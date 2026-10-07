"""
H2a pilot 实验 —— 非对称举证（ABP）vs 有罪推定（Guilty-Presumption）假阳性对比

假设 H2a：
    共识法庭旧的"有罪推定"规则初筛（consensus_court.RuleEngine.screen()：
    证据存在任何形式缺陷 → 整条剔除）会在"仅有形式缺陷、实质无矛盾"的案例上
    产生系统性假阳性；而"非对称举证"策略（rubric_engine.AsymmetricRubric.
    evaluate()：默认通过，仅检出矛盾对才拒绝）假阳性率显著更低，
    同时保持对真矛盾案例的高召回。

实验设计：
    两个策略都作用在同一个 DDB-L1 合成基准（tests/ddb/l1_generator.py 生成）上，
    案例级判定口径：
    - 有罪推定（guilty）：对齐 ConsensusCourt.hold_court 的真实初筛行为——
      任一证据被 RuleEngine.screen() 报出任何问题，即案例级"拒绝"。
    - 非对称举证（abp）：EvidenceScorer.score() 逐条评分后送入
      AsymmetricRubric.evaluate()，RubricVerdict.passed=False 即案例级"拒绝"。
    指标：
    - 召回 = 有矛盾案例上的拒绝率
    - FPR（假阳性率）= 无矛盾案例上的拒绝率（含 defect_only 子集单列）
    - 相对降幅 FPR_rel = (FPR_guilty − FPR_abp) / FPR_guilty

运行方式（在 backend/ 目录下）：
    python -m pytest tests/ddb/test_pilot_h2a.py -v    # 单元测试（80 例小样本）
    python -m tests.ddb.test_pilot_h2a                 # 全量 pilot（600 例，输出 JSON 报告）

依赖说明：
    AsymmetricRubric / RuleEngine / EvidenceScorer 均为纯内存对象（构造无
    DB 依赖），本实验不创建任何数据库 Session、不使用 conftest 的 DB fixture。
    导入 app 包要求环境变量 SCIMINER_API_KEY / DB_PASSWORD 非空
    （app/config.py 的 fail-closed 校验），本文件在导入 app 前用
    os.environ.setdefault 注入测试占位值——与 conftest 对
    TEST_ADMIN_PASSWORD 的处理模式一致，不会覆盖真实环境已有配置。
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# 导入前置：backend 根目录入 sys.path + 环境变量占位（必须在 import app 之前）
# ---------------------------------------------------------------------------
_BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

# app.config 对以下两项做非空强校验（fail-closed），pilot 仅用纯内存组件，
# 占位值不会触达任何真实服务；setdefault 保证不覆盖调用方已有的真实配置。
os.environ.setdefault("SCIMINER_API_KEY", "ddb-pilot-placeholder")
os.environ.setdefault("DB_PASSWORD", "ddb-pilot-placeholder")

from app.core.virtual_lab.consensus_court import (  # noqa: E402
    Evidence,
    EvidenceScorer,
    RuleEngine,
)
from app.core.virtual_lab.rubric_engine import AsymmetricRubric  # noqa: E402
from tests.ddb.l1_generator import DdbL1Generator, generate  # noqa: E402

#: pilot 默认随机种子（与生成器默认一致，写入报告保证可复现）
SEED = 42

#: pilot 单元测试使用的 8 类矛盾类别名（与 l1_generator 类别表一致）
_CONTRADICTION_CATEGORY_NAMES = {
    "有效-无效", "成功-失败", "阳性-阴性", "激活-抑制",
    "存在-不存在", "高表达-低表达", "促进-抑制", "上调-下调",
}


# ---------------------------------------------------------------------------
# 基准加载（测试用缓存，避免重复生成）
# ---------------------------------------------------------------------------

_SMALL_CASES_CACHE: Optional[list[dict]] = None


def _small_cases() -> list[dict]:
    """
    返回测试用小样本基准（40 有矛盾 + 40 无矛盾，其中 20 例 defect_only）。

    模块级缓存：三个 pytest 测试共享同一份样本，保证断言对象一致且省时。
    """
    global _SMALL_CASES_CACHE
    if _SMALL_CASES_CACHE is None:
        _SMALL_CASES_CACHE = generate(
            num_with_contradiction=40, num_no_contradiction=40, seed=SEED,
        )
    return _SMALL_CASES_CACHE


# ---------------------------------------------------------------------------
# 案例 dict → Evidence 对象（仅无损还原 + 一条 pilot 安全规则）
# ---------------------------------------------------------------------------

def _to_evidence(ev: dict) -> Evidence:
    """
    把基准案例中的证据字典还原为 consensus_court.Evidence 对象。

    pilot 安全规则：source 为 "unknown"（谱系外来源缺陷）的证据，其引用列表
    无条件补一个占位引用。原因：RuleEngine.screen() 的第 3 条有罪推定规则是
    "agent_opinion 且无引用"，若 unknown_source 案例恰好采到 agent_opinion，
    其缺陷计数会混入第 3 条规则，污染"缺陷类型 → 假阳性来源"的可解释性。
    该处理不添加任何文本内容，不影响两个策略的实质判定：
    - ABP 只看内容关键词，unknown 来源与引用与否均不参与矛盾检测；
    - 有罪推定对 unknown_source 证据仍保留其 short_content /
      missing_reference（其他证据）等真实缺陷命中。
    其余字段（含 timestamp=None 的时效缺省、0-6 年随机时效）无损还原，
    由 EvidenceScorer 的 recency 维度真实计算。

    Args:
        ev: 基准案例中的证据字典

    Returns:
        consensus_court.Evidence 实例
    """
    reference_ids = list(ev["reference_ids"])
    if ev["source"] == "unknown" and not reference_ids:
        reference_ids = [f"UNKNOWN-SRC-{ev['evidence_id']}"]
    timestamp: Optional[datetime] = None
    if ev["timestamp"]:
        timestamp = datetime.fromisoformat(ev["timestamp"])
    return Evidence(
        evidence_id=ev["evidence_id"],
        source=ev["source"],
        content=ev["content"],
        role=ev["role"],
        timestamp=timestamp,
        reference_ids=reference_ids,
        metadata=dict(ev.get("metadata") or {}),
    )


# ---------------------------------------------------------------------------
# 两种策略的案例级判定（全部真实调用源码 API）
# ---------------------------------------------------------------------------

def run_guilty_presumption(case: dict) -> bool:
    """
    有罪推定策略（旧路径）：真实调用 consensus_court.RuleEngine.screen()。

    与 ConsensusCourt.hold_court 的初筛行为对齐：任一证据被 screen() 报出
    问题（内容过短 / 来源为空 / agent_opinion 缺引用 / AF2 引用），
    该证据即被剔除，不会进入后续评分与裁决；本 pilot 把"案例存在被剔除
    证据"定义为案例级拒绝——这正是旧机制的假阳性来源（形式缺陷 ≠ 内容矛盾）。

    Args:
        case: DDB-L1 案例字典

    Returns:
        True 表示案例被拒绝（存在被初筛剔除的证据）
    """
    engine = RuleEngine()
    for ev_dict in case["evidences"]:
        issues = engine.screen(_to_evidence(ev_dict))
        if issues:
            return True
    return False


def run_abp(case: dict) -> bool:
    """
    非对称举证策略（ABP，新路径）：真实调用
    EvidenceScorer.score() + AsymmetricRubric.evaluate()。

    流程与 DrillBeforeDisposition.disposition 一致：逐条评分 → 构建
    evidence_map → rubric 评估；RubricVerdict.passed=False（检出矛盾对）
    即案例级拒绝。形式缺陷只进入 pending_issues，不影响通过——
    这正是 H2a 假设要验证的行为差异。

    Args:
        case: DDB-L1 案例字典

    Returns:
        True 表示案例被拒绝（检出矛盾对）
    """
    scorer = EvidenceScorer()
    rubric = AsymmetricRubric()
    evidences = [_to_evidence(ev) for ev in case["evidences"]]
    scores = [scorer.score(ev, case["topic"]) for ev in evidences]
    evidence_map = {ev.evidence_id: ev for ev in evidences}
    verdict = rubric.evaluate(scores, evidence_map, case["topic"])
    return not verdict.passed


# ---------------------------------------------------------------------------
# 指标计算
# ---------------------------------------------------------------------------

def _rejection_rate(cases: list[dict], strategy) -> tuple[int, int, float]:
    """
    计算某策略在一组案例上的拒绝率。

    Args:
        cases: 案例字典列表
        strategy: 案例级判定函数（run_guilty_presumption / run_abp）

    Returns:
        (拒绝数, 案例总数, 拒绝率)；空集时拒绝率定义为 0.0
    """
    rejected = sum(1 for c in cases if strategy(c))
    total = len(cases)
    rate = rejected / total if total else 0.0
    return rejected, total, rate


def compute_metrics(cases: list[dict]) -> dict:
    """
    对基准全量运行两种策略并计算 H2a 指标。

    Args:
        cases: DDB-L1 案例字典列表

    Returns:
        指标字典：两策略各自的召回 / FPR / defect_only 子集 FPR、
        相对降幅 FPR_rel，以及各组样本数
    """
    with_c = [c for c in cases if c["polarity"] == "with_contradiction"]
    without_c = [c for c in cases if c["polarity"] == "no_contradiction"]
    defect_only = [c for c in without_c if c["ground_truth"]["defect_only"]]
    clean_neutral = [c for c in without_c if not c["ground_truth"]["defect_only"]]

    g_recall_n, g_recall_d, g_recall = _rejection_rate(with_c, run_guilty_presumption)
    g_fpr_n, g_fpr_d, g_fpr = _rejection_rate(without_c, run_guilty_presumption)
    g_defect_n, _, g_defect = _rejection_rate(defect_only, run_guilty_presumption)
    g_clean_n, _, g_clean = _rejection_rate(clean_neutral, run_guilty_presumption)

    a_recall_n, a_recall_d, a_recall = _rejection_rate(with_c, run_abp)
    a_fpr_n, a_fpr_d, a_fpr = _rejection_rate(without_c, run_abp)
    a_defect_n, _, a_defect = _rejection_rate(defect_only, run_abp)
    a_clean_n, _, a_clean = _rejection_rate(clean_neutral, run_abp)

    # 相对降幅：分母为 0（有罪推定在该集上零假阳性）时按 0 处理并保留原始值供审计
    fpr_rel = (g_fpr - a_fpr) / g_fpr if g_fpr > 0 else 0.0

    return {
        "counts": {
            "total": len(cases),
            "with_contradiction": len(with_c),
            "no_contradiction": len(without_c),
            "defect_only": len(defect_only),
            "clean_neutral": len(clean_neutral),
        },
        "guilty_presumption": {
            "recall": {"rejected": g_recall_n, "denominator": g_recall_d, "rate": g_recall},
            "fpr_no_contradiction": {"rejected": g_fpr_n, "denominator": g_fpr_d, "rate": g_fpr},
            "fpr_defect_only": {"rejected": g_defect_n, "rate": g_defect},
            "fpr_clean_neutral": {"rejected": g_clean_n, "rate": g_clean},
        },
        "abp": {
            "recall": {"rejected": a_recall_n, "denominator": a_recall_d, "rate": a_recall},
            "fpr_no_contradiction": {"rejected": a_fpr_n, "denominator": a_fpr_d, "rate": a_fpr},
            "fpr_defect_only": {"rejected": a_defect_n, "rate": a_defect},
            "fpr_clean_neutral": {"rejected": a_clean_n, "rate": a_clean},
        },
        "fpr_relative_reduction": round(fpr_rel, 4),
    }


# ---------------------------------------------------------------------------
# pytest 单元测试（80 例小样本）
# ---------------------------------------------------------------------------

def test_generator_selfcheck():
    """
    测试 1：生成器基准防泄漏自检。
    - 有矛盾案例：必含至少一组 **跨来源** 矛盾关键词对（与源码
      _find_contradictions 跳过同源对的口径一致）；
    - 无矛盾案例：所有证据不得出现任何矛盾关键词对的任一侧关键词；
    - defect_only 案例必带 defect_flags；
    - DdbL1Generator.validate 全量校验通过。
    """
    cases = _small_cases()
    pairs = list(AsymmetricRubric.CONTRADICTION_KEYWORD_PAIRS)
    all_keywords = [kw for pair in pairs for kw in pair]

    for case in cases:
        evs = case["evidences"]
        if case["polarity"] == "with_contradiction":
            found = any(
                ea["source"] != eb["source"]
                and any(
                    (a in ea["content"] and b in eb["content"])
                    or (b in ea["content"] and a in eb["content"])
                    for a, b in pairs
                )
                for i, ea in enumerate(evs) for eb in evs[i + 1:]
            )
            assert found, f"{case['case_id']}: 有矛盾案例缺少跨来源矛盾对"
        else:
            for ev in evs:
                hit = [kw for kw in all_keywords if kw in ev["content"]]
                assert not hit, (
                    f"{case['case_id']}/{ev['evidence_id']}: "
                    f"无矛盾案例证据含矛盾关键词侧 {hit}"
                )
            if case["ground_truth"]["defect_only"]:
                assert case["defect_flags"], f"{case['case_id']}: defect_only 无缺陷标记"

    # 全量结构校验（同一校验在 generate() 内部也执行过一次，双保险）
    DdbL1Generator.validate(cases)


def test_abp_discrimination():
    """
    测试 2：ABP 策略（真实调用 AsymmetricRubric.evaluate）判别力。
    - 有矛盾案例拒绝率（召回）> 0.9；
    - 无矛盾案例通过率 > 0.9（等价于 FPR < 0.1）。
    """
    cases = _small_cases()
    with_c = [c for c in cases if c["polarity"] == "with_contradiction"]
    without_c = [c for c in cases if c["polarity"] == "no_contradiction"]

    _, _, recall = _rejection_rate(with_c, run_abp)
    rejected_no, total_no, fpr = _rejection_rate(without_c, run_abp)

    assert recall > 0.9, f"ABP 在有矛盾案例上召回 {recall:.3f} 未达 0.9"
    assert (1 - fpr) > 0.9, (
        f"ABP 在无矛盾案例上通过率 {1 - fpr:.3f} 未达 0.9"
        f"（拒绝 {rejected_no}/{total_no}）"
    )


def test_guilty_presumption_false_positive_differential():
    """
    测试 3：有罪推定策略的假阳性劣势（H2a 核心断言）。
    - 有罪推定在 defect_only 子集（形式缺陷、实质无矛盾）上拒绝率显著高；
    - 且其无矛盾案例整体 FPR 严格大于 ABP 的 FPR。
    """
    cases = _small_cases()
    without_c = [c for c in cases if c["polarity"] == "no_contradiction"]
    defect_only = [c for c in without_c if c["ground_truth"]["defect_only"]]

    _, _, g_defect = _rejection_rate(defect_only, run_guilty_presumption)
    _, _, g_fpr = _rejection_rate(without_c, run_guilty_presumption)
    _, _, a_fpr = _rejection_rate(without_c, run_abp)
    _, _, a_defect = _rejection_rate(defect_only, run_abp)

    assert g_defect > 0.5, f"有罪推定在 defect_only 上拒绝率 {g_defect:.3f} 过低，样本构造异常"
    assert g_fpr > a_fpr, (
        f"有罪推定 FPR {g_fpr:.3f} 未高于 ABP FPR {a_fpr:.3f}，H2a 断言失败"
    )
    assert a_defect <= a_fpr + 1e-9 and g_defect > a_defect, (
        f"defect_only 子集上两策略假阳性无差异（guilty={g_defect:.3f}, abp={a_defect:.3f}）"
    )


# ---------------------------------------------------------------------------
# 全量 pilot main()（600 例，输出 JSON 报告）
# ---------------------------------------------------------------------------

def main() -> dict:
    """
    全量 pilot：生成 600 例（300 有矛盾 + 300 无矛盾，其中 150 defect_only），
    运行两种策略计算 H2a 指标，写入 tests/ddb/pilot_h2a_report.json。

    Returns:
        报告字典（同写盘内容）
    """
    print("DDB-L1 pilot H2a：生成 600 例基准（seed=42）...")
    cases = generate(
        num_with_contradiction=300, num_no_contradiction=300, seed=SEED,
    )
    metrics = compute_metrics(cases)

    report = {
        "experiment": "H2a — ABP vs 有罪推定（规则初筛）假阳性对比 pilot",
        "benchmark": "DDB-L1（tests/ddb/l1_generator.py 合成争议基准）",
        "strategies": {
            "guilty_presumption": "consensus_court.RuleEngine.screen()：任一证据形式缺陷 → 案例拒绝",
            "abp": "rubric_engine.AsymmetricRubric.evaluate()：默认通过，仅检出矛盾对 → 案例拒绝",
        },
        "seed": SEED,
        "generated_at": datetime.now().astimezone().isoformat(),
        "metrics": metrics,
    }

    out_path = Path(__file__).resolve().parent / "pilot_h2a_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 控制台摘要（关键数字直接可读）
    m = metrics
    print(f"样本：总 {m['counts']['total']} | 有矛盾 {m['counts']['with_contradiction']}"
          f" | 无矛盾 {m['counts']['no_contradiction']}"
          f"（defect_only {m['counts']['defect_only']}）")
    for name, key in (("有罪推定", "guilty_presumption"), ("ABP", "abp")):
        s = m[key]
        print(
            f"{name}：召回 {s['recall']['rate']:.3f}"
            f"（{s['recall']['rejected']}/{s['recall']['denominator']}）| "
            f"FPR(无矛盾) {s['fpr_no_contradiction']['rate']:.3f}"
            f"（{s['fpr_no_contradiction']['rejected']}/{s['fpr_no_contradiction']['denominator']}）| "
            f"FPR(defect_only) {s['fpr_defect_only']['rate']:.3f}"
        )
    print(f"FPR 相对降幅 FPR_rel = {m['fpr_relative_reduction']:.3f}")
    print(f"报告已写入 {out_path}")
    return report


if __name__ == "__main__":
    # 本 pilot 全部为同步纯内存调用，无异步 API 需要 asyncio 包装
    main()
