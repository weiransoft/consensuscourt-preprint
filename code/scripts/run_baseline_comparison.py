#!/usr/bin/env python3
"""
对照基线端到端对比实验（论文大纲 §6.2 主实验 Tab.3 核心数据，v0.4.11）。

在 DDB-L1（合成争议，正/负各 N 例）+ DDB-L1.5（88 例负例对照）上，
让四个仲裁系统消费**完全相同**的证据评分（同一个 EvidenceScorer、
同一份证据集），对比矛盾检测的召回率 / 假阳性率 / 准确率：

- consensus_court_full：完整 ConsensusCourt 判定内核——
  AsymmetricRubric 全开关（aspect + ValidityGate + 强度分级），
  预测信号 = rubric 定罪（verdict.passed=False，即找到过门禁的矛盾对，
  含 v0.4.7 硬约束前置扫描）。
- majority_vote：等权多数决（baseline_arbiter.MajorityVote），
  预测信号 = PI 式裁决"驳回提案"；**无矛盾对检测通道**。
- elo_arbiter：角色加权 + Elo 校准（baseline_arbiter.EloArbiter），
  同样只有证据质量聚合，无矛盾对检测通道。
- single_judge：最高分证据直裁决（baseline_arbiter.SingleJudge）。

科学假设（大纲 §1 核心动机）：三个基线只聚合证据质量，两侧高质量对立
证据都会得高分 → 基线"采信"而漏检真矛盾（recall 显著低）；只有
ConsensusCourt 的非对称举证矛盾对通道能在高质量证据上检出真矛盾，
同时在 L1.5 负例对照上 FPR=0。

二分类口径：
- 正类 y=1：案件含真矛盾（L1 with_contradiction），系统应"拦下"；
- 负类 y=0：无真矛盾（L1 no_contradiction + L1.5 全部）；
- ConsensusCourt 预测正类 = 定罪（not verdict.passed）；
- 基线预测正类 = 裁决"驳回提案"（reject 口径，主指标）；
  另报告"非支持"口径（驳回或有条件支持都算拦下）作为敏感性分析，
  避免阈值口径选择导致的结论粉饰。

用法：
    backend$ .venv/bin/python tests/scripts/run_baseline_comparison.py
    backend$ .venv/bin/python tests/scripts/run_baseline_comparison.py --cases 300

输出：tests/ddb/baseline_comparison_report.json + stdout 对照表
退出码：0=红线全部满足；1=红线违例；2=运行异常
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# 导入前置：backend 根入 sys.path + 环境变量占位（必须在 import app 之前）
# ---------------------------------------------------------------------------
_BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

os.environ.setdefault("SCIMINER_API_KEY", "baseline-cmp-placeholder")
os.environ.setdefault("DB_PASSWORD", "baseline-cmp-placeholder")

from app.core.virtual_lab.baseline_arbiter import BASELINE_ARBITERS  # noqa: E402
from app.core.virtual_lab.consensus_court import Evidence, EvidenceScorer  # noqa: E402
from app.core.virtual_lab.rubric_engine import AsymmetricRubric  # noqa: E402
from tests.ddb.l1_generator import generate as generate_l1  # noqa: E402

#: 实验种子（与消融管线一致，保证可复现）
SEED = 42

#: Wilson 95% 分数区间的 z 值（默认 1.96，与 baseline_stats 单一事实源对齐）
_WILSON_Z = 1.96

#: 完整 ConsensusCourt 系统在对照表中的注册名
COURT_NAME = "consensus_court_full"


def wilson_interval(k: int, n: int, z: float = _WILSON_Z) -> Optional[list]:
    """Wilson 95% CI 的单一事实源包装（v1.0 §6 J2 裁决）。

    真实实现下沉到 tests.scripts.baseline_stats.wilson_interval（返回 tuple）；
    本函数保留原调用方签名：返回 list、各元素 round 到 4 位小数（论文
    Tab.2 CI 列精度）。返回 None 语义不变（n=0 时）。
    """
    from tests.scripts.baseline_stats import wilson_interval as _bs_wi

    tup = _bs_wi(k, n, z=z)
    if tup is None:
        return None
    # baseline_stats 返回 tuple[float, float]，round(4) 对齐 run_baseline_comparison
    # 原始契约——论文表格显示 4 位小数精度。
    return [round(tup[0], 4), round(tup[1], 4)]


def _to_evidence(ev: dict) -> Evidence:
    """把 L1/L1.5 案例证据字典无损还原为 consensus_court.Evidence 对象。

    与 run_ablation_v03._to_evidence 同口径：unknown 来源证据补占位引用，
    该处理只影响 RuleEngine 的 agent_opinion 规则，不影响 ABP 通道。

    Args:
        ev: L1/L1.5 案例中的证据字典

    Returns:
        consensus_court.Evidence 实例
    """
    from datetime import datetime as _dt

    reference_ids = list(ev["reference_ids"])
    if ev["source"] == "unknown" and not reference_ids:
        reference_ids = [f"UNKNOWN-SRC-{ev['evidence_id']}"]
    timestamp: Optional[_dt] = None
    if ev["timestamp"]:
        timestamp = _dt.fromisoformat(ev["timestamp"])
    return Evidence(
        evidence_id=ev["evidence_id"],
        source=ev["source"],
        content=ev["content"],
        role=ev["role"],
        timestamp=timestamp,
        reference_ids=reference_ids,
        metadata=dict(ev.get("metadata") or {}),
    )


def _empty_confusion() -> dict:
    """初始化二分类混淆计数字典（TP/FP/TN/FN 全零）。"""
    return {"tp": 0, "fp": 0, "tn": 0, "fn": 0}


def _rates(cm: dict) -> dict:
    """由混淆计数计算对照指标。

    Args:
        cm: 含 tp/fp/tn/fn 的字典

    Returns:
        recall/fpr/accuracy/precision/n 指标字典（分母为 0 时记 None，
        避免除零；n 恒为总样本数）
    """
    tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]
    n_pos = tp + fn
    n_neg = fp + tn
    total = tp + fp + tn + fn
    return {
        "n": total,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "recall": round(tp / n_pos, 4) if n_pos else None,
        "fpr": round(fp / n_neg, 4) if n_neg else None,
        "accuracy": round((tp + tn) / total, 4) if total else None,
        "precision": round(tp / (tp + fp), 4) if (tp + fp) else None,
        # Wilson 95% 置信区间（v0.4.16 评审 P1-6a）：跨案例二项波动仍然
        # 存在，论文主表必须带区间；recall 区间基于 TP/n_pos，FPR 基于 FP/n_neg
        "recall_ci95": wilson_interval(tp, n_pos),
        "fpr_ci95": wilson_interval(fp, n_neg),
    }


def evaluate_systems(cases: list[dict]) -> dict[str, dict]:
    """在同一案例池上运行全部四个系统，返回各系统的指标字典。

    每个案件只做一次证据还原与 EvidenceScorer 评分，四个系统复用同一
    批 EvidenceScore——保证对照的唯一变量是"裁决逻辑"，不含评分差异。

    Args:
        cases: L1 + L1.5 合并后的案例字典列表（须含 polarity 字段）

    Returns:
        {system_name: 指标字典}，指标字典含 reject / non_support 两种
        拦下口径的混淆指标（reject 为主指标，non_support 为敏感性分析）
    """
    scorer = EvidenceScorer()
    # 完整法庭内核：三机制全开（与 hold_court 中 AsymmetricRubric() 默认一致）
    rubric = AsymmetricRubric()
    # 三个基线各取一个无状态实例（arbitrate 不持有跨案状态）
    baselines = {name: cls() for name, cls in BASELINE_ARBITERS.items()}

    # 两套口径的混淆矩阵：reject=仅"驳回提案"算拦下；
    # non_support="驳回提案"或"有条件支持"都算拦下
    confusion: dict[str, dict[str, dict]] = {
        COURT_NAME: {"reject": _empty_confusion(), "non_support": _empty_confusion()},
    }
    for name in baselines:
        confusion[name] = {
            "reject": _empty_confusion(),
            "non_support": _empty_confusion(),
        }

    # 各系统裁决分布留痕（便于复查，不是指标）
    decision_dist: dict[str, dict[str, int]] = {
        name: {} for name in [COURT_NAME, *baselines.keys()]
    }

    for case in cases:
        has_truth = case["polarity"] == "with_contradiction"
        evidences = [_to_evidence(ev) for ev in case["evidences"]]
        scores = [scorer.score(ev, case["topic"]) for ev in evidences]
        emap = {ev.evidence_id: ev for ev in evidences}

        # --- 完整 ConsensusCourt：rubric 定罪信号 ---
        verdict = rubric.evaluate(scores, emap, case["topic"])
        court_flagged = not verdict.passed
        predictions: dict[str, bool] = {COURT_NAME: court_flagged}

        # --- 三个基线：仲裁裁决信号 ---
        for name, arb in baselines.items():
            decision, _rationale, _conf = arb.arbitrate(
                case["topic"], scores, emap
            )
            decision_dist[name][decision] = (
                decision_dist[name].get(decision, 0) + 1
            )
            # 主口径：只有明确驳回才算"拦下矛盾"
            predictions[name] = decision == "驳回提案"
            # 敏感性口径：非终局支持即拦下（有条件支持计入）
            ns_flagged = decision != "支持提案"
            ns_cm = confusion[name]["non_support"]
            if has_truth:
                ns_cm["tp" if ns_flagged else "fn"] += 1
            else:
                ns_cm["fp" if ns_flagged else "tn"] += 1

        # 记录各系统主口径混淆（ConsensusCourt 输出只有定罪/未定罪二元，
        # 不存在"有条件"档，故其 non_support 口径与 reject 恒等）
        for name, flagged in predictions.items():
            cm = confusion[name]["reject"]
            if has_truth:
                cm["tp" if flagged else "fn"] += 1
            else:
                cm["fp" if flagged else "tn"] += 1

    # ConsensusCourt 两口径完全一致：用最终主口径计数复制一份
    confusion[COURT_NAME]["non_support"] = dict(confusion[COURT_NAME]["reject"])

    # 汇总指标
    results: dict[str, dict] = {}
    for name, matrices in confusion.items():
        results[name] = {
            "reject": _rates(matrices["reject"]),
            "non_support": _rates(matrices["non_support"]),
            "decision_distribution": decision_dist.get(name, {}),
        }
    return results


def check_red_lines(results: dict[str, dict]) -> list[str]:
    """对照实验红线检查（论文主假设的可证伪条款）。

    Args:
        results: evaluate_systems 产出的各系统指标

    Returns:
        违例描述列表（空列表 = 全部满足）
    """
    violations: list[str] = []
    court = results[COURT_NAME]["reject"]
    baseline_names = [n for n in results if n != COURT_NAME]

    # 红线 1：完整法庭在正例零漏检、负例零误伤、准确率满分
    if court["recall"] != 1.0:
        violations.append(
            f"ConsensusCourt 召回红线违例：recall={court['recall']}（期望 1.0）"
        )
    if court["fpr"] != 0.0:
        violations.append(
            f"ConsensusCourt FPR 红线违例：fpr={court['fpr']}（期望 0.0）"
        )
    if court["accuracy"] != 1.0:
        violations.append(
            f"ConsensusCourt 准确率红线违例：accuracy={court['accuracy']}（期望 1.0）"
        )

    # 红线 2：逐基线严格优于（核心动机的可证伪条款——任一基线追平即削弱主张）
    for name in baseline_names:
        b = results[name]["reject"]
        if b["recall"] is not None and b["recall"] >= court["recall"]:
            violations.append(
                f"对照优势违例：{name} recall={b['recall']} ≥ "
                f"ConsensusCourt recall={court['recall']}"
                "（无矛盾检测通道的基线不应在正例追平完整法庭）"
            )
        if b["accuracy"] is not None and b["accuracy"] >= court["accuracy"]:
            violations.append(
                f"对照优势违例：{name} accuracy={b['accuracy']} ≥ "
                f"ConsensusCourt accuracy={court['accuracy']}"
            )
        if b["fpr"] is not None and b["fpr"] < court["fpr"]:
            violations.append(
                f"对照优势违例：{name} fpr={b['fpr']} < "
                f"ConsensusCourt fpr={court['fpr']}（完整法庭 FPR 不应高于基线）"
            )
    return violations


def main() -> int:
    """主流程：生成/加载 L1+L1.5 → 四系统评估 → 红线检查 → 落盘报告。"""
    parser = argparse.ArgumentParser(
        description="§6.2 ConsensusCourt vs 三个对照基线端到端对比实验"
    )
    parser.add_argument(
        "--cases", type=int, default=60,
        help="L1 每极性案例数（默认 60；主实验用 300）",
    )
    args = parser.parse_args()

    print(f"[1/3] 加载 DDB-L1（{args.cases}×2 例，seed={SEED}）+ DDB-L1.5 负例对照...")
    cases = generate_l1(
        num_with_contradiction=args.cases,
        num_no_contradiction=args.cases,
        seed=SEED,
    )
    from tests.ddb.l15_generator import generate as generate_l15

    l15_cases = generate_l15(per_template=8, seed=SEED)
    cases.extend(l15_cases)
    n_pos = sum(1 for c in cases if c["polarity"] == "with_contradiction")
    n_neg = len(cases) - n_pos
    print(f"      合并案例池：{len(cases)} 例（正例 {n_pos} / 负例 {n_neg}）")

    print("[2/3] 四系统同评分对照评估...")
    results = evaluate_systems(cases)
    print(
        f"      {'system':<24}{'recall':>9}{'recall_ci95':>18}"
        f"{'fpr':>9}{'fpr_ci95':>18}{'accuracy':>10}{'precision':>11}"
    )
    order = [COURT_NAME, "majority_vote", "elo_arbiter", "single_judge"]
    for name in order:
        m = results[name]["reject"]

        def _fmt_ci(ci: Optional[list]) -> str:
            """把 Wilson 区间格式化为 '[lo,hi]'（None → '-'，便于 stdout 对账）。"""
            return f"[{ci[0]:.3f},{ci[1]:.3f}]" if ci else "-"

        print(
            f"      {name:<24}"
            f"{(m['recall'] if m['recall'] is not None else float('nan')):>9.4f}"
            f"{_fmt_ci(m['recall_ci95']):>18}"
            f"{(m['fpr'] if m['fpr'] is not None else float('nan')):>9.4f}"
            f"{_fmt_ci(m['fpr_ci95']):>18}"
            f"{(m['accuracy'] if m['accuracy'] is not None else float('nan')):>10.4f}"
            f"{(m['precision'] if m['precision'] is not None else float('nan')):>11.4f}"
        )
    print("      （敏感性分析：非支持口径，含'有条件支持'）")
    for name in order:
        m = results[name]["non_support"]
        print(
            f"      {name + ' [non-support]':<24}"
            f"{(m['recall'] if m['recall'] is not None else float('nan')):>9.4f}"
            f"{(m['fpr'] if m['fpr'] is not None else float('nan')):>9.4f}"
            f"{(m['accuracy'] if m['accuracy'] is not None else float('nan')):>10.4f}"
        )

    print("[3/3] 红线检查 + 写报告...")
    violations = check_red_lines(results)
    for v in violations:
        print(f"      [违例] {v}")
    if not violations:
        print("      全部红线满足：完整法庭 FPR=0 / recall=1 / Acc=1，且严格优于三基线")

    report = {
        "experiment": "§6.2 对照基线端到端对比（Tab.3 冒烟数据）",
        "benchmark": "DDB-L1（合成争议）+ DDB-L1.5（负例对照 88 例）",
        "seed": SEED,
        "cases_per_polarity_l1": args.cases,
        "pool_composition": {"n_positive": n_pos, "n_negative": n_neg},
        "prediction_mapping": {
            "consensus_court_full": "AsymmetricRubric 三机制全开，定罪=not verdict.passed",
            "baselines_reject": "arbitrate 裁决 == '驳回提案'",
            "baselines_non_support": "arbitrate 裁决 != '支持提案'（敏感性分析）",
        },
        "systems": results,
        "red_line_violations": violations,
        "generated_at": datetime.now().isoformat(),
        "note": (
            "冒烟级（L1 60×2 + L1.5 88）；主实验 --cases 300 放大即可。"
            "四系统复用同一批 EvidenceScorer 评分，唯一变量为裁决逻辑。"
        ),
    }
    out_path = (
        Path(__file__).resolve().parents[1] / "ddb" / "baseline_comparison_report.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"      报告：{out_path}")

    if violations:
        print("结论：对照实验红线违例，需检查基线行为或 rubric 词典覆盖。")
        return 1
    print("结论：对照实验通过——矛盾对检测通道的必要性得到实证支持。")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 —— 管线异常统一归口退出码 2
        print(f"运行异常：{exc}", file=sys.stderr)
        raise
