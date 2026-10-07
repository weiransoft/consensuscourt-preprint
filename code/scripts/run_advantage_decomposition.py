#!/usr/bin/env python3
"""
定理 1-A 优势分解标定实验（论文大纲 §5.3/§6.5，v0.4.12）。

实验目标
========
定理 1-A 是零分布假设的逐案容斥恒等式：

    P(D) − P(M) = P(D\\M) − P(M\\D) = P(S) − P(M\\D)

其中底空间为"无真矛盾"案件（C=∅）：
- D = ∪ᵢ Dᵢ：任一证据携带 PENDING 级形式缺陷（有罪推定的拒绝事件）；
- M = ∪ₚ Mₚ：ABP rubric 检出至少一个过门禁矛盾对（ABP 的拒绝事件）；
- S = D\\M：纯缺陷无辜案件集（ABP 的优势来源）；
- M\\D：无缺陷却被矛盾对错配定罪（ABP 的独立代价）。

本实验在 DDB-L1 负例（30 defect_only + 30 clean）+ DDB-L1.5 三类
伪冲突负例（88 例：cross_aspect 40 + ambiguous_reference 24 +
different_object_gate 24）构成的 C=∅ 底空间上，对 full_v03（三机制全开）
与 baseline_v02（三机制全关）两套检测器分别：

1. 用真实 RuleEngine.screen_asymmetric 标定 D（PENDING 计数），
   CRITICAL 硬约束单独计数（本底空间理论上为 0，出现即红线违例）；
2. 用真实 AsymmetricRubric.evaluate 标定 M（contradictions 非空）；
3. 逐案归入 2×2 单元（D&M / S=D\\M / M\\D / neither），检验恒等式
   两侧是否精确闭合（集合代数，非不等式——必须残差为 0）；
4. 检验可证伪条款：full 配置 P(S) > P(M\\D)（该领域 ABP 有正收益），
   以及 aspect/门禁只能缩小 M 的事件集（M\\D_full ≤ M\\D_base）。

q_K / κ 标定（§5.4 引理 A-2）
=============================
在受控零假设合成案件（复用 run_robustness_experiments.build_null_case，
m=20、p₀=0.30、唯一来源）上：
- q_K = 错配对数 / 候选对总数 C(m,2)（单对边际错配概率）；
- κ = P(Mₚ∩Mₚ') / q_K²，p、p' 为共享一条证据的"相邻对"
  （每案 m·C(m−1,2) 个）。共享证据携带的极性词会同时决定两个对的
  命中，理论预期正相依 κ>1——这正是引理 A-2 不能用 Sidak 独立乘积、
  必须用 Rényi 界（Δ=2(m−2)）的经验依据。

用法：
    backend$ .venv/bin/python tests/scripts/run_advantage_decomposition.py
    backend$ .venv/bin/python tests/scripts/run_advantage_decomposition.py \\
        --cases 60 --null-trials 300

输出：tests/ddb/advantage_decomposition_report.json + stdout 分解表
退出码：0=全部红线满足；1=红线违例；2=运行异常
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

# ---------------------------------------------------------------------------
# 导入前置：backend 根入 sys.path + 环境变量占位（必须在 import app 之前）
# ---------------------------------------------------------------------------
_BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

os.environ.setdefault("SCIMINER_API_KEY", "advantage-decomp-placeholder")
os.environ.setdefault("DB_PASSWORD", "advantage-decomp-placeholder")

from app.core.virtual_lab.consensus_court import (  # noqa: E402
    Evidence,
    EvidenceScorer,
    RuleEngine,
)
from app.core.virtual_lab.rubric_engine import AsymmetricRubric  # noqa: E402
from tests.ddb.l1_generator import generate as generate_l1  # noqa: E402
from tests.ddb.l15_generator import generate as generate_l15  # noqa: E402

# 复用零假设案件构造（p₀=0.30、唯一来源、不植入对立词对），
# 保证 q_K/κ 与 §6.5 定理 1-B 扫描在同一受控底空间下标定。
from run_robustness_experiments import NULL_TOPIC, build_null_case  # noqa: E402

# 复用 L1/L1.5 证据字典 → Evidence 的统一还原口径（unknown 补占位引用）。
from run_ablation_v03 import _to_evidence  # noqa: E402

#: 实验种子（与消融/对照/鲁棒性管线一致）
SEED = 42

#: 恒等式闭合容差：集合代数逐案计数应为精确恒等，仅容浮点累加误差
CLOSURE_TOLERANCE = 1e-12

#: κ 联合事件最小样本量（相邻对共现次数）；不足视为统计功效不够，
#: 按红线违例处理（需调大 --null-trials），禁止小样本粉饰。
MIN_JOINT_EVENTS = 30

#: 零假设案件证据数（与 §6.5 n 扫描最大格点一致）
NULL_M = 20

#: 两档检测器配置：full_v03 = 三机制全开（ABP 完整版）；
#: baseline_v02 = 三机制全关（v0.2 有罪通道近似）。
DETECTOR_CONFIGS: dict[str, AsymmetricRubric] = {}  # 延迟到 main/函数内构造


def _make_configs() -> dict[str, AsymmetricRubric]:
    """构造两档检测器 rubric 实例（每次调用独立，避免状态串扰）。"""
    return {
        "full_v03": AsymmetricRubric(
            enable_aspect=True,
            enable_validity_gate=True,
            enable_intensity=True,
        ),
        "baseline_v02": AsymmetricRubric(
            enable_aspect=False,
            enable_validity_gate=False,
            enable_intensity=False,
        ),
    }


# ===========================================================================
# 底空间构造：L1 负例（分层）+ L1.5 负例（按类别分层）
# ===========================================================================

def build_negative_pool(cases_l1: int) -> list[tuple[str, dict]]:
    """构造 C=∅ 底空间案件池（全部为无真矛盾负例），带分层标签。

    分层口径（与大纲 §5.3 事件来源一一对应）：
    - "l1_defect_only"：L1 无矛盾但刻意注入形式缺陷 → S 集供给层；
    - "l1_clean"：L1 无矛盾且无缺陷 → 四格中的 neither 参照层；
    - "l15_{category}"：L1.5 伪冲突负例（cross_aspect /
      ambiguous_reference / different_object_gate 三类）→ M\\D 的
      对照层（v0.2 全关时误定罪，full 全开时应被机制拦截）。

    Args:
        cases_l1: L1 每极性生成数（负例总数等于该值，其中半数 defect_only）

    Returns:
        (stratum, case_dict) 元组列表
    """
    pool: list[tuple[str, dict]] = []
    l1_cases = generate_l1(
        num_with_contradiction=cases_l1,
        num_no_contradiction=cases_l1,
        seed=SEED,
    )
    for case in l1_cases:
        if case["polarity"] == "with_contradiction":
            continue  # 底空间仅负例
        stratum = (
            "l1_defect_only"
            if case["ground_truth"].get("defect_only")
            else "l1_clean"
        )
        pool.append((stratum, case))

    # L1.5：per_template=8 与 run_ablation_v03/run_baseline_comparison 一致
    l15_cases = generate_l15(per_template=8, seed=SEED)
    for case in l15_cases:
        pool.append((f"l15_{case['category']}", case))
    return pool


# ===========================================================================
# 事件标定：D（PENDING 缺陷）/ M（矛盾对定罪）
# ===========================================================================

def detect_defect_event(case: dict, rule_engine: RuleEngine) -> tuple[bool, int]:
    """标定案件级事件 D = ∪ᵢ Dᵢ，并单独统计 CRITICAL 硬约束数。

    D 的实现口径（§5.2）：至少一条证据经真实 screen_asymmetric
    返回 PENDING（内容过短 / 来源为空 / Agent 观点缺引用）。
    CRITICAL（AlphaFold2/3 双禁令）是真硬违规、不是"纯形式缺陷"，
    不计入 D；在合成底空间中必须为 0，否则基准被污染。

    Args:
        case: L1/L1.5 案例字典
        rule_engine: 真实规则引擎实例

    Returns:
        (has_pending, critical_evidence_count)
    """
    has_pending = False
    critical_count = 0
    for ev_dict in case["evidences"]:
        ev = _to_evidence(ev_dict)
        severities = [sev for _desc, sev in rule_engine.screen_asymmetric(ev)]
        if "PENDING" in severities:
            has_pending = True
        if "CRITICAL" in severities:
            critical_count += 1
    return has_pending, critical_count


def detect_mismatch_event(
    case: dict,
    rubric: AsymmetricRubric,
    scorer: EvidenceScorer,
) -> tuple[bool, float]:
    """标定案件级事件 M = ∪ₚ Mₚ（ABP rubric 是否检出过门禁矛盾对）。

    Args:
        case: L1/L1.5 案例字典
        rubric: 已配置开关的真实 AsymmetricRubric
        scorer: 真实证据评分器

    Returns:
        (convicted, confidence)：contradictions 非空即 M 发生
    """
    evidences = [_to_evidence(ev) for ev in case["evidences"]]
    scores = [scorer.score(ev, case["topic"]) for ev in evidences]
    emap = {ev.evidence_id: ev for ev in evidences}
    verdict = rubric.evaluate(scores, emap, case["topic"])
    return bool(verdict.contradictions), verdict.confidence


# ===========================================================================
# 2×2 单元分解 + 恒等式闭合
# ===========================================================================

def decompose(
    pool: list[tuple[str, dict]],
    rubric: AsymmetricRubric,
    scorer: EvidenceScorer,
    rule_engine: RuleEngine,
) -> dict:
    """在 C=∅ 底空间上做定理 1-A 的逐案事件分解。

    四格定义（互斥且穷尽）：
    - d_and_m：D∩M（有缺陷且被错配定罪）；
    - s：S=D\\M（有缺陷但无矛盾——ABP 救回的无辜主张）；
    - m_without_d：M\\D（无缺陷却被错配定罪——ABP 的独立代价）；
    - neither：Dᶜ∩Mᶜ。

    恒等式检验：
        lhs = P(D) − P(M)
        rhs = P(S) − P(M\\D)
    两式必须逐计数相等（残差 ≤ CLOSURE_TOLERANCE）。

    Args:
        pool: (stratum, case) 底空间案件列表
        rubric: 检测器配置
        scorer / rule_engine: 真实生产组件

    Returns:
        单元计数、各事件概率、闭合残差、分层明细、CRITICAL 计数
    """
    n = len(pool)
    cells = {"d_and_m": 0, "s": 0, "m_without_d": 0, "neither": 0}
    critical_total = 0
    # 分层聚合：每层独立统计四格事件数
    by_stratum: dict[str, dict] = {}

    for stratum, case in pool:
        has_d, n_critical = detect_defect_event(case, rule_engine)
        has_m, _conf = detect_mismatch_event(case, rubric, scorer)
        critical_total += n_critical

        if has_d and has_m:
            cell = "d_and_m"
        elif has_d and not has_m:
            cell = "s"
        elif has_m and not has_d:
            cell = "m_without_d"
        else:
            cell = "neither"
        cells[cell] += 1

        layer = by_stratum.setdefault(
            stratum,
            {"n": 0, "d_and_m": 0, "s": 0, "m_without_d": 0, "neither": 0},
        )
        layer["n"] += 1
        layer[cell] += 1

    p_d = (cells["d_and_m"] + cells["s"]) / n
    p_m = (cells["d_and_m"] + cells["m_without_d"]) / n
    p_s = cells["s"] / n
    p_m_without_d = cells["m_without_d"] / n
    lhs = p_d - p_m
    rhs = p_s - p_m_without_d

    # 分层表补比例字段（便于报告直接阅读）
    for layer in by_stratum.values():
        ln = layer["n"]
        layer["P(D)"] = round(
            (layer["d_and_m"] + layer["s"]) / ln, 4
        )
        layer["P(M)"] = round(
            (layer["d_and_m"] + layer["m_without_d"]) / ln, 4
        )
        layer["P(S)"] = round(layer["s"] / ln, 4)
        layer["P(M\\D)"] = round(layer["m_without_d"] / ln, 4)

    return {
        "n": n,
        "cells": cells,
        "P(D)_fpr_guilty": round(p_d, 6),
        "P(M)_fpr_abp": round(p_m, 6),
        "P(S)": round(p_s, 6),
        "P(M_without_D)": round(p_m_without_d, 6),
        "closure": {
            "lhs_PD_minus_PM": round(lhs, 6),
            "rhs_PS_minus_PMD": round(rhs, 6),
            "residual": abs(lhs - rhs),
        },
        "critical_evidence_count": critical_total,
        "by_stratum": by_stratum,
    }


# ===========================================================================
# q_K / κ 标定（零假设案件 + 相邻对枚举）
# ===========================================================================

def calibrate_null_dependence(
    rubric: AsymmetricRubric,
    scorer: EvidenceScorer,
    m: int = NULL_M,
    trials: int = 300,
    seed: int = SEED,
) -> dict:
    """在零假设合成案件上标定单对错配率 q_K 与相邻对相依系数 κ。

    计数口径：
    - 候选对：每案全部 C(m,2) 个证据对（来源唯一，无同源跳过），
      逐对用真实 rubric 评估后取其 contradictions 端点集判定 Mₚ；
    - 相邻对 (p,p')：共享恰好一条证据，即对三元组 {i,j,k} 取
      p=(i,j)、p'=(i,k)；每案 m·C(m−1,2) 个有序共享枚举
      （(i,j,k) 与 (i,k,j) 是同一对相邻对，故 j<k 去重——每个
      共享证据 i 恰好枚举 C(m−1,2) 个无序相邻对）；
    - q_K = Σ 1{Mₚ} / 候选对总数；
    - P_joint = Σ 1{Mₚ∩Mₚ'} / 相邻对总数；
    - κ = P_joint / q_K²。独立情形 κ=1；共享极性词诱导正相依 κ>1。

    Args:
        rubric: 检测器配置
        scorer: 真实评分器
        m: 每案证据数
        trials: 案件数
        seed: 随机器种子（与调用方其他实验隔离时可换种子）

    Returns:
        原始计数 + q_K / P_joint / κ
    """
    # 独立 RNG 流：q_K/κ 采样与 §6.5 n 扫描互不影响（不同 trials 参数），
    # 同种子同参数下结果可复现。
    rng = __import__("random").Random(seed)

    def mismatch_pairs(evidences: list[Evidence]) -> set[frozenset]:
        """返回该案被定罪的证据对端点 frozenset 集合。"""
        scores = [scorer.score(ev, NULL_TOPIC) for ev in evidences]
        emap = {ev.evidence_id: ev for ev in evidences}
        verdict = rubric.evaluate(scores, emap, NULL_TOPIC)
        return {
            frozenset((p.evidence_a_id, p.evidence_b_id))
            for p in verdict.contradictions
        }

    pair_opportunities = 0
    mismatch_events = 0
    adjacent_pairs = 0
    joint_events = 0

    for case_idx in range(trials):
        evidences = build_null_case(m, case_idx, rng)
        ids = [ev.evidence_id for ev in evidences]
        all_pairs = [frozenset((a, b)) for a, b in combinations(ids, 2)]
        convicted = mismatch_pairs(evidences)

        pair_opportunities += len(all_pairs)
        mismatch_events += sum(1 for p in all_pairs if p in convicted)

        # 相邻对枚举：固定共享证据 i，从其余 m−1 条中无序取两条
        for i in range(m):
            rest = [j for j in range(m) if j != i]
            for a, b in combinations(rest, 2):
                p = frozenset((ids[i], ids[a]))
                q = frozenset((ids[i], ids[b]))
                adjacent_pairs += 1
                if p in convicted and q in convicted:
                    joint_events += 1

    q_k = mismatch_events / pair_opportunities if pair_opportunities else 0.0
    p_joint = joint_events / adjacent_pairs if adjacent_pairs else 0.0
    kappa = p_joint / (q_k ** 2) if q_k > 0 else None

    return {
        "m": m,
        "trials": trials,
        "pair_opportunities": pair_opportunities,
        "mismatch_events": mismatch_events,
        "q_K": round(q_k, 6),
        "adjacent_pairs": adjacent_pairs,
        "joint_events": joint_events,
        "P_joint": round(p_joint, 8),
        "kappa": round(kappa, 4) if kappa is not None else None,
    }


# ===========================================================================
# 红线检查（预注册可证伪条款）
# ===========================================================================

def check_decomposition_red_lines(
    decomposition: dict[str, dict],
    null_calibration: dict[str, dict],
) -> list[str]:
    """定理 1-A 分解实验的红线检查。

    Args:
        decomposition: {config_name: decompose() 结果}
        null_calibration: {config_name: calibrate_null_dependence() 结果}

    Returns:
        违例描述列表（空 = 全部满足）
    """
    violations: list[str] = []

    # 红线 1：两配置恒等式残差必须为 0（容斥恒等式，非统计拟合）
    for name, res in decomposition.items():
        residual = res["closure"]["residual"]
        if residual > CLOSURE_TOLERANCE:
            violations.append(
                f"定理 1-A 闭合违例（{name}）：残差={residual}"
                f" > {CLOSURE_TOLERANCE}（容斥恒等式应精确成立）"
            )

    # 红线 2：底空间不得出现 CRITICAL 硬约束（合成负例无 AlphaFold 引用）
    for name, res in decomposition.items():
        if res["critical_evidence_count"] != 0:
            violations.append(
                f"底空间污染（{name}）：CRITICAL 硬约束证据 "
                f"{res['critical_evidence_count']} 条（C=∅ 底空间应为 0）"
            )

    # 红线 3：可证伪条款——full 配置 P(S) 必须严格大于 P(M\\D)，
    # 否则 ABP 在该底空间无正收益（论文核心主张被证伪）
    full = decomposition["full_v03"]
    if not full["P(S)"] > full["P(M_without_D)"]:
        violations.append(
            f"可证伪条款违例：P(S)={full['P(S)']} "
            f"≤ P(M\\D)={full['P(M_without_D)']}（ABP 在该领域无优势）"
        )

    # 红线 4：aspect/门禁只能缩小 M 的事件集——full 的 M\\D 不得高于 v0.2
    base = decomposition["baseline_v02"]
    if full["P(M_without_D)"] > base["P(M_without_D)"] + 1e-9:
        violations.append(
            f"机制方向违例：P(M\\D)_full={full['P(M_without_D)']} "
            f"> P(M\\D)_base={base['P(M_without_D)']}"
        )

    # 红线 5：q_K_full ≤ q_K_base（门禁/aspect 不增大边际错配率）
    q_full = null_calibration["full_v03"]["q_K"]
    q_base = null_calibration["baseline_v02"]["q_K"]
    if q_full > q_base + 1e-9:
        violations.append(
            f"q_K 方向违例：q_K_full={q_full} > q_K_base={q_base}"
        )

    # 红线 6：κ 标定有效性——baseline 下相邻对正相依 κ≥1（共享极性词），
    # 且联合事件计数达到最小样本量（禁止小样本估计）
    for name, cal in null_calibration.items():
        if cal["joint_events"] < MIN_JOINT_EVENTS:
            violations.append(
                f"κ 样本不足（{name}）：相邻对共现事件 "
                f"{cal['joint_events']} < {MIN_JOINT_EVENTS}"
                f"（请调大 --null-trials）"
            )
    base_cal = null_calibration["baseline_v02"]
    if (
        base_cal["joint_events"] >= MIN_JOINT_EVENTS
        and base_cal["kappa"] is not None
        and base_cal["kappa"] < 1.0
    ):
        violations.append(
            f"κ 相依方向违例（baseline_v02）：κ={base_cal['kappa']} < 1.0"
            "（共享证据极性词应诱导正相依）"
        )

    return violations


# ===========================================================================
# 主流程
# ===========================================================================

def main() -> int:
    """命令行入口：构造底空间 → 双配置分解 → q_K/κ 标定 → 红线 → 落盘。"""
    parser = argparse.ArgumentParser(
        description="§5.3/§6.5 定理 1-A 优势分解标定（P(S)/P(M\\D)/q_K/κ）"
    )
    parser.add_argument(
        "--cases", type=int, default=60,
        help="L1 每极性案件数（默认 60，负例=该值；另加 L1.5 88 例）",
    )
    parser.add_argument(
        "--null-trials", type=int, default=300,
        help="q_K/κ 标定的零假设案件数（默认 300，m=20）",
    )
    args = parser.parse_args()

    print("[1/4] 构造 C=∅ 负例底空间（L1 defect_only/clean + L1.5 三类）...")
    pool = build_negative_pool(args.cases)
    n_l1_def = sum(1 for s, _ in pool if s == "l1_defect_only")
    n_l1_clean = sum(1 for s, _ in pool if s == "l1_clean")
    n_l15 = sum(1 for s, _ in pool if s.startswith("l15_"))
    print(
        f"      底空间 {len(pool)} 例：L1 defect_only={n_l1_def} / "
        f"L1 clean={n_l1_clean} / L1.5={n_l15}"
    )

    print("[2/4] 双配置 2×2 事件分解（定理 1-A 恒等式闭合）...")
    scorer = EvidenceScorer()
    rule_engine = RuleEngine()
    configs = _make_configs()
    decomposition: dict[str, dict] = {}
    for name, rubric in configs.items():
        decomposition[name] = decompose(pool, rubric, scorer, rule_engine)
        r = decomposition[name]
        print(
            f"      [{name}] P(D)={r['P(D)_fpr_guilty']:.4f} "
            f"P(M)={r['P(M)_fpr_abp']:.4f} "
            f"P(S)={r['P(S)']:.4f} P(M\\D)={r['P(M_without_D)']:.4f} "
            f"四格={r['cells']} 残差={r['closure']['residual']:.2e}"
        )

    print(f"[3/4] q_K / κ 标定（零假设 m={NULL_M}，{args.null_trials} 案）...")
    null_calibration: dict[str, dict] = {}
    # full 配置使用偏移种子，避免与 baseline 共享同一批案件之外的任何
    # 隐式顺序耦合（案件构造本身逐案确定性，偏移种子提供独立复制）。
    for idx, (name, rubric) in enumerate(configs.items()):
        null_calibration[name] = calibrate_null_dependence(
            rubric, scorer, m=NULL_M, trials=args.null_trials, seed=SEED + idx,
        )
        cal = null_calibration[name]
        print(
            f"      [{name}] q_K={cal['q_K']:.5f} "
            f"P_joint={cal['P_joint']:.6f} κ={cal['kappa']} "
            f"（错配事件 {cal['mismatch_events']}/{cal['pair_opportunities']}，"
            f"相邻共现 {cal['joint_events']}/{cal['adjacent_pairs']}）"
        )

    print("[4/4] 红线检查 + 写报告...")
    violations = check_decomposition_red_lines(decomposition, null_calibration)
    for v in violations:
        print(f"      [违例] {v}")
    if not violations:
        print("      全部红线满足：恒等式精确闭合 + 可证伪方向 + κ 正相依。")

    report = {
        "experiment": "§5.3/§6.5 定理 1-A 优势分解标定（P(S)/P(M\\D)/q_K/κ）",
        "benchmark": "DDB-L1 负例 + DDB-L1.5 伪冲突负例（C=∅ 底空间）+ 受控零假设合成案件",
        "seed": SEED,
        "pool": {
            "l1_per_polarity": args.cases,
            "l1_defect_only": n_l1_def,
            "l1_clean": n_l1_clean,
            "l15": n_l15,
            "total": len(pool),
        },
        "decomposition": decomposition,
        "null_calibration": null_calibration,
        "red_line_violations": violations,
        "generated_at": datetime.now().isoformat(),
    }
    out_path = Path(__file__).resolve().parents[1] / "ddb" / (
        "advantage_decomposition_report.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"      报告：{out_path}")

    if violations:
        print("结论：定理 1-A 分解实验红线违例，需按违例项排查。")
        return 1
    print("结论：定理 1-A 优势分解实验通过——ABP 优势可测、恒等式闭合、κ>1。")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 —— 管线异常统一归口退出码 2
        print(f"运行异常：{exc}", file=sys.stderr)
        raise
