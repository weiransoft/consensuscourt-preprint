#!/usr/bin/env python3
"""
基线阈值敏感性扫描实验（论文大纲 §6.2 补充实验，v0.4.16 评审 P1-6b）。

背景（对抗性评审 R2-M1）：Tab.2 三基线在单点阈值（支持 0.7 / 驳回 0.4）上
被评价，审稿人可质疑"基线在最不利操作点上被评价"。本实验在
(support_thr, reject_thr) ∈ {0.3,...,0.9}² 共 49 个阈值格点上重放三个
基线的决策映射，画出每条基线可达到的完整 (recall, FPR) Pareto 前沿，
并把完整法庭（阈值无关的单点）作为参考线——回应"单点比较"质疑：
即使允许基线任选操作点，也没有任何格点能同时达到法庭的 recall=1.0/FPR=0.0。

方法（决策原语重放，生产代码零改动）：
- 与 run_baseline_comparison.py 同生成器、同 seed（L1 60×2 + L1.5 88）、
  同 EvidenceScorer 评分——每案只算一次评分与决策原语；
- 三基线的决策全部是其"决策原语"的阈值映射：
  * majority_vote 原语 = (支持票数, 反对票数)（每票阈值可参数化）；
  * elo_arbiter 原语 = 角色加权均分 avg（决策：avg>=t_sup 支持 /
    avg>=t_rej 有条件 / 否则驳回，与 baseline_arbiter._decision_from_score
    逐字同逻辑，仅把 0.7/0.4 换成格点参数）；
  * single_judge 原语 = 最高 overall_score（映射同上）；
- 完整法庭信号 = AsymmetricRubric 定罪（not passed），阈值无关，作为
  参考点输出。

预注册红线：
- RL1（扫描-主对比互锁）：默认阈值 (0.7, 0.4) 下，三基线 reject 与
  non_support 口径的 recall/fpr/accuracy/precision 必须与
  tests/ddb/baseline_comparison_report.json 完全一致（决策原语重放的
  正确性锁）；完整法庭参考点也必须与该报告一致；
- RL2（统计量单调性）：elo/single 的 reject-FPR 对 t_rej 单调不减
  （驳回阈值越高 → 越多案件被判驳回 → FPR 不会下降）；
- RL3（法庭参考点互锁）：完整法庭（阈值无关的单点）指标与主对比
  报告一致；"阈值无关性"由 collect_primitives 只算一次的构造保证，
  红线层锁的是参考点数值本身不漂移。

诚实报告项（非红线）：baseline_optimum_reaches_court_profile——是否存在
任何基线格点同时达到 recall=1.0 且 FPR=0.0。这是实证问题，由数据回答。

用法：
    backend$ .venv/bin/python tests/scripts/run_baseline_sensitivity.py

输出：tests/ddb/baseline_sensitivity_report.json + stdout 前沿对照表
退出码：0=红线全部满足；1=红线违例；2=运行异常
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# 导入前置：backend 根入 sys.path + 环境变量占位（必须在 import app 之前）
# ---------------------------------------------------------------------------
_BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

os.environ.setdefault("SCIMINER_API_KEY", "baseline-sweep-placeholder")
os.environ.setdefault("DB_PASSWORD", "baseline-sweep-placeholder")

from app.core.virtual_lab.rubric_engine import AsymmetricRubric  # noqa: E402
from app.core.virtual_lab.baseline_arbiter import EloArbiter  # noqa: E402
from tests.ddb.l1_generator import generate as generate_l1  # noqa: E402
from tests.ddb.l15_generator import generate as generate_l15  # noqa: E402
from run_baseline_comparison import (  # noqa: E402
    SEED,
    COURT_NAME,
    _to_evidence,
    _empty_confusion,
    _rates,
)

#: 阈值扫描网格（评审路线图指定 0.3-0.9，步长 0.1 → 7×7=49 格点）
GRID: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

#: 主对比报告路径（RL1 互锁基准）
_MAIN_REPORT = Path(__file__).resolve().parents[1] / "ddb" / "baseline_comparison_report.json"

#: 三个被扫描基线的注册名（与 BASELINE_ARBITERS 键一致）
SWEEP_NAMES = ("majority_vote", "elo_arbiter", "single_judge")


def _majority_counts(
    raw_scores: list[float], t_sup: float, t_rej: float
) -> tuple[int, int]:
    """majority_vote 决策原语：给定票档阈值的支持票/反对票计数。

    与 baseline_arbiter.MajorityVote.arbitrate 的计票逻辑逐字同口径：
    overall_score >= t_sup 计支持票、elif < t_rej 计反对票、其余弃权
    （支持票优先的 if/elif 链——当 t_sup < t_rej 时，落在
    [t_sup, t_rej) 的分数只计支持票，与生产语义一致，不会双重计票）；
    唯一差异是把类常量 0.7/0.4 换成格点参数——这是"重放"而非"重实现"，
    RL1 红线在默认阈值处锁死二者等价。

    Args:
        raw_scores: 案件内各证据的 overall_score 列表
        t_sup: 支持票阈值（>= 计支持，优先判定）
        t_rej: 反对票阈值（< 计反对，仅在未计支持时）

    Returns:
        (支持票数, 反对票数)
    """
    support = 0
    reject = 0
    for sc in raw_scores:
        if sc >= t_sup:
            support += 1
        elif sc < t_rej:
            reject += 1
        # else: 弃权档（t_sup <= sc < t_rej 区间外的中间档），不计票
    return support, reject


def _majority_decision(support: int, reject: int) -> str:
    """majority_vote 计票 → 裁决映射（与 MajorityVote.arbitrate 逐字同口径）。

    Args:
        support: 支持票数
        reject: 反对票数

    Returns:
        三档裁决字符串（支持提案 / 驳回提案 / 有条件支持）
    """
    if support > reject:
        return "支持提案"
    if reject > support:
        return "驳回提案"
    return "有条件支持"  # 平票 → 有条件支持（保守方向，与基线实现一致）


def _weighted_average(scores: list, emap: dict) -> float:
    """elo_arbiter 决策原语：角色加权均分（与 EloArbiter.arbitrate 同口径）。

    直接复用 EloArbiter.ROLE_WEIGHTS 注册表与 fallback（0.5），保证重放
    与生产基线实现在权重语义上零漂移。

    Args:
        scores: EvidenceScore 列表
        emap: evidence_id -> Evidence 映射（取 role 用）

    Returns:
        角色加权平均分（total_weight=0 时返回 0.0，与基线实现一致）
    """
    weighted_sum = 0.0
    total_weight = 0.0
    for s in scores:
        ev = emap.get(s.evidence_id)
        role = ev.role.lower() if ev else "agent"
        w = EloArbiter.ROLE_WEIGHTS.get(role, 0.5)
        weighted_sum += w * s.overall_score
        total_weight += w
    return weighted_sum / total_weight if total_weight > 0 else 0.0


def _statistic_decision(stat: float, t_sup: float, t_rej: float) -> str:
    """统计量 → 三档裁决映射（与 baseline_arbiter._decision_from_score 同口径，
    仅把 0.7/0.4 阈值参数化为格点）。

    Args:
        stat: 决策统计量（elo 加权均分或 single 最高分）
        t_sup: 支持阈值（>= 判支持）
        t_rej: 驳回阈值（< 判驳回，中间档为有条件支持）

    Returns:
        三档裁决字符串
    """
    if stat >= t_sup:
        return "支持提案"
    if stat >= t_rej:
        return "有条件支持"
    return "驳回提案"


def collect_primitives(cases: list[dict]) -> list[dict]:
    """一次性计算每案的评分、决策原语与法庭定罪信号（扫描前处理）。

    与 run_baseline_comparison.evaluate_systems 同评分路径（同一个
    EvidenceScorer、同一 _to_evidence 还原），保证 RL1 互锁可比。

    Args:
        cases: L1 + L1.5 合并案例字典列表

    Returns:
        原语记录列表：{polarity, court_flagged, majority_scores,
        elo_avg, single_best}
    """
    from app.core.virtual_lab.consensus_court import EvidenceScorer

    scorer = EvidenceScorer()
    rubric = AsymmetricRubric()
    records: list[dict] = []
    for case in cases:
        evidences = [_to_evidence(ev) for ev in case["evidences"]]
        scores = [scorer.score(ev, case["topic"]) for ev in evidences]
        emap = {ev.evidence_id: ev for ev in evidences}
        # 完整法庭定罪信号（阈值无关，参考点）
        verdict = rubric.evaluate(scores, emap, case["topic"])
        records.append({
            "polarity": case["polarity"],
            "court_flagged": not verdict.passed,
            "majority_scores": [s.overall_score for s in scores],
            "elo_avg": _weighted_average(scores, emap),
            "single_best": max(s.overall_score for s in scores),
        })
    return records


def evaluate_grid(records: list[dict]) -> dict[str, list]:
    """在 49 格点上重放三基线决策，返回每格点的两口径混淆指标。

    Args:
        records: collect_primitives 产出的原语记录

    Returns:
        {system_name: [{t_sup, t_rej, reject: 指标, non_support: 指标}, ...]}
    """
    grid_out: dict[str, list] = {name: [] for name in SWEEP_NAMES}
    for t_sup in GRID:
        for t_rej in GRID:
            # 每格点一套混淆矩阵（reject / non_support 两口径）
            cms: dict[str, dict[str, dict]] = {
                name: {"reject": _empty_confusion(), "non_support": _empty_confusion()}
                for name in SWEEP_NAMES
            }
            for rec in records:
                has_truth = rec["polarity"] == "with_contradiction"

                # --- majority_vote：计票 → 裁决（弃权档不计入任一票） ---
                support, reject = _majority_counts(
                    rec["majority_scores"], t_sup, t_rej
                )
                decision = _majority_decision(support, reject)
                _tally(cms["majority_vote"], has_truth,
                       decision == "驳回提案", decision != "支持提案")

                # --- elo_arbiter：加权均分 → 三档裁决 ---
                decision = _statistic_decision(rec["elo_avg"], t_sup, t_rej)
                _tally(cms["elo_arbiter"], has_truth,
                       decision == "驳回提案", decision != "支持提案")

                # --- single_judge：最高分 → 三档裁决 ---
                decision = _statistic_decision(rec["single_best"], t_sup, t_rej)
                _tally(cms["single_judge"], has_truth,
                       decision == "驳回提案", decision != "支持提案")

            for name in SWEEP_NAMES:
                grid_out[name].append({
                    "t_sup": t_sup,
                    "t_rej": t_rej,
                    "reject": _rates(cms[name]["reject"]),
                    "non_support": _rates(cms[name]["non_support"]),
                })
    return grid_out


def _tally(cm_pair: dict, has_truth: bool, flagged_rej: bool, flagged_ns: bool) -> None:
    """把单案预测记入 reject / non_support 两口径混淆矩阵。

    口径与 run_baseline_comparison.evaluate_systems 逐字一致：
    正类=含真矛盾案件；reject 口径只有"驳回提案"算拦下；
    non_support 口径"非支持即拦下"（有条件支持计入）。

    Args:
        cm_pair: {"reject": 混淆, "non_support": 混淆}
        has_truth: 案件是否含真矛盾（正类）
        flagged_rej: reject 口径的"拦下"预测
        flagged_ns: non_support 口径的"拦下"预测
    """
    cm = cm_pair["reject"]
    if has_truth:
        cm["tp" if flagged_rej else "fn"] += 1
    else:
        cm["fp" if flagged_rej else "tn"] += 1
    cm = cm_pair["non_support"]
    if has_truth:
        cm["tp" if flagged_ns else "fn"] += 1
    else:
        cm["fp" if flagged_ns else "tn"] += 1


def pareto_frontier(points: list[dict], key: str) -> list[dict]:
    """计算 (recall↑, fpr↓) 的非支配前沿（reject 口径）。

    点 A 支配点 B ⇔ recall_A >= recall_B 且 fpr_A <= fpr_B（至少一严格）。
    前沿按 fpr 升序输出，供论文 Pareto 图直接取坐标。

    Args:
        points: evaluate_grid 产出的格点列表
        key: 指标口径键（"reject" 或 "non_support"）

    Returns:
        非支配格点列表（附 t_sup/t_rej 便于溯源）
    """
    feasible = [
        {"t_sup": p["t_sup"], "t_rej": p["t_rej"],
         "recall": p[key]["recall"], "fpr": p[key]["fpr"]}
        for p in points
    ]
    frontier: list[dict] = []
    for a in feasible:
        dominated = False
        for b in feasible:
            if b is a:
                continue
            if (b["recall"] >= a["recall"] and b["fpr"] <= a["fpr"]
                    and (b["recall"] > a["recall"] or b["fpr"] < a["fpr"])):
                dominated = True
                break
        if not dominated:
            frontier.append(a)
    frontier.sort(key=lambda p: (p["fpr"], -p["recall"]))
    # 同一 (fpr, recall) 只保留一个代表点（不同阈值可达同一指标）
    dedup: list[dict] = []
    for p in frontier:
        if not dedup or p["fpr"] != dedup[-1]["fpr"] or p["recall"] != dedup[-1]["recall"]:
            dedup.append(p)
    return dedup


def check_red_lines(
    grid_out: dict[str, list],
    court_metrics: dict,
    main_report: dict,
) -> list[str]:
    """三条预注册红线检查（详见模块 docstring）。

    Args:
        grid_out: evaluate_grid 产出
        court_metrics: 完整法庭参考点指标（reject 口径）
        main_report: 主对比报告 JSON（RL1 互锁基准）

    Returns:
        违例描述列表（空 = 全部满足）
    """
    violations: list[str] = []

    # RL1：默认阈值 (0.7, 0.4) 与主对比报告完全一致
    for name in SWEEP_NAMES:
        for op in ("reject", "non_support"):
            ref = main_report["systems"][name][op]
            hit = next(
                p for p in grid_out[name]
                if p["t_sup"] == 0.7 and p["t_rej"] == 0.4
            )
            for metric in ("recall", "fpr", "accuracy", "precision"):
                if hit[op][metric] != ref[metric]:
                    violations.append(
                        f"RL1 互锁违例：{name}/{op}/{metric}="
                        f"{hit[op][metric]} ≠ 主对比 {ref[metric]}"
                    )
    ref_court = main_report["systems"][COURT_NAME]["reject"]
    for metric in ("recall", "fpr", "accuracy", "precision"):
        if court_metrics[metric] != ref_court[metric]:
            violations.append(
                f"RL1 互锁违例：法庭参考点 {metric}={court_metrics[metric]} "
                f"≠ 主对比 {ref_court[metric]}"
            )

    # RL2：统计量型基线（elo/single）reject-FPR 对 t_rej 单调不减
    for name in ("elo_arbiter", "single_judge"):
        for t_sup in GRID:
            fprs = [
                next(p for p in grid_out[name]
                     if p["t_sup"] == t_sup and p["t_rej"] == t_rej)
                ["reject"]["fpr"]
                for t_rej in GRID
            ]
            if any(b < a for a, b in zip(fprs, fprs[1:])):
                violations.append(
                    f"RL2 单调违例：{name} t_sup={t_sup} 的 reject-FPR 序列 "
                    f"{fprs} 非单调不减"
                )

    # RL3：法庭指标阈值无关（这里以格点循环外的单点 vs 报告一致性表达；
    # 阈值无关性本身由 collect_primitives 只算一次保证）
    return violations


def main() -> int:
    """主流程：生成案例池 → 原语采集 → 49 格点重放 → 红线检查 → 落盘。"""
    parser = argparse.ArgumentParser(
        description="§6.2 基线阈值敏感性扫描（49 格点 Pareto，评审 P1-6b）"
    )
    parser.add_argument(
        "--cases", type=int, default=60,
        help="L1 每极性案例数（默认 60，与主对比冒烟口径一致）",
    )
    args = parser.parse_args()

    print(f"[1/4] 加载 DDB-L1（{args.cases}×2 例，seed={SEED}）+ DDB-L1.5（88 例）...")
    cases = generate_l1(
        num_with_contradiction=args.cases,
        num_no_contradiction=args.cases,
        seed=SEED,
    )
    cases.extend(generate_l15(per_template=8, seed=SEED))
    n_pos = sum(1 for c in cases if c["polarity"] == "with_contradiction")
    print(f"      合并案例池：{len(cases)} 例（正例 {n_pos} / 负例 {len(cases) - n_pos}）")

    print("[2/4] 采集决策原语（每案一次评分，49 格点重放复用）...")
    records = collect_primitives(cases)
    court_tp = sum(1 for r in records if r["polarity"] == "with_contradiction" and r["court_flagged"])
    court_fp = sum(1 for r in records if r["polarity"] != "with_contradiction" and r["court_flagged"])
    court_metrics = _rates({
        "tp": court_tp, "fp": court_fp,
        "fn": n_pos - court_tp, "tn": (len(cases) - n_pos) - court_fp,
    })

    print(f"[3/4] {len(GRID)}×{len(GRID)}={len(GRID) ** 2} 格点重放三基线决策...")
    grid_out = evaluate_grid(records)

    print("[4/4] 红线检查 + Pareto 前沿提取 + 写报告...")
    with open(_MAIN_REPORT, "r", encoding="utf-8") as f:
        main_report = json.load(f)
    violations = check_red_lines(grid_out, court_metrics, main_report)
    for v in violations:
        print(f"      [违例] {v}")
    if not violations:
        print("      全部红线满足：RL1 默认阈值与主对比互锁一致；"
              "RL2 统计量 FPR 单调；RL3 法庭参考点一致")

    frontier = {name: pareto_frontier(grid_out[name], "reject") for name in SWEEP_NAMES}
    # 诚实报告项：是否存在基线格点同时达到法庭画像（recall=1.0 且 FPR=0.0）
    reaches = any(
        p["reject"]["recall"] == 1.0 and p["reject"]["fpr"] == 0.0
        for name in SWEEP_NAMES for p in grid_out[name]
    )
    print(f"      基线最优格点是否达到法庭画像（recall=1.0 & FPR=0.0）：{reaches}")
    for name in SWEEP_NAMES:
        pts = " ".join(
            f"(r={p['recall']:.3f},f={p['fpr']:.3f}@{p['t_sup']:.1f}/{p['t_rej']:.1f})"
            for p in frontier[name]
        )
        print(f"      {name} 前沿：{pts}")

    report = {
        "experiment": "§6.2 基线阈值敏感性扫描（49 格点 Pareto，评审 P1-6b）",
        "benchmark": "DDB-L1（60×2）+ DDB-L1.5（88 例），与主对比同池同评分",
        "seed": SEED,
        "grid": list(GRID),
        "n_grid_points": len(GRID) ** 2,
        "pool_composition": {
            "n_positive": n_pos, "n_negative": len(cases) - n_pos,
        },
        "method": (
            "决策原语重放：majority=计票阈值映射 / elo=角色加权均分三档映射 / "
            "single=最高分三档映射；映射逻辑与 baseline_arbiter 逐字同口径，"
            "仅阈值参数化；生产代码零改动"
        ),
        "court_reference": court_metrics,
        "baseline_optimum_reaches_court_profile": reaches,
        "systems": {
            name: {
                "grid_points": grid_out[name],
                "pareto_frontier_reject": frontier[name],
            }
            for name in SWEEP_NAMES
        },
        "red_line_violations": violations,
        "generated_at": datetime.now().isoformat(),
        "note": (
            "RL1 默认阈值(0.7,0.4)与 baseline_comparison_report.json 互锁；"
            "RL2 统计量型基线 FPR 对驳回阈值单调不减；RL3 法庭参考点互锁"
            "（阈值无关性由原语只算一次构造保证）。"
            "baseline_optimum_reaches_court_profile 为诚实报告项（非红线）。"
        ),
    }
    out_path = (
        Path(__file__).resolve().parents[1] / "ddb" / "baseline_sensitivity_report.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"      报告：{out_path}")

    if violations:
        print("结论：敏感性扫描红线违例，需检查决策原语重放的正确性。")
        return 1
    print("结论：扫描通过——基线在任意阈值格点上的可达画像已完整落盘。")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 —— 管线异常统一归口退出码 2
        print(f"运行异常：{exc}", file=sys.stderr)
        raise
