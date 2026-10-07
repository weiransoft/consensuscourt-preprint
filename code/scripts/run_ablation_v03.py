#!/usr/bin/env python3
"""
v0.3 三机制 + v0.4.5 位置加权消融冒烟实验脚本。

对应论文大纲 §6.4 消融设计（v0.3 三项机制 = 3 个一阶自变量 +
v0.4.5 位置加权 = 第四自变量）与 §6.3 指标口径。本脚本是**冒烟级**
（60 例小样本 × 6 配置），用于：
1. 验证消融管线端到端可运行（L1 生成 → 8 元组解包兼容 → ABP 评估）；
2. 产出 6 个关键配置的召回/FPR 对照表，验证保守公理：
   - 机制全开 FPR ≤ 全关 FPR（新机制只降假阳性）；
   - 机制全开召回 ≥ v0.2 基线召回 - 容差（不漏检红线）；
   - 位置加权开启时定罪置信度折扣效应可观测。
正式 13 项消融（13-600 例全量 + 全部 2^4 组合 + 显著性检验）
在第 3 月主实验中运行（同管线放大规模即可）。

实验配置（配置名 → 四开关）：
- baseline_v02   : aspect=off, gate=off, intensity=off, position=off（v0.2 等价基线）
- aspect_only    : aspect=on,  gate=off, intensity=off, position=off
- gate_only      : aspect=off, gate=on,  intensity=off, position=off
- position_only  : aspect=off, gate=off, intensity=off, position=on（v0.4.5 新增）
- full_v03       : aspect=on,  gate=on,  intensity=on,  position=off（v0.3 完整机制）
- full_v04       : aspect=on,  gate=on,  intensity=on,  position=on（v0.4 完整机制）

用法（默认 60 例）：
    backend$ .venv/bin/python tests/scripts/run_ablation_v03.py
    backend$ .venv/bin/python tests/scripts/run_ablation_v03.py --cases 600
    backend$ .venv/bin/python tests/scripts/run_ablation_v03.py --l15
    backend$ .venv/bin/python tests/scripts/run_ablation_v03.py --scorer-ablation

输出：tests/ddb/ablation_v03_report.json + stdout 汇总表
退出码：0=红线全部满足；1=红线违例；2=运行异常
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# 导入前置：backend 根入 sys.path + 环境变量占位（必须在 import app 之前）
# ---------------------------------------------------------------------------
_BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

os.environ.setdefault("SCIMINER_API_KEY", "v03-ablation-placeholder")
os.environ.setdefault("DB_PASSWORD", "v03-ablation-placeholder")

from app.core.virtual_lab.consensus_court import (  # noqa: E402
    Evidence,
    EvidenceScore,
    EvidenceScorer,
    HumanBOSSOverride,
    IndependentReviewer,
    PIAgentArbiter,
    RuleEngine,
)
from app.core.virtual_lab.rubric_engine import (  # noqa: E402
    AsymmetricRubric,
    EvidencePositionWeighter,
)
from tests.ddb.l1_generator import generate  # noqa: E402

#: 实验种子（与 pilot 一致，写入报告保证可复现）
SEED = 42

#: 消融配置表（§6.4 一阶效应的关键格点）
#: v0.4.5 扩展：position 维度（EvidencePositionWeighter）作为第四自变量，
#: 用于 §6.4 "去掉位置加权"消融。position_weighted_confidence 指标仅在
#: enable_position_weighting=True 的配置下计算，用于位置加权一阶效应验证。
CONFIGS: dict[str, dict[str, bool]] = {
    "baseline_v02": {"enable_aspect": False, "enable_validity_gate": False, "enable_intensity": False, "enable_position_weighting": False},
    "aspect_only": {"enable_aspect": True, "enable_validity_gate": False, "enable_intensity": False, "enable_position_weighting": False},
    "gate_only": {"enable_aspect": False, "enable_validity_gate": True, "enable_intensity": False, "enable_position_weighting": False},
    "position_only": {"enable_aspect": False, "enable_validity_gate": False, "enable_intensity": False, "enable_position_weighting": True},
    "full_v03": {"enable_aspect": True, "enable_validity_gate": True, "enable_intensity": True, "enable_position_weighting": False},
    "full_v04": {"enable_aspect": True, "enable_validity_gate": True, "enable_intensity": True, "enable_position_weighting": True},
}

#: 保守公理容差：全开召回相对 v0.2 基线的允许下降（60 例冒烟样本下
#: 1 例 = 1/30 ≈ 0.034；词典缺口导致的超限漏检即红线违例）
RECALL_TOLERANCE = 0.05


# ---------------------------------------------------------------------------
# scorer 级消融（v0.4.15 新增，大纲 §6.4 第 4/8 行：来源等权 / 时效衰减关闭）
#
# 与 6 个 rubric 开关配置的层次差异：rubric 配置消融的是"矛盾对检测内核"
# 的机制开关；scorer 级消融在**机制全开（full_v03 开关组合）不变**的前提
# 下，对 EvidenceScorer 的评分输出做实验侧后处理——**生产代码零改动**，
# 唯一变量 = 评分公式的单一分量，符合 §6.2 单变量隔离方法论：
# - no_source_pedigree（来源等权）：全部证据 source_score 置统一中值
#   （SOURCE_WEIGHTS 八键取值的中位数，随生产权重表自动同步），
#   抹平来源谱系差异后按 0.4/0.2/0.4 公式重算 overall；
# - no_recency（时效衰减关闭）：全部证据 recency_score 置 1.0
#   （时间衰减 exp(-age/H) 的上界，关闭=永不过时）后重算 overall。
#
# 效应通道说明（预印本主张严谨性）：overall_score 在 rubric 定罪通道
# （关键词对 + 硬约束短路）中**不参与**定罪判定（通道与评分解耦是
# ABP 的结构性设计），只影响无矛盾时的置信度（_compute_confidence 的
# avg overall 项）→ 消融的可观测效应在**置信度/升级层**：FPR/召回
# 的单调性红线（消融不得更优）+ 置信度效应方向可解释。
# ---------------------------------------------------------------------------

#: scorer 消融模式表：模式名 → 人读描述（写入报告供追溯）
SCORER_ABLATION_MODES: dict[str, str] = {
    "no_source_pedigree": "来源等权：source_score 置统一中值后重算 overall（§6.4 消融第 4 行）",
    "no_recency": "时效衰减关闭：recency_score 置 1.0 后重算 overall（§6.4 消融第 8 行）",
}

#: 来源等权模式使用的统一中值分数：取生产 EvidenceScorer.SOURCE_WEIGHTS
#: 八键（pubmed/chembl/uniprot/drugbank/experiment/llm/agent_opinion/unknown）
#: 的中位数——从生产权重表程序化推导，生产表更新时自动同步，避免硬编码漂移
UNIFORM_SOURCE_SCORE: float = round(median(EvidenceScorer.SOURCE_WEIGHTS.values()), 4)


def _recompute_overall(source_score: float, recency_score: float, relevance_score: float) -> float:
    """
    按生产 EvidenceScorer 的加权公式重算 overall 分数。

    与 consensus_court.py EvidenceScorer.score() L199 的公式逐项一致：
    overall = 0.4×source + 0.2×recency + 0.4×relevance，保留 3 位小数。
    单独抽出为纯函数是为了：(1) 消融置换分量后重算有唯一实现；
    (2) 单元测试可判别校验"重算函数与生产公式一致"（未置换时输出
    必须与原 overall 完全相等）。

    Args:
        source_score: 来源可信度分数（0-1）
        recency_score: 时效性分数（0-1）
        relevance_score: 相关性分数（0-1）

    Returns:
        重算后的 overall 分数（3 位小数）
    """
    return round(source_score * 0.4 + recency_score * 0.2 + relevance_score * 0.4, 3)


def apply_scorer_ablation(scores: list, mode: str) -> list:
    """
    对评分列表执行 scorer 级消融置换（返回新对象，不修改输入）。

    置换规则（唯一变量隔离，其余分量与 issues 原样保留）：
    - no_source_pedigree：source_score 置 UNIFORM_SOURCE_SCORE（来源等权），
      issues 原样保留（"来源未知"标记的是证据本身的未知状态，与权重
      是否等权无关，不应随消融消失）；
    - no_recency：recency_score 置 1.0（关闭时间衰减），并从 issues 中
      移除"证据时效性较低"——衰减关闭的-counterfactual 语义下该问题
      不会被生产评分器报出，保留会造成 pending 计数失真。

    Args:
        scores: EvidenceScore 列表（EvidenceScorer.score 的输出）
        mode: SCORER_ABLATION_MODES 的 key

    Returns:
        置换后的新 EvidenceScore 列表（overall 已按生产公式重算）

    Raises:
        ValueError: 未知消融模式
    """
    if mode not in SCORER_ABLATION_MODES:
        raise ValueError(f"未知 scorer 消融模式: {mode}")

    out = []
    for s in scores:
        if mode == "no_source_pedigree":
            new_source = UNIFORM_SOURCE_SCORE
            new_recency = s.recency_score
            new_issues = list(s.issues)
        else:  # no_recency
            new_source = s.source_score
            new_recency = 1.0
            # 时效衰减关闭的语义：低时效问题不应存在（counterfactual 一致性）
            new_issues = [i for i in s.issues if i != "证据时效性较低"]
        out.append(
            EvidenceScore(
                evidence_id=s.evidence_id,
                source_score=round(new_source, 3),
                recency_score=round(new_recency, 3),
                relevance_score=s.relevance_score,
                overall_score=_recompute_overall(new_source, new_recency, s.relevance_score),
                issues=new_issues,
            )
        )
    return out


def check_scorer_ablation_red_lines(full_metrics: dict, abl_metrics: dict, mode: str) -> list[str]:
    """
    scorer 级消融红线检查器（独立纯函数，供单元测试伪造违例反测）。

    红线口径（大纲 §6.4 第 4/8 行 + §6.2 单变量方法论）：
    - R1 单调性（两模式通用）：消融格点 FPR 不得高于 full_v03（等权/关衰减
      不得降低假阳性拦截能力）、召回不得低于 full_v03（评分后处理不改变
      定罪通道，若改变即实现错误）；
    - R2 效应方向可解释：
      · no_recency：关闭衰减后 recency_score 只会从 exp(-age/H) 抬升到 1.0
        → 通过案例平均置信度必须 ≥ full_v03（方向先验唯一）；
      · no_source_pedigree：置平必须实际改变评分（消融生效性）——通过案例
        平均置信度必须 ≠ full_v03；若恒等说明 L1 案例的来源谱系未参与
        置信度计算或消融未生效，实验无信息量。方向（抬升/压低取决于
        证据池的来源构成）如实写入报告，不预设符号。

    Args:
        full_metrics: full_v03 开关组合的指标字典（含 recall/fpr/avg_pass_confidence）
        abl_metrics: 消融格点的指标字典（同口径）
        mode: 消融模式名（SCORER_ABLATION_MODES 的 key）

    Returns:
        违例描述列表（空 = 红线全部满足）
    """
    violations: list[str] = []

    # 架构师审查加固（v0.4.15）：模式名校验——非法模式此前会静默落入
    # no_source_pedigree 的 else 分支，使红线检查形同虚设；现显式拒绝。
    if mode not in SCORER_ABLATION_MODES:
        raise ValueError(f"未知 scorer 消融模式: {mode!r}，合法值: {sorted(SCORER_ABLATION_MODES)}")

    # R1 单调性：消融不得更优（FPR 不升 / 召回不降）
    # 架构师审查加固（v0.4.15）：full_metrics["fpr"] 补 None 守卫（与召回侧
    # 双侧守卫对称），避免指标缺失时抛 TypeError 而非报告违例。
    if (abl_metrics["fpr"] is not None and full_metrics["fpr"] is not None
            and abl_metrics["fpr"] > full_metrics["fpr"]):
        violations.append(
            f"scorer 消融 FPR 单调性违例：{mode} fpr={abl_metrics['fpr']} > "
            f"full_v03 fpr={full_metrics['fpr']}（消融不得降低假阳性拦截能力）"
        )
    if (abl_metrics["recall"] is not None and full_metrics["recall"] is not None
            and abl_metrics["recall"] < full_metrics["recall"]):
        violations.append(
            f"scorer 消融召回单调性违例：{mode} recall={abl_metrics['recall']} < "
            f"full_v03 recall={full_metrics['recall']}"
            "（评分后处理不触碰定罪通道，召回变化即实现错误）"
        )

    # R2 效应方向可解释（通过案例平均置信度通道）
    full_conf = full_metrics.get("avg_pass_confidence")
    abl_conf = abl_metrics.get("avg_pass_confidence")
    # conf=None（零通过案例）时置信度通道不可观测，跳过 R2 而非记违例——
    # 此为口径性跳过：置信度比较只在两侧均有通过案例时才有意义。
    if full_conf is not None and abl_conf is not None:
        if mode == "no_recency":
            if abl_conf < full_conf:
                violations.append(
                    f"scorer 消融效应方向违例：no_recency 通过案例平均置信度 "
                    f"{abl_conf} < full_v03 {full_conf}"
                    "（关闭时效衰减只能抬升 recency 分量，置信度不得下降）"
                )
        else:  # no_source_pedigree
            if abl_conf == full_conf:
                violations.append(
                    f"scorer 消融生效性违例：no_source_pedigree 通过案例平均置信度 "
                    f"与 full_v03 恒等（{full_conf}）——来源谱系未参与评分或消融未生效"
                )
    return violations


def _to_evidence(ev: dict) -> Evidence:
    """把 L1 案例证据字典无损还原为 consensus_court.Evidence 对象。

    与 pilot 的唯一差异：unknown 来源证据补占位引用——该处理只影响
    RuleEngine 的 agent_opinion 规则，不影响 ABP 关键词/aspect/gate 通道。

    Args:
        ev: L1 案例中的证据字典

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


def evaluate_config(
    cases: list[dict],
    switches: dict[str, bool],
    score_ablation: Optional[str] = None,
) -> dict:
    """
    在整份 L1 基准上以指定开关组合运行 ABP，汇总案例级判别指标。

    指标口径（与 pilot H2a 一致）：
    - recall      : 有矛盾案例中被定罪（passed=False）的比例
    - fpr         : 无矛盾案例中被误定罪的比例
    - ambig_rate  : 无矛盾案例中出现 pending_pairs（AMBIGUOUS 降级）的比例
    - intensity_dist: 定罪矛盾对的强度分布（仅机制开启时非平凡）
    - avg_weighted_confidence : 定罪案例的位置加权平均置信度
      （enable_position_weighting=True 时按 EvidencePositionWeighter
      对最终置信度做位置折扣后取均值；否则等于未加权平均）
    - avg_pass_confidence : 通过案例（passed=True）的平均置信度——
      rubric 无矛盾路径的 _compute_confidence 输出（avg overall + 数量
      加成），是 scorer 级消融（v0.4.15）的可观测效应通道

    Args:
        cases: L1 案例字典列表
        switches: 消融开关（含三机制开关 + enable_position_weighting）
        score_ablation: scorer 级消融模式（SCORER_ABLATION_MODES 的 key，
            None = 不置换，走原始生产评分）

    Returns:
        指标字典
    """
    scorer = EvidenceScorer()
    # 分离位置加权开关（不属于 AsymmetricRubric 三开关，需剔除后传入）
    rubric_switches = {k: v for k, v in switches.items() if k != "enable_position_weighting"}
    rubric = AsymmetricRubric(**rubric_switches)
    pos_weighter = (
        EvidencePositionWeighter()
        if switches.get("enable_position_weighting", False)
        else None
    )

    n_pos_hit = 0
    n_neg_hit = 0
    n_neg_ambig = 0
    intensity_dist: dict[str, int] = {}
    missed_positives: list[str] = []
    weighted_confidences: list[float] = []
    pass_confidences: list[float] = []

    # 极性归属用 polarity 字段直接判断（案例对象同一性不依赖值相等）
    positives = [c for c in cases if c["polarity"] == "with_contradiction"]
    negatives = [c for c in cases if c["polarity"] != "with_contradiction"]
    positive_ids = {c["case_id"] for c in positives}

    for case in cases:
        evidences = [_to_evidence(ev) for ev in case["evidences"]]
        scores = [scorer.score(ev, case["topic"]) for ev in evidences]
        # scorer 级消融：评分后处理置换（None = 原始生产评分，零改动）
        if score_ablation is not None:
            scores = apply_scorer_ablation(scores, score_ablation)
        emap = {ev.evidence_id: ev for ev in evidences}
        verdict = rubric.evaluate(scores, emap, case["topic"])
        convicted = not verdict.passed
        # 通过案例置信度：scorer 消融的效应通道（定罪案例恒 0.9，无信息量）
        if verdict.passed:
            pass_confidences.append(verdict.confidence)

        if case["case_id"] in positive_ids:
            if convicted:
                n_pos_hit += 1
            else:
                missed_positives.append(case["case_id"])
        else:
            if convicted:
                n_neg_hit += 1
            if verdict.pending_pairs:
                n_neg_ambig += 1

        for c in verdict.contradictions:
            key = str(c.intensity)
            intensity_dist[key] = intensity_dist.get(key, 0) + 1

        # 位置加权置信度：定罪案例的最终置信度按证据列表中的位置折扣
        if convicted:
            base_conf = verdict.confidence
            n_ev = len(evidences)
            if n_ev > 1 and pos_weighter:
                # 多证据取均值折扣（单证据=中间位置，不折扣）
                w_sum = 0.0
                for i in range(n_ev):
                    ratio = i / (n_ev - 1)
                    w_sum += pos_weighter.weight(ratio)
                avg_w = w_sum / n_ev
                weighted_conf = round(base_conf * avg_w, 4)
            else:
                weighted_conf = base_conf
            weighted_confidences.append(weighted_conf)

    avg_w_conf = (
        round(sum(weighted_confidences) / len(weighted_confidences), 4)
        if weighted_confidences
        else None
    )
    avg_pass_conf = (
        round(sum(pass_confidences) / len(pass_confidences), 4)
        if pass_confidences
        else None
    )

    return {
        "switches": dict(switches),
        "score_ablation": score_ablation,
        "n_positive": len(positives),
        "n_negative": len(negatives),
        "recall": round(n_pos_hit / len(positives), 4) if positives else None,
        "fpr": round(n_neg_hit / len(negatives), 4) if negatives else None,
        "ambiguous_negative_rate": (
            round(n_neg_ambig / len(negatives), 4) if negatives else None
        ),
        "intensity_distribution": intensity_dist,
        "avg_weighted_confidence": avg_w_conf,
        "avg_pass_confidence": avg_pass_conf,
        "missed_positives": missed_positives[:20],  # 报告留痕截断
    }


def evaluate_l15_config(cases: list[dict], switches: dict[str, bool]) -> dict:
    """
    在整份 L1.5 负例对照基准上以指定开关组合运行 ABP（v0.4.3 新增）。

    L1.5 集全部是无真矛盾负例（polarity=no_contradiction），指标口径：
    - fpr        : 被误定罪的比例（full_v03 必须 0 = FPR=0 红线）
    - ambig_rate : 出现 AMBIGUOUS 降级的比例（ambiguous_reference 类
                   在门禁开启时应整类降级，gate 关闭应归 0）
    - by_category: 按类别（cross_aspect / different_object_gate /
                   ambiguous_reference）的分机制一阶效应 FPR——
                   aspect_only 配置下 cross_aspect 类应漂移为定罪、
                   gate_only 配置下两类门禁类别应漂移为定罪

    Args:
        cases: L1.5 案例字典列表
        switches: AsymmetricRubric 三开关

    Returns:
        指标字典（含 by_category 分解）
    """
    scorer = EvidenceScorer()
    # 分离位置加权开关（不属于 AsymmetricRubric 三开关）
    rubric_switches = {k: v for k, v in switches.items() if k != "enable_position_weighting"}
    rubric = AsymmetricRubric(**rubric_switches)

    n_neg_hit = 0
    n_neg_ambig = 0
    by_category: dict[str, dict[str, int]] = {}
    convicted_ids: list[str] = []

    for case in cases:
        evidences = [_to_evidence(ev) for ev in case["evidences"]]
        scores = [scorer.score(ev, case["topic"]) for ev in evidences]
        emap = {ev.evidence_id: ev for ev in evidences}
        verdict = rubric.evaluate(scores, emap, case["topic"])
        convicted = not verdict.passed

        cat = case["category"]
        bucket = by_category.setdefault(
            cat, {"n": 0, "convicted": 0, "ambiguous": 0}
        )
        bucket["n"] += 1
        if convicted:
            n_neg_hit += 1
            bucket["convicted"] += 1
            convicted_ids.append(case["case_id"])
        if verdict.pending_pairs:
            n_neg_ambig += 1
            bucket["ambiguous"] += 1

    for bucket in by_category.values():
        bucket["fpr"] = (
            round(bucket["convicted"] / bucket["n"], 4) if bucket["n"] else None
        )

    return {
        "switches": dict(switches),
        "n_negative": len(cases),
        "fpr": round(n_neg_hit / len(cases), 4) if cases else None,
        "ambiguous_negative_rate": (
            round(n_neg_ambig / len(cases), 4) if cases else None
        ),
        "by_category": by_category,
        "convicted_ids": convicted_ids[:20],  # 报告留痕截断
    }


def evaluate_l4_config(cases: list[dict], switches: dict[str, bool]) -> dict:
    """
    在整份 L4 强度标注基准上以指定开关组合运行 ABP（v0.4.3 新增）。

    L4 集全部是有真矛盾的正例（polarity=with_contradiction），指标口径：
    - recall          : 被定罪的比例（full_v03 必须 = 1.0，零漏检红线）
    - intensity_match : 定罪强度 == gold_intensity 的比例（强度一致性红线）
    - intensity_match_by_category : 分类别的强度匹配（gap_same /
                        gap_unknown / nogap_same / nogap_unknown）
    - intensity_distribution : 定罪矛盾对的强度分布（强度一阶效应验证）

    Args:
        cases: L4 案例字典列表
        switches: AsymmetricRubric 三开关

    Returns:
        指标字典（含 by_category / intensity_distribution）
    """
    scorer = EvidenceScorer()
    # 分离位置加权开关（不属于 AsymmetricRubric 三开关）
    rubric_switches = {k: v for k, v in switches.items() if k != "enable_position_weighting"}
    rubric = AsymmetricRubric(**rubric_switches)

    n_total = len(cases)
    n_convicted = 0
    n_match = 0
    n_mismatch = 0
    match_by_cat: dict[str, dict[str, int]] = {}
    intensity_dist: dict[str, int] = {}
    mismatches: list[str] = []
    misses: list[str] = []

    for case in cases:
        evidences = [_to_evidence(ev) for ev in case["evidences"]]
        scores = [scorer.score(ev, case["topic"]) for ev in evidences]
        emap = {ev.evidence_id: ev for ev in evidences}
        verdict = rubric.evaluate(scores, emap, case["topic"])
        convicted = not verdict.passed

        cat = case["category"]
        bucket = match_by_cat.setdefault(cat, {"n": 0, "match": 0, "mismatch": 0})
        bucket["n"] += 1

        gold = case["ground_truth"]["gold_intensity"]
        gold_pair = tuple(case["ground_truth"]["gold_pair"])

        if not convicted:
            # 漏检（正例必判有罪）
            misses.append(case["case_id"])
            bucket["mismatch"] += 1
            n_mismatch += 1
            continue

        n_convicted += 1
        # 定位 gold_pair
        located = None
        for c in verdict.contradictions:
            pair = tuple(sorted([c.evidence_a_id, c.evidence_b_id]))
            if pair == tuple(sorted(gold_pair)):
                located = c
                break
        if located is None:
            pairs_out = [(c.evidence_a_id, c.evidence_b_id)
                         for c in verdict.contradictions]
            mismatches.append(
                f"{case['case_id']} gold_pair 未命中: gold={gold_pair} "
                f"输出={pairs_out}"
            )
            bucket["mismatch"] += 1
            n_mismatch += 1
            continue

        if located.intensity == gold:
            n_match += 1
            bucket["match"] += 1
            intensity_dist[str(gold)] = intensity_dist.get(str(gold), 0) + 1
        else:
            mismatches.append(
                f"{case['case_id']} 强度不一致: gold={gold} "
                f"输出={located.intensity}")
            bucket["mismatch"] += 1
            n_mismatch += 1

    return {
        "switches": dict(switches),
        "n_total": n_total,
        "n_convicted": n_convicted,
        "n_misses": len(misses),
        "n_mismatches": len(mismatches),
        "recall": round(n_convicted / n_total, 4) if n_total else None,
        "intensity_match_rate": round(n_match / n_total, 4) if n_total else None,
        "intensity_match_by_category": match_by_cat,
        "intensity_distribution": intensity_dist,
        "misses_sample": misses[:10],
        "mismatches_sample": mismatches[:10],
    }


# ---------------------------------------------------------------------------
# 法院组件消融（v0.4.11 新增，大纲 §6.4 第 9-11 项可纯代码落地子集）：
# 去掉 ABP / 去掉独立评审 / 去掉 Human BOSS
#
# 与 6 个 rubric 开关配置的层次差异：rubric 配置度量的是"矛盾对检测内核"
# 的定罪指标（passed），而法院组件消融度量 ConsensusCourt.hold_court 的
# 两个外显结果——定罪（contradiction 通道）与升级（escalated，人工复核）。
# 无头流水线 run_court_headless 逐行复刻 hold_court 的组件编排
# （consensus_court.py hold_court 内"规则初筛→评分→rubric→PI→评审→
# Human BOSS→escalated"段），唯一区别是不做 DB 落库（实验不需要
# ProjectEvent 持久化）；parity 由 tests/test_court_ablation_pipeline.py
# 对真实 hold_court（内存 SQLite）逐字段比对保证，防止生产代码漂移。
# ---------------------------------------------------------------------------

#: 法院消融模式表：court_full 为参照行（全组件），其余三行各移除一个组件
COURT_MODES: dict[str, str] = {
    "court_full": "完整法庭（参照行：ABP rubric + 独立评审 + Human BOSS）",
    "court_no_abp": "去掉 ABP（不运行矛盾对 rubric；仅留逐证据 CRITICAL 硬约束）",
    "court_no_reviewer": "去掉独立评审（reviewer_notes 置空，条款 4/6 信号消失）",
    "court_no_boss": "去掉 Human BOSS（override_items 不参与 escalated）",
}


def run_court_headless(case: dict, mode: str, detail: bool = False) -> dict:
    """在单个案例上运行无头法庭流水线（hold_court 组件编排的实验镜像）。

    与 ConsensusCourt.hold_court 的严格对应关系：
      1. RuleEngine.screen（对称旧初筛）过滤证据   ↔ hold_court 规则引擎初筛段
      2. EvidenceScorer 评分                       ↔ hold_court 证据评分段
      3. AsymmetricRubric().evaluate（默认全开）   ↔ hold_court rubric 评估段
         · mode == court_no_abp 时整段跳过（max_intensity=0、rubric 不参与定罪）；
           定罪仅保留逐证据 screen_asymmetric 的 CRITICAL（硬约束，如 AF2/AF3）
      4. PIAgentArbiter.arbitrate                  ↔ hold_court PI 裁决段
      5. IndependentReviewer.review                ↔ hold_court 独立评审段
         · mode == court_no_reviewer 时 notes=[]、双轨分歧强制 False
      6. HumanBOSSOverride.generate_items          ↔ hold_court 覆盖清单段
         · mode == court_no_boss 时 items 照算但不参与 escalated（反事实留痕）
      7. escalated = bool(items) or conf<0.6 or max_intensity>=3
                                                    ↔ hold_court 升级判定式

    Args:
        case: L1/L1.5 案例字典
        mode: COURT_MODES 的 key
        detail: 为 True 时额外返回裁决链明细（初筛过滤留痕、逐证据评分、
            矛盾对全记录、PI 理由、评审意见、BOSS 清单）——供 §6.5 端到端
            案例研究（run_case_study.py）复用同一编排，避免第二份镜像漂移；
            默认 False 保持原返回结构（parity 测试口径不变）

    Returns:
        dict: convicted（矛盾对/硬约束定罪）、escalated（人工复核升级）、
        decision/confidence（PI 裁决留痕）、max_intensity（定罪峰值强度）、
        n_filtered（被对称初筛过滤的证据数，留痕用于解释召回差异）；
        detail=True 时附加 filtered_ids / evidence_scores /
        contradictions / pi_rationale / reviewer_notes / override_items
    """
    if mode not in COURT_MODES:
        raise ValueError(f"未知法院消融模式: {mode}")

    rule_engine = RuleEngine()
    scorer = EvidenceScorer()
    pi = PIAgentArbiter()
    reviewer = IndependentReviewer()
    boss = HumanBOSSOverride()
    rubric = AsymmetricRubric()

    all_evidence = [_to_evidence(ev) for ev in case["evidences"]]

    # 步骤 1：对称初筛过滤（与 hold_court 完全一致，用旧 screen()）
    filtered_evidence = []
    filtered_audit: list[dict] = []  # detail 模式：过滤留痕（证据 id + 命中问题）
    for ev in all_evidence:
        issues = rule_engine.screen(ev)
        if not issues:
            filtered_evidence.append(ev)
        elif detail:
            filtered_audit.append(
                {"evidence_id": ev.evidence_id, "screen_issues": list(issues)}
            )
    n_filtered = len(all_evidence) - len(filtered_evidence)
    emap = {ev.evidence_id: ev for ev in filtered_evidence}
    scores = [scorer.score(ev, case["topic"]) for ev in filtered_evidence]

    # 步骤 3：ABP 矛盾对通道（no_abp 模式整段移除）
    rubric_contradictions: list = []  # detail 模式：定罪矛盾对全记录
    if mode == "court_no_abp":
        max_intensity = 0
        rubric_convicted = False
        # 去掉 ABP 后唯一的定罪来源：逐证据 CRITICAL 硬约束
        # （screen_asymmetric 三态口径，PENDING 不定罪）
        critical = any(
            severity == "CRITICAL"
            for ev in all_evidence
            for _desc, severity in rule_engine.screen_asymmetric(ev)
        )
        convicted = critical
    else:
        rubric_verdict = rubric.evaluate(scores, emap, case["topic"])
        max_intensity = max(
            (c.intensity for c in rubric_verdict.contradictions), default=0
        )
        rubric_convicted = not rubric_verdict.passed
        convicted = rubric_convicted
        rubric_contradictions = list(rubric_verdict.contradictions)

    # 步骤 4：PI 仲裁
    pi_decision, pi_rationale, confidence = pi.arbitrate(
        case["topic"], scores, emap
    )

    # 步骤 5：独立评审（no_reviewer 模式移除该组件）
    if mode == "court_no_reviewer":
        reviewer_notes: list[str] = []
        dual_track_divergence = False
    else:
        reviewer_notes = reviewer.review(pi_decision, scores, emap)
        dual_track_divergence = reviewer.contradicts_pi_decision(
            pi_decision, reviewer_notes
        )

    # 步骤 6：Human BOSS 覆盖清单（no_boss 模式照算但不接入升级判定）
    override_items = boss.generate_items(
        pi_decision, confidence, scores, reviewer_notes,
        max_contradiction_intensity=max_intensity,
        dual_track_divergence=dual_track_divergence,
    )
    if mode == "court_no_boss":
        override_items = []

    # 步骤 7：升级判定（与 hold_court 同式）
    escalated = bool(override_items) or confidence < 0.6 or max_intensity >= 3

    out = {
        "convicted": convicted,
        "escalated": escalated,
        "decision": pi_decision,
        "confidence": confidence,
        "max_intensity": max_intensity,
        "n_filtered": n_filtered,
    }
    if detail:
        # 裁决链明细（案例研究用）：初筛过滤留痕 + 逐证据评分 + 矛盾对全记录
        # + PI 理由 + 评审意见 + BOSS 清单，全部来自本次运行的同一组件实例，
        # 与上方汇总字段天然一致（单次执行，无双跑分歧风险）
        out.update(
            {
                "filtered_audit": filtered_audit,
                "evidence_scores": [
                    {
                        "evidence_id": s.evidence_id,
                        "source_score": s.source_score,
                        "recency_score": s.recency_score,
                        "relevance_score": s.relevance_score,
                        "overall_score": s.overall_score,
                        "issues": list(s.issues),
                    }
                    for s in scores
                ],
                "contradictions": [
                    {
                        "evidence_a_id": c.evidence_a_id,
                        "evidence_b_id": c.evidence_b_id,
                        "intensity": c.intensity,
                        "severity": str(c.severity),
                        "desc": c.contradiction_desc,
                    }
                    for c in rubric_contradictions
                ],
                "pi_rationale": pi_rationale,
                "reviewer_notes": list(reviewer_notes),
                "override_items": list(override_items),
            }
        )
    return out


def evaluate_court_config(cases: list[dict], mode: str) -> dict:
    """在整份基准上运行指定法院消融模式，汇总定罪与升级两类指标。

    指标口径：
    - 定罪层（conviction_*）：矛盾对通道的召回/FPR——与 6 个 rubric
      配置同口径，但证据先经 hold_court 的对称初筛过滤，故数值可能低于
      rubric 直评（短内容证据被过滤是生产真实行为，差异本身即发现）。
    - 升级层（escalation_*）：人工复核触发的召回/FPR/precision——
      §6.3 "升级恰当性"指标，独立评审/Human BOSS 的一阶效应体现在这里。

    Args:
        cases: L1 或 L1.5 案例字典列表
        mode: COURT_MODES 的 key

    Returns:
        指标字典（混淆计数 + 率值 + 裁决分布）
    """
    conv_cm = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    esc_cm = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    decision_dist: dict[str, int] = {}
    total_filtered = 0

    for case in cases:
        has_truth = case["polarity"] == "with_contradiction"
        out = run_court_headless(case, mode)
        total_filtered += out["n_filtered"]
        decision_dist[out["decision"]] = decision_dist.get(out["decision"], 0) + 1

        if has_truth:
            conv_cm["tp" if out["convicted"] else "fn"] += 1
            esc_cm["tp" if out["escalated"] else "fn"] += 1
        else:
            conv_cm["fp" if out["convicted"] else "tn"] += 1
            esc_cm["fp" if out["escalated"] else "tn"] += 1

    def _cm_rates(cm: dict) -> dict:
        n_pos = cm["tp"] + cm["fn"]
        n_neg = cm["fp"] + cm["tn"]
        total = sum(cm.values())
        return {
            "recall": round(cm["tp"] / n_pos, 4) if n_pos else None,
            "fpr": round(cm["fp"] / n_neg, 4) if n_neg else None,
            "accuracy": round((cm["tp"] + cm["tn"]) / total, 4) if total else None,
            "precision": (
                round(cm["tp"] / (cm["tp"] + cm["fp"]), 4)
                if (cm["tp"] + cm["fp"]) else None
            ),
            "n_positive": n_pos,
            "n_negative": n_neg,
        }

    return {
        "mode": mode,
        "mode_description": COURT_MODES[mode],
        "conviction": _cm_rates(conv_cm),
        "escalation": _cm_rates(esc_cm),
        "decision_distribution": decision_dist,
        "evidence_filtered_by_legacy_screen": total_filtered,
    }


def main() -> int:
    """消融冒烟主流程：生成基准 → 4 配置评估 → 红线检查 → 落盘报告。"""
    parser = argparse.ArgumentParser(description="v0.3 三机制消融冒烟实验")
    parser.add_argument(
        "--cases", type=int, default=60,
        help="每极性案例数（默认 60；主实验用 300）",
    )
    parser.add_argument(
        "--l15", action="store_true",
        help="附加 DDB-L1.5 负例对照基准评估（大纲 v0.4.3 §6.1 层 1.5，"
             "FPR=0 红线 + 分机制一阶效应）",
    )
    parser.add_argument(
        "--l4", action="store_true",
        help="附加 DDB-L4 强度标注层评估（spec §2：强度判定表全分支"
             "可达性 + 全开配置一致性 + 强度一阶效应红线）",
    )
    parser.add_argument(
        "--court", action="store_true",
        help="附加法院组件消融（§6.4：去掉 ABP / 去掉独立评审 / "
             "去掉 Human BOSS），度量 hold_court 外显的定罪率与升级恰当性",
    )
    parser.add_argument(
        "--scorer-ablation", action="store_true",
        help="附加 scorer 级消融（v0.4.15 §6.4 第 4/8 行：来源等权 "
             "no_source_pedigree / 时效衰减关闭 no_recency），机制全开 "
             "（full_v03 开关）不变，仅对评分输出做实验侧后处理",
    )
    args = parser.parse_args()

    print(f"[1/5] 生成 DDB-L1 冒烟基准（{args.cases}×2 例，seed={SEED}）...")
    cases = generate(
        num_with_contradiction=args.cases,
        num_no_contradiction=args.cases,
        seed=SEED,
    )
    print(f"      基准生成并校验通过：{len(cases)} 例")

    print("[2/5] 消融配置评估...")
    results: dict[str, dict] = {}
    for name, switches in CONFIGS.items():
        results[name] = evaluate_config(cases, switches)
        r = results[name]
        print(
            f"      {name:<15} recall={r['recall']:.4f}  fpr={r['fpr']:.4f}"
            f"  ambig={r['ambiguous_negative_rate']:.4f}"
            f"  intensity={r['intensity_distribution']}"
        )

    print("[3/5] 保守公理红线检查...")
    base = results["baseline_v02"]
    full = results["full_v03"]
    violations: list[str] = []
    if full["fpr"] > base["fpr"]:
        violations.append(
            f"FPR 红线违例：full_v03 fpr={full['fpr']} > baseline fpr={base['fpr']}"
            "（新机制必须只降假阳性）"
        )
    if full["recall"] < base["recall"] - RECALL_TOLERANCE:
        violations.append(
            f"召回红线违例：full_v03 recall={full['recall']} < "
            f"baseline recall={base['recall']} - 容差 {RECALL_TOLERANCE}"
            "（机制不得引入系统性漏检；检查 Aspect/OBJECT 词典覆盖）"
        )
    for v in violations:
        print(f"      [违例] {v}")
    if not violations:
        print("      两条红线均满足（FPR 不升 / 召回不降）")

    # ------------------------------------------------------------------
    # L1.5 负例对照基准（大纲 v0.4.3 §6.1 层 1.5）：FPR=0 红线 +
    # 分机制一阶效应（aspect_only → cross_aspect 类漂移、
    # gate_only → 门禁两类漂移）
    # ------------------------------------------------------------------
    l15_results: Optional[dict[str, dict]] = None
    if args.l15:
        from tests.ddb.l15_generator import generate as generate_l15

        print("[4/5] DDB-L1.5 负例对照基准评估（FPR=0 红线）...")
        l15_cases = generate_l15(per_template=8, seed=SEED)
        l15_results = {}
        for name, switches in CONFIGS.items():
            l15_results[name] = evaluate_l15_config(l15_cases, switches)
            r = l15_results[name]
            cats = " ".join(
                f"{k}={v['fpr']}" for k, v in sorted(r["by_category"].items())
            )
            print(
                f"      {name:<15} fpr={r['fpr']:.4f}"
                f"  ambig={r['ambiguous_negative_rate']:.4f}  [{cats}]"
            )

        # L1.5 红线 1：full_v03 在负例集 FPR 必须为 0
        if l15_results["full_v03"]["fpr"] != 0:
            violations.append(
                f"L1.5 FPR=0 红线违例：full_v03 在负例集 fpr="
                f"{l15_results['full_v03']['fpr']}（存在未被机制拦截的伪冲突）"
            )
        # L1.5 红线 2：aspect 一阶效应——cross_aspect 类在 aspect_only
        # 与 full_v03 下必须全部拦截（FPR=0，aspect 独占拦截面），在
        # baseline_v02 与 gate_only（均关 aspect）下必须全部漂移为定罪
        # （FPR=1，证伪性：拦截确由 aspect 贡献）
        cross_key = "cross_aspect"
        expect_cross = {
            "aspect_only": 0.0, "full_v03": 0.0, "full_v04": 0.0,
            "baseline_v02": 1.0, "gate_only": 1.0, "position_only": 1.0,
        }
        for name, expected in expect_cross.items():
            got = l15_results[name]["by_category"].get(cross_key, {}).get("fpr")
            if got != expected:
                violations.append(
                    f"L1.5 一阶效应违例：cross_aspect 类在 {name} 下"
                    f" fpr={got}（期望 {expected}），aspect 拦截面归属不纯"
                )
        # L1.5 红线 3：门禁一阶效应——门禁两类在 baseline_v02 /
        # aspect_only（均关**门禁**）下必须漂移为定罪（FPR=1），在
        # gate_only / full_v03（门禁开启）下必须全部拦截（FPR=0；
        # AMBIGUOUS 降级不算定罪）
        gate_cats = ("different_object_gate", "ambiguous_reference")
        expect_gate = {
            "baseline_v02": 1.0, "aspect_only": 1.0, "position_only": 1.0,
            "gate_only": 0.0, "full_v03": 0.0, "full_v04": 0.0,
        }
        for name, expected in expect_gate.items():
            for cat in gate_cats:
                got = l15_results[name]["by_category"].get(cat, {}).get("fpr")
                if got != expected:
                    violations.append(
                        f"L1.5 一阶效应违例：{cat} 类在 {name} 下 fpr={got}"
                        f"（期望 {expected}），门禁拦截面归属不纯"
                    )
    else:
        print("[4/5] DDB-L1.5 评估：跳过（未传 --l15）")

    # ------------------------------------------------------------------
    # DDB-L4 强度标注层（spec §2）：强度判定表全分支可达性 +
    # 全开配置一致性红线 + 强度一阶效应
    # ------------------------------------------------------------------
    l4_results: Optional[dict[str, dict]] = None
    if args.l4:
        from tests.ddb.l4_generator import generate as generate_l4

        print("[4b/5] DDB-L4 强度标注基准评估（强度一致性红线）...")
        l4_cases = generate_l4(rounds=4, seed=SEED)
        l4_results = {}
        for name, switches in CONFIGS.items():
            l4_results[name] = evaluate_l4_config(l4_cases, switches)
            r = l4_results[name]
            cats = " ".join(
                f"{k}: m={v['match']}/x={v['mismatch']}"
                for k, v in sorted(r["intensity_match_by_category"].items())
            )
            print(
                f"      {name:<15} recall={r['recall']:.4f}"
                f"  match={r['intensity_match_rate']:.4f}"
                f"  dist={r['intensity_distribution']}  [{cats}]"
            )

        # L4 红线 1：full_v03 / full_v04 召回必须 = 1.0（零漏检）
        for full_name in ("full_v03", "full_v04"):
            if l4_results[full_name]["recall"] != 1.0:
                violations.append(
                    f"L4 零漏检红线违例：{full_name} 在正例集 recall="
                    f"{l4_results[full_name]['recall']}（存在漏检正例）"
                )
        # L4 红线 2：full_v03 / full_v04 强度匹配率必须 = 1.0（mismatch=0）
        for full_name in ("full_v03", "full_v04"):
            if l4_results[full_name]["intensity_match_rate"] != 1.0:
                violations.append(
                    f"L4 强度一致性红线违例：{full_name} match_rate="
                    f"{l4_results[full_name]['intensity_match_rate']}"
                    f"（{l4_results[full_name]['n_mismatches']} 例不一致）："
                    "\n".join(l4_results[full_name]["mismatches_sample"][:5])
                )
        # L4 红线 3：强度一阶效应——intensity 关闭时全部强度恒 2，
        # 开启时 gap_same 类必须提升至 3（S4 门禁佐证分支可达性）
        for name in ("baseline_v02", "aspect_only", "gate_only", "position_only"):
            dist = l4_results[name]["intensity_distribution"]
            if not (set(dist.keys()) <= {"2"}):
                violations.append(
                    f"L4 强度一阶效应违例：{name}（intensity=off）"
                    f" 应恒 2 但分布={dist}"
                )
        # gap_same 在 full_v03 / full_v04 下必须 match（intensity=3，S4 分支可达）
        for full_name in ("full_v03", "full_v04"):
            gs = l4_results[full_name]["intensity_match_by_category"].get(
                "gap_same", {})
            if gs.get("match", 0) == 0:
                violations.append(
                    f"L4 强度判定表分支违例：gap_same 类在 {full_name} 下"
                    " 全部 mismatch（S4 SAME_OBJECT 佐证分支不可达）"
                )
    else:
        print("[4b/5] DDB-L4 评估：跳过（未传 --l4）")

    # ------------------------------------------------------------------
    # 位置加权一阶效应（v0.4.5 新增，§6.4 "去掉位置加权"消融）：
    # full_v04（position=on）相对 full_v03（position=off）定罪
    # 置信度折扣效应必须可观测。
    # ------------------------------------------------------------------
    full_v03_w = results.get("full_v03", {}).get("avg_weighted_confidence")
    full_v04_w = results.get("full_v04", {}).get("avg_weighted_confidence")
    if full_v03_w is not None and full_v04_w is not None:
        if full_v04_w >= full_v03_w:
            violations.append(
                f"位置加权一阶效应违例：full_v04 avg_weighted_confidence="
                f"{full_v04_w} 未低于 full_v03={full_v03_w}"
                "（position=on 应产生可观测置信度折扣）"
            )
        else:
            print(
                f"[4c/5] 位置加权一阶效应：full_v03={full_v03_w} → "
                f"full_v04={full_v04_w}（折扣效应可观测）"
            )
    else:
        print("[4c/5] 位置加权一阶效应：跳过（无定罪案例或数据缺失）")

    # ------------------------------------------------------------------
    # 法院组件消融（v0.4.11 新增，§6.4 第 9-11 项可纯代码落地子集）：
    # court_full 参照行 + 去掉 ABP / 独立评审 / Human BOSS。
    # 定罪层验证"去掉矛盾检测通道必丢召回"；升级层验证评审/BOSS 的
    # 一阶效应（组件只增加升级信号 → 升级召回单调不降）。
    # ------------------------------------------------------------------
    court_results: Optional[dict[str, dict]] = None
    court_l15_results: Optional[dict[str, dict]] = None
    if args.court:
        print("[4d/5] 法院组件消融评估（hold_court 定罪层 + 升级层）...")
        court_results = {}
        for mode in COURT_MODES:
            court_results[mode] = evaluate_court_config(cases, mode)
            r = court_results[mode]
            cv, es = r["conviction"], r["escalation"]
            print(
                f"      {mode:<18} 定罪 recall={cv['recall']} fpr={cv['fpr']}"
                f"  | 升级 recall={es['recall']} fpr={es['fpr']}"
                f" prec={es['precision']}  | 过滤证据={r['evidence_filtered_by_legacy_screen']}"
            )

        full_cv = court_results["court_full"]["conviction"]
        full_es = court_results["court_full"]["escalation"]

        # 法院红线 1：评审/BOSS 不触碰矛盾对定罪通道——三模式定罪指标恒等
        for mode in ("court_no_reviewer", "court_no_boss"):
            cv = court_results[mode]["conviction"]
            if (cv["recall"] != full_cv["recall"]
                    or cv["fpr"] != full_cv["fpr"]):
                violations.append(
                    f"法院消融违例：{mode} 定罪指标 ({cv['recall']}, {cv['fpr']}) "
                    f"与 court_full ({full_cv['recall']}, {full_cv['fpr']}) 不一致"
                    "（评审/BOSS 位于定罪下游，不得改变矛盾对定罪结果）"
                )

        # 法院红线 2：去掉 ABP 必丢矛盾对召回（L1 合成集无 AF 硬约束，
        # no_abp 定罪召回必须严格小于完整法庭；FPR 不得更高）
        no_abp_cv = court_results["court_no_abp"]["conviction"]
        if not (no_abp_cv["recall"] < full_cv["recall"]):
            violations.append(
                f"法院消融违例：court_no_abp 定罪 recall={no_abp_cv['recall']} "
                f"未严格低于 court_full={full_cv['recall']}"
                "（去掉矛盾对通道必须损失检测能力，否则该通道无收益）"
            )
        if (no_abp_cv["fpr"] is not None and full_cv["fpr"] is not None
                and no_abp_cv["fpr"] > full_cv["fpr"]):
            violations.append(
                f"法院消融违例：court_no_abp 定罪 fpr={no_abp_cv['fpr']} "
                f"高于 court_full={full_cv['fpr']}"
            )

        # 法院红线 3：升级召回单调性——评审/BOSS 只增加升级信号
        for mode in ("court_no_reviewer", "court_no_boss"):
            es = court_results[mode]["escalation"]
            if (es["recall"] is not None and full_es["recall"] is not None
                    and es["recall"] > full_es["recall"]):
                violations.append(
                    f"法院消融违例：{mode} 升级 recall={es['recall']} "
                    f"> court_full={full_es['recall']}"
                    "（组件移除后升级召回不应上升——组件只产生升级信号）"
                )

        # L1.5 负例集上的法院消融：完整法庭定罪 FPR 必须为 0（rubric 口径一致）
        if args.l15:
            court_l15_results = {}
            print("      L1.5 负例集上的法院消融：")
            for mode in COURT_MODES:
                court_l15_results[mode] = evaluate_court_config(l15_cases, mode)
                r = court_l15_results[mode]
                cv, es = r["conviction"], r["escalation"]
                print(
                    f"      {mode:<18} 定罪 fpr={cv['fpr']}"
                    f"  | 升级 fpr={es['fpr']} prec={es['precision']}"
                )
            if court_l15_results["court_full"]["conviction"]["fpr"] != 0:
                violations.append(
                    "法院消融违例：court_full 在 L1.5 负例集定罪 fpr="
                    f"{court_l15_results['court_full']['conviction']['fpr']}"
                    "（与 rubric 全开 FPR=0 口径不一致）"
                )
    else:
        print("[4d/5] 法院组件消融：跳过（未传 --court）")

    # ------------------------------------------------------------------
    # scorer 级消融（v0.4.15 新增，§6.4 第 4/8 行：来源等权 / 时效衰减
    # 关闭）。机制全开（full_v03 开关组合）不变，唯一变量 = 评分公式的
    # 单一分量（实验侧后处理，生产代码零改动）。红线见
    # check_scorer_ablation_red_lines（R1 单调性 + R2 效应方向可解释）。
    # ------------------------------------------------------------------
    scorer_ablation_results: Optional[dict[str, dict]] = None
    if args.scorer_ablation:
        print("[4e/5] scorer 级消融评估（来源等权 / 时效衰减关闭，full_v03 开关不变）...")
        full_v03_switches = CONFIGS["full_v03"]
        scorer_ablation_results = {}
        for mode, desc in SCORER_ABLATION_MODES.items():
            scorer_ablation_results[mode] = evaluate_config(
                cases, full_v03_switches, score_ablation=mode
            )
            r = scorer_ablation_results[mode]
            print(
                f"      {mode:<20} recall={r['recall']:.4f}  fpr={r['fpr']:.4f}"
                f"  avg_pass_conf={r['avg_pass_confidence']}"
                f"  （{desc}）"
            )
        # 红线检查：full_v03 参照行 vs 两个消融格点
        for mode in SCORER_ABLATION_MODES:
            for v in check_scorer_ablation_red_lines(
                results["full_v03"], scorer_ablation_results[mode], mode
            ):
                violations.append(v)
                print(f"      [违例] {v}")
        if not violations:
            print(
                "      scorer 消融红线满足（FPR/召回单调性 + 置信度效应方向可解释）"
            )
    else:
        print("[4e/5] scorer 级消融：跳过（未传 --scorer-ablation）")

    print("[5/5] 写入消融报告...")
    report = {
        "experiment": "v0.3 三机制消融冒烟（Aspect 约束 / ValidityGate / 强度分级）",
        "benchmark": "DDB-L1（tests/ddb/l1_generator.py 合成争议基准）",
        "seed": SEED,
        "cases_per_polarity": args.cases,
        "recall_tolerance": RECALL_TOLERANCE,
        "configs": results,
        "l15_configs": l15_results,
        "l4_configs": l4_results,
        "court_configs": court_results,
        "court_l15_configs": court_l15_results,
        "scorer_ablation_configs": scorer_ablation_results,
        "scorer_ablation_uniform_source_score": UNIFORM_SOURCE_SCORE,
        "red_line_violations": violations,
        "generated_at": datetime.now().isoformat(),
        "note": (
            "冒烟级（60×2）；正式 13 项消融（600×2 全组合 + 显著性检验）"
            "在第 3 月主实验运行，同管线 --cases 300 放大即可。"
            "L1.5（--l15）：负例对照层 FPR=0 红线 + 分机制一阶效应。"
            "L4（--l4）：强度判定表全分支可达性 + 全开一致性 + 强度一阶效应。"
            "v0.4.5 扩展：position 维度（EvidencePositionWeighter）作为第四"
            "自变量，position_only / full_v04 两格点验证位置加权一阶效应"
            "（full_v04 定罪置信度折扣 < full_v03）。"
            "v0.4.11 扩展：--court 法院组件消融（§6.4 第 9-11 项可纯代码"
            "落地子集），court_full 参照 + no_abp/no_reviewer/no_boss，"
            "度量 hold_court 定罪层（矛盾对通道必要性）与升级层（评审/"
            "BOSS 升级信号一阶效应，含 §6.3 precision@escalation）。"
            "v0.4.15 扩展：--scorer-ablation scorer 级消融（§6.4 第 4/8 行），"
            "机制全开不变、评分输出实验侧后处理（生产代码零改动）——"
            "no_source_pedigree 来源等权（统一中值 "
            f"{UNIFORM_SOURCE_SCORE}）+ no_recency 时效衰减关闭（recency=1.0），"
            "可观测效应在通过案例置信度通道（overall 与定罪通道解耦是 "
            "ABP 结构性设计）。"
        ),
    }
    out_path = Path(__file__).resolve().parents[1] / "ddb" / "ablation_v03_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"      报告：{out_path}")

    if violations:
        print("结论：红线违例，需修复词典覆盖后重跑。")
        return 1
    print("结论：消融冒烟通过（管线可用 + 保守公理成立）。")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 —— 管线异常统一归口退出码 2
        print(f"运行异常：{exc}", file=sys.stderr)
        raise
