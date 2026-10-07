"""
共识法庭对照基线仲裁器（§6.2）

三个基线实现，用于对比"多数决"、"加权投票"、"单裁判直裁决"三类
经典策略在 DDB 基准上的 FPR/召回/Acc——这些基线的共同特点是**不做
矛盾对检测**（即无 ConsensusCourt 的 rubric 评估，直接基于证据评分
出裁决），是 ABP 思想实验"去掉矛盾检测模块"的实证载体。

当前版本仅实现 §6.2 中标注为"可纯代码落地"的三项基线；IMPACT-style、
Debate-then-Judge 等依赖额外审议辩论管线的基线属后续增量。

与论文大纲的对应关系：
  MajorityVote → §6.2 表"各 Agent 一票，等权多数决"
  EloArbiter   → §6.2 表"Google AI Co-Scientist 式 Elo 锦标赛复现"
                （本实现用角色权重 + Elo 置信度校准近似复现）
  SingleJudge  → §6.2 表"GPT-4o/Qwen3.6-35B 单裁判直接裁决"
                （本实现用最高置信度证据直裁决，作为规则版单裁判替代）

运行：backend$ .venv/bin/python -m pytest tests/test_baseline_arbiter.py -q
"""
from __future__ import annotations

import math
from typing import Any

from app.core.virtual_lab.consensus_court import Evidence, EvidenceScore


# ---------------------------------------------------------------------------
# 仲裁器公共契约：签名对齐 PIAgentArbiter.arbitrate，返回 (decision, rationale, confidence)
# ---------------------------------------------------------------------------

_DECISION_SUPPORT = "支持提案"
_DECISION_CONDITIONAL = "有条件支持"
_DECISION_REJECT = "驳回提案"


def _decision_from_score(score: float) -> str:
    """把 0-1 置信度分数映射为三档裁决（与 PIAgentArbiter 口径一致：
    ≥0.7 支持、≥0.4 有条件、<0.4 驳回）。"""
    if score >= 0.7:
        return _DECISION_SUPPORT
    if score >= 0.4:
        return _DECISION_CONDITIONAL
    return _DECISION_REJECT


# ---------------------------------------------------------------------------
# Baseline 1: MajorityVote（等权多数决）
# ---------------------------------------------------------------------------

class MajorityVote:
    """每个 Agent 一票，等权多数决。

    对应大纲 §6.2：各 Agent 一票，等权多数决。与 VoteAggregator 的
    VETO/UNANIMOUS/MAJORITY 判定不同——这里每个证据等权，不做角色区分；
    "一票否决"不存在于该基线，所有证据等权累加，这正是对照差异所在。

    阈值设计：每证据 overall_score ≥ 0.7 计支持票、< 0.4 计反对票、
    中间票记弃权；支持票 > 反对票 → 支持；反之驳回；平票 → 有条件支持。
    """

    # 支持票阈值：≥ 0.7（置信度 70% 以上算支持）
    SUPPORT_THRESHOLD = 0.7
    # 反对票阈值：< 0.4（置信度 40% 以下算反对）
    REJECT_THRESHOLD = 0.4

    def arbitrate(
        self,
        topic: str,
        evidence_scores: list[EvidenceScore],
        evidence_map: dict[str, Evidence],
    ) -> tuple[str, str, float]:
        """按等权多数决裁决。

        Args:
            topic: 争议主题
            evidence_scores: 证据评分列表
            evidence_map: evidence_id -> Evidence 映射（本基线不使用角色，
                保留签名对齐 PIAgentArbiter）

        Returns:
            (decision, rationale, confidence)
        """
        n = len(evidence_scores)
        if n == 0:
            return "无法裁决", "没有可用证据", 0.0

        support = reject = abstain = 0
        for s in evidence_scores:
            sc = s.overall_score
            if sc >= self.SUPPORT_THRESHOLD:
                support += 1
            elif sc < self.REJECT_THRESHOLD:
                reject += 1
            else:
                abstain += 1

        if support > reject:
            decision = _DECISION_SUPPORT
            conf = support / n
        elif reject > support:
            decision = _DECISION_REJECT
            conf = reject / n
        else:
            # 平票 → 有条件支持（保守方向：不轻易驳回）
            decision = _DECISION_CONDITIONAL
            conf = 0.5

        rationale = (
            f"多数决：共 {n} 条证据，支持 {support} / 反对 {reject} / 弃权 {abstain}，"
            f"裁决：{decision}（置信度 {conf:.2f}）"
        )
        return decision, rationale, conf


# ---------------------------------------------------------------------------
# Baseline 2: EloArbiter（角色加权 + Elo 置信度校准）
# ---------------------------------------------------------------------------

class EloArbiter:
    """Elo 锦标赛式加权裁决器。

    对应大纲 §6.2：Google AI Co-Scientist 式 Elo 锦标赛复现。完整 Elo
    锦标赛需要多轮"证据对赛"并根据结果重排 Elo，这里做规则近似：
    角色权重（与 PIAgentArbiter.ROLE_WEIGHTS 同口径，体现领域专家更可靠）
    乘以 overall_score 作为加权投票，再通过 Elo 公式校准置信度——
    E = 1 / (1 + 10^((R_ref - R_cand) / 400))，其中 R_ref=1000（基准 Elo），
    R_cand = Σ(role_weight * score * 100) / Σ(role_weight)，把加权平均分数
    映射到 0-1 置信度，避免硬阈值截断平滑。

    与 PIAgentArbiter 的区别：PI 直接用加权平均 → 阈值三档；本实现
    引入 Elo 平滑校准，输出置信度在分数区间连续分布，避免阈值边界的
    不稳定性。

    注：本实现为"单轮 Elo 近似"，未做完整多轮对赛 + Elo 重排；
    完整 Elo 锦标赛需外部 LLM 对赛判定，属后续增量。
    """

    # 角色权重（与 PIAgentArbiter 同口径，便于消融对比"仅仲裁逻辑差异"）。
    # v0.4.13 对齐论文 13 角色：与 PIAgentArbiter.ROLE_WEIGHTS 逐键同值（单测锁死），
    # data-specialist（论文名）= data-analyst（平台实现名），reviewer/agent 为历史
    # 兼容 fallback 键，未登记角色仍走 0.5 默认 fallback。
    ROLE_WEIGHTS: dict[str, float] = {
        "pi-agent": 1.0,
        "drug-researcher": 0.9,
        "structural-biologist": 0.9,
        "computational-chemist": 0.85,
        "synthetic-chemist": 0.8,
        "antibody-engineer": 0.8,
        "peptide-designer": 0.75,
        "data-specialist": 0.75,
        # data-specialist：论文名 = 平台实现名 data-analyst 的双命名映射
        # （附录 D D.3）。按计划红线不引入别名机制——论文侧统一用论文名；
        # 平台侧 data-analyst 证据走 0.5 fallback，口径差异已在 D.3 声明。
        "qc-inspector": 0.7,
        "knowledge-engineer": 0.65,
        "reflexion-agent": 0.6,
        "project-coordinator": 0.6,
        "report-expert": 0.5,
        # 历史 fallback 键（非论文 13 角色，仅为 v0.4.12 前证据流兼容保留）
        "reviewer": 0.7,
        "agent": 0.5,
    }
    # 基准 Elo 分（对比候选加权均分）
    BASELINE_ELO = 1000.0
    # Elo K 因子（控制校准曲线斜率，与 Chess Elo 标准口径 K=32 对齐）
    ELO_K = 32.0

    def arbitrate(
        self,
        topic: str,
        evidence_scores: list[EvidenceScore],
        evidence_map: dict[str, Evidence],
    ) -> tuple[str, str, float]:
        """角色加权 + Elo 校准裁决。"""
        if not evidence_scores:
            return "无法裁决", "没有可用证据", 0.0

        weighted_sum = 0.0
        total_weight = 0.0
        for s in evidence_scores:
            ev = evidence_map.get(s.evidence_id)
            role = ev.role.lower() if ev else "agent"
            w = self.ROLE_WEIGHTS.get(role, 0.5)
            weighted_sum += w * s.overall_score
            total_weight += w

        # 加权平均置信度（0-1）
        avg = weighted_sum / total_weight if total_weight > 0 else 0.0
        # 映射到 Elo 候选分：avg 0.0→Elo 0，avg 1.0→Elo 2000，基准 1000
        candidate_elo = avg * 2000.0
        # Elo 期望分：E = 1/(1+10^((R_ref - R_cand)/400))
        expected = 1.0 / (1.0 + 10.0 ** ((self.BASELINE_ELO - candidate_elo) / 400.0))
        # expected ∈ (0, 1)，作为校准后的置信度（avg 直接作为决策依据，
        # expected 作为连续置信度输出，二者近似但 expected 平滑）
        decision = _decision_from_score(avg)

        rationale = (
            f"Elo 仲裁：角色加权均分 {avg:.2f}，校准置信度 {expected:.2f}，"
            f"裁决：{decision}"
        )
        return decision, rationale, expected


# ---------------------------------------------------------------------------
# Baseline 3: SingleJudge（单裁判直裁决）
# ---------------------------------------------------------------------------

class SingleJudge:
    """单裁判直裁决（用最高置信度证据直接出判决）。

    对应大纲 §6.2：GPT-4o/Qwen3.6-35B 单裁判直接裁决。LLM 单裁判本质
    上是"最强证据直裁决"——直接看最有力的一条证据下结论，不做矛盾检测、
    不做多证据合取。这里用规则近似：取 overall_score 最高的证据作为
    "单裁判的目光落点"，直接出裁决。

    这是最脆弱的基线：一条噪声高分证据即可误导决策；适合用来说明
    "无矛盾检测时容易被伪冲突误伤"这一本文核心动机。
    """

    def arbitrate(
        self,
        topic: str,
        evidence_scores: list[EvidenceScore],
        evidence_map: dict[str, Evidence],
    ) -> tuple[str, str, float]:
        """取最高置信度证据直裁决。"""
        if not evidence_scores:
            return "无法裁决", "没有可用证据", 0.0

        best = max(evidence_scores, key=lambda s: s.overall_score)
        decision = _decision_from_score(best.overall_score)

        ev = evidence_map.get(best.evidence_id)
        role = ev.role if ev else "unknown"
        rationale = (
            f"单裁判：取 {best.evidence_id}（role={role}）置信度 "
            f"{best.overall_score:.2f} 直裁决 → {decision}"
        )
        return decision, rationale, best.overall_score


# ---------------------------------------------------------------------------
# 基线注册表（run_ablation_v03.py 消融管线统一入口）
# ---------------------------------------------------------------------------

BASELINE_ARBITERS: dict[str, Any] = {
    "majority_vote": MajorityVote,
    "elo_arbiter": EloArbiter,
    "single_judge": SingleJudge,
}
"""§6.2 对照基线注册表，key 用作消融配置名、值是类（调用 .arbitrate 实例方法）。"""


__all__ = [
    "MajorityVote",
    "EloArbiter",
    "SingleJudge",
    "BASELINE_ARBITERS",
]