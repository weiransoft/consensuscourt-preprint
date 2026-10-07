"""
ConsensusCourt - 共识法庭

职责：
1. 对 Agent 冲突进行结构化审理。
2. 证据评分器（EvidenceScorer）按来源、时效、相关性评分。
3. 规则引擎初筛明显违规或低质证据。
4. PI Agent 裁决：基于证据与角色权重生成裁决建议。
5. 独立评审 Agent 复核：从不同视角交叉验证。
6. Human BOSS 覆盖清单：列出需要人类最终确认的事项。
"""
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from app.core.virtual_lab.models import AgentChatMessage, AgentChatSession
from app.services.base import BaseService

logger = logging.getLogger(__name__)


class ConsensusCourtError(Exception):
    """共识法庭异常基类"""
    pass


@dataclass
class Evidence:
    """证据条目"""
    evidence_id: str
    source: str
    content: str
    role: str
    timestamp: Optional[datetime] = None
    reference_ids: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class EvidenceScore:
    """证据评分结果"""
    evidence_id: str
    source_score: float
    recency_score: float
    relevance_score: float
    overall_score: float
    issues: list[str] = field(default_factory=list)


COURT_VERDICT_EVENT_TYPE = "CONSENSUS_COURT_VERDICT"
"""共识法庭裁决事件类型（写入 ProjectEvent.event_type，用于可审计回放）。"""
# ---------------------------------------------------------------------------
# 三态引用验证（论文机制：valid / invalid / network_failed）。
#
# 生产内置语义（G1 修复，与论文 §三态引用验证严格对齐）：
# - valid          : 引用经核验确认存在且指向成立；
# - invalid        : 引用经核验确认不存在/伪造（确认级信号，升级人工复核）；
# - network_failed : 核验服务未配置或网络失败——**仅告警，绝不作为定罪依据**。
#                    核验能力缺失 ≠ 引用伪造，诚实降级为 network_failed，
#                    这正是论文"网络失败不能定罪"红线（RL1）的生产版。
# 状态判定优先级：上游真实核验结果（Evidence.metadata["citation_checks"]）
# > 未提供核验结果（一律 network_failed，不猜测不伪造）。
# ---------------------------------------------------------------------------
CITATION_STATE_VALID = "valid"
CITATION_STATE_INVALID = "invalid"
CITATION_STATE_NETWORK_FAILED = "network_failed"

_VALID_CITATION_STATES = frozenset(
    {CITATION_STATE_VALID, CITATION_STATE_INVALID, CITATION_STATE_NETWORK_FAILED}
)


@dataclass
class CitationCheck:
    """单条引用的三态核验结果

    Attributes:
        evidence_id: 所属证据 ID
        reference_id: 引用标识（如 PMID / ChEMBL ID / e2e 案卷引用）
        state: 三态之一（CITATION_STATE_* 常量）
        detail: 人读细节（如失败原因），供庭审详情与审计回放
    """
    evidence_id: str
    reference_id: str
    state: str
    detail: str = ""


class CitationVerifier:
    """引用核验器基类（三态引用验证的可替换注入点）

    生产部署可注入真实核验实现（如 PubMed/ChEMBL 在线查询）；
    默认实现 DefaultCitationVerifier 采用"预计算结果 + 诚实降级"策略，
    不在裁决同步路径上发起网络请求（保持零 LLM、确定性、无外部依赖）。
    """

    def verify(self, evidence: Evidence, topic: str = "") -> list[CitationCheck]:
        """对单条证据的全部引用做三态核验

        Args:
            evidence: 证据条目
            topic: 争议主题（供实现方参考，默认实现不使用）

        Returns:
            与 evidence.reference_ids 一一对应的核验结果列表
        """
        raise NotImplementedError


class DefaultCitationVerifier(CitationVerifier):
    """默认引用核验器：优先采信上游预计算结果，缺失则诚实降级 network_failed

    预计算结果来自 Evidence.metadata["citation_checks"]，支持两种形态：
    - {reference_id: state_str}：仅状态；
    - {reference_id: {"state": str, "detail": str}}：状态 + 人读细节。
    上游（如疫苗管线真实做过文献/序列库查询的模块）将真实核验结论
    写入 metadata 后，法庭即可消费 valid/invalid 确认级信号；
    未写入的一律按 network_failed 处理——不猜测、不伪造核验结论。
    """

    def verify(self, evidence: Evidence, topic: str = "") -> list[CitationCheck]:
        """对单条证据的全部引用做三态核验（见类 docstring 的状态优先级）"""
        precomputed = evidence.metadata.get("citation_checks")
        precomputed = precomputed if isinstance(precomputed, dict) else {}
        checks: list[CitationCheck] = []
        for ref in evidence.reference_ids:
            entry = precomputed.get(ref)
            if entry is None:
                checks.append(CitationCheck(
                    evidence_id=evidence.evidence_id,
                    reference_id=ref,
                    state=CITATION_STATE_NETWORK_FAILED,
                    detail="未提供引用核验结果（核验服务未配置），按网络失败处理（仅告警，不定罪）",
                ))
                continue
            if isinstance(entry, dict):
                state_raw = str(entry.get("state", "")).strip().lower()
                detail = str(entry.get("detail", ""))
            else:
                state_raw = str(entry).strip().lower()
                detail = ""
            if state_raw in _VALID_CITATION_STATES:
                state = state_raw
            else:
                # 未知状态字符串：保守按 network_failed 处理（不得臆判 invalid）
                state = CITATION_STATE_NETWORK_FAILED
                detail = detail or f"未知核验状态 '{state_raw}'，保守按网络失败处理"
            checks.append(CitationCheck(
                evidence_id=evidence.evidence_id,
                reference_id=ref,
                state=state,
                detail=detail,
            ))
        return checks


@dataclass
class CourtVerdict:
    """
    法庭裁决结果

    v0.3 扩展字段带默认值追加在末尾：既有四参/八字段的构造调用零改动，
    未走 rubric 定罪通道（如证据为空）时 max_contradiction_intensity=0，
    语义为"本次审理不存在任何定罪矛盾对"。
    """
    verdict: str
    decision: str
    confidence: float
    evidence_scores: list[EvidenceScore]
    pi_rationale: str
    reviewer_notes: list[str]
    human_override_items: list[str]
    escalated: bool
    max_contradiction_intensity: int = 0
    """定罪矛盾对的最大强度（v0.3 升级条款 5 依据）：0=无矛盾对，
    1=轻微（程度分歧），2=中度（方向存疑），3=对抗（方向直接相反）。"""
    dual_track_divergence: bool = False
    """双轨仲裁分歧标志（v0.4.9 升级条款 6 依据）：规则/模型仲裁轨（PIAgentArbiter）
    与证据型核查轨（IndependentReviewer）结论语义冲突时为 True，默认 False
    （既有八字段/九字段构造调用零改动、行为等价 v0.4.8）。"""
    citation_checks: list[CitationCheck] = field(default_factory=list)
    """三态引用验证结果（G1 生产内置）：逐引用 valid/invalid/network_failed。
    仅随案卷留痕与升级判定消费——network_failed 绝不定罪（论文三态语义）。"""


def verdict_to_payload(
    verdict: CourtVerdict,
    topic: str,
    evidence_list: list[Evidence],
) -> dict:
    """
    将 CourtVerdict 序列化为可入库的事件 payload。

    payload 结构（schema v1.0）：
    {
        "topic": str,
        "decision": str,
        "confidence": float,
        "verdict_text": str,
        "pi_rationale": str,
        "evidence_scores": [ {evidence_id, source_score, recency_score,
                              relevance_score, overall_score, issues} ],
        "reviewer_notes": [str],
        "human_override_items": [str],
        "escalated": bool,
        "max_contradiction_intensity": int,  # v0.3 升级条款 5 留痕
        "dual_track_divergence": bool,       # v0.4.9 升级条款 6 留痕
        "citation_checks": [ {evidence_id, reference_id, state, detail} ],  # G1 三态引用验证留痕
        "evidence_fingerprints": {evidence_id: sha256_hex},
    }

    Args:
        verdict: 法庭裁决结果
        topic: 争议主题
        evidence_list: 参与审理的证据列表（用于计算证据指纹）

    Returns:
        事件 payload 字典（不含 session_id / project_id 等关联字段）
    """
    return {
        "topic": topic,
        "decision": verdict.decision,
        "confidence": verdict.confidence,
        "verdict_text": verdict.verdict,
        "pi_rationale": verdict.pi_rationale,
        "evidence_scores": [
            {
                "evidence_id": s.evidence_id,
                "source_score": s.source_score,
                "recency_score": s.recency_score,
                "relevance_score": s.relevance_score,
                "overall_score": s.overall_score,
                "issues": list(s.issues),
            }
            for s in verdict.evidence_scores
        ],
        "reviewer_notes": list(verdict.reviewer_notes),
        "human_override_items": list(verdict.human_override_items),
        "escalated": verdict.escalated,
        # v0.3：定罪矛盾对最大强度随案卷留痕，供 L3 回放审计分诊依据
        "max_contradiction_intensity": verdict.max_contradiction_intensity,
        # v0.4.9：双轨仲裁分歧随案卷留痕（升级条款 6 依据，供 L3 回放审计分诊）
        "dual_track_divergence": verdict.dual_track_divergence,
        # G1 三态引用验证逐条留痕：valid/invalid/network_failed，
        # 供庭审详情结构化展示与 L3 回放审计（网络失败仅告警、不定罪）
        "citation_checks": [
            {
                "evidence_id": c.evidence_id,
                "reference_id": c.reference_id,
                "state": c.state,
                "detail": c.detail,
            }
            for c in verdict.citation_checks
        ],
        # 证据指纹：用于案卷完整性校验与去重（任何证据内容变更 → 指纹变化）
        "evidence_fingerprints": {
            ev.evidence_id: compute_evidence_fingerprint(ev) for ev in evidence_list
        },
    }


class EvidenceScorer:
    """
    证据评分器

    从来源可信度、时效性、相关性三个维度对证据打分。
    """

    # 来源可信度权重
    SOURCE_WEIGHTS = {
        "pubmed": 0.9,
        "chembl": 0.85,
        "uniprot": 0.85,
        "drugbank": 0.8,
        "experiment": 0.75,
        "llm": 0.5,
        "agent_opinion": 0.4,
        "unknown": 0.2,
    }

    # 时效性半衰期（天）
    RECENCY_HALFLIFE_DAYS = 365 * 2

    def score(self, evidence: Evidence, topic: str) -> EvidenceScore:
        """
        对单条证据评分

        Args:
            evidence: 证据条目
            topic: 争议主题

        Returns:
            EvidenceScore
        """
        issues = []
        source = evidence.source.strip().lower()
        source_score = self.SOURCE_WEIGHTS.get(source, 0.2)
        if source == "unknown":
            issues.append("来源未知")

        recency_score = self._recency_score(evidence.timestamp)
        if recency_score < 0.5:
            issues.append("证据时效性较低")

        relevance_score = self._relevance_score(evidence.content, topic)
        if relevance_score < 0.3:
            issues.append("证据与主题相关性弱")

        if not evidence.reference_ids:
            issues.append("缺少可验证引用")

        overall = round((source_score * 0.4 + recency_score * 0.2 + relevance_score * 0.4), 3)
        return EvidenceScore(
            evidence_id=evidence.evidence_id,
            source_score=round(source_score, 3),
            recency_score=round(recency_score, 3),
            relevance_score=round(relevance_score, 3),
            overall_score=overall,
            issues=issues,
        )

    def _recency_score(self, timestamp: Optional[datetime]) -> float:
        """基于时间衰减计算时效性分数

        时区口径（架构审查 INFO-3）：now 用 timezone-aware UTC；
        历史存量 timestamp 是 naive-UTC（utcnow 落库），按 UTC
        补时区后相减。否则 aware-naive 相减抛 TypeError，被下方
        except 吞掉后时效分静默退化为 0.5。
        """
        if timestamp is None:
            return 0.5
        try:
            now = datetime.now(timezone.utc)
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            age_days = (now - timestamp).total_seconds() / 86400
            import math
            return max(0.0, min(1.0, math.exp(-age_days / self.RECENCY_HALFLIFE_DAYS)))
        except Exception:
            return 0.5

    def _relevance_score(self, content: str, topic: str) -> float:
        """基于关键词重叠计算相关性分数"""
        content_lower = content.lower()
        topic_words = [w for w in topic.lower().split() if len(w) > 1]
        if not topic_words:
            return 0.5
        matched = sum(1 for w in topic_words if w in content_lower)
        return min(1.0, matched / len(topic_words))


class RuleEngine:
    """
    规则引擎初筛

    基于明确规则快速排除低质或违规证据。

    P0-2 改造：
    - screen() 保持向后兼容（返回问题字符串列表）
    - screen_asymmetric() 新增非对称举证初筛（返回带严重等级的问题）
    - evaluate_with_rubric() 新增非对称举证 rubric 评估
    """

    def screen(self, evidence: Evidence) -> list[str]:
        """
        对证据进行初筛，返回违规/低质问题列表（向后兼容）

        Args:
            evidence: 证据条目

        Returns:
            问题列表（空表示通过初筛）
        """
        issues = []
        if not evidence.content or len(evidence.content.strip()) < 10:
            issues.append("证据内容过短")
        if not evidence.source:
            issues.append("证据来源为空")
        if evidence.source.strip().lower() == "agent_opinion" and not evidence.reference_ids:
            issues.append("Agent 观点缺少引用")
        # 结构预测工具禁令（v0.4.8）：AlphaFold2/3 均系统级禁用，
        # 合规降级链 OpenDDE → Chai1 → Boltz（与 rubric_engine
        # HARD_CONSTRAINT_PATTERNS 同源）。
        lower_content = evidence.content.lower()
        if ("alphafold2" in lower_content or "af2" in lower_content
                or "alphafold3" in lower_content or "af3" in lower_content):
            issues.append("违反结构预测工具禁令：检测到 AlphaFold2/3 引用")
        return issues

    def screen_asymmetric(self, evidence: Evidence) -> list[tuple[str, str]]:
        """
        非对称举证初筛 - 返回带严重等级的问题列表

        P0-2 核心改造：从"有罪推定"改为"非对称举证"
        - 内容过短/来源为空等问题 → 标记为 PENDING（待举证），不直接拒绝
        - AlphaFold2/3 引用违反结构预测工具硬约束 → 标记为 CRITICAL（必须拒绝）

        Args:
            evidence: 证据条目

        Returns:
            (issue_description, severity) 列表，severity 取值：
            - "PENDING"：待举证，不直接拒绝
            - "CRITICAL"：硬性违规，必须拒绝
            - "WARNING"：潜在问题，需关注
        """
        issues: list[tuple[str, str]] = []
        if not evidence.content or len(evidence.content.strip()) < 10:
            # 非对称举证：内容过短标记为待举证，不直接拒绝
            issues.append(("证据内容过短", "PENDING"))
        if not evidence.source:
            issues.append(("证据来源为空", "PENDING"))
        if evidence.source.strip().lower() == "agent_opinion" and not evidence.reference_ids:
            issues.append(("Agent 观点缺少引用", "PENDING"))
        # 结构预测工具禁令（v0.4.8）：AlphaFold2/3 均系统级禁用，
        # 合规降级链 OpenDDE → Chai1 → Boltz（与 rubric_engine
        # HARD_CONSTRAINT_PATTERNS 同源）
        lower_content = evidence.content.lower() if evidence.content else ""
        if ("alphafold2" in lower_content or "af2" in lower_content
                or "alphafold3" in lower_content or "af3" in lower_content):
            # 硬性违规：AlphaFold2/3 是系统级禁用，必须拒绝
            issues.append(
                ("违反结构预测工具禁令：检测到 AlphaFold2/3 引用", "CRITICAL")
            )
        return issues

    def evaluate_with_rubric(
        self,
        evidence_scores: list[EvidenceScore],
        evidence_map: dict[str, Evidence],
        topic: str = "",
    ):
        """
        使用非对称举证 rubric 评估证据集

        P0-2 新增：把审查从"有罪推定"改为"非对称举证"
        - 收集所有证据的待举证问题（PENDING）
        - 寻找矛盾证据对（CRITICAL）
        - 只有找到矛盾才定罪，否则默认通过

        Args:
            evidence_scores: 证据评分列表
            evidence_map: evidence_id -> Evidence 映射
            topic: 争议主题

        Returns:
            RubricVerdict 裁决结果
        """
        # 延迟导入避免循环依赖
        from app.core.virtual_lab.rubric_engine import AsymmetricRubric

        rubric = AsymmetricRubric()
        return rubric.evaluate(evidence_scores, evidence_map, topic)


class PIAgentArbiter:
    """
    PI Agent 裁决器

    基于证据评分、角色权重与争议主题生成裁决建议。
    当前版本使用规则逻辑；后续可接入 LLM 增强。
    """

    # 角色权重：领域专家权重更高。
    # v0.4.13 对齐论文 13 角色（大纲 L137 / 附录 D 权重表）：
    # - 论文 13 角色全部显式登记，其中 data-specialist（论文名）= data-analyst（平台
    #   实现名，app/core/agents/profiles/data-analyst.yaml），双命名映射不改名不复制；
    # - reviewer / agent 为历史兼容 fallback 键（v0.4.12 前既有），保留原值不动；
    # - 未登记角色仍走 arbitrate() 中的 0.5 默认 fallback，向后兼容；
    # - 既有 7 键数值不得修改（保证 v0.4.12 前行为与既有实验可复现）。
    ROLE_WEIGHTS = {
        "pi-agent": 1.0,
        "drug-researcher": 0.9,
        "structural-biologist": 0.9,
        "computational-chemist": 0.85,
        "synthetic-chemist": 0.8,
        "antibody-engineer": 0.8,
        "peptide-designer": 0.75,
        "data-specialist": 0.75,
        # data-specialist：论文名 = 平台实现名 data-analyst 的双命名映射
        # （附录 D D.3）。按计划红线不引入别名机制——论文侧（ROLE_WEIGHTS/
        # ROLE_POOL/校准实验/DDB）统一用论文名；平台侧真实证据流的
        # data-analyst 角色证据走 0.5 fallback，该口径差异已在 D.3 显式声明。
        "qc-inspector": 0.7,
        "knowledge-engineer": 0.65,
        "reflexion-agent": 0.6,
        "project-coordinator": 0.6,
        "report-expert": 0.5,
        # 历史 fallback 键（非论文 13 角色，仅为 v0.4.12 前证据流兼容保留）
        "reviewer": 0.7,
        "agent": 0.5,
    }

    def arbitrate(
        self,
        topic: str,
        evidence_scores: list[EvidenceScore],
        evidence_map: dict[str, Evidence],
    ) -> tuple[str, str, float]:
        """
        生成 PI Agent 裁决

        Args:
            topic: 争议主题
            evidence_scores: 证据评分列表
            evidence_map: evidence_id -> Evidence 映射

        Returns:
            (verdict, rationale, confidence)
        """
        if not evidence_scores:
            return "无法裁决", "没有可用证据", 0.0

        # 按角色权重对证据 overall_score 做加权平均，
        # 避免旧实现中 weight 同时乘以 score 导致分子分母抵消、结果恒为 1.0 的 bug
        weighted_sum = 0.0
        total_weight = 0.0
        for score in evidence_scores:
            ev = evidence_map.get(score.evidence_id)
            if not ev:
                continue
            role_weight = self.ROLE_WEIGHTS.get(ev.role.strip().lower(), 0.5)
            weighted_sum += role_weight * score.overall_score
            total_weight += role_weight

        avg_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        if avg_score >= 0.7:
            decision = "支持提案"
        elif avg_score >= 0.4:
            decision = "有条件支持"
        else:
            decision = "驳回提案"

        rationale = (
            f"PI Agent 基于 {len(evidence_scores)} 条证据加权评分，"
            f"平均加权分 {avg_score:.2f}，决定：{decision}。"
        )
        confidence = round(min(1.0, avg_score + 0.1), 3)
        return decision, rationale, confidence


class IndependentReviewer:
    """
    独立评审 Agent

    从不同视角交叉验证裁决，检查潜在偏见与遗漏。
    """

    def review(
        self,
        pi_decision: str,
        evidence_scores: list[EvidenceScore],
        evidence_map: dict[str, Evidence],
    ) -> list[str]:
        """
        生成独立评审意见

        Args:
            pi_decision: PI Agent 裁决
            evidence_scores: 证据评分列表
            evidence_map: 证据映射

        Returns:
            评审意见列表
        """
        notes = []
        high_scores = [s for s in evidence_scores if s.overall_score >= 0.7]
        low_scores = [s for s in evidence_scores if s.overall_score < 0.4]

        if not high_scores:
            notes.append("缺少高质量证据支撑，建议补充实验或文献。")
        if low_scores:
            ids = [s.evidence_id for s in low_scores]
            notes.append(f"以下证据质量较低，应谨慎采信：{ids}")

        # 检查是否所有证据来自同一角色（潜在偏见）
        roles = {evidence_map[s.evidence_id].role for s in evidence_scores if s.evidence_id in evidence_map}
        if len(roles) == 1:
            notes.append("所有证据均来自同一角色，存在确认偏误风险。")

        if pi_decision == "驳回提案" and high_scores:
            notes.append("PI Agent 驳回了提案，但存在高质量支持证据，建议人工复核。")

        return notes

    def contradicts_pi_decision(
        self,
        pi_decision: str,
        reviewer_notes: list[str],
    ) -> bool:
        """
        判定独立评审（证据型核查轨）与 PI 仲裁（规则/模型轨）是否结论冲突。

        §4.7 升级条款 6（v0.4.9）：吸收 Cecil & Freedman 2026 双轨仲裁实证——
        规则/模型仲裁轨（PIAgentArbiter）与独立评审的证据型核查轨
        （IndependentReviewer，仅凭证据质量/多样性核查，不重算科学答案）
        结论冲突时，单轨残余分歧必须显式出口（强制升级），而非静默吞掉。

        判定口径（保守，避免误升级）：
        - 评审意见中出现建议人工复核信号，而 PI 仲裁恰恰给出终局性正面
          裁决（"支持提案"）→ 两轨分歧：评审认为证据基础不足以支撑终局采信，
          PI 却予采信；
        - 其余组合（PI 已驳回/有条件支持=本身已谨慎；或评审无实质异议）
          一律不判分歧，由既有升级条款 1-5 兜底。

        注：条款 4（评审异议即升级）捕获"任意异议"，条款 6 专捕"异议方向与
        裁决方向矛盾"的语义分歧子集——这是比"有异议"更强、更具体的分歧信号，
        用于满足大纲 Tab.5 分歧率统计需求，二者并存、互不替代。

        Args:
            pi_decision: PI Agent 裁决文本
            reviewer_notes: 独立评审意见列表

        Returns:
            True = 两轨结论语义冲突
        """
        # 评审建议人工复核 / 高质量证据被驳回 = 证据轨倾向"不应轻易采信"
        review_pushes_back = any(
            "建议人工复核" in note for note in reviewer_notes
        )
        # PI 给出终局性正面裁决 = 模型轨倾向"采信"
        pi_affirmative = pi_decision == "支持提案"
        # 分歧 = 一轨采信、另一轨要求复核（方向矛盾）
        return pi_affirmative and review_pushes_back


class HumanBOSSOverride:
    """
    Human BOSS 覆盖清单生成器

    列出需要人类最终确认的高风险事项。
    """

    def generate_items(
        self,
        pi_decision: str,
        confidence: float,
        evidence_scores: list[EvidenceScore],
        reviewer_notes: list[str],
        max_contradiction_intensity: int = 0,
        dual_track_divergence: bool = False,
    ) -> list[str]:
        """
        生成 Human BOSS 覆盖清单

        升级条款（论文大纲 §4.7，任一命中即进入清单）：
        1. confidence < 0.6
        2. decision ∈ {驳回提案, 有条件支持}（关键决策）
        3. 存在 overall_score < 0.3 的证据被采信
        4. 独立评审提出异议
        5. （v0.3 新增）存在 intensity=3 的定罪矛盾对
           ——吸收 Kumar et al. 强度分诊思想：对抗级矛盾强制人工复核
        6. （v0.4.9 新增）双轨仲裁分歧：规则/模型仲裁轨（PIAgentArbiter）
           与证据型核查轨（IndependentReviewer）结论语义冲突
           ——吸收 Cecil & Freedman 2026 双轨协议：单轨残余分歧必须显式出口

        Args:
            pi_decision: PI Agent 裁决
            confidence: 置信度
            evidence_scores: 证据评分列表
            reviewer_notes: 独立评审意见
            max_contradiction_intensity: 定罪矛盾对最大强度（v0.3 升级条款 5
                输入；默认 0=无矛盾对，既有四参调用零改动、行为等价 v0.2）
            dual_track_divergence: 双轨仲裁分歧标志（v0.4.9 升级条款 6 输入；
                默认 False，既有调用零改动、行为等价 v0.4.8）

        Returns:
            覆盖事项列表
        """
        items = []
        if confidence < 0.6:
            items.append(f"裁决置信度较低（{confidence:.2f}），需要人类确认。")
        if pi_decision in ("驳回提案", "有条件支持"):
            items.append(f"裁决结果为 '{pi_decision}'，涉及关键决策，需要人类确认。")
        if any(s.overall_score < 0.3 for s in evidence_scores):
            items.append("存在低质量证据被采信，需要人类审核。")
        if reviewer_notes:
            items.append("独立评审 Agent 提出异议，需要人类复核。")
        # 升级条款 5（v0.3）：对抗级（intensity=3）定罪矛盾对强制人工复核
        if max_contradiction_intensity >= 3:
            items.append(
                "存在强度 3/3（对抗级）的定罪矛盾对，"
                "依据升级条款 5 强制人工复核。"
            )
        # 升级条款 6（v0.4.9）：双轨仲裁结论分歧强制人工复核
        if dual_track_divergence:
            items.append(
                "规则/模型仲裁轨与证据型核查轨结论分歧，"
                "依据升级条款 6 强制人工复核。"
            )
        return items


class ConsensusCourt(BaseService):
    """
    共识法庭

    对 Agent 群聊中未达成共识的冲突进行结构化审理，
    输出裁决结果、独立评审意见与 Human BOSS 覆盖清单。
    """

    def __init__(self, db: Session, citation_verifier: Optional[CitationVerifier] = None):
        super().__init__(db)
        self.evidence_scorer = EvidenceScorer()
        self.rule_engine = RuleEngine()
        self.pi_arbiter = PIAgentArbiter()
        self.reviewer = IndependentReviewer()
        self.human_override = HumanBOSSOverride()
        # G1 三态引用验证：默认 DefaultCitationVerifier（预计算结果 + 诚实降级），
        # 生产可注入真实在线核验实现（CitationVerifier 子类）
        self.citation_verifier = citation_verifier or DefaultCitationVerifier()

    def hold_court(
        self,
        session_id: str,
        topic: str,
        evidence_list: list[Evidence],
    ) -> CourtVerdict:
        """
        召开共识法庭审理冲突

        裁决产生后通过 `_persist_verdict_event` 写入 ProjectEvent
        （event_type=CONSENSUS_COURT_VERDICT），满足论文可审计性要求：
        证据评分、PI 理由、评审异议、升级原因、证据指纹全链路留痕。
        落库失败不阻断裁决返回（审计尽力而为，与 audit_service 同一语义）。

        Args:
            session_id: 关联群聊会话 ID
            topic: 争议主题
            evidence_list: 证据列表

        Returns:
            CourtVerdict
        """
        logger.info("召开共识法庭: session=%s topic=%s evidence_count=%d", session_id, topic, len(evidence_list))

        # 规则引擎初筛
        filtered_evidence = []
        for ev in evidence_list:
            issues = self.rule_engine.screen(ev)
            if not issues:
                filtered_evidence.append(ev)
            else:
                logger.warning("证据 %s 未通过初筛: %s", ev.evidence_id, issues)

        evidence_map = {ev.evidence_id: ev for ev in filtered_evidence}

        # 证据评分
        scores = []
        for ev in filtered_evidence:
            score = self.evidence_scorer.score(ev, topic)
            scores.append(score)

        # 非对称举证 rubric 评估（v0.3：定罪矛盾对强度喂给升级条款 5）。
        # 延迟导入避免与 rubric_engine 的循环依赖。
        from app.core.virtual_lab.rubric_engine import AsymmetricRubric

        rubric = AsymmetricRubric()
        rubric_verdict = rubric.evaluate(scores, evidence_map, topic)
        # 定罪矛盾对最大强度：无矛盾对时 max([]) 兜底 0（语义=无对抗信号）
        max_intensity = max(
            (c.intensity for c in rubric_verdict.contradictions), default=0
        )

        # PI Agent 裁决
        pi_decision, rationale, confidence = self.pi_arbiter.arbitrate(topic, scores, evidence_map)

        # 独立评审 Agent 复核
        reviewer_notes = self.reviewer.review(pi_decision, scores, evidence_map)

        # v0.4.9 升级条款 6：双轨仲裁分歧判定——规则/模型仲裁轨（PI 裁决）
        # 与证据型核查轨（独立评审）结论语义冲突
        dual_track_divergence = self.reviewer.contradicts_pi_decision(
            pi_decision, reviewer_notes
        )

        # Human BOSS 覆盖清单（含 v0.3 升级条款 5：intensity=3 强制人工复核；
        # v0.4.9 升级条款 6：双轨仲裁分歧强制人工复核）
        override_items = self.human_override.generate_items(
            pi_decision, confidence, scores, reviewer_notes,
            max_contradiction_intensity=max_intensity,
            dual_track_divergence=dual_track_divergence,
        )

        # ------------------------------------------------------------------
        # G1 三态引用验证（论文机制生产内置）：
        # 对通过初筛的证据逐引用核验，产出 valid/invalid/network_failed。
        # - network_failed：仅告警（评审备注），绝不改判、绝不定罪、不强制升级；
        # - invalid（确认伪造引用）：确认级信号，记入该证据 issues 并强制人工复核；
        # - valid：无需额外动作。
        # 注意：措辞刻意避开"建议人工复核"（IndependentReviewer.
        # contradicts_pi_decision 的分歧触发短语），避免核验告警被误判为
        # 双轨仲裁分歧而虚假升级。
        # ------------------------------------------------------------------
        citation_checks: list[CitationCheck] = []
        for ev in filtered_evidence:
            citation_checks.extend(self.citation_verifier.verify(ev, topic))
        invalid_checks = [
            c for c in citation_checks if c.state == CITATION_STATE_INVALID
        ]
        network_failed_checks = [
            c for c in citation_checks if c.state == CITATION_STATE_NETWORK_FAILED
        ]
        # invalid 引用写入对应证据的评分 issues（确认级质量问题留痕）
        if invalid_checks:
            score_map = {s.evidence_id: s for s in scores}
            for c in invalid_checks:
                score = score_map.get(c.evidence_id)
                if score is not None:
                    score.issues.append(f"引用核验确认失效: {c.reference_id}")
            reviewer_notes = reviewer_notes + [
                f"证据 {c.evidence_id} 的引用 {c.reference_id} 经核验确认失效（疑似伪造）。"
                for c in invalid_checks
            ]
            override_items = override_items + [
                f"存在 {len(invalid_checks)} 条确认失效（疑似伪造）引用，"
                "需人工复核相关证据的真实性。"
            ]
        # network_failed 引用：仅告警（三态语义：网络失败不定罪）。
        # 刻意不写入 Human BOSS 覆盖清单、不影响 escalated——核验能力缺失
        # 不是高风险事项，强制升级会让所有含未核验引用的案卷都触发人工复核，
        # 违背"网络失败不定罪"红线（network_failure 实验红线 RL1 的生产版）。
        if network_failed_checks:
            reviewer_notes = reviewer_notes + [
                f"{len(network_failed_checks)} 条引用因网络失败未能核验"
                "（三态语义：仅告警，不参与定罪）。"
            ]

        # rubric 定罪（找到矛盾对）本身即关键裁决信号，并入升级判定；
        # 无矛盾对时 max_intensity=0，条件恒假，v0.2 行为等价。
        # 三态引用验证的确认级信号（invalid 引用）同样并入升级判定：
        # 存在疑似伪造引用的案卷必须升级人工复核（保守原则）。
        escalated = (
            bool(override_items)
            or confidence < 0.6
            or max_intensity >= 3
            or bool(invalid_checks)
        )

        verdict_text = (
            f"ConsensusCourt 裁决：{pi_decision}（置信度 {confidence:.2f}）。"
        )

        verdict = CourtVerdict(
            verdict=verdict_text,
            decision=pi_decision,
            confidence=confidence,
            evidence_scores=scores,
            pi_rationale=rationale,
            reviewer_notes=reviewer_notes,
            human_override_items=override_items,
            escalated=escalated,
            max_contradiction_intensity=max_intensity,
            dual_track_divergence=dual_track_divergence,
            citation_checks=citation_checks,
        )

        # 裁决落库（P0 可审计性改造）
        self._persist_verdict_event(session_id, topic, evidence_list, verdict)
        return verdict

    def _persist_verdict_event(
        self,
        session_id: str,
        topic: str,
        evidence_list: list[Evidence],
        verdict: CourtVerdict,
    ) -> Optional[str]:
        """
        将法庭裁决写入项目事件表（ProjectEvent）。

        关联链路：session_id → AgentChatSession.project_id → ProjectEvent.project_id；
        correlation_id 使用 session_id，供 L3 历史争议回放按会话聚合
        "辩论消息 → 裁决 → 审批" 全案卷。

        会话不存在 project_id（前端助手会话）或落库失败时仅记录日志，
        不影响裁决结果返回——审计尽力而为，业务不被阻断。

        Args:
            session_id: 群聊会话 ID
            topic: 争议主题
            evidence_list: 参与审理的证据列表
            verdict: 裁决结果

        Returns:
            成功时返回事件 ID；未落库时返回 None
        """
        try:
            session = self.db.get(AgentChatSession, session_id)
            if session is None or not session.project_id:
                logger.info(
                    "裁决未落库（会话无关联项目）: session=%s topic=%s decision=%s",
                    session_id, topic, verdict.decision,
                )
                return None
            # 延迟导入：consensus_court 被 deterministic_hub 下游引用路径覆盖时会成环
            from app.core.virtual_lab.deterministic_hub import DeterministicHub

            payload = verdict_to_payload(verdict, topic, evidence_list)
            payload["session_id"] = session_id
            event = DeterministicHub(self.db).record_event(
                project_id=session.project_id,
                event_type=COURT_VERDICT_EVENT_TYPE,
                actor_type="agent",
                actor_id="consensus-court",
                payload=payload,
                correlation_id=session_id,
            )
            return event.id
        except Exception as exc:  # noqa: BLE001 - 审计失败不阻断裁决
            logger.warning(
                "共识法庭裁决落库失败（不阻断裁决返回）: session=%s error=%s",
                session_id, exc,
            )
            return None

    def evidence_from_chat_messages(
        self,
        messages: list[AgentChatMessage],
        default_source: str = "agent_opinion",
    ) -> list[Evidence]:
        """
        从群聊消息中提取证据

        Args:
            messages: 群聊消息列表
            default_source: 默认证据来源

        Returns:
            Evidence 列表
        """
        evidence_list = []
        for msg in messages:
            if msg.sender_role == "system":
                continue
            ref_ids = []
            if msg.attachments:
                ref_ids = [a.get("reference_id", "") for a in msg.attachments if a.get("reference_id")]
            ev = Evidence(
                evidence_id=f"ev-{msg.id}",
                source=default_source,
                content=msg.content,
                role=msg.sender_role,
                timestamp=msg.created_at,
                reference_ids=ref_ids,
                metadata={"message_id": msg.id, "message_type": msg.message_type},
            )
            evidence_list.append(ev)
        return evidence_list


def compute_evidence_fingerprint(evidence: Evidence) -> str:
    """
    计算证据指纹

    用于证据去重与审计。
    """
    payload = {
        "source": evidence.source,
        "content": evidence.content,
        "role": evidence.role,
        "reference_ids": sorted(evidence.reference_ids),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
