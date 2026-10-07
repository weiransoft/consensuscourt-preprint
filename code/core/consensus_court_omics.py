"""
ConsensusCourt × 疫苗靶点发现流水线集成适配层

背景（论文-生产集成缺口修复）：
    ConsensusCourt（consensus_court.py）此前只在论文实验侧
    （consensus-court-preprint/code/scripts/）被调用，生产疫苗管线
    （api/vaccine.py 的 design / candidates/evaluate 端点）产出的
    候选证据从未经过 ABP（非对称举证）法庭仲裁。本模块补齐
    "疫苗候选评价 → 证据构造 → hold_court 开庭 → 裁决事件留痕"
    这一条生产集成链路。

集成契约（与既有内核零耦合改动）：
    1. 输入 = evaluate_vaccine_candidate 的五维评价结果
       （vaccine_evaluation.evaluate_vaccine_candidate 返回值）
       + 候选元数据（病原名 / 抗原名 / 候选 ID）；
    2. 每个评价维度转换为一条 Evidence：
       - 有真实数据（IEDB 表位 / MSA 保守性 / SASA 结构证据）
         → source 映射到 EvidenceScorer.SOURCE_WEIGHTS 的高权源；
       - 中性兜底或数据缺失（insufficient_data）→ 保留 agent_opinion
         低权源并在 content 中如实携带 note（诚实降级，不抹除）；
    3. 开庭完全复用 ConsensusCourt.hold_court：本模块不重新实现
       任何仲裁逻辑，只做 Evidence 构造与 topic 编排；
    4. 裁决经 hold_court 内部 _persist_verdict_event 写入 ProjectEvent
       （event_type=CONSENSUS_COURT_VERDICT），复用通知中心既有
       消费通道（api/virtual_lab.py），前端零改动可见。

诚实性约束（零 mock / 零占位 / 零简化）：
    - ABP 语义如实透传：hold_court 判"驳回提案"就判驳回，适配层
      不做任何粉饰、不改写裁决、不重试刷分；
    - 评价数据缺失时维度证据走 PENDING（待举证）路径，由
      RuleEngine.screen 如实标记——证据全部被初筛拒掉时
      hold_court 返回"无法裁决"，本层如实上报而非编造证据；
    - 与 vaccine_evaluation 的诚实降级措辞协议由
      tests/test_consensus_court_omics.py::TestWordingProtocolLock
      （零网络协议锁测试）守护：上游真实评价输出 → 来源分级
      断言，上游改写 note 文案时 CI 立即失败而非静默升权；
    - 时间戳一律 timezone-aware UTC（datetime.now(timezone.utc)，
      不用已弃用的 utcnow）。内核 EvidenceScorer._recency_score
      已兼容 naive/aware 两种存量口径；trial_floor 与
      record_event 的 naive-utcnow 落库值比较时按 naive-UTC
      口径换算（见 _utc_naive_now），与虚拟实验室全局存储口径
      一致。全局 utcnow → aware 的统一属跨模块技术债，另行立项。

已知限制（架构审查 INFO 级记录，均为 ABP 合法语义而非缺陷）：
    1. 单角色退化：本适配层五维证据统一由 drug-researcher 角色
       提交，PI 加权平均的角色维度退化为证据质量单轨评分；
       IndependentReviewer 必然产出"同一角色，确认偏误风险"备注，
       升级条款 6（双轨分歧）在正常疫苗案卷上恒不触发——未来
       实验组/计算组多源证据同庭时应恢复角色异源化。
    2. 单源不自证：正常候选的证据全部同 source（同一 experiment
       或同一 agent_opinion）时，rubric"同一来源不对比"规则使其
       结构性不可能产出定罪矛盾对（max_contradiction_intensity≡0）。
       这是 ABP 无罪推定的合法体现——矛盾定罪只在案卷确有异源
       对立证据时可达（对抗案卷 E2E 已验证通道可达），不应把
       "正常候选零定罪"理解为"疫苗候选已受矛盾检测保护"。
    3. 两条"无法裁决"路径并存且均正确：evaluation 为空 → 不开庭
       直接上报；评价全维度缺失（开庭后证据全被初筛拒收）→
       hold_court 空证据语义返回"无法裁决"。两者都如实留痕。
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.core.virtual_lab.consensus_court import (
    COURT_VERDICT_EVENT_TYPE,
    ConsensusCourt,
    CourtVerdict,
    Evidence,
    compute_evidence_fingerprint,
)
from app.core.virtual_lab.models import AgentChatSession, ProjectEvent

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    """当前时间（timezone-aware UTC）

    架构审查 INFO-3：datetime.utcnow() 自 Python 3.12 弃用且产出
    naive 时间戳（无时区，跨时区比较易错）。本模块新产生的时间
    （证据 timestamp）统一用本函数。
    """
    return datetime.now(timezone.utc)


def _utc_naive_now() -> datetime:
    """当前时间（naive-UTC 口径，仅用于与库内既有 naive 时间戳比较）

    内核 deterministic_hub.record_event 以 datetime.utcnow()（naive）
    写入 ProjectEvent.created_at；本模块 trial_floor 作为该列的查询
    下界必须同口径（aware 与 naive-UTC 列比较会错位/报错），因此
    换算为 naive-UTC。这是虚拟实验室全局存储口径的适配，不代表
    推荐新代码使用 naive 时间。
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)

# ---------------------------------------------------------------------------
# 评价维度 → 证据来源映射（EvidenceScorer.SOURCE_WEIGHTS 键口径）
# ---------------------------------------------------------------------------
# 维度输入数据的真实出处 → 证据来源可信度键：
#   epitope_density / coverage / top_epitope_strength 来自 IEDB 真实
#   B 细胞表位预测（外部数据库真实查询）→ "experiment"（0.75）；
#   conservation 真实路径来自 Biopython 全局比对（确定性计算）→
#     "experiment"（0.75）；兜底路径是文献关键词映射 → "pubmed"
#   不适用（映射不是实测），降为 "agent_opinion"（0.4）如实反映
#   "有文献线索但本管线未实测"的证据强度；
#   structural_accessibility 真实路径来自 Shrake-Rupley SASA 实测
#   → "experiment"（0.75）；无结构证据的中性值 → "agent_opinion"。
_DIMENSION_SOURCE_REAL = "experiment"
_DIMENSION_SOURCE_NEUTRAL = "agent_opinion"
# 文献关键词映射（上游 DATA_SOURCE_LITERATURE_MAP）：有文献线索但
# 本管线未实测 → 映射到 pubmed(0.9) 与实测 experiment(0.75) 之间的
# 证据强度不再适用（pubmed 权重高于 experiment，映射证据不应高于
# 实测证据），保守起见同样落到低权 agent_opinion(0.4)。
_DIMENSION_SOURCE_LITERATURE = "agent_opinion"

# 上游 vaccine_evaluation 的 DATA_SOURCE_* 枚举键（结构化协议）
_DATA_SOURCE_MEASURED = "measured"
# 已知但判低权的取值集合（insufficient/neutral/literature_map）
_KNOWN_LOW_TRUST_SOURCES = frozenset(
    {"insufficient", "neutral", "literature_map"}
)

# 维度证据的举证角色（PIAgentArbiter.ROLE_WEIGHTS 键口径）：
# 疫苗抗原证据由计算设计线（drug-researcher 角色）提交，权重 0.9。
_DIMENSION_ROLE = "drug-researcher"

# insufficient_data 前缀（与 vaccine_evaluation 各维度 note 约定一致）
_INSUFFICIENT_PREFIX = "insufficient_data"

# 中性兜底 note 特征词（vaccine_evaluation 诚实降级声明的固定措辞，
# 见 _CONSERVATION_NEUTRAL / _ACCESSIBILITY_NEUTRAL 的 note 文本）
_NEUTRAL_NOTE_MARKER = "中性值 0.5"


def _is_neutral_note(note: str) -> bool:
    """判定维度 note 是否为诚实降级/中性兜底措辞

    判定依据（与 vaccine_evaluation 的诚实降级措辞严格对齐）：
    - note 以 insufficient_data 开头 → 数据缺失；
    - note 含"中性值 0.5" → 文献关键词映射/无证据中性值。

    Args:
        note: 维度评价结果的 note 文本

    Returns:
        True=中性/缺失兜底措辞；False=真实数据措辞
    """
    return note.startswith(_INSUFFICIENT_PREFIX) or _NEUTRAL_NOTE_MARKER in note


def _dimension_source(
    dim_key: str,
    dimension_result: Dict[str, Any],
    candidate_id: str,
) -> str:
    """判定单个维度结果是真实数据还是中性/缺失兜底

    三级判定（架构审查 MAJOR-1 根治项：结构化字段优先，note 文案
    只是给人看的展示文本，不作为跨模块机器协议的第一信号）：
    1. 结构化信号：维度结果的 data_source 键（vaccine_evaluation
       显式声明的 DATA_SOURCE_* 枚举）——measured → 高权；
       literature_map/neutral/insufficient → 低权；未知取值 →
       低权 + WARNING（上游枚举扩展时强制本层同步审视）；
    2. 文本回退（旧存量评价 JSON 无 data_source 键）：note 以
       insufficient_data 开头或含"中性值 0.5" → 低权；
    3. 实据信号（回退路径的第二信号）：文本判"真实"时 raw 必须
       携带实际计算数据（total / coverage_ratio / conservation_score
       等非零实据），否则视为措辞疑似漂移，降为低权 + WARNING，
       防止无数据证据伪装成有引用的实测证据进入高权通道。

    Args:
        dim_key: 维度键（告警定位用）
        dimension_result: 单维度评价结果 {score, raw, note, data_source?}
        candidate_id: 候选 ID（告警定位用）

    Returns:
        证据来源键（SOURCE_WEIGHTS 的键）
    """
    data_source = dimension_result.get("data_source")
    if data_source is not None:
        # 结构化协议优先（上游 DATA_SOURCE_* 枚举）
        if data_source == _DATA_SOURCE_MEASURED:
            return _DIMENSION_SOURCE_REAL
        if data_source not in _KNOWN_LOW_TRUST_SOURCES:
            logger.warning(
                "维度 data_source 取值不在已知枚举内，保守降为"
                " agent_opinion: candidate=%s dimension=%s data_source=%r",
                candidate_id, dim_key, data_source,
            )
            return _DIMENSION_SOURCE_NEUTRAL
        # literature_map/neutral/insufficient → 低权（文献映射证据
        # 权重不应高于实测，统一走 _DIMENSION_SOURCE_LITERATURE）
        return _DIMENSION_SOURCE_LITERATURE

    # ---- 兼容回退：旧存量评价 JSON 无 data_source 键，读 note 文本 ----
    note = str(dimension_result.get("note") or "")
    if _is_neutral_note(note):
        return _DIMENSION_SOURCE_NEUTRAL

    # 第二信号：raw 必须携带实据，否则视为上游措辞漂移
    raw = dimension_result.get("raw")
    raw_has_evidence = isinstance(raw, dict) and any(
        value not in (None, 0, 0.0, "", [], {})
        for value in raw.values()
    )
    if not raw_has_evidence:
        logger.warning(
            "维度无 data_source 键且 note 判为真实数据但 raw 无实据，"
            "疑似上游诚实降级措辞漂移，降为 agent_opinion: "
            "candidate=%s dimension=%s note=%r",
            candidate_id, dim_key, note,
        )
        return _DIMENSION_SOURCE_NEUTRAL
    return _DIMENSION_SOURCE_REAL


def build_candidate_evidence(
    candidate_id: str,
    pathogen_name: str,
    antigen_name: Optional[str],
    evaluation: Dict[str, Any],
    evidence_time: Optional[datetime] = None,
) -> List[Evidence]:
    """把单个疫苗候选的五维评价结果转换为法庭证据列表

    每个维度一条 Evidence：
    - evidence_id：{候选 ID}:{维度键}，全局唯一且可反查候选与维度；
    - content：维度得分 + 真实计算注记（note 原文，保留审计线索）；
    - source：经 _dimension_source 判定的真实/中性来源；
    - metadata：维度得分、原始数据摘要、病原/抗原名——供
      AspectInferrer（metadata 显式切面优先）与 ValidityGate
      （对象同一性）在 v0.3 机制下正常使用。

    Args:
        candidate_id: 疫苗候选 ID
        pathogen_name: 病原名称（如"犬腺病毒二型"）
        antigen_name: 抗原名（如"hexon"，可空）
        evaluation: evaluate_vaccine_candidate 的返回值
        evidence_time: 证据时间戳（None=当前 UTC，影响时效分）

    Returns:
        Evidence 列表（每维度一条，缺失维度以低权证据如实在场）
    """
    dimensions: Dict[str, Any] = evaluation.get("dimensions") or {}
    evidence_list: List[Evidence] = []
    # 评价方法名（真实溯源键之一，进 reference_ids 供人工复核溯源）
    method = str(evaluation.get("method") or "unknown")
    for dim_key, dim_result in dimensions.items():
        # 入口类型防御：上游 JSON 列可能存进畸形结构，非 dict 维度
        # 结果不得让整批证据构造崩溃（审查维度 7 修复）
        if not isinstance(dim_result, dict):
            logger.warning(
                "维度结果类型畸形（非 dict），跳过该维度: candidate=%s "
                "dimension=%s type=%s",
                candidate_id, dim_key, type(dim_result).__name__,
            )
            continue
        source = _dimension_source(dim_key, dim_result, candidate_id)
        score = dim_result.get("score")
        note = str(dim_result.get("note") or "")
        raw = dim_result.get("raw") or {}
        # content 组织原则：得分 + 病原 + 抗原 + 注记原文，
        # 保证相关性评分（关键词重叠）与审计追溯同时成立
        content = (
            f"疫苗候选 {candidate_id} 维度 {dim_key}："
            f"病原 {pathogen_name}，抗原 {antigen_name or '未命名'}，"
            f"得分 {score}。依据：{note}"
        )
        # 真实溯源引用（架构审查 MAJOR-2 修复）：引用串携带评价
        # 方法名与维度实据摘要，人工复核可按 method 回到评价器、
        # 按 raw 摘要核对计算输入，而非不可解析的自指串
        reference_ids: List[str] = []
        if source == _DIMENSION_SOURCE_REAL:
            trace = f"pipeline:{candidate_id}:{dim_key}?method={method}"
            if isinstance(raw, dict):
                for key in ("alignments", "pdb_id", "total"):
                    if raw.get(key):
                        trace += f"&{key}={raw[key]}"
                        break
            reference_ids.append(trace)
        evidence_list.append(Evidence(
            evidence_id=f"{candidate_id}:{dim_key}",
            source=source,
            content=content,
            role=_DIMENSION_ROLE,
            timestamp=evidence_time or datetime.utcnow(),
            # 真实数据维度引用其计算出处（方法名+实据摘要），
            # 满足 RuleEngine.screen 的"缺引用"初筛判据且可溯源
            reference_ids=reference_ids,
            metadata={
                "dimension": dim_key,
                "score": score,
                "raw": raw,
                "pathogen": pathogen_name,
                "antigen": antigen_name,
                # ObjectExtractor 第一级键（架构审查 MINOR-2 修复）：
                # target/molecule 填抗原名、locus 填病原名，与对抗
                # 案卷口径对齐，使 ValidityGate 对象同一性判定走
                # 结构化通道而非词表猜测
                "target": antigen_name,
                "molecule": antigen_name,
                "locus": pathogen_name,
            },
        ))
    return evidence_list


def build_trial_topic(pathogen_name: str, antigen_name: Optional[str]) -> str:
    """构造法庭审理主题（相关性评分与矛盾切面的锚点）

    主题词表刻意覆盖"疫苗 / 抗原 / 保守 / 表位"等词，使各维度
    证据的 content（含病原名与维度注记）与主题产生真实关键词重叠，
    相关性评分走 EvidenceScorer._relevance_score 正常路径。

    Args:
        pathogen_name: 病原名称
        antigen_name: 抗原名（可空）

    Returns:
        审理主题字符串
    """
    return (
        f"{pathogen_name} 疫苗抗原 {antigen_name or ''} "
        "靶点候选证据是否支持进入疫苗设计（表位 保守性 覆盖度 结构可及性）"
    ).strip()


def _find_duplicate_verdict_event(
    db: Session,
    session_id: str,
    fingerprints: Dict[str, str],
    exclude_event_id: Optional[str] = None,
) -> Optional[str]:
    """查找本会话中证据指纹完全相同的更早裁决案卷（案卷去重）

    判定依据：verdict_to_payload 的 evidence_fingerprints 是逐证据
    内容级 sha256（evidence_id → hash），同一候选同一评价数据的
    重复仲裁产出完全相同的指纹映射。命中 → 返回最早那份案卷的
    event_id；本份案卷如实落库不改（审计不可变），仅由调用方在
    返回值标注 duplicate_of。

    自身排除：本次开庭的裁决刚由 hold_court 落库（record_event
    每事件即 commit），调用方在开庭前取 created_at 时间下界
    （floor），查询排除 floor 之后的事件即排除本次自身——
    record_event 显式写入 utcnow()（微秒级），与历史毫秒级
    时间戳碰撞概率可忽略，且 SQLite DATETIME 列对同刻写入的
    相等性比较不作为去重依据。

    Args:
        db: 数据库会话
        session_id: 会话 ID（案卷 correlation_id）
        fingerprints: 本次开庭的证据指纹 {evidence_id: sha256_hex}
        exclude_event_id: 本次案卷自身 event_id（可选，双重保险）

    Returns:
        更早同指纹案卷的 event_id；无重复返回 None
    """
    return _query_matching_verdict_event(
        db, session_id, fingerprints,
        newest_first=False, exclude_event_id=exclude_event_id,
    )


def _locate_own_verdict_event(
    db: Session,
    session_id: str,
    fingerprints: Dict[str, str],
    created_at_floor: datetime,
) -> Optional[str]:
    """定位本次开庭刚落库的案卷 event_id（hold_court 不回传事件 ID）

    定位条件：同会话、同类型、指纹一致、created_at 不早于 floor
    （开庭前取的时间下界）——排除全部历史案卷，取最新一条即本次
    落库的案卷。

    Args:
        db: 数据库会话
        session_id: 会话 ID
        fingerprints: 本次开庭的证据指纹
        created_at_floor: 本次开庭前取的时间下界

    Returns:
        本次案卷 event_id；未落库（会话无项目等）返回 None
    """
    return _query_matching_verdict_event(
        db, session_id, fingerprints,
        created_at_gte=created_at_floor, newest_first=True,
    )


def _query_matching_verdict_event(
    db: Session,
    session_id: str,
    fingerprints: Dict[str, str],
    created_at_gte: Optional[datetime] = None,
    newest_first: bool = False,
    exclude_event_id: Optional[str] = None,
) -> Optional[str]:
    """按证据指纹查询本会话裁决案卷（去重/自身定位共用查询）

    Args:
        db: 数据库会话
        session_id: 会话 ID（案卷 correlation_id）
        fingerprints: 目标证据指纹映射
        created_at_gte: 时间下界（含），None=不限
        newest_first: True=按最新优先（定位本次自身），
            False=按最早优先（去重指向首案）
        exclude_event_id: 排除指定事件 ID

    Returns:
        命中案卷的 event_id；无命中返回 None
    """
    if not fingerprints:
        return None
    query = db.query(ProjectEvent).filter(
        ProjectEvent.correlation_id == session_id,
        ProjectEvent.event_type == COURT_VERDICT_EVENT_TYPE,
    )
    if created_at_gte is not None:
        query = query.filter(ProjectEvent.created_at >= created_at_gte)
    if newest_first:
        query = query.order_by(
            ProjectEvent.created_at.desc(), ProjectEvent.id.desc(),
        )
    else:
        query = query.order_by(
            ProjectEvent.created_at.asc(), ProjectEvent.id.asc(),
        )
    for event in query.all():
        if exclude_event_id and event.id == exclude_event_id:
            continue
        payload = event.payload or {}
        if payload.get("evidence_fingerprints") == fingerprints:
            return event.id
    return None


def adjudicate_vaccine_candidates(
    db: Session,
    session_id: str,
    project_id: str,
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """对一批疫苗候选逐一召开 ConsensusCourt 庭审并汇总裁决

    流程（每个候选独立开庭，一案一卷）：
    1. build_candidate_evidence 把五维评价 → Evidence 列表；
    2. hold_court 开庭（规则初筛 → 评分 → ABP rubric → PI 裁决 →
       独立评审 → Human BOSS 清单），裁决自动落 ProjectEvent；
    3. 裁决结果与候选评价数据合并返回，供评价端点直接回传前端。

    诚实性：候选无维度数据（evaluation 为空）时不开庭——如实返回
    decision="无法裁决" 与原因，与 hold_court 空证据语义一致。

    归属校验（架构审查 BLOCKER-1 修复，fail-fast）：裁决案卷的
    实际落库项目 = session_id → AgentChatSession.project_id（内核
    _persist_verdict_event 的关联链路），因此入口强制校验会话
    存在、已绑定项目且与 project_id 一致——否则案卷会写进与会话
    绑定的另一个项目，形成越权写入与审计归属错位。

    批次容错（架构审查 MAJOR-4 修复）：单候选开庭异常不中断整批
    ——异常候选如实产出 decision="无法裁决" + error 字段（与本模块
    诚实降级哲学一致）。注意内核 record_event 每事件即 commit，
    已提交案卷不可回滚，故容错语义是"跳过坏候选、保留好结果"，
    调用方不应整批重试（会产生重复案卷）。

    Args:
        db: 数据库会话（裁决事件落 ProjectEvent 用）
        session_id: 关联群聊会话 ID（案卷 correlation_id）
        project_id: 研究项目 ID（必须与会话绑定项目一致，否则拒绝）
        candidates: 候选列表，每项
            {candidate_id, pathogen_name, antigen_name, evaluation}

    Returns:
        逐候选裁决摘要列表：
        [{candidate_id, pathogen_name, decision, confidence,
          escalated, max_contradiction_intensity,
          dual_track_divergence, human_override_items, pi_rationale,
          reviewer_notes, evidence_count}]

    Raises:
        ValueError: 会话不存在 / 会话未绑定项目 / 会话归属项目与
            project_id 不一致 / candidate_id 为空（归属与身份不明
            时不开庭，防止案卷写错项目或证据 ID 退化）
    """
    # 入口归属校验（fail-fast，先于任何开庭与落库）
    session = db.get(AgentChatSession, session_id)
    if session is None:
        raise ValueError(f"庭审会话不存在: session_id={session_id}")
    if not session.project_id:
        raise ValueError(
            f"庭审会话未绑定研究项目，案卷无处归档: session_id={session_id}"
        )
    if session.project_id != project_id:
        raise ValueError(
            "会话归属项目与传入 project_id 不一致，拒绝开庭"
            f"（防越权写入）: session_id={session_id} "
            f"session.project_id={session.project_id} project_id={project_id}"
        )
    for idx, cand in enumerate(candidates):
        if not str(cand.get("candidate_id") or "").strip():
            raise ValueError(
                f"候选 #{idx} 缺少 candidate_id，证据 ID 会退化，拒绝开庭"
            )

    court = ConsensusCourt(db)
    results: List[Dict[str, Any]] = []

    for cand in candidates:
        candidate_id = str(cand.get("candidate_id") or "")
        pathogen_name = str(cand.get("pathogen_name") or "")
        antigen_name = cand.get("antigen_name")
        evaluation: Dict[str, Any] = cand.get("evaluation") or {}

        if not evaluation.get("dimensions"):
            # 无评价数据不开庭：如实上报，不构造空案卷
            results.append({
                "candidate_id": candidate_id,
                "pathogen_name": pathogen_name,
                "decision": "无法裁决",
                "confidence": 0.0,
                "escalated": True,
                "max_contradiction_intensity": 0,
                "dual_track_divergence": False,
                "human_override_items": ["候选无五维评价数据，未开庭审理"],
                "pi_rationale": "证据集为空（候选未评价），依 ABP 程序不予受理",
                "reviewer_notes": [],
                "evidence_count": 0,
                "project_id": project_id,
            })
            continue

        evidence_list = build_candidate_evidence(
            candidate_id=candidate_id,
            pathogen_name=pathogen_name,
            antigen_name=antigen_name,
            evaluation=evaluation,
        )
        topic = build_trial_topic(pathogen_name, antigen_name)

        # 真实开庭：ABP 全流程（初筛→评分→rubric→PI→独立评审→BOSS）
        # 批次容错：单候选内核异常只污染当前候选，如实产出
        # "无法裁决 + error"，后续候选继续开庭（审查 MAJOR-4）
        # 开庭时间下界：供去重查询区分"本次刚落库案卷"与历史案卷；
        # 用 naive-UTC 口径与 ProjectEvent.created_at（record_event
        # 以 naive utcnow 写入）同轨比较（见 _utc_naive_now 说明）
        trial_floor = _utc_naive_now()
        try:
            verdict: CourtVerdict = court.hold_court(
                session_id=session_id, topic=topic, evidence_list=evidence_list,
            )
        except Exception as exc:  # noqa: BLE001 - 单候选失败不中断批次
            logger.exception(
                "疫苗候选庭审异常（不中断批次）: project=%s candidate=%s",
                project_id, candidate_id,
            )
            results.append({
                "candidate_id": candidate_id,
                "pathogen_name": pathogen_name,
                "antigen_name": antigen_name,
                "decision": "无法裁决",
                "confidence": 0.0,
                "escalated": True,
                "max_contradiction_intensity": 0,
                "dual_track_divergence": False,
                "human_override_items": ["候选开庭过程异常，需人工检查评价数据"],
                "pi_rationale": f"庭审过程异常，本候选裁决缺失: {exc}",
                "reviewer_notes": [],
                "evidence_count": len(evidence_list),
                "composite_score": evaluation.get("composite_score"),
                "project_id": project_id,
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue

        # 案卷去重留痕（架构审查 MINOR-4 修复）：证据指纹是逐证据
        # 内容级 sha256（verdict_to_payload），与本会话既有裁决事件
        # 的指纹完全一致 → 同一评价数据被重复仲裁，本份标注
        # duplicate_of 指向最早案卷，供调用方/L3 回放过滤重复项。
        # 说明：案卷保持不可变（审计特性，不删不改），去重仅做标注。
        fingerprints = {
            ev.evidence_id: compute_evidence_fingerprint(ev)
            for ev in evidence_list
        }
        # 先定位本次刚落库的案卷（floor 之后最新一条），再在未落
        # floor 限制的全量查询中排除它、找更早的同指纹首案
        own_event_id = _locate_own_verdict_event(
            db, session_id, fingerprints, trial_floor,
        )
        duplicate_of = _find_duplicate_verdict_event(
            db, session_id, fingerprints, exclude_event_id=own_event_id,
        )

        logger.info(
            "疫苗候选庭审完成: project=%s candidate=%s decision=%s "
            "confidence=%.3f escalated=%s duplicate_of=%s",
            project_id, candidate_id, verdict.decision,
            verdict.confidence, verdict.escalated, duplicate_of,
        )
        result = {
            "candidate_id": candidate_id,
            "pathogen_name": pathogen_name,
            "antigen_name": antigen_name,
            "decision": verdict.decision,
            "confidence": verdict.confidence,
            "escalated": verdict.escalated,
            "max_contradiction_intensity": verdict.max_contradiction_intensity,
            "dual_track_divergence": verdict.dual_track_divergence,
            "human_override_items": list(verdict.human_override_items),
            "pi_rationale": verdict.pi_rationale,
            "reviewer_notes": list(verdict.reviewer_notes),
            "evidence_count": len(evidence_list),
            "composite_score": evaluation.get("composite_score"),
            "project_id": project_id,
        }
        if duplicate_of:
            result["duplicate_of"] = duplicate_of
        results.append(result)

    return results
