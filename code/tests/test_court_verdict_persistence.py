"""
P0 可审计性改造：CourtVerdict 落库测试

覆盖目标（design/paper-consensus-court-action-report.md §下周队列 第 1 项）：
1. verdict_to_payload 序列化完整性：证据评分 / PI 理由 / 评审异议 /
   升级项 / 证据指纹 全字段落库 schema；
2. hold_court 裁决后自动写入 ProjectEvent（event_type=CONSENSUS_COURT_VERDICT），
   correlation_id=会话 ID，供 L3 历史争议按会话回放"辩论→裁决"全案卷；
3. 关联链路正确：ProjectEvent.project_id 取自会话所属项目；
4. 降级语义：会话无关联项目（前端助手会话）→ 不落库、裁决照常返回；
   落库异常 → 不阻断裁决返回（审计尽力而为）。

测试策略：SQLite 内存库 + 真实 ORM 全链路（conftest 共享 session），
零 mock 数据库交互；仅对"无项目会话 / 落库异常"两个降级分支使用
最小 stub（替换 db 层对象）。
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime

import pytest
from sqlalchemy import select

from app.core.virtual_lab.consensus_court import (
    COURT_VERDICT_EVENT_TYPE,
    ConsensusCourt,
    CourtVerdict,
    Evidence,
    EvidenceScore,
    verdict_to_payload,
)
from app.core.virtual_lab.models import (
    AgentChatMessage,
    AgentChatSession,
    AgentExecution,
    ProjectEvent,
    ResearchProject,
)


@pytest.fixture
def db(thread_safe_session_factory):
    """提供独立的数据库会话（函数级隔离）"""
    session = thread_safe_session_factory()
    try:
        yield session
    finally:
        session.close()


def _make_evidence(evidence_id: str, content: str = "某条证据内容") -> Evidence:
    """
    构造一条最小证据条目。

    使用有效角色（drug-researcher）与有效来源（pubmed），
    使 PI 仲裁产生有意义的裁决而非"无法裁决"。
    """
    return Evidence(
        evidence_id=evidence_id,
        source="pubmed",
        content=content,
        role="drug-researcher",
        timestamp=datetime.utcnow(),
    )


def _create_session_with_project(db, project_id: str, topic: str) -> str:
    """
    创建带项目关联的群聊会话（虚拟实验室场景）。

    Args:
        db: 数据库会话
        project_id: 所属项目 ID
        topic: 讨论主题

    Returns:
        新会话 ID
    """
    session = AgentChatSession(
        owner_user_id="audit-tester",
        project_id=project_id,
        topic=topic,
        mode="decision",
        title=f"audit-{topic[:20]}",
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session.id


def _add_conflicting_message(db, session_id: str, sender_role: str, content: str) -> None:
    """向会话追加一条辩论消息（供证据提取使用）。"""
    msg = AgentChatMessage(
        session_id=session_id,
        sender_role=sender_role,
        sender_name=f"{sender_role}-agent",
        role=sender_role,
        message_type="opinion",
        round=1,
        content=content,
    )
    db.add(msg)
    db.commit()


@pytest.fixture
def project(db):
    """构造最小可用研究项目（复用虚拟实验室真实构造方法）。"""
    from uuid import uuid4

    from app.core.virtual_lab.project_manager import ProjectManager
    from app.models import User

    user = User(
        id=str(uuid4()),
        email="court-audit@test.local",
        name="court-auditor",
        password_hash="x",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    manager = ProjectManager(db)
    project = manager.create_project(
        user_id=user.id,
        title="audit-persist-project",
        goal="验证 CourtVerdict 落库链路",
        drug_type="small_molecule",
    )
    return project


class TestVerdictToPayload:
    """verdict_to_payload 序列化 schema 测试"""

    def test_payload_contains_full_audit_trail(self):
        """裁决 payload 必须包含完整审计字段（论文可审计性主张的数据基础）"""
        evidence_list = [_make_evidence("ev-1"), _make_evidence("ev-2")]

        scores = [
            EvidenceScore(
                evidence_id="ev-1",
                source_score=0.9,
                recency_score=0.8,
                relevance_score=0.7,
                overall_score=0.8,
                issues=[],
            ),
            EvidenceScore(
                evidence_id="ev-2",
                source_score=0.5,
                recency_score=0.6,
                relevance_score=0.4,
                overall_score=0.5,
                issues=["无引用"],
            ),
        ]
        verdict = CourtVerdict(
            verdict="裁决文本",
            decision="AGREE",
            confidence=0.9,
            evidence_scores=scores,
            pi_rationale="证据一致",
            reviewer_notes=["评审认为 A 方证据引用完整"],
            human_override_items=[],
            escalated=False,
        )

        payload = verdict_to_payload(verdict, "EGFR 抑制策略", evidence_list)

        # 审计五要素必须齐全
        assert payload["topic"] == "EGFR 抑制策略"
        assert payload["decision"] == "AGREE"
        assert payload["confidence"] == 0.9
        assert payload["pi_rationale"] == "证据一致"
        assert payload["reviewer_notes"] == ["评审认为 A 方证据引用完整"]
        assert payload["human_override_items"] == []
        assert payload["escalated"] is False
        assert len(payload["evidence_scores"]) == 2
        assert payload["evidence_scores"][1]["issues"] == ["无引用"]
        # 证据指纹 = sha256(source|content|role|refs)，可离线校验案卷完整性
        assert set(payload["evidence_fingerprints"].keys()) == {"ev-1", "ev-2"}
        expected_fp = hashlib.sha256(
            json.dumps(
                {
                    "source": "pubmed",
                    "content": "某条证据内容",
                    "role": "drug-researcher",
                    "reference_ids": [],
                },
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        assert payload["evidence_fingerprints"]["ev-1"] == expected_fp
        # payload 必须 JSON 可序列化（JSON 列落库前提）
        json.dumps(payload, ensure_ascii=False)


class TestVerdictPersistence:
    """hold_court 裁决落库链路测试"""

    def test_hold_court_persists_verdict_event(self, db, project):
        """裁决写入 ProjectEvent，关联链路与 payload 完整"""
        session_id = _create_session_with_project(db, project.id, "持久化验证")

        court = ConsensusCourt(db)
        evidence_list = [_make_evidence("ev-1")]
        verdict = court.hold_court(session_id, "持久化验证", evidence_list)

        # 裁决本身不受影响
        assert verdict.decision in {"支持提案", "有条件支持", "驳回提案", "无法裁决"}

        # 事件已落库
        event = db.execute(
            select(ProjectEvent).where(
                ProjectEvent.event_type == COURT_VERDICT_EVENT_TYPE,
                ProjectEvent.project_id == project.id,
            )
        ).scalar_one()
        # 关联链路：correlation_id = 会话 ID（L3 回放按会话聚合）
        assert event.correlation_id == session_id
        assert event.actor_type == "agent"
        assert event.actor_id == "consensus-court"
        assert event.payload_schema_version == "1.0"
        # payload 完整
        p = event.payload
        assert p["topic"] == "持久化验证"
        assert p["decision"] == verdict.decision
        assert p["session_id"] == session_id
        assert p["evidence_fingerprints"]
        assert p["pi_rationale"]

    def test_no_project_session_skips_persistence(self, db, project):
        """前端助手会话（无 project_id）→ 不落库，裁决照常返回"""
        # 前端智能助手会话：project_id 为 None
        session = AgentChatSession(
            owner_user_id="front-user",
            topic="前端会话",
            title="no-project",
        )
        db.add(session)
        db.commit()
        db.refresh(session)

        court = ConsensusCourt(db)
        verdict = court.hold_court(session.id, "无项目会话", [_make_evidence("ev-1")])
        assert verdict.decision in {"支持提案", "有条件支持", "驳回提案", "无法裁决"}
        # 该项目下不应产生任何法庭裁决事件
        count = db.execute(
            select(ProjectEvent).where(
                ProjectEvent.event_type == COURT_VERDICT_EVENT_TYPE,
                ProjectEvent.project_id == project.id,
            )
        ).scalars().all()
        assert count == []

    def test_persistence_failure_does_not_block_verdict(self, db, project, monkeypatch):
        """落库异常（如 DB 故障）→ 不阻断裁决返回"""
        session_id = _create_session_with_project(db, project.id, "故障降级")

        court = ConsensusCourt(db)

        # 模拟落库失败：替换 court.db 为会抛异常的 stub（仅 _persist_verdict_event 用到）
        class _BrokenDB:
            """模拟 DB 故障的最小 stub：任何方法调用都抛异常"""

            def __getattr__(self, name):
                def _raise(*a, **k):
                    raise RuntimeError("simulated db failure")
                return _raise

        monkeypatch.setattr(court, "db", _BrokenDB())
        # 裁决必须正常返回（审计尽力而为）
        verdict = court.hold_court(session_id, "故障降级", [_make_evidence("ev-1")])
        assert verdict.decision in {"支持提案", "有条件支持", "驳回提案", "无法裁决"}
        # 该会话未产生事件
        assert db.execute(
            select(ProjectEvent).where(
                ProjectEvent.event_type == COURT_VERDICT_EVENT_TYPE,
                ProjectEvent.correlation_id == session_id,
            )
        ).scalars().all() == []

    def test_full_case_replay_by_correlation_id(self, db, project):
        """L3 回放场景：按 correlation_id（会话 ID）聚合辩论消息 + 裁决事件"""
        session_id = _create_session_with_project(db, project.id, "回放验证")

        # 模拟辩论：两条消息
        _add_conflicting_message(db, session_id, "researcher-a", "观点 A")
        _add_conflicting_message(db, session_id, "researcher-b", "观点 B")

        court = ConsensusCourt(db)
        evidence_list = court.evidence_from_chat_messages(
            db.execute(
                select(AgentChatMessage).where(AgentChatMessage.session_id == session_id)
            ).scalars().all()
        )
        court.hold_court(session_id, "回放验证", evidence_list)

        # 回放断言：同一 correlation_id 下辩论消息与裁决事件可完整关联
        messages = db.execute(
            select(AgentChatMessage).where(AgentChatMessage.session_id == session_id)
        ).scalars().all()
        events = db.execute(
            select(ProjectEvent).where(
                ProjectEvent.event_type == COURT_VERDICT_EVENT_TYPE,
                ProjectEvent.correlation_id == session_id,
            )
        ).scalars().all()
        assert len(messages) == 2
        assert len(events) == 1
        # 裁决 payload 中的证据数量与实际辩论消息数一致
        assert len(events[0].payload["evidence_fingerprints"]) == 2
