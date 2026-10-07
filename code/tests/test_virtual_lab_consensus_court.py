"""
ConsensusCourt 单元测试

覆盖证据评分、规则引擎初筛、PI Agent 裁决、独立评审、Human BOSS 覆盖清单。
"""
from datetime import datetime, timedelta


from app.core.virtual_lab.consensus_court import (
    ConsensusCourt,
    Evidence,
    EvidenceScorer,
    IndependentReviewer,
    PIAgentArbiter,
    RuleEngine,
)


class TestEvidenceScorer:
    """证据评分器测试。"""

    def test_pubmed_evidence_scores_high(self):
        """PubMed 来源、相关、近期的证据应得高分。"""
        scorer = EvidenceScorer()
        ev = Evidence(
            evidence_id="e1",
            source="pubmed",
            content="EGFR L858R mutation drives NSCLC resistance",
            role="drug-researcher",
            timestamp=datetime.utcnow(),
            reference_ids=["12345678"],
        )
        score = scorer.score(ev, topic="EGFR resistance")
        assert score.source_score >= 0.8
        assert score.relevance_score > 0.5
        assert score.overall_score > 0.5
        assert not score.issues

    def test_old_evidence_recency_low(self):
        """过期证据时效性分数应较低。"""
        scorer = EvidenceScorer()
        ev = Evidence(
            evidence_id="e2",
            source="pubmed",
            content="old study",
            role="drug-researcher",
            timestamp=datetime.utcnow() - timedelta(days=365 * 10),
            reference_ids=["11111111"],
        )
        score = scorer.score(ev, topic="EGFR")
        assert score.recency_score < 0.5
        assert any("时效性较低" in issue for issue in score.issues)

    def test_missing_reference_issue(self):
        """缺少引用的证据应被标记。"""
        scorer = EvidenceScorer()
        ev = Evidence(
            evidence_id="e3",
            source="agent_opinion",
            content="我认为应该这样",
            role="agent",
        )
        score = scorer.score(ev, topic="decision")
        assert any("缺少可验证引用" in issue for issue in score.issues)


class TestRuleEngine:
    """规则引擎初筛测试。"""

    def test_af2_reference_rejected(self):
        """包含 AlphaFold2 引用的证据应被初筛拒绝。"""
        engine = RuleEngine()
        ev = Evidence(
            evidence_id="e4",
            source="agent_opinion",
            content="使用 AlphaFold2 预测结构",
            role="agent",
        )
        issues = engine.screen(ev)
        assert any("结构预测工具禁令" in issue for issue in issues)

    def test_short_content_rejected(self):
        """内容过短的证据应被拒绝。"""
        engine = RuleEngine()
        ev = Evidence(
            evidence_id="e5",
            source="pubmed",
            content="ok",
            role="agent",
        )
        issues = engine.screen(ev)
        assert any("内容过短" in issue for issue in issues)

    def test_valid_evidence_passes(self):
        """合法证据应通过初筛。"""
        engine = RuleEngine()
        ev = Evidence(
            evidence_id="e6",
            source="pubmed",
            content="EGFR L858R is a common oncogenic mutation in NSCLC.",
            role="drug-researcher",
            reference_ids=["12345678"],
        )
        assert not engine.screen(ev)


class TestPIAgentArbiter:
    """PI Agent 裁决器测试。"""

    def test_high_score_supports(self):
        """高质量证据应支持提案。"""
        arbiter = PIAgentArbiter()
        ev_map = {
            "e1": Evidence("e1", "pubmed", "strong evidence related to topic", "drug-researcher"),
        }
        scores = [EvidenceScorer().score(ev_map["e1"], "topic")]
        decision, rationale, confidence = arbiter.arbitrate("topic", scores, ev_map)
        assert decision == "支持提案"
        assert confidence > 0.5

    def test_low_score_rejects(self):
        """低质量证据应驳回提案。"""
        arbiter = PIAgentArbiter()
        ev_map = {
            "e1": Evidence("e1", "agent_opinion", "weak", "agent"),
        }
        scores = [EvidenceScorer().score(ev_map["e1"], "topic")]
        decision, rationale, confidence = arbiter.arbitrate("topic", scores, ev_map)
        assert decision == "驳回提案"


class TestIndependentReviewer:
    """独立评审 Agent 测试。"""

    def test_single_role_bias_detected(self):
        """所有证据来自同一角色时应提示偏见风险。"""
        reviewer = IndependentReviewer()
        ev_map = {
            "e1": Evidence("e1", "pubmed", "evidence", "drug-researcher"),
        }
        scores = [EvidenceScorer().score(ev_map["e1"], "topic")]
        notes = reviewer.review("支持提案", scores, ev_map)
        assert any("同一角色" in note for note in notes)


class TestConsensusCourt:
    """共识法庭集成测试。"""

    def test_hold_court_with_valid_evidence(self, test_session_factory):
        """合法证据应生成裁决且无需升级。"""
        db = test_session_factory()
        try:
            court = ConsensusCourt(db)
            evidence = [
                Evidence("e1", "pubmed", "EGFR L858R drives resistance", "drug-researcher", reference_ids=["123"]),
                Evidence("e2", "chembl", "ChEMBL target data", "computational-chemist", reference_ids=["CHEMBL123"]),
            ]
            verdict = court.hold_court("session-1", "是否针对 L858R 设计抑制剂", evidence)
            assert verdict.decision in ("支持提案", "有条件支持")
            assert 0 <= verdict.confidence <= 1
            assert len(verdict.evidence_scores) == 2
        finally:
            db.close()

    def test_hold_court_escalates_when_low_confidence(self, test_session_factory):
        """低置信度裁决应触发 Human BOSS 升级。"""
        db = test_session_factory()
        try:
            court = ConsensusCourt(db)
            evidence = [
                Evidence("e1", "agent_opinion", "I am not sure", "agent"),
            ]
            verdict = court.hold_court("session-2", "争议", evidence)
            assert verdict.escalated is True
            assert len(verdict.human_override_items) > 0
        finally:
            db.close()
