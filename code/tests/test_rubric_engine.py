"""
RubricEngine 单元测试 - P0-2 非对称举证审查 rubric

覆盖三个核心组件：
1. AsymmetricRubric：非对称举证 rubric（找到矛盾才定罪）
2. EvidencePositionWeighter：按证据位置加权置信度
3. DrillBeforeDisposition：先收集所有证据再下结论

设计哲学验证：
- 追溯而非重算：基于已有证据追溯矛盾
- 非对称举证：指控方需举证矛盾，被控方默认无罪
- drill before disposition：先举证后裁决
"""
from datetime import datetime
from typing import List, Optional

import pytest

from app.core.virtual_lab.consensus_court import Evidence, EvidenceScore
from app.core.virtual_lab.rubric_engine import (
    AsymmetricRubric,
    ContradictionPair,
    DrillBeforeDisposition,
    EvidencePositionWeighter,
    RubricVerdict,
    Severity,
    create_default_rubric_engine,
)


# ============================================================================
# 辅助函数：构造测试用证据与评分
# ============================================================================

def _make_evidence(
    evidence_id: str,
    content: str,
    source: str = "pubmed",
    role: str = "drug-researcher",
    reference_ids: Optional[List[str]] = None,
) -> Evidence:
    """构造测试用证据条目"""
    return Evidence(
        evidence_id=evidence_id,
        source=source,
        content=content,
        role=role,
        timestamp=datetime.utcnow(),
        reference_ids=reference_ids or [],
    )


def _make_score(
    evidence_id: str,
    overall: float = 0.8,
    issues: Optional[List[str]] = None,
) -> EvidenceScore:
    """构造测试用证据评分"""
    return EvidenceScore(
        evidence_id=evidence_id,
        source_score=0.85,
        recency_score=0.9,
        relevance_score=0.8,
        overall_score=overall,
        issues=issues or [],
    )


# ============================================================================
# Severity 枚举测试
# ============================================================================

class TestSeverity:
    """严重等级枚举测试"""

    def test_severity_values(self):
        """四个严重等级值应正确"""
        assert Severity.CRITICAL.value == "CRITICAL"
        assert Severity.WARNING.value == "WARNING"
        assert Severity.INFO.value == "INFO"
        assert Severity.PENDING.value == "PENDING"

    def test_severity_is_string(self):
        """Severity 继承 str，可直接作为字符串使用"""
        assert isinstance(Severity.CRITICAL, str)
        assert Severity.CRITICAL == "CRITICAL"

    def test_severity_comparable(self):
        """Severity 枚举成员可比较"""
        assert Severity.CRITICAL != Severity.WARNING
        assert Severity.PENDING != Severity.CRITICAL

    def test_severity_from_value(self):
        """可通过字符串值构造 Severity"""
        assert Severity("CRITICAL") is Severity.CRITICAL
        assert Severity("PENDING") is Severity.PENDING


# ============================================================================
# ContradictionPair 数据类测试
# ============================================================================

class TestContradictionPair:
    """矛盾证据对数据类测试"""

    def test_required_fields(self):
        """必填字段应正确赋值"""
        pair = ContradictionPair(
            evidence_a_id="ev-1",
            evidence_b_id="ev-2",
            topic="药物有效性",
            contradiction_desc="证据 1 称有效，证据 2 称无效",
        )
        assert pair.evidence_a_id == "ev-1"
        assert pair.evidence_b_id == "ev-2"
        assert pair.topic == "药物有效性"
        assert pair.contradiction_desc == "证据 1 称有效，证据 2 称无效"

    def test_default_severity_is_critical(self):
        """默认 severity 应为 CRITICAL（确认的矛盾必须拒绝）"""
        pair = ContradictionPair("ev-1", "ev-2", "topic", "desc")
        assert pair.severity is Severity.CRITICAL

    def test_custom_severity(self):
        """可指定自定义 severity"""
        pair = ContradictionPair(
            "ev-1", "ev-2", "topic", "desc",
            severity=Severity.WARNING,
        )
        assert pair.severity is Severity.WARNING


# ============================================================================
# RubricVerdict 数据类测试
# ============================================================================

class TestRubricVerdict:
    """非对称举证裁决结果数据类测试"""

    def test_required_passed_field(self):
        """passed 是必填字段"""
        verdict = RubricVerdict(passed=True)
        assert verdict.passed is True

    def test_default_values(self):
        """默认值应正确（空列表、零置信度、空理由）"""
        verdict = RubricVerdict(passed=True)
        assert verdict.contradictions == []
        assert verdict.pending_issues == []
        assert verdict.confidence == 0.0
        assert verdict.rationale == ""

    def test_independent_default_lists(self):
        """默认列表应独立（不同实例不共享引用）"""
        v1 = RubricVerdict(passed=True)
        v2 = RubricVerdict(passed=True)
        v1.contradictions.append(ContradictionPair("a", "b", "t", "d"))
        v1.pending_issues.append("issue")
        # v2 的列表不应受影响
        assert v2.contradictions == []
        assert v2.pending_issues == []

    def test_full_construction(self):
        """完整构造应正确赋值"""
        pair = ContradictionPair("a", "b", "t", "d")
        verdict = RubricVerdict(
            passed=False,
            contradictions=[pair],
            pending_issues=["待补充"],
            confidence=0.85,
            rationale="找到矛盾",
        )
        assert verdict.passed is False
        assert len(verdict.contradictions) == 1
        assert verdict.pending_issues == ["待补充"]
        assert verdict.confidence == 0.85


# ============================================================================
# AsymmetricRubric 核心测试
# ============================================================================

class TestAsymmetricRubricEvaluate:
    """AsymmetricRubric.evaluate() 核心评估逻辑测试"""

    def test_no_contradiction_passes(self):
        """无矛盾时应该通过（非对称举证：默认无罪）"""
        rubric = AsymmetricRubric()
        ev_a = _make_evidence("ev-1", "该药物有效，pIC50=8.5", source="pubmed")
        ev_b = _make_evidence("ev-2", "实验成功，纯度>95%", source="experiment")
        scores = [_make_score("ev-1"), _make_score("ev-2")]
        evidence_map = {"ev-1": ev_a, "ev-2": ev_b}

        verdict = rubric.evaluate(scores, evidence_map, topic="药物活性")

        assert verdict.passed is True
        assert verdict.contradictions == []

    def test_with_contradiction_fails(self):
        """发现矛盾时应该不通过（找到矛盾才定罪）"""
        rubric = AsymmetricRubric()
        # 证据 A 说"有效"，证据 B 说"无效" → 矛盾
        ev_a = _make_evidence("ev-1", "该药物有效", source="pubmed")
        ev_b = _make_evidence("ev-2", "该药物无效", source="experiment")
        scores = [_make_score("ev-1"), _make_score("ev-2")]
        evidence_map = {"ev-1": ev_a, "ev-2": ev_b}

        verdict = rubric.evaluate(scores, evidence_map, topic="药物有效性")

        assert verdict.passed is False
        assert len(verdict.contradictions) == 1
        assert verdict.contradictions[0].evidence_a_id == "ev-1"
        assert verdict.contradictions[0].evidence_b_id == "ev-2"
        assert verdict.contradictions[0].severity == Severity.CRITICAL

    def test_empty_evidence_list_passes(self):
        """空证据列表应该通过（非对称举证：无证据不指控）"""
        rubric = AsymmetricRubric()
        verdict = rubric.evaluate([], {}, topic="空主题")
        assert verdict.passed is True
        assert verdict.contradictions == []
        assert verdict.pending_issues == []

    def test_single_evidence_no_contradiction(self):
        """单条证据无法形成矛盾对，应该通过"""
        rubric = AsymmetricRubric()
        ev = _make_evidence("ev-1", "该药物有效", source="pubmed")
        scores = [_make_score("ev-1")]
        evidence_map = {"ev-1": ev}

        verdict = rubric.evaluate(scores, evidence_map, topic="药物活性")

        assert verdict.passed is True
        assert verdict.contradictions == []

    def test_same_source_no_contradiction(self):
        """同一来源的证据不进行矛盾对比（不会自相矛盾）"""
        rubric = AsymmetricRubric()
        # 两条证据都来自 pubmed，即使内容矛盾也不检测
        ev_a = _make_evidence("ev-1", "该药物有效", source="pubmed")
        ev_b = _make_evidence("ev-2", "该药物无效", source="pubmed")
        scores = [_make_score("ev-1"), _make_score("ev-2")]
        evidence_map = {"ev-1": ev_a, "ev-2": ev_b}

        verdict = rubric.evaluate(scores, evidence_map, topic="药物活性")

        # 同源不对比，应该通过
        assert verdict.passed is True
        assert verdict.contradictions == []

    def test_confidence_high_when_contradiction_found(self):
        """发现矛盾时置信度应为 0.9（高置信度定罪）"""
        rubric = AsymmetricRubric()
        ev_a = _make_evidence("ev-1", "有效", source="pubmed")
        ev_b = _make_evidence("ev-2", "无效", source="experiment")
        scores = [_make_score("ev-1"), _make_score("ev-2")]
        evidence_map = {"ev-1": ev_a, "ev-2": ev_b}

        verdict = rubric.evaluate(scores, evidence_map, topic="topic")

        assert verdict.passed is False
        assert verdict.confidence == 0.9

    def test_rationale_contains_contradiction_count(self):
        """裁决理由应包含矛盾数量"""
        rubric = AsymmetricRubric()
        ev_a = _make_evidence("ev-1", "有效", source="pubmed")
        ev_b = _make_evidence("ev-2", "无效", source="experiment")
        scores = [_make_score("ev-1"), _make_score("ev-2")]
        evidence_map = {"ev-1": ev_a, "ev-2": ev_b}

        verdict = rubric.evaluate(scores, evidence_map, topic="topic")

        assert "1 个矛盾证据对" in verdict.rationale

    def test_rationale_contains_pending_count_when_passed(self):
        """通过时裁决理由应包含待举证问题数量"""
        rubric = AsymmetricRubric()
        ev_a = _make_evidence("ev-1", "该药物有效", source="pubmed")
        ev_b = _make_evidence("ev-2", "实验成功", source="experiment")
        scores = [_make_score("ev-1"), _make_score("ev-2")]
        evidence_map = {"ev-1": ev_a, "ev-2": ev_b}

        verdict = rubric.evaluate(scores, evidence_map, topic="topic")

        assert "默认通过" in verdict.rationale


class TestAsymmetricRubricContradictionKeywords:
    """矛盾关键词对检测测试"""

    @pytest.mark.parametrize(
        "kw_a,kw_b",
        AsymmetricRubric.CONTRADICTION_KEYWORD_PAIRS,
    )
    def test_all_keyword_pairs_detected(self, kw_a, kw_b):
        """所有预定义的矛盾关键词对应都能被检测到（双向）"""
        rubric = AsymmetricRubric()
        # 正向：A 含 kw_a，B 含 kw_b
        ev_a = _make_evidence("ev-a", f"结果显示{kw_a}", source="pubmed")
        ev_b = _make_evidence("ev-b", f"结果显示{kw_b}", source="experiment")
        scores = [_make_score("ev-a"), _make_score("ev-b")]
        evidence_map = {"ev-a": ev_a, "ev-b": ev_b}

        verdict = rubric.evaluate(scores, evidence_map, topic="topic")
        assert verdict.passed is False, f"未检测到矛盾关键词对: ({kw_a}, {kw_b})"

    @pytest.mark.parametrize(
        "kw_a,kw_b",
        AsymmetricRubric.CONTRADICTION_KEYWORD_PAIRS,
    )
    def test_keyword_pairs_bidirectional(self, kw_a, kw_b):
        """矛盾关键词对应双向检测（A 含 kw_b，B 含 kw_a 也算矛盾）"""
        rubric = AsymmetricRubric()
        # 反向：A 含 kw_b，B 含 kw_a
        ev_a = _make_evidence("ev-a", f"结果显示{kw_b}", source="pubmed")
        ev_b = _make_evidence("ev-b", f"结果显示{kw_a}", source="experiment")
        scores = [_make_score("ev-a"), _make_score("ev-b")]
        evidence_map = {"ev-a": ev_a, "ev-b": ev_b}

        verdict = rubric.evaluate(scores, evidence_map, topic="topic")
        assert verdict.passed is False, f"未检测到双向矛盾: ({kw_b}, {kw_a})"

    def test_non_contradiction_keywords_pass(self):
        """非矛盾关键词不应触发定罪"""
        rubric = AsymmetricRubric()
        ev_a = _make_evidence("ev-1", "pIC50 为 8.5", source="pubmed")
        ev_b = _make_evidence("ev-2", "溶解度良好", source="experiment")
        scores = [_make_score("ev-1"), _make_score("ev-2")]
        evidence_map = {"ev-1": ev_a, "ev-2": ev_b}

        verdict = rubric.evaluate(scores, evidence_map, topic="topic")
        assert verdict.passed is True

    def test_contradiction_desc_includes_topic(self):
        """矛盾描述应包含争议主题"""
        rubric = AsymmetricRubric()
        ev_a = _make_evidence("ev-1", "有效", source="pubmed")
        ev_b = _make_evidence("ev-2", "无效", source="experiment")
        scores = [_make_score("ev-1"), _make_score("ev-2")]
        evidence_map = {"ev-1": ev_a, "ev-2": ev_b}

        verdict = rubric.evaluate(scores, evidence_map, topic="EGFR 抑制活性")

        assert len(verdict.contradictions) == 1
        assert "EGFR 抑制活性" in verdict.contradictions[0].contradiction_desc

    def test_multiple_contradictions_count(self):
        """多对矛盾都应被检测到"""
        rubric = AsymmetricRubric()
        # 三条证据两两矛盾（有效/无效/失败 vs 成功）
        ev_a = _make_evidence("ev-1", "药物有效", source="pubmed")
        ev_b = _make_evidence("ev-2", "药物无效", source="experiment")
        ev_c = _make_evidence("ev-3", "实验失败", source="chembl")
        scores = [_make_score("ev-1"), _make_score("ev-2"), _make_score("ev-3")]
        evidence_map = {"ev-1": ev_a, "ev-2": ev_b, "ev-3": ev_c}

        verdict = rubric.evaluate(scores, evidence_map, topic="topic")

        # ev-1 vs ev-2 (有效/无效), ev-1 vs ev-3 (有效/失败 - 不在关键词对),
        # ev-2 vs ev-3 (无效/失败 - 不在关键词对)
        # 注意：有效/失败 不在预定义关键词对中，只有 有效/无效 是
        assert verdict.passed is False
        assert len(verdict.contradictions) == 1


class TestAsymmetricRubricPendingIssues:
    """AsymmetricRubric 待举证问题收集测试"""

    def test_short_content_marked_pending(self):
        """内容过短应标记为待举证（不直接拒绝）"""
        rubric = AsymmetricRubric()
        ev = _make_evidence("ev-1", "短", source="pubmed")  # 内容 < 10 字符
        scores = [_make_score("ev-1")]
        evidence_map = {"ev-1": ev}

        verdict = rubric.evaluate(scores, evidence_map, topic="topic")

        # 仍然通过（非对称举证：待举证不等于拒绝）
        assert verdict.passed is True
        assert any("内容过短" in issue for issue in verdict.pending_issues)

    def test_empty_source_marked_pending(self):
        """来源为空应标记为待举证"""
        rubric = AsymmetricRubric()
        ev = _make_evidence("ev-1", "该药物具有显著的抑制活性", source="")
        scores = [_make_score("ev-1")]
        evidence_map = {"ev-1": ev}

        verdict = rubric.evaluate(scores, evidence_map, topic="topic")

        assert verdict.passed is True
        assert any("来源为空" in issue for issue in verdict.pending_issues)

    def test_agent_opinion_without_refs_marked_pending(self):
        """Agent 观点缺少引用应标记为待举证"""
        rubric = AsymmetricRubric()
        ev = _make_evidence(
            "ev-1",
            "根据我的分析该药物有效",
            source="agent_opinion",
            reference_ids=[],
        )
        scores = [_make_score("ev-1")]
        evidence_map = {"ev-1": ev}

        verdict = rubric.evaluate(scores, evidence_map, topic="topic")

        assert verdict.passed is True
        assert any("Agent 观点" in issue and "缺少引用" in issue for issue in verdict.pending_issues)

    def test_score_issues_collected(self):
        """证据评分中的问题应被收集到 pending_issues"""
        rubric = AsymmetricRubric()
        ev = _make_evidence("ev-1", "该药物具有显著的抑制活性", source="pubmed")
        scores = [_make_score("ev-1", issues=["来源可信度低", "时效性差"])]
        evidence_map = {"ev-1": ev}

        verdict = rubric.evaluate(scores, evidence_map, topic="topic")

        assert any("来源可信度低" in issue for issue in verdict.pending_issues)
        assert any("时效性差" in issue for issue in verdict.pending_issues)

    def test_missing_evidence_id_skipped(self):
        """评分中引用了不存在的 evidence_id 时应跳过（不报错）"""
        rubric = AsymmetricRubric()
        ev = _make_evidence("ev-1", "该药物有效", source="pubmed")
        scores = [_make_score("ev-1"), _make_score("ev-missing")]
        evidence_map = {"ev-1": ev}  # 缺少 ev-missing

        # 不应抛出异常
        verdict = rubric.evaluate(scores, evidence_map, topic="topic")
        assert verdict.passed is True


class TestAsymmetricRubricConfidence:
    """AsymmetricRubric 置信度计算测试"""

    def test_no_evidence_low_confidence(self):
        """无证据时置信度应为 0.3（低置信度）"""
        rubric = AsymmetricRubric()
        verdict = rubric.evaluate([], {}, topic="topic")
        assert verdict.confidence == 0.3

    def test_single_evidence_confidence(self):
        """单条证据置信度应基于评分 + 少量 bonus"""
        rubric = AsymmetricRubric()
        ev = _make_evidence("ev-1", "该药物有效", source="pubmed")
        scores = [_make_score("ev-1", overall=0.8)]
        evidence_map = {"ev-1": ev}

        verdict = rubric.evaluate(scores, evidence_map, topic="topic")

        # 0.8 + 1*0.02 = 0.82
        assert verdict.confidence == 0.82

    def test_multiple_evidence_confidence_bonus(self):
        """多证据应有更高 bonus（但不超过 0.9）"""
        rubric = AsymmetricRubric()
        ev_a = _make_evidence("ev-1", "该药物有效", source="pubmed")
        ev_b = _make_evidence("ev-2", "溶解度良好", source="experiment")
        scores = [_make_score("ev-1", overall=0.7), _make_score("ev-2", overall=0.7)]
        evidence_map = {"ev-1": ev_a, "ev-2": ev_b}

        verdict = rubric.evaluate(scores, evidence_map, topic="topic")

        # 平均 0.7 + 2*0.02 = 0.74
        assert verdict.confidence == 0.74

    def test_confidence_capped_at_0_9(self):
        """置信度上限应为 0.9"""
        rubric = AsymmetricRubric()
        ev_a = _make_evidence("ev-1", "该药物有效", source="pubmed")
        ev_b = _make_evidence("ev-2", "溶解度良好", source="experiment")
        # 高评分 + 多证据 → 应被截断到 0.9
        scores = [_make_score("ev-1", overall=0.95), _make_score("ev-2", overall=0.95)]
        evidence_map = {"ev-1": ev_a, "ev-2": ev_b}

        verdict = rubric.evaluate(scores, evidence_map, topic="topic")

        # 0.95 + 0.04 = 0.99 → 截断到 0.9
        assert verdict.confidence == 0.9


# ============================================================================
# EvidencePositionWeighter 测试
# ============================================================================

class TestEvidencePositionWeighter:
    """证据位置加权器测试"""

    def test_opening_position_weight(self):
        """开头位置（0.0-0.33）权重应为 0.8"""
        weighter = EvidencePositionWeighter()
        assert weighter.weight(0.0) == 0.8
        assert weighter.weight(0.1) == 0.8
        assert weighter.weight(0.32) == 0.8

    def test_middle_position_weight(self):
        """中间位置（0.33-0.67）权重应为 1.0（最可靠）"""
        weighter = EvidencePositionWeighter()
        assert weighter.weight(0.33) == 1.0
        assert weighter.weight(0.5) == 1.0
        assert weighter.weight(0.66) == 1.0

    def test_closing_position_weight(self):
        """结尾位置（0.67-1.0）权重应为 0.9"""
        weighter = EvidencePositionWeighter()
        assert weighter.weight(0.67) == 0.9
        assert weighter.weight(0.8) == 0.9
        assert weighter.weight(1.0) == 0.9

    def test_boundary_0_33_is_middle(self):
        """边界值 0.33 应属于中间区间（< 0.33 是 opening）"""
        weighter = EvidencePositionWeighter()
        # 0.33 不 < 0.33，所以落入 middle
        assert weighter.weight(0.33) == 1.0
        # 0.32 < 0.33，落入 opening
        assert weighter.weight(0.32) == 0.8

    def test_boundary_0_67_is_closing(self):
        """边界值 0.67 应属于结尾区间"""
        weighter = EvidencePositionWeighter()
        # 0.67 不 < 0.67，所以落入 closing
        assert weighter.weight(0.67) == 0.9
        # 0.66 < 0.67，落入 middle
        assert weighter.weight(0.66) == 1.0

    def test_out_of_range_returns_default(self):
        """超出 [0, 1] 范围应返回默认权重 1.0"""
        weighter = EvidencePositionWeighter()
        assert weighter.weight(-0.1) == 1.0
        assert weighter.weight(1.5) == 1.0

    def test_weight_by_section_known(self):
        """已知段落名称应返回对应权重"""
        weighter = EvidencePositionWeighter()
        assert weighter.weight_by_section("opening") == 0.8
        assert weighter.weight_by_section("middle") == 1.0
        assert weighter.weight_by_section("closing") == 0.9

    def test_weight_by_section_unknown(self):
        """未知段落名称应返回默认权重 1.0"""
        weighter = EvidencePositionWeighter()
        assert weighter.weight_by_section("unknown") == 1.0
        assert weighter.weight_by_section("") == 1.0

    def test_adjust_confidence_opening(self):
        """开头位置应降低置信度（×0.8）"""
        weighter = EvidencePositionWeighter()
        adjusted = weighter.adjust_confidence(0.9, 0.1)
        assert adjusted == 0.72  # 0.9 * 0.8

    def test_adjust_confidence_middle(self):
        """中间位置应保持置信度（×1.0）"""
        weighter = EvidencePositionWeighter()
        adjusted = weighter.adjust_confidence(0.85, 0.5)
        assert adjusted == 0.85  # 0.85 * 1.0

    def test_adjust_confidence_closing(self):
        """结尾位置应略微降低置信度（×0.9）"""
        weighter = EvidencePositionWeighter()
        adjusted = weighter.adjust_confidence(0.9, 0.8)
        assert adjusted == 0.81  # 0.9 * 0.9

    def test_adjust_confidence_capped_at_1(self):
        """调整后置信度不应超过 1.0"""
        weighter = EvidencePositionWeighter()
        adjusted = weighter.adjust_confidence(1.5, 0.5)
        assert adjusted == 1.0

    def test_adjust_confidence_floored_at_0(self):
        """调整后置信度不应低于 0.0"""
        weighter = EvidencePositionWeighter()
        adjusted = weighter.adjust_confidence(-0.5, 0.5)
        assert adjusted == 0.0

    def test_adjust_confidence_rounded(self):
        """调整后置信度应四舍五入到 3 位小数"""
        weighter = EvidencePositionWeighter()
        adjusted = weighter.adjust_confidence(0.856, 0.1)
        # 0.856 * 0.8 = 0.6848 → 0.685
        assert adjusted == 0.685


# ============================================================================
# DrillBeforeDisposition 测试
# ============================================================================

class TestDrillBeforeDisposition:
    """先举证后裁决测试"""

    def test_initial_state(self):
        """初始状态：无证据，未完成"""
        drill = DrillBeforeDisposition()
        assert drill.evidence_count == 0
        assert drill.is_complete is False

    def test_collect_single_evidence(self):
        """collect() 应能收集单条证据"""
        drill = DrillBeforeDisposition()
        ev = _make_evidence("ev-1", "该药物有效", source="pubmed")
        drill.collect(ev)
        assert drill.evidence_count == 1
        assert drill.is_complete is False

    def test_collect_with_score(self):
        """collect() 应能同时收集证据和评分"""
        drill = DrillBeforeDisposition()
        ev = _make_evidence("ev-1", "该药物有效", source="pubmed")
        score = _make_score("ev-1")
        drill.collect(ev, score)
        assert drill.evidence_count == 1

    def test_collect_multiple_evidence(self):
        """collect() 应能收集多条证据"""
        drill = DrillBeforeDisposition()
        drill.collect(_make_evidence("ev-1", "有效", source="pubmed"))
        drill.collect(_make_evidence("ev-2", "活性高", source="experiment"))
        assert drill.evidence_count == 2

    def test_collect_after_complete_reopens(self):
        """收集完成后再次 collect 应重新开放收集"""
        drill = DrillBeforeDisposition()
        drill.collect(_make_evidence("ev-1", "有效", source="pubmed"))
        drill.complete_collection()
        assert drill.is_complete is True

        # 再次 collect 应重新开放
        drill.collect(_make_evidence("ev-2", "活性高", source="experiment"))
        assert drill.is_complete is False
        assert drill.evidence_count == 2

    def test_collect_batch(self):
        """collect_batch() 应能批量收集证据"""
        drill = DrillBeforeDisposition()
        ev_list = [
            _make_evidence("ev-1", "有效", source="pubmed"),
            _make_evidence("ev-2", "活性高", source="experiment"),
            _make_evidence("ev-3", "纯度好", source="chembl"),
        ]
        scores = [_make_score("ev-1"), _make_score("ev-2"), _make_score("ev-3")]
        drill.collect_batch(ev_list, scores)
        assert drill.evidence_count == 3

    def test_collect_batch_without_scores(self):
        """collect_batch() 不传 scores 时只收集证据"""
        drill = DrillBeforeDisposition()
        ev_list = [
            _make_evidence("ev-1", "有效", source="pubmed"),
            _make_evidence("ev-2", "活性高", source="experiment"),
        ]
        drill.collect_batch(ev_list)
        assert drill.evidence_count == 2

    def test_collect_batch_mismatched_lengths(self):
        """collect_batch() 长度不匹配时应警告但不报错"""
        drill = DrillBeforeDisposition()
        ev_list = [
            _make_evidence("ev-1", "有效", source="pubmed"),
            _make_evidence("ev-2", "活性高", source="experiment"),
        ]
        scores = [_make_score("ev-1")]  # 只有 1 个评分，但有 2 条证据
        drill.collect_batch(ev_list, scores)
        # 应该收集 2 条证据（第一条带评分，第二条不带）
        assert drill.evidence_count == 2

    def test_complete_collection(self):
        """complete_collection() 应标记收集完成"""
        drill = DrillBeforeDisposition()
        drill.collect(_make_evidence("ev-1", "有效", source="pubmed"))
        drill.complete_collection()
        assert drill.is_complete is True

    def test_disposition_before_complete_raises(self):
        """未完成收集时调用 disposition() 应抛出 RuntimeError"""
        drill = DrillBeforeDisposition()
        drill.collect(_make_evidence("ev-1", "有效", source="pubmed"))
        with pytest.raises(RuntimeError, match="收集未完成"):
            drill.disposition(topic="topic")

    def test_disposition_after_complete_returns_verdict(self):
        """完成收集后调用 disposition() 应返回裁决结果"""
        drill = DrillBeforeDisposition()
        drill.collect(_make_evidence("ev-1", "有效", source="pubmed"))
        drill.complete_collection()
        verdict = drill.disposition(topic="药物活性")
        assert isinstance(verdict, RubricVerdict)
        assert verdict.passed is True  # 无矛盾

    def test_disposition_detects_contradiction(self):
        """disposition() 应能检测收集到的矛盾证据"""
        drill = DrillBeforeDisposition()
        drill.collect(_make_evidence("ev-1", "药物有效", source="pubmed"))
        drill.collect(_make_evidence("ev-2", "药物无效", source="experiment"))
        drill.complete_collection()
        verdict = drill.disposition(topic="药物活性")
        assert verdict.passed is False
        assert len(verdict.contradictions) == 1

    def test_disposition_auto_scores_when_missing(self):
        """disposition() 无评分时应自动调用 EvidenceScorer 评分"""
        drill = DrillBeforeDisposition()
        # collect 时不传 score
        drill.collect(_make_evidence("ev-1", "该药物具有显著的抑制活性", source="pubmed"))
        drill.complete_collection()
        # 应该自动评分并返回结果（不报错）
        verdict = drill.disposition(topic="抑制活性")
        assert isinstance(verdict, RubricVerdict)
        assert verdict.passed is True

    def test_disposition_with_custom_rubric(self):
        """disposition() 应支持传入自定义 rubric"""
        drill = DrillBeforeDisposition()
        drill.collect(_make_evidence("ev-1", "有效", source="pubmed"))
        drill.complete_collection()
        custom_rubric = AsymmetricRubric()
        verdict = drill.disposition(topic="topic", rubric=custom_rubric)
        assert isinstance(verdict, RubricVerdict)

    def test_reset(self):
        """reset() 应清空所有状态"""
        drill = DrillBeforeDisposition()
        drill.collect(_make_evidence("ev-1", "有效", source="pubmed"))
        drill.complete_collection()
        drill.reset()
        assert drill.evidence_count == 0
        assert drill.is_complete is False

    def test_full_workflow_collect_complete_disposition(self):
        """完整流程：collect → complete → disposition → reset"""
        drill = DrillBeforeDisposition()

        # 1. 收集
        drill.collect(_make_evidence("ev-1", "药物有效", source="pubmed"))
        drill.collect(_make_evidence("ev-2", "实验成功", source="experiment"))
        assert drill.evidence_count == 2
        assert not drill.is_complete

        # 2. 完成收集
        drill.complete_collection()
        assert drill.is_complete

        # 3. 下结论
        verdict = drill.disposition(topic="药物活性")
        assert verdict.passed is True  # 无矛盾

        # 4. 重置
        drill.reset()
        assert drill.evidence_count == 0
        assert not drill.is_complete


# ============================================================================
# 便捷函数测试
# ============================================================================

class TestCreateDefaultRubricEngine:
    """create_default_rubric_engine() 便捷函数测试"""

    def test_returns_three_components(self):
        """应返回三个组件的元组"""
        rubric, weighter, drill = create_default_rubric_engine()
        assert isinstance(rubric, AsymmetricRubric)
        assert isinstance(weighter, EvidencePositionWeighter)
        assert isinstance(drill, DrillBeforeDisposition)

    def test_components_are_independent(self):
        """每次调用应返回新的独立实例"""
        r1, w1, d1 = create_default_rubric_engine()
        r2, w2, d2 = create_default_rubric_engine()
        assert r1 is not r2
        assert w1 is not w2
        assert d1 is not d2


# ============================================================================
# 端到端集成测试
# ============================================================================

class TestEndToEndIntegration:
    """端到端集成测试：完整审查流程"""

    def test_clean_evidence_passes_with_high_confidence(self):
        """干净证据集应通过审查并具有较高置信度"""
        rubric = AsymmetricRubric()
        # 三条互相不矛盾的证据
        ev_a = _make_evidence("ev-1", "该药物 pIC50=8.5", source="pubmed")
        ev_b = _make_evidence("ev-2", "溶解度 50mg/mL", source="experiment")
        ev_c = _make_evidence("ev-3", "小鼠实验显示良好耐受", source="pubmed")
        scores = [_make_score("ev-1", 0.9), _make_score("ev-2", 0.85), _make_score("ev-3", 0.8)]
        evidence_map = {"ev-1": ev_a, "ev-2": ev_b, "ev-3": ev_c}

        verdict = rubric.evaluate(scores, evidence_map, topic="药物成药性")

        assert verdict.passed is True
        assert verdict.contradictions == []
        # 平均 0.85 + 3*0.02=0.06 = 0.91 → 截断到 0.9
        assert verdict.confidence == 0.9

    def test_contradictory_evidence_fails_with_critical(self):
        """矛盾证据集应不通过并标记 CRITICAL"""
        rubric = AsymmetricRubric()
        ev_a = _make_evidence("ev-1", "实验组阳性", source="pubmed")
        ev_b = _make_evidence("ev-2", "实验组阴性", source="experiment")
        scores = [_make_score("ev-1"), _make_score("ev-2")]
        evidence_map = {"ev-1": ev_a, "ev-2": ev_b}

        verdict = rubric.evaluate(scores, evidence_map, topic="实验结果")

        assert verdict.passed is False
        assert len(verdict.contradictions) == 1
        assert verdict.contradictions[0].severity == Severity.CRITICAL
        assert verdict.confidence == 0.9

    def test_pending_issues_do_not_block_pass(self):
        """待举证问题不应阻止通过（非对称举证核心原则）"""
        rubric = AsymmetricRubric()
        # 内容过短 + 来源为空 + Agent 观点无引用 → 全部待举证
        ev_short = _make_evidence("ev-1", "短", source="pubmed")
        ev_no_source = _make_evidence("ev-2", "该药物有效", source="")
        ev_opinion = _make_evidence(
            "ev-3", "我认为该药物有效", source="agent_opinion", reference_ids=[]
        )
        scores = [_make_score("ev-1"), _make_score("ev-2"), _make_score("ev-3")]
        evidence_map = {"ev-1": ev_short, "ev-2": ev_no_source, "ev-3": ev_opinion}

        verdict = rubric.evaluate(scores, evidence_map, topic="topic")

        # 虽然有很多问题，但没有矛盾，应该通过
        assert verdict.passed is True
        assert len(verdict.pending_issues) >= 3

    def test_position_weighter_integrates_with_rubric(self):
        """位置加权器与 rubric 集成：调整后的置信度仍合理"""
        rubric = AsymmetricRubric()
        weighter = EvidencePositionWeighter()

        ev = _make_evidence("ev-1", "该药物有效", source="pubmed")
        scores = [_make_score("ev-1", overall=0.85)]
        evidence_map = {"ev-1": ev}

        verdict = rubric.evaluate(scores, evidence_map, topic="topic")
        # 假设该证据在回答开头位置（0.1），应被降权
        adjusted = weighter.adjust_confidence(verdict.confidence, 0.1)

        assert verdict.passed is True
        assert adjusted < verdict.confidence  # 开头位置降权
        assert 0.0 <= adjusted <= 1.0

    def test_drill_workflow_with_contradiction(self):
        """drill 工作流：收集矛盾证据后下结论"""
        drill = DrillBeforeDisposition()
        # 模拟渐进式收集证据
        drill.collect(_make_evidence("ev-1", "药物A激活该通路", source="pubmed"))
        drill.collect(_make_evidence("ev-2", "药物A抑制该通路", source="experiment"))
        drill.complete_collection()

        verdict = drill.disposition(topic="信号通路调控")

        assert verdict.passed is False
        assert len(verdict.contradictions) == 1
        assert "激活" in verdict.contradictions[0].contradiction_desc
        assert "抑制" in verdict.contradictions[0].contradiction_desc
