"""
rubric_engine v0.3 单元测试 —— Aspect 约束 / ValidityGate / 强度分级

对应论文大纲 §4.4 三项 v0.3 机制与 §4.7 升级条款 5，用例编号 U1-U5：
- U1  AspectInferrer 切面推断（显式字段 / 词典 / UNKNOWN 兜底 / 通配放行）
- U2  子串极性修复（"不存在"含"存在"陷阱始终生效，不受开关影响）
- U3  ValidityGate 三态（SAME/UNKNOWN 放行、DIFFERENT 拦截、AMBIGUOUS 降级）
- U4  强度分级 _grade_intensity（极性基线 / 来源差加成 / 同对象佐证 / 截断）
- U5  三开关差分（任一关闭精确回退 v0.2 对应行为；全关 = v0.2 等价）

运行：backend$ .venv/bin/pytest tests/test_rubric_engine_v03.py -q
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

# 导入前置：backend 根入 sys.path + 配置占位（与 tests/ddb pilot 同一模式）
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))
os.environ.setdefault("SCIMINER_API_KEY", "v03-unit-placeholder")
os.environ.setdefault("DB_PASSWORD", "v03-unit-placeholder")

import pytest  # noqa: E402

from app.core.virtual_lab.consensus_court import Evidence, EvidenceScore  # noqa: E402
from app.core.virtual_lab.rubric_engine import (  # noqa: E402
    Aspect,
    AspectInferrer,
    AsymmetricRubric,
    GateDecision,
    KEYWORD_PAIR_SPECS,
    POLARITY_DEGREE,
    POLARITY_DIRECTIONAL,
    Severity,
    ValidityGate,
    _grade_intensity,
    _pair_hit_by_polarity,
    _polarity_sides,
    KeywordPairSpec,
)


# ============================================================================
# 辅助构造
# ============================================================================

def _ev(
    evidence_id: str,
    content: str,
    source: str = "pubmed",
    metadata: dict | None = None,
) -> Evidence:
    """构造测试证据条目（默认 pubmed 来源、可注入 metadata 显式字段）。"""
    return Evidence(
        evidence_id=evidence_id,
        source=source,
        content=content,
        role="drug-researcher",
        timestamp=datetime(2024, 1, 1),
        reference_ids=["12345678"],
        metadata=metadata or {},
    )


def _score(evidence_id: str, overall: float = 0.8) -> EvidenceScore:
    """构造测试评分条目（v0.3 机制不读评分数值，仅满足 evaluate 入参结构）。"""
    return EvidenceScore(
        evidence_id=evidence_id,
        source_score=0.85,
        recency_score=0.9,
        relevance_score=0.8,
        overall_score=overall,
    )


def _spec(keyword_a: str, keyword_b: str, **kw) -> KeywordPairSpec:
    """按关键词从规格表取词对，允许覆盖字段（构造 degree 极性用）。"""
    for s in KEYWORD_PAIR_SPECS:
        if s.keyword_a == keyword_a and s.keyword_b == keyword_b:
            from dataclasses import replace
            return replace(s, **kw)
    raise KeyError(f"词对不在规格表: {keyword_a}/{keyword_b}")


# ============================================================================
# U1 Aspect 约束
# ============================================================================

class TestAspectInferrer:
    """AspectInferrer 三级瀑布推断测试。"""

    def test_explicit_metadata_aspects_wins(self):
        """metadata['aspects'] 显式切面优先于词典推断（枚举名 + 中文名混用）。"""
        inf = AspectInferrer()
        ev = _ev("ev-1", "完全无术语的普通句子", metadata={
            "aspects": ["ACTIVITY_DIRECTION", "效价"],
        })
        aspects = inf.infer(ev, topic="任意主题")
        assert aspects == frozenset({Aspect.ACTIVITY_DIRECTION, Aspect.POTENCY})

    def test_explicit_metadata_singular_aspect(self):
        """单数键 metadata['aspect'] 同样被采纳（复数键优先）。"""
        inf = AspectInferrer()
        ev = _ev("ev-1", "普通句子", metadata={"aspect": "安全性"})
        assert inf.infer(ev) == frozenset({Aspect.SAFETY})

    def test_explicit_invalid_falls_back_to_lexicon(self):
        """显式字段值全部非法时跳过，回退词典推断（不抛异常）。"""
        inf = AspectInferrer()
        ev = _ev("ev-1", "EGFR 激酶活性升高", metadata={"aspects": ["不存在的切面"]})
        assert Aspect.ACTIVITY_DIRECTION in inf.infer(ev)

    def test_lexicon_hit_multiple_aspects(self):
        """词典扫描：content + topic 联合语料，多切面术语同时命中。"""
        inf = AspectInferrer()
        ev = _ev("ev-1", "肿瘤组织中 HER2 高表达", source="pubmed")
        aspects = inf.infer(ev, topic="曲妥珠单抗疗效 ORR")
        assert Aspect.EXPRESSION_REGULATION in aspects  # 高表达 / HER2
        assert Aspect.POTENCY in aspects                # 疗效 / ORR

    def test_zero_hit_returns_unknown(self):
        """词典零命中 → 兜底 {UNKNOWN}（推断器永不返回空集）。"""
        inf = AspectInferrer()
        ev = _ev("ev-1", "今天天气不错适合出门散步")
        assert inf.infer(ev, topic="") == frozenset({Aspect.UNKNOWN})

    def test_ascii_word_boundary_no_substring_hit(self):
        """ASCII 键按词边界匹配：'IC50' 不应被 'C50' 类子串误中，反之亦然。"""
        inf = AspectInferrer()
        ev = _ev("ev-1", "补体成分 C50 水平测定")
        # "IC50" 是 ACTIVITY_DIRECTION 键；文本只有 "C50"，不应命中该切面
        assert Aspect.ACTIVITY_DIRECTION not in inf.infer(ev)


class TestAspectConstraint:
    """Aspect 约束端到端行为（跨切面伪冲突拦截 + UNKNOWN 通配）。"""

    def test_cross_aspect_pseudo_contradiction_blocked(self):
        """跨切面伪冲突（成药性'合成失败' vs 机制'靶点验证成功'）被拦截。"""
        rubric = AsymmetricRubric()  # 三机制默认全开
        ev_a = _ev("ev-a", "该候选分子的合成失败，路线收率无法放大", source="pubmed")
        ev_b = _ev("ev-b", "遗传学证据充分，靶点验证成功", source="experiment")
        scores = [_score("ev-a"), _score("ev-b")]
        emap = {"ev-a": ev_a, "ev-b": ev_b}

        verdict = rubric.evaluate(scores, emap, topic="候选分子推进决策")
        # v0.2 会在此定罪（成功/失败词对命中）；v0.3 aspect 约束拦截伪冲突
        assert verdict.passed is True
        assert verdict.contradictions == []

    def test_same_aspect_contradiction_still_convicted(self):
        """同切面真矛盾（同为表达调控的'上调'vs'下调'）照常定罪。"""
        rubric = AsymmetricRubric()
        ev_a = _ev("ev-a", "给药后 TP53 靶基因表达显著上调", source="pubmed")
        ev_b = _ev("ev-b", "给药后 TP53 靶基因表达明显下调", source="experiment")
        scores = [_score("ev-a"), _score("ev-b")]
        emap = {"ev-a": ev_a, "ev-b": ev_b}

        verdict = rubric.evaluate(scores, emap, topic="p53 通路调控方向")
        assert verdict.passed is False
        pair = verdict.contradictions[0]
        assert pair.matched_aspect is Aspect.EXPRESSION_REGULATION
        assert Aspect.EXPRESSION_REGULATION in pair.shared_aspects

    def test_unknown_aspect_wildcard_admits(self):
        """UNKNOWN 通配：一侧词典零命中时 aspect 判据退化为恒真（保守放行）。"""
        rubric = AsymmetricRubric()
        ev_a = _ev("ev-a", "该药物临床评价为有效", source="pubmed")
        ev_b = _ev("ev-b", "该药物临床评价为无效", source="experiment")
        scores = [_score("ev-a"), _score("ev-b")]
        emap = {"ev-a": ev_a, "ev-b": ev_b}

        verdict = rubric.evaluate(scores, emap, topic="治疗反应")
        # "有效/无效"词对全切面合法；即便双方切面判定有 UNKNOWN 也不拦截
        assert verdict.passed is False
        assert verdict.contradictions[0].matched_keyword_pair == ("有效", "无效")


# ============================================================================
# U2 子串极性修复（始终生效，与开关无关）
# ============================================================================

class TestPolaritySubstringFix:
    """_polarity_sides / _pair_hit_by_polarity 极性拆分测试。"""

    _SPEC_EXIST = _spec("存在", "不存在")
    _SPEC_ACTIVATE = _spec("激活", "抑制")

    def test_both_negated_no_hit(self):
        """两侧均写'不存在'：'存在'不得从'不存在'中误命中肯定侧。"""
        a = "耐药位点在该队列中不存在"
        b = "y96d 二次突变不存在"
        pos_a, neg_a = _polarity_sides(a, self._SPEC_EXIST)
        pos_b, neg_b = _polarity_sides(b, self._SPEC_EXIST)
        assert "存在" not in pos_a and "存在" not in pos_b
        assert "不存在" in neg_a and "不存在" in neg_b
        assert _pair_hit_by_polarity(a, b, self._SPEC_EXIST) is False

    def test_true_contradiction_hit(self):
        """一方'存在'（无否定前缀）+ 一方'不存在' → 正常命中。"""
        a = "患者中检测到 y96d 二次突变，该位点存在"
        b = "队列分析显示该位点不存在"
        assert _pair_hit_by_polarity(a.lower(), b.lower(), self._SPEC_EXIST) is True

    def test_negation_prefix_on_ka_not_affirmative(self):
        """单字否定前缀覆盖：'未激活'不得计入肯定侧（激活 ⊂ 未激活）。

        设计语义（保守方向）：(激活/抑制) 这类非 negated_b 词对只对
        ka 的肯定出现敏感——"未激活 vs 抑制"属同侧否定表述，
        不构成可定罪的方向对立，与"不存在 vs 无效"不配对的既有语义一致。
        """
        a = "通路未激活"
        pos_a, neg_a = _polarity_sides(a, self._SPEC_ACTIVATE)
        assert "激活" not in pos_a  # 被"未"前缀覆盖，肯定侧零命中
        assert neg_a                # 覆盖后的出现归否定侧留痕
        # 肯定侧被清零 → 与任何对侧都不再构成 (激活/抑制) 词对命中
        assert _pair_hit_by_polarity(a, "药物抑制该通路", self._SPEC_ACTIVATE) is False
        assert _pair_hit_by_polarity(a, "通路未激活", self._SPEC_ACTIVATE) is False

    def test_fix_active_even_with_all_switches_off(self):
        """三开关全关时子串修复仍生效（正确性修复不属于可选机制）。"""
        rubric = AsymmetricRubric(
            enable_aspect=False, enable_validity_gate=False, enable_intensity=False,
        )
        ev_a = _ev("ev-a", "队列分析显示 y96d 耐药位点不存在", source="pubmed")
        ev_b = _ev("ev-b", "同一队列的测序复核，二次突变不存在", source="experiment")
        scores = [_score("ev-a"), _score("ev-b")]
        emap = {"ev-a": ev_a, "ev-b": ev_b}

        verdict = rubric.evaluate(scores, emap, topic="耐药位点")
        assert verdict.passed is True
        assert verdict.contradictions == []


# ============================================================================
# U3 ValidityGate 三态
# ============================================================================

class TestValidityGate:
    """对象同一性门禁三态判定测试。"""

    def setup_method(self):
        """每个用例独立门禁实例（无内部状态，保持一致构造）。"""
        self.gate = ValidityGate()

    def test_no_object_signal_unknown_admits(self):
        """双方均无对象信号 → UNKNOWN（放行定罪，ABP 保守方向）。"""
        ev_a = _ev("ev-a", "临床评价为有效")
        ev_b = _ev("ev-b", "临床评价为无效", source="experiment")
        result = self.gate.check(ev_a, ev_b, topic="治疗反应")
        assert result.decision is GateDecision.UNKNOWN

    def test_shared_canonical_key_same_object(self):
        """同 canonical（含别名归一，如 CD276/B7-H3）→ SAME_OBJECT。"""
        ev_a = _ev("ev-a", "ADC 在 CD276 高表达肿瘤中显示阳性应答")
        ev_b = _ev("ev-b", "双抗在 B7-H3 靶向队列中应答为阴性", source="experiment")
        result = self.gate.check(ev_a, ev_b, topic="B7-H3 靶向策略")
        assert result.decision is GateDecision.SAME_OBJECT
        assert "CD276" in result.matched_keys

    def test_different_specific_objects_intercepted(self):
        """双方均为词典专名且键集不相交 → DIFFERENT_OBJECT（拦截不定罪）。"""
        ev_a = _ev("ev-a", "EGFR 抑制剂在一线治疗中有效")
        ev_b = _ev("ev-b", "KRAS G12C 抑制剂在一线治疗中无效", source="experiment")
        result = self.gate.check(ev_a, ev_b, topic="一线方案")
        assert result.decision is GateDecision.DIFFERENT_OBJECT

    def test_metadata_target_participates(self):
        """metadata['target'] 显式对象参与同一性判断（别名归一后相交）。"""
        ev_a = _ev("ev-a", "单药有效", metadata={"target": "ERBB1"})
        ev_b = _ev("ev-b", "单药无效", source="experiment", metadata={"target": "EGFR"})
        result = self.gate.check(ev_a, ev_b)
        assert result.decision is GateDecision.SAME_OBJECT
        assert "EGFR" in result.matched_keys

    def test_gate_different_blocks_conviction_end_to_end(self):
        """端到端：门禁 DIFFERENT 拦截 → 词对命中也不定罪。"""
        rubric = AsymmetricRubric()
        ev_a = _ev("ev-a", "EGFR 变构抑制剂对三重突变有效", source="pubmed")
        ev_b = _ev("ev-b", "KRAS G12C 抑制剂对原发灶无效", source="experiment")
        scores = [_score("ev-a"), _score("ev-b")]
        emap = {"ev-a": ev_a, "ev-b": ev_b}

        verdict = rubric.evaluate(scores, emap, topic="抑制剂选择")
        assert verdict.passed is True
        assert verdict.pending_pairs == []  # DIFFERENT 是"非对抗"，不是待举证

    def test_ambiguous_downgrade_via_injected_extractor(self):
        """端到端：仅泛指信号冲突（无 MOLECULE 专名佐证）→ AMBIGUOUS 降级。

        用注入式抽取器构造"双方均有信号但均非专名"场景：现象键不同
        （旁观者效应 vs 耐药机制 → 宿主键 DXd vs KRAS 相交为空，
        且双方只有 PHENOMENON/HOST 对象、无 MOLECULE 专名）。
        """
        class _OnlyPhenomenonExtractor:
            """只返回 PHENOMENON 对象的抽取器（消融注入点用法示范）。"""

            def __init__(self, phenomenon: str):
                self._p = phenomenon

            def extract(self, ev, topic=""):
                from app.core.virtual_lab.rubric_engine import ScientificObject
                hosts = {"旁观者效应": "DXd", "耐药机制": "KRAS"}
                return [
                    ScientificObject("PHENOMENON", self._p, self._p),
                    ScientificObject("HOST", hosts[self._p], hosts[self._p]),
                ]

        # 双证据各注入不同现象键：通过 Per-rubric 注入需要两个 gate，
        # 这里直接单测 ValidityGate.check 的 AMBIGUOUS 分支（键集不相交、
        # 双方无 MOLECULE → AMBIGUOUS）
        class _SplitExtractor:
            """按证据 id 分流现象键的抽取器。"""

            def extract(self, ev, topic=""):
                p = "旁观者效应" if ev.evidence_id.endswith("a") else "耐药机制"
                return _OnlyPhenomenonExtractor(p).extract(ev, topic)

        gate = ValidityGate(_SplitExtractor())
        ev_a = _ev("ev-a", "载药 ADC 有效")
        ev_b = _ev("ev-b", "载药 ADC 无效", source="experiment")
        result = gate.check(ev_a, ev_b, topic="ADC 策略")
        assert result.decision is GateDecision.AMBIGUOUS

        # 端到端（rubric 注入同一抽取器）：AMBIGUOUS → pending_pairs 降级
        rubric = AsymmetricRubric(object_extractor=_SplitExtractor())
        scores = [_score("ev-a"), _score("ev-b")]
        emap = {"ev-a": ev_a, "ev-b": ev_b}
        verdict = rubric.evaluate(scores, emap, topic="ADC 策略")
        assert verdict.passed is True
        assert verdict.contradictions == []
        assert len(verdict.pending_pairs) == 1
        pending = verdict.pending_pairs[0]
        assert pending.severity.value == "PENDING"
        assert pending.gate_detail.decision is GateDecision.AMBIGUOUS
        # 降级候选同步进入人读 pending_issues 供复核
        assert any("降为待举证" in issue for issue in verdict.pending_issues)

    def test_phenomenon_same_key_convicted(self):
        """现象同名（旁观者效应）经现象键相交判 SAME → 照常定罪。"""
        rubric = AsymmetricRubric()
        ev_a = _ev("ev-a", "旁观者效应显著，载药 ADC 有效", source="pubmed")
        ev_b = _ev("ev-b", "旁观者效应有限，载药 ADC 无效", source="experiment")
        scores = [_score("ev-a"), _score("ev-b")]
        emap = {"ev-a": ev_a, "ev-b": ev_b}

        verdict = rubric.evaluate(scores, emap, topic="ADC 旁观者效应")
        assert verdict.passed is False
        assert verdict.contradictions[0].gate_detail.decision is GateDecision.SAME_OBJECT


# ============================================================================
# U4 强度分级
# ============================================================================

class TestGradeIntensity:
    """_grade_intensity 纯函数分级测试。"""

    def test_degree_polarity_baseline_one(self):
        """degree 极性基线 = 1（无来源差、无门禁佐证）。"""
        spec = _spec("有效", "无效", polarity=POLARITY_DEGREE)
        assert _grade_intensity(spec, "pubmed", "pubmed", None) == 1

    def test_directional_baseline_two(self):
        """directional 极性基线 = 2。"""
        spec = _spec("有效", "无效", polarity=POLARITY_DIRECTIONAL)
        assert _grade_intensity(spec, "pubmed", "pubmed", None) == 2

    def test_source_gap_bonus(self):
        """来源可信度差 |Δ| ≥ 0.30 → +1（pubmed 0.9 vs llm 0.5）。"""
        from app.core.virtual_lab.rubric_engine import GateResult
        spec = _spec("激活", "抑制")
        gate_same = GateResult(decision=GateDecision.UNKNOWN)
        assert _grade_intensity(spec, "pubmed", "llm", gate_same) == 3

    def test_same_object_evidence_boost(self):
        """SAME_OBJECT 佐证把方向基线 2 提到 3（同一对象上的方向对立）。"""
        from app.core.virtual_lab.rubric_engine import GateResult
        spec = _spec("上调", "下调")
        gate = GateResult(decision=GateDecision.SAME_OBJECT)
        assert _grade_intensity(spec, "chembl", "chembl", gate) == 3

    def test_clamped_to_range(self):
        """多重加成叠乘时截断在 [1, 3]（degree+来源差=2，不越界）。"""
        spec = _spec("有效", "无效", polarity=POLARITY_DEGREE)
        # degree(1) + 来源差(+1) = 2；degree 基线不享受 SAME 加成
        from app.core.virtual_lab.rubric_engine import GateResult
        gate = GateResult(decision=GateDecision.SAME_OBJECT)
        assert _grade_intensity(spec, "pubmed", "unknown", gate) == 2

    def test_hard_constraint_shortcut(self):
        """硬约束词对直接短路为 3（v0.4 AF2 类硬约束接入预留）。"""
        spec = _spec("有效", "无效", hard_constraint=True)
        assert _grade_intensity(spec, "unknown", "unknown", None) == 3


# ============================================================================
# U5 开关差分（v0.2 等价性）
# ============================================================================

class TestSwitchDifferential:
    """三开关独立回退差分测试（§6.4 消融自变量的开关语义锁）。"""

    #: 跨切面伪冲突案例：aspect 开 → 通过；aspect 关 → 定罪
    def _pseudo_case(self) -> tuple:
        ev_a = _ev("ev-a", "该候选分子的合成失败，路线收率无法放大", source="pubmed")
        ev_b = _ev("ev-b", "遗传学证据充分，靶点验证成功", source="experiment")
        return [ev_a, ev_b]

    def test_aspect_off_restores_v02_conviction(self):
        """aspect 关闭：跨切面伪冲突回到 v0.2 定罪行为。"""
        ev_a, ev_b = self._pseudo_case()
        scores = [_score("ev-a"), _score("ev-b")]
        emap = {ev.evidence_id: ev for ev in (ev_a, ev_b)}

        on = AsymmetricRubric(enable_aspect=True).evaluate(scores, emap, "决策")
        off = AsymmetricRubric(enable_aspect=False).evaluate(scores, emap, "决策")
        assert on.passed is True
        assert off.passed is False
        # aspect 关闭时 matched_aspect 不填（机制不参与判定）
        assert off.contradictions[0].matched_aspect is None

    def test_gate_off_pending_pairs_always_empty(self):
        """gate 关闭：pending_pairs 恒空、DIFFERENT 对回到定罪通道。"""
        ev_a = _ev("ev-a", "EGFR 抑制剂对三重突变有效", source="pubmed")
        ev_b = _ev("ev-b", "KRAS G12C 抑制剂对原发灶无效", source="experiment")
        scores = [_score("ev-a"), _score("ev-b")]
        emap = {"ev-a": ev_a, "ev-b": ev_b}

        on = AsymmetricRubric(enable_validity_gate=True).evaluate(scores, emap, "方案")
        off = AsymmetricRubric(enable_validity_gate=False).evaluate(scores, emap, "方案")
        assert on.passed is True and on.pending_pairs == []
        assert off.passed is False and off.pending_pairs == []
        assert off.contradictions[0].gate_detail is None

    def test_intensity_off_classic_desc_and_constant_two(self):
        """intensity 关闭：intensity 恒 2、desc 逐字等于 v0.2 经典模板。"""
        from app.core.virtual_lab.rubric_engine import CLASSIC_DESC_TEMPLATE
        ev_a = _ev("ev-a", "该药物临床有效", source="pubmed")
        ev_b = _ev("ev-b", "该药物临床无效", source="agent_opinion")
        scores = [_score("ev-a"), _score("ev-b")]
        emap = {"ev-a": ev_a, "ev-b": ev_b}

        off = AsymmetricRubric(enable_intensity=False).evaluate(scores, emap, "疗效")
        pair = off.contradictions[0]
        assert pair.intensity == 2
        assert pair.contradiction_desc == CLASSIC_DESC_TEMPLATE.format(
            a_id="ev-a", b_id="ev-b", ka="有效", kb="无效", topic="疗效",
        )
        # 开→关对比：来源差（pubmed 0.9 vs agent_opinion 0.4）在开启时应加成
        on = AsymmetricRubric(enable_intensity=True).evaluate(scores, emap, "疗效")
        assert on.contradictions[0].intensity == 3

    def test_all_off_equal_v02_semantics(self):
        """三开关全关：定罪集合与 v0.2 语义一致（词对命中即定罪、intensity=2）。"""
        ev_a = _ev("ev-a", "给药后 TP53 表达上调", source="pubmed")
        ev_b = _ev("ev-b", "给药后 TP53 表达下调", source="experiment")
        scores = [_score("ev-a"), _score("ev-b")]
        emap = {"ev-a": ev_a, "ev-b": ev_b}

        v = AsymmetricRubric(
            enable_aspect=False, enable_validity_gate=False, enable_intensity=False,
        ).evaluate(scores, emap, "p53 表达")
        assert v.passed is False
        pair = v.contradictions[0]
        assert pair.intensity == 2
        assert pair.matched_aspect is None
        assert pair.gate_detail is None
        assert pair.matched_keyword_pair is None  # 经典路径不填结构化字段
        assert "上调" in pair.contradiction_desc and "下调" in pair.contradiction_desc

    def test_rationale_intensity_distribution_only_when_on(self):
        """rationale 的强度分布摘要仅在 intensity 开启时追加。"""
        ev_a = _ev("ev-a", "该药物临床有效", source="pubmed")
        ev_b = _ev("ev-b", "该药物临床无效", source="experiment")
        scores = [_score("ev-a"), _score("ev-b")]
        emap = {"ev-a": ev_a, "ev-b": ev_b}

        on = AsymmetricRubric(enable_intensity=True).evaluate(scores, emap, "疗效")
        off = AsymmetricRubric(enable_intensity=False).evaluate(scores, emap, "疗效")
        assert "强度分布" in on.rationale
        assert "强度分布" not in off.rationale


# ============================================================================
# U6 硬约束前置扫描（v0.4 AF2/AF3 禁令接入）
# ============================================================================

class TestHardConstraintScan:
    """P0 硬约束短路通道集成测试（§4.3/§4.4：CRITICAL 必须排除，
    不依赖词对匹配——单条证据命中即构成定罪矛盾对 intensity=3）。

    覆盖：
      - U6.1 _scan_hard_constraints 纯函数行为（AF2/AF3 双禁 + 误匹配排除）
      - U6.2 AsymmetricRubric.evaluate 端到端命中 AF2 引用
      - U6.3 AF3 引用同样触发（v0.4.8 双禁——合规降级 OpenDDE→Chai1→Boltz）
      - U6.4 无硬约束引用的证据集不受影响
      - U6.5 误匹配排除（"alphafold2" 不重复命中 "af2"）
      - U6.6 单证据双禁（AF2+AF3 同现）→ 仅 1 条族级 CRITICAL
    """

    # ---------- U6.1 纯函数 ----------

    def test_u6_1_scan_hit_alphafold2(self):
        """"AlphaFold2" 命中模式 1（alphafold2），排除词阻止模式 2（af2）重复命中。"""
        from app.core.virtual_lab.rubric_engine import _scan_hard_constraints
        hits = _scan_hard_constraints("该研究使用了 AlphaFold2 进行结构预测")
        assert len(hits) == 1
        assert hits[0][1] == "structure-tool-ban"
        assert hits[0][0] == "alphafold2"

    def test_u6_1_scan_hit_af2_token(self):
        """独立 "AF2" token（非 AlphaFold2 全拼）命中模式 2。"""
        from app.core.virtual_lab.rubric_engine import _scan_hard_constraints
        hits = _scan_hard_constraints("我们复现了 AF2 的基准结果")
        assert len(hits) == 1
        assert hits[0][0] == "af2"

    def test_u6_1_scan_hit_alphafold3(self):
        """v0.4.8：AlphaFold3 引用触发（双禁）——命中 alphafold3，排除词阻止 af3 重复。"""
        from app.core.virtual_lab.rubric_engine import _scan_hard_constraints
        hits = _scan_hard_constraints("该模型基于 AlphaFold3 进行预测")
        assert len(hits) == 1
        assert hits[0][1] == "structure-tool-ban"
        assert hits[0][0] == "alphafold3"

    def test_u6_1_scan_hit_af3_token(self):
        """独立 "AF3" token 命中模式 4（v0.4.8 双禁）。"""
        from app.core.virtual_lab.rubric_engine import _scan_hard_constraints
        hits = _scan_hard_constraints("我们复现了 AF3 的基准结果")
        assert len(hits) == 1
        assert hits[0][0] == "af3"

    def test_u6_1_scan_miss_no_reference(self):
        """无硬约束引用的文本不产生命中。"""
        from app.core.virtual_lab.rubric_engine import _scan_hard_constraints
        hits = _scan_hard_constraints("该候选分子的合成收率较高")
        assert hits == []

    # ---------- U6.2 端到端 ----------

    def test_u6_2_end_to_end_af2_violation(self):
        """单条证据含 AF2 引用 → evaluate 产出 1 条 CRITICAL 矛盾对（intensity=3）。"""
        ev_a = _ev("ev-a", "该研究使用 AF2 对靶点进行了结构建模", source="pubmed")
        ev_b = _ev("ev-b", "临床前数据显示该候选分子有效", source="experiment")
        scores = [_score("ev-a"), _score("ev-b")]
        emap = {"ev-a": ev_a, "ev-b": ev_b}

        verdict = AsymmetricRubric().evaluate(scores, emap, "候选分子推进决策")

        # 硬约束矛盾对：ev_a 自身即触发（evidence_a_id == evidence_b_id）
        hc_pairs = [
            c for c in verdict.contradictions
            if c.evidence_a_id == c.evidence_b_id
        ]
        assert len(hc_pairs) == 1, f"期望 1 条硬约束矛盾对，实际 {len(hc_pairs)}: {verdict.contradictions}"
        assert hc_pairs[0].severity == Severity.CRITICAL
        assert hc_pairs[0].intensity == 3
        assert "structure-tool-ban" in hc_pairs[0].contradiction_desc
        # 硬约束命中使整体 failed
        assert verdict.passed is False

    def test_u6_3_end_to_end_af3_violation(self):
        """v0.4.8 双禁：AF3 引用同样触发硬约束 → 1 条 CRITICAL 矛盾对。"""
        ev_a = _ev("ev-a", "该研究使用 AlphaFold3 进行结构预测", source="pubmed")
        ev_b = _ev("ev-b", "临床前数据显示该候选分子有效", source="experiment")
        scores = [_score("ev-a"), _score("ev-b")]
        emap = {"ev-a": ev_a, "ev-b": ev_b}

        verdict = AsymmetricRubric().evaluate(scores, emap, "候选分子推进决策")

        # AF3 触发硬约束（双禁）
        hc_pairs = [
            c for c in verdict.contradictions
            if c.evidence_a_id == c.evidence_b_id
        ]
        assert len(hc_pairs) == 1, f"AF3 应触发 1 条硬约束: {hc_pairs}"
        assert hc_pairs[0].severity == Severity.CRITICAL
        assert hc_pairs[0].intensity == 3
        assert verdict.passed is False

    def test_u6_6_end_to_end_dual_ban_single_critical(self):
        """v0.4.8 双禁：单证据同含 AF2+AF3 → 仅 1 条族级 CRITICAL（rule_name 去重）。"""
        ev_a = _ev("ev-a", "我们对比了 AlphaFold3 与 AF2 的预测", source="pubmed")
        scores = [_score("ev-a")]
        emap = {"ev-a": ev_a}

        verdict = AsymmetricRubric().evaluate(scores, emap, "结构预测")

        hc_pairs = [
            c for c in verdict.contradictions
            if c.evidence_a_id == c.evidence_b_id
        ]
        assert len(hc_pairs) == 1, f"AF2+AF3 同现应仅 1 条族级 CRITICAL，实际 {len(hc_pairs)}"
        assert hc_pairs[0].severity == Severity.CRITICAL
        assert hc_pairs[0].intensity == 3

    def test_u6_4_end_to_end_no_hard_constraint_unchanged(self):
        """无硬约束引用的证据集 → 矛盾对全部来自词对流水线（无 self-pair）。"""
        ev_a = _ev("ev-a", "该候选分子合成失败", source="pubmed")
        ev_b = _ev("ev-b", "实验显示合成成功", source="experiment")
        scores = [_score("ev-a"), _score("ev-b")]
        emap = {"ev-a": ev_a, "ev-b": ev_b}

        verdict = AsymmetricRubric().evaluate(scores, emap, "合成可行性")

        # 不应有硬约束 self-pair
        hc_pairs = [
            c for c in verdict.contradictions
            if c.evidence_a_id == c.evidence_b_id
        ]
        assert len(hc_pairs) == 0

    def test_u6_5_alphafold2_no_duplicate(self):
        """"AlphaFold2" 不重复命中：模式 1（alphafold2）命中 + 排除词阻止模式 2（af2）→ 仅 1 条。"""
        ev_a = _ev("ev-a", "结构预测使用了 AlphaFold2 模型", source="pubmed")
        scores = [_score("ev-a")]
        emap = {"ev-a": ev_a}

        verdict = AsymmetricRubric().evaluate(scores, emap, "结构预测")

        hc_pairs = [
            c for c in verdict.contradictions
            if c.evidence_a_id == c.evidence_b_id
        ]
        assert len(hc_pairs) == 1, f"AlphaFold2 应仅触发 1 条硬约束对，实际 {len(hc_pairs)}"
