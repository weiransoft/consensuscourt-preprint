"""
DDB-L2 v0.3 红线结构锁 + 消融冒烟 + 升级挂钩测试（U6 / U7）

对应大纲 §4.7 升级条款 5/6 与 §6.4 消融设计，用例编号：
- U6  升级挂钩：intensity=3 定罪矛盾对 → CourtVerdict.escalated=True +
      覆盖清单条款 5 + payload 留痕 max_contradiction_intensity
- U7  L2 红线结构锁：L2 正例金标矛盾对在 v0.3 机制全开下必须保持 100%
      召回（aspect/gate 只降 FPR 不降召回的保守公理）；词典覆盖锁
      （每个正例证据至少一个非 UNKNOWN 切面）；机制全开召回 ≥ 全关召回
- U8  升级条款 6（v0.4.9）：双轨仲裁分歧（PIAgentArbiter 规则/模型轨 vs
      IndependentReviewer 证据核查轨结论语义冲突）→ 强制升级 + payload 留痕
- 消融冒烟：三开关 8 种组合在 L2 上全部可运行、结果结构完整（无异常），
      为第 1-2 月正式消融（13 项）冒烟铺路

运行：backend$ .venv/bin/pytest tests/ddb/test_l2_v03_locks.py -q
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# 导入前置：backend 根目录入 sys.path + 环境变量占位（必须在 import app 之前）
# ---------------------------------------------------------------------------
_BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

os.environ.setdefault("SCIMINER_API_KEY", "ddb-v03-placeholder")
os.environ.setdefault("DB_PASSWORD", "ddb-v03-placeholder")

import pytest  # noqa: E402

from app.core.virtual_lab.consensus_court import (  # noqa: E402
    Evidence,
    EvidenceScorer,
    HumanBOSSOverride,
    verdict_to_payload,
)
from app.core.virtual_lab.rubric_engine import (  # noqa: E402
    Aspect,
    AspectInferrer,
    AsymmetricRubric,
)
from tests.ddb.l2_cases import load_cases  # noqa: E402


# ---------------------------------------------------------------------------
# 案例 dict → Evidence（与 test_l2_cases._to_evidence 同构）
# ---------------------------------------------------------------------------

def _to_evidence(ev: dict) -> Evidence:
    """把 L2 案例证据字典无损还原为 consensus_court.Evidence 对象。"""
    timestamp: Optional[datetime] = None
    if ev.get("timestamp"):
        timestamp = datetime.fromisoformat(ev["timestamp"])
    return Evidence(
        evidence_id=ev["evidence_id"],
        source=ev["source"],
        content=ev["content"],
        role=ev["role"],
        timestamp=timestamp,
        reference_ids=list(ev.get("reference_ids") or []),
        metadata=dict(ev.get("metadata") or {}),
    )


def _rubric_case(case: dict, **switches) -> tuple[bool, list]:
    """
    在单个 L2 案例上运行指定开关组合的 ABP，返回 (是否定罪, 矛盾对列表)。

    Args:
        case: L2 案例字典
        **switches: AsymmetricRubric 三开关（enable_* 系列）

    Returns:
        (not verdict.passed, verdict.contradictions)
    """
    scorer = EvidenceScorer()
    rubric = AsymmetricRubric(**switches)
    evidences = [_to_evidence(ev) for ev in case["evidences"]]
    scores = [scorer.score(ev, case["topic"]) for ev in evidences]
    evidence_map = {ev.evidence_id: ev for ev in evidences}
    verdict = rubric.evaluate(scores, evidence_map, case["topic"])
    return (not verdict.passed), verdict.contradictions


# ---------------------------------------------------------------------------
# 基准缓存（模块级共享）
# ---------------------------------------------------------------------------

_CASES_CACHE: Optional[list[dict]] = None


def _cases() -> list[dict]:
    """加载首批 L2 基准（10 例，模块级缓存）。"""
    global _CASES_CACHE
    if _CASES_CACHE is None:
        _CASES_CACHE = load_cases()
    return _CASES_CACHE


# ===========================================================================
# U6 升级条款 5 挂钩（consensus_court 层）
# ===========================================================================

class TestEscalationClause5:
    """升级条款 5：intensity=3 定罪矛盾对强制升级。"""

    def test_override_item_only_when_intensity_three(self):
        """generate_items：仅 max_contradiction_intensity>=3 追加条款 5 事项。"""
        hb = HumanBOSSOverride()
        # 高置信 + 支持提案 + 无评审异议 + 无低质证据 → 条款 1-4 全不命中
        from app.core.virtual_lab.consensus_court import EvidenceScore
        ok_score = EvidenceScore("ev-1", 0.9, 0.9, 0.9, 0.9)

        items_below = hb.generate_items("支持提案", 0.9, [ok_score], [], 2)
        assert items_below == []

        items_at3 = hb.generate_items("支持提案", 0.9, [ok_score], [], 3)
        assert len(items_at3) == 1
        assert "升级条款 5" in items_at3[0]

    def test_default_param_backward_compatible(self):
        """默认参数（不传强度）时行为与 v0.2 完全一致（无条款 5 事项）。"""
        hb = HumanBOSSOverride()
        from app.core.virtual_lab.consensus_court import EvidenceScore
        low = EvidenceScore("ev-1", 0.2, 0.2, 0.2, 0.2)
        items = hb.generate_items("驳回提案", 0.9, [low], ["异议"])
        assert not any("升级条款 5" in it for it in items)
        # 其余条款照常命中（驳回 + 低质证据 + 评审异议）
        assert len(items) == 3

    def test_verdict_to_payload_carries_intensity(self):
        """payload 序列化包含 max_contradiction_intensity 字段。"""
        from app.core.virtual_lab.consensus_court import CourtVerdict
        verdict = CourtVerdict(
            verdict="v", decision="支持提案", confidence=0.9,
            evidence_scores=[], pi_rationale="r", reviewer_notes=[],
            human_override_items=[], escalated=True,
            max_contradiction_intensity=3,
        )
        payload = verdict_to_payload(verdict, "topic", [])
        assert payload["max_contradiction_intensity"] == 3

    def test_verdict_default_intensity_zero(self):
        """既有八字段构造（v0.2 形态）默认强度为 0。"""
        from app.core.virtual_lab.consensus_court import CourtVerdict
        verdict = CourtVerdict(
            verdict="v", decision="支持提案", confidence=0.9,
            evidence_scores=[], pi_rationale="r", reviewer_notes=[],
            human_override_items=[], escalated=False,
        )
        assert verdict.max_contradiction_intensity == 0


# ===========================================================================
# U8 升级条款 6（v0.4.9）：双轨仲裁分歧强制升级
# ===========================================================================

class TestEscalationClause6:
    """升级条款 6：规则/模型仲裁轨（PIAgentArbiter）与证据型核查轨
    （IndependentReviewer）结论语义冲突 → 强制升级（§4.7 / Tab.5）。"""

    def test_divergence_true_when_pi_affirm_review_pushback(self):
        """PI 支持提案 + 评审建议人工复核 → 两轨分歧为真。"""
        from app.core.virtual_lab.consensus_court import IndependentReviewer
        rev = IndependentReviewer()
        notes = ["PI Agent 驳回了提案，但存在高质量支持证据，建议人工复核。"]
        assert rev.contradicts_pi_decision("支持提案", notes) is True

    def test_divergence_false_when_pi_reject_review_pushback(self):
        """PI 驳回提案（本身已谨慎）+ 评审异议 → 不算分歧（方向不矛盾）。"""
        from app.core.virtual_lab.consensus_court import IndependentReviewer
        rev = IndependentReviewer()
        notes = ["缺少高质量证据支撑，建议补充实验或文献。"]
        assert rev.contradicts_pi_decision("驳回提案", notes) is False

    def test_divergence_false_when_no_pushback(self):
        """PI 支持提案 + 评审无异议 → 两轨一致，不分歧。"""
        from app.core.virtual_lab.consensus_court import IndependentReviewer
        rev = IndependentReviewer()
        assert rev.contradicts_pi_decision("支持提案", []) is False

    def test_override_item_only_when_divergence(self):
        """generate_items：仅 dual_track_divergence=True 追加条款 6 事项。"""
        hb = HumanBOSSOverride()
        from app.core.virtual_lab.consensus_court import EvidenceScore
        ok_score = EvidenceScore("ev-1", 0.9, 0.9, 0.9, 0.9)

        items_no = hb.generate_items(
            "支持提案", 0.9, [ok_score], [], 0, False
        )
        assert not any("升级条款 6" in it for it in items_no)

        items_div = hb.generate_items(
            "支持提案", 0.9, [ok_score], [], 0, True
        )
        assert len(items_div) == 1
        assert "升级条款 6" in items_div[0]

    def test_default_param_backward_compatible(self):
        """默认不传分歧标志时行为与 v0.4.8 完全一致（无条款 6 事项）。"""
        hb = HumanBOSSOverride()
        from app.core.virtual_lab.consensus_court import EvidenceScore
        ok_score = EvidenceScore("ev-1", 0.9, 0.9, 0.9, 0.9)
        # 与 test_override_item_only_when_intensity_three 同口径调用（无第 6 参）
        items = hb.generate_items("支持提案", 0.9, [ok_score], [], 2)
        assert items == []

    def test_verdict_to_payload_carries_divergence(self):
        """payload 序列化包含 dual_track_divergence 字段。"""
        from app.core.virtual_lab.consensus_court import CourtVerdict
        verdict = CourtVerdict(
            verdict="v", decision="支持提案", confidence=0.9,
            evidence_scores=[], pi_rationale="r", reviewer_notes=[],
            human_override_items=[], escalated=True,
            dual_track_divergence=True,
        )
        payload = verdict_to_payload(verdict, "topic", [])
        assert payload["dual_track_divergence"] is True

    def test_verdict_default_divergence_false(self):
        """既有八字段构造（v0.2 形态）默认分歧标志为 False。"""
        from app.core.virtual_lab.consensus_court import CourtVerdict
        verdict = CourtVerdict(
            verdict="v", decision="支持提案", confidence=0.9,
            evidence_scores=[], pi_rationale="r", reviewer_notes=[],
            human_override_items=[], escalated=False,
        )
        assert verdict.dual_track_divergence is False


# ===========================================================================
# U7 L2 红线结构锁（召回不回退 + 词典覆盖）
# ===========================================================================

class TestL2RedLineLocks:
    """v0.3 机制在 L2 真实文献争议上的召回红线锁。"""

    def test_l2l_001_full_mechanism_recall_100(self):
        """正例锁：机制全开时，每个正例至少一对金标矛盾对被检出。"""
        cases = [c for c in _cases() if c["ground_truth"]["has_true_contradiction"]]
        missed: list[str] = []
        for case in cases:
            convicted, pairs = _rubric_case(case)
            assert convicted, f"{case['case_id']}: v0.3 全开未定罪（v0.2 召回 100%）"
            hit = {frozenset((p.evidence_a_id, p.evidence_b_id)) for p in pairs}
            gold = {frozenset(pair) for pair in case["ground_truth"]["contradiction_pairs"]}
            if not (hit & gold):
                missed.append(case["case_id"])
        assert not missed, f"金标矛盾对全部漏检（红线）: {missed}"

    def test_l2l_002_open_ge_closed_recall(self):
        """消融单调性冒烟锁：机制全开定罪案例集 ⊇ 全关闭罪案例集不强制，
        但全开召回数不得低于全关（保守公理：新机制只拦截伪冲突）。"""
        cases = [c for c in _cases() if c["ground_truth"]["has_true_contradiction"]]
        n_on = sum(1 for c in cases if _rubric_case(c)[0])
        n_off = sum(1 for c in cases if _rubric_case(
            c, enable_aspect=False, enable_validity_gate=False,
            enable_intensity=False)[0])
        assert n_on >= n_off, (
            f"机制全开召回 {n_on} < 全关召回 {n_off}，违反 ABP 保守方向"
        )

    def test_l2l_003_aspect_lexicon_coverage(self):
        """词典覆盖锁：每个正例的全部证据切面推断不得全为 UNKNOWN
        （Aspect 触发词典必须覆盖 L2 金标句术语）。"""
        inf = AspectInferrer()
        gaps: list[str] = []
        for case in _cases():
            if not case["ground_truth"]["has_true_contradiction"]:
                continue
            for ev in case["evidences"]:
                aspects = inf.infer(_to_evidence(ev), case["topic"])
                if aspects == frozenset({Aspect.UNKNOWN}):
                    gaps.append(f"{case['case_id']}/{ev['evidence_id']}")
        assert not gaps, f"Aspect 词典覆盖缺口（红线）: {gaps}"

    def test_l2l_004_no_new_false_conviction_on_negatives(self):
        """负例锁：无矛盾案例（若有）在机制全开下不得被新机制误定罪。
        L2 首批全部为正例时，本锁以空集合平凡通过并保持结构性存在。"""
        negatives = [c for c in _cases()
                     if not c["ground_truth"]["has_true_contradiction"]]
        for case in negatives:
            convicted, _ = _rubric_case(case)
            assert not convicted, f"{case['case_id']}: 负例被 v0.3 误定罪"


# ===========================================================================
# 消融冒烟（§6.4 自变量 8 组合可运行性）
# ===========================================================================

class TestAblationSmoke:
    """三开关 8 组合消融冒烟：全部可运行、结果结构完整。"""

    @pytest.mark.parametrize("aspect", [True, False])
    @pytest.mark.parametrize("gate", [True, False])
    @pytest.mark.parametrize("intensity", [True, False])
    def test_all_switch_combinations_run(self, aspect, gate, intensity):
        """8 种开关组合在全部 L2 案例上运行无异常且不变量成立。

        不变量：
        - gate 关闭 → pending_pairs 恒空；gate 开启 → pending_pairs 中
          全部是 PENDING 严重级；
        - intensity 关闭 → 词对路径矛盾对 intensity 恒 2（硬约束对恒 3，
          v0.4.8 AF2/AF3 双禁：结构预测工具禁令不依赖三机制，独立于
          enable_intensity 短路）；
        - aspect 关闭 → matched_aspect 恒 None。
        """
        switches = {
            "enable_aspect": aspect,
            "enable_validity_gate": gate,
            "enable_intensity": intensity,
        }
        for case in _cases():
            scorer = EvidenceScorer()
            rubric = AsymmetricRubric(**switches)
            evidences = [_to_evidence(ev) for ev in case["evidences"]]
            scores = [scorer.score(ev, case["topic"]) for ev in evidences]
            emap = {ev.evidence_id: ev for ev in evidences}
            verdict = rubric.evaluate(scores, emap, case["topic"])

            # pending_pairs 严重级不变量
            for p in verdict.pending_pairs:
                assert p.severity.value == "PENDING"
            if not gate:
                assert verdict.pending_pairs == []
            # 强度不变量（硬约束对恒 3，词对路径受 enable_intensity 控制）
            for c in verdict.contradictions:
                assert c.intensity in (1, 2, 3)
                if not intensity:
                    # 硬约束对（同证据，v0.4.8 AF2/AF3 双禁）恒 3；
                    # 词对路径对恒 2（_grade_intensity 被 enable_intensity 短路）
                    if c.evidence_a_id == c.evidence_b_id:
                        assert c.intensity == 3
                    else:
                        assert c.intensity == 2
                if not aspect:
                    assert c.matched_aspect is None
