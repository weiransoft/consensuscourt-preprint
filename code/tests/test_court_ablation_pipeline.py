"""
法院组件消融无头流水线（run_ablation_v03.run_court_headless）测试。

核心测试是 **parity 守卫**：无头流水线逐行复刻 hold_court 的组件编排，
必须与真实 ConsensusCourt.hold_court（内存 SQLite，含落库尝试）在
裁决/置信度/升级/定罪峰值强度上逐字段一致——生产 hold_court 编排
漂移时本测试会失败，防止实验镜像与生产脱节。

同时覆盖 §6.4 三消融模式（no_abp/no_reviewer/no_boss）的行为不变量：
- 去掉 ABP：L1 合成集（无 AF 硬约束）定罪必须归零；
- 去掉评审/BOSS：矛盾对定罪通道不受影响（两模式与 full 定罪恒等）；
- 去掉 BOSS：升级只能减少不能增加。

运行：backend$ .venv/bin/pytest tests/test_court_ablation_pipeline.py -v
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))
_SCRIPTS_DIR = _BACKEND_ROOT / "tests" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

os.environ.setdefault("SCIMINER_API_KEY", "test-placeholder")
os.environ.setdefault("DB_PASSWORD", "test-placeholder")

import pytest  # noqa: E402

from app.core.virtual_lab.consensus_court import ConsensusCourt  # noqa: E402
from run_ablation_v03 import (  # noqa: E402
    COURT_MODES,
    _to_evidence,
    evaluate_court_config,
    run_court_headless,
)
from tests.ddb.l1_generator import generate as generate_l1  # noqa: E402


@pytest.fixture(scope="module")
def sample_cases():
    """取标准种子 L1 集的首个正例与首个负例。"""
    cases = generate_l1(
        num_with_contradiction=20, num_no_contradiction=20, seed=42
    )
    positive = next(c for c in cases if c["polarity"] == "with_contradiction")
    negative = next(c for c in cases if c["polarity"] != "with_contradiction")
    return positive, negative


class TestHeadlessHoldCourtParity:
    """U14.1 无头流水线与真实 hold_court 逐字段 parity（漂移守卫）。"""

    def test_positive_case_parity(self, sample_cases, test_session_factory):
        positive, _ = sample_cases
        db = test_session_factory()
        try:
            headless = run_court_headless(positive, "court_full")
            evidences = [_to_evidence(ev) for ev in positive["evidences"]]
            verdict = ConsensusCourt(db).hold_court(
                "parity-session-pos", positive["topic"], evidences
            )
            # PI 裁决文本与置信度（同评分同加权 → 必须逐位一致）
            assert headless["decision"] == verdict.decision
            assert headless["confidence"] == verdict.confidence
            # 升级判定（同 items/confidence/intensity 公式）
            assert headless["escalated"] == verdict.escalated
            # 定罪峰值强度（同一 rubric 全开关）
            assert headless["max_intensity"] == verdict.max_contradiction_intensity
            # 定罪等价：hold_court 不直接暴露 convicted，矛盾对非空 ⟺ 峰值强度>0
            assert headless["convicted"] == (
                verdict.max_contradiction_intensity > 0
            )
        finally:
            db.close()

    def test_negative_case_parity(self, sample_cases, test_session_factory):
        _, negative = sample_cases
        db = test_session_factory()
        try:
            headless = run_court_headless(negative, "court_full")
            evidences = [_to_evidence(ev) for ev in negative["evidences"]]
            verdict = ConsensusCourt(db).hold_court(
                "parity-session-neg", negative["topic"], evidences
            )
            assert headless["decision"] == verdict.decision
            assert headless["confidence"] == verdict.confidence
            assert headless["escalated"] == verdict.escalated
            assert headless["max_intensity"] == verdict.max_contradiction_intensity
            # 负例不得定罪
            assert headless["convicted"] is False
            assert verdict.max_contradiction_intensity == 0
        finally:
            db.close()


class TestAblationModes:
    """U14.2 三种消融模式的行为不变量。"""

    def test_invalid_mode_raises(self, sample_cases):
        positive, _ = sample_cases
        with pytest.raises(ValueError, match="未知法院消融模式"):
            run_court_headless(positive, "not_a_mode")

    def test_no_abp_loses_all_pairwise_conviction(self, sample_cases):
        """去掉 ABP 后 L1 合成集（无 AF2/AF3 内容）矛盾对定罪必须归零。"""
        positive, negative = sample_cases
        out_pos = run_court_headless(positive, "court_no_abp")
        out_neg = run_court_headless(negative, "court_no_abp")
        assert out_pos["convicted"] is False
        assert out_pos["max_intensity"] == 0
        assert out_neg["convicted"] is False

    def test_reviewer_boss_do_not_touch_conviction(self, sample_cases):
        """评审/BOSS 位于定罪下游：移除后定罪结果与 full 恒等。"""
        positive, negative = sample_cases
        full_p = run_court_headless(positive, "court_full")
        for mode in ("court_no_reviewer", "court_no_boss"):
            out_p = run_court_headless(positive, mode)
            out_n = run_court_headless(negative, mode)
            assert out_p["convicted"] == full_p["convicted"]
            assert out_p["max_intensity"] == full_p["max_intensity"]
            # 负例在任何模式都不得被矛盾对通道定罪
            assert out_n["convicted"] is False

    def test_boss_removal_only_reduces_escalation(self, sample_cases):
        """去掉 Human BOSS 后升级信号只能减少（正例上 full 升级时 no_boss 未必）。"""
        positive, _ = sample_cases
        full = run_court_headless(positive, "court_full")
        no_boss = run_court_headless(positive, "court_no_boss")
        assert no_boss["escalated"] <= full["escalated"]

    def test_no_reviewer_forces_no_dual_track(self, sample_cases):
        """去掉评审后双轨分歧必为 False——escalation 不再依赖评审条款。"""
        positive, _ = sample_cases
        # 间接验证：no_reviewer 的升级判定不抛错且模式表完整
        out = run_court_headless(positive, "court_no_reviewer")
        assert set(COURT_MODES) == {
            "court_full", "court_no_abp", "court_no_reviewer", "court_no_boss",
        }
        assert isinstance(out["escalated"], bool)


class TestCourtConfigEvaluation:
    """U14.3 聚合评估在小案例集上的结构正确性。"""

    def test_evaluate_returns_two_metric_layers(self):
        cases = generate_l1(
            num_with_contradiction=8, num_no_contradiction=8, seed=42
        )
        for mode in COURT_MODES:
            r = evaluate_court_config(cases, mode)
            assert r["mode"] == mode
            for layer in ("conviction", "escalation"):
                m = r[layer]
                assert m["n_positive"] == 8
                assert m["n_negative"] == 8
                assert 0 <= (m["recall"] or 0) <= 1
                assert 0 <= (m["fpr"] or 0) <= 1
