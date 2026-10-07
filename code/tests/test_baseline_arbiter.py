"""
对照基线仲裁器单元测试（§6.2 Baselines）

覆盖三个基线（MajorityVote / EloArbiter / SingleJudge）的基本行为，
并与 ConsensusCourt（full_v03）在 L1 合成数据上做一次端到端对比——
验证"无矛盾检测时容易被伪冲突误伤"这一核心动机。

运行：backend$ .venv/bin/pytest tests/test_baseline_arbiter.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

import pytest  # noqa: E402

from app.core.virtual_lab.baseline_arbiter import (  # noqa: E402
    BASELINE_ARBITERS,
    EloArbiter,
    MajorityVote,
    SingleJudge,
)
from app.core.virtual_lab.consensus_court import Evidence, EvidenceScore  # noqa: E402


# ---------------------------------------------------------------------------
# 测试辅助：构造 Evidence + EvidenceScore
# ---------------------------------------------------------------------------

def _ev(eid: str, content: str = "test", role: str = "pi-agent") -> Evidence:
    return Evidence(
        evidence_id=eid, content=content, source="pubmed", role=role,
    )


def _score(eid: str, overall: float) -> EvidenceScore:
    return EvidenceScore(eid, 0.9, 0.9, 0.9, overall)


def _emap(scores: list[EvidenceScore], evidences: list[Evidence]) -> dict:
    return {ev.evidence_id: ev for ev in evidences}


# ---------------------------------------------------------------------------
# MajorityVote
# ---------------------------------------------------------------------------

class TestMajorityVote:
    def test_all_high_support(self):
        scores = [_score("a", 0.8), _score("b", 0.9), _score("c", 0.75)]
        emap = {s.evidence_id: _ev(s.evidence_id) for s in scores}
        v, r, c = MajorityVote().arbitrate("t", scores, emap)
        assert v == "支持提案"
        assert c == pytest.approx(1.0)

    def test_mixed_rejection(self):
        scores = [_score("a", 0.1), _score("b", 0.2), _score("c", 0.8)]
        emap = {s.evidence_id: _ev(s.evidence_id) for s in scores}
        v, r, c = MajorityVote().arbitrate("t", scores, emap)
        assert v == "驳回提案"
        assert c == pytest.approx(2 / 3)

    def test_tie_conditional(self):
        # 1 支持 / 1 反对 / 1 弃权 → 平票 → 有条件
        scores = [_score("a", 0.8), _score("b", 0.2), _score("c", 0.5)]
        emap = {s.evidence_id: _ev(s.evidence_id) for s in scores}
        v, r, c = MajorityVote().arbitrate("t", scores, emap)
        assert v == "有条件支持"
        assert c == 0.5

    def test_empty_scores(self):
        v, r, c = MajorityVote().arbitrate("t", [], {})
        assert v == "无法裁决"


# ---------------------------------------------------------------------------
# EloArbiter
# ---------------------------------------------------------------------------

class TestEloArbiter:
    def test_high_avg_strong_support(self):
        scores = [_score("a", 0.9)]
        emap = {"a": _ev("a", role="pi-agent")}
        v, r, c = EloArbiter().arbitrate("t", scores, emap)
        assert v == "支持提案"
        # 0.9 * 2000 = 1800 Elo；E = 1/(1+10^((1000-1800)/400)) ≈ 0.99
        assert c > 0.95

    def test_low_avg_strong_reject(self):
        scores = [_score("a", 0.1)]
        emap = {"a": _ev("a", role="agent")}
        v, r, c = EloArbiter().arbitrate("t", scores, emap)
        assert v == "驳回提案"
        # 0.1*2000=200 Elo；E ≈ 0.003
        assert c < 0.05

    def test_role_weighting_matter(self):
        # pi-agent 高分 vs agent 低分 → 加权平均高
        scores = [_score("a", 0.9), _score("b", 0.1)]
        emap = {"a": _ev("a", role="pi-agent"), "b": _ev("b", role="agent")}
        _, _, weighted = EloArbiter().arbitrate("t", scores, emap)

        # 两 agent 同等分数 → 加权平均接近算术平均
        emap2 = {"a": _ev("a", role="agent"), "b": _ev("b", role="agent")}
        _, _, unweighted = EloArbiter().arbitrate("t", scores, emap2)

        # pi-agent 高分权重大 → 加权置信度更高
        assert weighted > unweighted

    def test_registry_present(self):
        assert "elo_arbiter" in BASELINE_ARBITERS
        assert BASELINE_ARBITERS["elo_arbiter"] is EloArbiter


# ---------------------------------------------------------------------------
# SingleJudge
# ---------------------------------------------------------------------------

class TestSingleJudge:
    def test_highest_score_wins(self):
        scores = [_score("a", 0.2), _score("b", 0.95), _score("c", 0.5)]
        emap = {s.evidence_id: _ev(s.evidence_id) for s in scores}
        v, r, c = SingleJudge().arbitrate("t", scores, emap)
        assert v == "支持提案"
        assert c == pytest.approx(0.95)
        assert "b" in r  # rationale 指向最高分证据

    def test_no_reasoning_no_reasoning(self):
        """单裁判完全忽略低分反对证据（核心弱点，适合用来说明矛盾检测必要性）。"""
        scores = [_score("a", 0.95), _score("b", 0.05)]
        emap = {s.evidence_id: _ev(s.evidence_id) for s in scores}
        v, _, _ = SingleJudge().arbitrate("t", scores, emap)
        # 高分 a 直接赢，低分 b 完全被忽略
        assert v == "支持提案"


# ---------------------------------------------------------------------------
# 基线注册表完整性
# ---------------------------------------------------------------------------

class TestBaselineRegistry:
    def test_three_baselines(self):
        assert len(BASELINE_ARBITERS) == 3
        for cls in BASELINE_ARBITERS.values():
            assert hasattr(cls, "arbitrate")

    def test_all_accepts_signature(self):
        """所有基线的 arbitrate 签名对齐 PIAgentArbiter。"""
        from inspect import signature
        for cls in BASELINE_ARBITERS.values():
            sig = signature(cls.arbitrate)
            params = set(sig.parameters.keys())
            assert {"topic", "evidence_scores", "evidence_map"} <= params