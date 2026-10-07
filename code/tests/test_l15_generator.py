"""
DDB-L1.5 负例对照基准测试（U8 / U9 / U10）

对应大纲 v0.4.3 §6.1 L1.5 负例对照层与 §5.3 定理 1-A 的 P(S) 估计需求：
- U8  生成完整性锁：负例基准可复现生成（固定 seed）、全量 validate
      通过、类别覆盖齐全、ground_truth 标签干净
- U9  全开配置 FPR 锁：v0.3 三机制全开下，每个负例案例必须判无罪
      （FPR=0）——aspect / 门禁两类拦截面在案例级全部生效；
      分机制一阶效应：仅关 aspect（cross_aspect 类必须出现定罪漂移）、
      仅关门禁（different_object/ambiguous 类必须出现漂移）
- U10 全关差分锁（v0.2 行为等价面）：三开关全关下每个负例必须被判
      定罪（expected_v02=convicted 恒成立）——证明负例的"鉴别力"：
      这些案例正是 v0.3 机制要拦截、而 v0.2 会误定罪的对象

运行：backend$ .venv/bin/pytest tests/ddb/test_l15_generator.py -q
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
)
from app.core.virtual_lab.rubric_engine import (  # noqa: E402
    Aspect,
    AspectInferrer,
    AsymmetricRubric,
)
from tests.ddb.l15_generator import (  # noqa: E402
    L15_NEUTRAL_SENTENCES,
    DdbL15Generator,
    generate,
)


# ---------------------------------------------------------------------------
# 案例 dict → Evidence（与 test_l2_v03_locks._to_evidence 同构）
# ---------------------------------------------------------------------------

def _to_evidence(ev: dict) -> Evidence:
    """把 L1.5 案例证据字典无损还原为 consensus_court.Evidence 对象。"""
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
    在单个 L1.5 案例上运行指定开关组合的 ABP，返回 (是否定罪, 矛盾对列表)。

    Args:
        case: L1.5 案例字典
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
# 基准缓存（模块级共享，固定 seed=42 与落盘 l15_cases.json 同源）
# ---------------------------------------------------------------------------

_CASES_CACHE: Optional[list[dict]] = None


def _cases() -> list[dict]:
    """加载 L1.5 负例基准（默认规模、固定种子，模块级缓存）。"""
    global _CASES_CACHE
    if _CASES_CACHE is None:
        _CASES_CACHE = generate(per_template=8, seed=42)
    return _CASES_CACHE


# ===========================================================================
# U8 生成完整性锁
# ===========================================================================

class TestL15GenerationIntegrity:
    """生成期完整性：可复现、全量校验通过、标签干净。"""

    def test_l15_001_deterministic_reproducible(self):
        """同 seed 两次生成必须逐字段一致（基准可复现红线）。
        时间戳基于 utcnow 实时值——复现比较前把非空时间戳归一化，
        其余全部采样字段（来源/角色/干扰数量与位置/引用 ID）必须一致。"""

        def _normalize(case: dict) -> dict:
            normalized = {k: v for k, v in case.items() if k != "evidences"}
            normalized["evidences"] = [
                {**ev, "timestamp": ev["timestamp"] is not None}
                for ev in case["evidences"]
            ]
            return normalized

        first = [_normalize(c) for c in generate(per_template=3, seed=7)]
        second = [_normalize(c) for c in generate(per_template=3, seed=7)]
        assert first == second, "固定 seed 下生成结果不一致，基准不可复现"

    def test_l15_002_full_validate_pass(self):
        """默认规模全量 validate 通过且类别覆盖齐全。"""
        cases = _cases()
        DdbL15Generator.validate(cases)
        categories = {c["category"] for c in cases}
        assert {"cross_aspect", "different_object_gate",
                "ambiguous_reference"} <= categories

    def test_l15_003_ground_truth_labels_clean(self):
        """每个案例 ground_truth 标签干净：负例恒无罪、期望标签齐全。
        ambiguous_reference 类的全开期望是 pending_ambiguous（AMBIGUOUS
        降待举证），其余类别是 no_conviction。"""
        for case in _cases():
            gt = case["ground_truth"]
            assert gt["has_true_contradiction"] is False
            assert gt["defect_only"] is False
            assert gt["mechanism"] in ("aspect", "validity_gate")
            expected = (
                "pending_ambiguous"
                if case["category"] == "ambiguous_reference"
                else "no_conviction"
            )
            assert gt["expected_full"] == expected
            assert gt["expected_v02"] == "convicted"

    def test_l15_004_neutral_sentences_zero_object(self):
        """中性干扰句全池必须零对象词典命中（L15_NEUTRAL_ZERO_OBJECT）。"""
        extractor = DdbL15Generator(seed=1)._extractor  # 借用其抽取器实例
        offenders: list[str] = []
        for sentence in L15_NEUTRAL_SENTENCES:
            ev = Evidence(evidence_id="NEU", source="", content=sentence, role="")
            if extractor.extract(ev, ""):
                offenders.append(sentence[:40])
        assert not offenders, f"中性句命中对象词典（纪律破坏）: {offenders}"

    def test_l15_005_aspect_lexicon_coverage(self):
        """词典覆盖锁：每个案例的核心证据（EV01/EV02）切面推断不得
        全 UNKNOWN（aspect/门禁拦截面依赖词典对句面术语的覆盖）。"""
        inf = AspectInferrer()
        gaps: list[str] = []
        for case in _cases():
            by_id = {ev["evidence_id"]: ev for ev in case["evidences"]}
            core_ids = sorted(
                eid for eid in by_id if eid.endswith(("EV01", "EV02"))
            )
            aspects = [
                inf.infer(_to_evidence(by_id[eid]), case["topic"])
                for eid in core_ids
            ]
            if all(a == frozenset({Aspect.UNKNOWN}) for a in aspects):
                gaps.append(case["case_id"])
        assert not gaps, f"Aspect 词典对核心句覆盖缺口: {gaps}"


# ===========================================================================
# U9 全开配置 FPR 锁（拦截面生效性）
# ===========================================================================

class TestL15FullOpenFPR:
    """v0.3 全开配置在 L1.5 负例集上的 FPR 与一阶机制效应。"""

    def test_l15_010_full_open_fpr_zero(self):
        """FPR=0 红线：全开配置下每个负例必须判无罪。"""
        convicted: list[str] = []
        for case in _cases():
            ok, _ = _rubric_case(case)
            if ok:
                convicted.append(case["case_id"])
        assert not convicted, f"v0.3 全开误定罪（FPR>0）: {convicted}"

    def test_l15_011_aspect_mechanism_effect(self):
        """aspect 一阶效应：cross_aspect 类在仅关 aspect 时必须出现
        定罪漂移（证明拦截面确由 aspect 贡献，而非门禁顺带拦截）。"""
        cross = [c for c in _cases() if c["category"] == "cross_aspect"]
        assert cross, "cross_aspect 类别缺失，一阶效应不可测"
        drifted = [
            c["case_id"] for c in cross
            if _rubric_case(
                c, enable_aspect=False,
                enable_validity_gate=True, enable_intensity=True)[0]
        ]
        # 全部 cross 案例都应从"无罪"漂移为"定罪"（aspect 独占拦截面）
        assert len(drifted) == len(cross), (
            f"cross_aspect 仅关 aspect 后仅 {len(drifted)}/{len(cross)} 例"
            f"定罪，aspect 拦截面归属不纯: {cross[0]['case_id']} 起未漂移"
        )

    def test_l15_012_gate_mechanism_effect(self):
        """门禁一阶效应：different_object / ambiguous 两类在仅关门禁
        时必须全部出现定罪漂移（门禁独占拦截面）。"""
        gated = [c for c in _cases()
                 if c["category"] in ("different_object_gate",
                                      "ambiguous_reference")]
        assert gated, "门禁类类别缺失，一阶效应不可测"
        drifted = [
            c["case_id"] for c in gated
            if _rubric_case(
                c, enable_aspect=True,
                enable_validity_gate=False, enable_intensity=True)[0]
        ]
        assert len(drifted) == len(gated), (
            f"门禁类仅关门禁后仅 {len(drifted)}/{len(gated)} 例定罪，"
            f"门禁拦截面归属不纯"
        )


# ===========================================================================
# U10 全关差分锁（v0.2 行为等价面 = 负例鉴别力）
# ===========================================================================

class TestL15FullClosedDifferential:
    """三开关全关（≈v0.2 行为）必须把全部负例误定罪——鉴别力锁。"""

    def test_l15_020_full_closed_convicts_all(self):
        """每个负例在全关配置必须定罪（expected_v02=convicted）。"""
        missed: list[str] = []
        for case in _cases():
            ok, _ = _rubric_case(
                case, enable_aspect=False,
                enable_validity_gate=False, enable_intensity=False)
            if not ok:
                missed.append(f"{case['case_id']}({case['category']})")
        assert not missed, f"全关配置漏定罪（负例鉴别力缺口）: {missed}"

    def test_l15_021_open_closed_gap_full(self):
        """机制效应差分锁：全关定罪数 - 全开定罪数 = 全部案例数
        （L1.5 集上 v0.3 机制的贡献面 = 100% 误定罪被拦截）。"""
        cases = _cases()
        n_closed = sum(
            1 for c in cases
            if _rubric_case(c, enable_aspect=False,
                            enable_validity_gate=False,
                            enable_intensity=False)[0])
        n_open = sum(1 for c in cases if _rubric_case(c)[0])
        assert n_open == 0 and n_closed == len(cases), (
            f"差分不成立：全开定罪 {n_open}（期望 0）、"
            f"全关定罪 {n_closed}/{len(cases)}（期望全部）"
        )


# ===========================================================================
# 消融冒烟（§6.4 自变量 8 组合可运行性，与 L2 冒烟同构）
# ===========================================================================

class TestL15AblationSmoke:
    """三开关 8 组合消融冒烟：负例集上全部可运行且结构不变量成立。"""

    @pytest.mark.parametrize("aspect", [True, False])
    @pytest.mark.parametrize("gate", [True, False])
    @pytest.mark.parametrize("intensity", [True, False])
    def test_all_switch_combinations_run(self, aspect, gate, intensity):
        """8 种开关组合在全部 L1.5 案例上运行无异常且不变量成立。

        不变量（与 test_l2_v03_locks 消融冒烟一致）：
        - gate 关闭 → pending_pairs 恒空；开启 → 全部 PENDING 严重级；
        - intensity 关闭 → 定罪矛盾对 intensity 恒 2；
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

            for p in verdict.pending_pairs:
                assert p.severity.value == "PENDING"
            if not gate:
                assert verdict.pending_pairs == []
            for c in verdict.contradictions:
                assert c.intensity in (1, 2, 3)
                if not intensity:
                    assert c.intensity == 2
                if not aspect:
                    assert c.matched_aspect is None
