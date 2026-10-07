"""
DDB-L4 强度标注层生成器单元测试（U11 / U12）

对应 design/l4-intensity-annotation-spec.md §2 与大纲 §6.1 L4：
- U11 生成完整性锁：L4 基准可复现生成（固定 seed）、类别覆盖齐全、
      gold 标签与类别绑定干净、SAME_OBJECT_POOL 专名切面存活；
- U12 全开配置一致性锁：v0.3 三机制全开下每个正例必须定罪且强度
      == gold_intensity（分类别 match=全部、mismatch=0）。

运行：backend$ .venv/bin/pytest tests/ddb/test_l4_generator.py -q
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
from app.core.virtual_lab.rubric_engine import AsymmetricRubric  # noqa: E402
from tests.ddb.l4_generator import (  # noqa: E402
    CATEGORY_GOLD,
    L4_DEFAULT_ROUNDS,
    L4_DEFAULT_SEED,
    SAME_OBJECT_POOL,
    UNKNOWN_OBJECT_POOL,
    generate,
    gold_intensity,
)


# ---------------------------------------------------------------------------
# 案例 dict → Evidence（与 run_l4_smoke._to_evidence 同构）
# ---------------------------------------------------------------------------

def _to_evidence(ev: dict) -> Evidence:
    """把 L4 案例证据字典无损还原为 consensus_court.Evidence 对象。"""
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
    在单个 L4 案例上运行指定开关组合的 ABP，返回 (是否定罪, 矛盾对列表)。

    Args:
        case: L4 案例字典
        **switches: AsymmetricRubric 三开关（enable_* 系列）

    Returns:
        (not verdict.passed, verdict.contradictions)
    """
    scorer = EvidenceScorer()
    rubric = AsymmetricRubric(**switches)
    evidences = [_to_evidence(ev) for ev in case["evidences"]]
    scores = [scorer.score(ev, case["topic"]) for ev in evidences]
    emap = {ev.evidence_id: ev for ev in evidences}
    verdict = rubric.evaluate(scores, emap, case["topic"])
    return (not verdict.passed), verdict.contradictions


# ---------------------------------------------------------------------------
# 基准缓存（模块级共享，固定 seed=42 与落盘 l4_cases.json 同源）
# ---------------------------------------------------------------------------

_CASES_CACHE: Optional[list[dict]] = None


def _cases() -> list[dict]:
    """加载 L4 基准（默认规模、固定种子，模块级缓存）。"""
    global _CASES_CACHE
    if _CASES_CACHE is None:
        _CASES_CACHE = generate(rounds=L4_DEFAULT_ROUNDS, seed=L4_DEFAULT_SEED)
    return _CASES_CACHE


# ===========================================================================
# U11 生成完整性锁
# ===========================================================================

class TestL4GenerationIntegrity:
    """生成期完整性：可复现、类别齐全、gold 标签干净。"""

    def test_l4_001_deterministic_reproducible(self):
        """同 seed 两次生成必须逐字段一致（基准可复现红线）。
        时间戳基于 utcnow 实时值——复现比较前把时间戳归一化，
        其余全部字段（来源/角色/引用 ID/content/强度/元数据）必须一致。"""
        first = [_normalize(c) for c in generate(rounds=1, seed=7)]
        second = [_normalize(c) for c in generate(rounds=1, seed=7)]
        assert first == second, "固定 seed 下生成结果不一致，基准不可复现"

    def test_l4_002_default_scale_and_categories(self):
        """默认规模 = len(TEMPLATES)×rounds×4 且类别覆盖齐全。"""
        cases = _cases()
        # 默认 rounds=4，类别 = gap_unknown/gap_same/nogap_unknown/nogap_same
        cats = {c["category"] for c in cases}
        assert cats == set(CATEGORY_GOLD), (
            f"类别不齐: {cats} vs {set(CATEGORY_GOLD)}")
        # 每类别例数相同（= 模板数 × rounds）
        counts = {cat: sum(1 for c in cases if c["category"] == cat)
                  for cat in CATEGORY_GOLD}
        per_cat = list(counts.values())
        assert len(set(per_cat)) == 1, f"类别分布不均: {counts}"

    def test_l4_003_gold_binding_clean(self):
        """每个案例 ground_truth 标签干净：正例恒有矛盾、强度=类别 gold。"""
        for case in _cases():
            gt = case["ground_truth"]
            assert gt["has_true_contradiction"] is True
            assert gt["gold_pair"] == ["EV-A", "EV-B"]
            cat = case["category"]
            assert gt["gold_intensity"] == CATEGORY_GOLD[cat], (
                f"{case['case_id']} gold={gt['gold_intensity']} "
                f"期望 {CATEGORY_GOLD[cat]}（类别绑定破坏）")

    def test_l4_004_intensity_gold_formula(self):
        """gold_intensity 判定表与类别绑定一致：directional 基线 2、
        gap → S3 +1 → 3；same → S4 提升至 3；nogap_unknown → 恒 2。"""
        for case in _cases():
            gt = case["ground_truth"]
            cat = case["category"]
            # 重算 gold（与 case 内 gold 对照）
            src_a = case["evidences"][0]["source"]
            src_b = case["evidences"][1]["source"]
            pair = tuple(gt["target_pair_spec"])
            gate = gt["gate_expectation"]
            recalc, _ = gold_intensity(pair, src_a, src_b, gate)
            assert recalc == gt["gold_intensity"] == CATEGORY_GOLD[cat], (
                f"{case['case_id']} 重算 gold={recalc} "
                f"记录={gt['gold_intensity']} 类别期望={CATEGORY_GOLD[cat]}")

    def test_l4_005_source_delta_correct(self):
        """source_delta 必须等于 |w_a - w_b| 且 Δ 桶正确。"""
        for case in _cases():
            gt = case["ground_truth"]
            w_a = EvidenceScorer.SOURCE_WEIGHTS.get(
                case["evidences"][0]["source"].lower())
            w_b = EvidenceScorer.SOURCE_WEIGHTS.get(
                case["evidences"][1]["source"].lower())
            delta = round(abs(w_a - w_b), 4)
            assert gt["source_delta"] == delta
            cat = case["category"]
            if "gap" in cat and "nogap" not in cat:
                assert delta >= 0.30, f"{case['case_id']} gap 桶 Δ={delta}"
            elif "nogap" in cat:
                assert delta < 0.30, f"{case['case_id']} nogap 桶 Δ={delta}"

    def test_l4_006_gate_bucket_gate_expectation(self):
        """gate_bucket 必须与 gate_expectation 一致：same→SAME_OBJECT、
        unknown→UNKNOWN。"""
        for case in _cases():
            bucket = case["metadata"]["gate_bucket"]
            expect = case["ground_truth"]["gate_expectation"]
            if bucket == "same":
                assert expect == "SAME_OBJECT"
            else:
                assert expect == "UNKNOWN"

    def test_l4_007_same_object_pool_survival(self):
        """SAME_OBJECT_POOL 专名切面存活自检等价验证：每个专名
        在全部模板完整语料上均使词对存活（模块加载即检查等价复算）。"""
        from tests.ddb.l4_generator import _full_aspect_hit, _spec_of, TEMPLATES
        from app.core.virtual_lab.rubric_engine import Aspect
        for obj_word in SAME_OBJECT_POOL:
            for topic, side_a, side_b, pair in TEMPLATES:
                spec = _spec_of(pair)
                if Aspect.UNKNOWN in spec.aspects:
                    continue  # UNKNOWN 通配恒存活
                neg_form = (spec.keyword_b if spec.negated_b
                            else "不" + spec.keyword_a)
                rendered = (side_a.replace("{TPN}", neg_form)
                            .replace("{TP}", pair[0])
                            .replace("{OBJ}", obj_word)
                            .replace("{FILL}", ""))
                full = _full_aspect_hit(f"{rendered} {topic}")
                assert spec.aspects & full, (
                    f"same 轮专名切面不存活: {obj_word} [{topic}] {pair}")

    def test_l4_008_unknown_object_pool_zero_signal(self):
        """UNKNOWN_OBJECT_POOL 专名必须零词典信号（门禁零信号前提）。"""
        from app.core.virtual_lab.rubric_engine import ObjectExtractor
        extractor = ObjectExtractor()
        offenders = []
        for token in UNKNOWN_OBJECT_POOL:
            ev = Evidence(evidence_id="UKN", source="", content=token,
                          role="", metadata={})
            if extractor.extract(ev, ""):
                offenders.append(token)
        assert not offenders, f"零信号专名命中对象词典: {offenders}"


# ===========================================================================
# U12 全开配置一致性锁（强度判定表全分支可达）
# ===========================================================================

class TestL4FullOpenIntensity:
    """v0.3 全开配置在 L4 正例集上的定罪率与强度一致性。"""

    def test_l4_010_full_open_convicts_all(self):
        """全开配置下每个正例必须定罪（has_true_contradiction=True）。"""
        misses = []
        for case in _cases():
            ok, _ = _rubric_case(case)
            if not ok:
                misses.append(case["case_id"])
        assert not misses, (
            f"全开配置漏检（{len(misses)}/{len(_cases())} 例）: {misses[:5]}")

    def test_l4_010b_full_open_intensity_match_all(self):
        """全开配置下定罪强度必须 == gold_intensity（mismatch=0 红线）。"""
        mismatches = []
        for case in _cases():
            ok, contradictions = _rubric_case(case)
            if not ok:
                continue  # 漏检已上锁
            gold = case["ground_truth"]["gold_intensity"]
            gold_pair = tuple(case["ground_truth"]["gold_pair"])
            located = None
            for c in contradictions:
                pair = tuple(sorted([c.evidence_a_id, c.evidence_b_id]))
                if pair == tuple(sorted(gold_pair)):
                    located = c
                    break
            assert located is not None, (
                f"{case['case_id']} gold_pair 未被 ABP 记录")
            if located.intensity != gold:
                mismatches.append(
                    f"{case['case_id']}: gold={gold} "
                    f"输出={located.intensity}")
        assert not mismatches, (
            f"强度一致性违例（{len(mismatches)} 例）: {mismatches[:5]}")

    def test_l4_011_intensity_distribution_matches_gold(self):
        """全开配置强度分布必须与 gold 分布一致
        （gap_unknown/gap_same/nogap_same→3、nogap_unknown→2）。"""
        # gold 分布
        gold_dist: dict[int, int] = {}
        for case in _cases():
            g = case["ground_truth"]["gold_intensity"]
            gold_dist[g] = gold_dist.get(g, 0) + 1
        # ABP 全开输出分布
        abp_dist: dict[int, int] = {}
        for case in _cases():
            ok, contradictions = _rubric_case(case)
            if not ok or not contradictions:
                continue
            for c in contradictions:
                abp_dist[c.intensity] = abp_dist.get(c.intensity, 0) + 1
        assert abp_dist == gold_dist, (
            f"强度分布不一致: ABP={abp_dist} gold={gold_dist}")


# ===========================================================================
# 消融冒烟（三开关 8 组合可运行性 + 结构不变量）
# ===========================================================================

class TestL4AblationSmoke:
    """三开关 8 组合消融冒烟：正例集上全部可运行且不变量成立。"""

    @pytest.mark.parametrize("aspect", [True, False])
    @pytest.mark.parametrize("gate", [True, False])
    @pytest.mark.parametrize("intensity", [True, False])
    def test_all_switch_combinations_run(self, aspect, gate, intensity):
        """8 种开关组合在全部 L4 案例上运行无异常且不变量成立。

        不变量：
        - intensity 关闭 → 定罪矛盾对 intensity 恒 2；
        - gate 关闭 → pending_pairs 恒空；开启 → 全部 PENDING 严重级；
        - aspect 关闭 → matched_aspect 恒 None；
        - 任意开关组合下正例不得判无罪（词对极性命中在所有配置下保留）。
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


# ---------------------------------------------------------------------------
# 内部辅助：时间戳归一化（test_l4_001 专用）
# ---------------------------------------------------------------------------

def _normalize(case: dict) -> dict:
    """非空时间戳归一化为布尔占位（utcnow 实时值不可逐字段比对）。"""
    normalized = {k: v for k, v in case.items() if k != "evidences"}
    normalized["evidences"] = [
        {**ev, "timestamp": ev["timestamp"] is not None}
        for ev in case["evidences"]
    ]
    return normalized
