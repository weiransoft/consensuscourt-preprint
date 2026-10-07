"""
鲁棒性专项实验（run_robustness_experiments.py）单元测试。

覆盖：
- 零假设案件构造器（唯一来源、缺陷注入、长度约束）；
- 顺序置换小样本零分歧（定理 2）；
- n 扫描聚合结构与机制方向（全机制 q̂ ≤ 全关，定理 1-B）；
- log-log 标度回归的解析正确性；
- 红线检查器对各类违例的捕获能力。

统计标度律的红线（斜率区间）在脚本正式规模（300 案/格点）验证；
单元测试只做小规模结构检查 + 检查器逻辑测试，避免小样本噪声。

运行：backend$ .venv/bin/pytest tests/test_robustness_experiments.py -v
"""
from __future__ import annotations

import math
import os
import random
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

from datetime import datetime  # noqa: E402

from app.core.virtual_lab.consensus_court import Evidence  # noqa: E402

import run_robustness_experiments as rr  # noqa: E402


# ---------------------------------------------------------------------------
# 共享构造器：证据小积木（全部经真实评分器/规则引擎验证语义）
# ---------------------------------------------------------------------------

#: 审理主题（含 EGFR 对象，与 L1 正例一致）
_STREAM_TOPIC = "EGFR 抑制剂疗效争议"

#: 长内容中性文本：pubmed 来源下 overall_score≈0.76（强干净证据）
_STRONG_CONTENT = (
    "本实验在 EGFR 突变细胞系中观察到激酶活性显著上调的完整记录与充分对照"
)
#: 短内容文本：真实触发 screen_asymmetric 的 PENDING（证据内容过短）
_PENDING_CONTENT = "数据待补充"


def _make_evidence(
    eid: str,
    source: str = "pubmed",
    content: str = _STRONG_CONTENT,
    reference_ids: list[str] | None = None,
    role: str = "supporting",
) -> Evidence:
    """构造字段完整的真实 Evidence（非 mock，走真实评分/规则通道）。"""
    return Evidence(
        evidence_id=eid,
        source=source,
        content=content,
        role=role,
        # 评分器含时间新鲜度权重：固定 2024 年会使 overall 降至 0.7 以下，
        # 故夹具采用当前时间（L1 真实案件同样携带当代时间戳）。
        timestamp=datetime.utcnow(),
        reference_ids=reference_ids if reference_ids is not None else [f"PMID-{eid}"],
        metadata={},
    )


def _strong_evidence(eid: str) -> Evidence:
    """强干净证据：pubmed 长内容，overall_score≥0.7，无 PENDING。"""
    return _make_evidence(eid, source="pubmed", content=_STRONG_CONTENT)


def _weak_evidence(eid: str) -> Evidence:
    """弱证据：llm 来源同内容 overall_score≈0.6（打断强证据连续计数）。"""
    return _make_evidence(eid, source="llm", content=_STRONG_CONTENT)


def _pending_evidence(eid: str) -> Evidence:
    """缺陷证据：短内容触发 PENDING（overall_score≈0.56）。"""
    return _make_evidence(eid, source="pubmed", content=_PENDING_CONTENT)


def _good_perm_fixture() -> dict:
    """检查器测试用：零分歧的实验 A 结果。"""
    return {"total_permutations": 1000, "divergent_permutations": 0}


def _good_scan_fixture() -> dict:
    """检查器测试用：全部 B 红线满足的 n 扫描结果。"""
    return {
        "rows": [
            {
                "m": m, "p0_hat": 0.30, "q_hat_base": 0.035,
                "q_hat_full": 0.005, "mean_false_pair_base": 0.035 * c,
                "ratio_over_bound": 1.0,
            }
            for m, c in [(3, 3), (5, 10), (10, 45), (20, 190)]
        ],
        "scaling": {
            "slope_false_pairs_vs_Cm2": 1.0,
            "slope_ratio_vs_m_minus_1": -1.0,
        },
    }


class TestNullCaseBuilder:
    """U15.1 零假设案件构造纪律。"""

    def test_cardinality_and_unique_sources(self):
        rng = random.Random(1)
        evs = rr.build_null_case(20, case_idx=0, rng=rng)
        assert len(evs) == 20
        sources = {ev.source for ev in evs}
        assert len(sources) == 20  # 来源唯一 → 候选对恰为 C(m,2)

    def test_defect_rate_near_design(self):
        """2000 条证据大样本下缺陷率收敛到 p0=0.30（±0.03）。"""
        rng = random.Random(7)
        n_short = 0
        total = 0
        for idx in range(100):
            evs = rr.build_null_case(20, case_idx=idx, rng=rng)
            for ev in evs:
                total += 1
                if len(ev.content.strip()) < 10:
                    n_short += 1
        assert abs(n_short / total - rr.P0_DESIGNED) < 0.03

    def test_defective_evidence_carries_no_keyword(self):
        """缺陷纪律：短证据不得携带任何极性词（D_i 与词表解耦）。"""
        rng = random.Random(11)
        for idx in range(50):
            for ev in rr.build_null_case(10, case_idx=idx, rng=rng):
                if len(ev.content.strip()) < 10:
                    assert not any(
                        kw in ev.content for kw in rr.KEYWORD_POOL
                    )

    def test_nondefective_evidence_carries_exactly_one_keyword(self):
        """非缺陷证据恰好植入 1 个极性词（不植入完整对立词对）。

        唯一允许的多重命中：植入"不存在"时子串附带"存在"——这是
        极性机 _polarity_sides 明确处理的子串情形，不是额外植入。
        """
        rng = random.Random(13)
        for idx in range(20):
            for ev in rr.build_null_case(10, case_idx=idx, rng=rng):
                if len(ev.content.strip()) >= 10:
                    hits = {kw for kw in rr.KEYWORD_POOL if kw in ev.content}
                    assert hits <= {"存在", "不存在"} or len(hits) == 1
                    if hits == {"存在", "不存在"}:
                        assert "不存在" in ev.content
                    assert len(hits) >= 1

    def test_pending_counter_matches_short_content(self):
        """真实 screen_asymmetric 的 PENDING 计数与短内容注入一致。"""
        rng = random.Random(17)
        evs = rr.build_null_case(30, case_idx=99, rng=rng)
        n_short = sum(1 for ev in evs if len(ev.content.strip()) < 10)
        assert rr._count_defects_pending(evs) == n_short
        assert 0 < n_short < 30  # 30 证据下 p0=0.3，全有/全无概率极低


class TestOrderPermutation:
    """U15.2 顺序置换（定理 2）小样本。"""

    def test_zero_divergence_small_run(self):
        result = rr.run_order_permutation(total_permutations=30, seed=42)
        assert result["divergent_permutations"] == 0
        assert result["invariant_rate"] == 1.0
        assert result["total_permutations"] == 30

    def test_exact_distribution_across_cases(self):
        """总置换数恰好分配完（无丢失/重复）。"""
        result = rr.run_order_permutation(total_permutations=100, seed=42)
        counted = sum(
            cfg["permutations"]
            for case in result["per_case"]
            for cfg in case["configs"].values()
        ) // 2  # 每案两个配置各跑相同次数
        assert counted == 100


class TestNScan:
    """U15.3 n 扫描结构与机制方向。"""

    def test_scan_grid_and_rows(self):
        scan = rr.run_n_scan(trials=40, seed=42)
        assert [r["m"] for r in scan["rows"]] == list(rr.N_GRID)
        for r in scan["rows"]:
            assert r["n_pair_opportunities"] == r["m"] * (r["m"] - 1) // 2
            assert r["trials"] == 40
            assert 0.15 <= r["p0_hat"] <= 0.45
            # 全机制（aspect+门禁）错配率不得高于全关
            assert r["q_hat_full"] <= r["q_hat_base"] + 1e-9

    def test_gate_reduces_false_pairs_on_average(self):
        """4 个互异 canonical 对象下，全关错配期望必须明显高于全开。"""
        scan = rr.run_n_scan(trials=80, seed=42)
        base_total = sum(r["mean_false_pair_base"] for r in scan["rows"])
        full_total = sum(r["mean_false_pair_full"] for r in scan["rows"])
        assert full_total < base_total


class TestLogLogSlope:
    """U15.4 标度回归解析正确性。"""

    def test_perfect_power_law_slope(self):
        """y = 3·x^2 在对数坐标斜率必须 ≈ 2.0。"""
        xs = [1.0, 2.0, 4.0, 8.0]
        ys = [3.0 * x ** 2 for x in xs]
        assert rr._loglog_slope(xs, ys) == pytest.approx(2.0, abs=1e-9)

    def test_inverse_decay_slope(self):
        """y = 5/x 的斜率必须 ≈ −1.0。"""
        xs = [2.0, 4.0, 9.0, 19.0]
        ys = [5.0 / x for x in xs]
        assert rr._loglog_slope(xs, ys) == pytest.approx(-1.0, abs=1e-9)


class TestRedLineChecker:
    """U15.5 红线检查器对违例的捕获（构造输入验证检查器本身）。"""

    def _good_perm(self):
        return {"total_permutations": 1000, "divergent_permutations": 0}

    def _good_scan(self):
        return {
            "rows": [
                {
                    "m": m, "p0_hat": 0.30, "q_hat_base": 0.035,
                    "q_hat_full": 0.005, "mean_false_pair_base": 0.035 * c,
                    "ratio_over_bound": 1.0,
                }
                for m, c in [(3, 3), (5, 10), (10, 45), (20, 190)]
            ],
            "scaling": {
                "slope_false_pairs_vs_Cm2": 1.0,
                "slope_ratio_vs_m_minus_1": -1.0,
            },
        }

    def test_all_green_passes(self):
        assert rr.check_robustness_red_lines(
            self._good_perm(), self._good_scan()
        ) == []

    def test_permutation_divergence_detected(self):
        bad_perm = {"total_permutations": 1000, "divergent_permutations": 3}
        violations = rr.check_robustness_red_lines(bad_perm, self._good_scan())
        assert any("定理 2" in v for v in violations)

    def test_p0_drift_detected(self):
        scan = self._good_scan()
        scan["rows"][0]["p0_hat"] = 0.05
        violations = rr.check_robustness_red_lines(self._good_perm(), scan)
        assert any("1-B-(i)" in v for v in violations)

    def test_pair_slope_anomaly_detected(self):
        scan = self._good_scan()
        scan["scaling"]["slope_false_pairs_vs_Cm2"] = 0.3
        violations = rr.check_robustness_red_lines(self._good_perm(), scan)
        assert any("1-B-(ii)" in v for v in violations)

    def test_ratio_slope_anomaly_detected(self):
        scan = self._good_scan()
        scan["scaling"]["slope_ratio_vs_m_minus_1"] = 0.2
        violations = rr.check_robustness_red_lines(self._good_perm(), scan)
        assert any("1-B-(iii)" in v for v in violations)

    def test_mechanism_direction_violation_detected(self):
        scan = self._good_scan()
        scan["rows"][2]["q_hat_full"] = 0.10
        scan["rows"][2]["q_hat_base"] = 0.035
        violations = rr.check_robustness_red_lines(self._good_perm(), scan)
        assert any("机制方向" in v for v in violations)


class TestStreamingLockSemantics:
    """U15.6 流式即判基线的终局/锁定/重置语义（规则式单元测试）。"""

    def _engine(self):
        return rr.EvidenceScorer(), rr.RuleEngine(), rr.AsymmetricRubric(
            enable_aspect=True,
            enable_validity_gate=True,
            enable_intensity=True,
        )

    def test_support_lock_k1_on_first_strong(self):
        """k=1：首条强干净证据到达即"支持"终局（首因终局）。"""
        scorer, engine, rubric = self._engine()
        evs = [_strong_evidence("S1")]
        decision, step, reason = rr.streaming_disposition(
            evs, _STREAM_TOPIC, rubric, scorer, engine, streak_k=1,
        )
        assert decision == "支持提案"
        assert step == 1 and reason == "支持终局"

    def test_support_lock_k2_requires_two_consecutive(self):
        """k=2：两条连续强证据才终局；弱证据打断计数后不终局。"""
        scorer, engine, rubric = self._engine()
        decision, step, reason = rr.streaming_disposition(
            [_strong_evidence("S1"), _strong_evidence("S2")],
            _STREAM_TOPIC, rubric, scorer, engine, streak_k=2,
        )
        assert decision == "支持提案" and step == 2 and reason == "支持终局"
        # 强 → 弱 → 强：连续计数在弱证据处归零，耗尽后默认无罪
        decision2, step2, reason2 = rr.streaming_disposition(
            [_strong_evidence("S1"), _weak_evidence("W1"), _strong_evidence("S2")],
            _STREAM_TOPIC, rubric, scorer, engine, streak_k=2,
        )
        assert decision2 == "支持提案"
        assert step2 == 3 and reason2 == "默认无罪"

    def test_pending_blocks_support_lock(self):
        """在案 PENDING 未消解期间禁止支持终局（k=1/k=2 均被阻断）。"""
        scorer, engine, rubric = self._engine()
        for k in (1, 2):
            decision, step, reason = rr.streaming_disposition(
                [_pending_evidence("P1"), _strong_evidence("S1"),
                 _strong_evidence("S2")],
                _STREAM_TOPIC, rubric, scorer, engine, streak_k=k,
            )
            assert decision == "支持提案"
            assert step == 3 and reason == "默认无罪"

    def test_default_acquittal_on_weak_evidence(self):
        """无矛盾、无强证据：耗尽后无罪推定兜底。"""
        scorer, engine, rubric = self._engine()
        decision, step, reason = rr.streaming_disposition(
            [_weak_evidence("W1")],
            _STREAM_TOPIC, rubric, scorer, engine, streak_k=1,
        )
        assert (decision, step, reason) == ("支持提案", 1, "默认无罪")

    def test_contradiction_lock_is_earliest_prefix(self):
        """真实 L1 正例：矛盾终局步必须是首个形成矛盾对的前缀长度。"""
        scorer, engine, rubric = self._engine()
        cases = rr.generate_l1(
            num_with_contradiction=20, num_no_contradiction=20, seed=rr.SEED
        )
        positives = [c for c in cases if c["polarity"] == "with_contradiction"]
        locked = None
        for case in positives:
            evs = [rr._to_evidence(e) for e in case["evidences"]]
            decision, step, reason = rr.streaming_disposition(
                evs, case["topic"], rubric, scorer, engine, streak_k=1,
            )
            if reason == "矛盾终局":
                locked = (case, evs, step)
                break
        assert locked is not None, "20 个正例中至少存在一个矛盾终局案件"
        case, evs, step = locked
        # 终局当步确实有矛盾
        scores = [scorer.score(e, case["topic"]) for e in evs[:step]]
        emap = {e.evidence_id: e for e in evs[:step]}
        assert rubric.evaluate(scores, emap, case["topic"]).contradictions
        # 提前终止的必要性：少一条证据的前缀无矛盾
        prev_scores = [scorer.score(e, case["topic"]) for e in evs[: step - 1]]
        prev_map = {e.evidence_id: e for e in evs[: step - 1]}
        assert not rubric.evaluate(
            prev_scores, prev_map, case["topic"]
        ).contradictions

    def test_early_support_lock_can_preempt_real_contradiction(self):
        """存在正例：支持终局先于矛盾形成，但全案合议仍定罪（错放机制实证）。"""
        scorer, engine, rubric = self._engine()
        cases = rr.generate_l1(
            num_with_contradiction=60, num_no_contradiction=60, seed=rr.SEED
        )
        positives = [c for c in cases if c["polarity"] == "with_contradiction"]
        preempted = 0
        for case in positives:
            evs = [rr._to_evidence(e) for e in case["evidences"]]
            decision, _step, reason = rr.streaming_disposition(
                evs, case["topic"], rubric, scorer, engine, streak_k=1,
            )
            if decision == "支持提案":
                # 规范流程对完整证据集合议必须仍能定罪
                scores = [scorer.score(e, case["topic"]) for e in evs]
                emap = {e.evidence_id: e for e in evs}
                assert rubric.evaluate(scores, emap, case["topic"]).contradictions
                preempted += 1
        # 60 个规范序正例中应有若干例被提前支持终局（首因效应非空）
        assert preempted > 0


class TestStreamingExperiment:
    """U15.7 流式置换实验小规模端到端。"""

    def test_small_run_structure(self):
        result = rr.run_streaming_experiment(total_permutations=20, seed=43)
        assert result["n_positive_cases"] == 60
        assert result["court_recall"] == 1.0  # 规范流程全案合议 60/60
        assert set(result["configs"]) == {"k1", "k2"}
        for c in result["configs"].values():
            assert c["permutations"] == 20
            assert 0 <= c["divergence_rate"] <= 1
            assert 0 <= c["affected_cases"] <= 60
            assert 0 <= c["permuted_recall"] <= 1
        # k=2 更保守，分歧不超过 k=1
        k1, k2 = result["configs"]["k1"], result["configs"]["k2"]
        assert k2["divergent_permutations"] <= k1["divergent_permutations"]

    def test_independent_rng_stream_does_not_change_experiment_a(self):
        """实验 C 使用独立种子流；同参数两次运行结果必须逐计数一致。"""
        a = rr.run_streaming_experiment(total_permutations=20, seed=rr.SEED + 1)
        b = rr.run_streaming_experiment(total_permutations=20, seed=rr.SEED + 1)
        assert a["configs"]["k1"]["divergent_permutations"] == (
            b["configs"]["k1"]["divergent_permutations"]
        )


class TestCorruptSources:
    """U15.8 来源标签扰动器纪律：恰改 k 条、仅 source 变、原对象不改。"""

    def test_exact_victim_count_and_allowed_labels(self):
        evs = [_strong_evidence(f"E{i}") for i in range(10)]
        rng = random.Random(123)
        victims = rng.sample(range(10), max(1, round(0.2 * 10)))
        rng = random.Random(123)  # 同种子重放：与扰动器内部抽样一致
        corrupted = rr.corrupt_sources(evs, rng, ratio=0.20)
        allowed = set(rr.EvidenceScorer.SOURCE_WEIGHTS.keys())
        assert len(victims) == 2
        for i in victims:
            assert corrupted[i].source in allowed
        # 非受害位置必须与原对象同一（dataclasses.replace 只替换受害条目）
        for i in range(10):
            if i not in victims:
                assert corrupted[i] is evs[i]

    def test_only_source_field_changes(self):
        evs = [
            _make_evidence(
                f"E{i}", source=("pubmed" if i % 2 else "chembl"),
                reference_ids=[f"PMID-{i}"], role="supporting",
            )
            for i in range(10)
        ]
        rng = random.Random(99)
        victims = rng.sample(range(10), 2)
        rng = random.Random(99)
        corrupted = rr.corrupt_sources(evs, rng, ratio=0.20)
        for i in victims:
            o, n = evs[i], corrupted[i]
            assert o.content == n.content
            assert o.evidence_id == n.evidence_id
            assert o.reference_ids == n.reference_ids
            assert o.role == n.role
            assert o.timestamp == n.timestamp
            assert o.metadata == n.metadata

    def test_input_list_not_mutated(self):
        evs = [_strong_evidence(f"E{i}") for i in range(5)]
        before = [(e.evidence_id, e.source) for e in evs]
        rr.corrupt_sources(evs, random.Random(7), ratio=0.4)
        assert [(e.evidence_id, e.source) for e in evs] == before

    def test_rounding_rule_and_minimum_one(self):
        """m=6/20%→1 条（round(1.2)=1）；m=10/50%→5 条；m=3/20%→至少 1 条。"""
        for m, ratio, expected in [(10, 0.5, 5), (6, 0.2, 1), (3, 0.2, 1)]:
            evs = [_strong_evidence(f"E{i}") for i in range(m)]
            rng = random.Random(5)
            victim_set = set(rng.sample(range(m), expected))
            rng = random.Random(5)
            corrupted = rr.corrupt_sources(evs, rng, ratio=ratio)
            replaced = sum(1 for i in range(m) if corrupted[i] is not evs[i])
            assert replaced == expected
            assert len(victim_set) == expected

    def test_reproducible(self):
        evs = [_strong_evidence(f"E{i}") for i in range(10)]
        a = rr.corrupt_sources(evs, random.Random(2024), ratio=0.3)
        b = rr.corrupt_sources(evs, random.Random(2024), ratio=0.3)
        assert [e.source for e in a] == [e.source for e in b]


class TestSourceNoiseExperiment:
    """U15.9 来源噪声实验小规模端到端（结构 + 零构陷不变量）。"""

    def test_pool_split_sizes(self):
        positives, negatives = rr.build_l1_l15_pool(cases_l1=60)
        assert len(positives) == 60
        assert len(negatives) == 60 + 88
        neg_l1 = [c for c in negatives if c["case_id"].startswith("DDB-L1-")]
        neg_l15 = [c for c in negatives if c["case_id"].startswith("DDB-L15-")]
        assert len(neg_l1) == 60 and len(neg_l15) == 88

    def test_small_run_zero_false_gain_and_drift_bounds(self):
        """零构陷是结构性结论：即使 reps=2 小样本，负例误增也必须为 0。"""
        result = rr.run_source_noise_experiment(replicates=2, seed=44)
        pos, neg = result["positives"], result["negatives"]
        assert pos["replicates"] == 60 * 2
        assert neg["replicates"] == 148 * 2
        assert neg["false_gain"] == 0 and neg["false_gain_rate"] == 0
        # oracle 审计字段：零构陷下误增案件清单必须存在且为空（正例同理，
        # 基准全定罪 court_recall=1.0，不存在"基准未定罪→噪声后定罪"路径）
        assert neg["false_gain_cases"] == []
        assert pos["false_gain_cases"] == []
        assert result["negatives_l1"]["cases"] == 60
        assert result["negatives_l15"]["cases"] == 88
        assert 0.0 <= pos["noisy_recall"] <= 1.0
        # 平均漂移不超过最大漂移
        assert pos["mean_abs_confidence_drift"] <= (
            pos["max_abs_confidence_drift"] + 1e-9
        )


class TestStreamingNoiseRedLines:
    """U15.10 实验 C/D 红线检查器违例捕获（伪造输入）。"""

    def _good_streaming(self):
        return {
            "court_recall": 1.0,
            "configs": {
                "k1": {
                    "divergent_permutations": 63,
                    "divergence_rate": 0.063,
                    "permuted_recall": 0.916,
                    "canonical_recall": 0.933,
                },
                "k2": {
                    "divergent_permutations": 8,
                    "divergence_rate": 0.008,
                    "permuted_recall": 0.992,
                    "canonical_recall": 1.0,
                },
            },
        }

    def _good_noise(self):
        return {
            "negatives": {"false_gain_rate": 0},
            "positives": {"noisy_recall": 0.967, "mean_abs_confidence_drift": 0.019},
        }

    def test_cd_green_passes(self):
        assert rr.check_robustness_red_lines(
            _good_perm_fixture(), _good_scan_fixture(),
            streaming=self._good_streaming(), noise=self._good_noise(),
        ) == []

    def test_streaming_zero_divergence_detected(self):
        st = self._good_streaming()
        st["configs"]["k1"]["divergent_permutations"] = 0
        st["configs"]["k1"]["divergence_rate"] = 0.0
        violations = rr.check_robustness_red_lines(
            _good_perm_fixture(), _good_scan_fixture(), streaming=st,
        )
        assert any("实验 C" in v and "零分歧" in v for v in violations)

    def test_streaming_low_divergence_detected(self):
        st = self._good_streaming()
        st["configs"]["k1"]["divergence_rate"] = 0.01
        violations = rr.check_robustness_red_lines(
            _good_perm_fixture(), _good_scan_fixture(), streaming=st,
        )
        assert any("实验 C" in v and "0.02" in v for v in violations)

    def test_streaming_recall_not_below_court_detected(self):
        st = self._good_streaming()
        st["configs"]["k1"]["permuted_recall"] = 1.0
        violations = rr.check_robustness_red_lines(
            _good_perm_fixture(), _good_scan_fixture(), streaming=st,
        )
        assert any("实验 C" in v and "规范流程召回" in v for v in violations)

    def test_noise_false_gain_detected(self):
        nz = self._good_noise()
        nz["negatives"]["false_gain_rate"] = 0.01
        violations = rr.check_robustness_red_lines(
            _good_perm_fixture(), _good_scan_fixture(), noise=nz,
        )
        assert any("实验 D" in v and "误增" in v for v in violations)

    def test_noise_recall_drop_detected(self):
        nz = self._good_noise()
        nz["positives"]["noisy_recall"] = 0.90
        violations = rr.check_robustness_red_lines(
            _good_perm_fixture(), _good_scan_fixture(), noise=nz,
        )
        assert any("实验 D" in v and "召回" in v for v in violations)

    def test_noise_confidence_drift_detected(self):
        nz = self._good_noise()
        nz["positives"]["mean_abs_confidence_drift"] = 0.08
        violations = rr.check_robustness_red_lines(
            _good_perm_fixture(), _good_scan_fixture(), noise=nz,
        )
        assert any("实验 D" in v and "漂移" in v for v in violations)
