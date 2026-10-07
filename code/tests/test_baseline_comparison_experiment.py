"""
对照基线端到端对比实验（run_baseline_comparison.py）单元测试。

测试策略：不依赖报告 JSON，直接在小规模 L1/L1.5 案例池上调用
evaluate_systems / check_red_lines，验证：
1. 四系统指标结构完整、计数自洽；
2. 完整 ConsensusCourt 在 L1+L1.5 上 recall=1.0 / FPR=0.0；
3. 红线检查在标准种子案例池上全部满足（与冒烟脚本同口径）。

运行：backend$ .venv/bin/pytest tests/test_baseline_comparison_experiment.py -v
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

import run_baseline_comparison as rbc  # noqa: E402
from tests.ddb.l1_generator import generate as generate_l1  # noqa: E402
from tests.ddb.l15_generator import generate as generate_l15  # noqa: E402

#: 四个被测系统的注册名（COURT_NAME 与三基线 key）
SYSTEM_NAMES = [
    rbc.COURT_NAME, "majority_vote", "elo_arbiter", "single_judge",
]


@pytest.fixture(scope="module")
def small_pool():
    """小案例池：L1 12×2 + L1.5 每模板 2 例（22 例），模块级复用。"""
    cases = generate_l1(
        num_with_contradiction=12, num_no_contradiction=12, seed=42
    )
    cases.extend(generate_l15(per_template=2, seed=42))
    return cases


@pytest.fixture(scope="module")
def standard_pool():
    """与冒烟脚本同口径的标准池：L1 60×2（红线判定需要稳定分布）。"""
    return generate_l1(
        num_with_contradiction=60, num_no_contradiction=60, seed=42
    )


class TestEvaluateSystemsStructure:
    """U13.1 指标结构与计数自洽。"""

    def test_four_systems_present(self, small_pool):
        results = rbc.evaluate_systems(small_pool)
        assert set(results.keys()) == set(SYSTEM_NAMES)

    def test_metric_blocks_complete(self, small_pool):
        results = rbc.evaluate_systems(small_pool)
        for name in SYSTEM_NAMES:
            for block in ("reject", "non_support"):
                m = results[name][block]
                for key in (
                    "n", "n_positive", "n_negative",
                    "recall", "fpr", "accuracy", "precision",
                ):
                    assert key in m
            assert "decision_distribution" in results[name]

    def test_counts_conserve(self, small_pool):
        """每个系统的混淆计数总和等于案例池大小，正负例计数四系统一致。"""
        results = rbc.evaluate_systems(small_pool)
        n_total = len(small_pool)
        n_pos = sum(1 for c in small_pool if c["polarity"] == "with_contradiction")
        for name in SYSTEM_NAMES:
            m = results[name]["reject"]
            assert m["n"] == n_total
            assert m["n_positive"] == n_pos
            assert m["n_negative"] == n_total - n_pos

    def test_court_two_mappings_identical(self, small_pool):
        """ConsensusCourt 二元输出下 reject 与 non_support 口径必须完全一致。"""
        results = rbc.evaluate_systems(small_pool)
        court = results[rbc.COURT_NAME]
        assert court["reject"] == court["non_support"]

    def test_non_support_more_permissive_for_baselines(self, small_pool):
        """非支持口径（含'有条件支持'）对基线只能拦下更多，不能更少。"""
        results = rbc.evaluate_systems(small_pool)
        for name in ("majority_vote", "elo_arbiter", "single_judge"):
            rej = results[name]["reject"]
            ns = results[name]["non_support"]
            # 非支持口径的 TP+FP（拦下数）≥ 严格驳回口径
            flagged_rej = _flagged(rej)
            flagged_ns = _flagged(ns)
            assert flagged_ns >= flagged_rej


def _flagged(m: dict) -> int:
    """由率值块反算拦下数（TP+FP）。"""
    return round(m["recall"] * m["n_positive"] + m["fpr"] * m["n_negative"])


class TestCourtPerfectSeparation:
    """U13.2 完整法庭在 L1+L1.5 上零漏检零误伤。"""

    def test_court_perfect_on_l1_l15(self, small_pool):
        results = rbc.evaluate_systems(small_pool)
        court = results[rbc.COURT_NAME]["reject"]
        assert court["recall"] == 1.0
        assert court["fpr"] == 0.0
        assert court["accuracy"] == 1.0
        assert court["precision"] == 1.0


class TestRedLines:
    """U13.3 红线检查在标准池（60×2）上全部满足。"""

    def test_no_violations_standard_pool(self, standard_pool):
        results = rbc.evaluate_systems(standard_pool)
        violations = rbc.check_red_lines(results)
        assert violations == []

    def test_baselines_cannot_match_court_recall(self, standard_pool):
        """核心动机可证伪条款：三基线召回必须严格低于完整法庭。"""
        results = rbc.evaluate_systems(standard_pool)
        court_recall = results[rbc.COURT_NAME]["reject"]["recall"]
        for name in ("majority_vote", "elo_arbiter", "single_judge"):
            assert results[name]["reject"]["recall"] < court_recall

    def test_red_line_detects_perfect_baseline(self):
        """红线检查必须能捕获'基线追平法庭'的伪造结果（检查器有效性）。"""
        fake = {
            rbc.COURT_NAME: {"reject": _rates(1.0, 0.0, 10, 10)},
            "majority_vote": {"reject": _rates(1.0, 0.0, 10, 10)},
        }
        violations = rbc.check_red_lines(fake)
        assert any("recall" in v for v in violations)


def _rates(recall: float, fpr: float, n_pos: int, n_neg: int) -> dict:
    """构造红线检查器输入用的率值块（仅用于检查器有效性测试）。"""
    tp = round(recall * n_pos)
    fn = n_pos - tp
    fp = round(fpr * n_neg)
    tn = n_neg - fp
    return {
        "recall": recall, "fpr": fpr,
        "accuracy": round((tp + tn) / (n_pos + n_neg), 4),
        "precision": recall, "n": n_pos + n_neg,
        "n_positive": n_pos, "n_negative": n_neg,
    }


class TestWilsonConfidenceIntervals:
    """U13.4 Wilson 95% 置信区间（v0.4.16 评审 P1-6a）：函数正确性 + 字段结构锁。"""

    def test_wilson_perfect_recall_n60(self):
        """已知值锁：k=60, n=60（法庭召回）→ [0.9398, 1.0]，与报告 JSON 一致。"""
        lo, hi = rbc.wilson_interval(60, 60)
        assert lo == pytest.approx(0.9398, abs=5e-4)
        assert hi == pytest.approx(1.0)

    def test_wilson_perfect_recall_n10(self):
        """已知值锁：k=10, n=10（L2 层召回）→ 下界 0.7225，论文 H2b 注记取 0.72。"""
        lo, hi = rbc.wilson_interval(10, 10)
        assert lo == pytest.approx(0.7225, abs=5e-4)
        assert hi == pytest.approx(1.0)

    def test_wilson_zero_fpr_n148(self):
        """已知值锁：k=0, n=148（法庭 FPR）→ [0.0, 0.0253]。"""
        lo, hi = rbc.wilson_interval(0, 148)
        assert lo == 0.0
        assert hi == pytest.approx(0.0253, abs=5e-4)

    def test_wilson_zero_n_returns_none(self):
        """n=0（负例集为空等退化情形）→ None，不抛异常。"""
        assert rbc.wilson_interval(0, 0) is None

    def test_wilson_rejects_k_out_of_range(self):
        """k>n 或负数属调用方 bug，必须显式抛错而非静默产出错误区间。"""
        with pytest.raises(ValueError):
            rbc.wilson_interval(61, 60)
        with pytest.raises(ValueError):
            rbc.wilson_interval(-1, 10)

    def test_metric_blocks_carry_ci_fields(self, small_pool):
        """结构锁：reject/non_support 率值块都带 recall_ci95/fpr_ci95 字段。"""
        results = rbc.evaluate_systems(small_pool)
        for name in SYSTEM_NAMES:
            for block in ("reject", "non_support"):
                m = results[name][block]
                for key in ("recall_ci95", "fpr_ci95"):
                    assert key in m, f"{name}/{block} 缺 {key}"
                    ci = m[key]
                    # 空负例集等退化情形允许 None，否则必须是 [lo, hi] 二元组
                    if ci is not None:
                        assert len(ci) == 2
                        lo, hi = ci
                        assert 0.0 <= lo <= hi <= 1.0

    def test_court_ci_consistent_with_point_estimate(self, small_pool):
        """一致性锁：点估计必须落在 Wilson 区间内（区间构造的必要条件）。"""
        results = rbc.evaluate_systems(small_pool)
        for name in SYSTEM_NAMES:
            m = results[name]["reject"]
            lo_r, hi_r = m["recall_ci95"]
            assert lo_r <= m["recall"] <= hi_r
            if m["fpr_ci95"] is not None:
                lo_f, hi_f = m["fpr_ci95"]
                assert lo_f <= m["fpr"] <= hi_f
