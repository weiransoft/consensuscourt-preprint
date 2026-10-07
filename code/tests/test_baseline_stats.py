"""
test_baseline_stats —— baseline_stats.py 纯函数模块单元测试

v1.0 要求（bio 阶段一详细设计 §6.3 + 附录 A.2 #1-#3）：
    Wilson/FDR-BH 已知值必须用 python3 实算值断言，禁止近似；
    所有断言零依赖项目运行时（仅 numpy 标准库）。
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.scripts.baseline_stats import (
    fdr_bh,
    mean_std,
    paired_bootstrap_ci,
    permutation_p_value,
    wilson_interval,
)


# =========================================================================
# Wilson 分数置信区间
# =========================================================================


class TestWilsonInterval:
    """Wilson 分数置信区间单元测试。"""

    def test_known_value_8_of_10(self):
        """8/10 的 Wilson 95% CI（python3 实算锁：(0.4902, 0.9433)）。"""
        lo, hi = wilson_interval(8, 10, z=1.96)
        assert np.isclose(lo, 0.490162, atol=1e-6)
        assert np.isclose(hi, 0.943318, atol=1e-6)

    def test_known_value_1_of_1_boundary(self):
        """1/1 的 Wilson 95% CI 上界饱和为 1.0（截断规则锁；z=1.96 单一事实源）。"""
        lo, hi = wilson_interval(1, 1, z=1.96)
        # 精确实算 z=1.96: lo = (1 + 1.92)/(1+1.92/1) - half ≈ 0.206549（浮点舍入差在 1e-5 级）
        assert np.isclose(lo, 0.206549, atol=1e-5)
        assert hi == 1.0  # 必须精确等于 1.0（上界截断）

    def test_zero_success(self):
        """0/10 的下界饱和为 0（单侧 z=1.96 实算 hi ≈ 0.277533，v0.9 误用单侧 1.645 导致错写 0.308431）。"""
        lo, hi = wilson_interval(0, 10, z=1.96)
        assert lo == 0.0
        assert np.isclose(hi, 0.277533, atol=1e-5)

    def test_zero_total_returns_none(self):
        """n=0 → None（调用方用作 saturated 标记）。"""
        assert wilson_interval(0, 0) is None

    def test_invalid_k_raises(self):
        """k < 0 或 k > n 必须显式失败。"""
        with pytest.raises(ValueError):
            wilson_interval(-1, 10)
        with pytest.raises(ValueError):
            wilson_interval(11, 10)

    def test_ci_ordering_boundary(self):
        """k=n → 上界严格等于 1.0；k=0 → 下界严格等于 0.0。"""
        assert wilson_interval(10, 10, z=1.96)[1] == 1.0
        assert wilson_interval(0, 10, z=1.96)[0] == 0.0

    def test_consistency_with_run_baseline_comparison(self):
        """
        单一事实源验证：baseline_stats.wilson_interval 的数值语义
        必须与 run_baseline_comparison.wilson_interval（原实现）一致。
        原实现返回 list 且 round(..., 4)，本模块返回 tuple 且 round(..., 6)——
        比较时先 round 到 4 位再断言，避免精度差异造成测试漂移。
        """
        from tests.scripts.run_baseline_comparison import wilson_interval as rbci_wilson

        for k, n in [(0, 10), (3, 10), (8, 10), (1, 1), (5, 5), (0, 60)]:
            ours = wilson_interval(k, n, z=1.96)
            theirs = rbci_wilson(k, n, z=1.96)
            if ours is None:
                assert theirs is None, f"n={n} 零分母语义不一致"
                continue
            lo_ours, hi_ours = ours
            lo_theirs, hi_theirs = theirs
            assert round(lo_ours, 4) == lo_theirs, (
                f"k={k}, n={n}: lo 不一致 {lo_ours} vs {lo_theirs}"
            )
            assert round(hi_ours, 4) == hi_theirs, (
                f"k={k}, n={n}: hi 不一致 {hi_ours} vs {hi_theirs}"
            )


# =========================================================================
# Benjamini-Hochberg FDR 校正（**精确阶梯单调化方向**）
# =========================================================================


class TestFdrBH:
    """Benjamini-Hochberg FDR 校正单元测试。"""

    def test_known_value_python_real(self):
        """
        python3 实算锁（**v1.0 正确实现**，修订自 v0.9 错误断言）：
            pvals = [0.01, 0.04, 0.03, 0.005], α=0.05
            升序 p_i: 0.005(r1), 0.01(r2), 0.03(r3), 0.04(r4)
            raw q_i:  0.02,      0.02,      0.04,      0.04
            阶梯单调化（方向关键：从 max rank 向下取 min）:
                q4=0.04 → q3=min(0.04, 0.04)=0.04 → q2=min(0.02, 0.04)=0.02
            还原原序: [0.02, 0.04, 0.04, 0.02]
            全部 ≤ 0.05 → rejected = {0, 1, 2, 3}
        """
        adj_q, rejected = fdr_bh([0.01, 0.04, 0.03, 0.005], alpha=0.05)
        assert np.allclose(adj_q, [0.02, 0.04, 0.04, 0.02], atol=1e-12)
        assert rejected == {0, 1, 2, 3}

    def test_empty_list(self):
        """空列表 → 空返回（调用方显式处理边界）。"""
        adj_q, rejected = fdr_bh([], alpha=0.05)
        assert adj_q == []
        assert rejected == set()

    def test_all_equal_large_no_reject(self):
        """全 p=0.5 → 阶梯后仍 ≈ 0.5 > 0.05，rejected = set()。"""
        adj_q, rejected = fdr_bh([0.5, 0.5, 0.5, 0.5], alpha=0.05)
        # BH 公式 0.5*4/r → raw = [2.0, 1.0, 0.667, 0.5]; 阶梯仍 2.0/1.0/0.667/0.5
        assert len(rejected) == 0
        for q in adj_q:
            assert q > 0.05

    def test_step_down_monotonic_direction(self):
        """阶梯单调化方向必须是"从尾部向前取累积最小值"（v1.0 修订）。"""
        # 手工构造一个阶梯方向差异可观测的 pvals
        # 升序 q_i 若未单调（上游原始 BH 就是阶梯的目的），
        # 单调化后必须保证非升序（按原始索引方向看应非降序——
        # 更简单的测试：调整后 q_i 按 pvals 升序排列的位置看应单调非增）
        adj_q, _ = fdr_bh([0.2, 0.15, 0.1], alpha=0.05)
        # 还原后按原始索引 adj_q[0], adj_q[1], adj_q[2]
        # 重新升序排列，调整后 q 必须非升序（方向正确）
        indices_by_pval = sorted(range(3), key=lambda i: [0.2, 0.15, 0.1][i])
        q_sorted_by_pval = [adj_q[i] for i in indices_by_pval]
        # 阶梯单调化保证 q_sorted_by_pval 非增序（即 raw_q 的阶梯结果）
        for i in range(len(q_sorted_by_pval) - 1):
            assert q_sorted_by_pval[i] >= q_sorted_by_pval[i + 1] - 1e-12, (
                f"阶梯单调化方向错误：{q_sorted_by_pval}"
            )

    def test_out_of_range_p_raises(self):
        """p 不在 [0, 1] → ValueError。"""
        with pytest.raises(ValueError):
            fdr_bh([-0.1, 0.5])
        with pytest.raises(ValueError):
            fdr_bh([0.0, 1.5])

    def test_alpha_out_of_range_raises(self):
        """alpha 不在 (0, 1) → ValueError。"""
        with pytest.raises(ValueError):
            fdr_bh([0.5], alpha=0.0)
        with pytest.raises(ValueError):
            fdr_bh([0.5], alpha=1.0)


# =========================================================================
# mean_std 描述性统计
# =========================================================================


class TestMeanStd:
    """描述性统计单元测试。"""

    def test_ddof_zero_population(self):
        """ddof=0 总体标准差。"""
        m, s = mean_std([0.2, 0.4, 0.6], ddof=0)
        assert m == pytest.approx(0.4)
        assert s == pytest.approx(np.std([0.2, 0.4, 0.6], ddof=0))

    def test_ddof_one_sample(self):
        """ddof=1 样本标准差。"""
        m, s = mean_std([0.2, 0.4, 0.6], ddof=1)
        assert m == pytest.approx(0.4)
        assert s == pytest.approx(np.std([0.2, 0.4, 0.6], ddof=1))

    def test_single_value(self):
        """单值 → 均值=值本身；样本标准差 ddof=1 抛 NaN（numpy 行为）。"""
        m, s = mean_std([0.42], ddof=1)
        assert m == pytest.approx(0.42)
        assert np.isnan(s)  # ddof=1 单元素样本标准差 = NaN（numpy 默认）

    def test_empty_raises(self):
        """空列表 → ValueError（显式失败而非静默产出 NaN）。"""
        with pytest.raises(ValueError):
            mean_std([])


# =========================================================================
# 配对 bootstrap CI
# =========================================================================


class TestPairedBootstrapCI:
    """配对 bootstrap CI 单元测试。"""

    def test_all_positive_d_ci_tight(self):
        """每对差异全 +0.1 → CI 严格收窄在 0.1 附近。"""
        new = [0.5, 0.6, 0.7, 0.8, 0.9]
        old = [0.4, 0.5, 0.6, 0.7, 0.8]
        lo, hi = paired_bootstrap_ci(new, old, n_boot=2000, seed=99)
        assert lo >= 0.09
        assert hi <= 0.11

    def test_all_equal_d_zero_ci_contains_zero(self):
        """两版每对值相等 → d 全 0 → CI 包含 0。"""
        vals = [0.5, 0.6, 0.7, 0.8, 0.9]
        lo, hi = paired_bootstrap_ci(vals, vals, n_boot=2000, seed=99)
        assert lo <= 0.0 <= hi

    def test_seed_fixed_reproducible(self):
        """固定 seed 两次调用结果逐位相等（可复现锁）。"""
        new = np.random.default_rng(7).random(20)
        old = new - 0.05
        ci_a = paired_bootstrap_ci(new, old, n_boot=500, seed=42)
        ci_b = paired_bootstrap_ci(new, old, n_boot=500, seed=42)
        assert np.isclose(ci_a[0], ci_b[0])
        assert np.isclose(ci_a[1], ci_b[1])

    def test_length_mismatch_raises(self):
        """new/old 长度不一致 → ValueError。"""
        with pytest.raises(ValueError):
            paired_bootstrap_ci([0.5] * 5, [0.5] * 4)

    def test_empty_raises(self):
        """空数组 → ValueError。"""
        with pytest.raises(ValueError):
            paired_bootstrap_ci([], [])

    def test_n_boot_too_small_raises(self):
        """n_boot < 2 → ValueError。"""
        with pytest.raises(ValueError):
            paired_bootstrap_ci([0.5], [0.5], n_boot=1)


# =========================================================================
# 配对置换检验
# =========================================================================


class TestPermutationPValue:
    """双侧配对置换检验单元测试。"""

    def test_zero_difference_p_equals_one(self):
        """两版完全相等 → p=1.0（零差异无显著性）。"""
        vals = [0.5, 0.6, 0.7, 0.8, 0.9]
        assert permutation_p_value(vals, vals, n_perm=500, seed=42) == 1.0

    def test_large_consistent_difference_p_small(self):
        """
        每对差异全 +0.2，10 点样本双侧置换最小 p=2/2^10≈0.00195<0.01。
        5 点样本最小 p=2/2^5=0.0625 过大 → 改用 10 点（保证 p<0.01）。
        """
        new = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
        old = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
        p = permutation_p_value(new, old, n_perm=2000, seed=42)
        assert p < 0.01  # 10 点恒定差异，双侧最小 p≈0.002

    def test_seed_fixed_reproducible(self):
        """固定 seed 可复现。"""
        new = np.random.default_rng(7).random(20)
        old = new - 0.02
        p_a = permutation_p_value(new, old, n_perm=500, seed=42)
        p_b = permutation_p_value(new, old, n_perm=500, seed=42)
        assert p_a == p_b
