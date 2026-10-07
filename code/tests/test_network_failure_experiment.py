#!/usr/bin/env python3
"""
网络失败区分实验单元测试（论文大纲 §6.5，v0.4.14，U18）。

覆盖范围：
1. FlakyPubMedTransport 注入器：失败概率语义（p=0 恒成功 / p=1 恒失败 /
   中间确定性）、valid / 不存在两种应答语义、传输层计数器；
2. 案例生成器：同 seed 确定性、A 层引用全有效 / B 层引用确认不存在、
   共享热点池跨案重用（缓存效应基础）；
3. 二态直拒基线映射：network_failed / invalid 一律定罪、valid 不定罪
   （判别设计，防"直拒映射恒真/恒假"退化）；
4. 策略格点核心函数（小案例集端到端）：三态 FPR=0（含全断网）、
   二态直拒全断网 FPR=1.0、B 层 invalid → 两策略一致定罪、
   格点双跑确定性；
5. 红线检查器：六条红线逐一伪造违例反测（检查器有效性）+ 真实格点全过；
6. run_experiment 端到端（缩小规模）：报告结构完整 + 红线全过。

纪律：无业务 mock——网络层模拟使用 httpx 官方 MockTransport 传输层接口
（失败注入即实验处理本身），验证/重试/缓存/映射逻辑全部走真实实现。
"""
from __future__ import annotations

import asyncio
import random

import httpx
import pytest

from tests.scripts.run_network_failure_experiment import (
    POLICIES,
    POLICY_THREE_STATE,
    POLICY_TWO_STATE,
    CitationCase,
    FlakyPubMedTransport,
    _rng_for_cell,
    check_red_lines,
    generate_cases,
    run_experiment,
    run_policy_cell,
    two_state_direct_mapping,
)

_PUBMED_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


def _make_request(pmid: str) -> httpx.Request:
    """构造指向 PubMed esummary 端点的 GET 请求（注入器单测用）。"""
    return httpx.Request(
        "GET", _PUBMED_URL,
        params={"db": "pubmed", "id": pmid, "retmode": "json"},
    )


# ============================================================================
# 1. FlakyPubMedTransport 注入器
# ============================================================================

class TestFlakyPubMedTransport:
    """确定性失败注入器的概率语义、应答语义与计数器测试。"""

    @pytest.mark.asyncio
    async def test_p0_all_success(self):
        """p=0 无故障基准：全部请求成功应答，无失败计数。"""
        transport = FlakyPubMedTransport(0.0, {"88000001"}, random.Random(42))
        for _ in range(10):
            resp = await transport(_make_request("88000001"))
            assert resp.status_code == 200
        assert transport.total_requests == 10
        assert transport.failed_requests == 0
        assert transport.resolved_requests == 10

    @pytest.mark.asyncio
    async def test_p1_all_fail(self):
        """p=1 全断网：每次请求都注入 ConnectError（重试逻辑的上游输入）。"""
        transport = FlakyPubMedTransport(1.0, {"88000001"}, random.Random(42))
        for _ in range(10):
            with pytest.raises(httpx.ConnectError):
                await transport(_make_request("88000001"))
        assert transport.total_requests == 10
        assert transport.failed_requests == 10
        assert transport.resolved_requests == 0

    def test_deterministic_same_seed(self):
        """同 seed 注入器失败判定序列完全一致（逐请求可复现）。"""
        outcomes_a = []
        ta = FlakyPubMedTransport(0.5, set(), random.Random(7))
        for _ in range(50):
            try:
                asyncio.run(ta(_make_request("88000001")))
                outcomes_a.append("ok")
            except httpx.ConnectError:
                outcomes_a.append("fail")

        outcomes_b = []
        tb = FlakyPubMedTransport(0.5, set(), random.Random(7))
        for _ in range(50):
            try:
                asyncio.run(tb(_make_request("88000001")))
                outcomes_b.append("ok")
            except httpx.ConnectError:
                outcomes_b.append("fail")

        assert outcomes_a == outcomes_b
        # 中间失败率下两种结果都应出现（概率语义非退化）
        assert "ok" in outcomes_a and "fail" in outcomes_a

    @pytest.mark.asyncio
    async def test_valid_pmid_response_semantics(self):
        """有效 PMID 应答 valid 语义：result 含该 PMID 且无 error 键。"""
        transport = FlakyPubMedTransport(0.0, {"88000001"}, random.Random(42))
        resp = await transport(_make_request("88000001"))
        data = resp.json()
        assert "88000001" in data["result"]
        assert "error" not in data["result"]["88000001"]

    @pytest.mark.asyncio
    async def test_invalid_pmid_response_semantics(self):
        """无效 PMID 应答"不存在"语义：result 为空 → 上游判 invalid。"""
        transport = FlakyPubMedTransport(0.0, {"88000001"}, random.Random(42))
        resp = await transport(_make_request("99000001"))
        data = resp.json()
        assert data["result"] == {}

    @pytest.mark.asyncio
    async def test_unexpected_endpoint_treated_as_failure(self):
        """非 PubMed 端点请求按网络失败处理（实验只授权单一端点）。"""
        transport = FlakyPubMedTransport(0.0, set(), random.Random(42))
        req = httpx.Request("GET", "https://www.ebi.ac.uk/chembl/api/data/chembl1.json")
        with pytest.raises(httpx.ConnectError):
            await transport(req)


# ============================================================================
# 2. 案例生成器
# ============================================================================

class TestGenerateCases:
    """A/B 两层案例生成器的确定性、真值正确性与共享池重用测试。"""

    def test_deterministic_same_seed(self):
        """同 seed 双跑案例集逐案例相同。"""
        cases_a, valid_a, invalid_a = generate_cases(42, 10, 4)
        cases_b, valid_b, invalid_b = generate_cases(42, 10, 4)
        assert [(c.case_id, c.claim_text, c.declared_refs, c.stratum) for c in cases_a] \
            == [(c.case_id, c.claim_text, c.declared_refs, c.stratum) for c in cases_b]
        assert valid_a == valid_b and invalid_a == invalid_b

    def test_clean_stratum_refs_all_valid(self):
        """A 层：引用全部来自有效池，每案 2-4 条（共享池无放回抽取）。"""
        cases, valid_pool, _ = generate_cases(42, 20, 0)
        clean = [c for c in cases if c.stratum == "clean"]
        assert len(clean) == 20
        for c in clean:
            assert 2 <= len(c.declared_refs) <= 4
            assert len(set(c.declared_refs)) == len(c.declared_refs)  # 案内无重复
            assert all(r in valid_pool for r in c.declared_refs)

    def test_invalid_stratum_refs_disjoint_from_valid(self):
        """B 层：引用确认不存在，与有效池零交集（真值正确性）。"""
        cases, valid_pool, invalid_pool = generate_cases(42, 5, 6)
        bad = [c for c in cases if c.stratum == "invalid_ref"]
        assert len(bad) == 6
        for c in bad:
            assert all(r not in valid_pool for r in c.declared_refs)
            assert all(r in invalid_pool for r in c.declared_refs)

    def test_shared_pool_reuse(self):
        """共享热点池跨案重用：总引用查找次数 > 去重后 rid 数（缓存效应基础）。"""
        cases, _, _ = generate_cases(42, 30, 0)
        total_refs = sum(len(c.declared_refs) for c in cases)
        unique_rids = {r for c in cases for r in c.declared_refs}
        assert total_refs > len(unique_rids)


# ============================================================================
# 3. 二态直拒基线映射（社区现状对照，判别设计）
# ============================================================================

class TestTwoStateDirectMapping:
    """二态直拒映射的判别测试：非 valid 一律定罪、valid 不定罪。"""

    def test_network_failed_convicted(self):
        """network_failed（网络失败）被直拒定罪——与生产三态的核心分歧点。"""
        results = {"123": {"status": "network_failed", "valid": False}}
        assert two_state_direct_mapping(results) is True

    def test_invalid_convicted(self):
        """invalid（确认不存在）被定罪——与生产三态一致的部分。"""
        results = {"123": {"status": "invalid", "valid": False}}
        assert two_state_direct_mapping(results) is True

    def test_format_invalid_convicted(self):
        """format_invalid（格式问题）也被直拒定罪——二态压缩语义的体现。"""
        results = {"abc": {"status": "format_invalid", "valid": False}}
        assert two_state_direct_mapping(results) is True

    def test_valid_not_convicted(self):
        """全部引用 valid → 不定罪（防恒真退化）。"""
        results = {"123": {"status": "valid", "valid": True}}
        assert two_state_direct_mapping(results) is False

    def test_empty_results_not_convicted(self):
        """空结果映射（无引用声明）→ 不定罪（边界一致性）。"""
        assert two_state_direct_mapping({}) is False


# ============================================================================
# 4. 策略格点核心函数（小案例集端到端）
# ============================================================================

def _mini_cases() -> tuple[list[CitationCase], set[str]]:
    """构造迷你案例集（3 干净 + 2 无效引用），供格点端到端测试。"""
    cases, valid_pmids, _ = generate_cases(42, 3, 2)
    return cases, valid_pmids


class TestPolicyCells:
    """（策略 × 失败率）格点执行器的端到端行为测试。"""

    @pytest.mark.asyncio
    async def test_three_state_fpr_zero_under_full_outage(self):
        """三态策略全断网（p=1.0）：A 层 FPR=0（降级 WARNING 不定罪）。"""
        cases, valid_pmids = _mini_cases()
        cell = await run_policy_cell(POLICY_THREE_STATE, 1.0, cases, valid_pmids, 42)
        assert cell["fpr_clean"] == 0.0
        assert cell["convicted_clean"] == 0
        # 全断网下全部引用查找 3 次尝试后降级 network_failed
        assert cell["network_failed_rate"] == 1.0
        assert cell["avg_attempts"] == 3.0

    @pytest.mark.asyncio
    async def test_three_state_full_outage_abstains_on_b(self):
        """三态策略全断网：B 层召回 0（弃权而非误判——设计语义断言）。"""
        cases, valid_pmids = _mini_cases()
        cell = await run_policy_cell(POLICY_THREE_STATE, 1.0, cases, valid_pmids, 42)
        assert cell["recall_invalid"] == 0.0

    @pytest.mark.asyncio
    async def test_two_state_full_outage_pollutes_all(self):
        """二态直拒全断网：A 层 FPR=1.0（基础设施故障污染的上界）。"""
        cases, valid_pmids = _mini_cases()
        cell = await run_policy_cell(POLICY_TWO_STATE, 1.0, cases, valid_pmids, 42)
        assert cell["fpr_clean"] == 1.0
        assert cell["recall_invalid"] == 1.0

    @pytest.mark.asyncio
    async def test_b_stratum_convicted_both_policies_no_failure(self):
        """p=0 无故障：B 层确认不存在引用在两策略下都被定罪（召回 1.0）。"""
        cases, valid_pmids = _mini_cases()
        for policy in POLICIES:
            cell = await run_policy_cell(policy, 0.0, cases, valid_pmids, 42)
            assert cell["recall_invalid"] == 1.0, f"策略 {policy} B 层召回非 1.0"
            assert cell["fpr_clean"] == 0.0, f"策略 {policy} p=0 出现假阳性"

    @pytest.mark.asyncio
    async def test_cell_metrics_consistency(self):
        """格点指标内部一致性：裁决分布计数与分层计数吻合。"""
        cases, valid_pmids = _mini_cases()
        cell = await run_policy_cell(POLICY_TWO_STATE, 0.5, cases, valid_pmids, 42)
        total = cell["verdict_distribution"]["convicted"] \
            + cell["verdict_distribution"]["acquitted"]
        assert total == cell["n_clean"] + cell["n_invalid"]
        assert cell["n_clean"] == 3 and cell["n_invalid"] == 2
        # 二态直拒无跨案缓存 → 缓存命中恒 0
        assert cell["cache_hits"] == 0

    @pytest.mark.asyncio
    async def test_three_state_cache_hits_across_cases(self):
        """三态策略跨案共享验证器：共享池重用产生缓存命中（结构下界>0）。"""
        cases, valid_pmids, _ = generate_cases(42, 12, 0)
        cell = await run_policy_cell(POLICY_THREE_STATE, 0.0, cases, valid_pmids, 42)
        assert cell["cache_hits"] > 0

    @pytest.mark.asyncio
    async def test_cell_determinism_same_seed(self):
        """格点双跑确定性：同 seed 同参数逐字段相等。"""
        cases, valid_pmids = _mini_cases()
        a = await run_policy_cell(POLICY_THREE_STATE, 0.5, cases, valid_pmids, 42)
        b = await run_policy_cell(POLICY_THREE_STATE, 0.5, cases, valid_pmids, 42)
        assert a == b

    @pytest.mark.asyncio
    async def test_unknown_policy_raises(self):
        """未知策略名抛 ValueError（格点矩阵完整性守卫）。"""
        cases, valid_pmids = _mini_cases()
        with pytest.raises(ValueError):
            await run_policy_cell("no_such_policy", 0.5, cases, valid_pmids, 42)


# ============================================================================
# 5. 红线检查器（伪造违例反测 + 真实格点全过）
# ============================================================================

def _fabricated_cells(**overrides) -> list[dict]:
    """构造理想通过态的格点矩阵（伪造违例反测的基线夹具）。

    默认值取自真实实验的理论预期形态（三态 FPR=0 / 二态 FPR 随 p 上升至 1.0），
    可按字段覆盖注入违例。
    """
    rates = [0.0, 0.3, 0.5, 1.0]
    cells = []
    for policy, fpr_seq, recall_seq in [
        (POLICY_THREE_STATE, [0.0, 0.0, 0.0, 0.0], [1.0, 0.95, 0.9, 0.0]),
        (POLICY_TWO_STATE, [0.0, 0.6, 0.8, 1.0], [1.0, 1.0, 1.0, 1.0]),
    ]:
        for rate, fpr, recall in zip(rates, fpr_seq, recall_seq):
            cells.append({
                "policy": policy,
                "failure_rate": rate,
                "n_clean": 60, "n_invalid": 20,
                "fpr_clean": fpr, "recall_invalid": recall,
                "convicted_clean": int(fpr * 60),
                "convicted_invalid": int(recall * 20),
                "total_ref_lookups": 180,
                "avg_attempts": 1.0 + rate * 2,
                "network_failed_rate": rate,
                "cache_hits": 100,
                "transport_total_requests": 200,
                "transport_failed_requests": int(rate * 200),
                "verdict_distribution": {"convicted": 5, "acquitted": 75},
            })
            if abs(rate - 0.5) < 1e-9:
                rerun = dict(cells[-1])
                rerun["policy"] = f"{policy}（复跑）"
                cells.append(rerun)
    return cells


class TestRedLineChecker:
    """六条预注册红线的检查器有效性测试（逐一伪造违例反测）。"""

    def test_all_pass_on_ideal_cells(self):
        """理想通过态格点矩阵：六条红线全过。"""
        results = check_red_lines(_fabricated_cells())
        assert len(results) == 6
        assert all(r["passed"] for r in results), \
            [r["name"] for r in results if not r["passed"]]

    def _run_with_override(self, policy: str, rate: float, **field_overrides) -> list[dict]:
        """在理想夹具上覆盖指定格点字段后重跑红线检查。"""
        cells = _fabricated_cells()
        for c in cells:
            if c["policy"] == policy and abs(c["failure_rate"] - rate) < 1e-9:
                c.update(field_overrides)
        return check_red_lines(cells)

    def test_rl1_forged_fpr_violation(self):
        """红线 1 反测：三态任一格点 FPR>0 必须被判违例。"""
        results = self._run_with_override(
            POLICY_THREE_STATE, 0.5, fpr_clean=0.1, convicted_clean=6,
        )
        rl1 = next(r for r in results if r["name"] == "RL1-three-state-fpr-zero")
        assert rl1["passed"] is False

    def test_rl2_forged_recall_violation(self):
        """红线 2 反测：三态 p=0.3 召回 <0.9 必须被判违例。"""
        results = self._run_with_override(
            POLICY_THREE_STATE, 0.3, recall_invalid=0.8, convicted_invalid=16,
        )
        rl2 = next(r for r in results if r["name"] == "RL2-three-state-recall-profile")
        assert rl2["passed"] is False

    def test_rl2_forged_abstention_violation(self):
        """红线 2 反测：三态全断网召回 >0（误判泄漏）必须被判违例。"""
        results = self._run_with_override(
            POLICY_THREE_STATE, 1.0, recall_invalid=0.2, convicted_invalid=4,
        )
        rl2 = next(r for r in results if r["name"] == "RL2-three-state-recall-profile")
        assert rl2["passed"] is False

    def test_rl3_forged_monotone_violation(self):
        """红线 3 反测：二态 FPR 序列非单调必须被判违例。"""
        results = self._run_with_override(
            POLICY_TWO_STATE, 0.5, fpr_clean=0.3, convicted_clean=18,
        )
        rl3 = next(r for r in results if r["name"] == "RL3-two-state-fpr-pollution-curve")
        assert rl3["passed"] is False

    def test_rl3_forged_zero_fpr_at_03(self):
        """红线 3 反测：二态 FPR(0.3)=0（无污染）必须被判违例。"""
        results = self._run_with_override(
            POLICY_TWO_STATE, 0.3, fpr_clean=0.0, convicted_clean=0,
        )
        rl3 = next(r for r in results if r["name"] == "RL3-two-state-fpr-pollution-curve")
        assert rl3["passed"] is False

    def test_rl4_forged_recall_drop(self):
        """红线 4 反测：二态任一格点召回 <1.0 必须被判违例。"""
        results = self._run_with_override(
            POLICY_TWO_STATE, 1.0, recall_invalid=0.9, convicted_invalid=18,
        )
        rl4 = next(r for r in results if r["name"] == "RL4-two-state-recall-always-one")
        assert rl4["passed"] is False

    def test_rl5_missing_rerun(self):
        """红线 5 反测：缺少 p=0.5 复跑格点必须被判违例。"""
        cells = [c for c in _fabricated_cells() if "（复跑）" not in c["policy"]]
        results = check_red_lines(cells)
        rl5 = next(r for r in results if r["name"] == "RL5-determinism-rerun")
        assert rl5["passed"] is False

    def test_rl5_forged_rerun_mismatch(self):
        """红线 5 反测：复跑格点与首次运行字段不一致必须被判违例。"""
        cells = _fabricated_cells()
        for c in cells:
            if c["policy"] == f"{POLICY_THREE_STATE}（复跑）":
                c["avg_attempts"] = 9.99
        results = check_red_lines(cells)
        rl5 = next(r for r in results if r["name"] == "RL5-determinism-rerun")
        assert rl5["passed"] is False

    def test_rl6_forged_zero_cache(self):
        """红线 6 反测：三态 p=0.3 缓存命中 0 必须被判违例。"""
        results = self._run_with_override(POLICY_THREE_STATE, 0.3, cache_hits=0)
        rl6 = next(r for r in results if r["name"] == "RL6-cache-effect")
        assert rl6["passed"] is False


# ============================================================================
# 6. run_experiment 端到端（缩小规模）
# ============================================================================

class TestRunExperimentE2E:
    """完整实验矩阵端到端（缩小规模）：报告结构 + 红线全过。"""

    @pytest.mark.asyncio
    async def test_full_matrix_structure_and_red_lines(self):
        """缩小规模全矩阵（2 策略 × 4 失败率 + 复跑）：结构完整且红线全过。"""
        report = await run_experiment(seed=42, n_clean=12, n_invalid=4)
        # 结构完整性
        assert set(report) == {"meta", "cells", "red_lines", "all_passed"}
        # 10 个格点（8 主格点 + 2 复跑）
        assert len(report["cells"]) == 10
        # 六条红线
        assert len(report["red_lines"]) == 6
        assert report["all_passed"] is True, \
            [r for r in report["red_lines"] if not r["passed"]]

    @pytest.mark.asyncio
    async def test_full_matrix_determinism(self):
        """端到端确定性：同 seed 双跑报告 cells 逐字段相等。"""
        a = await run_experiment(seed=42, n_clean=6, n_invalid=2)
        b = await run_experiment(seed=42, n_clean=6, n_invalid=2)
        assert a["cells"] == b["cells"]
