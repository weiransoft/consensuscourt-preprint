"""
非对称举证审查 - P0-2 改造的端到端测试

覆盖 CitationVerifier / HallucinationDetector / RuleEngine 的 P0-2 新增功能：
1. CitationVerifier：重试机制、缓存机制、status 字段区分
2. HallucinationDetector：network_failed 降级为 WARNING（避免假阳性）
3. RuleEngine：screen_asymmetric() / evaluate_with_rubric()
4. 端到端：从证据收集到裁决的完整非对称举证流程

核心验证点：
- 网络失败 ≠ 引用不存在（避免假阳性 CRITICAL）
- 内容过短 → PENDING（不直接拒绝，非对称举证）
- 找到矛盾才定罪（非对称举证核心原则）
"""
from datetime import datetime
from typing import List, Optional

import httpx
import pytest
import respx
from httpx import Response

from app.core.virtual_lab.auto_validator import (
    CitationVerifier,
    HallucinationDetector,
)
from app.core.virtual_lab.consensus_court import (
    Evidence,
    EvidenceScore,
    RuleEngine,
)
from app.core.virtual_lab.rubric_engine import (
    AsymmetricRubric,
    RubricVerdict,
    Severity,
)


# ============================================================================
# 辅助函数
# ============================================================================

def _make_evidence(
    evidence_id: str,
    content: str,
    source: str = "pubmed",
    role: str = "drug-researcher",
    reference_ids: Optional[List[str]] = None,
) -> Evidence:
    """构造测试用证据条目"""
    return Evidence(
        evidence_id=evidence_id,
        source=source,
        content=content,
        role=role,
        timestamp=datetime.utcnow(),
        reference_ids=reference_ids or [],
    )


def _make_score(
    evidence_id: str,
    overall: float = 0.8,
    issues: Optional[List[str]] = None,
) -> EvidenceScore:
    """构造测试用证据评分"""
    return EvidenceScore(
        evidence_id=evidence_id,
        source_score=0.85,
        recency_score=0.9,
        relevance_score=0.8,
        overall_score=overall,
        issues=issues or [],
    )


# ============================================================================
# CitationVerifier 重试机制测试
# ============================================================================

class TestCitationVerifierRetry:
    """CitationVerifier 重试机制测试（P0-2 核心改造）"""

    @respx.mock
    @pytest.mark.asyncio
    async def test_retry_succeeds_on_third_attempt(self):
        """前 2 次失败，第 3 次成功 → 应该返回 valid"""
        # 模拟前 2 次 500 错误，第 3 次 200 成功
        route = respx.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        ).mock(
            side_effect=[
                Response(500),  # 第 1 次失败
                Response(500),  # 第 2 次失败
                Response(200, json={"result": {"12345678": {"uid": "12345678"}}}),  # 第 3 次成功
            ]
        )
        verifier = CitationVerifier(max_retries=3)
        results = await verifier.verify(["12345678"])

        assert results["12345678"]["valid"] is True
        assert results["12345678"]["status"] == "valid"
        assert results["12345678"]["attempts"] == 3
        assert route.call_count == 3

    @respx.mock
    @pytest.mark.asyncio
    async def test_all_retries_fail_returns_network_failed(self):
        """3 次都失败 → 应该返回 network_failed（不是 invalid）"""
        route = respx.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        ).mock(return_value=Response(500))
        verifier = CitationVerifier(max_retries=3)
        results = await verifier.verify(["12345678"])

        assert results["12345678"]["valid"] is False
        assert results["12345678"]["status"] == "network_failed"
        assert results["12345678"]["attempts"] == 3
        assert route.call_count == 3

    @respx.mock
    @pytest.mark.asyncio
    async def test_retry_on_429_rate_limit(self):
        """HTTP 429（频率限制）应触发重试"""
        route = respx.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        ).mock(
            side_effect=[
                Response(429),  # 第 1 次频率限制
                Response(200, json={"result": {"12345678": {"uid": "12345678"}}}),
            ]
        )
        verifier = CitationVerifier(max_retries=3)
        results = await verifier.verify(["12345678"])

        assert results["12345678"]["status"] == "valid"
        assert results["12345678"]["attempts"] == 2
        assert route.call_count == 2

    @respx.mock
    @pytest.mark.asyncio
    async def test_retry_on_connection_error(self):
        """连接异常（如超时）应触发重试"""
        _route = respx.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        ).mock(
            side_effect=[
                httpx.ConnectError("连接超时"),
                httpx.ConnectError("连接超时"),
                Response(200, json={"result": {"12345678": {"uid": "12345678"}}}),
            ]
        )
        verifier = CitationVerifier(max_retries=3)
        results = await verifier.verify(["12345678"])

        assert results["12345678"]["status"] == "valid"
        assert results["12345678"]["attempts"] == 3

    @respx.mock
    @pytest.mark.asyncio
    async def test_no_retry_on_404_pubmed(self):
        """PubMed 404 不应触发重试（实际上是 raise_for_status 抛异常，会被当作异常处理）

        注意：PubMed 的 _check_pubmed 不像 ChEMBL 那样单独处理 404，
        而是走 raise_for_status() → 非 _NetworkError → 不重试，返回 invalid
        """
        route = respx.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        ).mock(return_value=Response(404))
        verifier = CitationVerifier(max_retries=3)
        results = await verifier.verify(["12345678"])

        # 404 走 raise_for_status → 非 _NetworkError → 不重试 → invalid
        assert results["12345678"]["status"] == "invalid"
        assert results["12345678"]["attempts"] == 1
        assert route.call_count == 1

    @respx.mock
    @pytest.mark.asyncio
    async def test_custom_max_retries(self):
        """应支持自定义最大重试次数"""
        route = respx.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        ).mock(return_value=Response(500))
        verifier = CitationVerifier(max_retries=5)
        results = await verifier.verify(["12345678"])

        assert results["12345678"]["status"] == "network_failed"
        assert results["12345678"]["attempts"] == 5
        assert route.call_count == 5

    @respx.mock
    @pytest.mark.asyncio
    async def test_default_max_retries_is_3(self):
        """默认最大重试次数应为 3"""
        assert CitationVerifier.DEFAULT_MAX_RETRIES == 3

        respx.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        ).mock(return_value=Response(500))
        verifier = CitationVerifier()  # 使用默认值
        results = await verifier.verify(["12345678"])

        assert results["12345678"]["attempts"] == 3


# ============================================================================
# CitationVerifier 缓存机制测试
# ============================================================================

class TestCitationVerifierCache:
    """CitationVerifier 缓存机制测试（P0-2 核心改造）"""

    @respx.mock
    @pytest.mark.asyncio
    async def test_cache_hit_avoids_http_request(self):
        """缓存命中时应避免重复 HTTP 请求"""
        route = respx.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        ).mock(return_value=Response(200, json={"result": {"12345678": {"uid": "12345678"}}}))
        verifier = CitationVerifier()

        # 第一次查询：发起 HTTP 请求
        results1 = await verifier.verify(["12345678"])
        assert route.call_count == 1
        assert results1["12345678"]["status"] == "valid"

        # 第二次查询：应命中缓存，不发起 HTTP 请求
        results2 = await verifier.verify(["12345678"])
        assert route.call_count == 1  # 仍然是 1，没有新增请求
        assert results2["12345678"]["status"] == "valid"

    @respx.mock
    @pytest.mark.asyncio
    async def test_cache_size_increases_with_entries(self):
        """缓存大小应随条目数增加"""
        respx.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        ).mock(return_value=Response(200, json={"result": {"12345678": {"uid": "12345678"}}}))
        respx.get("https://www.ebi.ac.uk/chembl/api/data/chembl123.json").mock(
            return_value=Response(200, json={"molecule_chembl_id": "CHEMBL123"})
        )
        verifier = CitationVerifier()
        assert verifier.get_cache_size() == 0

        await verifier.verify(["12345678"])
        assert verifier.get_cache_size() == 1

        await verifier.verify(["CHEMBL123"])
        assert verifier.get_cache_size() == 2

    @respx.mock
    @pytest.mark.asyncio
    async def test_clear_cache_resets_size(self):
        """clear_cache() 应清空缓存"""
        respx.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        ).mock(return_value=Response(200, json={"result": {"12345678": {"uid": "12345678"}}}))
        verifier = CitationVerifier()
        await verifier.verify(["12345678"])
        assert verifier.get_cache_size() == 1

        verifier.clear_cache()
        assert verifier.get_cache_size() == 0

    @respx.mock
    @pytest.mark.asyncio
    async def test_cache_stores_network_failed_results(self):
        """network_failed 结果也应被缓存（避免短时间重复失败请求）"""
        route = respx.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        ).mock(return_value=Response(500))
        verifier = CitationVerifier(max_retries=3)

        # 第一次查询：3 次重试都失败，缓存 network_failed 结果
        results1 = await verifier.verify(["12345678"])
        assert results1["12345678"]["status"] == "network_failed"
        assert route.call_count == 3

        # 第二次查询：应命中缓存，不再发起请求
        results2 = await verifier.verify(["12345678"])
        assert results2["12345678"]["status"] == "network_failed"
        assert route.call_count == 3  # 仍然是 3，没有新增请求

    @respx.mock
    @pytest.mark.asyncio
    async def test_cache_ttl_24_hours(self):
        """缓存 TTL 应为 24 小时"""
        assert CitationVerifier.CACHE_TTL_SECONDS == 24 * 60 * 60

    @respx.mock
    @pytest.mark.asyncio
    async def test_cache_expired_requeries(self):
        """缓存过期后应重新发起请求"""
        import time

        route = respx.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        ).mock(return_value=Response(200, json={"result": {"12345678": {"uid": "12345678"}}}))
        verifier = CitationVerifier()

        # 第一次查询
        await verifier.verify(["12345678"])
        assert route.call_count == 1

        # 手动将缓存时间戳设置为 25 小时前（过期）
        cached = verifier._cache["12345678"]
        result, _ = cached
        verifier._cache["12345678"] = (result, time.time() - 25 * 3600)

        # 第二次查询：缓存已过期，应重新发起请求
        await verifier.verify(["12345678"])
        assert route.call_count == 2

    @respx.mock
    @pytest.mark.asyncio
    async def test_batch_verify_uses_cache(self):
        """批量验证时应利用缓存（同一 ID 只查询一次）"""
        route = respx.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        ).mock(return_value=Response(200, json={"result": {"12345678": {"uid": "12345678"}}}))
        verifier = CitationVerifier()

        # 批量验证，包含重复的 ID
        await verifier.verify(["12345678", "12345678", "12345678"])
        # 由于缓存，应该只发起 1 次 HTTP 请求
        assert route.call_count == 1


# ============================================================================
# CitationVerifier 状态区分测试
# ============================================================================

class TestCitationVerifierStatus:
    """CitationVerifier status 字段测试（P0-2 核心改造：区分 4 种状态）"""

    @respx.mock
    @pytest.mark.asyncio
    async def test_status_valid_for_existing_pubmed(self):
        """存在的 PMID → status="valid" """
        respx.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        ).mock(return_value=Response(200, json={"result": {"12345678": {"uid": "12345678"}}}))
        verifier = CitationVerifier()
        results = await verifier.verify(["12345678"])

        assert results["12345678"]["status"] == "valid"
        assert results["12345678"]["valid"] is True
        assert results["12345678"]["source"] == "pubmed"

    @respx.mock
    @pytest.mark.asyncio
    async def test_status_valid_for_existing_chembl(self):
        """存在的 ChEMBL ID → status="valid" """
        respx.get("https://www.ebi.ac.uk/chembl/api/data/chembl123.json").mock(
            return_value=Response(200, json={"molecule_chembl_id": "CHEMBL123"})
        )
        verifier = CitationVerifier()
        results = await verifier.verify(["CHEMBL123"])

        assert results["CHEMBL123"]["status"] == "valid"
        assert results["CHEMBL123"]["valid"] is True
        assert results["CHEMBL123"]["source"] == "chembl"

    @respx.mock
    @pytest.mark.asyncio
    async def test_status_invalid_for_nonexistent_chembl(self):
        """ChEMBL 404 → status="invalid"（引用不存在，不是网络错误）"""
        route = respx.get("https://www.ebi.ac.uk/chembl/api/data/chembl999999.json").mock(
            return_value=Response(404)
        )
        verifier = CitationVerifier()
        results = await verifier.verify(["CHEMBL999999"])

        assert results["CHEMBL999999"]["status"] == "invalid"
        assert results["CHEMBL999999"]["valid"] is False
        # 404 不应触发重试
        assert route.call_count == 1

    @respx.mock
    @pytest.mark.asyncio
    async def test_status_network_failed_on_chembl_500(self):
        """ChEMBL 500 → status="network_failed"（网络错误，不是引用不存在）"""
        respx.get("https://www.ebi.ac.uk/chembl/api/data/chembl123.json").mock(
            return_value=Response(500)
        )
        verifier = CitationVerifier(max_retries=3)
        results = await verifier.verify(["CHEMBL123"])

        assert results["CHEMBL123"]["status"] == "network_failed"
        assert results["CHEMBL123"]["valid"] is False

    @respx.mock
    @pytest.mark.asyncio
    async def test_status_network_failed_on_chembl_429(self):
        """ChEMBL 429 → status="network_failed" """
        respx.get("https://www.ebi.ac.uk/chembl/api/data/chembl123.json").mock(
            return_value=Response(429)
        )
        verifier = CitationVerifier(max_retries=2)
        results = await verifier.verify(["CHEMBL123"])

        assert results["CHEMBL123"]["status"] == "network_failed"

    def test_status_format_invalid_for_bad_pmid(self):
        """非法 PMID 格式 → status="format_invalid"（无 HTTP 请求）"""
        # 同步测试：format_invalid 不需要 HTTP 请求
        # 但 verify() 是 async，所以用 asyncio.run
        import asyncio

        verifier = CitationVerifier()
        results = asyncio.get_event_loop().run_until_complete(verifier.verify(["abc"]))
        assert results["abc"]["status"] == "format_invalid"
        assert results["abc"]["valid"] is False
        assert results["abc"]["source"] is None

    def test_status_format_invalid_for_bad_chembl(self):
        """非法 ChEMBL 格式 → status="format_invalid"

        注意：ChEMBL 格式要求 CHEMBL+数字（如 CHEMBL123）。
        用 "CHEMBL"（无数字）测试，它不匹配 PMID 也不匹配 ChEMBL。
        """
        import asyncio

        verifier = CitationVerifier()
        results = asyncio.get_event_loop().run_until_complete(verifier.verify(["CHEMBL"]))
        assert results["CHEMBL"]["status"] == "format_invalid"
        assert results["CHEMBL"]["source"] is None

    @respx.mock
    @pytest.mark.asyncio
    async def test_result_contains_all_required_fields(self):
        """验证结果应包含所有 P0-2 新增字段"""
        respx.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        ).mock(return_value=Response(200, json={"result": {"12345678": {"uid": "12345678"}}}))
        verifier = CitationVerifier()
        results = await verifier.verify(["12345678"])

        result = results["12345678"]
        # 必须包含的字段
        assert "reference_id" in result
        assert "valid" in result
        assert "status" in result
        assert "source" in result
        assert "attempts" in result
        # error 字段可选
        assert "error" in result

    @respx.mock
    @pytest.mark.asyncio
    async def test_pubmed_with_error_field_is_invalid(self):
        """PubMed 返回中包含 error 字段 → status="invalid" """
        respx.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        ).mock(
            return_value=Response(
                200,
                json={"result": {"12345678": {"error": "cannot get document summary"}}},
            )
        )
        verifier = CitationVerifier()
        results = await verifier.verify(["12345678"])

        assert results["12345678"]["status"] == "invalid"
        assert results["12345678"]["valid"] is False


# ============================================================================
# HallucinationDetector 降级测试（P0-2 核心：避免假阳性）
# ============================================================================

class TestHallucinationDetectorSeverity:
    """HallucinationDetector 严重等级测试（P0-2 核心改造：避免假阳性）"""

    @respx.mock
    @pytest.mark.asyncio
    async def test_network_failed_is_warning_not_critical(self):
        """network_failed → WARNING（不等于不存在，避免假阳性）"""
        # 模拟 3 次 500 错误 → network_failed
        respx.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        ).mock(return_value=Response(500))
        verifier = CitationVerifier(max_retries=3)
        detector = HallucinationDetector(verifier)

        issues = await detector.detect("研究表明该药物有效", ["12345678"])

        # 应该有 WARNING 而不是 CRITICAL
        network_issues = [i for i in issues if "12345678" in (i.field or "")]
        assert len(network_issues) == 1
        assert network_issues[0].severity == "WARNING"
        assert "网络验证失败" in network_issues[0].message
        assert "不等于不存在" in network_issues[0].message

    @respx.mock
    @pytest.mark.asyncio
    async def test_invalid_reference_is_critical(self):
        """invalid（确认不存在）→ CRITICAL（可能是幻觉）"""
        # ChEMBL 404 → invalid
        respx.get("https://www.ebi.ac.uk/chembl/api/data/chembl999999.json").mock(
            return_value=Response(404)
        )
        verifier = CitationVerifier()
        detector = HallucinationDetector(verifier)

        issues = await detector.detect("引用 CHEMBL999999 支持该结论", ["CHEMBL999999"])

        # 应该有 CRITICAL
        critical_issues = [i for i in issues if i.severity == "CRITICAL"]
        assert len(critical_issues) == 1
        assert "CHEMBL999999" in critical_issues[0].message

    @respx.mock
    @pytest.mark.asyncio
    async def test_valid_reference_no_issue(self):
        """valid 引用 → 不产生 hallucination 问题"""
        respx.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        ).mock(return_value=Response(200, json={"result": {"12345678": {"uid": "12345678"}}}))
        verifier = CitationVerifier()
        detector = HallucinationDetector(verifier)

        issues = await detector.detect("研究表明该药物有效", ["12345678"])

        # 不应有针对 12345678 的问题
        ref_issues = [i for i in issues if i.field == "12345678"]
        assert len(ref_issues) == 0

    @pytest.mark.asyncio
    async def test_format_invalid_is_warning(self):
        """format_invalid → WARNING（格式问题，不是幻觉）"""
        verifier = CitationVerifier()
        detector = HallucinationDetector(verifier)

        issues = await detector.detect("引用 abc 支持该结论", ["abc"])

        # 应该有 WARNING（格式不合法）
        format_issues = [i for i in issues if i.field == "abc"]
        assert len(format_issues) == 1
        assert format_issues[0].severity == "WARNING"
        assert "格式不合法" in format_issues[0].message

    @respx.mock
    @pytest.mark.asyncio
    async def test_mixed_status_references(self):
        """混合状态的引用应分别处理"""
        # 12345678 → valid
        respx.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        ).mock(return_value=Response(200, json={"result": {"12345678": {"uid": "12345678"}}}))
        # CHEMBL999999 → 404 invalid
        respx.get("https://www.ebi.ac.uk/chembl/api/data/chembl999999.json").mock(
            return_value=Response(404)
        )
        verifier = CitationVerifier()
        detector = HallucinationDetector(verifier)

        issues = await detector.detect(
            "研究有多个引用支持",
            ["12345678", "CHEMBL999999", "bad-format"],
        )

        # 12345678 (valid) → 不产生问题
        # CHEMBL999999 (invalid) → CRITICAL
        # bad-format (format_invalid) → WARNING
        critical = [i for i in issues if i.severity == "CRITICAL"]
        warnings = [i for i in issues if i.severity == "WARNING"]

        assert any(i.field == "CHEMBL999999" for i in critical)
        assert any(i.field == "bad-format" for i in warnings)
        # 12345678 不应出现在任何问题中
        assert not any(i.field == "12345678" for i in issues)

    @respx.mock
    @pytest.mark.asyncio
    async def test_network_failed_suggestion_recommends_retry(self):
        """network_failed 的建议应推荐重试或手动确认"""
        respx.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        ).mock(return_value=Response(500))
        verifier = CitationVerifier(max_retries=3)
        detector = HallucinationDetector(verifier)

        issues = await detector.detect("引用 12345678", ["12345678"])

        network_issues = [i for i in issues if i.field == "12345678"]
        assert len(network_issues) == 1
        assert network_issues[0].suggestion is not None
        assert "重试" in network_issues[0].suggestion or "手动确认" in network_issues[0].suggestion

    @respx.mock
    @pytest.mark.asyncio
    async def test_invalid_suggestion_recommends_remove(self):
        """invalid 的建议应推荐移除或替换"""
        respx.get("https://www.ebi.ac.uk/chembl/api/data/chembl999999.json").mock(
            return_value=Response(404)
        )
        verifier = CitationVerifier()
        detector = HallucinationDetector(verifier)

        issues = await detector.detect("引用 CHEMBL999999", ["CHEMBL999999"])

        critical = [i for i in issues if i.severity == "CRITICAL"]
        assert len(critical) == 1
        assert critical[0].suggestion is not None
        assert "移除" in critical[0].suggestion or "替换" in critical[0].suggestion


# ============================================================================
# RuleEngine 非对称举证测试
# ============================================================================

class TestRuleEngineScreenAsymmetric:
    """RuleEngine.screen_asymmetric() 测试（P0-2 核心改造）"""

    def test_screen_backwards_compatible(self):
        """screen() 应保持向后兼容（返回 list[str]）"""
        engine = RuleEngine()
        # 正常证据
        ev = _make_evidence("ev-1", "该药物具有显著的抑制活性", source="pubmed")
        issues = engine.screen(ev)
        assert isinstance(issues, list)
        assert len(issues) == 0  # 无问题

    def test_screen_backwards_compatible_with_issues(self):
        """screen() 有问题时返回字符串列表"""
        engine = RuleEngine()
        # 内容过短
        ev = _make_evidence("ev-1", "短", source="")
        issues = engine.screen(ev)
        assert isinstance(issues, list)
        assert all(isinstance(i, str) for i in issues)
        assert len(issues) >= 2  # 内容过短 + 来源为空

    def test_screen_asymmetric_returns_tuples(self):
        """screen_asymmetric() 应返回 list[tuple[str, str]]"""
        engine = RuleEngine()
        ev = _make_evidence("ev-1", "短", source="")
        issues = engine.screen_asymmetric(ev)
        assert isinstance(issues, list)
        assert all(isinstance(i, tuple) and len(i) == 2 for i in issues)

    def test_short_content_is_pending_not_rejected(self):
        """内容过短 → PENDING（非对称举证：不直接拒绝）"""
        engine = RuleEngine()
        ev = _make_evidence("ev-1", "短", source="pubmed")
        issues = engine.screen_asymmetric(ev)

        # 应该有 PENDING 而不是 CRITICAL
        short_issues = [i for i in issues if "内容过短" in i[0]]
        assert len(short_issues) == 1
        assert short_issues[0][1] == "PENDING"

    def test_empty_source_is_pending(self):
        """来源为空 → PENDING"""
        engine = RuleEngine()
        ev = _make_evidence("ev-1", "该药物具有显著的抑制活性", source="")
        issues = engine.screen_asymmetric(ev)

        source_issues = [i for i in issues if "来源为空" in i[0]]
        assert len(source_issues) == 1
        assert source_issues[0][1] == "PENDING"

    def test_agent_opinion_without_refs_is_pending(self):
        """Agent 观点缺少引用 → PENDING"""
        engine = RuleEngine()
        ev = _make_evidence(
            "ev-1",
            "根据我的分析该药物有效",
            source="agent_opinion",
            reference_ids=[],
        )
        issues = engine.screen_asymmetric(ev)

        opinion_issues = [i for i in issues if "Agent 观点" in i[0]]
        assert len(opinion_issues) == 1
        assert opinion_issues[0][1] == "PENDING"

    def test_af2_reference_is_critical(self):
        """AF2 引用 → CRITICAL（硬性违规，必须拒绝）"""
        engine = RuleEngine()
        ev = _make_evidence(
            "ev-1",
            "我们使用 AlphaFold2 预测了结构",
            source="pubmed",
        )
        issues = engine.screen_asymmetric(ev)

        af2_issues = [i for i in issues if "AlphaFold2" in i[0] or "AF2" in i[0]]
        assert len(af2_issues) == 1
        assert af2_issues[0][1] == "CRITICAL"

    def test_af2_lowercase_is_critical(self):
        """小写 af2 引用 → CRITICAL"""
        engine = RuleEngine()
        ev = _make_evidence(
            "ev-1",
            "我们使用 af2 预测了结构",
            source="pubmed",
        )
        issues = engine.screen_asymmetric(ev)
        af2_issues = [i for i in issues if "AF2" in i[0] or "AlphaFold2" in i[0]]
        assert len(af2_issues) == 1
        assert af2_issues[0][1] == "CRITICAL"

    def test_af3_reference_is_critical(self):
        """v0.4.8：AlphaFold3 引用同样违反结构预测工具禁令 → CRITICAL
        （合规降级链 OpenDDE→Chai1→Boltz，AF2/AF3 双禁）"""
        engine = RuleEngine()
        ev = _make_evidence(
            "ev-1",
            "我们使用 AlphaFold3 预测了结构",
            source="pubmed",
        )
        issues = engine.screen_asymmetric(ev)
        tool_issues = [i for i in issues if "AlphaFold2/3" in i[0]]
        assert len(tool_issues) == 1
        assert tool_issues[0][1] == "CRITICAL"

    def test_clean_evidence_no_issues(self):
        """干净证据 → 无问题"""
        engine = RuleEngine()
        ev = _make_evidence(
            "ev-1",
            "该药物具有显著的抑制活性，pIC50=8.5",
            source="pubmed",
            reference_ids=["12345678"],
        )
        issues = engine.screen_asymmetric(ev)
        assert len(issues) == 0

    def test_empty_content_is_pending(self):
        """空内容 → PENDING"""
        engine = RuleEngine()
        ev = _make_evidence("ev-1", "", source="pubmed")
        issues = engine.screen_asymmetric(ev)
        empty_issues = [i for i in issues if "内容过短" in i[0]]
        assert len(empty_issues) == 1
        assert empty_issues[0][1] == "PENDING"


class TestRuleEngineEvaluateWithRubric:
    """RuleEngine.evaluate_with_rubric() 测试（P0-2 核心改造）"""

    def test_returns_rubric_verdict(self):
        """应返回 RubricVerdict 实例"""
        engine = RuleEngine()
        ev = _make_evidence("ev-1", "该药物有效", source="pubmed")
        scores = [_make_score("ev-1")]
        evidence_map = {"ev-1": ev}

        verdict = engine.evaluate_with_rubric(scores, evidence_map, topic="药物活性")

        assert isinstance(verdict, RubricVerdict)

    def test_no_contradiction_passes(self):
        """无矛盾时应该通过"""
        engine = RuleEngine()
        ev_a = _make_evidence("ev-1", "该药物有效", source="pubmed")
        ev_b = _make_evidence("ev-2", "溶解度良好", source="experiment")
        scores = [_make_score("ev-1"), _make_score("ev-2")]
        evidence_map = {"ev-1": ev_a, "ev-2": ev_b}

        verdict = engine.evaluate_with_rubric(scores, evidence_map, topic="药物活性")

        assert verdict.passed is True

    def test_contradiction_fails(self):
        """发现矛盾时应该不通过"""
        engine = RuleEngine()
        ev_a = _make_evidence("ev-1", "药物有效", source="pubmed")
        ev_b = _make_evidence("ev-2", "药物无效", source="experiment")
        scores = [_make_score("ev-1"), _make_score("ev-2")]
        evidence_map = {"ev-1": ev_a, "ev-2": ev_b}

        verdict = engine.evaluate_with_rubric(scores, evidence_map, topic="药物活性")

        assert verdict.passed is False
        assert len(verdict.contradictions) == 1

    def test_empty_evidence_passes(self):
        """空证据集应该通过（非对称举证：默认无罪）"""
        engine = RuleEngine()
        verdict = engine.evaluate_with_rubric([], {}, topic="topic")
        assert verdict.passed is True

    def test_topic_passed_through(self):
        """topic 应传递到矛盾描述中"""
        engine = RuleEngine()
        ev_a = _make_evidence("ev-1", "有效", source="pubmed")
        ev_b = _make_evidence("ev-2", "无效", source="experiment")
        scores = [_make_score("ev-1"), _make_score("ev-2")]
        evidence_map = {"ev-1": ev_a, "ev-2": ev_b}

        verdict = engine.evaluate_with_rubric(scores, evidence_map, topic="特定主题")

        assert "特定主题" in verdict.contradictions[0].contradiction_desc


# ============================================================================
# 端到端集成测试
# ============================================================================

class TestEndToEndAsymmetricJudgement:
    """端到端非对称举证审查测试"""

    @respx.mock
    @pytest.mark.asyncio
    async def test_clean_evidence_with_valid_citations_passes(self):
        """干净证据 + 有效引用 → 通过审查"""
        # 模拟有效的 PubMed 引用
        respx.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        ).mock(return_value=Response(200, json={"result": {"12345678": {"uid": "12345678"}}}))

        # 1. 验证引用
        verifier = CitationVerifier()
        detector = HallucinationDetector(verifier)
        issues = await detector.detect(
            "研究表明该药物具有抑制活性",
            ["12345678"],
        )

        # 2. 无 CRITICAL 问题
        critical = [i for i in issues if i.severity == "CRITICAL"]
        assert len(critical) == 0

        # 3. 进行非对称举证审查
        engine = RuleEngine()
        ev = _make_evidence(
            "ev-1",
            "研究表明该药物具有抑制活性",
            source="pubmed",
            reference_ids=["12345678"],
        )
        scores = [_make_score("ev-1")]
        verdict = engine.evaluate_with_rubric(scores, {"ev-1": ev}, topic="抑制活性")

        assert verdict.passed is True

    @respx.mock
    @pytest.mark.asyncio
    async def test_network_failure_does_not_block_pass(self):
        """网络失败不应阻止证据通过（避免假阳性）"""
        # 模拟网络失败
        respx.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        ).mock(return_value=Response(500))

        verifier = CitationVerifier(max_retries=3)
        detector = HallucinationDetector(verifier)

        # 引用验证失败，但只是 WARNING 不是 CRITICAL
        issues = await detector.detect("研究表明该药物有效", ["12345678"])

        critical = [i for i in issues if i.severity == "CRITICAL"]
        warnings = [i for i in issues if i.severity == "WARNING"]
        assert len(critical) == 0  # 无 CRITICAL
        assert len(warnings) >= 1  # 有 WARNING

        # 非对称举证审查仍然可以通过
        engine = RuleEngine()
        ev = _make_evidence(
            "ev-1",
            "研究表明该药物有效",
            source="pubmed",
            reference_ids=["12345678"],
        )
        scores = [_make_score("ev-1")]
        verdict = engine.evaluate_with_rubric(scores, {"ev-1": ev}, topic="药物活性")

        assert verdict.passed is True

    @respx.mock
    @pytest.mark.asyncio
    async def test_invalid_citation_blocks_pass(self):
        """确认不存在的引用 → CRITICAL → 阻止通过"""
        # ChEMBL 404 → invalid
        respx.get("https://www.ebi.ac.uk/chembl/api/data/chembl999999.json").mock(
            return_value=Response(404)
        )

        verifier = CitationVerifier()
        detector = HallucinationDetector(verifier)
        issues = await detector.detect("引用 CHEMBL999999", ["CHEMBL999999"])

        # 应该有 CRITICAL
        critical = [i for i in issues if i.severity == "CRITICAL"]
        assert len(critical) >= 1

    def test_contradictory_evidence_blocks_pass(self):
        """矛盾证据 → 阻止通过"""
        engine = RuleEngine()
        ev_a = _make_evidence("ev-1", "实验组阳性", source="pubmed")
        ev_b = _make_evidence("ev-2", "实验组阴性", source="experiment")
        scores = [_make_score("ev-1"), _make_score("ev-2")]
        evidence_map = {"ev-1": ev_a, "ev-2": ev_b}

        verdict = engine.evaluate_with_rubric(scores, evidence_map, topic="实验结果")

        assert verdict.passed is False
        assert len(verdict.contradictions) == 1
        assert verdict.contradictions[0].severity == Severity.CRITICAL

    def test_pending_issues_do_not_block_pass(self):
        """待举证问题不阻止通过（非对称举证核心原则）"""
        engine = RuleEngine()
        # 多个 PENDING 问题但无矛盾
        ev_short = _make_evidence("ev-1", "短", source="pubmed")
        ev_no_source = _make_evidence("ev-2", "该药物有效", source="")
        ev_opinion = _make_evidence(
            "ev-3", "我认为有效", source="agent_opinion", reference_ids=[]
        )
        scores = [_make_score("ev-1"), _make_score("ev-2"), _make_score("ev-3")]
        evidence_map = {"ev-1": ev_short, "ev-2": ev_no_source, "ev-3": ev_opinion}

        verdict = engine.evaluate_with_rubric(scores, evidence_map, topic="topic")

        # 虽然有很多问题，但没有矛盾，应该通过
        assert verdict.passed is True
        assert len(verdict.pending_issues) >= 3

    def test_af2_reference_blocks_pass(self):
        """AF2 引用 → CRITICAL → 阻止通过"""
        engine = RuleEngine()
        ev = _make_evidence(
            "ev-1",
            "我们使用 AlphaFold2 预测了蛋白质结构",
            source="pubmed",
        )
        issues = engine.screen_asymmetric(ev)

        # AF2 是硬约束，应该是 CRITICAL
        af2_issues = [i for i in issues if i[1] == "CRITICAL"]
        assert len(af2_issues) >= 1
        assert any("AlphaFold2" in i[0] or "AF2" in i[0] for i in af2_issues)

    @respx.mock
    @pytest.mark.asyncio
    async def test_full_workflow_collect_verify_judge(self):
        """完整工作流：收集证据 → 验证引用 → 非对称举证裁决"""
        from app.core.virtual_lab.rubric_engine import DrillBeforeDisposition

        # 1. 模拟有效引用
        respx.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        ).mock(return_value=Response(200, json={"result": {"12345678": {"uid": "12345678"}}}))

        # 2. 收集证据
        drill = DrillBeforeDisposition()
        drill.collect(
            _make_evidence(
                "ev-1",
                "该药物 pIC50=8.5，具有抑制活性",
                source="pubmed",
                reference_ids=["12345678"],
            )
        )
        drill.collect(
            _make_evidence(
                "ev-2",
                "溶解度 50mg/mL，成药性良好",
                source="experiment",
            )
        )
        drill.complete_collection()

        # 3. 验证引用
        verifier = CitationVerifier()
        detector = HallucinationDetector(verifier)
        issues = await detector.detect(
            "研究表明该药物 pIC50=8.5",
            ["12345678"],
        )
        critical = [i for i in issues if i.severity == "CRITICAL"]
        assert len(critical) == 0  # 引用有效

        # 4. 非对称举证裁决
        verdict = drill.disposition(topic="药物成药性")

        assert verdict.passed is True
        assert verdict.contradictions == []
        assert verdict.confidence > 0.5

    def test_rubric_engine_integration_with_rule_engine(self):
        """rubric_engine 与 RuleEngine 集成测试"""
        rubric = AsymmetricRubric()
        engine = RuleEngine()

        # 构造矛盾证据
        ev_a = _make_evidence("ev-1", "药物激活通路", source="pubmed")
        ev_b = _make_evidence("ev-2", "药物抑制通路", source="experiment")
        scores = [_make_score("ev-1"), _make_score("ev-2")]
        evidence_map = {"ev-1": ev_a, "ev-2": ev_b}

        # 通过 RuleEngine 调用 rubric
        verdict = engine.evaluate_with_rubric(scores, evidence_map, topic="通路调控")

        # 直接调用 rubric（应得到相同结果）
        direct_verdict = rubric.evaluate(scores, evidence_map, topic="通路调控")

        assert verdict.passed == direct_verdict.passed
        assert len(verdict.contradictions) == len(direct_verdict.contradictions)
        assert verdict.confidence == direct_verdict.confidence
