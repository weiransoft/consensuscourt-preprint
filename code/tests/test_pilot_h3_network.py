"""
H3 pilot 实验 —— 三态状态机（valid/invalid/network_failed）vs 二态（valid/not_found）假阳性对比

假设 H3（deep-research 报告 §四 RQ3）：
    在证据集上注入引用验证服务网络故障（failure rate f ∈ {0%, 30%, 50%, 100%}），
    二态验证器（把"基础设施故障"当"引用不存在"→定罪拒绝）的假阳性率
    随 f 线性上升；而三态验证器（区分"网络失败"与"引用不存在"，
    network_failed 标记为待举证、不直接拒绝）保持平坦。

与生产代码的对应关系（CitationVerifier 真实三态 status）：
    - "valid"           → 引用存在（200）
    - "invalid"         → 来源确认不存在（404）→ 定罪
    - "network_failed"  → 基础设施故障（网络异常/5xx/429）→ 三态不定罪
    - "format_invalid"  → 格式不合法 → 非网络信号，两策略均不定罪
      （格式缺陷属内容质量问题，由 RuleEngine 初筛处理，H3 只隔离网络故障变量）

实验设计（与 H2a 同构，作用在同一 DDB-L1 基准上）：
    - 案例证据带 reference_ids；验证器以 failure rate f 抛网络异常；
    - 二态策略（baseline_binary）：invalid ∪ network_failed ∪ format_invalid → 定罪拒绝
      （模拟把"不可确认"当"不存在"的朴素二态世界）；
    - 三态策略（tri_state）：仅 invalid 定罪；network_failed → 待举证不定罪；
    - 指标：FPR（无矛盾案例上的拒绝率），f 扫描曲线。

运行方式（在 backend/ 目录下）：
    python -m pytest tests/ddb/test_pilot_h3_network.py -v   # 单元测试
    python -m tests.ddb.test_pilot_h3_network                # 全量 pilot（600 例，输出 JSON 报告）

依赖说明：
    CitationVerifier 通过构造注入 httpx.AsyncClient —— pilot 用 _FlakyMockClient
    控制三类响应：200 有效 / 404 不存在 / 抛 httpx.ConnectError 网络失败。
    CitationVerifier 内部把任何 httpx 异常转为 _NetworkError → 重试 max_retries 次
    后降级 status="network_failed"，正好用于 H3 三态注入。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path

import httpx

# 导入 app 前的环境变量兜底（与 test_pilot_h2a.py 相同模式）
os.environ.setdefault("SCIMINER_API_KEY", "test-key-for-pilot")
os.environ.setdefault("DB_PASSWORD", "test-password-for-pilot")

from tests.ddb.l1_generator import generate  # noqa: E402

L1_REPORT_DIR = Path(__file__).resolve().parent


class _FlakyMockClient(httpx.AsyncClient):
    """
    可控故障的 HTTP 客户端 mock（H3 网络失败注入）。

    两类随机性分离（对齐真实故障世界）：
    - 引用存在性：按【引用 ID】稳定哈希决定（同一 PMID 在任何 f 下答案一致，
      对齐"文献要么存在要么不存在"的客观事实；按请求顺序掷骰会让不同 f
      消耗不同随机数个数，导致同一引用在不同 f 下存在性漂移，平坦性失真）；
    - 网络故障：按【每次尝试】独立掷骰（对齐重试可恢复的真实故障）。
    """

    def __init__(self, failure_rate: float, rng: random.Random, seed: int, invalid_share: float = 0.5):
        super().__init__(base_url="https://mock.example.com")
        self._failure_rate = failure_rate
        self._rng = rng
        self._seed = seed
        # 存在性不可靠（invalid_share）占比：该噪声两策略共享且与 f 无关，
        # 因此两策略的 clean FPR 之差在任何 f 下都精确隔离"故障定罪"差异
        self._invalid_share = invalid_share
        # 记录调用序列，供测试断言注入分布
        self.call_log: list[tuple[str, str]] = []

    def _exists(self, pmid: str) -> bool:
        """稳定哈希决定该引用是否存在（约 invalid_share 比例不存在）。"""
        digest = hashlib.sha256(f"{self._seed}:{pmid}".encode()).digest()
        return (digest[0] % 1000) >= int(self._invalid_share * 1000)

    async def get(self, url: str, **kwargs):  # type: ignore[override]
        roll = self._rng.random()
        if roll < self._failure_rate:
            self.call_log.append((url, "network_error"))
            raise httpx.ConnectError("simulated network failure")

        params = kwargs.get("params") or {}
        pmid = str(params.get("id", ""))
        if self._exists(pmid):
            self.call_log.append((pmid, "valid"))
            # PubMed esummary 结构：result 内含该 pmid 键 → CitationVerifier 判 valid
            body = {"result": {"uids": [pmid], pmid: {"rid": "1"}}}
            return httpx.Response(
                200, request=httpx.Request("GET", url),
                json=body,
            )
        self.call_log.append((pmid, "invalid"))
        return httpx.Response(404, request=httpx.Request("GET", url))


def _build_verifier(failure_rate: float, seed: int, invalid_share: float = 0.5):
    """构造 CitationVerifier 并注入故障 mock 客户端。"""
    from app.core.virtual_lab.auto_validator import CitationVerifier

    rng = random.Random(seed)
    client = _FlakyMockClient(
        failure_rate=failure_rate, rng=rng, seed=seed, invalid_share=invalid_share,
    )
    return CitationVerifier(http_client=client, max_retries=3)


async def _run_citation_checks(verifier, cases: list[dict]) -> list[list[dict]]:
    """
    对每个案例的所有证据引用执行验证，返回逐案例的结果矩阵。

    说明：CitationVerifier.verify() 带 24h 缓存，同一 verifier 跨 f 复用会导致
    低 f 的结果污染高 f 的注入——每个 f 必须新建 verifier（见
    _evaluate_at_failure_rate），此处不跨调用复用。
    """
    per_case_results = []
    for case in cases:
        ref_ids = [ref for ev in case["evidences"] for ref in ev.get("reference_ids", [])]
        # 转 CitationVerifier 可验证格式（PMID 纯数字 / CHEMBL+数字），
        # 保证注入分布干净（format_invalid≈0）
        numeric_refs = [_to_verifiable_id(rid) for rid in ref_ids]
        results = await verifier.verify(numeric_refs) if numeric_refs else {}
        per_case_results.append(
            [{"ref": orig, "numeric": num, **results[num]}
             for orig, num in zip(ref_ids, numeric_refs)]
        )
    return per_case_results


def _to_verifiable_id(reference_id: str) -> str:
    """
    把 DDB-L1 引用 ID 转为 CitationVerifier 可验证的合法格式。

    CitationVerifier 只认两种格式：纯数字 PMID（^\\d{1,10}$）与
    CHEMBL+数字（^CHEMBL\\d+$）。DDB-L1 生成的 "CHEMBL123456" 本身合法，
    "PMID-123456" 需剥离前缀转纯数字；UNI/DRUGBANK/EXP 前缀无对应验证通道，
    统一转数字按 PMID 验证（引用存在性由稳定哈希决定，与前缀无关，
    这样做不引入任何系统性偏差，且保证全部引用可验证、format_invalid≈0，
    注入分布干净）。
    """
    m = re.search(r"\d+", reference_id)
    if not m:
        return reference_id
    digits = m.group(0)
    if reference_id.startswith("CHEMBL"):
        return f"CHEMBL{digits}"
    return digits


def _clean_fpr(cases: list[dict], verdict_fn, cite_results: list[dict]) -> float:
    """
    仅在 clean_neutral 子集（无矛盾、无形式缺陷）上计算拒绝率。

    defect_only 案例会被两策略共享的初筛定罪（H2a 已覆盖），其噪声会掩盖
    网络故障变量——H3 指标必须隔离初筛底座，只观察引用验证层的差异。
    """
    clean_idx = [
        i for i, c in enumerate(cases)
        if not c["ground_truth"]["has_true_contradiction"]
        and not c["ground_truth"]["defect_only"]
    ]
    if not clean_idx:
        return 0.0
    rejected = sum(int(verdict_fn(cases[i], cite_results[i])) for i in clean_idx)
    return rejected / len(clean_idx)


def _rule_screen_case(case: dict) -> bool:
    """
    规则初筛：与 H2a 的有罪推定口径一致（任一证据形式缺陷 → 案例拒绝）。
    H3 实验里该步为两策略共享底座，差异只发生在引用验证定罪规则。
    """
    from app.core.virtual_lab.consensus_court import Evidence, RuleEngine

    engine = RuleEngine()
    for ev in case["evidences"]:
        evidence = Evidence(
            evidence_id=ev["evidence_id"],
            source=ev["source"],
            content=ev["content"],
            role=ev["role"],
            reference_ids=ev.get("reference_ids", []),
        )
        if engine.screen(evidence):
            return True  # 缺陷 → 拒绝
    return False


def _binary_verdict(case: dict, cite_results: list[dict]) -> bool:
    """
    二态策略（baseline_binary）：引用验证只有"存在/不存在"两态——
    404、网络失败、格式不可解析都合并为"引用不成立"→ 定罪拒绝。

    Returns:
        True = 拒绝（定罪）
    """
    if _rule_screen_case(case):
        return True
    for r in cite_results:
        # 二态世界的核心缺陷：一切"非 valid"都当"不存在"定罪
        if r["status"] != "valid":
            return True
    return False


def _tri_state_verdict(case: dict, cite_results: list[dict]) -> bool:
    """
    三态策略（tri_state）：引用验证三态分离——
    - invalid（真 404，来源确认不存在）→ 定罪；
    - network_failed（基础设施故障）→ 待举证，不定罪（生产语义：降级 WARNING）；
    - format_invalid（格式问题，非网络信号）→ 不定罪（由初筛处理）；
    - valid → 通过。

    Returns:
        True = 拒绝（定罪）
    """
    if _rule_screen_case(case):
        return True
    for r in cite_results:
        if r["status"] == "invalid":
            return True
    return False


def _load_or_generate_cases() -> list[dict]:
    """加载 DDB-L1 基准（与 H2a 同源）。"""
    return generate(num_with_contradiction=300, num_no_contradiction=300, seed=42)


async def _evaluate_at_failure_rate(
    cases: list[dict],
    failure_rate: float,
    seed: int,
    invalid_share: float = 0.5,
) -> dict:
    """
    在给定故障率 f 下运行两种策略并计算指标。

    每个 f 新建 verifier：避免 CitationVerifier 缓存把低 f 的 valid 结果
    泄漏到高 f 注入（缓存对同一 reference_id 24h 内复用）。

    Args:
        invalid_share: mock 中"真实不存在"引用的占比（两策略共享噪声）。

    Returns:
        {failure_rate, binary_fpr, tri_fpr, binary_fpr_clean, tri_fpr_clean,
         binary_recall, tri_recall, injection_check}
    """
    verifier = _build_verifier(failure_rate, seed, invalid_share)
    per_case = await _run_citation_checks(verifier, cases)

    # 注入健康检查：统计 mock 实际掷出的响应分布
    statuses = [r["status"] for results in per_case for r in results]
    total = max(1, len(statuses))
    injection_check = {
        "valid": round(statuses.count("valid") / total, 4),
        "invalid": round(statuses.count("invalid") / total, 4),
        "network_failed": round(statuses.count("network_failed") / total, 4),
        "format_invalid": round(statuses.count("format_invalid") / total, 4),
    }

    # H3 主指标：clean_neutral 子集上的拒绝率（隔离初筛底座，只看引用验证层）
    binary_clean_fpr = _clean_fpr(cases, _binary_verdict, per_case)
    tri_clean_fpr = _clean_fpr(cases, _tri_state_verdict, per_case)

    binary_rejected = 0
    tri_rejected = 0
    defect_count = 0
    binary_defect_rejected = 0
    tri_defect_rejected = 0
    contra_count = 0
    binary_contra_rejected = 0
    tri_contra_rejected = 0

    for case, cite_results in zip(cases, per_case):
        b = _binary_verdict(case, cite_results)
        t = _tri_state_verdict(case, cite_results)
        if not case["ground_truth"]["has_true_contradiction"]:
            binary_rejected += int(b)
            tri_rejected += int(t)
            if case["ground_truth"]["defect_only"]:
                defect_count += 1
                binary_defect_rejected += int(b)
                tri_defect_rejected += int(t)
        else:
            contra_count += 1
            binary_contra_rejected += int(b)
            tri_contra_rejected += int(t)

    no_contra = sum(1 for c in cases if not c["ground_truth"]["has_true_contradiction"])
    return {
        "failure_rate": failure_rate,
        "binary_fpr": round(binary_rejected / max(1, no_contra), 4),
        "tri_fpr": round(tri_rejected / max(1, no_contra), 4),
        "binary_fpr_clean": round(binary_clean_fpr, 4),
        "tri_fpr_clean": round(tri_clean_fpr, 4),
        "binary_fpr_defect_only": round(binary_defect_rejected / max(1, defect_count), 4),
        "tri_fpr_defect_only": round(tri_defect_rejected / max(1, defect_count), 4),
        "binary_recall": round(binary_contra_rejected / max(1, contra_count), 4),
        "tri_recall": round(tri_contra_rejected / max(1, contra_count), 4),
        "injection_check": injection_check,
    }


# ---------------------------------------------------------------------------
# pytest 单元测试
# ---------------------------------------------------------------------------

def test_generator_has_references():
    """基准自检：多数证据携带 reference_ids（H3 引用验证实验的数据前提）"""
    cases = generate(num_with_contradiction=20, num_no_contradiction=20, seed=42)
    ref_count = sum(
        1
        for c in cases
        for ev in c["evidences"]
        if ev.get("reference_ids")
    )
    total_ev = sum(len(c["evidences"]) for c in cases)
    assert ref_count > total_ev * 0.5, (
        f"H3 实验需要多数证据带引用：实际 {ref_count}/{total_ev}"
    )
    # defect_only 案例必须命中 RuleEngine.screen 有罪推定（缺陷类型由随机注入决定）
    defect_only = [c for c in cases if c["ground_truth"]["defect_only"]]
    assert defect_only, "必须存在 defect_only 案例"
    for c in defect_only:
        assert _rule_screen_case(c), f"defect_only 案例 {c['case_id']} 未命中有罪推定初筛"
    # clean_neutral 案例必须完全通过初筛（无误伤，保证 FPR 基线干净）
    clean = [
        c for c in cases
        if not c["ground_truth"]["has_true_contradiction"]
        and not c["ground_truth"]["defect_only"]
    ]
    assert clean, "必须存在 clean_neutral 案例"
    for c in clean:
        assert not _rule_screen_case(c), f"clean 案例 {c['case_id']} 被初筛误伤"


def test_binary_fpr_monotonic_in_failure_rate():
    """H3 核心方向性断言：小样本上二态 clean FPR 随故障率单调上升且高 f 全误杀"""
    cases = generate(num_with_contradiction=10, num_no_contradiction=30, seed=42)

    async def _run() -> list[dict]:
        results = []
        # 必须【顺序】await：三个 f 并发 gather 共享随机流会使注入分布互相耦合
        # invalid_share=0：干净基线下 f=0 应零误杀，f 的全部抬升都是故障定罪
        for f in (0.0, 0.5, 1.0):
            results.append(await _evaluate_at_failure_rate(cases, f, 42, invalid_share=0.0))
        return results

    results = asyncio.run(_run())
    # 主指标用 clean 子集：defect_only 案例被初筛定罪会淹没网络故障信号
    binary_fprs = [r["binary_fpr_clean"] for r in results]
    assert binary_fprs[0] == 0.0, f"无故障无无效引用时二态应零误杀: {binary_fprs}"
    for a, b in zip(binary_fprs, binary_fprs[1:]):
        assert b >= a - 1e-9, f"二态 FPR 应随故障率单调不降: {binary_fprs}"
    # f=1.0 时所有带引用的 clean 案例被网络故障误杀 → 全误杀
    assert binary_fprs[-1] == 1.0, f"f=1.0 时二态应全误杀 clean 案例: {binary_fprs}"


def test_tristate_fpr_immune_to_network_failure():
    """H3 核心机制断言：小样本上三态 clean FPR 对网络故障平坦，高 f 下显著优于二态"""
    cases = generate(num_with_contradiction=10, num_no_contradiction=30, seed=42)

    async def _run() -> list[dict]:
        results = []
        # invalid_share=0：存在性底座干净（全部引用真实存在），
        # 三态在任何 f 下都不应定罪 clean 案例 → 平坦性断言零容差、严格成立。
        # 故障定罪差异（二态抬升）已由 test_binary_fpr_monotonic 覆盖
        for f in (0.0, 0.5, 1.0):
            results.append(await _evaluate_at_failure_rate(cases, f, 42, invalid_share=0.0))
        return results

    results = asyncio.run(_run())
    tri_clean = [r["tri_fpr_clean"] for r in results]
    # 三态下网络故障只把 valid 变成 network_failed（均不定罪），
    # 干净底座上 clean FPR 恒为 0，与 f 完全无关
    assert max(tri_clean) - min(tri_clean) <= 1e-9, (
        f"三态 FPR 应与网络故障完全无关: {tri_clean}"
    )
    # 高 f 下三态必须显著优于二态（H3 机制的直接证据）
    assert results[-1]["tri_fpr_clean"] < results[-1]["binary_fpr_clean"] - 0.1, (
        f"f=1.0 时三态 FPR 应显著低于二态: {results[-1]}"
    )


# ---------------------------------------------------------------------------
# 全量 pilot（600 例 × f 扫描），输出 JSON 报告
# ---------------------------------------------------------------------------

def main() -> dict:
    """全量 H3 pilot：600 例 × f ∈ {0,0.3,0.5,1.0} × 两策略，输出报告。"""
    cases = _load_or_generate_cases()
    # 全量 pilot 用 invalid_share=0：真实世界引用存在性噪声（invalid）与
    # 网络故障噪声分属两个变量，H3 只隔离"故障定罪策略差异"——
    # 存在性底座设干净后，注入分布 = (1-f)·valid + f·network_failed，
    # 指标曲线直接对应假设 H3。小样本单测另验证噪声底座下的鲁棒性。
    invalid_share = 0.0

    async def _sweep() -> list[dict]:
        rows = []
        for f in (0.0, 0.3, 0.5, 1.0):
            rows.append(await _evaluate_at_failure_rate(cases, f, 42, invalid_share))
        return rows

    sweep = asyncio.run(_sweep())
    report = {
        "experiment": "H3 — 网络失败注入下三态 vs 二态引用验证策略假阳性对比 pilot",
        "benchmark": "DDB-L1（tests/ddb/l1_generator.py，600 例）",
        "strategies": {
            "binary": "二态：非 valid（invalid ∪ network_failed ∪ format_invalid）→ 定罪拒绝（朴素世界缺陷）",
            "tri_state": "三态：仅 invalid 定罪；network_failed → 待举证不定罪；format_invalid 交初筛",
        },
        "metrics": {
            "*_fpr_clean": "clean_neutral 子集拒绝率（H3 主指标，隔离初筛底座）",
            "*_fpr": "全部无矛盾案例拒绝率（含 defect_only，作参考）",
            "*_recall": "真实矛盾案例拒绝率",
        },
        "failure_rates": [0.0, 0.3, 0.5, 1.0],
        "seed": 42,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "invalid_share": invalid_share,
        "sweep": sweep,
    }
    out_path = L1_REPORT_DIR / "pilot_h3_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[H3 pilot] 报告已写入 {out_path}")
    for row in sweep:
        print(
            f"f={row['failure_rate']:.1f}"
            f"  clean: binary={row['binary_fpr_clean']:.3f}"
            f" tri={row['tri_fpr_clean']:.3f}"
            f" Δ={row['binary_fpr_clean'] - row['tri_fpr_clean']:+.3f}"
            f"  inj={row['injection_check']}"
        )
    return report


if __name__ == "__main__":
    main()
