#!/usr/bin/env python3
"""
§6.5 网络失败区分实验（论文大纲 §6.5 首条专项实验，P0-2 核心，Fig.4 数据源，v0.4.14）。

实验目标
========
大纲 §6.5 承诺："注入 30%/50%/100% 引用验证网络失败，对比'三态 + 重试 + 缓存'
vs '二态直拒'的假阳性曲线 → 直接回应'基础设施故障污染科学裁决'这一被社区
忽视的问题"。

背景：现有引用验证工作把"验证失败"直接当"引用错误"（二态：valid / invalid）。
本文 CitationVerifier（auto_validator.py）已实现 P0-2 三态改造：
status ∈ {valid, invalid, network_failed, format_invalid} + 3 次指数重试 +
24h TTL 缓存，HallucinationDetector 据此分级——invalid（确认不存在）→
CRITICAL（幻觉定罪），network_failed（网络失败）→ WARNING（不等于不存在，
随案移交不定罪）。

本实验把该机制放到系统性网络故障下量化检验：

1. 确定性注入器按失败率 p ∈ {0, 0.3, 0.5, 1.0} 对 PubMed 传输层注入
   连接失败（httpx.MockTransport 官方传输层接口，网络层模拟即实验处理本身）；
2. 两种验证失败处置策略消费同一套生产验证/重试/缓存实现，唯一变量 =
   失败结果的处置映射：
   - three_state_resilient（生产策略）：三态分级 + 3 次重试 + 跨案缓存，
     走生产 HallucinationDetector 原始路径定罪；
   - two_state_direct（社区现状基线，实验侧实现）：无重试（1 次尝试）、
     无跨案缓存，任何非 valid 状态一律按"引用不存在"定罪（直拒）；
3. 在干净主张集（A 层，全部引用真实有效，正确裁决=放行）上量 FPR，
   在确认不存在引用集（B 层，引用格式合法但确认不存在，应被定罪）上量召回；
4. 六条预注册红线（见 check_red_lines）全部满足才退出 0，任何违例退出 1。

科学预期
========
- 三态策略：network_failed 仅产生 WARNING，结构性不进入定罪通道 →
  A 层 FPR 在全部失败率格点保持 0（该红线是对"降级接线正确性"的
  可证伪检验，而非理论保证）；代价是全断网（p=1.0）时无法确认引用
  不存在 → B 层召回降为 0——系统选择"弃权"而非"误判"，与 ABP
  无罪推定语义一致。
- 二态直拒：网络失败被当引用错误 → A 层 FPR 随 p 单调上升，
  p=1.0 时全部干净主张被误判（FPR=1.0）——量化"基础设施故障污染
  科学裁决"的上界；其 B 层召回恒 1.0 的高召回完全以假阳性为代价。

用法：
    backend$ .venv/bin/python tests/scripts/run_network_failure_experiment.py
    backend$ .venv/bin/python tests/scripts/run_network_failure_experiment.py \\
        --cases 120 --invalid-cases 40

输出：tests/ddb/network_failure_report.json + stdout 摘要表
退出码：0=全部红线满足；1=红线违例；2=运行异常
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# 导入前置：backend 根入 sys.path + 环境变量占位（必须在 import app 之前）
# ---------------------------------------------------------------------------
_BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

os.environ.setdefault("SCIMINER_API_KEY", "network-failure-exp-placeholder")
os.environ.setdefault("DB_PASSWORD", "network-failure-exp-placeholder")

import httpx  # noqa: E402

from app.core.virtual_lab.auto_validator import (  # noqa: E402
    CitationVerifier,
    HallucinationDetector,
)

#: 实验种子（与消融/对照/鲁棒性/优势分解/权重校准管线一致）
SEED = 42

#: 失败率格点：0 为无故障基准，1.0 为全断网
FAILURE_RATES: list[float] = [0.0, 0.3, 0.5, 1.0]

#: 生产策略名（三态 + 重试 + 缓存，走生产 HallucinationDetector 定罪路径）
POLICY_THREE_STATE = "three_state_resilient"

#: 社区现状基线名（二态直拒：任何非 valid 一律定罪；实验侧对照实现，非生产路径）
POLICY_TWO_STATE = "two_state_direct"

#: 全部策略格点（实验矩阵的行）
POLICIES: list[str] = [POLICY_THREE_STATE, POLICY_TWO_STATE]

#: PubMed esummary 端点（与 CitationVerifier._check_pubmed 一致）
_PUBMED_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


# ---------------------------------------------------------------------------
# 案例模型与生成器
# ---------------------------------------------------------------------------

@dataclass
class CitationCase:
    """引用验证实验案例（主张文本 + 声明引用列表 + 层级标签）。

    attributes:
        case_id: 案例唯一标识（层内顺序编号，保证报告可追溯）
        claim_text: 主张文本（Agent 证据的自然语言表述，触发幻觉检测管线）
        declared_refs: 主张声明的引用 ID 列表（全部为合法格式 PMID）
        stratum: 层级标签——"clean"=A 层（引用全部真实有效，正确裁决=放行），
                 "invalid_ref"=B 层（引用确认不存在，正确裁决=定罪）
    """

    case_id: str
    claim_text: str
    declared_refs: list[str]
    stratum: str


#: 共享热点 PMID 池大小：A 层全部案例从该池重复抽取引用，
#: 复现"跨案重复引用同一批经典文献"的真实形态（缓存效应的度量基础）
POOL_SIZE = 40

#: 池内 PMID 统一前缀（8800xxxx 为实验专用保留段，格式合法的 8 位纯数字）
_VALID_PMID_BASE = 88000001

#: B 层确认不存在 PMID 前缀（9900xxxx，格式合法但传输层返回"不存在"语义）
_INVALID_PMID_BASE = 99000001

#: A 层主张文本模板（轮转使用；裁决仅由声明引用的验证结果驱动，文本只提供语境）
_CLAIM_TEMPLATES: list[str] = [
    "研究表明 {pmid} 支持该靶点抑制剂在体模型中的活性结论",
    "文献显示依据 {pmid} 该化合物选择性窗口满足设计要求",
    "据报道 {pmid} 的动物实验数据与本项目假设一致",
]


def generate_cases(
    seed: int,
    n_clean: int,
    n_invalid: int,
) -> tuple[list[CitationCase], set[str], set[str]]:
    """确定性生成 A/B 两层实验案例集。

    生成纪律（与 DDB 各层生成器一致——"构造即验证"）：
    - A 层：每案从 40 个共享热点 PMID 池无放回抽取 2-4 条引用（跨案高重用），
      全部真实有效（传输层按 valid 语义应答），无任何真矛盾 → 正确裁决 = 放行；
    - B 层：每案声明 2 条确认不存在的 PMID（传输层按 invalid 语义应答）→
      正确裁决 = 定罪，作为三态降级"不牺牲真幻觉召回"的对照组。

    Args:
        seed: 随机种子（同 seed 逐案例可复现）
        n_clean: A 层（干净主张）案例数
        n_invalid: B 层（确认不存在引用）案例数

    Returns:
        (案例列表, 有效 PMID 集, 无效 PMID 集) 三元组；
        两个 PMID 集供注入器构造应答语义。
    """
    rng = random.Random(seed)

    # 共享热点引用池：POOL_SIZE 个格式合法 PMID（传输层将其应答为 valid）
    valid_pool = [str(_VALID_PMID_BASE + i) for i in range(POOL_SIZE)]
    # B 层专用确认不存在池：每案 2 条，总量 2*n_invalid，与有效池零交集
    invalid_pool = [str(_INVALID_PMID_BASE + i) for i in range(2 * n_invalid)]

    cases: list[CitationCase] = []

    # A 层：干净主张 + 全部真实有效的声明引用（共享池无放回抽取，2-4 条/案）
    for i in range(n_clean):
        n_refs = rng.randint(2, 4)
        refs = rng.sample(valid_pool, n_refs)
        # 主张文本按案例序号轮转模板，引用 ID 以顿号拼接嵌入
        template = _CLAIM_TEMPLATES[i % len(_CLAIM_TEMPLATES)]
        claim = template.format(pmid="、".join(refs))
        cases.append(CitationCase(
            case_id=f"clean-{i:03d}",
            claim_text=claim,
            declared_refs=refs,
            stratum="clean",
        ))

    # B 层：确认不存在引用（每案 2 条，正确裁决 = 定罪）
    for i in range(n_invalid):
        refs = invalid_pool[i * 2:(i + 1) * 2]
        template = _CLAIM_TEMPLATES[i % len(_CLAIM_TEMPLATES)]
        claim = template.format(pmid="、".join(refs))
        cases.append(CitationCase(
            case_id=f"invalid-{i:03d}",
            claim_text=claim,
            declared_refs=refs,
            stratum="invalid_ref",
        ))

    return cases, set(valid_pool), set(invalid_pool)


# ---------------------------------------------------------------------------
# 网络失败注入器（实验装置：httpx 官方 MockTransport 传输层接口）
# ---------------------------------------------------------------------------

class FlakyPubMedTransport:
    """确定性网络失败注入传输层（httpx.MockTransport 的异步 handler）。

    职责边界（实验装置，非业务 mock）：
    - 网络层模拟即本实验的处理变量——按概率 p 对 PubMed esummary 请求抛出
      httpx.ConnectError，模拟连接失败（DNS/超时/服务不可达同语义）；
    - 未命中失败时按引用真值应答：有效 PMID → 200 + result 含该 PMID（valid）；
      无效 PMID → 200 + result 为空（CitationVerifier._check_pubmed 据此判
      invalid，与 PubMed "引用不存在"的真实应答语义一致）；
    - 确定性保证：每次请求按固定顺序消耗 random.Random 的随机流；
      CitationVerifier.verify() 对引用列表顺序 await、重试循环顺序执行，
      单线程 asyncio 下整个格点的 RNG 消耗顺序固定 → 同 seed 逐计数可复现。

    attributes:
        failure_rate: 单次请求失败概率 p ∈ [0, 1]
        valid_pmids: 传输层应答为 valid 的 PMID 集
        rng: 确定性随机源（由格点种子派生，见 _rng_for_cell）
        total_requests / failed_requests / resolved_requests: 传输层计数器
    """

    def __init__(
        self,
        failure_rate: float,
        valid_pmids: set[str],
        rng: random.Random,
    ):
        """初始化注入器。

        Args:
            failure_rate: 单次请求失败概率（0=无故障基准，1.0=全断网）
            valid_pmids: 应答为 valid 的 PMID 集（其余合法 PMID 应答为不存在）
            rng: 确定性随机源（格点种子派生）
        """
        self.failure_rate = failure_rate
        self.valid_pmids = valid_pmids
        self.rng = rng
        # 传输层计数器（写入报告，供重试/缓存效应交叉核验）
        self.total_requests = 0
        self.failed_requests = 0
        self.resolved_requests = 0

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        """MockTransport 异步 handler：注入失败或按真值应答。

        Args:
            request: httpx 请求对象（CitationVerifier 经 AsyncClient 发出）

        Returns:
            未命中失败时的 200 应答（valid / 不存在两种语义）。

        Raises:
            httpx.ConnectError: 按概率 p 注入的连接失败（被生产重试逻辑捕获）。
        """
        # 本实验只走 PubMed esummary 端点；其他端点一律按网络失败处理
        if _PUBMED_URL not in str(request.url):
            self.total_requests += 1
            self.failed_requests += 1
            raise httpx.ConnectError("simulated network failure: unexpected endpoint")

        self.total_requests += 1

        # 按概率 p 注入连接失败——这是实验的核心处理变量
        if self.rng.random() < self.failure_rate:
            self.failed_requests += 1
            raise httpx.ConnectError(
                f"simulated network failure (p={self.failure_rate})"
            )

        # 未命中失败：解析 PMID 并按真值应答
        self.resolved_requests += 1
        pmid = request.url.params.get("id", "")
        if pmid in self.valid_pmids:
            # valid 语义：result 字典含该 PMID 且无 error 键
            return httpx.Response(200, json={"result": {pmid: {"uid": pmid}}})
        # 不存在语义：result 为空 → _check_pubmed 返回 False → status=invalid
        return httpx.Response(200, json={"result": {}})


def _rng_for_cell(seed: int, policy: str, failure_rate: float) -> random.Random:
    """为（策略 × 失败率）格点派生确定性随机源。

    用字符串元组做种子（Python random.Random 对 str 种子完全确定），
    保证不同格点独立、同格点跨进程可复现。

    Args:
        seed: 实验全局种子
        policy: 策略名
        failure_rate: 失败率格点

    Returns:
        该格点专用的 random.Random 实例。
    """
    return random.Random(f"{seed}-{policy}-{failure_rate}")


# ---------------------------------------------------------------------------
# 验证失败处置映射（实验的唯一变量）
# ---------------------------------------------------------------------------

def two_state_direct_mapping(results: dict[str, dict]) -> bool:
    """社区现状基线映射：二态直拒——任何非 valid 状态一律定罪。

    【对照基线声明】该函数是实验侧实现的"社区现状"对照（对应大纲 §2.2
    "现有工作把'验证失败'直接当'引用错误'"），**不是生产路径**，不注册进
    任何生产策略表；生产侧三态分级见 HallucinationDetector.detect。
    与生产策略的关系：消费同一 CitationVerifier.verify() 原始结果，
    唯一差异是失败处置映射（二态直拒 vs 三态分级）+ 重试/缓存配置
    （格点执行器控制），与 §6.2 对照基线方法论一致。

    二态语义：verify 结果被压缩为 {valid, invalid} 二值——network_failed
    （网络失败）与 format_invalid（格式问题）都被并入 invalid 直裁定罪，
    即"验证失败 = 引用错误"的有罪推定。

    Args:
        results: CitationVerifier.verify() 的结果映射
                 （reference_id → 含 status/valid 字段的结果字典）

    Returns:
        True=存在任一非 valid 引用 → 主张应被定罪；False=全部引用 valid。
    """
    return any(info.get("status") != "valid" for info in results.values())


# ---------------------------------------------------------------------------
# 格点执行器
# ---------------------------------------------------------------------------

def _blank_cell(policy: str, failure_rate: float) -> dict:
    """构造空格点统计结构（保证两策略报告字段完全同构，便于曲线绘图）。

    Args:
        policy: 策略名
        failure_rate: 失败率

    Returns:
        含全部指标字段、计数器归零的格点字典。
    """
    return {
        "policy": policy,
        "failure_rate": failure_rate,
        "n_clean": 0,
        "n_invalid": 0,
        "fpr_clean": 0.0,
        "recall_invalid": 0.0,
        "convicted_clean": 0,
        "convicted_invalid": 0,
        # 验证层过程指标（重试/缓存效应度量）
        "total_ref_lookups": 0,
        "avg_attempts": 0.0,
        "network_failed_rate": 0.0,
        "cache_hits": 0,
        # 传输层计数（与验证层指标交叉核验）
        "transport_total_requests": 0,
        "transport_failed_requests": 0,
        # 裁决分布（全案例）
        "verdict_distribution": {"convicted": 0, "acquitted": 0},
    }


async def run_policy_cell(
    policy: str,
    failure_rate: float,
    cases: list[CitationCase],
    valid_pmids: set[str],
    seed: int,
) -> dict:
    """执行单个（策略 × 失败率）格点，返回该格点全部指标。

    两策略的差异实现（唯一变量 = 失败处置策略）：
    - three_state_resilient：整个格点共享一个 CitationVerifier 实例
      （max_retries=3，缓存跨案生效——共享热点池被复用即缓存效应），
      定罪走生产 HallucinationDetector.detect 原始路径
      （invalid→CRITICAL 定罪，network_failed/format_invalid→WARNING 不定罪）；
    - two_state_direct：每案新建 CitationVerifier（max_retries=1，
      无跨案缓存——等价于"每次验证都直连、失败即定罪"的社区现状），
      定罪走 two_state_direct_mapping 基线映射。

    统计口径：
    - A 层 FPR = A 层被定罪案数 / A 层案数（干净主张被误判 = 假阳性）；
    - B 层召回 = B 层被定罪案数 / B 层案数（真幻觉被捕获）；
    - avg_attempts / network_failed_rate 对全部引用查找次数取均值；
    - cache_hits = 跨案重复引用次数（三态策略下 = 缓存命中数的结构下界，
      因首次查找后同 rid 必命中缓存；二态策略每案新建实例恒为 0）。

    Args:
        policy: 策略名（POLICY_THREE_STATE / POLICY_TWO_STATE）
        failure_rate: 失败率格点
        cases: 案例列表（A/B 两层混排，按 stratum 分层统计）
        valid_pmids: 有效 PMID 集（注入器应答语义）
        seed: 实验全局种子（格点随机源由此派生）

    Returns:
        该格点指标字典（结构见 _blank_cell）。

    Raises:
        ValueError: 未知策略名。
    """
    cell = _blank_cell(policy, failure_rate)
    rng = _rng_for_cell(seed, policy, failure_rate)
    # 注入器与 httpx 客户端（格点级；三态策略下客户端与验证器跨案共享）
    transport = FlakyPubMedTransport(failure_rate, valid_pmids, rng)

    clean_total = 0
    clean_convicted = 0
    invalid_total = 0
    invalid_convicted = 0
    # 验证层过程统计累加器
    attempts_sum = 0
    network_failed_count = 0
    total_lookups = 0
    cache_hits = 0
    # 已出现过的引用 rid 集（三态策略下用于计算跨案缓存命中结构下界）
    seen_rids: set[str] = set()

    if policy == POLICY_THREE_STATE:
        # 生产策略：格点级共享验证器（缓存跨案生效）+ 生产幻觉检测器
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(transport)
        ) as client:
            verifier = CitationVerifier(http_client=client, max_retries=3)
            detector = HallucinationDetector(verifier)
            for case in cases:
                # 过程统计：先显式跑一遍 verify 收集 status/attempts
                #（结果全部写入缓存，随后 detect 内部的 verify 全部命中缓存，
                #  不消耗额外网络尝试，保证统计与定罪消费同一批验证结果）
                stats = await verifier.verify(case.declared_refs)
                for rid in case.declared_refs:
                    total_lookups += 1
                    if rid in seen_rids:
                        cache_hits += 1
                    seen_rids.add(rid)
                    attempts_sum += stats[rid].get("attempts", 0)
                    if stats[rid].get("status") == "network_failed":
                        network_failed_count += 1

                # 定罪：生产 HallucinationDetector 原始路径
                #（invalid → CRITICAL；network_failed/format_invalid → WARNING）
                issues = await detector.detect(case.claim_text, case.declared_refs)
                convicted = any(
                    i.severity == "CRITICAL" and i.category == "hallucination"
                    for i in issues
                )
                if case.stratum == "clean":
                    clean_total += 1
                    clean_convicted += int(convicted)
                else:
                    invalid_total += 1
                    invalid_convicted += int(convicted)
                cell["verdict_distribution"][
                    "convicted" if convicted else "acquitted"
                ] += 1
    elif policy == POLICY_TWO_STATE:
        # 社区现状基线：每案新建验证器（1 次尝试、无跨案缓存）+ 基线映射
        for case in cases:
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(transport)
            ) as client:
                verifier = CitationVerifier(http_client=client, max_retries=1)
                stats = await verifier.verify(case.declared_refs)
            for rid in case.declared_refs:
                total_lookups += 1
                attempts_sum += stats[rid].get("attempts", 0)
                if stats[rid].get("status") == "network_failed":
                    network_failed_count += 1

            # 定罪：二态直拒映射（任何非 valid 一律定罪）
            convicted = two_state_direct_mapping(stats)
            if case.stratum == "clean":
                clean_total += 1
                clean_convicted += int(convicted)
            else:
                invalid_total += 1
                invalid_convicted += int(convicted)
            cell["verdict_distribution"][
                "convicted" if convicted else "acquitted"
            ] += 1
    else:
        raise ValueError(f"未知策略: {policy}")

    # 汇总格点指标（层内案数为 0 时比率置 0——分层实验允许单层缺失；
    # 除数非零时由构造保证：每案引用 ≥2，total_lookups 不可能为 0）
    cell["n_clean"] = clean_total
    cell["n_invalid"] = invalid_total
    cell["convicted_clean"] = clean_convicted
    cell["convicted_invalid"] = invalid_convicted
    cell["fpr_clean"] = clean_convicted / clean_total if clean_total else 0.0
    cell["recall_invalid"] = invalid_convicted / invalid_total if invalid_total else 0.0
    cell["total_ref_lookups"] = total_lookups
    cell["avg_attempts"] = attempts_sum / total_lookups
    cell["network_failed_rate"] = network_failed_count / total_lookups
    cell["cache_hits"] = cache_hits
    cell["transport_total_requests"] = transport.total_requests
    cell["transport_failed_requests"] = transport.failed_requests
    return cell


# ---------------------------------------------------------------------------
# 预注册红线检查
# ---------------------------------------------------------------------------

def check_red_lines(cells: list[dict]) -> list[dict]:
    """六条预注册红线的确定性检查（任何违例使实验退出码为 1）。

    红线清单（大纲 §6.5 落地段同步，预注册后不得依结果调整）：
    1. 三态策略 A 层 FPR = 0 对全部失败率格点（降级接线正确性的可证伪检验）；
    2. 三态策略 B 层召回：p=0 时 = 1.0（无故障时真幻觉必须全捕获）；
       p=0.3 时 ≥ 0.9（重试效应：有效失败率 0.3³≈0.027）；p=1.0 时 = 0
       （全断网 → 拒绝裁决，弃权而非误判——设计语义的结构性断言）；
    3. 二态直拒 A 层 FPR 随 p 单调不减、FPR(0.3) > 0、FPR(1.0) = 1.0
       （量化"基础设施故障污染科学裁决"的上界）；
    4. 二态直拒 B 层召回 = 1.0 对全部格点（直拒连断网都定罪——
       其高召回完全以假阳性为代价，方向性发现）；
    5. 确定性：p=0.5 格点双跑逐字段相等（同 seed 可复现）；
    6. 缓存效应：p=0.3 下三态策略跨案缓存命中数 > 0（共享热点池被复用）。

    Args:
        cells: 全部格点指标列表（含 p=0.5 的复跑格点，其 policy 名
               带 "（复跑）" 后缀仅用于区分，红线 5 在此比对）

    Returns:
        红线判定列表：每项 {"name", "passed", "detail"}。
    """
    results: list[dict] = []

    def _get(policy: str, rate: float) -> dict:
        """按（策略, 失败率）取格点；缺失即断言失败（格点矩阵完整性）。"""
        for c in cells:
            if c["policy"] == policy and abs(c["failure_rate"] - rate) < 1e-9:
                return c
        raise KeyError(f"缺少格点: policy={policy}, p={rate}")

    three = POLICY_THREE_STATE
    two = POLICY_TWO_STATE

    # 红线 1：三态 A 层 FPR 恒 0
    rates = sorted({c["failure_rate"] for c in cells if "（复跑）" not in c["policy"]})
    rl1_bad = [(three, p, _get(three, p)["fpr_clean"]) for p in rates
               if _get(three, p)["fpr_clean"] != 0.0]
    results.append({
        "name": "RL1-three-state-fpr-zero",
        "passed": not rl1_bad,
        "detail": ("全部格点 FPR=0" if not rl1_bad
                   else f"违例格点: {rl1_bad}"),
    })

    # 红线 2：三态 B 层召回分段断言（p=0 → 1.0；p=0.3 → ≥0.9；p=1.0 → 0）
    rl2_checks = [
        (0.0, _get(three, 0.0)["recall_invalid"] == 1.0),
        (0.3, _get(three, 0.3)["recall_invalid"] >= 0.9),
        (1.0, _get(three, 1.0)["recall_invalid"] == 0.0),
    ]
    rl2_bad = [(p, _get(three, p)["recall_invalid"]) for p, ok in rl2_checks if not ok]
    results.append({
        "name": "RL2-three-state-recall-profile",
        "passed": not rl2_bad,
        "detail": ("p=0 召回 1.0 / p=0.3 ≥0.9 / p=1.0 弃权 0 全部满足" if not rl2_bad
                   else f"违例格点: {rl2_bad}"),
    })

    # 红线 3：二态直拒 FPR 单调不减 + FPR(0.3)>0 + FPR(1.0)=1.0
    two_rates = [0.0, 0.3, 0.5, 1.0]
    fpr_seq = [_get(two, p)["fpr_clean"] for p in two_rates]
    monotone = all(fpr_seq[i] <= fpr_seq[i + 1] + 1e-12 for i in range(len(fpr_seq) - 1))
    rl3_ok = monotone and fpr_seq[1] > 0.0 and fpr_seq[3] == 1.0
    results.append({
        "name": "RL3-two-state-fpr-pollution-curve",
        "passed": rl3_ok,
        "detail": (f"FPR 序列 {fpr_seq} 满足单调/0.3>0/1.0=1.0" if rl3_ok
                   else f"FPR 序列 {fpr_seq} 违例（单调={monotone}）"),
    })

    # 红线 4：二态直拒 B 层召回恒 1.0
    rl4_bad = [p for p in two_rates if _get(two, p)["recall_invalid"] != 1.0]
    results.append({
        "name": "RL4-two-state-recall-always-one",
        "passed": not rl4_bad,
        "detail": ("全部格点召回=1.0" if not rl4_bad
                   else f"违例格点: {rl4_bad}"),
    })

    # 红线 5：确定性——p=0.5 双跑逐字段相等
    rerun_cells = [c for c in cells if c["policy"].endswith("（复跑）")]
    rl5_ok = True
    rl5_detail = "p=0.5 双跑逐字段相等"
    if not rerun_cells:
        rl5_ok = False
        rl5_detail = "缺少复跑格点（实验矩阵构造缺陷）"
    else:
        for rr in rerun_cells:
            base_policy = rr["policy"].replace("（复跑）", "")
            orig = _get(base_policy, 0.5)
            for key in orig:
                if key == "policy":
                    continue
                if orig[key] != rr.get(key):
                    rl5_ok = False
                    rl5_detail = f"字段 {key} 不一致: {orig[key]} vs {rr.get(key)}"
                    break
            if not rl5_ok:
                break
    results.append({"name": "RL5-determinism-rerun", "passed": rl5_ok, "detail": rl5_detail})

    # 红线 6：缓存效应——p=0.3 三态策略缓存命中数 > 0
    rl6_ok = _get(three, 0.3)["cache_hits"] > 0
    results.append({
        "name": "RL6-cache-effect",
        "passed": rl6_ok,
        "detail": (f"缓存命中 {_get(three, 0.3)['cache_hits']} 次 > 0" if rl6_ok
                   else "p=0.3 三态策略缓存命中数为 0（共享池复用未生效）"),
    })

    return results


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def _fmt(v: float) -> str:
    """浮点格式化（百分比四位小数，红线汇总表用）。"""
    return f"{v:.4f}"


async def run_experiment(
    seed: int,
    n_clean: int,
    n_invalid: int,
) -> dict:
    """执行完整实验矩阵并返回报告字典。

    矩阵 = 2 策略 × 4 失败率格点 + p=0.5 双跑复跑格点（红线 5）；
    案例集由 generate_cases 确定性生成，全格点共享同一批案例
    （保证跨格点可比性——唯一变量是失败率与处置策略）。

    Args:
        seed: 实验全局种子
        n_clean: A 层案例数
        n_invalid: B 层案例数

    Returns:
        报告字典（meta / cells / red_lines / all_passed）。
    """
    cases, valid_pmids, _invalid_pmids = generate_cases(seed, n_clean, n_invalid)

    cells: list[dict] = []
    for policy in POLICIES:
        for rate in FAILURE_RATES:
            cell = await run_policy_cell(policy, rate, cases, valid_pmids, seed)
            cells.append(cell)

    # 红线 5 专用复跑：p=0.5 两策略各重跑一次（独立格点随机源同种子派生 →
    # 与首次运行完全一致才通过）
    for policy in POLICIES:
        rr = await run_policy_cell(policy, 0.5, cases, valid_pmids, seed)
        rr["policy"] = f"{policy}（复跑）"
        cells.append(rr)

    red_lines = check_red_lines(cells)
    report = {
        "meta": {
            "experiment": "network_failure_discrimination_v0414",
            "outline_section": "§6.5 网络失败区分实验（P0-2 核心，Fig.4 数据源）",
            "seed": seed,
            "n_clean": n_clean,
            "n_invalid": n_invalid,
            "failure_rates": FAILURE_RATES,
            "policies": {
                POLICY_THREE_STATE: "三态+重试3+跨案缓存，生产 HallucinationDetector 定罪路径",
                POLICY_TWO_STATE: "二态直拒（社区现状对照，实验侧实现）：1 次尝试、无跨案缓存、非 valid 即定罪",
            },
            "generated_at": datetime.now().isoformat(),
        },
        "cells": cells,
        "red_lines": red_lines,
        "all_passed": all(r["passed"] for r in red_lines),
    }
    return report


def main() -> int:
    """命令行入口：跑实验 → 落盘报告 → 打印摘要表 → 按红线结果返回退出码。

    Returns:
        0=全部红线满足；1=红线违例；2=运行异常。
    """
    parser = argparse.ArgumentParser(description="§6.5 网络失败区分实验（v0.4.14）")
    parser.add_argument("--seed", type=int, default=SEED, help="实验全局种子")
    parser.add_argument("--cases", type=int, default=60, help="A 层干净主张案例数")
    parser.add_argument("--invalid-cases", type=int, default=20, help="B 层对照案例数")
    parser.add_argument(
        "--output", type=str,
        default=str(_BACKEND_ROOT / "tests" / "ddb" / "network_failure_report.json"),
        help="报告 JSON 输出路径",
    )
    args = parser.parse_args()

    try:
        report = asyncio.run(run_experiment(args.seed, args.cases, args.invalid_cases))
    except Exception as exc:  # noqa: BLE001 —— 顶层兜底：运行异常统一退出码 2
        print(f"[network-failure-exp] 运行异常: {exc}", file=sys.stderr)
        return 2

    # 报告落盘（Fig.4 绘图数据源）
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 控制台摘要表：p × 策略 → 核心指标
    print("=" * 96)
    print("§6.5 网络失败区分实验（三态+重试+缓存 vs 二态直拒）——Fig.4 数据源")
    print(f"seed={args.seed}  A层(干净)={args.cases} 案  B层(确认不存在引用)={args.invalid_cases} 案")
    print("=" * 96)
    header = (
        f"{'策略':<24}{'p':>5}{'FPR(A层)':>10}{'召回(B层)':>10}"
        f"{'均尝试':>8}{'netfail率':>10}{'缓存命中':>9}"
    )
    print(header)
    print("-" * 96)
    for cell in report["cells"]:
        print(
            f"{cell['policy']:<24}{cell['failure_rate']:>5.1f}"
            f"{_fmt(cell['fpr_clean']):>10}{_fmt(cell['recall_invalid']):>10}"
            f"{cell['avg_attempts']:>8.2f}{_fmt(cell['network_failed_rate']):>10}"
            f"{cell['cache_hits']:>9}"
        )
    print("=" * 96)
    print("预注册红线判定：")
    for rl in report["red_lines"]:
        mark = "PASS" if rl["passed"] else "FAIL"
        print(f"  [{mark}] {rl['name']}: {rl['detail']}")
    print("=" * 96)

    if report["all_passed"]:
        print(f"全部红线满足，报告已写入: {output_path}")
        return 0
    print("存在红线违例！", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
