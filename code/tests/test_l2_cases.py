"""
DDB-L2 文献真实争议基准 —— 单元测试

覆盖范围：
    1. l2_cases.load_cases()：结构自检（case_id/evidence_id 唯一、金标对
       引用存在）、strict_cross_source 模式、深拷贝隔离；
    2. l2_cases.citation_audit() / pending_citations()：审计清单结构与
       待复核队列（诚实边界：首批 pending 必须恰为声明的 3 条）；
    3. H2b 换源召回集成：L2 正例上跑 ABP（AsymmetricRubric）与有罪推定
       （RuleEngine.screen），断言 ABP 案例级召回 = 100%，并验证
       "金标对至少一条被关键词通道检出"（L2-04/L2-06 故意含
       agent_opinion 无引用证据——有罪推定与 ABP 的真实分歧样本）。

运行方式（在 backend/ 目录下）：
    python -m pytest tests/ddb/test_l2_cases.py -v

依赖说明：
    与 test_pilot_h2a.py 相同：AsymmetricRubric / RuleEngine / EvidenceScorer
    均为纯内存对象，导入 app 前注入 fail-closed 校验所需的占位环境变量。
"""

from __future__ import annotations

import json
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
_SCRIPTS_DIR = _BACKEND_ROOT / "tests" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

os.environ.setdefault("SCIMINER_API_KEY", "ddb-l2-placeholder")
os.environ.setdefault("DB_PASSWORD", "ddb-l2-placeholder")

from app.core.virtual_lab.consensus_court import (  # noqa: E402
    Evidence,
    EvidenceScorer,
    RuleEngine,
)
from app.core.virtual_lab.rubric_engine import AsymmetricRubric  # noqa: E402
from tests.ddb.l2_cases import (  # noqa: E402
    CITATION_VERIFIED_AT,
    L2_CASES,
    citation_audit,
    load_cases,
    pending_citations,
)


# ---------------------------------------------------------------------------
# 案例 dict → Evidence 对象（与 test_pilot_h2a._to_evidence 同构）
# ---------------------------------------------------------------------------

def _to_evidence(ev: dict) -> Evidence:
    """
    把 L2 案例中的证据字典无损还原为 consensus_court.Evidence 对象。

    Args:
        ev: L2 案例中的证据字典

    Returns:
        consensus_court.Evidence 实例
    """
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


# ---------------------------------------------------------------------------
# 两种策略的案例级判定（复用 test_pilot_h2a 模式，全部真实调用源码 API）
# ---------------------------------------------------------------------------

def run_guilty_presumption(case: dict) -> bool:
    """
    有罪推定策略：任一证据被 RuleEngine.screen() 报出问题即案例级拒绝。

    Args:
        case: L2 案例字典

    Returns:
        True 表示案例被拒绝
    """
    engine = RuleEngine()
    for ev_dict in case["evidences"]:
        if engine.screen(_to_evidence(ev_dict)):
            return True
    return False


def run_abp(case: dict) -> bool:
    """
    非对称举证策略（ABP）：EvidenceScorer 逐条评分 + AsymmetricRubric 评估，
    RubricVerdict.passed=False（检出矛盾对）即案例级拒绝。

    Args:
        case: L2 案例字典

    Returns:
        True 表示案例被拒绝
    """
    scorer = EvidenceScorer()
    rubric = AsymmetricRubric()
    evidences = [_to_evidence(ev) for ev in case["evidences"]]
    scores = [scorer.score(ev, case["topic"]) for ev in evidences]
    evidence_map = {ev.evidence_id: ev for ev in evidences}
    verdict = rubric.evaluate(scores, evidence_map, case["topic"])
    return not verdict.passed


def run_abp_contradiction_ids(case: dict) -> set[frozenset[str]]:
    """
    运行 ABP 并返回检出的矛盾对集合（frozenset 无序对，便于与金标对齐）。

    Args:
        case: L2 案例字典

    Returns:
        检出矛盾对的无序 evidence_id 集合
    """
    scorer = EvidenceScorer()
    rubric = AsymmetricRubric()
    evidences = [_to_evidence(ev) for ev in case["evidences"]]
    scores = [scorer.score(ev, case["topic"]) for ev in evidences]
    evidence_map = {ev.evidence_id: ev for ev in evidences}
    verdict = rubric.evaluate(scores, evidence_map, case["topic"])
    return {
        frozenset((c.evidence_a_id, c.evidence_b_id))
        for c in verdict.contradictions
    }


# ---------------------------------------------------------------------------
# 基准缓存（模块级，多个测试共享同一份样本）
# ---------------------------------------------------------------------------

_CASES_CACHE: Optional[list[dict]] = None


def _cases() -> list[dict]:
    """加载首批 L2 基准（模块级缓存）。"""
    global _CASES_CACHE
    if _CASES_CACHE is None:
        _CASES_CACHE = load_cases()
    return _CASES_CACHE


# ---------------------------------------------------------------------------
# 1. l2_cases 结构自检
# ---------------------------------------------------------------------------

class TestL2Structure:
    """L2 基准结构与数据忠实性断言。"""

    def test_load_cases_returns_ten_positive_cases(self) -> None:
        """首批应为 10 例、全部正例（with_contradiction）、含金标对。"""
        cases = _cases()
        assert len(cases) == 10
        for case in cases:
            assert case["polarity"] == "with_contradiction"
            assert case["evidence_type"] == "l2_real"
            assert case["defect_flags"] == []
            assert case["ground_truth"]["has_true_contradiction"] is True
            assert case["ground_truth"]["defect_only"] is False
            assert case["ground_truth"]["contradiction_pairs"], case["case_id"]
            assert case["ground_truth"]["expert_adjudication"] == "pending"

    def test_load_cases_strict_cross_source_passes(self) -> None:
        """strict 模式：全部金标矛盾对必须跨来源（可被关键词通道检出）。"""
        cases = load_cases(strict_cross_source=True)
        assert len(cases) == 10

    def test_load_cases_returns_deep_copy(self) -> None:
        """load_cases 返回深拷贝，修改不应污染模块级 L2_CASES。"""
        cases = load_cases()
        cases[0]["evidences"][0]["content"] = "被污染的内容"
        cases[0]["topic"] = "被污染的 topic"
        assert L2_CASES[0]["topic"] != "被污染的 topic"
        assert L2_CASES[0]["evidences"][0]["content"] != "被污染的内容"

    def test_evidence_ids_globally_unique_within_case(self) -> None:
        """案例内 evidence_id 唯一（load_cases 已断言，此处再验证返回值）。"""
        for case in _cases():
            ids = [ev["evidence_id"] for ev in case["evidences"]]
            assert len(ids) == len(set(ids)), case["case_id"]

    def test_metadata_fidelity_fields_present(self) -> None:
        """每条证据的 metadata 均带 paper_title/journal/year/stance 审计字段。"""
        for case in _cases():
            for ev in case["evidences"]:
                md = ev["metadata"]
                for key in ("paper_title", "paper_journal", "paper_year",
                            "stance", "citation_status"):
                    assert key in md, (case["case_id"], ev["evidence_id"], key)
                assert md["stance"] in ("A", "B", "neutral", "both")

    def test_citation_audit_row_count(self) -> None:
        """审计清单应覆盖全部证据条目。"""
        rows = citation_audit()
        total_ev = sum(len(c["evidences"]) for c in L2_CASES)
        assert len(rows) == total_ev

    def test_pending_citations_exact_expected_set(self) -> None:
        """
        待复核引用必须恰为声明的 1 条（诚实边界回归锁）。

        agent_opinion 类证据（L2-04-EV-04 / L2-06-EV-03）的
        citation_status="opinion" 属设计内故意无引用（Guilty 分歧种子），
        不在 pending 队列。新增 pending 引用属正常扩展，但必须同步更新
        本断言与 l2_cases.py 文件头的数据忠实性声明，防止静默漂移。
        """
        pend = {(r["case_id"], r["evidence_id"]) for r in pending_citations()}
        expected = {
            ("DDB-L2-0005", "L2-05-EV-02"),  # AF3-GPCR 第三方评测无 PMID
        }
        assert pend == expected


# ---------------------------------------------------------------------------
# 2. H2b 集成：ABP 在 L2 真实文献争议上的换源召回
# ---------------------------------------------------------------------------

class TestL2H2bRecall:
    """H2b（换源召回）：L2 正例上 ABP 必须保持高召回。"""

    def test_abp_rejects_all_positive_cases(self) -> None:
        """10/10 正例全部被 ABP 检出矛盾并拒绝（案例级召回 = 1.0）。"""
        missed = [c["case_id"] for c in _cases() if not run_abp(c)]
        assert not missed, f"ABP 漏检案例: {missed}"

    def test_abp_detects_at_least_one_gold_pair_per_case(self) -> None:
        """每个案例的金标矛盾对中至少一对被 ABP 关键词通道检出。"""
        undetected = []
        for case in _cases():
            detected = run_abp_contradiction_ids(case)
            gold = {
                frozenset((a, b))
                for a, b in case["ground_truth"]["contradiction_pairs"]
            }
            if not (detected & gold):
                undetected.append((case["case_id"], gold, detected))
        assert not undetected, f"金标对未被检出的案例: {undetected}"

    def test_guilty_presumption_diverges_on_agent_opinion_seeds(self) -> None:
        """
        L2-04 / L2-06 故意含 agent_opinion 无引用证据：
        有罪推定应命中（形式缺陷即拒绝），ABP 仅因内容矛盾才拒绝——
        两策略在此类样本上行为一致（均拒绝），但 ABP 的拒绝理由必须是
        内容矛盾而非形式缺陷（通过 rubric verdict 验证矛盾对被检出）。
        """
        seed_ids = {"DDB-L2-0004", "DDB-L2-0006"}
        for case in _cases():
            if case["case_id"] not in seed_ids:
                continue
            assert run_guilty_presumption(case), case["case_id"]
            assert run_abp_contradiction_ids(case), case["case_id"]


# ---------------------------------------------------------------------------
# 3. l2_cases.json 文件 ↔ 代码一致性锁（v0.4.15，§6.1 L2 文件指针）
# ---------------------------------------------------------------------------

class TestL2JsonFileConsistency:
    """
    落盘的 tests/ddb/l2_cases.json 必须与 l2_cases.py 当前版本逐案例一致。

    JSON 文件由 tests/scripts/export_l2_cases.py 从代码单一事实源确定性
    导出（无运行时时间戳 → 同代码版本重复导出字节级一致）。本测试锁死
    "文件随代码同步"：l2_cases.py 案例内容变更而未重跑导出时失败，
    防止论文引用的 JSON 与代码基准静默漂移。
    """

    def _load_json(self) -> dict:
        """读取落盘的 l2_cases.json（文件缺失时给出可操作的失败信息）。"""
        path = Path(__file__).resolve().parent / "l2_cases.json"
        assert path.exists(), (
            f"{path} 不存在：请运行 "
            ".venv/bin/python tests/scripts/export_l2_cases.py 重新导出"
        )
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def test_json_metadata_matches_code(self) -> None:
        """元信息（核验日/案例数/极性分布）必须与代码常量一致。"""
        payload = self._load_json()
        assert payload["generator"] == "tests/ddb/l2_cases.py"
        assert payload["citation_verified_at"] == CITATION_VERIFIED_AT
        assert payload["n_cases"] == len(L2_CASES)
        expected_dist: dict[str, int] = {}
        for case in L2_CASES:
            expected_dist[case["polarity"]] = expected_dist.get(case["polarity"], 0) + 1
        assert payload["polarity_distribution"] == expected_dist

    def test_json_cases_deep_equal_code(self) -> None:
        """全量深比较：JSON cases 与代码 L2_CASES（经 load_cases 深拷贝 +
        JSON 往返规范化）逐案例、逐证据、逐字段相等（含 metadata 审计
        字段与金标对；元组金标对按 JSON 语义规范化为列表后比较）。"""
        from export_l2_cases import _jsonify

        payload = self._load_json()
        assert payload["cases"] == _jsonify(load_cases())

    def test_json_export_is_deterministic(self) -> None:
        """确定性：同代码版本重复导出的载荷与已落盘文件完全一致
        （导出脚本无运行时时间戳/随机性，保证可复现）。"""
        from export_l2_cases import build_export_payload

        assert build_export_payload() == self._load_json()
