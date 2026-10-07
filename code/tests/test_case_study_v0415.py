"""
端到端案例研究（v0.4.15，大纲 §6.5 DDB-L2 落地版）单元测试。

覆盖范围：
    1. 报告结构锁：精选案例齐全、七环节时间线、ABP 收敛点、裁决链、
       证据评分明细、初筛过滤留痕、金标对检测、诚实性声明（DDB 来源
       而非平台生产运行）逐字段存在且类型正确；
    2. 确定性：build_report 双跑除 generated_at 外逐字段一致
       （报告无随机性，同代码版本可复现）；
    3. 裁决与既有 L2 锁一致：
       - 僵局型 L2-0001（零过滤）：定罪结果与 rubric 直评（H2b 锁
         10/10 定罪）一致，且金标矛盾对被关键词通道检出；
       - 升级型 L2-0005（全过滤，AF3 禁令）：法庭弃权但必须升级人工
         （正例不得静默放行），初筛留痕须给出禁令命中依据。

运行：backend$ .venv/bin/pytest tests/test_case_study_v0415.py -v
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

from run_case_study import (  # noqa: E402
    SELECTED_CASES,
    build_report,
)
from tests.ddb.l2_cases import L2_CASES  # noqa: E402

#: 七环节时间线的步骤名（与 run_case_study._TIMELINE_STEPS 对齐）
EXPECTED_STEPS = [
    "step1_legacy_screen",
    "step2_evidence_scoring",
    "step3_abp_rubric",
    "step4_pi_arbitration",
    "step5_independent_review",
    "step6_boss_override",
    "step7_escalation",
]


@pytest.fixture(scope="module")
def report() -> dict:
    """构建一次报告供全模块共享（build_report 为确定性纯计算）。"""
    return build_report()


# ---------------------------------------------------------------------------
# 1. 报告结构锁
# ---------------------------------------------------------------------------

class TestCaseStudyStructure:
    """报告顶层与逐案例结构锁。"""

    def test_top_level_keys(self, report: dict) -> None:
        """顶层键齐全：实验名 / 诚实性声明 / 法院模式 / 精选清单 / 案例条目。"""
        for key in (
            "experiment", "honesty_note", "court_mode",
            "selected_cases", "case_studies", "generated_at",
        ):
            assert key in report, key

    def test_honesty_note_labels_ddb_source(self, report: dict) -> None:
        """诚实性声明必须明确 DDB-L2 来源且声明非平台生产运行。"""
        note = report["honesty_note"]
        assert "DDB-L2" in note
        assert "非平台生产运行" in note
        assert "群聊 5 轮僵局" in note  # 对大纲 §6.5 环节的如实替代说明

    def test_selected_cases_are_deadlock_and_escalation(self, report: dict) -> None:
        """精选案例 = 僵局型 + 升级型各一，选择理由非空（可审计）。"""
        roles = [c["study_role"] for c in report["selected_cases"]]
        assert any("deadlock" in r for r in roles)
        assert any("escalation" in r for r in roles)
        for c in report["selected_cases"]:
            assert c["selection_rationale"].strip(), c["case_id"]

    def test_entry_structure_complete(self, report: dict) -> None:
        """逐案例条目：七要素齐备（时间线/收敛点/裁决链/评分/过滤/金标/元信息）。"""
        for entry in report["case_studies"]:
            for key in (
                "case_id", "study_role", "selection_rationale", "topic",
                "dispute_sides", "timeline", "abp_convergence",
                "verdict_chain", "evidence_scores", "filtered_audit",
                "gold_pair_detection",
            ):
                assert key in entry, (entry.get("case_id"), key)

    def test_timeline_has_seven_ordered_steps(self, report: dict) -> None:
        """时间线必须按流水线顺序覆盖七个环节（与 run_court_headless 编排对应）。"""
        for entry in report["case_studies"]:
            steps = [t["step"] for t in entry["timeline"]]
            assert steps == EXPECTED_STEPS, entry["case_id"]

    def test_verdict_chain_fields(self, report: dict) -> None:
        """裁决链字段齐全且类型正确（convicted/escalated 布尔、决策/理由留痕）。"""
        for entry in report["case_studies"]:
            vc = entry["verdict_chain"]
            assert isinstance(vc["convicted"], bool)
            assert isinstance(vc["escalated"], bool)
            assert vc["decision"] in ("支持", "驳回提案", "有条件支持", "无法裁决")
            assert isinstance(vc["reviewer_notes"], list)
            assert isinstance(vc["override_items"], list)
            assert vc["pi_rationale"].strip()


# ---------------------------------------------------------------------------
# 2. 确定性
# ---------------------------------------------------------------------------

class TestCaseStudyDeterminism:
    """报告双跑一致性（同代码版本可复现）。"""

    def test_double_run_identical_except_timestamp(self, report: dict) -> None:
        """build_report 双跑除 generated_at 外逐字段相等。"""
        second = build_report()
        first_no_ts = {k: v for k, v in report.items() if k != "generated_at"}
        second_no_ts = {k: v for k, v in second.items() if k != "generated_at"}
        assert first_no_ts == second_no_ts


# ---------------------------------------------------------------------------
# 3. 裁决与既有 L2 锁一致
# ---------------------------------------------------------------------------

class TestCaseStudyConsistencyWithL2Locks:
    """裁决结果与 test_l2_cases（H2b 换源召回锁）的口径对齐。"""

    def _entry(self, report: dict, case_id: str) -> dict:
        """按 case_id 取案例条目（精选清单必须覆盖）。"""
        for entry in report["case_studies"]:
            if entry["case_id"] == case_id:
                return entry
        raise AssertionError(f"报告缺少精选案例 {case_id}")

    def test_deadlock_case_convicted_with_gold_pair(self, report: dict) -> None:
        """僵局型 L2-0001：零过滤 → 定罪与 H2b 锁（10/10 定罪）一致，
        金标矛盾对 L2-01-EV-02/L2-01-EV-03 被关键词通道检出。"""
        entry = self._entry(report, "DDB-L2-0001")
        assert entry["filtered_audit"] == []  # 零过滤前提
        assert entry["verdict_chain"]["convicted"] is True
        assert entry["abp_convergence"]["max_intensity"] == 3
        assert entry["gold_pair_detection"]["gold_pair_detected"] is True
        assert "L2-01-EV-02/L2-01-EV-03" in entry["gold_pair_detection"]["detected_gold_pairs"]

    def test_escalation_case_abstains_and_escalates(self, report: dict) -> None:
        """升级型 L2-0005：全证据命中 AF3 禁令被初筛过滤 → 法庭弃权
        （无法裁决）但必须升级人工（正例不得静默放行），过滤留痕给出依据。"""
        entry = self._entry(report, "DDB-L2-0005")
        vc = entry["verdict_chain"]
        # 全过滤：留痕对账（架构师审查加固）——过滤数 + 评分数必须等于
        # L2 源案例证据总数，且全过滤时评分为空、留痕数恰等于总数。
        # 原断言 `filtered == (filtered + scores) - scores` 为同义反复，
        # 未对账证据总数，现以代码单一事实源 L2_CASES 为对账基准。
        total_evidences = len(
            next(c for c in L2_CASES if c["case_id"] == "DDB-L2-0005")["evidences"]
        )
        assert len(entry["filtered_audit"]) + len(entry["evidence_scores"]) == total_evidences
        assert len(entry["evidence_scores"]) == 0
        assert len(entry["filtered_audit"]) == total_evidences
        assert len(entry["filtered_audit"]) > 0
        for row in entry["filtered_audit"]:
            assert "结构预测工具禁令" in "".join(row["screen_issues"]), row
        # 弃权 + 升级：不静默放行
        assert vc["convicted"] is False
        assert vc["decision"] == "无法裁决"
        assert vc["escalated"] is True

    def test_selected_cases_match_registry(self, report: dict) -> None:
        """报告精选清单与 SELECTED_CASES 注册表逐键一致（防止双处漂移）。"""
        ids = [c["case_id"] for c in report["selected_cases"]]
        assert sorted(ids) == sorted(SELECTED_CASES)
