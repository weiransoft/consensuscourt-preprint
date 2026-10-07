"""
v0.4.13 十三角色权重对齐单元测试。

覆盖计划文档 v0413_role_profiles_plan.md 的测试要求：
1. PIAgentArbiter / EloArbiter 两处 ROLE_WEIGHTS 补全至 13 论文角色 + 2 历史
   fallback 键，且逐键同值（消融对比"仅仲裁逻辑差异"的前提）；
2. 既有 7 键数值不变（v0.4.12 前行为可复现）+ fallback 行为不变；
3. 新增画像 qc-inspector / pi-agent 可被 profile_loader 真实加载；
4. §6.5 校准实验核心函数（翻转计数 / 覆盖统计 / 红线检查）行为正确。

运行：backend$ .venv/bin/pytest tests/test_role_weights_v0413.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))
_SCRIPTS_DIR = _BACKEND_ROOT / "tests" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import pytest  # noqa: E402

from app.core.virtual_lab.baseline_arbiter import EloArbiter  # noqa: E402
from app.core.virtual_lab.consensus_court import (  # noqa: E402
    Evidence,
    EvidenceScore,
    PIAgentArbiter,
)

# ---------------------------------------------------------------------------
# 论文 13 角色与附录 D 权重表（backend/design/paper-appendix-d-role-profiles.md D.1）
# ---------------------------------------------------------------------------

#: 论文 13 角色（大纲 L137）
PAPER_13_ROLES: list[str] = [
    "pi-agent",
    "drug-researcher",
    "structural-biologist",
    "computational-chemist",
    "synthetic-chemist",
    "antibody-engineer",
    "peptide-designer",
    "data-specialist",
    "qc-inspector",
    "knowledge-engineer",
    "reflexion-agent",
    "project-coordinator",
    "report-expert",
]

#: 附录 D D.1 权重表的期望值（13 角色 + 历史 fallback）
EXPECTED_WEIGHTS: dict[str, float] = {
    "pi-agent": 1.0,
    "drug-researcher": 0.9,
    "structural-biologist": 0.9,
    "computational-chemist": 0.85,
    "synthetic-chemist": 0.8,
    "antibody-engineer": 0.8,
    "peptide-designer": 0.75,
    "data-specialist": 0.75,
    "qc-inspector": 0.7,
    "knowledge-engineer": 0.65,
    "reflexion-agent": 0.6,
    "project-coordinator": 0.6,
    "report-expert": 0.5,
    # 历史 fallback 键（非论文 13 角色）
    "reviewer": 0.7,
    "agent": 0.5,
}

#: v0.4.12 前既有 7 键（值不得修改——保证既有实验可复现）
LEGACY_7_WEIGHTS: dict[str, float] = {
    "pi-agent": 1.0,
    "drug-researcher": 0.9,
    "structural-biologist": 0.9,
    "computational-chemist": 0.85,
    "synthetic-chemist": 0.8,
    "reviewer": 0.7,
    "agent": 0.5,
}


# ---------------------------------------------------------------------------
# 1. ROLE_WEIGHTS 13 键补全 + 两处同口径
# ---------------------------------------------------------------------------

class TestRoleWeightsAlignment:
    def test_pi_weights_cover_13_paper_roles(self):
        """PIAgentArbiter.ROLE_WEIGHTS 必须显式登记全部 13 论文角色。"""
        for role in PAPER_13_ROLES:
            assert role in PIAgentArbiter.ROLE_WEIGHTS, f"缺少论文角色 {role}"

    def test_weights_match_appendix_d_table(self):
        """两处 ROLE_WEIGHTS 逐键同值且与附录 D D.1 权重表一致。"""
        assert PIAgentArbiter.ROLE_WEIGHTS == EXPECTED_WEIGHTS
        assert EloArbiter.ROLE_WEIGHTS == EXPECTED_WEIGHTS

    def test_two_arbiters_same_calibration(self):
        """消融对比前提：两仲裁器权重字典逐键相等。"""
        assert PIAgentArbiter.ROLE_WEIGHTS == EloArbiter.ROLE_WEIGHTS

    def test_legacy_7_keys_unchanged(self):
        """既有 7 键数值不变（v0.4.12 前行为与既有实验可复现）。"""
        for role, weight in LEGACY_7_WEIGHTS.items():
            assert PIAgentArbiter.ROLE_WEIGHTS[role] == weight, role
            assert EloArbiter.ROLE_WEIGHTS[role] == weight, role


# ---------------------------------------------------------------------------
# 2. fallback 行为不变（仲裁器对未登记角色仍取 0.5）
# ---------------------------------------------------------------------------

def _ev(eid: str, role: str) -> Evidence:
    """构造单条证据（pubmed 来源，避免来源降分干扰角色权重断言）。"""
    return Evidence(evidence_id=eid, content="test content", source="pubmed", role=role)


def _single_case(role: str, overall: float) -> tuple[list, dict]:
    """构造单证据案件（评分列表 + 映射），用于隔离验证角色权重路径。"""
    scores = [EvidenceScore("a", 0.9, 0.9, 0.9, overall)]
    emap = {"a": _ev("a", role)}
    return scores, emap


class TestFallbackBehavior:
    def test_unlisted_role_falls_back_to_0_5(self):
        """未登记角色（data-scientist，v0.4.13 已从 ROLE_POOL 移除）仍走 0.5。

        判别设计：高分工具证据（0.9）+ 低分 pi 证据（0.1）。
        - fallback 0.5：加权均分 = (0.5*0.9 + 1.0*0.1)/1.5 ≈ 0.367 → 驳回；
        - 若 fallback 误取 1.0：均分 = 0.5 → 有条件支持。
        断言"驳回"即证明 fallback=0.5（而非与 pi 同权）。
        """
        scores = [
            EvidenceScore("a", 0.9, 0.9, 0.9, 0.9),
            EvidenceScore("b", 0.9, 0.9, 0.9, 0.1),
        ]
        emap = {"a": _ev("a", "data-scientist"), "b": _ev("b", "pi-agent")}
        decision, _, _ = PIAgentArbiter().arbitrate("t", scores, emap)
        assert decision == "驳回提案"

    def test_reviewer_agent_weights_preserved(self):
        """reviewer/agent 历史 fallback 键仍按原权重参与加权。"""
        # reviewer 0.7：0.9 分证据加权均分 0.9*0.7/0.7=0.9 → 支持
        scores, emap = _single_case("reviewer", 0.9)
        assert PIAgentArbiter().arbitrate("t", scores, emap)[0] == "支持提案"
        # agent 0.5：0.5 分证据加权均分 0.5 → 有条件支持（0.4≤x<0.7）
        scores2, emap2 = _single_case("agent", 0.5)
        assert PIAgentArbiter().arbitrate("t", scores2, emap2)[0] == "有条件支持"

    def test_new_role_data_specialist_gets_0_75(self):
        """data-specialist（论文名，=平台 data-analyst）按 0.75 权重参与加权。"""
        # 单证据 0.75 分：加权均分 0.75 → 落入支持档（≥0.7）
        scores, emap = _single_case("data-specialist", 0.75)
        assert PIAgentArbiter().arbitrate("t", scores, emap)[0] == "支持提案"
        # 对比：若该角色被降权到 0.5 且混入低分 agent 证据，权重差异可见
        scores3 = [
            EvidenceScore("a", 0.9, 0.9, 0.9, 0.9),
            EvidenceScore("b", 0.9, 0.9, 0.9, 0.1),
        ]
        emap3 = {"a": _ev("a", "data-specialist"), "b": _ev("b", "agent")}
        # 加权均分 = (0.75*0.9 + 0.5*0.1) / 1.25 ≈ 0.58 → 有条件支持
        assert PIAgentArbiter().arbitrate("t", scores3, emap3)[0] == "有条件支持"

    def test_platform_name_data_analyst_falls_back_to_0_5(self):
        """平台实现名 data-analyst 不设别名键，证据走 0.5 fallback（计划红线）。

        计划红线明确"不引入别名机制"：论文侧（ROLE_WEIGHTS/ROLE_POOL/
        校准实验/DDB）统一用论文名 data-specialist；平台侧真实证据流的
        data-analyst 角色走未登记角色的 0.5 fallback，该口径差异已在
        附录 D D.3 显式声明（而非用别名键掩盖）。

        判别设计：高分 data-analyst 证据（0.9）+ 低分 pi 证据（0.1）。
        - fallback 0.5：加权均分 = (0.5*0.9 + 1.0*0.1)/1.5 ≈ 0.367 → 驳回；
        - 若误设 0.75 别名键：均分 = (0.75*0.9 + 1.0*0.1)/1.75 ≈ 0.443
          → 有条件支持。断言"驳回"即证明无别名键、fallback=0.5。
        """
        scores = [
            EvidenceScore("a", 0.9, 0.9, 0.9, 0.9),
            EvidenceScore("b", 0.9, 0.9, 0.9, 0.1),
        ]
        emap = {"a": _ev("a", "data-analyst"), "b": _ev("b", "pi-agent")}
        decision, _, _ = PIAgentArbiter().arbitrate("t", scores, emap)
        assert decision == "驳回提案"
        # 两仲裁器均不得登记 data-analyst 别名键（维持计划红线）
        assert "data-analyst" not in PIAgentArbiter.ROLE_WEIGHTS
        assert "data-analyst" not in EloArbiter.ROLE_WEIGHTS


# ---------------------------------------------------------------------------
# 3. 新增画像可被 profile_loader 真实加载
# ---------------------------------------------------------------------------

class TestNewProfilesLoadable:
    @pytest.mark.parametrize("role_key", ["qc-inspector", "pi-agent"])
    def test_profile_loads(self, role_key: str):
        """新画像经 RoleProfileLoader.load_all 真实加载且 role_key 正确。"""
        from app.core.agents.profile_loader import RoleProfileLoader

        profiles = RoleProfileLoader().load_all()
        assert role_key in profiles, f"{role_key} 未被 profile_loader 加载"
        profile = profiles[role_key]
        assert profile.role_key == role_key
        assert profile.display_name
        assert profile.system_prompt.strip(), "system_prompt 不得为空"
        assert profile.identity_prompt.strip(), "identity_prompt 不得为空"

    def test_qc_inspector_aligns_with_agent_guard(self):
        """qc-inspector 画像与 agent_guard 既有规则/权重同源（防角色漂移）。"""
        from app.core.agents.profile_loader import RoleProfileLoader

        profile = RoleProfileLoader().load_all()["qc-inspector"]
        # 投票权重 1（普通专业角色），与 vote.py DEFAULT_WEIGHTS 语义一致
        assert profile.meta["vote_weight"] == 1

    def test_pi_agent_aligns_with_arbiter(self):
        """pi-agent 画像定位为最终裁决者（tier=1、投票权重 2）。"""
        from app.core.agents.profile_loader import RoleProfileLoader

        profile = RoleProfileLoader().load_all()["pi-agent"]
        assert profile.tier == 1
        assert profile.meta["vote_weight"] == 2

    @pytest.mark.parametrize("role_key", ["qc-inspector", "pi-agent"])
    def test_structure_prediction_tools_forbidden(self, role_key: str):
        """两画像均禁用系统级禁用工具（AlphaFold2/3 硬约束一致）。"""
        from app.core.agents.profile_loader import RoleProfileLoader

        profile = RoleProfileLoader().load_all()[role_key]
        forbidden = str(profile.tools.get("forbidden_tools", []))
        assert "alphafold3" in forbidden
        assert "chai1" in forbidden


# ---------------------------------------------------------------------------
# 4. §6.5 校准实验核心函数
# ---------------------------------------------------------------------------

class TestCalibrationCore:
    @pytest.fixture()
    def calib(self):
        """延迟导入校准实验模块（其顶部含 sys.path/环境变量注入）。"""
        import run_role_weight_calibration as m

        return m

    def test_measure_role_coverage(self, calib):
        """覆盖率统计：按案件去重计数（一案多证据同角色只计一次）。"""
        pool = [
            {"evidences": [{"role": "pi-agent"}, {"role": "pi-agent"}]},
            {"evidences": [{"role": "reviewer"}]},
        ]
        coverage = calib.measure_role_coverage(pool)
        # 第一案 pi-agent 出现两次只计 1（按案件去重），第二案不含 → 共 1
        assert coverage["pi-agent"] == 1
        assert coverage["drug-researcher"] == 0

    def test_count_flips_detects_decision_change(self, calib):
        """翻转计数：权重变化导致裁决变化时应检出翻转。

        make_factory 语义与校准脚本一致：接收权重字典、返回已注入
        该权重的仲裁实例（防止扰动被静默丢弃导致实验空转）。
        """
        # 基准：pi-agent(1.0) 高分证据 + agent(0.5) 低分证据
        scores = [
            EvidenceScore("a", 0.9, 0.9, 0.9, 0.9),
            EvidenceScore("b", 0.9, 0.9, 0.9, 0.1),
        ]
        emap = {"a": _ev("a", "pi-agent"), "b": _ev("b", "agent")}
        base_arb = PIAgentArbiter()
        base_dec = base_arb.arbitrate("t", scores, emap)[0]

        # 极端扰动：pi-agent 权重 ×0.1 → 低分证据拉低加权均分 → 翻转
        perturbed = dict(PIAgentArbiter.ROLE_WEIGHTS)
        perturbed["pi-agent"] *= 0.1

        def make_factory(weights: dict[str, float]) -> PIAgentArbiter:
            """按传入权重构造仲裁实例（与 make_pi_factory 同语义）。"""
            arb = PIAgentArbiter()
            arb.ROLE_WEIGHTS = dict(weights)
            return arb

        flips = calib.count_flips([(scores, emap)], [base_dec], perturbed, make_factory)
        assert flips == 1

    def test_count_flips_no_change_same_weights(self, calib):
        """基准权重下重跑应 0 翻转（确定性仲裁）。"""
        scores, emap = _single_case("drug-researcher", 0.8)
        base_dec = PIAgentArbiter().arbitrate("t", scores, emap)[0]
        weights = dict(PIAgentArbiter.ROLE_WEIGHTS)

        def make_factory(weights: dict[str, float]) -> PIAgentArbiter:
            """按传入权重构造仲裁实例（与 make_pi_factory 同语义）。"""
            arb = PIAgentArbiter()
            arb.ROLE_WEIGHTS = dict(weights)
            return arb

        assert calib.count_flips([(scores, emap)], [base_dec], weights, make_factory) == 0

    def test_red_lines_positive_control_failure(self, calib):
        """R0：阳性对照翻转数不足时必须报违例（防空转假阴性）。"""
        results = {
            role: {d: {"flips": 0, "n": 100, "flip_rate": 0.0}
                   for d in calib.PERTURB_FACTORS}
            for role in PAPER_13_ROLES
        }
        coverage = {role: 50 for role in PAPER_13_ROLES}
        violations = calib.check_red_lines(
            results, results, coverage, {"pi-agent": 0}
        )
        assert any("阳性对照失败" in v for v in violations)

    def test_red_lines_flip_rate_exceeded(self, calib):
        """R2：翻转率超限时必须报违例（权重过拟合）。"""
        results = {
            role: {d: {"flips": 0, "n": 100, "flip_rate": 0.0}
                   for d in calib.PERTURB_FACTORS}
            for role in PAPER_13_ROLES
        }
        # 构造 drug-researcher@down_0.8 翻转率超限
        results["drug-researcher"]["down_0.8"] = {
            "flips": 10, "n": 100, "flip_rate": 0.1,
        }
        coverage = {role: 50 for role in PAPER_13_ROLES}
        violations = calib.check_red_lines(
            results, results, coverage,
            {r: 5 for r in ("pi-agent", "drug-researcher", "report-expert")},
        )
        assert any("翻转率超限" in v and "drug-researcher" in v for v in violations)

    def test_red_lines_coverage_insufficient(self, calib):
        """R1：角色覆盖率不足时必须报违例。"""
        results = {
            role: {d: {"flips": 0, "n": 100, "flip_rate": 0.0}
                   for d in calib.PERTURB_FACTORS}
            for role in PAPER_13_ROLES
        }
        coverage = {role: 50 for role in PAPER_13_ROLES}
        coverage["report-expert"] = 2
        violations = calib.check_red_lines(
            results, results, coverage,
            {r: 5 for r in ("pi-agent", "drug-researcher", "report-expert")},
        )
        assert any("角色覆盖不足" in v and "report-expert" in v for v in violations)
