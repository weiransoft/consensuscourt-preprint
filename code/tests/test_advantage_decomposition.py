"""
定理 1-A 优势分解标定实验（run_advantage_decomposition.py）单元测试 U16。

覆盖：
- C=∅ 负例底空间构造（L1 defect_only/clean 分层 + L1.5 四类）；
- D 事件（真实 screen_asymmetric 的 PENDING）与 M 事件（真实 rubric
  矛盾对）的逐案标定；
- 2×2 四格事件分类（D∩M / S=D\\M / M\\D / neither）与容斥恒等式
  P(D)−P(M)=P(S)−P(M\\D) 的精确闭合（集合代数，非统计拟合）；
- q_K/κ 标定的枚举计数（m=4 时候选对 6、相邻对 12/案）、κ=P_joint/q_K²
  公式、同种子可复现性；
- 预注册红线检查器对全部六类违例的捕获（伪造输入验证检查器本身）。

运行：backend$ .venv/bin/pytest tests/test_advantage_decomposition.py -v
"""
from __future__ import annotations

import copy
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

from app.core.virtual_lab.consensus_court import (  # noqa: E402
    EvidenceScorer,
    RuleEngine,
)
from app.core.virtual_lab.rubric_engine import AsymmetricRubric  # noqa: E402
from tests.ddb.l1_generator import generate as generate_l1  # noqa: E402
from tests.ddb.l15_generator import generate as generate_l15  # noqa: E402

import run_advantage_decomposition as ad  # noqa: E402


def _full_rubric() -> AsymmetricRubric:
    """三机制全开 rubric（ABP full_v03 配置）。"""
    return AsymmetricRubric(
        enable_aspect=True,
        enable_validity_gate=True,
        enable_intensity=True,
    )


def _base_rubric() -> AsymmetricRubric:
    """三机制全关 rubric（baseline_v02 配置）。"""
    return AsymmetricRubric(
        enable_aspect=False,
        enable_validity_gate=False,
        enable_intensity=False,
    )


def _short_defect_evidence_dict(eid: str = "HYB-E900") -> dict:
    """构造一条必触发 PENDING（内容过短）的证据字典。

    内容与 build_null_case 缺陷证据同口径（短文本、无极性词），
    来源 pubmed 且引用完整，确保只有 short_content 通道触发。
    """
    return {
        "evidence_id": eid,
        "source": "pubmed",
        "content": "数据待补充",
        "role": "supporting",
        "timestamp": None,
        "reference_ids": ["PMID-HYB-900"],
        "metadata": {},
    }


class TestNegativePool:
    """U16.1 C=∅ 底空间构造与分层标签。"""

    def test_pool_composition_small(self):
        """cases_l1=10 时：10 个 L1 负例 + 88 个 L1.5 = 98 例。"""
        pool = ad.build_negative_pool(cases_l1=10)
        n_l1 = sum(1 for s, _ in pool if s.startswith("l1_"))
        n_l15 = sum(1 for s, _ in pool if s.startswith("l15_"))
        assert n_l1 == 10
        assert n_l15 == 88
        assert len(pool) == 98

    def test_l1_strata_split(self):
        """L1 负例必须按 defect_only/clean 两层标注且两层均非空。"""
        pool = ad.build_negative_pool(cases_l1=20)
        strata = {s for s, _ in pool if s.startswith("l1_")}
        assert strata == {"l1_defect_only", "l1_clean"}
        n_defect = sum(1 for s, _ in pool if s == "l1_defect_only")
        n_clean = sum(1 for s, _ in pool if s == "l1_clean")
        assert n_defect == 10 and n_clean == 10  # 生成器半数口径

    def test_l15_strata_categories(self):
        """L1.5 分层标签必须覆盖三类伪冲突：40/24/24（per_template=8）。"""
        pool = ad.build_negative_pool(cases_l1=4)
        l15_strata = [
            s for s, _ in pool if s.startswith("l15_")
        ]
        from collections import Counter

        counts = Counter(l15_strata)
        # 三类：cross_aspect 含 5 个模板 → 40；其余两类各 3 模板 → 24
        assert counts["l15_cross_aspect"] == 40
        assert counts["l15_ambiguous_reference"] == 24
        assert counts["l15_different_object_gate"] == 24
        assert sum(counts.values()) == 88


class TestEventDetection:
    """U16.2 D/M 事件标定（真实生产组件，无 mock）。"""

    def test_detect_defect_pending_mini(self):
        """短内容证据必须标定 D=True，且无 CRITICAL。"""
        case = {"topic": ad.NULL_TOPIC, "evidences": [_short_defect_evidence_dict()]}
        has_pending, n_critical = ad.detect_defect_event(case, RuleEngine())
        assert has_pending is True
        assert n_critical == 0

    def test_detect_defect_clean_mini(self):
        """正常长度的完整证据不得标定 D。"""
        ev = _short_defect_evidence_dict()
        ev["content"] = (
            "本实验在 EGFR 突变细胞系中观察到激酶活性显著上调的完整记录"
        )
        case = {"topic": ad.NULL_TOPIC, "evidences": [ev]}
        has_pending, n_critical = ad.detect_defect_event(case, RuleEngine())
        assert has_pending is False
        assert n_critical == 0

    def test_mismatch_l15_gated_by_full_mechanism(self):
        """L1.5 伪冲突：full 全开拦截（M=False），baseline 全关误定罪（M=True）。"""
        l15 = generate_l15(per_template=8, seed=ad.SEED)
        scorer = EvidenceScorer()
        full = _full_rubric()
        base = _base_rubric()
        n_full = sum(
            ad.detect_mismatch_event(c, full, scorer)[0] for c in l15
        )
        n_base = sum(
            ad.detect_mismatch_event(c, base, scorer)[0] for c in l15
        )
        assert n_full == 0          # 三机制全开：88 例全部拦截
        assert n_base == len(l15)   # 三机制全关：88 例全部误定罪


class TestFourCellDecomposition:
    """U16.3 四格分类与恒等式闭合。"""

    def _mini_pool_full_covers_all_cells(self) -> list[tuple[str, dict]]:
        """构造在 full 配置下恰好覆盖四格的 4 案迷你底空间。

        - d_and_m：L1 正例（真矛盾 → M）追加一条短缺陷证据（→ D）；
        - m_without_d：L1 正例原样（M=True、D=False）；
        - s：L1 defect_only 负例（D=True、M=False）；
        - neither：L1 clean 负例（D=False、M=False）。
        """
        l1 = generate_l1(
            num_with_contradiction=20, num_no_contradiction=20, seed=ad.SEED
        )
        positive = next(c for c in l1 if c["polarity"] == "with_contradiction")
        defect_only = next(
            c for c in l1
            if c["polarity"] == "no_contradiction"
            and c["ground_truth"].get("defect_only")
        )
        clean = next(
            c for c in l1
            if c["polarity"] == "no_contradiction"
            and not c["ground_truth"].get("defect_only")
        )
        hybrid = copy.deepcopy(positive)
        hybrid["evidences"].append(_short_defect_evidence_dict())
        return [
            ("mini_d_and_m", hybrid),
            ("mini_m_without_d", positive),
            ("mini_s", defect_only),
            ("mini_neither", clean),
        ]

    def test_all_four_cells_hit_under_full(self):
        """full rubric 下四格必须各命中恰好 1 例。"""
        pool = self._mini_pool_full_covers_all_cells()
        result = ad.decompose(
            pool, _full_rubric(), EvidenceScorer(), RuleEngine()
        )
        assert result["cells"] == {
            "d_and_m": 1, "s": 1, "m_without_d": 1, "neither": 1,
        }

    def test_probabilities_derived_from_cells(self):
        """四个概率必须与四格计数严格一致（手算复核）。"""
        pool = self._mini_pool_full_covers_all_cells()
        r = ad.decompose(pool, _full_rubric(), EvidenceScorer(), RuleEngine())
        n = r["n"]
        c = r["cells"]
        # 报告字段保留 6 位小数，容差 1e-5
        assert r["P(D)_fpr_guilty"] == pytest.approx(
            (c["d_and_m"] + c["s"]) / n, abs=1e-5
        )
        assert r["P(M)_fpr_abp"] == pytest.approx(
            (c["d_and_m"] + c["m_without_d"]) / n, abs=1e-5
        )
        assert r["P(S)"] == pytest.approx(c["s"] / n, abs=1e-5)
        assert r["P(M_without_D)"] == pytest.approx(
            c["m_without_d"] / n, abs=1e-5
        )

    def test_identity_closure_exact(self):
        """容斥恒等式必须精确闭合（残差 0），迷你池与真实池双重验证。"""
        # 迷你池
        pool = self._mini_pool_full_covers_all_cells()
        r = ad.decompose(pool, _full_rubric(), EvidenceScorer(), RuleEngine())
        assert r["closure"]["residual"] == 0.0
        assert r["closure"]["lhs_PD_minus_PM"] == pytest.approx(
            r["closure"]["rhs_PS_minus_PMD"]
        )
        # 真实底空间（两配置）
        real_pool = ad.build_negative_pool(cases_l1=30)
        for rubric in (_full_rubric(), _base_rubric()):
            rr = ad.decompose(real_pool, rubric, EvidenceScorer(), RuleEngine())
            assert rr["closure"]["residual"] <= ad.CLOSURE_TOLERANCE

    def test_full_vs_base_advantage_direction_on_real_pool(self):
        """真实池：full 救回全部 L1.5（M\\D=0），base 误定罪全部 L1.5。"""
        pool = ad.build_negative_pool(cases_l1=30)
        scorer, engine = EvidenceScorer(), RuleEngine()
        full = ad.decompose(pool, _full_rubric(), scorer, engine)
        base = ad.decompose(pool, _base_rubric(), scorer, engine)
        n = len(pool)  # 30 L1 负例（半数 defect_only=15）+ 88 L1.5
        assert full["cells"]["m_without_d"] == 0
        assert base["cells"]["m_without_d"] == 88
        # 报告字段四舍五入到 6 位小数，容差 1e-5
        assert full["P(M_without_D)"] == pytest.approx(0.0, abs=1e-5)
        assert base["P(M_without_D)"] == pytest.approx(88 / n, abs=1e-5)
        # S 集由 L1 defect_only 供给（30 负例中半数 = 15 例）
        assert full["cells"]["s"] == 15
        assert full["P(S)"] == pytest.approx(15 / n, abs=1e-5)

    def test_by_stratum_partition_sums_to_total(self):
        """分层四格之和必须等于总体四格（分层是总体的一个划分）。"""
        pool = ad.build_negative_pool(cases_l1=10)
        r = ad.decompose(pool, _full_rubric(), EvidenceScorer(), RuleEngine())
        for cell in ("d_and_m", "s", "m_without_d", "neither"):
            assert sum(
                layer[cell] for layer in r["by_stratum"].values()
            ) == r["cells"][cell]
        assert sum(
            layer["n"] for layer in r["by_stratum"].values()
        ) == r["n"]


class TestNullDependenceCalibration:
    """U16.4 q_K/κ 枚举计数与公式。"""

    def test_enumeration_counts_m4(self):
        """m=4 时每案候选对 C(4,2)=6、相邻对 4·C(3,2)=12。"""
        cal = ad.calibrate_null_dependence(
            _base_rubric(), EvidenceScorer(), m=4, trials=10, seed=7
        )
        assert cal["pair_opportunities"] == 10 * 6
        assert cal["adjacent_pairs"] == 10 * 4 * 3

    def test_qk_and_kappa_formulas(self):
        """q_K=事件/候选对；κ=P_joint/q_K²（从返回计数独立重算）。"""
        cal = ad.calibrate_null_dependence(
            _base_rubric(), EvidenceScorer(), m=10, trials=80, seed=11
        )
        qk = cal["mismatch_events"] / cal["pair_opportunities"]
        pj = cal["joint_events"] / cal["adjacent_pairs"]
        assert cal["q_K"] == pytest.approx(qk, abs=1e-6)
        assert cal["P_joint"] == pytest.approx(pj, abs=1e-8)
        if qk > 0:
            assert cal["kappa"] == pytest.approx(pj / qk ** 2, abs=5e-3)
        else:
            assert cal["kappa"] is None

    def test_calibration_reproducible(self):
        """同种子同参数必须逐计数可复现。"""
        kw = dict(m=8, trials=30, seed=21)
        a = ad.calibrate_null_dependence(_base_rubric(), EvidenceScorer(), **kw)
        b = ad.calibrate_null_dependence(_base_rubric(), EvidenceScorer(), **kw)
        assert a["mismatch_events"] == b["mismatch_events"]
        assert a["joint_events"] == b["joint_events"]

    def test_full_qk_not_greater_than_base_small(self):
        """全开配置边际错配率不得高于全关（m=12，60 案小规模方向检查）。"""
        kw = dict(m=12, trials=60, seed=31)
        full = ad.calibrate_null_dependence(_full_rubric(), EvidenceScorer(), **kw)
        base = ad.calibrate_null_dependence(_base_rubric(), EvidenceScorer(), **kw)
        assert full["q_K"] <= base["q_K"] + 1e-9


class TestDecompositionRedLineChecker:
    """U16.5 六类红线违例的捕获（伪造输入，验证检查器逻辑）。"""

    def _good_decomp(self, ps=0.20, pmd=0.0):
        """构造一份闭合且方向正确的分解结果。"""
        n = 100
        s = round(ps * n)
        md = round(pmd * n)
        d_and_m = 0
        neither = n - s - md - d_and_m
        p_d = (d_and_m + s) / n
        p_m = (d_and_m + md) / n
        return {
            "n": n,
            "cells": {
                "d_and_m": d_and_m, "s": s,
                "m_without_d": md, "neither": neither,
            },
            "P(D)_fpr_guilty": p_d,
            "P(M)_fpr_abp": p_m,
            "P(S)": ps,
            "P(M_without_D)": pmd,
            "closure": {
                "lhs_PD_minus_PM": p_d - p_m,
                "rhs_PS_minus_PMD": ps - pmd,
                "residual": 0.0,
            },
            "critical_evidence_count": 0,
            "by_stratum": {},
        }

    def _good_cal(self, qk=0.03, joint=100, p_joint=0.0015, kappa=1.5):
        return {
            "q_K": qk, "joint_events": joint,
            "P_joint": p_joint, "kappa": kappa,
        }

    def test_all_green_passes(self):
        decomp = {
            "full_v03": self._good_decomp(ps=0.20, pmd=0.0),
            "baseline_v02": self._good_decomp(ps=0.20, pmd=0.59),
        }
        cal = {
            "full_v03": self._good_cal(qk=0.006),
            "baseline_v02": self._good_cal(qk=0.035),
        }
        assert ad.check_decomposition_red_lines(decomp, cal) == []

    def test_closure_residual_violation_detected(self):
        decomp = {
            "full_v03": self._good_decomp(),
            "baseline_v02": self._good_decomp(pmd=0.59),
        }
        decomp["full_v03"]["closure"]["residual"] = 1e-6
        violations = ad.check_decomposition_red_lines(
            decomp, {"full_v03": self._good_cal(qk=0.006),
                     "baseline_v02": self._good_cal(qk=0.035)}
        )
        assert any("闭合违例" in v for v in violations)

    def test_critical_pollution_detected(self):
        decomp = {
            "full_v03": self._good_decomp(),
            "baseline_v02": self._good_decomp(pmd=0.59),
        }
        decomp["baseline_v02"]["critical_evidence_count"] = 2
        violations = ad.check_decomposition_red_lines(
            decomp, {"full_v03": self._good_cal(qk=0.006),
                     "baseline_v02": self._good_cal(qk=0.035)}
        )
        assert any("CRITICAL" in v and "污染" in v for v in violations)

    def test_no_advantage_detected(self):
        """P(S) ≤ P(M\\D) 必须被识别为核心主张证伪。"""
        decomp = {
            "full_v03": self._good_decomp(ps=0.10, pmd=0.30),
            "baseline_v02": self._good_decomp(pmd=0.59),
        }
        violations = ad.check_decomposition_red_lines(
            decomp, {"full_v03": self._good_cal(qk=0.006),
                     "baseline_v02": self._good_cal(qk=0.035)}
        )
        assert any("无优势" in v for v in violations)

    def test_mechanism_direction_detected(self):
        """full 的 M\\D 高于 base 必须报机制方向违例。"""
        decomp = {
            "full_v03": self._good_decomp(ps=0.20, pmd=0.60),
            "baseline_v02": self._good_decomp(pmd=0.59),
        }
        violations = ad.check_decomposition_red_lines(
            decomp, {"full_v03": self._good_cal(qk=0.006),
                     "baseline_v02": self._good_cal(qk=0.035)}
        )
        assert any("机制方向" in v for v in violations)

    def test_qk_direction_detected(self):
        cal = {
            "full_v03": self._good_cal(qk=0.05),
            "baseline_v02": self._good_cal(qk=0.035),
        }
        violations = ad.check_decomposition_red_lines(
            {"full_v03": self._good_decomp(),
             "baseline_v02": self._good_decomp(pmd=0.59)},
            cal,
        )
        assert any("q_K 方向" in v for v in violations)

    def test_insufficient_joint_events_detected(self):
        """joint_events < 30 必须报样本不足（禁止小样本粉饰 κ）。"""
        cal = {
            "full_v03": self._good_cal(qk=0.006, joint=5, kappa=None),
            "baseline_v02": self._good_cal(qk=0.035),
        }
        violations = ad.check_decomposition_red_lines(
            {"full_v03": self._good_decomp(),
             "baseline_v02": self._good_decomp(pmd=0.59)},
            cal,
        )
        assert any("样本不足" in v and "full_v03" in v for v in violations)

    def test_kappa_below_one_detected(self):
        """样本充足但 κ<1 必须报相依方向违例。"""
        cal = {
            "full_v03": self._good_cal(qk=0.006),
            "baseline_v02": self._good_cal(qk=0.035, kappa=0.8),
        }
        violations = ad.check_decomposition_red_lines(
            {"full_v03": self._good_decomp(),
             "baseline_v02": self._good_decomp(pmd=0.59)},
            cal,
        )
        assert any("κ" in v and "baseline_v02" in v for v in violations)
