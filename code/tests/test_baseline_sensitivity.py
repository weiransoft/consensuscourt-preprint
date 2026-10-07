"""
基线阈值敏感性扫描实验（run_baseline_sensitivity.py）单元测试。

测试策略：不重跑完整 main()（避免覆写报告文件），在标准案例池上直接
调用脚本的 collect_primitives / evaluate_grid / check_red_lines /
pareto_frontier，并对照已落盘报告 JSON 验证：

1. 决策原语重放的确定性（同池双跑逐字段一致）；
2. 49 格点结构锁（网格完整、口径键齐全、计数守恒）；
3. RL1 默认阈值 (0.7, 0.4) 与主对比报告互锁（重放正确性锁）；
4. RL2 统计量型基线 reject-FPR 对 t_rej 单调不减；
5. Pareto 前沿正确性（非支配性 + fpr 升序 + 去重）与论文引用坐标一致；
6. 报告落盘锁：baseline_sensitivity_report.json 结构、红线零违例、
   诚实报告项 baseline_optimum_reaches_court_profile=False；
7. 检查器有效性反测试：伪造 RL1 漂移 / RL2 单调违例必须被捕获。

运行：backend$ .venv/bin/pytest tests/test_baseline_sensitivity.py -v
"""
from __future__ import annotations

import json
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

import run_baseline_sensitivity as rbs  # noqa: E402
from tests.ddb.l1_generator import generate as generate_l1  # noqa: E402
from tests.ddb.l15_generator import generate as generate_l15  # noqa: E402

#: 主对比报告路径（RL1 互锁基准，与脚本 _MAIN_REPORT 同源）
_MAIN_REPORT = (
    _BACKEND_ROOT / "tests" / "ddb" / "baseline_comparison_report.json"
)
#: 敏感性扫描报告路径（落盘锁对象）
_SENS_REPORT = (
    _BACKEND_ROOT / "tests" / "ddb" / "baseline_sensitivity_report.json"
)


@pytest.fixture(scope="module")
def primitives():
    """标准池（60×2 L1 + 88 L1.5，seed 42）的决策原语，模块级只算一次。"""
    cases = generate_l1(
        num_with_contradiction=60, num_no_contradiction=60, seed=42
    )
    cases.extend(generate_l15(per_template=8, seed=42))
    return rbs.collect_primitives(cases)


@pytest.fixture(scope="module")
def grid_out(primitives):
    """49 格点重放结果（基于模块级原语缓存，避免重复评分）。"""
    return rbs.evaluate_grid(primitives)


class TestPrimitiveDeterminism:
    """U19.1 决策原语重放的确定性：同池双跑逐字段一致。"""

    def test_double_run_identical(self, primitives):
        """确定性管线的基本要求：同样的池必须产出逐字段相同的原语记录。"""
        cases = generate_l1(
            num_with_contradiction=60, num_no_contradiction=60, seed=42
        )
        cases.extend(generate_l15(per_template=8, seed=42))
        replay = rbs.collect_primitives(cases)
        assert replay == primitives

    def test_majority_counts_support_priority_subdiagonal(self):
        """亚对角格点（t_sup < t_rej）语义锁：支持票优先，不双重计票。

        生产 MajorityVote.arbitrate 用 if/elif 链——分数落在
        [t_sup, t_rej) 时只计支持票；重放必须镜像该语义（架构审查必改项）。
        """
        # 0.55 落在 [0.5, 0.6)：elif 语义只计支持票；独立计票会同时
        # 计支持票（>=0.5）与反对票（<0.6），（1,0）锁死不回归
        support, reject = rbs._majority_counts([0.55], 0.5, 0.6)
        assert (support, reject) == (1, 0)
        # 中间弃权档不受影响：0.55 ∈ [0.5,0.6) 但 t_sup=0.7 时为弃权
        assert rbs._majority_counts([0.55], 0.7, 0.4) == (0, 0)

    def test_primitive_fields_complete(self, primitives):
        """原语记录必须携带极性、法庭信号与三条基线的决策统计量。"""
        for rec in primitives:
            for key in (
                "polarity", "court_flagged",
                "majority_scores", "elo_avg", "single_best",
            ):
                assert key in rec
            # 统计量取值合法性：分数在 [0,1]，且最高分 ≥ 加权均分未必成立，
            # 但二者都必须在分数包络内
            assert 0.0 <= rec["elo_avg"] <= 1.0
            assert 0.0 <= rec["single_best"] <= 1.0


class TestGridStructure:
    """U19.2 49 格点结构锁。"""

    def test_grid_shape(self, grid_out):
        """三基线各 49 格点，且 (t_sup, t_rej) 组合恰为 GRID 的笛卡尔积。"""
        expected = {(a, b) for a in rbs.GRID for b in rbs.GRID}
        assert len(rbs.GRID) == 7 and len(expected) == 49
        for name in rbs.SWEEP_NAMES:
            assert len(grid_out[name]) == 49
            got = {(p["t_sup"], p["t_rej"]) for p in grid_out[name]}
            assert got == expected

    def test_metric_blocks_complete(self, grid_out):
        """每个格点的 reject/non_support 两口径都带完整指标。"""
        for name in rbs.SWEEP_NAMES:
            for p in grid_out[name]:
                for op in ("reject", "non_support"):
                    for key in ("recall", "fpr", "accuracy", "precision",
                                "recall_ci95", "fpr_ci95"):
                        assert key in p[op], f"{name}@{p['t_sup']}/{p['t_rej']} 缺 {op}/{key}"

    def test_counts_conserve(self, primitives, grid_out):
        """每个格点两口径的 n/n_positive/n_negative 与案例池构成一致。"""
        n_pos = sum(1 for r in primitives if r["polarity"] == "with_contradiction")
        n_total = len(primitives)
        for name in rbs.SWEEP_NAMES:
            for p in grid_out[name]:
                for op in ("reject", "non_support"):
                    m = p[op]
                    assert m["n"] == n_total
                    assert m["n_positive"] == n_pos
                    assert m["n_negative"] == n_total - n_pos


class TestRedLines:
    """U19.3 三条预注册红线在标准池上全部满足。"""

    def test_no_violations(self, grid_out, primitives):
        """RL1 互锁 + RL2 单调：红线检查器零违例。"""
        n_pos = sum(1 for r in primitives if r["polarity"] == "with_contradiction")
        court_tp = sum(
            1 for r in primitives
            if r["polarity"] == "with_contradiction" and r["court_flagged"]
        )
        court_fp = sum(
            1 for r in primitives
            if r["polarity"] != "with_contradiction" and r["court_flagged"]
        )
        court_metrics = rbs._rates({
            "tp": court_tp, "fp": court_fp,
            "fn": n_pos - court_tp,
            "tn": (len(primitives) - n_pos) - court_fp,
        })
        with open(_MAIN_REPORT, "r", encoding="utf-8") as f:
            main_report = json.load(f)
        violations = rbs.check_red_lines(grid_out, court_metrics, main_report)
        assert violations == []

    def test_statistic_fpr_monotone_in_t_rej(self, grid_out):
        """RL2 直接验证：elo/single 的 reject-FPR 对 t_rej 单调不减。"""
        for name in ("elo_arbiter", "single_judge"):
            for t_sup in rbs.GRID:
                fprs = [
                    next(p for p in grid_out[name]
                         if p["t_sup"] == t_sup and p["t_rej"] == t_rej)
                    ["reject"]["fpr"]
                    for t_rej in rbs.GRID
                ]
                assert all(b >= a for a, b in zip(fprs, fprs[1:])), \
                    f"{name}@t_sup={t_sup} FPR 序列 {fprs} 非单调不减"


class TestParetoFrontier:
    """U19.4 Pareto 前沿正确性与论文坐标锁。"""

    def test_frontier_nondominated_and_sorted(self, grid_out):
        """前沿点必须非支配、按 fpr 升序、同 (fpr, recall) 去重。"""
        for name in rbs.SWEEP_NAMES:
            frontier = rbs.pareto_frontier(grid_out[name], "reject")
            assert frontier, f"{name} 前沿为空"
            fprs = [p["fpr"] for p in frontier]
            assert fprs == sorted(fprs)
            for i in range(1, len(frontier)):
                prev, cur = frontier[i - 1], frontier[i]
                # 升序去重：相邻点不许完全相同
                assert (cur["fpr"], cur["recall"]) != (prev["fpr"], prev["recall"])
            # 非支配性暴力复核：任何前沿点都不被其它前沿点支配
            for a in frontier:
                for b in frontier:
                    if b is a:
                        continue
                    dominates = (
                        b["recall"] >= a["recall"] and b["fpr"] <= a["fpr"]
                        and (b["recall"] > a["recall"] or b["fpr"] < a["fpr"])
                    )
                    assert not dominates, f"{name} 前沿点 {a} 被 {b} 支配"

    def test_frontier_coords_match_paper_figure(self, grid_out):
        """论文 Fig.（Pareto 图）坐标锁：关键拐点与 JSON 前沿一致（3 位小数）。"""
        elo = rbs.pareto_frontier(grid_out["elo_arbiter"], "reject")
        single = rbs.pareto_frontier(grid_out["single_judge"], "reject")
        majority = rbs.pareto_frontier(grid_out["majority_vote"], "reject")
        # elo 四拐点（论文图坐标）
        assert [(round(p["fpr"], 3), round(p["recall"], 3)) for p in elo] == [
            (0.034, 0.0), (0.628, 0.05), (0.946, 0.467), (1.0, 1.0),
        ]
        # single 五拐点
        assert [(round(p["fpr"], 3), round(p["recall"], 3)) for p in single] == [
            (0.0, 0.0), (0.669, 0.033), (0.831, 0.3), (0.939, 0.833), (1.0, 1.0),
        ]
        # majority 至少含近零 FPR 起点 (0.027, 0) 与端点 (1, 1)
        assert (round(majority[0]["fpr"], 3), round(majority[0]["recall"], 3)) == (0.027, 0.0)
        assert (majority[-1]["fpr"], majority[-1]["recall"]) == (1.0, 1.0)


class TestReportFile:
    """U19.5 报告落盘锁：结构、红线状态与诚实报告项。"""

    @pytest.fixture(scope="class")
    def report(self):
        """读取已落盘的敏感性扫描报告（由回归管线生成，测试不覆写）。

        报告文件不在版本库提交范围时跳过（而非误报失败），提示先跑管线。
        """
        if not _SENS_REPORT.exists():
            pytest.skip(
                "baseline_sensitivity_report.json 不存在，"
                "先运行 tests/scripts/run_baseline_sensitivity.py 生成"
            )
        with open(_SENS_REPORT, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_report_schema(self, report):
        """报告必须含网格/池构成/法庭参考点/三基线前沿/红线状态等键。"""
        for key in (
            "seed", "grid", "n_grid_points", "pool_composition",
            "court_reference", "baseline_optimum_reaches_court_profile",
            "systems", "red_line_violations",
        ):
            assert key in report
        assert report["n_grid_points"] == 49
        assert set(report["systems"].keys()) == set(rbs.SWEEP_NAMES)

    def test_report_red_lines_clean(self, report):
        """落盘报告的红线违例必须为空。"""
        assert report["red_line_violations"] == []

    def test_report_court_reference_matches_main_report(self, report):
        """法庭参考点与主对比报告的 reject 口径逐字段一致（RL3 表达）。"""
        with open(_MAIN_REPORT, "r", encoding="utf-8") as f:
            main_report = json.load(f)
        ref = main_report["systems"][rbs.COURT_NAME]["reject"]
        for metric in ("recall", "fpr", "accuracy", "precision"):
            assert report["court_reference"][metric] == ref[metric]

    def test_no_baseline_reaches_court_profile(self, report):
        """诚实报告项锁：49×3 格点中无任何点同时达到 recall=1.0 & FPR=0.0。"""
        assert report["baseline_optimum_reaches_court_profile"] is False


class TestCheckerEffectiveness:
    """U19.6 检查器有效性反测试：伪造违例必须被捕获。"""

    def _fake_grid(self, grid_out, name, t_sup, t_rej, mutate):
        """深拷贝格点并对指定系统/格点做指标篡改。"""
        import copy
        fake = copy.deepcopy(grid_out)
        for p in fake[name]:
            if p["t_sup"] == t_sup and p["t_rej"] == t_rej:
                mutate(p)
        return fake

    def test_detects_rl1_drift(self, grid_out, primitives):
        """RL1 反测试：把默认阈值格点的 recall 改掉，检查器必须报违例。"""
        def mutate(p):
            p["reject"]["recall"] = 0.99
        fake = self._fake_grid(grid_out, "majority_vote", 0.7, 0.4, mutate)
        with open(_MAIN_REPORT, "r", encoding="utf-8") as f:
            main_report = json.load(f)
        violations = rbs.check_red_lines(
            fake, {"recall": 1.0, "fpr": 0.0, "accuracy": 1.0, "precision": 1.0},
            main_report,
        )
        assert any("RL1" in v and "majority_vote" in v for v in violations)

    def test_detects_rl2_nonmonotone(self, grid_out, primitives):
        """RL2 反测试：把 elo 某格点 FPR 压到前格点之下，检查器必须报违例。"""
        def mutate(p):
            if p["t_sup"] == 0.7 and p["t_rej"] == 0.8:
                p["reject"]["fpr"] = 0.0
        fake = self._fake_grid(grid_out, "elo_arbiter", 0.7, 0.8, mutate)
        with open(_MAIN_REPORT, "r", encoding="utf-8") as f:
            main_report = json.load(f)
        violations = rbs.check_red_lines(
            fake, {"recall": 1.0, "fpr": 0.0, "accuracy": 1.0, "precision": 1.0},
            main_report,
        )
        assert any("RL2" in v and "elo_arbiter" in v for v in violations)
