"""
scorer 级消融（v0.4.15，大纲 §6.4 第 4/8 行）单元测试。

覆盖范围：
    1. 重算函数判别设计（_recompute_overall / apply_scorer_ablation）：
       - 来源置平改变 overall（pubmed 0.9 / unknown 0.2 → 统一中值 0.775）；
       - recency 置 1.0 改变 overall 且只升不降（时效衰减关闭的语义）；
       - 原 overall 不动时输出一致（recency 已为 1.0 的证据经 no_recency
         置换后 overall 逐条相等——重算函数与生产公式一致性的判别）；
       - 低时效问题标记（"证据时效性较低"）在 no_recency 下被正确移除、
         在 no_source_pedigree 下原样保留；
       - 未知模式 ValueError（防御性契约）。
    2. 红线检查器（check_scorer_ablation_red_lines）伪造违例反测：
       - FPR 单调性违例 / 召回单调性违例 / no_recency 效应方向违例 /
         no_source_pedigree 生效性违例各自被检出；
       - 全部满足时返回空列表（无误报）。
    3. 管线集成（真实 L1 生成器小样本）：no_recency 消融跑通 evaluate_config，
       召回/FPR 与 full_v03 恒等（评分后处理不触碰定罪通道的结构不变量），
       通过案例平均置信度 ≥ full_v03。

运行：backend$ .venv/bin/pytest tests/test_scorer_ablation_v0415.py -v
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
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
    Evidence,
    EvidenceScorer,
)
from run_ablation_v03 import (  # noqa: E402
    UNIFORM_SOURCE_SCORE,
    apply_scorer_ablation,
    check_scorer_ablation_red_lines,
    evaluate_config,
    _recompute_overall,
)
from tests.ddb.l1_generator import generate as generate_l1  # noqa: E402

#: 与 run_ablation_v03.CONFIGS["full_v03"] 同口径的机制全开开关
FULL_V03_SWITCHES = {
    "enable_aspect": True,
    "enable_validity_gate": True,
    "enable_intensity": True,
    "enable_position_weighting": False,
}


# ---------------------------------------------------------------------------
# 辅助：构造证据并评分（真实调用生产 EvidenceScorer）
# ---------------------------------------------------------------------------

def _score_one(content: str, source: str, timestamp=None, topic: str = "EGFR 抑制剂 耐药"):
    """构造单条证据并用生产 EvidenceScorer 评分，返回 (evidence, score)。"""
    ev = Evidence(
        evidence_id="EV-TEST-01",
        source=source,
        content=content,
        role="drug-researcher",
        timestamp=timestamp,
        reference_ids=["12345678"],
    )
    return ev, EvidenceScorer().score(ev, topic)


# ---------------------------------------------------------------------------
# 1. 重算函数判别设计
# ---------------------------------------------------------------------------

class TestScorerAblationTransforms:
    """apply_scorer_ablation 置换判别 + _recompute_overall 公式一致性。"""

    def test_no_source_pedigree_flattens_pubmed_score(self) -> None:
        """来源置平：pubmed（0.9）证据经消融后 source_score 置统一中值、overall 改变。"""
        _, score = _score_one(
            "osimertinib 在 EGFR 突变 NSCLC 中有效，mPFS 显著延长。",
            "pubmed", timestamp=datetime(2024, 1, 1),
        )
        assert score.source_score == 0.9
        (abl,) = apply_scorer_ablation([score], "no_source_pedigree")
        # 来源分数被置平为生产权重表的中位数
        assert abl.source_score == pytest.approx(UNIFORM_SOURCE_SCORE)
        # overall 必须实际改变（0.9→0.775 的 0.4 权重差不会被其他分量抵消）
        assert abl.overall_score != score.overall_score
        # 其余分量与证据 id 原样保留（单变量隔离）
        assert abl.evidence_id == score.evidence_id
        assert abl.recency_score == score.recency_score
        assert abl.relevance_score == score.relevance_score
        # issues 原样保留（"来源未知"与权重等权无关）
        assert abl.issues == score.issues

    def test_no_source_pedigree_raises_unknown_source(self) -> None:
        """来源置平：unknown（0.2）证据的 overall 被抬高——噪声来源受益即消融动机。"""
        _, score = _score_one(
            "某内部实验显示化合物 X 对靶点 Y 有效。",
            "unknown", timestamp=datetime(2024, 1, 1),
        )
        assert score.source_score == 0.2
        (abl,) = apply_scorer_ablation([score], "no_source_pedigree")
        assert abl.overall_score > score.overall_score

    def test_no_recency_raises_overall_and_keeps_source(self) -> None:
        """时效衰减关闭：老文献 recency 置 1.0，overall 只升不降、来源分量不动。"""
        _, score = _score_one(
            "早期研究显示化合物 X 在小鼠模型中抑制肿瘤生长。",
            "pubmed", timestamp=datetime(2010, 1, 1),
        )
        assert score.recency_score < 0.5  # 16 年历史文献必然低时效
        (abl,) = apply_scorer_ablation([score], "no_recency")
        assert abl.recency_score == 1.0
        assert abl.overall_score > score.overall_score
        assert abl.source_score == score.source_score
        # 低时效问题标记随衰减关闭而消失（counterfactual 语义）
        assert "证据时效性较低" not in abl.issues
        assert "证据时效性较低" in score.issues

    def test_no_recency_idempotent_when_recency_already_one(self) -> None:
        """原 overall 不动时输出一致：recency 已为 1.0 的证据经 no_recency 置换后
        overall 逐条相等（重算函数与生产公式一致性的判别测试）。"""
        # 时间戳取"现在"：exp(-age/H) ≈ 1.0，生产评分四舍五入后即为 1.0
        _, score = _score_one(
            "最新临床研究显示药物 Z 在 III 期试验中达到主要终点。",
            "pubmed", timestamp=datetime.utcnow(),
        )
        assert score.recency_score == 1.0
        (abl,) = apply_scorer_ablation([score], "no_recency")
        assert abl.overall_score == score.overall_score
        assert abl.recency_score == score.recency_score

    def test_no_source_pedigree_preserves_recency_issue(self) -> None:
        """来源置平不触碰时效通道：低时效问题在 no_source_pedigree 下原样保留。"""
        _, score = _score_one(
            "早期研究显示化合物 X 在小鼠模型中抑制肿瘤生长。",
            "pubmed", timestamp=datetime(2010, 1, 1),
        )
        (abl,) = apply_scorer_ablation([score], "no_source_pedigree")
        assert "证据时效性较低" in abl.issues

    def test_unknown_mode_raises_value_error(self) -> None:
        """未知消融模式必须拒绝（防御性契约，防止静默走原始评分）。"""
        _, score = _score_one("测试内容足够长以通过初筛。", "pubmed")
        with pytest.raises(ValueError):
            apply_scorer_ablation([score], "nonexistent_mode")

    def test_recompute_overall_matches_production_formula(self) -> None:
        """重算函数与生产 EvidenceScorer 公式一致：对任意分量取值，
        _recompute_overall(s, r, rel) 与手算 0.4/0.2/0.4 加权逐项相等。"""
        for s, r, rel in [(0.9, 1.0, 0.5), (0.775, 0.3, 1.0), (0.2, 0.0, 0.25)]:
            expected = round(s * 0.4 + r * 0.2 + rel * 0.4, 3)
            assert _recompute_overall(s, r, rel) == expected

    def test_ablation_does_not_mutate_input(self) -> None:
        """消融置换返回新对象：输入评分列表不被修改（实验可重复性）。"""
        _, score = _score_one(
            "osimertinib 在 EGFR 突变 NSCLC 中有效。",
            "pubmed", timestamp=datetime(2024, 1, 1),
        )
        before = (score.source_score, score.recency_score, score.overall_score)
        apply_scorer_ablation([score], "no_source_pedigree")
        assert (score.source_score, score.recency_score, score.overall_score) == before


# ---------------------------------------------------------------------------
# 2. 红线检查器伪造违例反测
# ---------------------------------------------------------------------------

def _metrics(recall: float, fpr: float, conf: float) -> dict:
    """构造指标字典（只含红线检查器消费的字段）。"""
    return {"recall": recall, "fpr": fpr, "avg_pass_confidence": conf}


class TestScorerAblationRedLineChecker:
    """check_scorer_ablation_red_lines 的伪造违例反测（检查器本身可信）。"""

    FULL = _metrics(recall=1.0, fpr=0.0, conf=0.75)

    def test_unknown_mode_rejected(self) -> None:
        """架构师审查加固锁：非法模式必须显式拒绝——此前会静默落入
        no_source_pedigree 的 else 分支，使红线检查形同虚设。"""
        with pytest.raises(ValueError, match="未知 scorer 消融模式"):
            check_scorer_ablation_red_lines(self.FULL, self.FULL, "not_a_mode")

    def test_full_fpr_none_guarded(self) -> None:
        """架构师审查加固锁：full_metrics["fpr"] 为 None 时不得抛
        TypeError（与召回侧双侧守卫对称），应正常返回（无违例可判）。"""
        full_none = _metrics(recall=1.0, fpr=None, conf=0.75)
        abl = _metrics(recall=1.0, fpr=0.1, conf=0.80)
        v = check_scorer_ablation_red_lines(full_none, abl, "no_recency")
        # fpr 通道因 full 侧缺失不可判，召回侧正常工作
        assert v == []

    def test_fpr_monotonicity_violation_detected(self) -> None:
        """伪造 FPR 违例：消融 FPR 高于 full_v03 必须被检出。"""
        abl = _metrics(recall=1.0, fpr=0.1, conf=0.80)
        v = check_scorer_ablation_red_lines(self.FULL, abl, "no_recency")
        assert len(v) == 1 and "FPR" in v[0]

    def test_recall_monotonicity_violation_detected(self) -> None:
        """伪造召回违例：消融召回低于 full_v03 必须被检出。"""
        abl = _metrics(recall=0.9, fpr=0.0, conf=0.80)
        v = check_scorer_ablation_red_lines(self.FULL, abl, "no_recency")
        assert len(v) == 1 and "召回" in v[0]

    def test_no_recency_direction_violation_detected(self) -> None:
        """伪造效应方向违例：no_recency 置信度低于 full_v03 必须被检出。"""
        abl = _metrics(recall=1.0, fpr=0.0, conf=0.70)
        v = check_scorer_ablation_red_lines(self.FULL, abl, "no_recency")
        assert len(v) == 1 and "效应方向" in v[0]

    def test_no_source_pedigree_engagement_violation_detected(self) -> None:
        """伪造生效性违例：no_source_pedigree 置信度与 full_v03 恒等必须被检出。"""
        abl = _metrics(recall=1.0, fpr=0.0, conf=0.75)
        v = check_scorer_ablation_red_lines(self.FULL, abl, "no_source_pedigree")
        assert len(v) == 1 and "恒等" in v[0]

    def test_all_passing_returns_empty(self) -> None:
        """全部满足（FPR/召回单调 + 方向正确/生效）时无误报。"""
        ok_recency = _metrics(recall=1.0, fpr=0.0, conf=0.80)
        assert check_scorer_ablation_red_lines(self.FULL, ok_recency, "no_recency") == []
        ok_pedigree = _metrics(recall=1.0, fpr=0.0, conf=0.70)
        assert check_scorer_ablation_red_lines(self.FULL, ok_pedigree, "no_source_pedigree") == []

    def test_equal_metrics_pass_monotonicity(self) -> None:
        """单调性允许相等（消融与 full_v03 恒等不算 FPR/召回违例）。"""
        equal = _metrics(recall=1.0, fpr=0.0, conf=0.80)
        assert check_scorer_ablation_red_lines(self.FULL, equal, "no_recency") == []


# ---------------------------------------------------------------------------
# 3. 管线集成（真实 L1 生成器小样本）
# ---------------------------------------------------------------------------

class TestScorerAblationPipelineIntegration:
    """evaluate_config(score_ablation=...) 与 full_v03 参照行的结构不变量。"""

    @pytest.fixture(scope="class")
    def small_cases(self):
        """小样本 L1 基准（2×2 例，seed 固定，保证测试确定性）。"""
        return generate_l1(num_with_contradiction=2, num_no_contradiction=2, seed=42)

    def test_ablation_recall_fpr_identical_to_full_v03(self, small_cases) -> None:
        """结构不变量：评分后处理不触碰定罪通道 → 召回/FPR 与 full_v03 恒等。"""
        full = evaluate_config(small_cases, FULL_V03_SWITCHES)
        for mode in ("no_source_pedigree", "no_recency"):
            abl = evaluate_config(small_cases, FULL_V03_SWITCHES, score_ablation=mode)
            assert abl["recall"] == full["recall"], mode
            assert abl["fpr"] == full["fpr"], mode

    def test_no_recency_pass_confidence_not_lower(self, small_cases) -> None:
        """效应方向：no_recency 的通过案例平均置信度 ≥ full_v03（方向先验唯一）。"""
        full = evaluate_config(small_cases, FULL_V03_SWITCHES)
        abl = evaluate_config(small_cases, FULL_V03_SWITCHES, score_ablation="no_recency")
        assert abl["avg_pass_confidence"] is not None
        # 架构师审查加固：full 侧同判别——None 比较会抛 TypeError 而非给出
        # 可读断言；夹具小样本必有通过案例，此处显式锚定该前提。
        assert full["avg_pass_confidence"] is not None
        assert abl["avg_pass_confidence"] >= full["avg_pass_confidence"]

    def test_metrics_carry_ablation_provenance(self, small_cases) -> None:
        """溯源字段：指标字典必须记录 score_ablation 模式（None=原始评分）。"""
        full = evaluate_config(small_cases, FULL_V03_SWITCHES)
        abl = evaluate_config(small_cases, FULL_V03_SWITCHES, score_ablation="no_recency")
        assert full["score_ablation"] is None
        assert abl["score_ablation"] == "no_recency"
