#!/usr/bin/env python3
"""
鲁棒性专项实验（论文大纲 §6.5，v0.4.11）：定理 2 顺序无关性 + 定理 1-B
证据数量扫描。两个实验均使用真实生产组件（AsymmetricRubric /
RuleEngine.screen_asymmetric / EvidenceScorer），无 mock、无外部服务依赖。

实验 A：证据顺序置换 1000 次（§5.5 定理 2 的证伪性预测）
------------------------------------------------------------------
定理 2：裁决函数 V 对任意证据置换 π 不变，当且仅当 V 是证据对上布尔
关系的合取（不含"首个命中/提前终止"等顺序原语）。ABP 的
DrillBeforeDisposition 结构性禁止提前终止 + 全量矛盾对合取必须满足：
对同一案件的 1000 次随机证据重排，裁决签名（passed + 矛盾对集合
（含强度）+ AMBIGUOUS 待举证对集合）完全一致；PI 仲裁的裁决/置信度
同为加权聚合，亦应逐案不变。在 full_v03 与 baseline_v02 两个开关
配置下重复检验（合取结构不依赖开关）。

实验 B：证据数量 n=3/5/10/20 扫描（§5.4 定理 1-B 及推论）
------------------------------------------------------------------
零假设底空间（无真矛盾）下构造随机案件，每条证据：
· 以概率 p₀=0.30 携带形式缺陷（内容过短，真实 screen_asymmetric 计 PENDING）；
· 否则携带 1 个从 8 词对 15 个极性词中均匀随机抽取的触发词，
  并随机指派 4 个对象之一——不植入任何对立词对，凡命中的矛盾对
  都是机会错配 M_p；
· 每证据来源唯一（journal_i），使候选对数恰为 C(m,2)（排除同来源跳过）。
定理 1-B 的可证伪预测：
  (i)  缺陷计数 D_m ≈ p₀·m（线性，p̂₀ 跨 m 稳定）；
  (ii) 错配对数 K_m ≈ q_K·C(m,2)（二次，q̂_K 跨 m 稳定）；
  (iii) 比 D_m/K_m ∝ 1/(m−1)（log-log 斜率 ≈ −1），且不劣于
       Bonferroni 下界 2p₀/(q_K(m−1))；
  (iv) 全机制（aspect+门禁）q̂_K ≤ 全关 q̂_K（机制只缩 M 生成事件集）。
推论交叉点 m* = 2p₀/q_K + 1（错配预算收紧点）随报告输出，不设红线。

用法：
    backend$ .venv/bin/python tests/scripts/run_robustness_experiments.py
    backend$ .venv/bin/python tests/scripts/run_robustness_experiments.py \
        --permutations 1000 --trials 300

输出：tests/ddb/robustness_report.json + stdout 结果
退出码：0=全部预测满足；1=预测违例；2=运行异常
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import replace as dc_replace
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# 导入前置：backend 根入 sys.path + 环境变量占位（必须在 import app 之前）
# ---------------------------------------------------------------------------
_BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))
# tests/scripts 不是包（无 __init__.py）：直接以脚本目录导入同目录的
# run_ablation_v03（脚本直接运行时 sys.path[0] 即本目录；pytest 导入
# 场景显式补入），复用其 _to_evidence 证据还原口径，避免双份漂移。
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

os.environ.setdefault("SCIMINER_API_KEY", "robustness-placeholder")
os.environ.setdefault("DB_PASSWORD", "robustness-placeholder")

from app.core.virtual_lab.consensus_court import (  # noqa: E402
    Evidence,
    EvidenceScorer,
    RuleEngine,
)
from app.core.virtual_lab.rubric_engine import AsymmetricRubric  # noqa: E402
from tests.ddb.l1_generator import generate as generate_l1  # noqa: E402
from tests.ddb.l15_generator import generate as generate_l15  # noqa: E402

# 复用 run_ablation_v03._to_evidence 的证据还原口径（unknown 补占位引用），
# 与消融/对照/优势分解实验保持同一份证据重建逻辑，避免口径漂移。
from run_ablation_v03 import _to_evidence  # noqa: E402

#: 实验种子（与消融管线一致）
SEED = 42

#: 定理 1-B 扫描的证据数量格点（§6.5：n=3/5/10/20）
N_GRID = (3, 5, 10, 20)

#: 零假设案件的单证据形式缺陷率（D_i 的边际概率，内容过短 PENDING）
P0_DESIGNED = 0.30

#: 对象池：取 OBJECT_ALIASES 中 4 个互不同 canonical 的可识别专名
#: （EGFR/KRAS/TP53/HER2）。ValidityGate 真实解析同族/异族：双方同
#: canonical → SAME 放行；异族 → DIFFERENT 拦截。故全关配置下
#: N_pairs=C(m,2)，全开配置下期望同对象概率 1/POOL_SIZE——门禁对机会
#: 错配 q_K 的缩减量可由 q̂_full vs q̂_base 直接观测。
#: 这些专名同时向切面注入真实切面键（EGFR/KRAS→活性方向，
#: p53/HER2→表达调控），与 L2 真实语料的切面-对象耦合结构一致。
OBJECT_POOL = ("EGFR", "KRAS", "p53", "HER2")

#: 15 个极性词（取自 rubric_engine.KEYWORD_PAIR_SPECS 的 ka/kb，顺序无关；
#: "抑制"同时属于两个词对，与生产词典真实结构保持一致）
KEYWORD_POOL = (
    "有效", "无效", "成功", "失败", "阳性", "阴性", "激活", "抑制",
    "存在", "不存在", "高表达", "低表达", "促进", "上调", "下调",
)

#: 零假设案件的中立审理主题（不含任何触发词，避免主题文本污染切面/极性命中）
NULL_TOPIC = "多团队证据汇总审理议题"

#: 流式即判基线的"支持终局"连续强干净证据数档位：
#: k=1 = 首因终局（首条强干净证据即终止听审，主口径）；
#: k=2 = 连续两条一致强干净证据才终局（敏感性口径，检验效应强弱依赖）。
STREAMING_STREAK_GRID = (1, 2)

#: 强干净证据阈值：无 PENDING 且 EvidenceScore.overall_score ≥ 0.7
#: （0.7 与 MajorityVote 支持票、PIAgentArbiter 支持档同口径，跨实验一致）
STRONG_SUPPORT_SCORE = 0.7

#: 来源标签噪声改标比例（§6.5：±20% 错标——每案恰 20% 证据的来源被替换）
NOISE_RATIO = 0.20

#: 每案独立噪声复制次数（多次改标以估计翻转率的期望）
NOISE_REPLICATES = 50

#: 流式/噪声实验的裁决常量（与 baseline_arbiter 中文裁决口径一致）
_DECISION_REJECT = "驳回提案"
_DECISION_SUPPORT = "支持提案"


# ===========================================================================
# 实验 A：证据顺序置换（定理 2）
# ===========================================================================

def _court_signature(
    rubric: AsymmetricRubric,
    evidences: list[Evidence],
    topic: str,
) -> tuple:
    """计算一个案件在给定证据顺序下的裁决签名。

    签名包含全部顺序应不变的裁决要素（端点排序后做集合化）：
    - passed：是否通过（定罪与否）；
    - contradictions：(证据对端点排序, 强度) 多重集；
    - pending_pairs：AMBIGUOUS 待举证对多重集；
    - PI 等价的置信度不在此处——本函数只刻画 rubric 判定 V。

    Args:
        rubric: 已配置开关的 AsymmetricRubric 实例
        evidences: 按某一顺序排列的证据列表
        topic: 争议主题

    Returns:
        可哈希的裁决签名元组
    """
    scorer = EvidenceScorer()
    scores = [scorer.score(ev, topic) for ev in evidences]
    emap = {ev.evidence_id: ev for ev in evidences}
    verdict = rubric.evaluate(scores, emap, topic)
    contra = tuple(sorted(
        (tuple(sorted((c.evidence_a_id, c.evidence_b_id))), c.intensity)
        for c in verdict.contradictions
    ))
    # pending_pairs 元素与 contradictions 同为 ContradictionPair 对象，
    # 按相同口径规范化（端点排序 + 强度多重集）
    pending = tuple(sorted(
        (tuple(sorted((p.evidence_a_id, p.evidence_b_id))), p.intensity)
        for p in verdict.pending_pairs
    ))
    return (verdict.passed, contra, pending)


def run_order_permutation(
    total_permutations: int = 1000,
    seed: int = SEED,
) -> dict:
    """定理 2 顺序无关性检验：对 L1 全部正例做共计 total_permutations 次重排。

    Args:
        total_permutations: 总置换次数（均匀分配到全部正例，默认恰好 1000）
        seed: 随机种子

    Returns:
        结果字典：总置换数、分歧次数、逐案一致性、两配置汇总
    """
    cases = generate_l1(
        num_with_contradiction=60, num_no_contradiction=60, seed=seed
    )
    positives = [c for c in cases if c["polarity"] == "with_contradiction"]

    # 1000 次均匀分配：前 remainder 个案件多分 1 次
    base, remainder = divmod(total_permutations, len(positives))
    per_case = [base + (1 if i < remainder else 0) for i in range(len(positives))]

    rng = random.Random(seed)
    configs = {
        "full_v03": AsymmetricRubric(
            enable_aspect=True,
            enable_validity_gate=True,
            enable_intensity=True,
        ),
        "baseline_v02": AsymmetricRubric(
            enable_aspect=False,
            enable_validity_gate=False,
            enable_intensity=False,
        ),
    }
    per_case_results = []
    divergent = 0
    actual_total = 0

    for idx, case in enumerate(positives):
        evidences = [_to_evidence(ev) for ev in case["evidences"]]
        case_record = {"case_id": case["case_id"], "configs": {}}
        for config_name, rubric in configs.items():
            baseline_sig = _court_signature(rubric, evidences, case["topic"])
            n_perm = per_case[idx]
            mismatches = 0
            for _ in range(n_perm):
                shuffled = list(evidences)
                rng.shuffle(shuffled)
                sig = _court_signature(rubric, shuffled, case["topic"])
                if sig != baseline_sig:
                    mismatches += 1
            case_record["configs"][config_name] = {
                "permutations": n_perm,
                "mismatches": mismatches,
                "invariant": mismatches == 0,
            }
            divergent += mismatches
        actual_total += n_perm
        per_case_results.append(case_record)

    return {
        "total_permutations": actual_total,
        "n_positive_cases": len(positives),
        "divergent_permutations": divergent,
        "invariant_rate": (
            round(1 - divergent / actual_total, 6) if actual_total else None
        ),
        "per_case": per_case_results,
    }


# ===========================================================================
# 实验 B：证据数量扫描（定理 1-B）
# ===========================================================================

def build_null_case(m: int, case_idx: int, rng: random.Random) -> list[Evidence]:
    """构造一个零假设（无真矛盾）随机案件的证据集。

    构造纪律（与定理 1-B 底空间 C=∅ 对齐）：
    - 绝不植入完整对立词对：每条证据恰好 0 或 1 个极性词，对立只能由
      不同证据的独立随机抽取偶然形成（=机会错配 M_p，正是被测对象）；
    - 缺陷与极性词互斥：缺陷证据写短文本（真实触发 screen_asymmetric
      的"证据内容过短"PENDING），不携带触发词，保证 D_i 与词表解耦；
    - 来源唯一 journal_i：rubric 同来源证据对跳过，唯一来源使候选对数
      恰为 C(m,2)，与定理记号 N_pairs 一致；
    - 对象从 OBJECT_POOL 均匀抽取（全关配置不使用；全开配置下同对象
      概率 = 1/|pool|，门禁对错配的缩减可被 q̂ 差值观测）。

    Args:
        m: 证据数量
        case_idx: 案件序号（用于生成不冲突的证据/引用 ID）
        rng: 已播种随机数生成器

    Returns:
        Evidence 列表（长度恰为 m）
    """
    evidences: list[Evidence] = []
    now = datetime.utcnow()
    for i in range(m):
        eid = f"N{case_idx:05d}-E{i:03d}"
        if rng.random() < P0_DESIGNED:
            # 形式缺陷：strip 后长度 < 10，真实命中 PENDING，且不含任何极性词
            content = "数据待补充"
        else:
            obj = rng.choice(OBJECT_POOL)
            kw = rng.choice(KEYWORD_POOL)
            # 长文本（>10 字符）中性叙事包裹单个随机极性词
            content = f"独立研究组重复观测显示{obj}在该模型中呈现{kw}表型，实验记录完整可追溯。"
        evidences.append(Evidence(
            evidence_id=eid,
            source=f"journal_{case_idx}_{i}",  # 来源唯一 → 候选对 = C(m,2)
            content=content,
            role="drug-researcher",
            timestamp=now - timedelta(days=i),
            reference_ids=[f"REF-{case_idx}-{i}"],
            metadata={"synthetic_null": True},
        ))
    return evidences


def _count_defects_pending(evidences: list[Evidence]) -> int:
    """用真实 RuleEngine.screen_asymmetric 统计 PENDING 形式缺陷证据数。

    Args:
        evidences: 案件证据集

    Returns:
        至少携带一条 PENDING 问题的证据条数（|{i : D_i}| 的实现口径）
    """
    rule_engine = RuleEngine()
    count = 0
    for ev in evidences:
        severities = [sev for _desc, sev in rule_engine.screen_asymmetric(ev)]
        if "PENDING" in severities:
            count += 1
    return count


def _count_false_pairs(
    rubric: AsymmetricRubric,
    evidences: list[Evidence],
) -> tuple[int, int]:
    """用真实 AsymmetricRubric 统计机会错配对数与 AMBIGUOUS 待举证对数。

    Args:
        rubric: 配置好的 rubric（全关/全开两套）
        evidences: 零假设案件证据集

    Returns:
        (定罪矛盾对数 |{p : M_p 且过门禁}|, AMBIGUOUS 待举证对数)
    """
    scorer = EvidenceScorer()
    scores = [scorer.score(ev, NULL_TOPIC) for ev in evidences]
    emap = {ev.evidence_id: ev for ev in evidences}
    verdict = rubric.evaluate(scores, emap, NULL_TOPIC)
    return len(verdict.contradictions), len(verdict.pending_pairs)


def _loglog_slope(xs: list[float], ys: list[float]) -> float:
    """对 (x,y) 做 log-log 一元线性回归，返回斜率（最小二乘）。

    Args:
        xs: 自变量（必须全部 > 0）
        ys: 因变量（必须全部 > 0；调用方需保证零计数时不传）

    Returns:
        最小二乘斜率 b（y = a·x^b 指数律）
    """
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    n = len(lx)
    mx, my = sum(lx) / n, sum(ly) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(lx, ly))
    var = sum((x - mx) ** 2 for x in lx)
    return cov / var if var > 0 else 0.0


def run_n_scan(trials: int = 300, seed: int = SEED) -> dict:
    """定理 1-B 证据数量扫描：n=3/5/10/20 各 trials 个零假设案件。

    Args:
        trials: 每个 m 的随机案件数（默认 300）
        seed: 随机种子

    Returns:
        逐 m 计数统计 + 标度斜率 + Bonferroni 下界对照 + 红线检查所需原始量
    """
    rng = random.Random(seed)
    rubric_base = AsymmetricRubric(
        enable_aspect=False, enable_validity_gate=False, enable_intensity=False,
    )
    rubric_full = AsymmetricRubric(
        enable_aspect=True, enable_validity_gate=True, enable_intensity=True,
    )

    rows = []
    for m in N_GRID:
        n_pairs = m * (m - 1) // 2
        sum_defects = 0
        sum_pairs_base = 0
        sum_pairs_full = 0
        sum_ambiguous_full = 0
        for case_idx in range(trials):
            evidences = build_null_case(m, case_idx, rng)
            sum_defects += _count_defects_pending(evidences)
            kb, _ab = _count_false_pairs(rubric_base, evidences)
            kf, amb = _count_false_pairs(rubric_full, evidences)
            sum_pairs_base += kb
            sum_pairs_full += kf
            sum_ambiguous_full += amb

        d_bar = sum_defects / trials
        k_bar_base = sum_pairs_base / trials
        k_bar_full = sum_pairs_full / trials
        p0_hat = d_bar / m
        q_hat_base = k_bar_base / n_pairs
        q_hat_full = k_bar_full / n_pairs
        row = {
            "m": m,
            "n_pair_opportunities": n_pairs,
            "trials": trials,
            "mean_defect_count": round(d_bar, 4),
            "mean_false_pair_base": round(k_bar_base, 4),
            "mean_false_pair_full": round(k_bar_full, 4),
            "mean_ambiguous_full": round(sum_ambiguous_full / trials, 4),
            "p0_hat": round(p0_hat, 4),
            "q_hat_base": round(q_hat_base, 6),
            "q_hat_full": round(q_hat_full, 6),
        }
        # 比与 Bonferroni 下界（全关配置，无 aspect：N_pairs=C(m,2)）
        if k_bar_base > 0:
            row["ratio_empirical"] = round(d_bar / k_bar_base, 4)
            row["bonferroni_bound"] = round(
                2 * p0_hat / (q_hat_base * (m - 1)), 4
            )
            row["ratio_over_bound"] = round(
                (d_bar / k_bar_base)
                / (2 * p0_hat / (q_hat_base * (m - 1))),
                4,
            )
        rows.append(row)

    # 标度律回归（仅对有错配计数的格点；m 越大计数越大，通常全部非零）
    fit_rows = [r for r in rows if r["mean_false_pair_base"] > 0]
    slope_pairs = _loglog_slope(
        [r["n_pair_opportunities"] for r in fit_rows],
        [r["mean_false_pair_base"] for r in fit_rows],
    )
    ratio_rows = [r for r in fit_rows if r.get("ratio_empirical")]
    slope_ratio = _loglog_slope(
        [r["m"] - 1 for r in ratio_rows],
        [r["ratio_empirical"] for r in ratio_rows],
    )

    # 推论交叉点 m* = 2p0/q + 1（用全关 q 的跨 m 合并估计）
    total_opp_base = sum(r["n_pair_opportunities"] * r["trials"] for r in rows)
    total_pairs_base = sum(
        r["mean_false_pair_base"] * r["trials"] for r in rows
    )
    q_pooled_base = (
        total_pairs_base / total_opp_base if total_opp_base else 0.0
    )
    p0_pooled = P0_DESIGNED  # 构造参数，实测 p0_hat 跨 m 稳定性另做红线
    crossover_m = (
        round(2 * p0_pooled / q_pooled_base + 1, 2)
        if q_pooled_base > 0 else None
    )

    return {
        "p0_designed": P0_DESIGNED,
        "object_pool_size": len(OBJECT_POOL),
        "keyword_pool_size": len(KEYWORD_POOL),
        "rows": rows,
        "scaling": {
            "slope_false_pairs_vs_Cm2": round(slope_pairs, 4),
            "slope_ratio_vs_m_minus_1": round(slope_ratio, 4),
            "q_hat_base_pooled": round(q_pooled_base, 6),
            "crossover_m_star": crossover_m,
        },
    }


# ===========================================================================
# 实验 C：流式即判基线（§5.5 定理 2 推论的对照侧 / §6.4 去 DrillBeforeDisposition）
# ===========================================================================

def _prefix_has_pending(rule_engine: RuleEngine, prefix: list[Evidence]) -> bool:
    """判断当前前缀中是否存在任何 PENDING 级形式缺陷未消解。

    流式基线下 PENDING 不直接终局（忠实 ABP"缺陷不等于定罪"的处理），
    但只要在案缺陷未消解，就不得作"支持"终局——待举证未完成。

    Args:
        rule_engine: 真实规则引擎
        prefix: 已到达证据前缀

    Returns:
        True 表示前缀内至少一条证据携带 PENDING
    """
    for ev in prefix:
        severities = [sev for _d, sev in rule_engine.screen_asymmetric(ev)]
        if "PENDING" in severities:
            return True
    return False


def streaming_disposition(
    evidences: list[Evidence],
    topic: str,
    rubric: AsymmetricRubric,
    scorer: EvidenceScorer,
    rule_engine: RuleEngine,
    streak_k: int = 1,
) -> tuple[str, int, str]:
    """流式即判基线：按证据到达序逐条审理，命中终局条件立即结案。

    本构造是论文大纲 §6.4 消融项"去掉 DrillBeforeDisposition（流式即判）"
    与 §5.5 定理 2 推论（"流式即判基线按构造违反顺序不变性"）的规则式
    实验基线，不是某个生产系统的复刻：它刻意保留顺序原语（首因终局 +
    提前终止），与法庭的 DrillBeforeDisposition（审完全部证据再合议）
    形成结构性对照。

    终局规则（按证据到达序执行，每到达一条就在当前前缀上用真实
    AsymmetricRubric 重新合议）：
    1. 矛盾终局：前缀首次形成过门禁矛盾对 → 立即"驳回提案"结案，
       后续证据不再听取（提前终止）；
    2. 支持终局：在案无任何未消解 PENDING 时，连续 streak_k 条
       "强干净证据"（overall_score ≥ 0.7）到达 → 立即"支持提案"结案；
       任一条不满足即打断连续计数；
    3. 全部证据审理完毕仍无终局 → 默认"支持提案"（无罪推定兜底）。

    Args:
        evidences: 按某一到达顺序排列的证据列表
        topic: 争议主题
        rubric: 真实 ABP rubric（全开配置，与法庭同检测口径）
        scorer: 真实证据评分器
        rule_engine: 真实规则引擎（PENDING 判定）
        streak_k: 支持终局所需连续强干净证据数（1=首因终局，2=双证终局）

    Returns:
        (decision, lock_step, lock_reason)：decision ∈ {驳回提案, 支持提案}；
        lock_step 为结案时已审理的前缀长度（1..m）；lock_reason 标注
        "矛盾终局"/"支持终局"/"默认无罪"。
    """
    prefix: list[Evidence] = []
    strong_streak = 0
    for step, ev in enumerate(evidences, start=1):
        prefix.append(ev)
        # 规则 1：在当前前缀上用真实 rubric 合议——矛盾对首次形成即终局
        scores = [scorer.score(e, topic) for e in prefix]
        emap = {e.evidence_id: e for e in prefix}
        verdict = rubric.evaluate(scores, emap, topic)
        if verdict.contradictions:
            return _DECISION_REJECT, step, "矛盾终局"
        # 规则 2：在案有未消解 PENDING 时禁止支持终局（待举证未完成）
        if _prefix_has_pending(rule_engine, prefix):
            strong_streak = 0
        elif scores[-1].overall_score >= STRONG_SUPPORT_SCORE:
            strong_streak += 1
            if strong_streak >= streak_k:
                return _DECISION_SUPPORT, step, "支持终局"
        else:
            strong_streak = 0
    return _DECISION_SUPPORT, len(prefix), "默认无罪"


def run_streaming_experiment(
    total_permutations: int = 1000,
    seed: int = SEED + 1,
) -> dict:
    """定理 2 对照侧：流式即判基线在证据置换下的顺序敏感性检验。

    与实验 A 相同的置换分配（60 个 L1 正例均匀分配 total_permutations 次），
    但裁决器换成 streaming_disposition。可证伪预测：
    - 流式基线的裁决随置换发生分歧（分歧率严格大于 0，主口径 k=1
      预注册红线 ≥0.02）；
    - 顺序扰动产生实际错放后果——置换平均召回严格低于规范顺序召回。

    Args:
        total_permutations: 总置换次数（默认 1000）
        seed: 随机种子（默认 SEED+1，与实验 A 的 RNG 流独立）

    Returns:
        逐 k 值的分歧统计 + 规范/置换召回对比
    """
    cases = generate_l1(
        num_with_contradiction=60, num_no_contradiction=60, seed=SEED
    )
    positives = [c for c in cases if c["polarity"] == "with_contradiction"]

    base, remainder = divmod(total_permutations, len(positives))
    per_case = [base + (1 if i < remainder else 0) for i in range(len(positives))]

    scorer = EvidenceScorer()
    rule_engine = RuleEngine()
    rubric = AsymmetricRubric(
        enable_aspect=True, enable_validity_gate=True, enable_intensity=True,
    )
    rng = random.Random(seed)
    configs_out: dict[str, dict] = {}

    # 规范流程（DrillBeforeDisposition：先审完全部证据再合议）在同一批正例
    # 上的召回——全开 rubric 对完整证据集一次性裁决。这是论文口径中的
    # "规范召回"，也是定理 2 主侧（实验 A）与流式对照侧的共同参照。
    court_reject = 0
    for case in positives:
        full_evidences = [_to_evidence(ev) for ev in case["evidences"]]
        full_scores = [scorer.score(e, case["topic"]) for e in full_evidences]
        full_map = {e.evidence_id: e for e in full_evidences}
        full_verdict = rubric.evaluate(full_scores, full_map, case["topic"])
        if full_verdict.contradictions:
            court_reject += 1
    court_recall = round(court_reject / len(positives), 4)

    for k in STREAMING_STREAK_GRID:
        canonical_reject = 0
        divergent = 0
        perm_reject = 0
        actual_total = 0
        affected_cases = 0

        for idx, case in enumerate(positives):
            evidences = [_to_evidence(ev) for ev in case["evidences"]]
            canonical, _step, _reason = streaming_disposition(
                evidences, case["topic"], rubric, scorer, rule_engine, streak_k=k,
            )
            if canonical == _DECISION_REJECT:
                canonical_reject += 1
            case_mismatch = 0
            for _ in range(per_case[idx]):
                shuffled = list(evidences)
                rng.shuffle(shuffled)
                decision, _s, _r = streaming_disposition(
                    shuffled, case["topic"], rubric, scorer, rule_engine,
                    streak_k=k,
                )
                actual_total += 1
                if decision == _DECISION_REJECT:
                    perm_reject += 1
                if decision != canonical:
                    divergent += 1
                    case_mismatch += 1
            if case_mismatch:
                affected_cases += 1

        configs_out[f"k{k}"] = {
            "streak_k": k,
            "permutations": actual_total,
            "divergent_permutations": divergent,
            "divergence_rate": round(divergent / actual_total, 6) if actual_total else None,
            "affected_cases": affected_cases,
            "canonical_recall": round(canonical_reject / len(positives), 4),
            "permuted_recall": round(perm_reject / actual_total, 4) if actual_total else None,
        }

    return {
        "total_permutations": configs_out[f"k{STREAMING_STREAK_GRID[0]}"]["permutations"],
        "n_positive_cases": len(positives),
        "court_recall": court_recall,
        "configs": configs_out,
    }


# ===========================================================================
# 实验 D：来源标签噪声 ±20%（§6.5 来源鲁棒性）
# ===========================================================================

def build_l1_l15_pool(cases_l1: int = 60) -> tuple[list[dict], list[dict]]:
    """构造来源噪声实验案件池（L1 60×2 + L1.5 88），按极性分正/负两组。

    Args:
        cases_l1: L1 每极性案件数

    Returns:
        (positives, negatives)：正例 = L1 with_contradiction；
        负例 = L1 no_contradiction + 全部 L1.5 伪冲突负例
    """
    l1_cases = generate_l1(
        num_with_contradiction=cases_l1,
        num_no_contradiction=cases_l1,
        seed=SEED,
    )
    positives = [c for c in l1_cases if c["polarity"] == "with_contradiction"]
    negatives = [c for c in l1_cases if c["polarity"] != "with_contradiction"]
    negatives.extend(generate_l15(per_template=8, seed=SEED))
    return positives, negatives


def corrupt_sources(
    evidences: list[Evidence],
    rng: random.Random,
    ratio: float = NOISE_RATIO,
) -> list[Evidence]:
    """对案件证据集施加来源标签噪声：恰 ceil(ratio·m) 条证据的 source 被替换。

    噪声模型（§6.5"来源标签噪声 ±20% 错标"）：
    - 被选中的证据 source 替换为来源谱系（EvidenceScorer.SOURCE_WEIGHTS
      八键）中均匀随机的一个键，允许与原标签相同（保守的非条件采样，
      不利用后验信息挑"最具破坏性"的标签）；
    - **仅改 source**：content / evidence_id / reference_ids / role /
      timestamp / metadata 全部原样保留——本实验隔离来源标签单变量。

    Args:
        evidences: 原始证据列表
        rng: 已播种随机数生成器（逐复制独立）
        ratio: 改标比例（默认 0.20）

    Returns:
        改标后的新证据列表（dataclasses.replace 浅拷贝，原对象不被修改）
    """
    m = len(evidences)
    n_corrupt = max(1, round(ratio * m))
    source_keys = list(EvidenceScorer.SOURCE_WEIGHTS.keys())
    victims = rng.sample(range(m), n_corrupt)
    corrupted = list(evidences)
    for idx in victims:
        corrupted[idx] = dc_replace(
            corrupted[idx], source=rng.choice(source_keys)
        )
    return corrupted


def _court_conviction_signature(
    evidences: list[Evidence],
    topic: str,
    rubric: AsymmetricRubric,
    scorer: EvidenceScorer,
) -> tuple[bool, frozenset, int, float]:
    """计算案件在全开法庭下的定罪签名。

    Args:
        evidences: 证据列表（原始或噪声改标后）
        topic: 争议主题
        rubric: 全开 rubric
        scorer: 评分器

    Returns:
        (convicted, pair_endpoints_set, max_intensity, confidence)
    """
    scores = [scorer.score(ev, topic) for ev in evidences]
    emap = {ev.evidence_id: ev for ev in evidences}
    verdict = rubric.evaluate(scores, emap, topic)
    pairs = frozenset(
        tuple(sorted((p.evidence_a_id, p.evidence_b_id)))
        for p in verdict.contradictions
    )
    max_intensity = max(
        (p.intensity for p in verdict.contradictions), default=0
    )
    return bool(verdict.contradictions), pairs, max_intensity, verdict.confidence


def run_source_noise_experiment(
    replicates: int = NOISE_REPLICATES,
    ratio: float = NOISE_RATIO,
    seed: int = SEED + 2,
) -> dict:
    """±20% 来源标签噪声下的法庭鲁棒性检验。

    预测（全部来自结构分析，预注册红线）：
    - 负例误增定罪率 = 0（**基准级结构论断**，非普遍定理）：本底空间的
      构造纪律排除了来源通道的全部构陷路径——L1 负例证据零矛盾关键词
      （生成器 _selfcheck_clean_case 强制），L1.5 核心词对恒跨来源采样
      （_sample_source_pair）且中性干扰零关键词，因此唯一的理论构陷通道
      "同源跳过解除"（改标使原本同源被跳过的内容命中对恢复比较）在负例中
      无被跳过的命中对可供解除。红线为该论断的**经验可证伪检验**：任何
      误增定罪都会连同矛盾对端点写入 false_gain_cases 审计字段供排查；
    - 正例噪声后召回 ≥0.95：唯一的漏放通道是"同源不对比"——核心对立
      双方恰被改标成同一来源时该对被跳过（效应方向只可能压制、不可能构陷）；
    - 置信度漂移有界：来源占 overall_score 的 0.4 权重，单次改标影响有限。

    Args:
        replicates: 每案独立噪声复制次数（默认 50）
        ratio: 每复制的改标比例（默认 0.20）
        seed: 随机种子（默认 SEED+2，独立 RNG 流）

    Returns:
        正负例翻转统计、矛盾对/强度翻转率、置信度漂移（含最大值）
    """
    positives, negatives = build_l1_l15_pool()
    scorer = EvidenceScorer()
    rubric = AsymmetricRubric(
        enable_aspect=True, enable_validity_gate=True, enable_intensity=True,
    )
    rng = random.Random(seed)

    def _sweep(cases: list[dict]) -> dict:
        """对一组案件跑基准签名 + replicates 次噪声复制，累计翻转统计。"""
        n_reps = 0
        false_gain = 0          # 基准未定罪 → 噪声后定罪（负例专用指标）
        false_gain_cases: list[dict] = []  # oracle 审计：误增案件的矛盾对端点
        conviction_lost = 0     # 基准定罪 → 噪声后未定罪（正例专用指标）
        pair_set_flips = 0      # 矛盾对端点集合发生变化
        intensity_flips = 0     # 最大强度发生变化
        conf_drift_sum = 0.0
        conf_drift_max = 0.0

        for case in cases:
            evidences = [_to_evidence(ev) for ev in case["evidences"]]
            base_conv, base_pairs, base_int, base_conf = (
                _court_conviction_signature(evidences, case["topic"], rubric, scorer)
            )
            for _ in range(replicates):
                corrupted = corrupt_sources(evidences, rng, ratio=ratio)
                # 不变量守卫（oracle）：除 source 外任何字段变化都是扰动器 bug
                for orig, new in zip(evidences, corrupted):
                    assert orig.content == new.content
                    assert orig.evidence_id == new.evidence_id
                    assert orig.reference_ids == new.reference_ids
                    assert orig.role == new.role
                    assert orig.timestamp == new.timestamp
                    assert orig.metadata == new.metadata
                conv, pairs, inten, conf = _court_conviction_signature(
                    corrupted, case["topic"], rubric, scorer
                )
                n_reps += 1
                if (not base_conv) and conv:
                    false_gain += 1
                    # oracle 审计：记录误增案件与其矛盾对端点（应为空列表）
                    false_gain_cases.append({
                        "case_id": case["case_id"],
                        "pairs": sorted(tuple(sorted(p)) for p in pairs),
                    })
                if base_conv and (not conv):
                    conviction_lost += 1
                if pairs != base_pairs:
                    pair_set_flips += 1
                if inten != base_int:
                    intensity_flips += 1
                drift = abs(conf - base_conf)
                conf_drift_sum += drift
                conf_drift_max = max(conf_drift_max, drift)

        return {
            "cases": len(cases),
            "replicates": n_reps,
            "false_gain": false_gain,
            "false_gain_rate": round(false_gain / n_reps, 6) if n_reps else None,
            "false_gain_cases": false_gain_cases,
            "conviction_lost": conviction_lost,
            "conviction_lost_rate": (
                round(conviction_lost / n_reps, 6) if n_reps else None
            ),
            "pair_set_flip_rate": (
                round(pair_set_flips / n_reps, 6) if n_reps else None
            ),
            "intensity_flip_rate": (
                round(intensity_flips / n_reps, 6) if n_reps else None
            ),
            "mean_abs_confidence_drift": (
                round(conf_drift_sum / n_reps, 6) if n_reps else None
            ),
            "max_abs_confidence_drift": round(conf_drift_max, 6),
        }

    pos_stats = _sweep(positives)
    neg_stats = _sweep(negatives)
    # 正例噪声后召回 = 1 − 漏放率（唯一敏感通道是同源跳过）
    pos_stats["noisy_recall"] = (
        round(1 - pos_stats["conviction_lost_rate"], 6)
        if pos_stats["conviction_lost_rate"] is not None else None
    )

    # 负例按 L1/L1.5 分层再扫一遍，给出"零构陷"结论的分层证据
    neg_l1 = [c for c in negatives if c["case_id"].startswith("DDB-L1-")]
    neg_l15 = [c for c in negatives if c["case_id"].startswith("DDB-L15-")]
    neg_l1_stats = _sweep(neg_l1)
    neg_l15_stats = _sweep(neg_l15)

    return {
        "noise_ratio": ratio,
        "replicates_per_case": replicates,
        "positives": pos_stats,
        "negatives": neg_stats,
        "negatives_l1": {
            "cases": neg_l1_stats["cases"],
            "false_gain": neg_l1_stats["false_gain"],
            "false_gain_rate": neg_l1_stats["false_gain_rate"],
        },
        "negatives_l15": {
            "cases": neg_l15_stats["cases"],
            "false_gain": neg_l15_stats["false_gain"],
            "false_gain_rate": neg_l15_stats["false_gain_rate"],
        },
    }


def check_robustness_red_lines(
    perm: dict,
    scan: dict,
    streaming: dict | None = None,
    noise: dict | None = None,
) -> list[str]:
    """鲁棒性实验的可证伪条款检查。

    Args:
        perm: run_order_permutation 结果
        scan: run_n_scan 结果
        streaming: run_streaming_experiment 结果（可选；传入时检查实验 C）
        noise: run_source_noise_experiment 结果（可选；传入时检查实验 D）

    Returns:
        违例描述列表（空 = 全部满足）
    """
    violations: list[str] = []

    # A 红线：1000 次置换零分歧（定理 2）
    if perm["divergent_permutations"] != 0:
        violations.append(
            f"定理 2 违例：{perm['total_permutations']} 次置换中 "
            f"{perm['divergent_permutations']} 次裁决分歧（合取顺序不变性失败）"
        )

    rows = scan["rows"]

    # B 红线 1：缺陷率 p̂₀ 跨 m 稳定（线性计数），容差 ±0.10
    for r in rows:
        if abs(r["p0_hat"] - P0_DESIGNED) > 0.10:
            violations.append(
                f"定理 1-B-(i) 违例：m={r['m']} 时 p0_hat={r['p0_hat']}"
                f" 偏离设计值 {P0_DESIGNED}（缺陷计数应线性于 m）"
            )

    # B 红线 2：错配率 q̂_K 跨 m 稳定（边际率不随 m 漂移），
    # 最大/最小比 ≤ 2.0（抽样波动容差，机会计数在 m=3 最稀疏）
    q_vals = [r["q_hat_base"] for r in rows if r["q_hat_base"] > 0]
    if q_vals and max(q_vals) / min(q_vals) > 2.0:
        violations.append(
            f"定理 1-B-(ii) 违例：q_hat_base 跨 m 波动过大：{q_vals}"
            "（单对错配率应是与 m 无关的边际量）"
        )

    # B 红线 3：错配对数二次增长（log-log 斜率 ∈ [0.85, 1.15]）
    sp = scan["scaling"]["slope_false_pairs_vs_Cm2"]
    if not (0.85 <= sp <= 1.15):
        violations.append(
            f"定理 1-B-(ii) 违例：错配对数对 C(m,2) 的 log-log 斜率={sp}"
            "（期望 ≈ 1.0 的二次增长）"
        )

    # B 红线 4：比值 1/(m−1) 衰减（斜率 ∈ [−1.3, −0.7]）
    sr = scan["scaling"]["slope_ratio_vs_m_minus_1"]
    if not (-1.3 <= sr <= -0.7):
        violations.append(
            f"定理 1-B-(iii) 违例：D/K 比对 (m−1) 的 log-log 斜率={sr}"
            "（期望 ≈ −1.0 的 1/(m−1) 衰减）"
        )

    # B 红线 5：经验比不劣于 Bonferroni 下界（允许 −20% 抽样容差）
    for r in rows:
        if "ratio_over_bound" in r and r["ratio_over_bound"] < 0.8:
            violations.append(
                f"定理 1-B 界违例：m={r['m']} 经验比/下界="
                f"{r['ratio_over_bound']} < 0.8（期望计数不得系统性劣于下界）"
            )

    # B 红线 6：全机制 q̂ ≤ 全关 q̂（aspect/门禁只缩减错配生成事件集）
    for r in rows:
        if r["q_hat_full"] > r["q_hat_base"] + 1e-9:
            violations.append(
                f"机制方向违例：m={r['m']} q_hat_full={r['q_hat_full']}"
                f" > q_hat_base={r['q_hat_base']}"
                "（aspect/门禁只能缩小 M 的生成事件集）"
            )

    if streaming is not None:
        # C 红线 1（主口径 k=1）：流式即判在置换下必须显著顺序敏感——
        # 分歧率严格 > 0 且 ≥0.02（探针预注册值 0.063，留 3 倍余量）
        k1 = streaming["configs"]["k1"]
        if k1["divergent_permutations"] <= 0:
            violations.append(
                "实验 C 违例：流式即判 k=1 在置换下零分歧"
                "（顺序原语未产生顺序效应，定理 2 对照侧证伪）"
            )
        if k1["divergence_rate"] < 0.02:
            violations.append(
                f"实验 C 违例：流式 k=1 分歧率 {k1['divergence_rate']} < 0.02"
                "（首因终局的顺序敏感性应显著）"
            )
        # C 红线 2：顺序扰动必须产生真实错放——流式置换平均召回严格低于
        # 规范流程召回（DrillBeforeDisposition 全案合议，对照口径）
        pr = k1["permuted_recall"]
        cr = streaming["court_recall"]
        if pr is not None and pr >= cr:
            violations.append(
                f"实验 C 违例：流式 k=1 置换召回 {pr} ≥ 规范流程召回 {cr}"
                "（顺序扰动应有可测的错放后果）"
            )
        # 注：实验 A 法庭零分歧由 A 红线统一检查（同次运行 perm 结果），
        # 形成"法庭顺序不变 / 流式顺序敏感"的同场对照。

    if noise is not None:
        # D 红线 1：负例误增定罪率严格为 0（基准级结构论断的经验检验：
        # 构造纪律排除"同源跳过解除"构陷通道，详见 run_source_noise_experiment）
        neg_rate = noise["negatives"]["false_gain_rate"]
        if neg_rate != 0:
            audit = noise["negatives"].get("false_gain_cases") or []
            violations.append(
                f"实验 D 违例：负例误增定罪率 {neg_rate} > 0"
                f"（本基准构造纪律下来源改标不得凭空制造矛盾；"
                f"审计 {len(audit)} 案："
                f"{[c['case_id'] for c in audit[:5]]}）"
            )
        # D 红线 2：正例噪声后召回 ≥0.95（同源跳过是唯一且窄的漏放通道）
        recall = noise["positives"]["noisy_recall"]
        if recall is not None and recall < 0.95:
            violations.append(
                f"实验 D 违例：正例噪声后召回 {recall} < 0.95"
                "（来源噪声对真实矛盾的压制超出预注册容差）"
            )
        # D 红线 3：置信度平均漂移有界（来源权重 0.4，单次改标影响有限）
        mean_drift = noise["positives"]["mean_abs_confidence_drift"]
        if mean_drift is not None and mean_drift > 0.05:
            violations.append(
                f"实验 D 违例：置信度漂移 |Δconf| 均值 {mean_drift} > 0.05"
                "（20% 来源错标对置信度的扰动应有限）"
            )

    return violations


def main() -> int:
    """主流程：置换检验 → n 扫描 → 流式对照 → 来源噪声 → 红线检查 → 落盘报告。"""
    parser = argparse.ArgumentParser(
        description="§6.5 鲁棒性专项实验（定理 2 + 定理 1-B + 流式对照 + 来源噪声）"
    )
    parser.add_argument(
        "--permutations", type=int, default=1000,
        help="顺序置换总次数（实验 A 法庭 / 实验 C 流式共用，默认 1000）",
    )
    parser.add_argument(
        "--trials", type=int, default=300,
        help="每个 n 格点的零假设案件数（默认 300）",
    )
    parser.add_argument(
        "--noise-replicates", type=int, default=NOISE_REPLICATES,
        help=f"来源噪声每案独立复制次数（实验 D，默认 {NOISE_REPLICATES}）",
    )
    args = parser.parse_args()

    print(f"[1/5] 实验 A：法庭证据顺序置换 {args.permutations} 次（定理 2）...")
    perm = run_order_permutation(total_permutations=args.permutations, seed=SEED)
    print(
        f"      置换 {perm['total_permutations']} 次 / "
        f"{perm['n_positive_cases']} 个正例 / "
        f"分歧 {perm['divergent_permutations']} 次 / "
        f"顺序不变率 {perm['invariant_rate']}"
    )

    print(f"[2/5] 实验 B：n={list(N_GRID)} 零假设扫描（每格点 {args.trials} 案，定理 1-B）...")
    scan = run_n_scan(trials=args.trials, seed=SEED)
    print(
        f"      {'m':>3}{'C(m,2)':>8}{'D_m':>9}{'K_base':>9}{'K_full':>9}"
        f"{'p0_hat':>9}{'q_base':>9}{'q_full':>9}{'D/K':>8}{'下界':>8}"
    )
    for r in scan["rows"]:
        print(
            f"      {r['m']:>3}{r['n_pair_opportunities']:>8}"
            f"{r['mean_defect_count']:>9.3f}{r['mean_false_pair_base']:>9.3f}"
            f"{r['mean_false_pair_full']:>9.3f}{r['p0_hat']:>9.3f}"
            f"{r['q_hat_base']:>9.4f}{r['q_hat_full']:>9.4f}"
            f"{r.get('ratio_empirical', float('nan')):>8.3f}"
            f"{r.get('bonferroni_bound', float('nan')):>8.3f}"
        )
    sc = scan["scaling"]
    print(
        f"      标度：K~C(m,2) 斜率={sc['slope_false_pairs_vs_Cm2']}"
        f"（期望≈1.0）；D/K~(m−1) 斜率={sc['slope_ratio_vs_m_minus_1']}"
        f"（期望≈−1.0）；交叉点 m*={sc['crossover_m_star']}"
    )

    print(
        f"[3/5] 实验 C：流式即判基线顺序置换 {args.permutations} 次"
        f"（k=1 主口径 + k=2 敏感性，定理 2 对照侧）..."
    )
    streaming = run_streaming_experiment(
        total_permutations=args.permutations, seed=SEED + 1,
    )
    for k in STREAMING_STREAK_GRID:
        c = streaming["configs"][f"k{k}"]
        print(
            f"      [k={k}] 分歧 {c['divergent_permutations']}/{c['permutations']}"
            f"={c['divergence_rate']} / 受影响案件 {c['affected_cases']} / "
            f"流式规范序召回 {c['canonical_recall']} → 置换召回 {c['permuted_recall']}"
            f"（规范流程召回 {streaming['court_recall']}）"
        )

    print(
        f"[4/5] 实验 D：来源标签 ±{int(NOISE_RATIO * 100)}% 噪声"
        f"（每案 {args.noise_replicates} 复制，L1 120 + L1.5 88 案）..."
    )
    noise = run_source_noise_experiment(
        replicates=args.noise_replicates, ratio=NOISE_RATIO, seed=SEED + 2,
    )
    print(
        f"      负例误增定罪 {noise['negatives']['false_gain']}"
        f"/{noise['negatives']['replicates']}"
        f"={noise['negatives']['false_gain_rate']}"
        f"（L1={noise['negatives_l1']['false_gain']} / "
        f"L1.5={noise['negatives_l15']['false_gain']}）"
    )
    print(
        f"      正例定罪丢失 {noise['positives']['conviction_lost']}"
        f"/{noise['positives']['replicates']}"
        f"={noise['positives']['conviction_lost_rate']}"
        f"（噪声后召回 {noise['positives']['noisy_recall']}）；"
        f"矛盾对集合翻转率 {noise['positives']['pair_set_flip_rate']}；"
        f"|Δconf| 均值 {noise['positives']['mean_abs_confidence_drift']} / "
        f"max {noise['positives']['max_abs_confidence_drift']}"
    )

    print("[5/5] 红线检查 + 写报告...")
    violations = check_robustness_red_lines(perm, scan, streaming, noise)
    for v in violations:
        print(f"      [违例] {v}")
    if not violations:
        print(
            "      全部预测满足：定理 2 顺序不变性 + 定理 1-B 计数标度律 "
            "+ 流式顺序敏感对照 + 来源噪声非构陷性"
        )

    report = {
        "experiment": (
            "§6.5 鲁棒性专项实验（定理 2 顺序无关性 + 定理 1-B n 扫描 "
            "+ 流式即判对照 + 来源标签噪声）"
        ),
        "benchmark": (
            "DDB-L1 正例（置换/噪声）+ DDB-L1 负例（噪声）+ "
            "DDB-L1.5 伪冲突负例（噪声）+ 受控零假设合成案件（n 扫描）"
        ),
        "seed": SEED,
        "order_permutation": perm,
        "n_scan": scan,
        "streaming_disposition": streaming,
        "source_noise": noise,
        "red_line_violations": violations,
        "generated_at": datetime.now().isoformat(),
    }
    # 逐案置换明细不写入 JSON 主报告（体积大），仅保留汇总
    report["order_permutation"]["per_case"] = [
        {"case_id": x["case_id"], "configs": x["configs"]}
        for x in perm["per_case"][:10]
    ]
    out_path = Path(__file__).resolve().parents[1] / "ddb" / "robustness_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"      报告：{out_path}")

    if violations:
        print("结论：鲁棒性实验预测违例，需检查顺序原语或词典相依结构。")
        return 1
    print(
        "结论：鲁棒性实验通过——定理 2 与定理 1-B 的可证伪预测得到支持；"
        "流式即判顺序敏感（对照侧）、来源噪声非构陷性同样得到支持。"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 —— 管线异常统一归口退出码 2
        print(f"运行异常：{exc}", file=sys.stderr)
        raise
