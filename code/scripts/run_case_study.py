#!/usr/bin/env python3
"""
端到端案例研究数据源脚本（v0.4.15，大纲 §6.5"端到端案例研究"落地）。

设计（诚实边界，预印本主张严谨性）：
    - 案例来自 DDB-L2 文献真实争议层（tests/ddb/l2_cases.py，引用已对
      NCBI esummary 核验），**不是**平台生产运行——大纲 §6.5 的
      "分歧产生 → 群聊 5 轮僵局 → 开庭 → …" 全程叙事中，群聊环节属
      平台生产运行数据，本脚本如实以"开庭流水线时间线"替代并在报告
      honesty_note 中声明；平台真实运行案例（ASFV/犬源疫苗管线等）
      留给 L3 层（论文 limitation 如实声明）。
    - 走生产 run_court_headless 完整管线（hold_court 组件编排的实验
      镜像，parity 由 test_court_ablation_pipeline.py 锁死），产出
      时间线（初筛→评分→ABP→PI→评审→BOSS→升级判定）、各环节发言/
      理由、ABP 收敛点（定罪矛盾对）、裁决链与证据引用记录。

精选案例（僵局型 + 升级型各一，选择理由写入报告可审计）：
    - DDB-L2-0001（EGFR 三重突变抑制剂设计，僵局型）：A/B 双方均有
      顶级期刊证据对峙（Nature/Cancer Lett 的变构路线 vs Nature/NEJM
      的 ATP 共价路线），无形式缺陷（n_filtered=0），矛盾只能靠证据
      交叉裁决——对应"群聊多轮僵局后升庭"的典型形态；
    - DDB-L2-0005（AlphaFold3 蛋白-配体预测可信度，升级型）：全部
      证据命中结构预测工具禁令（v0.4.8 双禁令）被对称初筛过滤 →
      法庭无可裁证据、弃权（decision=无法裁决，confidence=0）→
      整案升级人工复核——对应"升级条款兜底"的典型形态。

用法：
    backend$ .venv/bin/python tests/scripts/run_case_study.py

输出：tests/ddb/case_study_report.json + stdout 摘要
退出码：0=成功且双跑确定性一致；1=裁决与 L2 锁不一致；2=运行异常
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# 导入前置：backend 根入 sys.path + 环境变量占位（零 LLM / 零网络，占位即可）
# ---------------------------------------------------------------------------
_BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

os.environ.setdefault("SCIMINER_API_KEY", "case-study-placeholder")
os.environ.setdefault("DB_PASSWORD", "case-study-placeholder")

from run_ablation_v03 import run_court_headless  # noqa: E402
from tests.ddb.l2_cases import load_cases  # noqa: E402

#: 实验种子（run_court_headless 为纯确定性组件编排，种子仅作报告溯源留痕）
SEED = 42

#: 精选案例表：case_id → (研究类型, 选择理由)——理由写入报告供审阅者审计
SELECTED_CASES: dict[str, dict[str, str]] = {
    "DDB-L2-0001": {
        "study_role": "deadlock（僵局型）",
        "selection_rationale": (
            "A/B 双方均有顶级期刊证据对峙（变构路线：Nature 2016/Cancer Lett 2017；"
            "ATP 共价路线：NEJM FLAURA/OS），无形式缺陷（初筛 0 过滤），"
            "矛盾只能靠证据交叉裁决——对应大纲 §6.5'群聊多轮僵局后升庭'的典型形态"
        ),
    },
    "DDB-L2-0005": {
        "study_role": "escalation（升级型）",
        "selection_rationale": (
            "全部证据命中结构预测工具禁令（v0.4.8 AlphaFold2/3 双禁令）被对称初筛"
            "过滤 → 法庭无可裁证据、弃权（decision=无法裁决，confidence=0）→ "
            "整案升级人工复核——对应'升级条款兜底不可裁案件'的典型形态"
        ),
    },
}

#: 案例研究使用的法院模式（完整法庭 = 全组件参照）
COURT_MODE = "court_full"

#: 时间线各环节的人读名称（与 run_court_headless 编排步骤一一对应）
_TIMELINE_STEPS = [
    ("step1_legacy_screen", "对称初筛（RuleEngine.screen）"),
    ("step2_evidence_scoring", "证据评分（EvidenceScorer，0.4/0.2/0.4）"),
    ("step3_abp_rubric", "ABP 矛盾对评估（AsymmetricRubric 全开）"),
    ("step4_pi_arbitration", "PI Agent 仲裁（PIAgentArbiter）"),
    ("step5_independent_review", "独立评审（IndependentReviewer）"),
    ("step6_boss_override", "Human BOSS 覆盖清单（HumanBOSSOverride）"),
    ("step7_escalation", "升级判定（items or conf<0.6 or intensity>=3）"),
]


def build_case_study_entry(case: dict) -> dict:
    """
    对单个精选案例运行完整法庭流水线并组装案例研究条目。

    结构（大纲 §6.5 案例研究四要素的 DDB 落地）：
    - timeline：七个环节的顺序时间线（每环节 = 环节名 + 关键输出）；
    - abp_convergence：ABP 收敛点 = 定罪矛盾对全记录（含金标对命中标记）；
    - verdict_chain：裁决链（rubric→PI→评审→BOSS→升级，全留痕）；
    - evidence_scores / filtered_audit：证据评分明细与初筛过滤留痕；
    - gold_pair_detection：金标矛盾对是否被关键词通道检出（与 H2b 锁对齐）。

    Args:
        case: L2 案例字典（load_cases 深拷贝）

    Returns:
        案例研究条目字典（确定性，双跑一致）
    """
    out = run_court_headless(case, COURT_MODE, detail=True)

    # 金标对命中检测：与 test_l2_cases 的 frozenset 无序对口径一致
    detected_pairs = {
        frozenset((c["evidence_a_id"], c["evidence_b_id"]))
        for c in out["contradictions"]
    }
    gold_pairs = {
        frozenset((a, b))
        for a, b in case["ground_truth"]["contradiction_pairs"]
    }
    gold_hits = sorted("/".join(sorted(p)) for p in detected_pairs & gold_pairs)

    # 时间线：七环节顺序组装（每环节只留该环节的真实输出，无叙事加工）
    timeline = [
        {
            "step": name,
            "component": label,
            "output": {
                "step1_legacy_screen": {
                    "n_filtered": out["n_filtered"],
                    "filtered_audit": out["filtered_audit"],
                },
                "step2_evidence_scoring": {
                    "scored_evidence": [
                        s["evidence_id"] for s in out["evidence_scores"]
                    ]
                },
                "step3_abp_rubric": {
                    "convicted_by_abp": bool(out["contradictions"]),
                    "max_intensity": out["max_intensity"],
                },
                "step4_pi_arbitration": {
                    "decision": out["decision"],
                    "confidence": out["confidence"],
                },
                "step5_independent_review": {
                    "reviewer_notes": out["reviewer_notes"],
                },
                "step6_boss_override": {
                    "override_items": out["override_items"],
                },
                "step7_escalation": {
                    "escalated": out["escalated"],
                    "reasons": [
                        r for r, hit in (
                            ("BOSS 覆盖清单非空", bool(out["override_items"])),
                            ("置信度 < 0.6", out["confidence"] < 0.6),
                            ("矛盾峰值强度 ≥ 3", out["max_intensity"] >= 3),
                        ) if hit
                    ],
                },
            }[name],
        }
        for name, label in _TIMELINE_STEPS
    ]

    meta = SELECTED_CASES[case["case_id"]]
    return {
        "case_id": case["case_id"],
        "study_role": meta["study_role"],
        "selection_rationale": meta["selection_rationale"],
        "topic": case["topic"],
        "category": case["category"],
        "dispute_sides": case["dispute_sides"],
        "benchmark_source": "DDB-L2（tests/ddb/l2_cases.py，文献真实争议层）",
        "timeline": timeline,
        "abp_convergence": {
            "contradictions": out["contradictions"],
            "max_intensity": out["max_intensity"],
        },
        "verdict_chain": {
            "court_mode": COURT_MODE,
            "convicted": out["convicted"],
            "decision": out["decision"],
            "confidence": out["confidence"],
            "pi_rationale": out["pi_rationale"],
            "reviewer_notes": out["reviewer_notes"],
            "override_items": out["override_items"],
            "escalated": out["escalated"],
        },
        "evidence_scores": out["evidence_scores"],
        "filtered_audit": out["filtered_audit"],
        "gold_pair_detection": {
            "gold_pairs": [sorted(p) for p in sorted(gold_pairs, key=sorted)],
            "detected_gold_pairs": gold_hits,
            "gold_pair_detected": bool(gold_hits),
        },
    }


def build_report() -> dict:
    """
    构建完整案例研究报告（确定性：除 generated_at 外双跑逐字段一致）。

    Returns:
        报告字典（honesty_note / selected_cases / case_studies / seed）
    """
    cases = {c["case_id"]: c for c in load_cases()}
    missing = sorted(set(SELECTED_CASES) - set(cases))
    assert not missing, f"精选案例在 L2 基准中不存在: {missing}"

    entries = [build_case_study_entry(cases[cid]) for cid in sorted(SELECTED_CASES)]
    return {
        "experiment": "端到端案例研究（大纲 §6.5，DDB-L2 落地版）",
        "honesty_note": (
            "案例来自 DDB-L2 文献真实争议层（tests/ddb/l2_cases.py，PMID 已对 "
            "NCBI esummary 核验），非平台生产运行；大纲 §6.5 的'群聊 5 轮僵局'"
            "环节属平台生产运行叙事，本报告以开庭流水线七环节时间线如实替代，"
            "平台真实运行案例（ASFV/犬源疫苗管线等）留待 L3 层（论文 limitation "
            "如实声明）。"
        ),
        "court_mode": COURT_MODE,
        "seed": SEED,
        "selected_cases": [
            {"case_id": cid, **meta} for cid, meta in sorted(SELECTED_CASES.items())
        ],
        "case_studies": entries,
        "generated_at": datetime.now().isoformat(),
    }


def main() -> int:
    """主流程：构建报告 → 裁决一致性自检 → 落盘 → 双跑确定性校验。"""
    print("[1/3] 运行精选案例的完整法庭流水线（run_court_headless, court_full）...")
    report = build_report()
    for entry in report["case_studies"]:
        vc = entry["verdict_chain"]
        gp = entry["gold_pair_detection"]
        print(
            f"      {entry['case_id']} [{entry['study_role']}]  "
            f"convicted={vc['convicted']}  decision={vc['decision']}  "
            f"confidence={vc['confidence']}  escalated={vc['escalated']}  "
            f"金标对命中={gp['detected_gold_pairs'] or '无'}"
        )

    # 裁决一致性自检：与既有 L2 锁（H2b）对齐——零过滤案例的定罪结果必须
    # 与 rubric 直评一致（test_l2_cases.test_abp_rejects_all_positive_cases：
    # 10/10 正例 rubric 直评全部定罪）；本报告两个精选案例均零过滤
    # （L2-0001）或全过滤（L2-0005），后者定罪=弃权升级，属生产初筛真实行为，
    # 两类都必须 escalated（升级层不放过任何正例）
    print("[2/3] 裁决一致性自检（对齐 L2 H2b 锁）...")
    for entry in report["case_studies"]:
        vc = entry["verdict_chain"]
        if entry["filtered_audit"]:
            # 全过滤案例：法庭弃权 → 必须升级人工（不得静默放行正例）。
            # 金标对命中检查不适用：证据未进入 ABP 通道（对称初筛先行过滤
            # 是生产真实行为，H2b 锁的换源召回在 rubric 直评口径达成，
            # 该口径差异本身即案例研究的发现之一，报告中如实留痕）
            if not vc["escalated"]:
                print(
                    f"      [违例] {entry['case_id']} 全证据被初筛过滤但未升级"
                    "（正例不得静默放行）"
                )
                return 1
        else:
            # 零过滤案例：定罪结果必须与 rubric 直评（H2b 锁：全部定罪）一致，
            # 且金标矛盾对必须被关键词通道检出
            if not vc["convicted"]:
                print(
                    f"      [违例] {entry['case_id']} 零过滤但未定罪，"
                    "与 L2 H2b 锁（正例全部定罪）不一致"
                )
                return 1
            if not entry["gold_pair_detection"]["gold_pair_detected"]:
                print(
                    f"      [违例] {entry['case_id']} 金标矛盾对未被关键词通道检出"
                    "（与 H2b 换源召回锁不一致）"
                )
                return 1
    print("      一致性自检通过（定罪/升级/金标对命中与 L2 锁对齐）")

    # 双跑确定性：报告重建一次，除 generated_at 外逐字段一致
    print("[3/3] 双跑确定性校验...")
    second = build_report()
    first_no_ts = {k: v for k, v in report.items() if k != "generated_at"}
    second_no_ts = {k: v for k, v in second.items() if k != "generated_at"}
    if first_no_ts != second_no_ts:
        print("      [违例] 双跑报告不一致（确定性破坏）")
        return 1
    print("      双跑逐字段一致（除 generated_at）")

    out_path = Path(__file__).resolve().parents[1] / "ddb" / "case_study_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"      报告：{out_path}")
    print("结论：案例研究数据源产出成功（结构完整 + 确定性 + L2 锁对齐）。")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 —— 管线异常统一归口非零退出
        print(f"运行异常：{exc}", file=sys.stderr)
        raise
