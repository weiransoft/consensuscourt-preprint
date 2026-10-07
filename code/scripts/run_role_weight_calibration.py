#!/usr/bin/env python3
"""
§6.5 十三角色权重校准实验（论文大纲 §6.5 "13 角色权重校准"，v0.4.13）。

实验目标
========
大纲 §6.5 承诺："单角色权重扰动 ±20%，决策翻转率 → 权重鲁棒性分析"。
本实验在 DDB 案例池上对该承诺做实证：

1. 对论文 13 角色（见 backend/design/paper-appendix-d-role-profiles.md D.1 权重表）
   逐一做单角色权重扰动（×0.8 / ×1.2，其余角色权重不变）；
2. 在扰动后的 ROLE_WEIGHTS 下重跑两套仲裁器（PIAgentArbiter 规则仲裁轨 +
   EloArbiter 对照基线，两者 v0.4.13 起逐键同口径），统计裁决结果
   （支持/有条件支持/驳回）相对基准权重的翻转率；
3. 输出每角色 × 每扰动方向 × 每仲裁器的翻转率与角色案例覆盖率，
   作为论文"权重鲁棒性"主张的实证依据。

科学预期
========
权重设计（附录 D）中相邻角色权重间距 ≥0.05，单角色 ±20% 扰动
（如 0.9→0.72/1.08）只会移动该角色证据在加权均分中的贡献占比，
不会系统性跨越 0.7/0.4 裁决阈值——预期翻转率显著低于随机扰动水平。
若实测翻转率异常高（红线），说明裁决对权重过拟合，
论文的"权重鲁棒性"主张被证伪。

用法：
    backend$ .venv/bin/python tests/scripts/run_role_weight_calibration.py
    backend$ .venv/bin/python tests/scripts/run_role_weight_calibration.py \\
        --cases 120

输出：tests/ddb/role_weight_calibration.json + stdout 摘要表
退出码：0=全部红线满足；1=红线违例；2=运行异常
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# 导入前置：backend 根入 sys.path + 环境变量占位（必须在 import app 之前）
# ---------------------------------------------------------------------------
_BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

os.environ.setdefault("SCIMINER_API_KEY", "role-weight-calib-placeholder")
os.environ.setdefault("DB_PASSWORD", "role-weight-calib-placeholder")

from app.core.virtual_lab.baseline_arbiter import EloArbiter  # noqa: E402
from app.core.virtual_lab.consensus_court import (  # noqa: E402
    EvidenceScorer,
    PIAgentArbiter,
)
from tests.ddb.l1_generator import generate as generate_l1  # noqa: E402
from tests.ddb.l15_generator import generate as generate_l15  # noqa: E402

# 复用 L1/L1.5 证据字典 → Evidence 的统一还原口径（unknown 补占位引用）
from run_ablation_v03 import _to_evidence  # noqa: E402

#: 实验种子（与消融/对照/鲁棒性/优势分解管线一致）
SEED = 42

#: 论文 13 角色（大纲 L137，与附录 D D.1 权重表一一对应）
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

#: 扰动方向（大纲 §6.5 口径：±20%）
PERTURB_FACTORS: dict[str, float] = {"down_0.8": 0.8, "up_1.2": 1.2}

#: 每角色最小案例覆盖数：被扰动角色的证据必须出现在足够多的案件中，
#: 否则翻转率分母虽大但"可翻转面"为空，校准无意义。
MIN_ROLE_COVERAGE = 5

#: 翻转率红线：单角色 ±20% 扰动下的决策翻转率上限。
#: 依据（v0.4.13 实测标定，工厂构造器注入修复后）：规则式仲裁确定性可复现，
#: 实测 13 角色 ×2 方向 ×2 仲裁器共 52 个翻转率中最大值为 4/208≈0.0192
#: （data-specialist@×0.8，翻转仅出现在 ×0.8 降权方向）；留出容差至 5%
#: 吸收未来权重微调的波动，超过即视为"裁决对权重过拟合"，触发红线排查。
MAX_FLIP_RATE = 0.05

#: 阳性对照（R0）：用 ×0.1 的极端扰动验证翻转检测管线本身有效——
#: 若极端扰动下仍 0 翻转，说明实验空转（扰动未真正注入仲裁），
#: 此时"±20% 翻转率 0"是假阴性而非鲁棒性证据。
#: 实测依据（v0.4.13，回归口径 208 例）：pi-agent/drug-researcher/report-expert
#: 任一角色 ×0.1 扰动下分别翻转 7/6/6 案。
POSITIVE_CONTROL_FACTOR = 0.1
MIN_POSITIVE_CONTROL_FLIPS = 1


def build_case_pool(cases_l1: int) -> list[dict]:
    """构造校准实验的 DDB 案例池（L1 双极性 + L1.5 负例对照）。

    案例池口径与 §6.2 对照基线 / §5.3 优势分解管线一致（同种子同规模），
    保证校准结论与其他实验落在同一底空间上。

    Args:
        cases_l1: L1 每极性案件数（正例/负例各该值）

    Returns:
        案例字典列表（每例含 topic / evidences / ground_truth）
    """
    pool: list[dict] = []
    # L1：有真矛盾（正例）与无矛盾（负例）双极性
    pool.extend(
        generate_l1(
            num_with_contradiction=cases_l1,
            num_no_contradiction=cases_l1,
            seed=SEED,
        )
    )
    # L1.5：三类伪冲突负例（per_template=8 与对照管线一致）
    pool.extend(generate_l15(per_template=8, seed=SEED))
    return pool


def prepare_arbitration_inputs(case: dict) -> tuple[list, dict]:
    """把案例字典还原为仲裁器输入（证据评分列表 + evidence_id 映射）。

    Args:
        case: L1/L1.5 案例字典

    Returns:
        (scores, emap)：EvidenceScore 列表与 evidence_id -> Evidence 映射
    """
    evidences = [_to_evidence(ev) for ev in case["evidences"]]
    scorer = EvidenceScorer()
    scores = [scorer.score(ev, case["topic"]) for ev in evidences]
    emap = {ev.evidence_id: ev for ev in evidences}
    return scores, emap


def measure_role_coverage(pool: list[dict]) -> dict[str, int]:
    """统计每个论文角色在案例池中出现的案件数（角色覆盖率）。

    覆盖率是翻转率可解释性的前提：某角色覆盖为 0 时其扰动必然翻转率 0，
    属"无可翻转面"而非"鲁棒"，必须单独标记。

    Args:
        pool: 案例池

    Returns:
        {role_key: 出现该角色证据的案件数}
    """
    coverage = {role: 0 for role in PAPER_13_ROLES}
    for case in pool:
        roles_in_case = {
            ev.get("role", "").lower() for ev in case["evidences"]
        }
        for role in PAPER_13_ROLES:
            if role in roles_in_case:
                coverage[role] += 1
    return coverage


def count_flips(
    prepared: list[tuple[list, dict]],
    baseline_decisions: list[str],
    weights: dict[str, float],
    make_factory,
) -> int:
    """在给定权重下重跑全部案件并统计与基准裁决不同的案件数。

    Args:
        prepared: prepare_arbitration_inputs 的输出列表（评分+映射缓存）
        baseline_decisions: 基准权重下的逐案裁决字符串
        weights: 待测权重字典（本函数内用其构造仲裁实例，确保注入生效）
        make_factory: 工厂构造器——接收权重字典、返回已注入该权重的
            仲裁实例（make_pi_factory / make_elo_factory），保证扰动
            权重真正进入仲裁

    Returns:
        裁决翻转案件数
    """
    # 关键：必须用本次传入的 weights 构造仲裁实例——
    # 若复用绑定基准权重的外部工厂，扰动会被静默丢弃（实验空转）
    arbiter = make_factory(weights)
    return sum(
        1
        for (scores, emap), base_decision in zip(prepared, baseline_decisions)
        if arbiter.arbitrate("", scores, emap)[0] != base_decision
    )


def run_perturbation(
    pool: list[dict],
    base_weights: dict[str, float],
    make_factory,
) -> tuple[dict[str, dict[str, dict]], list[tuple[list, dict]], list[str]]:
    """执行"基准 → 逐角色扰动 → 翻转计数"的完整校准流程。

    流程：
    1. 以 base_weights 计算每案的基准裁决（decision 字符串）；
    2. 对 13 角色逐一、每方向构造扰动权重副本（仅乘该角色权重，
       其余角色不动，未登记角色不受影响）；
    3. 在扰动权重下重跑全部案件，统计与基准裁决不同的案件数
       （decision 字符串严格相等比较：支持/有条件支持/驳回/无法裁决）。

    Args:
        pool: 案例池
        base_weights: 基准角色权重（PIAgentArbiter.ROLE_WEIGHTS）
        make_factory: 工厂构造器——接收权重字典、返回已注入该权重的
            仲裁实例（make_pi_factory / make_elo_factory），保证每个
            扰动权重都真实注入对应仲裁实例

    Returns:
        (results, prepared, baseline_decisions)：
        {role: {direction: {"flip_rate", "flips", "n"}}}
        以及缓存的仲裁输入/基准裁决（供阳性对照复用）
    """
    # 第 1 步：基准裁决（缓存输入与基准 decision，避免重复评分）
    prepared = [prepare_arbitration_inputs(case) for case in pool]
    base_arbiter = make_factory(base_weights)
    baseline_decisions = [
        base_arbiter.arbitrate("", scores, emap)[0]
        for scores, emap in prepared
    ]

    results: dict[str, dict[str, dict]] = {}
    n = len(pool)
    for role in PAPER_13_ROLES:
        results[role] = {}
        for direction, factor in PERTURB_FACTORS.items():
            # 仅扰动该角色权重，其余键值保持不变（单角色扰动口径）
            perturbed = dict(base_weights)
            perturbed[role] = base_weights[role] * factor
            flips = count_flips(
                prepared, baseline_decisions, perturbed, make_factory
            )
            results[role][direction] = {
                "flips": flips,
                "n": n,
                "flip_rate": round(flips / n, 6) if n else 0.0,
            }
    return results, prepared, baseline_decisions


def make_pi_factory(weights: dict[str, float]) -> PIAgentArbiter:
    """构造使用给定权重的 PIAgentArbiter 仲裁实例（工厂构造器）。

    PIAgentArbiter.arbitrate 读取 self.ROLE_WEIGHTS 实例属性，
    因此通过实例属性覆盖实现权重注入（类属性保持基准不动）。
    语义约定：接收权重字典、直接返回仲裁实例——count_flips /
    run_perturbation 内部以 ``arbiter = make_factory(weights)`` 一步
    取得可用仲裁器，保证每次扰动权重都被真实注入（防空转）。

    Args:
        weights: 注入的角色权重字典

    Returns:
        已注入该权重的 PIAgentArbiter 实例
    """
    arbiter = PIAgentArbiter()
    # 实例属性覆盖：仅影响本实验的仲裁实例，不改类属性基准
    arbiter.ROLE_WEIGHTS = dict(weights)
    return arbiter


def make_elo_factory(weights: dict[str, float]) -> EloArbiter:
    """构造使用给定权重的 EloArbiter 仲裁实例（与 make_pi_factory 同理）。

    Args:
        weights: 注入的角色权重字典

    Returns:
        已注入该权重的 EloArbiter 实例
    """
    arbiter = EloArbiter()
    arbiter.ROLE_WEIGHTS = dict(weights)
    return arbiter


def check_red_lines(
    pi_results: dict,
    elo_results: dict,
    coverage: dict[str, int],
    positive_control_flips: dict[str, int],
) -> list[str]:
    """§6.5 权重校准实验的红线检查。

    红线清单：
    - R0 阳性对照：极端扰动（×0.1）下每个对照角色必须产生 ≥1 次翻转，
      否则翻转检测管线空转，"±20% 翻转率 0"是假阴性而非鲁棒性证据；
    - R1 覆盖率：每个论文角色的证据至少出现在 MIN_ROLE_COVERAGE 个案件中，
      否则该角色的翻转率不具统计意义（无可翻转面）；
    - R2 翻转率上限：所有 角色×方向×仲裁器 组合的翻转率 ≤ MAX_FLIP_RATE，
      超限即"裁决对权重过拟合"，权重鲁棒性主张被证伪。

    Args:
        pi_results: PIAgentArbiter 扰动结果
        elo_results: EloArbiter 扰动结果
        coverage: 角色覆盖率
        positive_control_flips: {对照角色: ×0.1 扰动下的翻转数}

    Returns:
        违例描述列表（空 = 全部满足）
    """
    violations: list[str] = []

    # R0：阳性对照（翻转检测管线有效性）
    for role, flips in positive_control_flips.items():
        if flips < MIN_POSITIVE_CONTROL_FLIPS:
            violations.append(
                f"阳性对照失败：{role}@×{POSITIVE_CONTROL_FACTOR} 极端扰动下 "
                f"翻转数 {flips} < {MIN_POSITIVE_CONTROL_FLIPS}"
                "（翻转检测管线空转，结果无效）"
            )

    # R1：角色覆盖率下限（无可翻转面的角色校准无意义）
    for role, cnt in coverage.items():
        if cnt < MIN_ROLE_COVERAGE:
            violations.append(
                f"角色覆盖不足：{role} 仅出现在 {cnt} 案 "
                f"< {MIN_ROLE_COVERAGE}（翻转率不具统计意义，请增大 --cases）"
            )

    # R2：翻转率上限（权重鲁棒性）
    for arbiter_name, results in (
        ("PIAgentArbiter", pi_results),
        ("EloArbiter", elo_results),
    ):
        for role, by_dir in results.items():
            for direction, stat in by_dir.items():
                if stat["flip_rate"] > MAX_FLIP_RATE:
                    violations.append(
                        f"翻转率超限（{arbiter_name}）：{role}@{direction} "
                        f"flip_rate={stat['flip_rate']:.4f} "
                        f"> {MAX_FLIP_RATE}（裁决对权重过拟合）"
                    )
    return violations


def main() -> int:
    """命令行入口：构造案例池 → 双仲裁器扰动校准 → 红线 → 落盘。"""
    parser = argparse.ArgumentParser(
        description="§6.5 十三角色权重校准（单角色 ±20% 扰动决策翻转率）"
    )
    parser.add_argument(
        "--cases", type=int, default=120,
        help="L1 每极性案件数（默认 120，正例/负例各该值；另加 L1.5 88 例）",
    )
    args = parser.parse_args()

    print("[1/4] 构造 DDB 案例池（L1 双极性 + L1.5 负例对照）...")
    pool = build_case_pool(args.cases)
    print(f"      案例池 {len(pool)} 例（种子 {SEED}）")

    print("[2/4] 角色覆盖率统计...")
    coverage = measure_role_coverage(pool)
    print(
        "      "
        + ", ".join(f"{r}={c}" for r, c in coverage.items())
    )

    print("[3/4] 单角色 ±20% 扰动校准（PIAgentArbiter + EloArbiter 同口径）...")
    base_weights = dict(PIAgentArbiter.ROLE_WEIGHTS)
    # 同口径断言前置：EloArbiter 权重与 PI 逐键同值（v0.4.13 契约），
    # 不一致则实验底座失效，直接报错而非带病出报告。
    if EloArbiter.ROLE_WEIGHTS != base_weights:
        print("      [错误] EloArbiter.ROLE_WEIGHTS 与 PIAgentArbiter 不同口径！")
        return 1
    pi_results, prepared, baseline_decisions = run_perturbation(
        pool, base_weights, make_pi_factory
    )
    elo_results, _, _ = run_perturbation(
        pool, base_weights, make_elo_factory
    )
    max_pi = max(
        s["flip_rate"] for by_dir in pi_results.values() for s in by_dir.values()
    )
    max_elo = max(
        s["flip_rate"] for by_dir in elo_results.values() for s in by_dir.values()
    )
    print(f"      PIAgentArbiter 最大翻转率 = {max_pi:.4f}")
    print(f"      EloArbiter     最大翻转率 = {max_elo:.4f}")

    # 阳性对照：极端扰动（×0.1）下必须产生翻转，证明检测管线有效
    positive_control_flips: dict[str, int] = {}
    for role in ("pi-agent", "drug-researcher", "report-expert"):
        control_weights = dict(base_weights)
        control_weights[role] = base_weights[role] * POSITIVE_CONTROL_FACTOR
        flips = count_flips(
            prepared,
            baseline_decisions,
            control_weights,
            make_pi_factory,
        )
        positive_control_flips[role] = flips
        print(f"      阳性对照 {role}@×{POSITIVE_CONTROL_FACTOR} -> 翻转 {flips}")

    print("[4/4] 红线检查 + 写报告...")
    violations = check_red_lines(
        pi_results, elo_results, coverage, positive_control_flips
    )
    for v in violations:
        print(f"      [违例] {v}")
    if not violations:
        print(
            f"      全部红线满足：13 角色覆盖 ≥{MIN_ROLE_COVERAGE} 案，"
            f"翻转率 ≤{MAX_FLIP_RATE:.0%}（权重鲁棒性成立）。"
        )

    report = {
        "experiment": "§6.5 十三角色权重校准（单角色 ±20% 扰动决策翻转率）",
        "benchmark": "DDB-L1（合成争议双极性）+ DDB-L1.5（伪冲突负例对照）",
        "seed": SEED,
        "paper_13_roles": PAPER_13_ROLES,
        "perturb_factors": PERTURB_FACTORS,
        "n_cases": len(pool),
        "role_coverage": coverage,
        "base_weights": base_weights,
        "pi_agent_arbiter": pi_results,
        "elo_arbiter": elo_results,
        "max_flip_rate_observed": {
            "pi_agent_arbiter": max_pi,
            "elo_arbiter": max_elo,
        },
        "red_line_thresholds": {
            "min_role_coverage": MIN_ROLE_COVERAGE,
            "max_flip_rate": MAX_FLIP_RATE,
            "positive_control_factor": POSITIVE_CONTROL_FACTOR,
            "min_positive_control_flips": MIN_POSITIVE_CONTROL_FLIPS,
        },
        "positive_control_flips": positive_control_flips,
        "red_line_violations": violations,
        "generated_at": datetime.now().isoformat(),
    }
    out_path = Path(__file__).resolve().parents[1] / "ddb" / (
        "role_weight_calibration.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"      报告：{out_path}")

    if violations:
        print("结论：权重校准实验红线违例，需按违例项排查。")
        return 1
    print("结论：§6.5 权重校准实验通过——单角色 ±20% 扰动决策翻转率受控。")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 —— 管线异常统一归口退出码 2
        print(f"运行异常：{exc}", file=sys.stderr)
        raise
