#!/usr/bin/env python3
"""
实验 B：LLM-debate 头对头（论文 §8 登记项 2，PROClaim/DebateCV 式，v0.4.17）。

语义
====
每案 3 次托管 qwen-flash 调用，复刻 PROClaim/DebateCV 式"控辩裁"辩论
协议（这是社区 LLM 科学裁决的代表性协议，用作与本法庭的头对头对照）：
① prosecution（控方）：只给证据池，论证"证据池含真正矛盾，提案不成立"；
② skeptic（辩方）：同一证据池，论证"证据一致无矛盾"；
③ judge（裁判）：输入证据清单 + 双方陈词 → 三档裁决 JSON
   {"verdict":"support|conditional|reject","confidence":0-1,"rationale":"<=80 words"}。
温度 0；裁判输出决定案件预测（与实验 A 同口径映射）。
法庭输出冻结（baseline_comparison_report.json，不重跑）。

目的：回答"多角色辩论协议能否替代规则核达到法庭同级 (recall, FPR)"。
结果无论方向都如实写入论文（预注册无结果红线）。

两阶段（防重跑烧钱，与实验 A 同构）
====================================
- Phase collect：异步并发 4、单调用重试 3（指数退避）、逐调用追加
  tests/ddb/llm_debate_responses.jsonl（key = case_id#role），断点续跑；
  控辩任一失败则该案跳过裁判调用（诚实记 call_failure，不伪造陈词）；
- Phase analyze：离线解析 → 两口径混淆矩阵（按裁判裁决）→ 指标 +
  Wilson CI → 与冻结法庭参考对比 → 报告 tests/ddb/llm_debate_report.json。

预注册结构红线（不设结果红线）
==============================
- RL1：208 案全部有完整控-辩-裁三元组（裁判调用成功）；
- RL2：裁判裁决要么在枚举内、要么计入诚实字段 parse_failures
  （保守处理与实验 A 逐字一致）；
- RL3：冻结法庭参考与预注册常量逐字段一致（互锁不漂移）。

用法：
    backend$ ../.venv-bio/bin/python tests/scripts/run_llm_debate_experiment.py
    backend$ ../.venv-bio/bin/python tests/scripts/run_llm_debate_experiment.py --phase analyze

配置：LLM_EXPERIMENT_BASE_URL / _API_KEY / _MODEL（backend/.env 或环境变量）；
缺配置 exit 2，不静默降级为模拟。

输出：tests/ddb/llm_debate_responses.jsonl + tests/ddb/llm_debate_report.json
退出码：0=红线全部满足；1=红线违例；2=运行异常/配置缺失
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from llm_experiment_lib import (  # noqa: E402
    CASES_PER_POLARITY_L1,
    SEED,
    CallStats,
    ConfigError,
    VERDICT_ENUM,
    VerdictParseError,
    append_jsonl,
    build_case_pool,
    chat_with_retry,
    check_frozen_against_canonical,
    check_frozen_reference_integrity,
    load_experiment_config,
    load_frozen_court_reference,
    load_jsonl_all,
    load_jsonl_latest,
    parse_verdict_json,
    prompt_sha256,
    render_evidence_pool,
    tally_case,
    utc_now_iso,
    _empty_confusion,
    _rates,
)

#: 响应与报告落盘路径
_RESPONSES_PATH = Path(__file__).resolve().parents[1] / "ddb" / "llm_debate_responses.jsonl"
_REPORT_PATH = Path(__file__).resolve().parents[1] / "ddb" / "llm_debate_report.json"

#: 三个辩论角色（每案三次调用的顺序编排：控 → 辩 → 裁）
ROLES: tuple[str, ...] = ("prosecution", "skeptic", "judge")

#: 控方系统提示词（只给证据池，立场给定——辩论协议的核心；不泄漏真值）
PROSECUTION_SYSTEM = (
    "你是科学证据辩论中的控方。给定一个争议议题和一组证据，"
    "你的任务是尽最大努力论证：证据池中存在真正的矛盾，"
    "该议题下的研究提案不成立。你必须只依据给定的证据内容论证，"
    "不得臆造证据之外的实验结果。\n"
    "只输出你的辩论陈词（中文，不超过120词），不要输出 JSON 或其它格式。"
)

#: 辩方系统提示词（与控方对称的对抗立场；同一证据池）
SKEPTIC_SYSTEM = (
    "你是科学证据辩论中的辩方。给定一个争议议题和一组证据，"
    "你的任务是尽最大努力论证：证据相互一致、不存在真正矛盾，"
    "该议题下的研究提案可以成立。你必须只依据给定的证据内容论证，"
    "不得臆造证据之外的实验结果。\n"
    "只输出你的辩论陈词（中文，不超过120词），不要输出 JSON 或其它格式。"
)

#: 裁判系统提示词（输入含双方陈词；输出 schema 与实验 A 完全一致）
JUDGE_SYSTEM = (
    "你是科学证据辩论中的独立裁判。给定一个争议议题、一组证据"
    "以及控辩双方的辩论陈词，你需要独立裁决该议题下的研究提案：\n"
    "- support：证据相互一致，足以支持提案；\n"
    "- conditional：证据大体一致但存在局限或缺口，只能有条件支持；\n"
    "- reject：证据之间存在实质矛盾或不足以支撑提案，应予驳回。\n"
    "你必须独立依据给定的证据与陈词判断，不要臆造证据之外的实验结果。\n"
    '只输出一个 JSON 对象，格式：{"verdict":"support|conditional|reject",'
    '"confidence":0到1的小数,"rationale":"不超过80词的中文理由"}，'
    "不要输出任何其它文字。"
)


def _evidence_block(case: dict) -> str:
    """
    渲染议题 + 盲评证据池（三个角色共用；无真值/评分/metadata 泄漏）。

    Args:
        case: 案例字典

    Returns:
        议题与证据池文本块
    """
    return (
        f"争议议题：{case['topic']}\n\n"
        f"证据池（共 {len(case['evidences'])} 条）：\n\n"
        f"{render_evidence_pool(case)}"
    )


def build_messages(role: str, case: dict, statements: dict[str, str]) -> list[dict]:
    """
    构建指定角色的 chat 消息（三次调用的编排原语）。

    Args:
        role: prosecution / skeptic / judge
        case: 案例字典
        statements: 已取得的陈词 {role: text}（judge 需要控辩双方）

    Returns:
        OpenAI chat messages 列表
    """
    if role in ("prosecution", "skeptic"):
        system = PROSECUTION_SYSTEM if role == "prosecution" else SKEPTIC_SYSTEM
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": _evidence_block(case)},
        ]
    # judge：证据清单 + 双方陈词
    user = (
        f"{_evidence_block(case)}\n\n"
        f"控方（prosecution）陈词：\n{statements.get('prosecution', '')}\n\n"
        f"辩方（skeptic）陈词：\n{statements.get('skeptic', '')}\n\n"
        "请按系统指令输出裁决 JSON。"
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]


async def phase_collect(
    cases: list[dict],
    concurrency: int,
    max_retries: int,
    client: Any = None,
    model: str | None = None,
) -> CallStats:
    """
    Phase collect：每案控→辩→裁三次调用，逐调用增量落盘（断点续跑）。

    断点语义按调用粒度：已成功的调用（key=case_id#role）跳过；
    控辩任一调用最终失败则该案跳过裁判（诚实记案级 call_failure，
    不用占位陈词伪造裁判输入）。

    Args:
        cases: 案例池（208 例）
        concurrency: 并发上限（计划规定 4）
        max_retries: 单调用最大重试次数（计划规定 3）
        client: 预构建客户端（离线单测注入；None 时按配置真实构建）
        model: 模型 ID（client 注入时必须给定）

    Returns:
        本轮 collect 的调用量统计

    Raises:
        ConfigError: 配置缺失（由 main 转 exit 2）
        ValueError: 注入 client 时未同步给定 model
    """
    from openai import AsyncOpenAI

    owns_client = client is None
    if owns_client:
        cfg = load_experiment_config()
        # max_retries=0：关闭 openai SDK 内建隐藏重试（默认 2 次），
        # 避免与脚本层重试叠加导致 n_calls/n_retries/tokens 诚实统计
        # 被低估（架构师审查必须修复项 1，与实验 A 一致）
        client = AsyncOpenAI(
            base_url=cfg.base_url, api_key=cfg.api_key, max_retries=0
        )
        model = cfg.model
    if not model:
        raise ValueError("注入 client 时必须同步给定 model（留痕一致性）")
    stats = CallStats()
    semaphore = asyncio.Semaphore(concurrency)

    done = load_jsonl_latest(_RESPONSES_PATH)

    def _ok(key: str, fingerprint: str) -> bool:
        """
        该调用是否可安全跳过（断点续跑判据，架构师审查必须修复项 2）。

        必须同时满足：调用成功且解析合法（裁判解析失败案续跑补调）；
        model 一致；prompt 指纹一致——换模型或 prompt 漂移后旧响应
        一律重调，严禁新旧采集条件混池。裁判指纹依赖控辩陈词，由
        run_case 在加载陈词后现场重算比对。
        """
        rec = done.get(key)
        return bool(
            rec
            and rec.get("call_ok")
            and rec.get("parse_ok", False)
            and rec.get("model") == model
            and rec.get("prompt_sha256") == fingerprint
        )

    async def run_role(case: dict, role: str, statements: dict[str, str]) -> dict:
        """单角色调用 → 解析 → 逐调用落盘；返回该调用记录。"""
        key = f"{case['case_id']}#{role}"
        messages = build_messages(role, case, statements)
        fingerprint = prompt_sha256(messages[0]["content"], messages[1]["content"])
        text, err = await chat_with_retry(
            client,
            model,
            messages,
            stats=stats,
            semaphore=semaphore,
            max_retries=max_retries,
            # 控辩陈词与裁判裁决统一 2048 上限（qwen-flash 思考模式
            # 需预留推理 token，见 llm_experiment_lib.chat_with_retry 登记）
            max_tokens=2048,
        )
        record: dict = {
            "key": key,
            "case_id": case["case_id"],
            "role": role,
            "model": model,
            "call_ok": text is not None,
            "call_error": err,
            "raw_response": text,
            "parsed_verdict": None,
            "confidence": None,
            "rationale": None,
            "parse_ok": False,
            "parse_error": None,
            "prompt_sha256": fingerprint,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if text is not None:
            if role == "judge":
                # 裁判输出走三档裁决 JSON 解析（与实验 A 同解析器）
                try:
                    parsed = parse_verdict_json(text)
                    record.update(
                        parsed_verdict=parsed["verdict"],
                        confidence=parsed["confidence"],
                        rationale=parsed["rationale"],
                        parse_ok=True,
                    )
                except VerdictParseError as perr:
                    record["parse_error"] = str(perr)
            else:
                # 控辩陈词为自由文本：非空即解析成功（raw_response 即陈词）
                record.update(parse_ok=True)
        append_jsonl(_RESPONSES_PATH, record)
        return record

    async def run_case(case: dict) -> None:
        """单案编排：控 → 辩 → 裁（后一步依赖前两步陈词）。

        控辩任一调用最终失败即短路返回——不伪造陈词、不调用裁判；
        缺失的裁判记录由 analyze 按案级 call_failure 诚实计数（RL1）。
        续跑跳过前现场重算 prompt 指纹并校验 model 一致（架构师审查
        必须修复项 2；裁判指纹依赖控辩陈词，须在陈词加载后计算）。
        """
        statements: dict[str, str] = {}
        for role in ROLES:
            key = f"{case['case_id']}#{role}"
            messages = build_messages(role, case, statements)
            fingerprint = prompt_sha256(
                messages[0]["content"], messages[1]["content"]
            )
            if _ok(key, fingerprint):
                statements[role] = done[key]["raw_response"]
                continue
            rec = await run_role(case, role, statements)
            if not rec["call_ok"]:
                return  # 该案本轮失败；下次运行断点续跑重试
            statements[role] = rec["raw_response"]

    await asyncio.gather(*(run_case(c) for c in cases))
    if owns_client:
        await client.close()
    print(
        f"[collect] 完成：调用 {stats.n_calls} 次（重试 {stats.n_retries}），"
        f"tokens prompt={stats.total_prompt_tokens} "
        f"completion={stats.total_completion_tokens}"
    )
    return stats


def phase_analyze() -> tuple[dict, list[str]]:
    """
    Phase analyze：离线解析 jsonl → 按裁判裁决计口径 → 报告（不联网）。

    Returns:
        (报告字典, 红线违例列表)
    """
    cases = build_case_pool()
    truth_by_id = {
        c["case_id"]: c["polarity"] == "with_contradiction" for c in cases
    }
    records = load_jsonl_latest(_RESPONSES_PATH)

    # 一致性审计（架构师审查必须修复项 2 的 analyze 侧）：全量历史
    # 记录的 model 唯一性与同 key prompt 指纹唯一性——混用/漂移即违例
    all_records = load_jsonl_all(_RESPONSES_PATH)
    models_present = sorted({r.get("model", "") for r in all_records})
    fp_by_key: dict[str, set] = {}
    for r in all_records:
        fp_by_key.setdefault(r.get("key", ""), set()).add(r.get("prompt_sha256", ""))
    mixed_fp_keys = sorted(k for k, fps in fp_by_key.items() if len(fps) > 1)

    # 两口径混淆矩阵 + 裁决分布 + 诚实计数器；
    # cm_clean_pair 为剔除调用失败/解析失败案后的敏感性矩阵
    #（防御"保守口径推高 FPR"的口径选择性质疑，架构师审查建议 5）
    cm_pair = {"reject": _empty_confusion(), "non_support": _empty_confusion()}
    cm_clean_pair = {"reject": _empty_confusion(), "non_support": _empty_confusion()}
    n_clean = 0
    decision_dist: dict[str, int] = {v: 0 for v in VERDICT_ENUM}
    decision_dist["parse_failure"] = 0
    decision_dist["call_failure"] = 0
    parse_failure_ids: list[str] = []
    call_failure_ids: list[str] = []
    illegal_verdict_ids: list[str] = []
    role_failures: dict[str, int] = {role: 0 for role in ROLES}

    for case_id, has_truth in truth_by_id.items():
        # 角色级失败统计（诚实报告项：控/辩/裁各自失败数）
        for role in ROLES:
            rec = records.get(f"{case_id}#{role}")
            if rec is not None and not rec.get("call_ok"):
                role_failures[role] += 1
        judge_rec = records.get(f"{case_id}#judge")
        if judge_rec is None or not judge_rec.get("call_ok"):
            # RL1 违例载体：无成功裁判响应（含控辩失败导致的跳过）；
            # 保守方向与解析失败统一（reject 未拦下、non_support 拦下，
            # 不给 LLM fail-open 通道——架构师审查建议 1）
            call_failure_ids.append(case_id)
            tally_case(cm_pair, has_truth, False, True)
            decision_dist["call_failure"] += 1
            continue
        if not judge_rec.get("parse_ok"):
            # RL2 诚实通道：裁判解析失败保守处理（与实验 A 逐字一致）
            parse_failure_ids.append(case_id)
            tally_case(cm_pair, has_truth, False, True)
            decision_dist["parse_failure"] += 1
            continue
        # 防篡改复核（架构师审查建议 4）：声称解析成功的裁判记录必须
        # 能用原始输出重解析出一致裁决，否则按解析失败保守计口径并
        # 触发 RL2 违例（防 parsed_verdict 与 raw_response 被单独篡改）
        try:
            reparsed = parse_verdict_json(judge_rec.get("raw_response") or "")
        except VerdictParseError:
            reparsed = None
        verdict = judge_rec.get("parsed_verdict")
        if (
            reparsed is None
            or reparsed["verdict"] != verdict
            or verdict not in VERDICT_ENUM
        ):
            parse_failure_ids.append(case_id)
            tally_case(cm_pair, has_truth, False, True)
            decision_dist["parse_failure"] += 1
            illegal_verdict_ids.append(case_id)
            continue
        decision_dist[verdict] += 1
        if verdict == "reject":
            tally_case(cm_pair, has_truth, True, True)
        elif verdict == "conditional":
            tally_case(cm_pair, has_truth, False, True)
        else:  # support
            tally_case(cm_pair, has_truth, False, False)
        # 干净子集（全部校验通过）同步计入敏感性矩阵
        tally_case(cm_clean_pair, has_truth, verdict == "reject", verdict != "support")
        n_clean += 1

    llm_metrics = {
        "reject": _rates(cm_pair["reject"]),
        "non_support": _rates(cm_pair["non_support"]),
        "decision_distribution": decision_dist,
        "role_call_failures": role_failures,
    }

    frozen = load_frozen_court_reference()
    reference = dict(frozen)
    ref_drift = check_frozen_reference_integrity(reference, frozen)
    ref_drift += check_frozen_against_canonical(frozen)

    deltas = {
        "recall": (
            round(llm_metrics["reject"]["recall"] - frozen["recall"], 4)
            if llm_metrics["reject"]["recall"] is not None
            else None
        ),
        "fpr": (
            round(llm_metrics["reject"]["fpr"] - frozen["fpr"], 4)
            if llm_metrics["reject"]["fpr"] is not None
            else None
        ),
    }

    n_pos = sum(truth_by_id.values())
    clean_metrics = {
        "description": (
            "剔除调用失败与解析失败案后的敏感性分析"
            "（防御保守口径推高 FPR 的口径选择性质疑）"
        ),
        "n_cases": n_clean,
        "reject": _rates(cm_clean_pair["reject"]),
        "non_support": _rates(cm_clean_pair["non_support"]),
    }
    report = {
        "experiment": "§8 登记项 2：LLM-debate 头对头（PROClaim/DebateCV 式控辩裁协议）",
        "benchmark": "DDB-L1（合成争议 60×2）+ DDB-L1.5（负例对照 88 例），seed=42",
        "model": models_present[-1] if models_present else None,
        "models_present": models_present,
        "temperature": 0.0,
        "calls_per_case": 3,
        "seed": SEED,
        "max_tokens": 2048,
        "max_tokens_deviation": (
            "预注册计划 §3 为 300；实测 qwen-flash 经网关默认启用思考模式，"
            "300 上限被推理内容耗尽导致正文恒空，登记偏离为 2048"
        ),
        "cases_per_polarity_l1": CASES_PER_POLARITY_L1,
        "pool_composition": {"n_positive": n_pos, "n_negative": len(cases) - n_pos},
        "prediction_mapping": {
            "debate_reject": '裁判裁决 == "reject"（与基线"驳回提案"同口径）',
            "debate_non_support": '裁判裁决 != "support"（敏感性口径）',
            "parse_failure_conservative": (
                "裁判解析失败案：reject 口径计未拦下、non_support 口径保守计拦下"
            ),
        },
        "frozen_court_reference": {
            "source": "tests/ddb/baseline_comparison_report.json#systems"
                      ".consensus_court_full.reject",
            **reference,
        },
        "llm_debate": llm_metrics,
        "deltas_vs_court_reject": deltas,
        "sensitivity_exclude_unparsed": clean_metrics,
        "parse_failures": len(parse_failure_ids),
        "parse_failure_case_ids": parse_failure_ids,
        "call_failures": len(call_failure_ids),
        "call_failure_case_ids": call_failure_ids,
        "responses_path": str(_RESPONSES_PATH.relative_to(_RESPONSES_PATH.parents[2])),
        "generated_at": utc_now_iso(),
    }

    violations: list[str] = []
    if call_failure_ids:
        violations.append(
            f"RL1 违例：{len(call_failure_ids)} 例无完整控-辩-裁三元组: "
            f"{call_failure_ids[:10]}{'...' if len(call_failure_ids) > 10 else ''}"
        )
    if illegal_verdict_ids:
        violations.append(
            f"RL2 违例：{len(illegal_verdict_ids)} 例裁判裁决非法或与原始"
            f"输出不一致（已按解析失败保守计入）: {illegal_verdict_ids[:10]}"
        )
    # 采集条件一致性（架构师审查必须修复项 2）：模型混用与 prompt
    # 指纹漂移都意味着新旧响应不可比，必须显式违例
    if len(models_present) > 1:
        violations.append(f"模型混用：响应池包含多个 model 记录: {models_present}")
    if mixed_fp_keys:
        violations.append(
            f"prompt 指纹漂移：{len(mixed_fp_keys)} 个 key 存在多指纹"
            f"（prompt 中途变更，新旧响应混池）: {mixed_fp_keys[:10]}"
        )
    violations.extend(ref_drift)
    report["red_line_violations"] = violations
    return report, violations


def main() -> int:
    """主流程：配置检查 → collect（可选）→ analyze → 红线 → 落盘。"""
    parser = argparse.ArgumentParser(description="实验 B：LLM-debate 头对头")
    parser.add_argument(
        "--phase", choices=["both", "collect", "analyze"], default="both",
        help="执行阶段（默认 both；analyze 纯离线可复跑）",
    )
    parser.add_argument("--concurrency", type=int, default=4, help="并发上限（默认 4）")
    parser.add_argument("--max-retries", type=int, default=3, help="单调用最大重试（默认 3）")
    args = parser.parse_args()

    cases = build_case_pool()
    n_pos = sum(1 for c in cases if c["polarity"] == "with_contradiction")
    print(f"案例池：{len(cases)} 例（正例 {n_pos} / 负例 {len(cases) - n_pos}），每案 3 次调用")

    if args.phase in ("both", "collect"):
        load_experiment_config()
        asyncio.run(phase_collect(cases, args.concurrency, args.max_retries))

    report, violations = phase_analyze()
    with open(_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    m = report["llm_debate"]
    print(f"报告：{_REPORT_PATH}")
    print(
        f"LLM-debate [reject]    recall={m['reject']['recall']} "
        f"ci={m['reject']['recall_ci95']} fpr={m['reject']['fpr']} "
        f"ci={m['reject']['fpr_ci95']} acc={m['reject']['accuracy']}"
    )
    print(
        f"LLM-debate [non-sup]   recall={m['non_support']['recall']} "
        f"fpr={m['non_support']['fpr']} acc={m['non_support']['accuracy']}"
    )
    print(
        f"Δ vs 法庭（reject 口径）：recall={report['deltas_vs_court_reject']['recall']} "
        f"fpr={report['deltas_vs_court_reject']['fpr']}；"
        f"parse_failures={report['parse_failures']} "
        f"call_failures={report['call_failures']}；"
        f"role_failures={m['role_call_failures']}"
    )
    for v in violations:
        print(f"[红线违例] {v}")
    if not violations:
        print("结构红线全部满足（RL1/RL2/RL3）；结果无论方向均如实落盘。")
        return 0
    print("结论：结构红线违例，报告已落盘供人工复核。")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ConfigError as exc:
        print(f"配置缺失：{exc}", file=sys.stderr)
        sys.exit(2)
    except Exception as exc:  # noqa: BLE001 —— 管线异常统一归口退出码 2
        print(f"运行异常：{exc}", file=sys.stderr)
        raise
