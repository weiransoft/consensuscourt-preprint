#!/usr/bin/env python3
"""
实验 A：LLM 仲裁器消融（论文 §8 登记项 1，v0.4.17）。

语义
====
用托管 qwen-flash 直接替代规则 PI 仲裁器：每案输入 topic + 证据池
（每条证据仅给 evidence_id/source/role/content——盲评纪律，不给任何
规则评分、不给 polarity 真值、不给 metadata），LLM 输出三档裁决
JSON {"verdict":"support|conditional|reject","confidence":0-1,
"rationale":"<=80 words"}。temperature=0；max_tokens 登记偏离计划
§2 的 300 为 2048（qwen-flash 网关思考模式需预留推理 token，
300 会耗尽导致正文恒空——详见 llm_experiment_lib.chat_with_retry）。

目的：回答"三档裁决信号是否必须来自规则核"——若 LLM 直评能达到
法庭同级的 (recall, FPR) 剖面，则规则核必要性主张被削弱；反之被
加强。结果无论方向都如实写入论文（预注册无结果红线）。

两阶段（防重跑烧钱）
====================
- Phase collect：异步并发 4、单案重试 3（指数退避）、逐案追加
  tests/ddb/llm_arbiter_responses.jsonl（含 case_id/原文/解析结果/
  usage/时间戳/prompt 指纹），断点续跑（已成功 case_id 跳过）；
- Phase analyze：离线解析 jsonl → 两口径混淆矩阵 → 指标 + Wilson CI
  → 与冻结法庭参考对比 → 报告 tests/ddb/llm_arbiter_report.json。

预注册结构红线（不设结果红线）
==============================
- RL1：208 案全部有成功响应（调用失败的案计入 call_failures 并违例）；
- RL2：每案最新记录要么解析为三枚举之一，要么计入诚实字段
  parse_failures（解析失败的案在 reject 口径按"未拦下"计、在
  non_support 口径按保守"拦下"计，并在报告声明）；
- RL3：报告内嵌法庭参考与 baseline_comparison_report.json 冻结值
  逐字段一致（互锁不漂移）。

用法：
    backend$ ../.venv-bio/bin/python tests/scripts/run_llm_arbiter_experiment.py
    backend$ ../.venv-bio/bin/python tests/scripts/run_llm_arbiter_experiment.py --phase analyze

配置：LLM_EXPERIMENT_BASE_URL / _API_KEY / _MODEL（backend/.env 或环境变量）；
缺配置 exit 2，不静默降级为模拟。

输出：tests/ddb/llm_arbiter_responses.jsonl + tests/ddb/llm_arbiter_report.json
退出码：0=红线全部满足；1=红线违例；2=运行异常/配置缺失
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from llm_experiment_lib import (  # noqa: E402
    CASES_PER_POLARITY_L1,
    FROZEN_REPORT_PATH,
    SEED,
    CallStats,
    ConfigError,
    VERDICT_ENUM,
    VerdictParseError,
    append_jsonl,
    build_case_pool,
    chat_with_retry,
    check_frozen_reference_integrity,
    check_frozen_against_canonical,
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

#: 响应与报告落盘路径（tests/ddb/ 下，与其它实验报告同级）
_RESPONSES_PATH = Path(__file__).resolve().parents[1] / "ddb" / "llm_arbiter_responses.jsonl"
_REPORT_PATH = Path(__file__).resolve().parents[1] / "ddb" / "llm_arbiter_report.json"

#: 系统提示词（盲评纪律的 prompt 层锁：只描述任务与输出格式，
#: 不含任何真值/评分/口径倾向信息）
SYSTEM_PROMPT = (
    "你是药物发现共识法庭的独立仲裁员。给定一个争议议题和一组证据，"
    "你需要判断证据池整体是否支持该议题下的研究提案：\n"
    "- support：证据相互一致，足以支持提案；\n"
    "- conditional：证据大体一致但存在局限或缺口，只能有条件支持；\n"
    "- reject：证据之间存在实质矛盾或不足以支撑提案，应予驳回。\n"
    "你必须独立依据给定的证据内容判断，不要臆造证据之外的实验结果。\n"
    '只输出一个 JSON 对象，格式：{"verdict":"support|conditional|reject",'
    '"confidence":0到1的小数,"rationale":"不超过80词的中文理由"}，'
    "不要输出任何其它文字。"
)


def build_user_prompt(case: dict) -> str:
    """
    构建单案用户提示词（议题 + 盲评证据池）。

    Args:
        case: 案例字典

    Returns:
        用户提示词文本（不含 polarity/ground_truth/metadata/评分）
    """
    return (
        f"争议议题：{case['topic']}\n\n"
        f"证据池（共 {len(case['evidences'])} 条）：\n\n"
        f"{render_evidence_pool(case)}\n\n"
        "请按系统指令输出裁决 JSON。"
    )


async def phase_collect(
    cases: list[dict],
    concurrency: int,
    max_retries: int,
    client: Any = None,
    model: str | None = None,
) -> CallStats:
    """
    Phase collect：逐案调用 LLM 并增量落盘（断点续跑）。

    已有成功响应（call_ok=True）的 case_id 跳过——烧钱保护；
    此前调用失败的记录会被重试（旧记录留在 jsonl，分析取最新）。

    Args:
        cases: 案例池（208 例）
        concurrency: 并发上限（计划规定 4）
        max_retries: 单案最大重试次数（计划规定 3）
        client: 预构建的 OpenAI 客户端（离线单测注入 MockTransport 版本；
            None 时按配置真实构建）
        model: 模型 ID（client 注入时必须一并给定，保持留痕一致）

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
        # 被低估（架构师审查必须修复项 1）
        client = AsyncOpenAI(
            base_url=cfg.base_url, api_key=cfg.api_key, max_retries=0
        )
        model = cfg.model
    if not model:
        raise ValueError("注入 client 时必须同步给定 model（留痕一致性）")
    stats = CallStats()
    semaphore = asyncio.Semaphore(concurrency)

    # 断点续跑跳过判据（架构师审查必须修复项 2）：必须同时满足
    # ① call_ok 且 parse_ok（解析失败案在续跑时补调，与实验 B 裁判
    #   语义对齐，消除"A 不重试 parse 失败"的两实验不一致）；
    # ② model 与 prompt 指纹一致——换模型或 prompt 漂移后旧响应
    #   一律重调，严禁新旧采集条件混池。
    done = load_jsonl_latest(_RESPONSES_PATH)
    done_ok: set[str] = set()
    for c in cases:
        rec = done.get(c["case_id"])
        if not (rec and rec.get("call_ok") and rec.get("parse_ok")):
            continue
        fp = prompt_sha256(SYSTEM_PROMPT, build_user_prompt(c))
        if rec.get("model") == model and rec.get("prompt_sha256") == fp:
            done_ok.add(c["case_id"])
    pending = [c for c in cases if c["case_id"] not in done_ok]
    print(
        f"[collect] 案例池 {len(cases)} 例，已有成功响应 {len(done_ok)} 例，"
        f"待调用 {len(pending)} 例（model={model}, 并发={concurrency}, "
        f"重试={max_retries}）"
    )

    async def run_one(case: dict) -> None:
        """单案：构建 prompt → 调用（含重试）→ 解析 → 逐案追加落盘。"""
        user_prompt = build_user_prompt(case)
        fingerprint = prompt_sha256(SYSTEM_PROMPT, user_prompt)
        text, err = await chat_with_retry(
            client,
            model,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            stats=stats,
            semaphore=semaphore,
            max_retries=max_retries,
        )
        record: dict = {
            "key": case["case_id"],
            "case_id": case["case_id"],
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
            try:
                parsed = parse_verdict_json(text)
                record.update(
                    parsed_verdict=parsed["verdict"],
                    confidence=parsed["confidence"],
                    rationale=parsed["rationale"],
                    parse_ok=True,
                )
            except VerdictParseError as perr:
                # 解析失败诚实落盘（不悄悄丢弃），分析阶段保守计入口径
                record["parse_error"] = str(perr)
        append_jsonl(_RESPONSES_PATH, record)

    await asyncio.gather(*(run_one(c) for c in pending))
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
    Phase analyze：离线解析 jsonl → 指标 → 报告（不联网）。

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

    for case_id, has_truth in truth_by_id.items():
        rec = records.get(case_id)
        if rec is None or not rec.get("call_ok"):
            # RL1 违例载体：无成功响应的案计入 call_failures；
            # 指标层按保守处理：reject 口径记"未拦下"（fn/tn）、
            # non_support 口径记"拦下"（与解析失败的保守方向统一，
            # 不给 LLM fail-open 通道——架构师审查建议 1）
            call_failure_ids.append(case_id)
            tally_case(cm_pair, has_truth, False, True)
            decision_dist["call_failure"] += 1
            continue
        if not rec.get("parse_ok"):
            # RL2 诚实通道：解析失败按保守处理——reject 口径"未拦下"
            # （无法给出驳回裁决）、non_support 口径"拦下"
            # （无法给出支持，保守不计入支持）
            parse_failure_ids.append(case_id)
            tally_case(cm_pair, has_truth, False, True)
            decision_dist["parse_failure"] += 1
            continue
        # 防篡改复核（架构师审查建议 4）：声称解析成功的记录必须能用
        # 原始输出重解析出一致裁决，否则按解析失败保守计口径并触发
        # RL2 违例（防 parsed_verdict 与 raw_response 被单独篡改）
        try:
            reparsed = parse_verdict_json(rec.get("raw_response") or "")
        except VerdictParseError:
            reparsed = None
        verdict = rec.get("parsed_verdict")
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
    }

    # 冻结法庭参考（RL3 互锁基准）
    frozen = load_frozen_court_reference()
    reference = dict(frozen)
    ref_drift = check_frozen_reference_integrity(reference, frozen)
    # 第二道锁：冻结文件实际值与预注册常量逐字段比对（防重跑覆盖漂移）
    ref_drift += check_frozen_against_canonical(frozen)

    # Δ 对比（reject 主口径；方向如实报告，不利也是结果）
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
        "experiment": "§8 登记项 1：LLM 仲裁器消融（qwen-flash 替代规则 PI 仲裁器）",
        "benchmark": "DDB-L1（合成争议 60×2）+ DDB-L1.5（负例对照 88 例），seed=42",
        "model": models_present[-1] if models_present else None,
        "models_present": models_present,
        "temperature": 0.0,
        "seed": SEED,
        "max_tokens": 2048,
        "max_tokens_deviation": (
            "预注册计划 §2 为 300；实测 qwen-flash 经网关默认启用思考模式，"
            "300 上限被推理内容耗尽导致正文恒空，登记偏离为 2048"
        ),
        "cases_per_polarity_l1": CASES_PER_POLARITY_L1,
        "pool_composition": {
            "n_positive": n_pos,
            "n_negative": len(cases) - n_pos,
        },
        "prediction_mapping": {
            "llm_reject": 'LLM 裁决 == "reject"（与基线"驳回提案"同口径）',
            "llm_non_support": 'LLM 裁决 != "support"（敏感性口径）',
            "parse_failure_conservative": (
                "解析失败案：reject 口径计未拦下、non_support 口径保守计拦下"
            ),
        },
        "frozen_court_reference": {
            "source": "tests/ddb/baseline_comparison_report.json#systems"
                      ".consensus_court_full.reject",
            **reference,
        },
        "llm_arbiter": llm_metrics,
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
    # RL1：208 案全部有成功响应
    if call_failure_ids:
        violations.append(
            f"RL1 违例：{len(call_failure_ids)} 例无成功响应: "
            f"{call_failure_ids[:10]}{'...' if len(call_failure_ids) > 10 else ''}"
        )
    # RL2：每案最新记录要么解析为三枚举之一，要么计入 parse_failures；
    # "声称解析成功但裁决非法/重解析不一致"既保守计口径又必须显式违例
    if illegal_verdict_ids:
        violations.append(
            f"RL2 违例：{len(illegal_verdict_ids)} 例裁决非法或与原始输出"
            f"不一致（已按解析失败保守计入）: {illegal_verdict_ids[:10]}"
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
    # RL3：内嵌法庭参考与冻结源逐字段一致
    violations.extend(ref_drift)
    report["red_line_violations"] = violations
    return report, violations


def main() -> int:
    """主流程：配置检查 → collect（可选）→ analyze → 红线 → 落盘。"""
    parser = argparse.ArgumentParser(description="实验 A：LLM 仲裁器消融")
    parser.add_argument(
        "--phase", choices=["both", "collect", "analyze"], default="both",
        help="执行阶段（默认 both；analyze 纯离线可复跑）",
    )
    parser.add_argument("--concurrency", type=int, default=4, help="并发上限（默认 4）")
    parser.add_argument("--max-retries", type=int, default=3, help="单案最大重试（默认 3）")
    args = parser.parse_args()

    cases = build_case_pool()
    n_pos = sum(1 for c in cases if c["polarity"] == "with_contradiction")
    print(f"案例池：{len(cases)} 例（正例 {n_pos} / 负例 {len(cases) - n_pos}）")

    if args.phase in ("both", "collect"):
        # 配置检查前置：缺配置立即 exit 2，不发起任何调用
        load_experiment_config()
        asyncio.run(phase_collect(cases, args.concurrency, args.max_retries))

    report, violations = phase_analyze()
    with open(_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    m = report["llm_arbiter"]
    print(f"报告：{_REPORT_PATH}")
    print(
        f"LLM 仲裁器 [reject]    recall={m['reject']['recall']} "
        f"ci={m['reject']['recall_ci95']} fpr={m['reject']['fpr']} "
        f"ci={m['reject']['fpr_ci95']} acc={m['reject']['accuracy']}"
    )
    print(
        f"LLM 仲裁器 [non-sup]   recall={m['non_support']['recall']} "
        f"fpr={m['non_support']['fpr']} acc={m['non_support']['accuracy']}"
    )
    print(
        f"Δ vs 法庭（reject 口径）：recall={report['deltas_vs_court_reject']['recall']} "
        f"fpr={report['deltas_vs_court_reject']['fpr']}；"
        f"parse_failures={report['parse_failures']} "
        f"call_failures={report['call_failures']}"
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
