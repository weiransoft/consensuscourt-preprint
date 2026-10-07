# 图表数据引用清单（DATA_PROVENANCE）

本清单将预印本 `main.tex` 中每一处定量主张映射到其唯一数据源——
`backend/tests/ddb/` 下的机器可读实验报告（确定性、零 LLM 管线生成，seed=42；
§6.10 两项 LLM 消融为例外，调用托管 qwen-flash，端点配置见附录 Reproducibility）。
任何数值修改必须先更新报告、再更新论文，双向可追溯。

## 表格 ↔ 数据源映射

| 论文位置 | 内容 | 数据源（相对 backend/tests/ddb/） | 关键字段 |
|---|---|---|---|
| Tab.1 (`tab:ddb`) | DDB 四层规模（600/88/10/256） | `l1_cases.json` / `l15_cases.json` / `l2_cases.json` / `l4_cases.json` | 案例数组长度；L2 元数据 `citation_verified_at=2026-09-20` |
| Tab.2 (`tab:main`) | 四系统对照（n=208，recall/FPR/Acc/Precision 双口径 + reject 口径 Wilson 95% CI 列） | `baseline_comparison_report.json` | `systems.{consensus_court_full,majority_vote,elo_arbiter,single_judge}.{reject,non_support}.*`、`{recall,fpr}_ci95`（`wilson_interval(k,n)` 程序化计算，z=1.96） |
| §6.2 敏感性扫描小节 + Fig.2 (`fig:pareto`) | 基线 49 格点 (0.3..0.9)² 阈值 Pareto 前沿（majority 12 点 / elo 4 点 / single 5 点）+ 法庭参考星点 (0,1) + "no baseline threshold point reaches the court profile" | `baseline_sensitivity_report.json` | `systems.*.pareto_frontier_reject`、`court_reference`、`baseline_optimum_reaches_court_profile=false`、`red_line_violations=[]` |
| §5（H2b 注记） | L2 层 n=10 召回 1.0 的 Wilson 95% 下界 0.72 | `run_baseline_comparison.py:wilson_interval(10,10)` | `[0.7225, 1.0]`（由 `test_baseline_comparison_experiment.py::TestWilsonConfidenceIntervals` 已知值锁） |
| §6.3（正文数值） | H2a pilot：guilty-presumption recall 0.6 / FPR 0.5；ABP 1.0/0.0 | `pilot_h2a_report.json` | `metrics.{guilty_presumption,abp}.*` |
| Tab.3 (`tab:courtabl`) | 法院组件消融（conviction/escalation 双层） | `ablation_v03_report.json` | `court_configs.{court_full,court_no_abp,court_no_reviewer,court_no_boss}.*`；L1.5 升级 precision=0.0 见 `court_l15_configs` |
| Tab.4 (`tab:scorerabl`) | scorer 级消融（0.4994 / 0.5248 / 0.625） | `ablation_v03_report.json` | `configs.full_v03.avg_pass_confidence`、`scorer_ablation_configs.{no_source_pedigree,no_recency}.avg_pass_confidence`、`scorer_ablation_uniform_source_score=0.775` |
| §6.6（正文数值） | 定理 2：1000 次置换 0 分歧；流式 k=1 分歧 0.070/召回 0.924、k=2 0.002/0.998；计数标度斜率 0.942 / −0.857、m*≈18.5；来源噪声 0/7400、62/3000=2.07%、召回 0.9793、翻转 13.9%/14.1%、\|Δconf\| 0.0076/max 0.578 | `robustness_report.json` | `order_permutation.*`、`streaming_disposition.*`、`n_scan.*`、`source_noise.*` |
| Tab.5 + Fig.2 (`tab:network`/`fig:net`) | 网络失败 FPR/召回曲线、重试（1.00/1.33/1.75/3.00）、缓存 133 次 | `network_failure_report.json` | `cells[*]`（含复跑格点）、`red_lines[6]`、`all_passed=true` |
| Tab.6 (`tab:decomp`) | 恒等式闭合（P(D)=0.2027、P(M)=0/0.5946、残差 0） | `advantage_decomposition_report.json` | `decomposition.{full_v03,baseline_v02}.*` |
| §6.7（正文数值） | q_K 0.0355/0.0058、κ 1.42/2.47、joint 1829/86 | `advantage_decomposition_report.json` | `null_calibration.{baseline_v02,full_v03}.*` |
| §6.8（正文数值） | 角色覆盖率 27–55、最大翻转 4/208=0.0192、阳性对照 7/6/6 | `role_weight_calibration.json` | `role_coverage`、`max_flip_rate_observed`、`positive_control_flips`、`red_line_thresholds` |
| §6.9（案例研究） | DDB-L2-0001（convicted/有条件支持/0.528/intensity=3/金标对命中）+ DDB-L2-0005（无法裁决/0.0/escalated/全过滤） | `case_study_report.json` | `case_studies[*].{verdict_chain,abp_convergence,gold_pair_detection,timeline}`、`honesty_note` |
| §6.9（H2b 锁） | L2 10/10 正例 rubric 直评检出金标对 | `test_l2_v03_locks.py`（金标召回结构锁）+ `l2_cases.json` `ground_truth.contradiction_pairs` | — |
| §6.10 Tab.7 (`tab:llmabl`) | LLM 仲裁器消融（reject 口径 recall 0.983 ci[0.9114,0.9971] / FPR 0.5 ci[0.4205,0.5795]）+ LLM-debate 头对头（recall 0.7 ci[0.5749,0.801] / FPR 0.3716 ci[0.2979,0.4518]）+ 冻结法庭参考（1.0/0.0） | `llm_arbiter_report.json` / `llm_debate_report.json` | `llm_arbiter.reject.*`、`llm_debate.reject.*`、`frozen_court_reference.*`、`deltas_vs_court_reject.*`、`sensitivity_exclude_unparsed.*`、`parse_failures=0`、`call_failures=0`、`red_line_violations=[]`（托管 qwen-flash，temperature=0，seed=42） |
| 附录 A | 13 角色权重表 | `role_weight_calibration.json` `base_weights` + `backend/design/paper-appendix-d-role-profiles.md` | — |

## 诚实边界（论文 Limitations 的数据源侧对应）

1. 案例研究数据源为 DDB-L2 文献真实争议基准（PMID 已核验），**非平台生产运行**——
   `case_study_report.json.honesty_note` 原文声明。
2. 用户研究、L3 平台层、LLM 结构角色标注消融**未执行**，论文中仅以规划项
   表述，无任何虚构数据；两项 LLM 消融（仲裁器替代、协议级辩论头对头）已执行
   并如实报告（含对 LLM 不利与有利方向的全部数字），见
   `llm_arbiter_report.json` / `llm_debate_report.json` 与论文 §6.10。
3. 全部实验为冒烟/回归规模（60–300 例/格点），主实验规模放大走同一管线。
4. 主基准与全部红线实验零 LLM 调用；平台 LLM 栈（vLLM + Qwen）不在本文
   实验范围内；§6.10 两项消融调用托管 qwen-flash（非平台栈），端点与模型配置
   在 `backend/.env` 的 `LLM_EXPERIMENT_*` 变量（不入仓库）。

## 引用真实性

- v0.4.16 新增 7 键（chowdhury2026proclaim / he2026debatecv / venkata2026trustagents
  / saxena2024reasons / qian2026atomcite / chen2026redagentbench /
  gptzero2026hallucination）已逐条对 arXiv 摘要页/出版方页面核验标题、作者、
  年份与标识符；其中 REASONS（saxena2024reasons）实际最新版为 v3（2025-04）且
  摘要仅提 hallucination rate，论文措辞不使用评审报告中"v5/Abstention Rate
  双指标"的失实表述。
- `references.bib` 中 Cecil & Freedman（arXiv:2609.03953）与 Kumar et al.
  （arXiv:2605.10171）的标识符取自 `design/paper-consensus-court-outline.md` §2.5
  （项目 Deep Research 精读记录）；其余为公认已发表文献。
- L2 案例全部 PMID 已对 NCBI esummary 二次核验（2026-09-20，含纠错复核），
  无法定位的条目在 `l2_cases.py` 审计清单中以 `citation_status=pending_verification` 显式留痕。
