# 预印本提交前最终检查清单（ConsensusCourt → arXiv）

- 基准：paper/main.tex、paper/references.bib、paper/DATA_PROVENANCE.md
- 状态标记：✅ 已核验完成 / ⬜ 提交前必须完成 / ⚠️ 建议完成（可带病提交但有风险）
- 最后更新：2026-10-09（他山 thesis-audit-reviewer 终审全量修复落地：11 项必须修改 + 12 项建议修改全部完成；摘要压缩至 1917 字符达标；重编译重打包，bib/bbl 25 键零差异；四条红线数值完好）

## A. 编译与 LaTeX 格式（arXiv 技术硬门槛）

- ✅ **A1 本地完整编译通过**（2026-09-21 修复）：本机无 MacTeX，已安装 Tectonic 0.17.0（自包含引擎，`~/.local/bin/tectonic`）执行 `tectonic --keep-intermediates main.tex` 三轮迭代编译。最终 EXIT=0、零 "!" 错误、零 undefined，产出 main.pdf（149,278 bytes，**14 页**，pypdf 确认）。编译过程发现并修复 2 个未定义宏（`\eg`、`\Ddb` 大小写敏感），证明静态校验确实无法替代真实编译。
- ✅ **A2 真实编译期风险项确认**（2026-09-21 修复）：真实编译通过后确认 pgfplots（compat=1.17）与 hyperref 无编译期冲突、theorem 环境编号正常、全部 \ref/\cite 解析无 undefined 警告。
- ✅ **A3 生成 .bbl 并随源码上传**（2026-10-09 终审刷新）：tectonic `--keep-intermediates` 重编已产出 main.bbl（10,212 bytes，**25 条 \bibitem**）。与 references.bib 的 25 个键做 python 正则双向对账（natbib 带可选参数的 `\bibitem[...]{key}` 口径），零差异；新增 wang2017eai045 / ramalingam2020os 两条案例文献均入 bbl。**上传时必须把 main.bbl 打进源码包**（arXiv 不执行 bibtex）。
- ⬜ **A4 图表完整性**：编译已通过（图/表无编译错误），但 PDF 渲染效果（表格列宽是否溢出、pgfplots 图是否被裁切）仍需人工打开 main.pdf 目检一遍——这是编译通过也查不出的视觉项。
- ✅ **A5 arXiv 打包**（2026-10-09 终审修复后重打包定稿）：`/tmp/arxiv_submission.tar.gz`（**34,442 字节，md5 `8dc011c940f04c80e1e0a66386555aaf`**）——仅含 main.tex + main.bbl + references.bib。干净目录 Tectonic `--keep-intermediates` 编译 EXIT=0、零 Overfull、**20 页**（main.pdf 198,762 bytes）。历史修复保留：四处溢出（三处 `sloppypar` + 一处集合表达式改 display 公式）+ 两处图例遮挡（fig:pareto `legend to name` 轴框上方摆放；fig:net 图例移回轴内西北窄空区，数据坐标零改动、与表 7 逐值一致）。本轮终审新增修复不触碰任何图表数据坐标。main.bbl/main.pdf/ConsensusCourt.pdf/ConsensusCourt1.pdf 已同步回 paper/ 目录（PDF md5 `8c03a61c…` 四份一致）。

## B. 元数据（arXiv 提交表单）

- ✅ **B1 作者信息替换**（2026-09-21 修复）：`\author{Wei Wang\thanks{Correspondence: weiransoft@126.com ...}}` 已落盘（main.tex L42-45），grep 验证无 "Anonymous" 残留。**注意**：未提供单位信息，作者栏当前只有姓名 + 通讯邮箱；若投稿时有单位可补充 `\affiliation`，arXiv 允许 v1 后在 v2 修改作者元数据。
- ⬜ **B2 标题与摘要录入 arXiv 元数据表单**：与 main.tex 内逐字一致（arXiv 元数据是独立字段，不是从 PDF 提取）。**摘要已压缩达标（2026-10-09 终审）**：约 2,730 字符 → **1,917 字符 ≤ 1,920 上限**（宏展开后计数），已同步 LOG-001 定罪通道、三态边界、residual 定性、`\S8` 限定、zero-LLM 范围、gold-labels 独立性六处口径。**录入时使用本清单末尾"修正版 arXiv 表单摘要"纯文本**（严禁外部模板数字 18,000 trials / 8.8–13.9pp / 80.3pp / 5×5——均不存在于论文）。
- ⬜ **B3 主分类选择**：建议 `cs.MA`（Multi-Agent Systems）为主分类，交叉 `cs.AI` / `q-bio.QM`（生物医学应用场景）；按投稿目标（如 cs.MA vs cs.AI 流量）定夺。
- ⬜ **B4 许可证选择**（✅ 已选定 2026-10-07）：**arXiv perpetual, non-exclusive license**（非 CC 系——独立作者首发 + 后续拟投期刊场景下最保守安全，与所有期刊政策兼容）。与仓库许可口径已对齐：代码/数据 MIT、稿件 All Rights Reserved（见仓库 LICENSE 附注）。
- ⬜ **B5 投稿政策声明**：确认论文未同时在其他期刊/会议审稿中（arXiv 预印本与投稿不冲突，但双投声明需如实填写）。

## C. 内容与主张严谨性（红线复查）

- ✅ **C1 论文-数据一致性**：24/24 项数值自动核对通过（Tab.2/3/4、网络失败、分解、角色、案例、L1.5 precision 全部与 tests/ddb/*.json 原值一致，见 DATA_PROVENANCE.md 映射表）。
- ✅ **C2 未执行实验零虚构**：limitations 如实声明（无用户研究 / 无 L3 / LLM 结构角色标注消融未做 / L2 规模与标注状态 / 冒烟规模 / 零 LLM 范围含两项 LLM 消融 carve-out）；正文无占位数字；§6.10 两项 LLM 消融为真实执行结果（含对 LLM 不利方向数字），数据源 `llm_arbiter_report.json` / `llm_debate_report.json`（红线全过、0 调用失败、0 解析失败）。
- ⬜ **C3 提交前人工通读一遍 Abstract 与 Conclusion**：确认每个主张都能对应 §6 的表/图（skill 的 claim–evidence 硬约束）；机器核对覆盖数值，覆盖不了措辞（如 "first to" / "significantly outperforms" 这类强主张）。
- ⚠️ **C4 DDB-L2 案例研究口径复核**：正文已如实标注案例来自 DDB-L2 文献争议基准、非平台生产运行；通读 §6 案例研究节，确认没有把 L2 叙事误写成生产运行记录。
- ⚠️ **C5 定性形容词降温**：检查 "novel" / "guarantee" / "proves" 用词——定理 1/1-B/2 是确定性代码性质命题，不是统计证明；"guarantee" 仅在引用定理时使用。

## D. 引用与数据溯源

- ⚠️ **D1 两篇 2026 近邻论文复核**（✅ 2026-10-07 复核完成，2026-10-09 终审再核）：arXiv API 直接核验——`2609.03953` = *Beyond Majority Vote: ...*，**Joe Cecil & Marjorie Freedman**（EMNLP 2026 Findings，与 bib 主题相符）；`2605.10171` = *"When Reviews Disagree: Fine-Grained Contradiction Analysis in Scientific Peer Reviews"*，Kumar/Kamdar/Hossain/Kumari/Saikh/**Ekbal**（**ACL 2026 Findings**，bib note 已按此登记）。references.bib 已修正作者名（原 `Cecil, R. and Freedman, J.` 与 `Kumar, A. and others` 为近似占位写法，已改为真实全名）并补 pages 14346–14370。**Ekbal 组由此确认为可联系的背书候选（kumar2026when 通讯作者）**。
- ✅ **D2 案例文献 PMID 复核**（2026-10-07 首核，2026-10-09 终审扩核）：PubMed eutils esummary 实拉——PMID 29151359：Soria et al., J Hematol Oncol 2018;11(1):17，标题逐字一致；PMID 27251290：Jia et al., Nature 2016;534(7607):340-5，标题逐字一致。**终审新增**：PMID 27840244（Wang S, Song Y, Liu D; Cancer Lett 385:51-54, 2017 = EAI045）与 PMID 31751012（Ramalingam SS et al; NEJM 382:41-50, 2020 = FLAURA OS）核验通过，新增 bib 条目 `wang2017eai045` / `ramalingam2020os` 并替换 p.13 原误引（原将 Cancer Lett 2017 挂到 jia2016eai045、NEJM OS 只引 Soria 2018）。
- ✅ **D3 引用 parity**：\cite↔bib 双向核对通过，无未引用键、无缺失键。
- ✅ **D4 代码可用性声明**（2026-10-07 完成）：公开仓库 https://github.com/weiransoft/consensuscourt-preprint 已上线（main=`d127f34`，tag `v0.4.17`，Release 已发布），含 paper/（tex+bbl+bib+PDF+DATA_PROVENANCE）、code/core（被测实现）、code/scripts（逐表/图确定性管线）、code/tests（pytest+DDB 生成器）、code/data（全部机器可读报告）、LICENSE（代码 MIT/稿件 ARR）、README、.zenodo.json。**Zenodo DOI 已生成（2026-10-07，API 核验）**：concept DOI **10.5281/zenodo.23218390**、version DOI **10.5281/zenodo.23218391**（记录 23218391，标题/作者 Wang, Wei 正确）。**main.tex \thanks 已落实仓库引用（2026-10-09 终审）**：补入 `\url{https://github.com/weiransoft/consensuscourt-preprint} (tag \texttt{v0.4.17}; DOI 10.5281/zenodo.23218391)`；正文路径已统一为 `backend/tests/ddb/`、`backend/tests/scripts`（与仓库结构一致）。

## E. 附属材料与复现包

- ✅ **E1 仓库可复现性抽查**（2026-10-07 重跑）：`bash backend/tests/scripts/run_v03_regression.sh` **REGRESSION_EXIT=0**——Step 1（v0.3 专项）通过、既有回归**零回归**通过、实验管线 3/3b/3b-2/3c/3d/3e/3f/3g 全通过（三态 FPR=0 全格点、优势分解恒等式、案例研究 L2 锁对齐均在日志确认）。日志 `/tmp/regression_run3.log`。注意：脚本要求 **Python ≥ 3.10**（数据库层 `typing.ParamSpec`）——首跑用系统 python3.9 曾致 252 处收集期 ImportError（exit 1），换 Python 3.11 解释器后全绿；仓库 README 已写明此要求。
- ⬜ **E2 论文复现命令表与仓库对齐**：附录 D 的复现命令逐条在干净环境验证过一遍（特别是 --scorer-ablation 与 --mode court_full）。
- ⚠️ **E3 补充材料决策**：tests/ddb/*.json 报告是否作为 arXiv ancillary files 或仓库 release 附件公开（建议仓库公开即可，论文脚注指向 repo + tag）。

## F. arXiv 提交流程操作项

- ⬜ **F1 endorsement（唯一无法由工具代办的关键缺口）**：作者在 cs.MA/cs.AI 若无发表记录，arXiv 要求该分类 endorsement。**2026-10-07 状态更新**：q-bio 实测同样需背书（q-bio 整体为单一 endorsement 域，per-domain 不覆盖 cs）；候选锁定：**① Yash Saxena（Case Western，saxena2024reasons 一作，REASONS 为本文直接对比对象）→ 冷邮件已起草待发；② Asif Ekbal（IIT Patna，kumar2026when 通讯，IMPACT/RevCI 为本文 aspect-gate/graded-intensity 设计来源）→ 邮件已起草，间隔≥5 天后发；③ 备选 Joe Cecil/Manas Gaur**。操作规则：邮箱一律从 arXiv 摘要页 "Submission history" 取登记地址；同一人不追发；附论文 PDF + endorsement code。具体操作：
  1. 访问 https://arxiv.org/auth/endorsement，输入分类（cs.MA / cs.AI）与论文元数据，系统生成 endorsement 链接；
  2. 寻找 endorser 的标准：需在**同一主分类**下近 5 年有至少 N 篇（通常 4-8 篇）arXiv 作者身份提交记录，且邮箱来自学术机构域名可信度更高；
  3. 找 endorser 途径：合作者 / 同领域作者（本文相关工作引用的作者如 du2023debate、wu2023autogen 等不合适——只找认识或可建立联系的人）、导师 / 同事已有的 endorsement 网络；把系统生成的 endorsement 链接发给对方，由其登录 arXiv 账号确认；
  4. endorser 确认后 endorsement 有效期 30 天，期内完成首次提交即可；
  5. 若 1-2 周内找不到 endorser：可先提交到不需要 endorsement 的分类（如 cs.LG 通常也不豁免——以 endorsement 页面实际提示为准），或联系 arXiv help（moderation@arxiv.org）说明情况。
  **时间线建议**：今天发起 endorsement → 收到确认后再走 F2 的提交时间选择。
- ⬜ **F2 提交时间选择**：arXiv 工作日 14:00 ET 前提交通常当日晚间公布（避免周五下午/公告假日，影响占坑时间戳）。
- ⬜ **F3 公布后 24h 复查**：公布后核对 PDF 渲染（公式/表格/图）、元数据、参考文献完整性；发现问题立即提交修正版（v2 保留 v1 时间戳，不损失占坑优先权）。

## 已完成事项存档（本轮冲刺证据）

| 事项 | 证据 |
|---|---|
| 论文数值 ↔ 报告 JSON 一致性（24 项） | 自动核对脚本 24/24 PASS |
| LaTeX 静态校验（环境/ref/cite/花括号） | 四类检查全过（含 \% 转义修复） |
| 全量回归 exit 0 | 276 passed + 119 passed，Step 3/3b-3g 全过 |
| 架构师审查 | 结论"可合并"，3 处加固已落地 + 锁测试 |
| 诚实边界声明 | main.tex §8 六条 limitations + DATA_PROVENANCE.md 四条 |
| 引用真实性 | 15/17 条已核验，2 条待 D1 复核 |
| **A1 真实编译通过**（2026-09-21） | Tectonic 0.17.0 三轮迭代，EXIT=0 / 零 error / 零 undefined，main.pdf 14 页（pypdf 确认）；修复未定义宏 \eg、\Ddb |
| **A3 main.bbl 生成与对账**（2026-09-21） | main.bbl 6,521 bytes / 16 条 \bibitem，与 references.bib 16 键双向对账零差异 |
| **B1 作者信息替换**（2026-09-21） | Wei Wang / weiransoft@126.com 落盘 main.tex L42-45，无 Anonymous 残留 |
| **v0.4.16 对抗性评审路线图修复**（2026-09-21） | P0 五项（§2 machine-side 段 / first 主张收敛 / Abstract-Conclusion 重心重排 + synthetic-layer 限定 / 定理 1/3 降格 Proposition / 引用层）+ P1 四项（Tab.2 Wilson CI 列、§6.2 阈值敏感性扫描 49 格点 + Pareto 图 fig:pareto、§7 escalation cost model、§6.1 agent-eval 定位段 + §6.5 REASONS 对话）+ P2 三项登记（§8 limitation 3/4 扩写 + 第三方基准交叉验证条目，只登记设计不虚构实验）；references.bib 16→23 键（7 新键逐条核验，D3 parity 复验零差异）；架构师审查 2 项必须修复已落地（majority 亚对角 elif 语义镜像 + §6.2 前沿措辞与数据对齐）；tectonic 重编译 EXIT=0，main.bbl 23 bibitem 对账零差异；新增 test_baseline_sensitivity.py（U19）+ TestWilsonConfidenceIntervals（U13.4），Step 1 相关 32 例全过；回归脚本 Step 1 + Step 3b-2 接入 |
| **v0.4.16 后待跑**（2026-09-21） | 全量回归 run_v03_regression.sh（Step 1 现含 308 项专项测试）exit 0 确认；C3/C5 人工通读按修订后 Abstract/Conclusion 重新执行 |
| **v0.4.17 两项真实 LLM 消融**（2026-09-21） | 实验 A（LLM 仲裁器消融，208 案）+ 实验 B（LLM-debate 控辩裁头对头，208×3=624 调用）真实执行完成（托管 qwen-flash，temperature=0，seed=42，红线 RL1/RL2/RL3 全过，0 调用失败 / 0 解析失败）；报告 `llm_arbiter_report.json` / `llm_debate_report.json`；论文 §6.10（tab:llmabl）+ 摘要/引言/贡献/related work/§6.1/§8/结论/附录 repro 九处同步限定；单测 56/56（A 30+13、B 13，含架构师审查修复项回归锁：SDK 隐藏重试关闭、断点续跑 model/指纹一致性、raw_response 防篡改重解析、敏感性剔除口径）；架构师审查 2 必须修复 + 4 建议已落地；tectonic 编译 EXIT=0、新增 overfull 全部清零（剩余 4 处为 v0.4.16 前遗留） |
| **v0.4.17 审查修订：LLM 实验支撑性 + 表述对齐**（2026-09-21） | 他山科研 thesis-audit-reviewer 审查：① 数据支撑性结论——两项 LLM 消融强力支持论文核心主张（ABP 决定 FPR 而非 recall：LLM 仲裁器 recall 0.983 接近法庭 1.0 但 FPR 0.500 vs 法庭 0.0；辩论结构无定罪规则反而退化 recall 0.700 且 0 support/208）；② 全文 zero-LLM 声明 9 处一致排除两项消融；③ Tab.7 数字与 JSON 逐字段一致；④ 4 处表述修订落地——(a) §6.10 honesty note 补充 max_tokens 预注册偏离声明（300→2048，qwen-flash 思考模式）；(b) Tab.7 caption 说明断点续跑机制（each case retains first successful response）；(c) 观察(3) 因果性主张措辞降温（"consistent with prosecution's framing prevailing"替代"prosecutor dominates"）；(d) Tab.7 调用计数改为"208 cases"/"208 cases × 3 roles, 0 unresolved failures"避免重试歧义；tectonic 编译 EXIT=0，无新增 overfull |
| **v0.4.17 方法论论述补充**（2026-09-21） | 回应"为何零LLM而非直接整合LLM"的质疑：① §6.1 新增"Why zero-LLM for the core experiments"段——三点方法论辩护（变量隔离纯化ABP架构效果、预注册红线需确定性可验证、一键复现无API依赖），明确零LLM是方法论选择非排斥LLM；② Discussion 新增"ABP and LLMs are complementary, not competing"段——明确ABP是与仲裁器解耦的程序性架构，LLM可在三个slot接入（scorer/矛盾检测器/仲裁器），ABP定罪规则必须始终保留，目标架构是"LLM components inside an ABP envelope"；tectonic 编译 EXIT=0，无新增 overfull |
| **终审全量修复（他山 thesis-audit-reviewer + 四角色专家）**（2026-10-09） | 11 项必须修改全部落地——REF-001..006（gottweis arXiv ID 2502.18864、jia2016eai045/kapoor/saxena/hong/du 作者与 venue 逐条按 arXiv API / NCBI / PMLR 权威源更正）+ 新增 wang2017eai045/ramalingam2020os 修复 p.13 误引 + DAT-001/002/003（150 defect-only、0/150 clean-neutral、2/208 翻转方向）+ LOG-001/002（定罪通道唯一性、acquit→escalate 语义）+ MET-001（摘要 2730→**1917 字符**）；12 项建议修改全部落地（provably×2 删除、P2 限定、zero-LLM 三处统一、Δ/K 符号定义、\S8、孤立表 tab:ddb/tab:decomp 补引、\thanks 补 URL+tag+DOI、路径统一、elo 限定、流式聚合口径、rigour 拼写、Limitations 两项新披露、Conclusion gold-labels、Theorem/§6.7 advantage-ratio 方向改"decays/coincides with the Bonferroni form"）；复核：tectonic 编译 EXIT=0 零 Overfull 20 页、bib/bbl 25 键双向零差异、四条红线数值完好（FPR 0.000×23、kappa=0、residual=0、图-表-JSON 一致）、tar 包 34,442 bytes md5 `8dc011c9…`、PDF md5 `8c03a61c…` 四份一致 |

## 修正版 arXiv 表单摘要（B2 录入用，1917 字符 ≤ 1920 上限）

> 来源：main.tex 摘要宏展开纯文本（与编译后 PDF 第 1 页逐字一致，仅 smallcaps `\ddb`→DDB、`\S\ref`→Section 编号展开）。**严禁**使用任何外部模板数字（18,000 trials / 8.8–13.9pp / 80.3pp / 5×5——均不存在于论文）。

```text
Multi-agent LLM research pipelines generate conflicting evidence streams; existing aggregation resolves them by rhetorical weight or ignores them, leaving two failure modes: false conviction of well-formed claims built on low-quality evidence, and verification outages silently polluting verdicts. We present ConsensusCourt, courtroom-style procedural arbitration: every claim is presumed innocent until a detected contradiction pair or a hard-constraint violation convicts it (asymmetric burden of proof, ABP); evidence is weighted by a source-pedigree/recency/relevance scorer whose confidence channel is structurally decoupled from conviction; an escalation layer routes unadjudicable cases to humans. Three results independent of the benchmark's hand-adjudicated gold labels: (i) a counting law for the advantage over guilty-presumption screening (slopes 0.942/-0.857 vs. 1.0/-1.0 expected); (ii) a network-failure experiment where a three-state verification policy holds the clean-claim FPR at 0 under failure rates up to 100%-abstaining (recall 0) at full outage rather than fabricating-while the common two-state policy convicts up to 100% of innocent claims, sole variable: the failure-mapping policy; and (iii) conviction-confidence decoupling preserved exactly under scorer ablation. On the author-built synthetic layer of our 954-case benchmark (DDB, regression-scale, Section 8) the court alone reaches recall 1.0 at FPR 0.0; majority vote, Elo, and single-judge baselines fail on at least one Pareto side at every scanned threshold. A per-case advantage decomposition closes by construction with pure layer attribution (residual = 0 on 148 negatives); verdicts are order-invariant over 1,000 permutations. The benchmark and every deterministic red-line experiment are zero-LLM; two registered LLM-side ablations run against a hosted LLM (Section 6.12); all pipelines carry pre-registered red lines in CI.
```
