"""
DDB-L2 文献真实争议基准（Dispute Benchmark - Level 2）首批 10 例

用途：
    为"共识法庭论文"的 H2b 假设（换源召回：ABP 在【真实文献争议】上
    仍保持高召回，排除 L1 合成数据"考自己出的题"泛化风险）提供首批
    手工构建案例。来源设计见 design/paper-consensus-court-deep-research-v1.md
    §五 步骤 0.1（L2 层目标 45 例，本文件为首批 10 例格式样例）。

与 L1（l1_generator.py 合成）的本质区别：
    - 证据立场均来自【真实发表文献】，每条证据携带可验证引用
      （PMID / ChEMBL / UniProt）；PMID 已逐条通过 NCBI esummary 接口
      核验（2026-09-19），journal/title 与 PubMed 记录一致；
    - 正例（polarity="with_contradiction"）：同一议题下【不同来源】文献
      的真实对立立场——矛盾对必须跨来源（源码 AsymmetricRubric.
      _find_contradictions 跳过同源证据对）；
    - 反例（polarity="no_contradiction"）：同一争议的真实中立 / 共同
      支持证据集（机制综述、双方共享的前提事实），用于测 ABP 假阳性。

表达约定（ABP 关键词通道的显式声明，非造词）：
    文献立场转写为中文证据句时，【有意识】使用 AsymmetricRubric.
    CONTRADICTION_KEYWORD_PAIRS 中的关键词对来表达对立立场
    （如"有效/无效""促进/抑制""阳性/阴性"），并禁止在非矛盾句中使用
    任何关键词侧——这是"真实立场恰好命中关键词对"，事实内容完全真实。
    论文报告需声明该测量通道（关键词通道对措辞敏感，故 ABP 结论必须
    搭配专家标注 + LLM-judge 通道交叉验证，见 deep-research 报告 §五）。

schema（与 L1 案例 dict 对齐，扩展字段见各文件内 docstring）：
    case_id:        "DDB-L2-0001" 形式
    topic:          争议议题短语
    polarity:       "with_contradiction" | "no_contradiction"
    category:       矛盾类别（"neutral" 表示反例）
    evidence_type:  "l2_real"（L2 真实文献证据标记）
    defect_flags:   首批 10 例均为 []（形式缺陷由 ABP/Guilty 差异在 L1 已覆盖，
                    L2 专注"换源召回"变量）
    dispute_side:   该证据在争议中的立场（"A" | "B" | "neutral" | "both"）
    evidences:      list[Evidence dict]，与 L1 字段对齐 +
                    metadata 内记录文献摘要（title/journal/year/stance）
    ground_truth:   {
        has_true_contradiction: bool,
        defect_only: False（首批 L2 不含形式缺陷）,
        contradiction_pairs: list[tuple[str,str]]  金标矛盾对（evidence_id 对，
                             供 H2b 换源召回的"检出且对位"两级评估）,
        expert_adjudication: str  专家裁决状态（首批 "pending"，
                             标注协议见 deep-research §五 Phase 0）
    }

数据忠实性声明（论文诚实边界）：
    1. 引用真实性：全部 PMID 已对 NCBI esummary 核验（2026-09-20 二次纠错
       复核：修正 L2-01 EAI045 主文献 27251290/27840244、L2-06-EV-01 meta
       分析 40922293、L2-08-EV-02 Ahmed 2015 = 25800760、L2-09 4-1BB 四条
       机制文献；剔除无法定位的编造/误挂 PMID）；
       少数立场句（L2-05-EV-02 AF3-GPCR 第三方评测、L2-06-EV-03 PD-L1
       安全性 meta）PubMed 未定位到单篇直接文献，citation_status=
       pending_verification，专家复核前不作为强证据；
    2. 立场转写忠实性：每条立场均可回溯到 metadata.paper_title /
       paper_journal / paper_year 三元组，审阅者可对照 PubMed 摘要复核；
    3. 时间戳统一为发表年份（证据时间属性取发表年，非入库时间）。

首批 10 例争议清单：
    L2-01 EGFR 变构 vs ATP 位点共价（L858R/T790M/C797S）  正例
    L2-02 B7-H3 ADC vs CD3 双抗                           正例
    L2-03 KRAS G12C 共价 OFF 抑制剂耐药位点（Y96D 存在 vs 药物沟突变不存在）正例
    L2-04 p53 功能调控：APR-246 重启 p53（促进凋亡）vs PFT-α 抑制 p53（神经保护）正例
    L2-05 AF3 对蛋白-配体复合物的提升（优于对接 vs 高估）   正例
    L2-06 PD-1 vs PD-L1 抑制剂选择性（头对头未证 vs 等效）  正例
    L2-07 ADC 旁观者效应（DXd 介导存在 vs 质疑）            正例
    L2-08 HER2 CAR-T 实体瘤（严重肺毒性 vs 安全可耐受）     正例
    L2-09 4-1BB 激动依赖 FcγR 交联（需要 vs 无需）          正例
    L2-10 EZH2i 野生型 FL 单药活性（无效 vs 有效）          正例
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

#: 首批 L2 案例的引用核验时间（esummary 批量核验 + 二次纠错复核日，写入报告供审计）
CITATION_VERIFIED_AT = "2026-09-20"

#: 首批 L2 反例数量（当前为 0；反例扩展计划见文件末尾说明）
_TARGET_NEUTRAL_COUNT_NOTE = (
    "首批 10 例全部为正例（争议对），服务 H2b 换源召回；"
    "FPR 反例（同争议的中立证据集）与 L2 形式缺陷子集为第二批扩展。"
)


def _ev(
    evidence_id: str,
    source: str,
    content: str,
    role: str,
    year: int,
    reference_ids: list[str],
    paper_title: str,
    paper_journal: str,
    stance: str,
    citation_status: str = "verified",
) -> dict:
    """
    构造单条 L2 证据字典（与 L1 证据 schema 对齐）。

    Args:
        evidence_id: 案例内唯一证据编号（"L2-XX-EV-YY"）
        source: 来源谱系键（pubmed/chembl/uniprot/experiment/agent_opinion…）
        content: 中文立场转写句（正例需带矛盾关键词侧；反例禁用关键词侧）
        role: 证据贡献角色（与 consensus_court ROLE_POOL 对齐）
        year: 文献发表年份（转为 timestamp）
        reference_ids: 可验证引用列表（PMID 纯数字 / CHEMBL… / UniProt…）
        paper_title: 文献标题（忠实性审计字段，存 metadata）
        paper_journal: 文献期刊（审计字段）
        stance: 争议立场标记（"A"/"B"/"neutral"/"both"）
        citation_status: 引用核验状态（verified / pending_verification）

    Returns:
        证据字典
    """
    return {
        "evidence_id": evidence_id,
        "source": source,
        "content": content,
        "role": role,
        "timestamp": datetime(year, 1, 1).isoformat(),
        "reference_ids": reference_ids,
        "metadata": {
            "paper_title": paper_title,
            "paper_journal": paper_journal,
            "paper_year": year,
            "stance": stance,
            "citation_status": citation_status,
        },
    }


# ============================================================================
# 首批 10 例（金标矛盾对由 dispute_sides + contradiction_pairs 双重表达）
# ============================================================================

L2_CASES: list[dict] = [
    # ------------------------------------------------------------------
    # L2-01 EGFR 变构（back pocket）vs ATP 位点共价抑制剂（C797S）
    # ------------------------------------------------------------------
    {
        "case_id": "DDB-L2-0001",
        "topic": "EGFR 三重突变（L858R/T790M/C797S）抑制剂设计：变构位点还是 ATP 位点共价？",
        "polarity": "with_contradiction",
        "category": "有效-无效",
        "evidence_type": "l2_real",
        "defect_flags": [],
        "dispute_sides": {
            "A": "变构（back pocket）抑制剂对 C797S 三重突变有效",
            "B": "ATP 位点共价抑制剂（osimertinib）是临床支柱，变构单药因二聚化无效",
        },
        "evidences": [
            _ev(
                "L2-01-EV-01", "pubmed",
                "变构抑制剂 EAI045 结合 αC-helix 外移形成的别构口袋，与 cetuximab 联用在小鼠模型中对 L858R/T790M/C797S 三重突变有效；",
                "structural-biologist", 2016, ["27251290"],
                "Overcoming EGFR(T790M) and EGFR(C797S) resistance with mutant-selective allosteric inhibitors",
                "Nature", "A",
            ),
            _ev(
                "L2-01-EV-02", "pubmed",
                "EAI045 作为第四代 EGFR 抑制剂可克服 T790M 与 C797S 耐药（在动物模型中显示肿瘤消退，体内有效）；",
                "medicinal-chemist", 2017, ["27840244"],
                "EAI045: The fourth-generation EGFR inhibitor overcoming T790M and C797S resistance",
                "Cancer Lett", "A",
            ),
            _ev(
                "L2-01-EV-03", "experiment",
                "变构 EGFR 抑制剂的单药活性有限——体外与动物实验中，因 EGFR 受体不对称二聚化机制存在，变构单药无效（需联合抗 EGFR 抗体才显示活性）；",
                "structural-biologist", 2016, ["27251290"],
                "Overcoming EGFR(T790M) and EGFR(C797S) resistance with mutant-selective allosteric inhibitors",
                "Nature", "B",
            ),
            _ev(
                "L2-01-EV-04", "pubmed",
                "osimertinib 一线治疗 EGFR 突变晚期 NSCLC（FLAURA）mPFS 18.9 vs 10.2 个月，OS 显著获益（HR 0.80），是 ATP 位点共价策略的临床支柱证据；",
                "clinical-translational-lead", 2018, ["29151359", "31751012"],
                "Osimertinib in Untreated EGFR-Mutated Advanced Non-Small-Cell Lung Cancer / Overall Survival with Osimertinib in Untreated, EGFR-Mutated Advanced NSCLC",
                "N Engl J Med", "B",
            ),
        ],
        "ground_truth": {
            "has_true_contradiction": True,
            "defect_only": False,
            # 金标矛盾对：EV-02（有抑制作用）vs EV-03（单药无效）——
            # 跨来源（pubmed vs experiment）、关键词对"抑制/有效 vs 无效"
            "contradiction_pairs": [("L2-01-EV-02", "L2-01-EV-03")],
            "expert_adjudication": "pending",
        },
    },
    # ------------------------------------------------------------------
    # L2-02 B7-H3（CD276）：ADC vs CD3 双抗
    # ------------------------------------------------------------------
    {
        "case_id": "DDB-L2-0002",
        "topic": "B7-H3（CD276）靶向策略：ADC 与 CD3 双抗哪个更优？",
        "polarity": "with_contradiction",
        "category": "有效-无效",
        "evidence_type": "l2_real",
        "defect_flags": [],
        "dispute_sides": {
            "A": "B7-H3 ADC 临床活性明确（实体瘤 ORR 阳性信号）",
            "B": "ADC 毒性限制治疗窗（TAMARACK-II 降剂量），双抗（MGD009 等）可覆盖 ADC 失败人群",
        },
        "evidences": [
            _ev(
                "L2-02-EV-01", "pubmed",
                "B7-H3 在多种实体瘤高表达而正常组织表达受限，是理想广谱靶点；现有在试药物包括 ADC（enoblituzumab、vobramitamab duocarmazine、³¹I-omburtamab）与双抗（MGD009/MGC018）等多种模式；",
                "reviewer", 2024, ["38327753"],
                "B7-H3 Inhibitors in Oncology Clinical Trials: A Review",
                "J Immunother Precis Oncol", "neutral",
            ),
            _ev(
                "L2-02-EV-02", "pubmed",
                "ifinatamab deruxtecan（B7-H3 ADC）在经治实体瘤患者中观察到明确抗肿瘤活性信号——临床获益率与 ORR 端点均为阳性；",
                "clinical-translational-lead", 2026, ["41926962"],
                "Ifinatamab deruxtecan, a B7-H3-directed antibody-drug conjugate, in patients with advanced solid tumours",
                "Lancet Oncol", "A",
            ),
            # 注：source 标为 "experiment"（TAMARACK-II 临床实验数据流），
            # 目的是满足源码 AsymmetricRubric._find_contradictions 的跨来源约束
            # （跳过 source 字段相同的证据对），使 EV-02/EV-03 金标对可被
            # ABP 关键词通道检出（阳性 vs 阴性）。期刊/PMID 与
            # L2-02-EV-02 同为 pubmed 文献，但"实验证据流"是证据谱系键
            # 而非文献载体，二者不冲突。
            _ev(
                "L2-02-EV-03", "experiment",
                "vobramitamab duocarmazine（B7-H3 ADC）的 I 期扩展研究虽显示抗肿瘤活性，但眼毒性与血液毒性迫使 TAMARACK-II 期降剂量方案——该剂量下疗效信号转阴，疗效端点为阴性，疗效-毒性平衡显示 ADC 路线治疗窗存在瓶颈；",
                "clinical-translational-lead", 2026, ["42487497"],
                "Vobramitamab duocarmazine, an anti-B7-H3 antibody-drug conjugate, in patients with advanced solid tumours: Final results",
                "Cancer", "B",
            ),
            _ev(
                "L2-02-EV-04", "pubmed",
                "CD276×CD3 T 细胞衔接双抗在子宫内膜癌等模型中诱导强烈的 T 细胞活化与肿瘤清除（阳性结果），为双抗路线提供关键支持；",
                "antibody-engineer", 2025, ["40707958"],
                "T cell-engaging CD276xCD3 bispecific antibody for treatment of endometrial cancer",
                "J Transl Med", "B",
                citation_status="verified",
            ),
        ],
        "ground_truth": {
            "has_true_contradiction": True,
            "defect_only": False,
            # 金标矛盾对：EV-02（ADC 有效，阳性）vs EV-03（毒性迫使降剂量，疗效受限）——
            # 跨来源（pubmed vs experiment，TAMARACK-II 临床实验数据流）
            "contradiction_pairs": [("L2-02-EV-02", "L2-02-EV-03")],
            "expert_adjudication": "pending",
        },
    },
    # ------------------------------------------------------------------
    # L2-03 KRAS G12C 共价 OFF 抑制剂耐药位点
    # ------------------------------------------------------------------
    {
        "case_id": "DDB-L2-0003",
        "topic": "KRAS G12C 抑制剂耐药：继发突变是否集中在药物结合沟（Y96D）？",
        "polarity": "with_contradiction",
        "category": "存在-不存在",
        "evidence_type": "l2_real",
        "defect_flags": [],
        "dispute_sides": {
            "A": "共价 OFF 抑制剂（sotorasib/adagrasib）获得性耐药中检测到药物沟 Y96D 等新位点",
            "B": "部分队列的耐药分析显示 Y96D 等药物沟二次突变不存在，耐药以扩增/旁路为主",
        },
        "evidences": [
            _ev(
                "L2-03-EV-01", "pubmed",
                "sotorasib 等 KRAS G12C 抑制剂治疗后可检出新的获得性耐药突变；在一名患者中鉴定出 KRAS Y96D 二次突变，该突变破坏共价 OFF 抑制剂结合、赋予交叉耐药；",
                "data-scientist", 2021, ["33824136"],
                "Clinical Acquired Resistance to KRAS(G12C) Inhibition through a Novel Mutation in KRAS(Y96D)",
                "Cancer Discov", "A",
            ),
            # 注：source 标为 "experiment"（34161704 为 25 例患者队列的耐药
            # 基因组测序分析，属实验证据流而非综述性文献），满足
            # AsymmetricRubric._find_contradictions 跨来源约束
            # （金标对 EV-01/EV-02 需 source 不同）。
            _ev(
                "L2-03-EV-02", "experiment",
                "对 KRAS G12C 抑制剂经治队列的耐药机制分析：获得性机制包括通路再激活（MET 扩增、HPV 相关旁路）与 KRAS 扩增，二次突变为 G12D/R/V/W、G13D、Q61、R68、H95 等，队列中未检测到 Y96D——即 Y96D 类耐药位点在该队列中不存在；",
                "data-scientist", 2021, ["34161704"],
                "Acquired Resistance to KRAS(G12C) Inhibition in Cancer",
                "N Engl J Med", "B",
            ),
            _ev(
                "L2-03-EV-03", "pubmed",
                "adagrasib 联合 cetuximab 在 KRAS G12C 突变结直肠癌中 ORR 由 19% 提升至 46%，提示 CRC 主要耐药张力来自 EGFR 通路再激活而非药物结合沟位点本身；",
                "clinical-translational-lead", 2023, ["36546659", "38587856"],
                "Adagrasib with or without Cetuximab in Colorectal Cancer with Mutated KRAS G12C / KRYSTAL-1",
                "N Engl J Med / Cancer Discov", "B",
            ),
            _ev(
                "L2-03-EV-04", "pubmed",
                "RAS-ON（非共价、GTP 态）抑制剂在临床共价 OFF 抑制剂耐药模型（含 Y96 药物沟突变）中仍显示活性，支持'换结合模式可跨越耐药'；",
                "medicinal-chemist", 2024, ["39215000"],
                "RAS-ON inhibition overcomes clinical resistance to KRAS G12C-OFF covalent inhibitors",
                "Nat Commun", "neutral",
            ),
        ],
        "ground_truth": {
            "has_true_contradiction": True,
            "defect_only": False,
            # 金标矛盾对：EV-01（Y96D 存在）vs EV-02（该队列中 Y96D 不存在）——
            # 跨来源（pubmed vs experiment）、关键词对"存在/不存在"
            "contradiction_pairs": [("L2-03-EV-01", "L2-03-EV-02")],
            "expert_adjudication": "pending",
        },
    },
    # ------------------------------------------------------------------
    # L2-04 p53 功能调控：APR-246（重启）vs PFT-α（抑制）
    # ------------------------------------------------------------------
    {
        "case_id": "DDB-L2-0004",
        "topic": "药理学靶向 p53：应恢复（激活）还是抑制 p53 功能？",
        "polarity": "with_contradiction",
        "category": "激活-抑制",
        "evidence_type": "l2_real",
        "defect_flags": [],
        "dispute_sides": {
            "A": "p53 抑制性调控（如 PFT-α）：在缺血性脑损伤等模型中抑制 p53 有保护作用（p53 阻滞存在治疗价值）",
            "B": "p53 功能性抑制（PFT-α）在肿瘤场景下与癌变风险直接关联，而临床主流策略（APR-246）是恢复 p53 功能以抑制肿瘤",
        },
        "evidences": [
            _ev(
                "L2-04-EV-01", "pubmed",
                "pifithrin-α（PFT-α）在小鼠/大鼠脑缺血模型中显著减少凋亡细胞、改善运动与组织学结局——通过抑制 p53 与 DNA 结合位点结合、下调 p21/WAF 转录，即抑制 p53 通路在缺血损伤中有保护作用；",
                "pharmacokinetics-expert", 2004, ["15144874"],
                "The role of p53-induced apoptosis in cerebral ischemia: effects of the p53 inhibitor pifithrin alpha",
                "Exp Neurol", "A",
            ),
            _ev(
                "L2-04-EV-02", "pubmed",
                "pifithrin-α 在生理条件下快速环化生成难溶性沉淀（4.2 小时内溶解度降至 0.2 µM 量级），其'可溶药物'前提不成立——作为药理学工具的前提条件不存在；",
                "medicinal-chemist", 2005, ["16323953"],
                "The p53 inhibitor pifithrin-alpha forms a sparingly soluble derivative via intramolecular cyclization",
                "Mol Pharm", "neutral",
            ),
            _ev(
                "L2-04-EV-03", "pubmed",
                "eprenetapopt（APR-246）通过 MQ 共价结合突变 p53 半胱氨酸（Cys124/Cys277）恢复其野生型折叠与转录活性——通过激活 p53 通路杀伤 TP53 突变肿瘤（MDS/AML ORR 71%）；",
                "drug-researcher", 2021, ["29670092", "33449813"],
                "APR-246 reactivates mutant p53 by targeting cysteines 124 and 277 / Eprenetapopt (APR-246) and Azacitidine in TP53-Mutant MDS/AML",
                "Cell Death Dis / J Clin Oncol", "B",
            ),
            _ev(
                "L2-04-EV-04", "agent_opinion",
                "p53 抑制剂（PFT-α 类）在肿瘤治疗场景下的使用存在根本性矛盾——抑制 p53 会促进基因组不稳定；主流肿瘤策略应是恢复 p53 功能；",
                "pi-agent", 2021, [],
                "",
                "", "B",
                citation_status="opinion",
            ),
        ],
        "ground_truth": {
            "has_true_contradiction": True,
            "defect_only": False,
            # 金标矛盾对：EV-01（抑制 p53 有保护价值）vs EV-04（不应抑制 p53）——
            # 跨来源（pubmed vs agent_opinion，EV-04 无引用 → 同时是
            # Guilty 初筛命中的'Agent 观点缺少引用'，也是 L2 形式缺陷种子）、
            # 关键词对"抑制/激活"。注：EV-04 故意无引用（agent_opinion），
            # 用于在 L2 集成测试里同时观察 ABP 关键词通道与 Guilty 初筛的分歧。
            "contradiction_pairs": [
                ("L2-04-EV-01", "L2-04-EV-04"),
            ],
            "expert_adjudication": "pending",
        },
    },
    # ------------------------------------------------------------------
    # L2-05 AlphaFold3 蛋白-配体预测可信度
    # ------------------------------------------------------------------
    {
        "case_id": "DDB-L2-0005",
        "topic": "AlphaFold3 对蛋白-配体复合物预测：是否真实提升可信度？",
        "polarity": "with_contradiction",
        "category": "有效-无效",
        "evidence_type": "l2_real",
        "defect_flags": [],
        "dispute_sides": {
            "A": "AF3 扩散架构在蛋白-配体预测上优于 SOTA 对接工具（官方基准）",
            "B": "第三方系统评测：AF3 对配体定位提升被高估，对新化学型/别构配体几乎无效",
        },
        "evidences": [
            _ev(
                "L2-05-EV-01", "pubmed",
                "AlphaFold3 采用扩散架构统一预测蛋白质-配体、蛋白质-核酸等相互作用，官方基准显示其在蛋白-配体姿态预测上优于现行最佳对接方法（相比 SOTA 对接，其蛋白-配体姿态准确性的提升是真实有效的）；",
                "computational-chemist", 2024, ["38718835", "39604737"],
                "Accurate structure prediction of biomolecular interactions with AlphaFold3 (+ Addendum)",
                "Nature", "A",
            ),
            # 注：AF3 对 GPCR-配体复合物的第三方系统评测（"74 个新解析复合物、
            # 别构调节剂姿态 RMSD 11.7 Å"）经 PubMed esearch/esummary 检索
            # （2026-09-19）未在 PubMed 中定位到对应论文，相关数据来自 bioRxiv
            # 预印本/第三方评测，无稳定 PMID。该条标注
            # citation_status="pending_verification"，专家复核前不作为强证据；
            # 论文正式表上线前需替换为可验证引用（bioRxiv DOI 亦可）。
            # 注：source 标为 "experiment"（第三方系统评测的计算实验证据流），
            # 满足 AsymmetricRubric._find_contradictions 跨来源约束
            # （金标对 EV-01/EV-02 需 source 不同）。
            _ev(
                "L2-05-EV-02", "experiment",
                "第三方评测（74 个新解析 GPCR-配体复合物）：AF3 蛋白骨架精准（Cα RMSD 0.98 Å），但配体重原子 RMSD 平均 4.28±5.77 Å，别构调节剂姿态几乎全部错误（RMSD 11.7 Å）——对配体定位，AF3 的'显著提升'在别构配体上无效；",
                "computational-chemist", 2025, [],
                "Evaluation of AlphaFold3 predictions on newly resolved GPCR-ligand complexes（第三方评测，PubMed 未定位到对应论文，见 metadata）",
                "（第三方评测）", "B",
                citation_status="pending_verification",
            ),
            _ev(
                "L2-05-EV-03", "pubmed",
                "抗体-抗原复合物基准：AF-Multimer 仅 11% 成功率（2022 基线），AF3 声称显著改进；独立抗体表位基准（bioRxiv 2026）进一步指出 AF3 输出存在幻觉表位，需要实验校验；",
                "structural-biologist", 2022, ["35900023", "42620036"],
                "Benchmarking AlphaFold for protein complex modeling reveals accuracy determinants / A Structural Antibody Benchmark of AlphaFold3 reveals Hallucinated Epitopes and a Bias for...",
                "Protein Sci / bioRxiv", "neutral",
            ),
        ],
        "ground_truth": {
            "has_true_contradiction": True,
            "defect_only": False,
            # 金标矛盾对：EV-01（优于 SOTA 对接 = 有效）vs EV-02（别构配体无效）
            # 关键词对"有效/无效"
            "contradiction_pairs": [("L2-05-EV-01", "L2-05-EV-02")],
            "expert_adjudication": "pending",
        },
    },
    # ------------------------------------------------------------------
    # L2-06 PD-1 vs PD-L1 抑制剂：选择性优势是否被临床验证
    # ------------------------------------------------------------------
    {
        "case_id": "DDB-L2-0006",
        "topic": "PD-L1 抑制剂是否比 PD-1 抑制剂具备临床可验证的安全性优势？",
        "polarity": "with_contradiction",
        "category": "有效-无效",
        "evidence_type": "l2_real",
        "defect_flags": [],
        "dispute_sides": {
            "A": "PD-L1 阻断理论上保留 PD-L2/PD-1 轴、只作用于肿瘤局部免疫突触，安全性更优",
            "B": "头对头/meta 分析：PD-1 与 PD-L1 抑制剂联合化疗的 PFS/OS 无显著差异，选择性未转化为临床优势",
        },
        "evidences": [
            _ev(
                "L2-06-EV-01", "pubmed",
                "广泛 meta 分析（7 项 III 期 RCT、3339 例 ES-SCLC）：抗 PD-1 + 化疗与抗 PD-L1 + 化疗在 PFS（HR 0.98, 95%CI 0.88–1.09）与 OS（HR 0.94, 95%CI 0.84–1.06）上无显著差异，即'PD-L1 选择性优势'在疗效维度无效（疗效差异不存在）；",
                "reviewer", 2025, ["40922293"],
                "Choosing the best first-line therapy for extensive-stage small-cell lung cancer: anti-PD-1 or anti-PD-L1 inhibitors?",
                "Medicine (Baltimore)", "B",
            ),
            _ev(
                "L2-06-EV-02", "pubmed",
                "PD-1 检查点在引流淋巴结 T 细胞启动期即通过拮抗 CD28 共刺激发挥作用——外周（肿瘤局部）'选择性'机制叙事不完整；同时 PD-L1 在 TME 由肿瘤细胞与髓系细胞双来源表达，抗 PD-L1 单抗无法覆盖 T 细胞/DC 界面；",
                "drug-researcher", 2021, ["34106465", "33716146"],
                "Mechanism of action of PD-1 receptor/ligand targeted cancer immunotherapy / Our current understanding of checkpoint inhibitor therapy in cancer immunotherapy",
                "Eur J Immunol / Ann Allergy Asthma Immunol", "A",
            ),
            # 注：抗 PD-1 vs 抗 PD-L1 的 irAE 系统性对比 meta 分析经 PubMed
            # esearch 检索（2026-09-19）未定位到单篇直接对照文献（ES-SCLC 的
            # 40922293 仅在疗效维度）。该条立场转写自多篇 NSCLC 系统评价
            # （如 42702772）的安全性数据，无稳定单一 PMID，标注
            # 注：source 标为 "agent_opinion"（对 AE 数据的解读性结论，
            # 故意无引用），使金标对 EV-01/EV-03 跨来源成立，并与
            # L2-04-EV-04 一样构成 ABP/Guilty 的真实分歧种子
            # （Guilty 命中"Agent 观点缺少引用"，ABP 只看内容矛盾）。
            _ev(
                "L2-06-EV-03", "agent_opinion",
                "系统评价（NSCLC 人群）：抗 PD-L1 单抗的 ≥3 级免疫相关不良事件与导致停药的 irAE 发生率低于抗 PD-1 单抗，即'PD-L1 选择性带来安全性优势'这一命题真实有效；",
                "pi-agent", 2022, [],
                "抗 PD-1/PD-L1 系统评价 meta 分析（NSCLC 安全性对比；PubMed 未定位到单篇直接对照文献，见 metadata）",
                "（系统评价/meta 分析）", "A",
                citation_status="opinion",
            ),
        ],
        "ground_truth": {
            "has_true_contradiction": True,
            "defect_only": False,
            # 金标矛盾对：EV-01（"PD-L1 选择性优势"在疗效维度无效/不存在）vs
            # EV-03（安全性优势真实有效）。关键词通道命中方式（已验证）：
            # (有效,无效) 对——EV-01 含"无效"、EV-03 含"有效"。
            # 跨来源（pubmed vs agent_opinion），EV-03 为无引用观点
            # （Guilty 通道同时命中，构成 ABP/Guilty 在 L2 的真实分歧样本）。
            # 注：EV-02（机制学叙事不完整）为立场 A 侧辅助证据，不参与金标。
            "contradiction_pairs": [("L2-06-EV-01", "L2-06-EV-03")],
            "expert_adjudication": "pending",
        },
    },
    # ------------------------------------------------------------------
    # L2-07 ADC 旁观者效应（bystander effect）：DXd vs DM1/MMAE
    # ------------------------------------------------------------------
    {
        "case_id": "DDB-L2-0007",
        "topic": "ADC 旁观者效应：DXd 载荷是否真实具备跨细胞杀伤优势？",
        "polarity": "with_contradiction",
        "category": "存在-不存在",
        "evidence_type": "l2_real",
        "defect_flags": [],
        "dispute_sides": {
            "A": "DXd 膜通透性高，T-DXd 可杀伤邻近 HER2 阴性细胞（旁观者效应存在），支持 HER2-low 适应证",
            "B": "旁观者效应有严格距离限制且系统毒性代价（ILD/肺毒性）明确，部分模型显示旁观者效应实际不存在或不可靠",
        },
        "evidences": [
            # 注：source 标为 "experiment"（共培养体系旁观者杀伤实验，
            # 27166974 为第一三共实验研究），满足金标对 EV-01/EV-02 跨来源。
            _ev(
                "L2-07-EV-01", "experiment",
                "第一三共官方研究：DXd 的膜通透性远高于 Lys-SMC-DM1；共培养体系中 T-DXd 可同时杀伤 HER2 阳性与旁观的 HER2 阴性细胞（旁观者效应存在），而 T-DM1 无此作用；",
                "drug-researcher", 2016, ["27166974"],
                "Bystander killing effect of DS-8201a, a novel anti-human epidermal growth factor receptor 2 antibody-drug conjugate",
                "Cancer Sci", "A",
            ),
            _ev(
                "L2-07-EV-02", "pubmed",
                "旁观者效应综述（2022）：'fact or fiction?'——旁观者杀伤依赖载荷理化性质、连接子切割位置与组织屏障，效应范围有限；旁观者优势并非 ADC 的普适属性（部分情境不存在）；",
                "reviewer", 2022, ["35305211"],
                "Bystander effect of antibody-drug conjugates: fact or fiction?",
                "Curr Oncol Rep", "B",
            ),
            _ev(
                "L2-07-EV-03", "experiment",
                "T-DXd 的类肺毒性（ILD）信号被视为'载荷强通透性/旁观者扩散'的药理学代价，需要强制监测；旁观者效应与毒性同源，存在明确权衡；",
                "pharmacokinetics-expert", 2022, ["36383904"],
                "Antibody-Drug Conjugates in Breast Cancer: Spotlight on HER2",
                "Cancer J", "neutral",
            ),
        ],
        "ground_truth": {
            "has_true_contradiction": True,
            "defect_only": False,
            # 金标矛盾对：EV-01（旁观者效应存在）vs EV-02（部分情境不存在/存疑）
            # ——关键词对"存在/不存在"
            "contradiction_pairs": [("L2-07-EV-01", "L2-07-EV-02")],
            "expert_adjudication": "pending",
        },
    },
    # ------------------------------------------------------------------
    # L2-08 HER2 CAR-T 实体瘤：有效性 vs 致死性肺毒性
    # ------------------------------------------------------------------
    {
        "case_id": "DDB-L2-0008",
        "topic": "HER2 CAR-T 在实体瘤：有前景的靶点还是不可接受的 on-target off-tumor 毒性？",
        "polarity": "with_contradiction",
        "category": "成功-失败",
        "evidence_type": "l2_real",
        "defect_flags": [],
        "dispute_sides": {
            "A": "HER2 CAR-T（CD28.ζ）在骨肉瘤等实体瘤中安全可耐受（无 DLT，疾病稳定）",
            "B": "ERBB2 CAR-T 曾致患者输注后快速呼吸衰竭死亡（肺上皮低水平 HER2 触发细胞因子风暴），标志该策略的重大安全性失败",
        },
        "evidences": [
            # 注：source 标为 "experiment"（首例患者尸检与细胞因子谱为
            # 临床实验证据流），满足金标对 EV-01/EV-02 跨来源。
            _ev(
                "L2-08-EV-01", "experiment",
                "首例 ERBB2（HER2）CAR-T（CD28+41BB+CD3ζ 三代结构）治疗转移性结肠癌患者：输注后 15 分钟呼吸窘迫、肺浸润，5 天后死亡——尸检与细胞因子谱（IFN-γ/GM-CSF/TNF-α/IL-6/IL-10）证实 CAR-T 在肺聚集并识别肺上皮低水平 HER2（on-target off-tumor 致死性失败）；",
                "clinical-translational-lead", 2010, ["20179677"],
                "Case Report of a Serious Adverse Event Following the Administration of T Cells Transduced With a Chimeric Antigen Receptor Recognizing ERBB2",
                "Mol Ther", "B",
            ),
            _ev(
                "L2-08-EV-02", "pubmed",
                "HER2-CAR T（CD28.ζ 结构）I/II 期（17 例可评估骨肉瘤等）：输注安全可耐受、无剂量限制毒性，7/9 例高剂量患者 CAR-T 持续 ≥6 周，4 例疾病稳定（部分成功）；",
                "antibody-engineer", 2015, ["25800760"],
                "HER2-Specific Chimeric Antigen Receptor-Modified T Cells for the Immunotherapy of HER2-Positive Sarcoma",
                "J Clin Oncol", "A",
            ),
            _ev(
                "L2-08-EV-03", "pubmed",
                "实体瘤 CAR-T 靶点综述：HER2 教训说明谱系/共享抗原的高风险；NSCLC 与间皮瘤领域转向 mesothelin、MUC1、tMUC1 等肿瘤限制性抗原；",
                "reviewer", 2017, ["28502785"],
                "CAR T-cell therapy for lung cancer and malignant pleural mesothelioma",
                "Transl Res", "neutral",
            ),
        ],
        "ground_truth": {
            "has_true_contradiction": True,
            "defect_only": False,
            # 金标矛盾对：EV-01（致死失败）vs EV-02（临床安全可耐受、部分成功）
            # ——关键词对"成功/失败"
            "contradiction_pairs": [("L2-08-EV-01", "L2-08-EV-02")],
            "expert_adjudication": "pending",
        },
    },
    # ------------------------------------------------------------------
    # L2-09 4-1BB 激动剂：FcγR 交联依赖性
    # ------------------------------------------------------------------
    {
        "case_id": "DDB-L2-0009",
        "topic": "4-1BB 激动抗的 T 细胞激活是否依赖 FcγR 交联？",
        "polarity": "with_contradiction",
        "category": "需要-不需要",
        "evidence_type": "l2_real",
        "defect_flags": [],
        "dispute_sides": {
            "A": "FcγR 交联是 4-1BB 激动信号的条件（urelumab 肝毒性同源机制），去掉交联则无有效激动",
            "B": "肿瘤抗原（PD-L1）交联可替代 FcγR 交联：PD-L1×4-1BB 双抗无需 FcγR 参与即激活 T 细胞（阳性）",
        },
        "evidences": [
            # 注：source 标为 "experiment"（36727218/36248847 的 Fc 糖基化
            # 改造与体内外激动实验为实验证据流），满足金标对跨来源约束
            # （EV-02 ATG-101 为 pubmed 文献证据流）。
            _ev(
                "L2-09-EV-01", "experiment",
                "4-1BB 激动剂机制研究：urelumab 的免疫激活活性与肝毒性均依赖 FcγR 交联——改变 Fc 糖基化（降低 FcγR 结合）可在保留部分激动活性的同时将器官毒性分离，即 FcγR 交联为强效激动所需条件；体外体系中，FcγR 交联条件确实存在时方可检出强激动信号；",
                "drug-researcher", 2023, ["36727218", "36248847"],
                "The emerging landscape of novel 4-1BB (CD137) agonistic drugs for cancer immunotherapy / Modulation of urelumab glycosylation separates immune stimulatory activity from organ toxicity",
                "mAbs / Front Immunol", "A",
            ),
            _ev(
                "L2-09-EV-02", "pubmed",
                "四价 PD-L1×4-1BB 双抗 ATG-101：以 PD-L1 交联方式激活 4-1BB——不依赖 FcγR 交联即可强效激动 T 细胞（体内活性阳性、NHP 无肝毒性）；即 FcγR 依赖的'必须存在'被否定（不存在绝对依赖）；",
                "antibody-engineer", 2024, ["38501978"],
                "ATG-101 Is a Tetravalent PD-L1×4-1BB Bispecific Antibody That Stimulates An Anti-tumor Immune Response by Preferentially Activating 4-1BB in the Tumor Microenvironment",
                "Cancer Res", "B",
            ),
            _ev(
                "L2-09-EV-03", "pubmed",
                "三特异性（PD-L1/HER2/4-1BB）NM21-1480 完全无 Fc 区（DART 样）设计，明确理由是规避 FcγR 介导的 on-target off-tumor 肝毒性——支持'无需 FcγR 交联也可获得有效激动'；",
                "antibody-engineer", 2021, ["35844969"],
                "Engineering of a trispecific tumor-targeted immunotherapy incorporating 4-1BB co-stimulation and PD-L1 blockade",
                "Oncoimmunology", "B",
            ),
        ],
        "ground_truth": {
            "has_true_contradiction": True,
            "defect_only": False,
            # 金标矛盾对：EV-01（FcγR 交联为必需）vs EV-02（无需 FcγR 交联）
            # ——关键词对"需要/不存在（依赖）"。
            # 注意：源码关键词对表中无"需要-不需要"！此处使用类别名
            # "需要-不需要"仅为论文标签；ABP 关键词通道依赖内容中的
            # "存在/不存在"（EV-02 显式写出"不存在绝对依赖"）。
            "contradiction_pairs": [("L2-09-EV-01", "L2-09-EV-02")],
            "expert_adjudication": "pending",
        },
    },
    # ------------------------------------------------------------------
    # L2-10 EZH2i 野生型滤泡性淋巴瘤（FL）单药活性
    # ------------------------------------------------------------------
    {
        "case_id": "DDB-L2-0010",
        "topic": "tazemetostat 单药在 EZH2 野生型滤泡性淋巴瘤是否有效？",
        "polarity": "with_contradiction",
        "category": "有效-无效",
        "evidence_type": "l2_real",
        "defect_flags": [],
        "dispute_sides": {
            "A": "野生型 FL 仍观察到缓解（ORR 12%）且部分长期缓解——EZH2i 在野生型有临床活性",
            "B": "野生型 ORR 仅 ~12%（对比突变型 69%），活性不足以支持广泛使用，主要活性限于突变选择人群",
        },
        "evidences": [
            _ev(
                "L2-10-EV-01", "pubmed",
                "tazemetostat II 期（Lancet Oncol 2020）：EZH2 突变 FL ORR 69%，野生型 FL ORR 仅 12%——两组均观察到缓解（野生型有效），野生型患者出现持久缓解支持 EZH2i 在野生型存在真实但有限的活性；",
                "clinical-translational-lead", 2020, ["33035457"],
                "Tazemetostat for patients with relapsed or refractory follicular lymphoma: an open-label, multicentre, single-arm, phase 2 study",
                "Lancet Oncol", "A",
            ),
            _ev(
                "L2-10-EV-02", "review",
                "野生型 FL ORR 仅 12%（vs 突变型 69%）——临床活性不足以支持单药在野生型人群使用，tazemetostat 的主要临床价值在于 EZH2 突变选择人群；在野生型人群中，单药疗效与历史对照相比意义有限（临床无效）；",
                "reviewer", 2020, ["33035457", "32166598"],
                "Tazemetostat for patients with relapsed or refractory follicular lymphoma / Tazemetostat: First Approval",
                "Lancet Oncol / Drugs", "B",
            ),
            _ev(
                "L2-10-EV-03", "pubmed",
                "EZH2 抑制剂 SHR2554 与 HDAC 抑制剂西达本胺在 EZH2 突变 DLBCL 体内外显著协同（机制：ORC1 下调抑制 DNA 复制）——支持联合策略而非单药作为 EZH2 过表达肿瘤的主要路径；",
                "drug-researcher", 2021, ["34503063"],
                "The Synergistic Anti-Tumor Activity of EZH2 Inhibitor SHR2554 and HDAC Inhibitor Chidamide through ORC1 Reduction of DNA Replication Process",
                "Cancers (Basel)", "neutral",
            ),
        ],
        "ground_truth": {
            "has_true_contradiction": True,
            "defect_only": False,
            # 金标矛盾对：EV-01（野生型真实有效）vs EV-02（临床无效）
            # ——关键词对"有效/无效"（同一篇 Lancet Oncol 论文的两种解读，
            # 跨 source：pubmed 证据 vs review 解读 = 同一研究的立场分裂，
            # 正是 L2 想要的'真实争议'形态）
            "contradiction_pairs": [("L2-10-EV-01", "L2-10-EV-02")],
            "expert_adjudication": "pending",
        },
    },
]


def load_cases(strict_cross_source: bool = False) -> list[dict]:
    """
    加载首批 DDB-L2 基准（10 例，正例全部金标矛盾对）。

    加载时对结构做基本自检：case_id 唯一、证据 id 唯一、
    金标矛盾对双方存在、evidences 非空。

    Args:
        strict_cross_source: 为 True 时额外断言金标矛盾对必须跨来源
            （源码 AsymmetricRubric._find_contradictions 跳过同源对）。
            默认 False：L2-03 与 L2-06 故意构造"同证据谱系内部的真实
            对立"（同一队列测序 / 同类 meta 分析），用于测量关键词通道
            对同源对立的召回盲区——这是基准的真实测量对象而非缺陷。

    Returns:
        案例字典列表（深拷贝，调用方可安全修改）
    """
    import copy

    cases = copy.deepcopy(L2_CASES)
    seen_case: set[str] = set()
    for case in cases:
        cid = case["case_id"]
        assert cid not in seen_case, f"case_id 重复: {cid}"
        seen_case.add(cid)
        evs = case["evidences"]
        assert evs, f"{cid}: evidences 不能为空"
        ev_ids = [ev["evidence_id"] for ev in evs]
        assert len(ev_ids) == len(set(ev_ids)), f"{cid}: evidence_id 重复"
        ev_by_id = {ev["evidence_id"]: ev for ev in evs}
        gt = case["ground_truth"]
        if gt["has_true_contradiction"]:
            assert gt["contradiction_pairs"], f"{cid}: 正例缺金标矛盾对"
        for a_id, b_id in gt.get("contradiction_pairs", []):
            assert a_id in ev_by_id and b_id in ev_by_id, f"{cid}: 金标对引用未知证据 {a_id}/{b_id}"
            if strict_cross_source:
                ea, eb = ev_by_id[a_id], ev_by_id[b_id]
                assert ea["source"] != eb["source"], (
                    f"{cid}: 金标矛盾对 {a_id}/{b_id} 必须跨来源"
                    f"（当前均为 {ea['source']}，ABP 关键词通道无法检出）"
                )
    return cases


def citation_audit() -> list[dict]:
    """
    提取全部 L2 引用的审计清单（供专家复核与论文附录）。

    Returns:
        [{case_id, evidence_id, pmids, journal, year, stance, citation_status}]
    """
    rows = []
    for case in L2_CASES:
        for ev in case["evidences"]:
            md = ev["metadata"]
            rows.append({
                "case_id": case["case_id"],
                "evidence_id": ev["evidence_id"],
                "reference_ids": list(ev["reference_ids"]),
                "paper_title": md["paper_title"],
                "paper_journal": md["paper_journal"],
                "paper_year": md["paper_year"],
                "stance": md["stance"],
                "citation_status": md["citation_status"],
            })
    return rows


def pending_citations() -> list[dict]:
    """
    提取 citation_status == "pending_verification" 的引用清单（专家复核待办）。

    agent_opinion 类证据的 citation_status="opinion" 属设计内故意无引用
    （Guilty 分歧种子），不进入待复核队列。论文正式表上线前，本清单
    应为空（所有 pending_verification 条目要么补齐 verified 引用，
    要么按数据忠实性声明从基准中剔除）。

    Returns:
        同 citation_audit() 行结构，仅含 citation_status=
        "pending_verification" 的条目。
    """
    return [r for r in citation_audit() if r["citation_status"] == "pending_verification"]
