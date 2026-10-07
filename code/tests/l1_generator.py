"""
DDB-L1 合成争议基准生成器（Dispute Benchmark - Level 1）

用途：
    为"共识法庭论文"的 H2a 假设（非对称举证 ABP 相比旧"有罪推定"规则初筛
    显著降低假阳性率）提供可复现的合成争议案例基准。

    每个案例模拟一次共识法庭审理的输入证据集，标签（ground_truth）由构造
    过程保证，无需人工标注：
    - with_contradiction（有矛盾）：案例内至少两条 **不同来源** 的证据分别
      包含 AsymmetricRubric.CONTRADICTION_KEYWORD_PAIRS 中同一对的两个关键词
      → 真实 ABP 必然检出矛盾对并拒绝（测召回）。
      注意：源码 _find_contradictions 会跳过"同一来源"的证据对，因此矛盾
      关键词必须分布在至少两种不同 source 上。
    - no_contradiction（无矛盾）：全部证据内容保证不含任何矛盾关键词对的
      "任一侧"关键词（生成后自检 + 运行时用源码关键词表二次验证），
      其中约一半注入形式缺陷（内容过短 / Agent 观点缺引用）
      → 真实 ABP 应默认通过（defect_only 子集用于测两策略假阳性差异）。

schema（案例 dict，见 DdbCase.to_dict）：
    case_id:        str   "DDB-L1-0001" 形式的全局唯一编号
    topic:          str   药物发现议题短语
    polarity:       str   "with_contradiction" | "no_contradiction"
    category:       str   8 类矛盾对之一（"neutral" 表示无矛盾案例）
    evidence_type:  str   clean（无缺陷）| short_content | missing_reference |
                          short_and_missing_reference（矛盾案例的可有形式缺陷）
    defect_flags:   list  故意注入的形式缺陷清单：
                          short_content / missing_reference（unknown_source
                          仅在缺陷计数后从内容中剥离，形成"来源未知"的纯形式
                          缺陷，内容仍无矛盾关键词）
    evidences:      list[Evidence dict]，字段与 consensus_court.Evidence 对齐：
                          evidence_id / source / content / role /
                          timestamp(ISO 或 null) / reference_ids / metadata
    ground_truth:   {
        has_true_contradiction: bool  polarity 为有矛盾时 True
        defect_only:            bool  无矛盾但含形式缺陷（测假阳性子集）
    }

运行：
    cd backend && python -m tests.ddb.l1_generator
    → 生成 tests/ddb/l1_cases.json（UTF-8, ensure_ascii=False, indent=2）

一致性约束（与源码对齐，严禁修改核心模块）：
    - 关键词对唯一权威来源：
      app.core.virtual_lab.rubric_engine.AsymmetricRubric.CONTRADICTION_KEYWORD_PAIRS
      （源码当前仅 8 对，正好覆盖本基准 8 个类别）
    - 形式缺陷必须命中 consensus_court.RuleEngine.screen() 的有罪推定判定：
      1) 内容为空或 strip 后长度 < 10        → "证据内容过短"
      2) 来源为空                             → "证据来源为空"
         （pilot 使用 source="unknown"：screen() 因 truthy 不拦截、
          EvidenceScorer 给 0.2 低分；unknown_source 仅出现在缺陷计数后，
          用于保证 no_contradiction 案例内容不含任何矛盾关键词）
      3) source=agent_opinion 且无引用        → "Agent 观点缺少引用"
"""

from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# 允许从仓库根目录直接运行本文件（python tests/ddb/l1_generator.py）
# 时，把 backend 根目录加入 sys.path；从 backend/ 以 -m 方式运行时无副作用。
_BACKEND_ROOT = str(Path(__file__).resolve().parents[2])
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

# app.config 对以下两项做非空强校验（fail-closed），本模块仅使用纯内存的
# rubric 常量，占位值不会触达任何真实服务；setdefault 不覆盖已有真实配置。
os.environ.setdefault("SCIMINER_API_KEY", "ddb-pilot-placeholder")
os.environ.setdefault("DB_PASSWORD", "ddb-pilot-placeholder")

from app.core.virtual_lab.rubric_engine import AsymmetricRubric

# ============================================================================
# 常量与词表（与 consensus_court / rubric_engine 源码严格对齐）
# ============================================================================

#: 证据来源谱系（键 = consensus_court.EvidenceScorer.SOURCE_WEIGHTS 的来源键，
#: 值 = 采样权重：高可信数据库常见，低可信来源少见但仍会出现）
SOURCE_POOL: dict[str, float] = {
    "pubmed": 22.0,
    "chembl": 16.0,
    "uniprot": 14.0,
    "drugbank": 12.0,
    "experiment": 12.0,
    "llm": 14.0,
    "agent_opinion": 10.0,
}

#: 角色池（v0.4.13 对齐：论文 13 角色全覆盖 + 虚拟实验室平台扩展角色）。
#: - 论文 13 角色见大纲 L137 / 附录 D 权重表，与 PIAgentArbiter.ROLE_WEIGHTS
#:   显式登记一致；data-specialist（论文名）= data-analyst（平台实现名）；
#: - assay-scientist / medicinal-chemist / pharmacokinetics-expert /
#:   clinical-translational-lead 为平台扩展角色（仲裁走 0.5 fallback 权重）；
#: - data-scientist 已移除（无任何权重登记，与 data-specialist 混淆）。
ROLE_POOL: list[str] = [
    # --- 论文 13 角色 ---
    "drug-researcher",
    "structural-biologist",
    "computational-chemist",
    "synthetic-chemist",
    "antibody-engineer",
    "peptide-designer",
    "qc-inspector",
    "data-specialist",
    "project-coordinator",
    "report-expert",
    "knowledge-engineer",
    "reflexion-agent",
    "pi-agent",
    # --- 历史兼容 fallback 角色 ---
    "reviewer",
    "agent",
    # --- 平台扩展角色 ---
    "assay-scientist",
    "medicinal-chemist",
    "pharmacokinetics-expert",
    "clinical-translational-lead",
]

#: 8 个药物发现矛盾类别。每个类别的 keywords 必须与
#: AsymmetricRubric.CONTRADICTION_KEYWORD_PAIRS 中的某一组双向兼容
#: （即该组两个词都出现在 keywords 列表里），模板保证单句只出现一侧关键词。
#: 适配说明：激活/抑制 与 促进/抑制 共享"抑制"，为消除双向匹配歧义，
#: 激活类模板只使用 {a}=激活/{b}=抑制，促进类模板只使用 {a}=促进/{b}=抑制。
CONTRADICTION_CATEGORIES: list[dict] = [
    {
        "name": "有效-无效",
        "keywords": ("有效", "无效"),
        "topic": "EGFR L858R 突变体对小分子抑制剂的敏感性",
        "subject": "候选化合物 WQ-101 对 EGFR L858R 突变细胞",
        "a": "在构效关系分析中显示{kw}，IC50 数据支持该判断",
        "b": "在重复给药实验中表现{kw}，剂量-效应曲线未见明显改善",
    },
    {
        "name": "成功-失败",
        "keywords": ("成功", "失败"),
        "topic": "ADC 药物偶联工艺的稳定性争议",
        "subject": "ADC 候选药 WQ-207 的偶联纯化工艺",
        "a": "三轮放大实验均{kw}，偶联均一性达到放行标准",
        "b": "批次间质检报告判定为{kw}，偶联率波动超出验收区间",
    },
    {
        "name": "阳性-阴性",
        "keywords": ("阳性", "阴性"),
        "topic": "hERG 心脏安全性筛选结果的解释分歧",
        "subject": "先导化合物 WQ-118 在 hERG 通道筛选中结果",
        "a": "多次独立复筛为{kw}，提示存在剂量依赖的心脏安全性风险",
        "b": "同一靶点的复核实验判为{kw}，未见有统计学意义的抑制率",
    },
    {
        "name": "激活-抑制",
        "keywords": ("激活", "抑制"),
        "topic": "变构调节剂对 AKT1 激酶活性的作用方向",
        "subject": "变构调节剂 WQ-305 对 AKT1 激酶活性",
        "a": "酶活检测提示显著{kw}效应，磷酸化水平同步升高",
        "b": "FRET 底物实验却观察到明显{kw}，初速度下降约四成",
    },
    {
        "name": "存在-不存在",
        "keywords": ("存在", "不存在"),
        "topic": "KRAS G12C 口袋隐秘结合位点的可及性争论",
        "subject": "KRAS G12C switch-II pocket 的隐秘结合位点",
        "a": "低温冷冻电镜密度图显示该位点{kw}，可容纳小分子配体",
        "b": "分子动力学轨迹分析则认为该位点{kw}，仅在诱导后瞬时开放",
    },
    {
        "name": "高表达-低表达",
        "keywords": ("高表达", "低表达"),
        "topic": "生物标志物 HER2 在入组队列中的分布",
        "subject": "入组队列中 HER2 蛋白在肿瘤组织",
        "a": "免疫组化统计显示{kw}，与既往文献报道的分布一致",
        "b": "同一队列的 RNA 测序结果却是{kw}，蛋白与转录层结果背离",
    },
    {
        "name": "促进-抑制",
        "keywords": ("促进", "抑制"),
        "topic": "辅料聚山梨酯-80 对蛋白聚集的影响",
        "subject": "辅料聚山梨酯-80 对治疗性抗体聚集行为",
        "a": "加速稳定性实验显示其显著{kw}聚集体生成，SEC 峰值抬升",
        "b": "正交方法 light scattering 数据却显示{kw}聚集，粒径保持平稳",
    },
    {
        "name": "上调-下调",
        "keywords": ("上调", "下调"),
        "topic": "WQ 系列化合物对 CYP3A4 代谢酶的调控方向",
        "subject": "候选药 WQ-066 对肝细胞 CYP3A4 表达",
        "a": "转录组分析发现明显{kw}，代谢通路富集评分居前",
        "b": "qPCR 验证却提示{kw}，酶活水平同步走低",
    },
]

#: 无矛盾案例的中性议题池（主题 + 陈述模板 + 侧重点）。
#: 所有模板字符串严禁包含任何矛盾关键词对的任一侧关键词（生成后自检强制校验）。
NEUTRAL_TOPICS: list[dict] = [
    {
        "topic": "口服小分子化合物 WQ-142 的 PK 特征",
        "subject": "化合物 WQ-142 在大鼠 PK 实验中",
        "focus": "药代特征",
        "assert": "{v}，暴露量随剂量近似线性增加",
    },
    {
        "topic": "先导化合物 WQ-330 的激酶谱选择性",
        "subject": "激酶谱筛选显示先导化合物 WQ-330 对 c-MET",
        "focus": "选择性结论",
        "assert": "{v}，同名家族其余靶点未见同等强度结合",
    },
    {
        "topic": "IgG1 单抗 WQ-Ab6 的稳定性处方筛选",
        "subject": "加速稳定性试验中单抗 WQ-Ab6 在候选处方内",
        "focus": "稳定性结论",
        "assert": "{v}，高分子聚合物比例维持在放行限度以下",
    },
    {
        "topic": "PROTAC 分子 WQ-781 的透膜性评估",
        "subject": "平行渗透性实验提示 PROTAC 分子 WQ-781 的 Caco-2 透膜性",
        "focus": "透膜性结论",
        "assert": "{v}，外排比处于可开发窗口",
    },
    {
        "topic": "WQ 平台高通量筛选的假阳性控制",
        "subject": "针对化合物库的剂量反应复筛流程",
        "focus": "流程判定",
        "assert": "{v}，曲率拟合残差处于验收标准之内",
    },
    {
        "topic": "氘代类似物 WQ-450 的代谢位点封闭效果",
        "subject": "微粒体代谢实验显示氘代类似物 WQ-450 的代谢清除率",
        "focus": "代谢结论",
        "assert": "{v}，主要软代谢位点的氧化比例明显收窄",
    },
]

#: 中性案例内容形容词池（不含任何矛盾关键词）
NEUTRAL_ADJECTIVES: list[str] = [
    "理想", "平稳", "良好", "可接受", "稳定",
    "符合预期", "有待观察", "仍需复核", "清晰", "一致",
]


# ============================================================================
# 数据类
# ============================================================================

@dataclass
class EvidenceDict:
    """
    单条证据的可序列化字典表示（字段与 consensus_court.Evidence 一一对应）。

    Attributes:
        evidence_id: 证据全局唯一 ID（案例内唯一）
        source: 来源谱系键（pubmed/chembl/.../unknown）
        content: 证据正文（中文模板句，可能含或不含矛盾关键词）
        role: 提交证据的虚拟实验室角色
        timestamp: ISO 8601 字符串或 None（制造时效性差异，供 EvidenceScorer
            的 recency 维度真实计算）
        reference_ids: 引用 ID 列表（可为空列表 → 触发"缺少可验证引用"类缺陷）
        metadata: 附加元数据（含生成时的类别与缺陷标记，便于审计）
    """

    evidence_id: str
    source: str
    content: str
    role: str
    timestamp: Optional[str]
    reference_ids: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class DdbCase:
    """
    DDB-L1 单个合成争议案例。

    Attributes:
        case_id: 案例编号（DDB-L1-0001 形式）
        topic: 药物发现议题短语
        polarity: "with_contradiction" / "no_contradiction"
        category: 矛盾类别名（无矛盾案例为 "neutral"）
        evidence_type: clean / short_content / missing_reference /
            short_and_missing_reference
        defect_flags: 注入的形式缺陷清单（short_content/missing_reference/
            unknown_source）
        evidences: 证据条目列表
        ground_truth: 真值标签（has_true_contradiction / defect_only）
    """

    case_id: str
    topic: str
    polarity: str
    category: str
    evidence_type: str
    defect_flags: list[str]
    evidences: list[EvidenceDict]
    ground_truth: dict

    def to_dict(self) -> dict:
        """转为可 JSON 序列化的普通字典（dataclass 深拷贝语义由 asdict 提供）。"""
        return asdict(self)


# ============================================================================
# 辅助函数
# ============================================================================

def _keyword_conflict(text: str, keywords) -> bool:
    """
    判断 text 是否同时命中某组矛盾关键词的两个关键词（与源码双向匹配口径一致）。

    用于类别模板与中性模板的编译期自检：同一句话里若同时含一对矛盾词，
    会造成无矛盾案例被真实 ABP 误判，必须在生成阶段杜绝。

    Args:
        text: 待检查文本
        keywords: 关键词序列（元组/列表）

    Returns:
        True 表示存在同时命中的关键词对
    """
    lowered = text.lower()
    for i, a in enumerate(keywords):
        for b in keywords[i + 1:]:
            if a in lowered and b in lowered:
                return True
    return False


def _build_all_keyword_table() -> list[tuple[str, str]]:
    """
    返回运行时权威关键词对表（直接取自源码常量，不做任何本地修改）。

    Raises:
        RuntimeError: 若源码关键词对数量不是预期的 8 对（论文附表需要同步更新）
    """
    pairs = list(AsymmetricRubric.CONTRADICTION_KEYWORD_PAIRS)
    if len(pairs) != 8:
        raise RuntimeError(
            f"AsymmetricRubric.CONTRADICTION_KEYWORD_PAIRS 数量为 {len(pairs)}，"
            f"与本生成器假设的 8 类不符，请对照源码更新类别表"
        )
    return pairs


def _assert_category_alignment(pairs: list[tuple[str, str]]) -> None:
    """
    校验 8 个类别的关键词组与源码关键词表双向兼容：
    每个类别的 (a, b) 必须作为某一组的双向兼容对存在于源码表中
    （即 (a,b) 或 (b,a) 恰好等于某一组），防止源码演进后基准悄然失效。
    """
    for cat in CONTRADICTION_CATEGORIES:
        a, b = cat["keywords"]
        if (a, b) not in pairs and (b, a) not in pairs:
            raise RuntimeError(
                f"类别 {cat['name']} 的关键词对与源码 CONTRADICTION_KEYWORD_PAIRS 不匹配"
            )


def _verify_no_contradiction(content: str, pairs: list[tuple[str, str]]) -> None:
    """
    逐对校验文本不含任何"双向命中"（单侧关键词允许出现，如类别内部模板；
    但同一文本同时含一对矛盾词的两方即报错）。
    """
    for kw_a, kw_b in pairs:
        if kw_a in content and kw_b in content:
            raise AssertionError(
                f"文本同时命中矛盾对 ({kw_a}, {kw_b})，违反模板无矛盾约束: {content[:60]}"
            )


def _normalize_content(content: str) -> str:
    """
    把内容长度规范化到 >= 10（strip 后），确保"内容过短"缺陷不会
    误伤非 short_content 案例（RuleEngine.screen 阈值为 strip 后长度 < 10）。

    实现：内容偏短时在句尾追加中性修饰语（不含任何矛盾关键词）补足长度。
    """
    padding = "，复核意见与组内既有结论保持一致"
    while len(content.strip()) < 12:
        content = content + padding
    return content


# ============================================================================
# 生成器核心
# ============================================================================

class DdbL1Generator:
    """
    DDB-L1 合成争议基准生成器。

    通过固定 seed 的 random.Random 实例保证完全可复现；所有案例在生成时
    即时自检（矛盾案例必含跨来源矛盾对、无矛盾案例必不含双向命中），
    自检失败立即抛异常而不是静默产出坏样本。
    """

    def __init__(self, seed: int = 42):
        """
        Args:
            seed: 随机种子（写入报告，保证实验可复现）
        """
        self._rng = random.Random(seed)
        self._pairs = _build_all_keyword_table()
        _assert_category_alignment(self._pairs)
        self._case_seq = 0

    # ------------------------------------------------------------------
    # 证据级构造
    # ------------------------------------------------------------------

    def _sample_source(self) -> str:
        """按权重从来源谱系采样一个来源键。"""
        return self._rng.choices(
            population=list(SOURCE_POOL.keys()),
            weights=list(SOURCE_POOL.values()),
            k=1,
        )[0]

    def _sample_timestamp(self) -> Optional[str]:
        """
        采样证据时间戳：约 1/3 为 None（时效缺省，EvidenceScorer 记 0.5 分），
        其余均匀落在距今 0-6 年（1826 天）内，制造真实的 recency 分数梯度。
        """
        if self._rng.random() < 0.33:
            return None
        age_days = self._rng.randint(0, 6 * 365)
        ts = datetime.utcnow() - timedelta(days=age_days)
        return ts.isoformat()

    def _make_reference_ids(self, missing: bool) -> list[str]:
        """
        生成引用 ID 列表。

        Args:
            missing: True → 返回空列表（制造"缺少可验证引用"的形式缺陷前提，
                是否真正触发缺陷取决于来源是否为 agent_opinion）
        """
        if missing:
            return []
        prefix = self._rng.choice(["PMID", "CHEMBL", "UNI", "DRUGBANK", "EXP"])
        return [f"{prefix}-{self._rng.randint(100000, 9999999)}"]

    def _make_evidence(
        self,
        case_id: str,
        index: int,
        content: str,
        source: Optional[str] = None,
        missing_reference: bool = False,
        category: str = "",
    ) -> EvidenceDict:
        """
        组装单条证据（自动补长度、采样角色与时间戳）。

        Args:
            case_id: 所属案例编号（用于 evidence_id 唯一性）
            index: 案例内序号
            content: 证据正文（模板生成，缺陷补丁在此之外完成）
            source: 指定来源；None 时按权重采样
            missing_reference: 是否刻意留空引用列表
            category: 生成类别（写入 metadata 供审计）

        Returns:
            EvidenceDict
        """
        content = _normalize_content(content)
        return EvidenceDict(
            evidence_id=f"{case_id}-EV{index:02d}",
            source=source or self._sample_source(),
            content=content,
            role=self._rng.choice(ROLE_POOL),
            timestamp=self._sample_timestamp(),
            reference_ids=self._make_reference_ids(missing_reference),
            metadata={"category": category},
        )

    # ------------------------------------------------------------------
    # 有矛盾案例
    # ------------------------------------------------------------------

    def _make_contradiction_case(self, forced_cat: dict, form_defect: Optional[str]) -> DdbCase:
        """
        生成一个有矛盾案例。

        构造保证：
        1. 恰好 2 条"矛盾核心证据"，关键词分属 **不同来源**
           （源码 _find_contradictions 跳过同源证据对，必须跨来源才可被检出）；
        2. 其余干扰证据（1-3 条）：中性模板句 或 与核心证据同侧关键词句，
           不额外注入缺陷（缺陷作为噪声按 form_defect 只注入干扰证据，
           绝不污染核心矛盾证据，保证召回子集完整）；
        3. form_defect 从 {None, short_content, missing_reference,
           short_and_missing_reference} 采样，注入方式为：
           - short_content → 把选中干扰证据内容替换为 <10 字符短句
           - missing_reference → 把选中干扰证据改为 agent_opinion 且清空引用
             （命中 RuleEngine.screen 第 3 条有罪推定规则）
           - 二者叠加 → 两条干扰证据各注入一种

        Args:
            forced_cat: 外部 round-robin 循环指定的矛盾类别（保证 8 类均衡）
            form_defect: 缺陷类型（None 表示干净案例）

        Returns:
            DdbCase（polarity=with_contradiction）
        """
        self._case_seq += 1
        case_id = f"DDB-L1-{self._case_seq:04d}"
        cat = forced_cat
        a_kw, b_kw = cat["keywords"]

        # 两条核心证据必须来自不同来源（保证跨来源矛盾对被源码检测到）
        source_a, source_b = self._rng.sample(list(SOURCE_POOL.keys()), 2)
        core_a = self._make_evidence(
            case_id, 1,
            content=f"{cat['subject']}{cat['a'].format(kw=a_kw)}",
            source=source_a, category=cat["name"],
        )
        core_b = self._make_evidence(
            case_id, 2,
            content=f"{cat['subject']}{cat['b'].format(kw=b_kw)}",
            source=source_b, category=cat["name"],
        )
        evidences: list[EvidenceDict] = [core_a, core_b]

        # 干扰证据：约一半用同类别同侧关键词句，一半用中性句（不含任何矛盾词）
        for idx in range(3, self._rng.randint(4, 6)):
            if self._rng.random() < 0.5:
                side_key = self._rng.choice(["a", "b"])
                side_kw = a_kw if side_key == "a" else b_kw
                text = f"{cat['subject']}{cat[side_key].format(kw=side_kw)}"
            else:
                text = self._neutral_sentence()
            evidences.append(
                self._make_evidence(case_id, idx, content=text, category=cat["name"])
            )

        # 按 form_defect 注入形式缺陷（只污染干扰证据，不破坏矛盾对的完整性）
        distractors = evidences[2:]
        if form_defect in ("short_content", "short_and_missing_reference") and distractors:
            victim = self._rng.choice(distractors)
            victim.content = "数据待补充。"  # 6 字符 → 触发"证据内容过短"
        if form_defect in ("missing_reference", "short_and_missing_reference"):
            # 选一条尚未注入短缺陷的干扰证据；若无干扰证据可选则跳过
            # （干扰证据数 randint(4,6) 保证至少 1 条，理论上不会走到 or 分支）
            candidates = [ev for ev in distractors if ev.content != "数据待补充。"]
            if candidates or distractors:
                victim = self._rng.choice(candidates or distractors)
                victim.source = "agent_opinion"
                victim.reference_ids = []
            else:
                # 无干扰证据时改在核心证据 B 上注入（不影响矛盾对关键词本身）
                core_b.source = "agent_opinion"
                core_b.reference_ids = []

        defect_flags: list[str] = []
        if form_defect in ("short_content", "short_and_missing_reference"):
            defect_flags.append("short_content")
        if form_defect in ("missing_reference", "short_and_missing_reference"):
            defect_flags.append("missing_reference")

        case = DdbCase(
            case_id=case_id,
            topic=cat["topic"],
            polarity="with_contradiction",
            category=cat["name"],
            evidence_type=form_defect or "clean",
            defect_flags=defect_flags,
            evidences=evidences,
            ground_truth={"has_true_contradiction": True, "defect_only": False},
        )
        self._selfcheck_contradiction_case(case)
        return case

    # ------------------------------------------------------------------
    # 无矛盾案例
    # ------------------------------------------------------------------

    def _neutral_sentence(self) -> str:
        """
        生成一句中性证据句（模板 + 形容词池组合），生成后立即校验：
        1) 单句内不得双向命中任何矛盾对；
        2) 单句内不得出现任何矛盾对的任一侧关键词
           （更严格，防止"无矛盾"定义泄漏争议语义，测试 1 按此口径断言）。
        """
        nt = self._rng.choice(NEUTRAL_TOPICS)
        adj = self._rng.choice(NEUTRAL_ADJECTIVES)
        sentence = f"{nt['subject']}{nt['assert'].format(v=adj)}"
        all_keywords = [kw for pair in self._pairs for kw in pair]
        hit = [kw for kw in all_keywords if kw in sentence]
        if hit:
            raise AssertionError(
                f"中性模板句出现了矛盾关键词侧 {hit}，违反无矛盾约束: {sentence}"
            )
        _verify_no_contradiction(sentence, self._pairs)
        return sentence

    def _make_clean_neutral_case(self) -> DdbCase:
        """
        生成"干净"无矛盾案例（defect_only=False）：
        3-5 条证据全部中性句、来源在已知谱系内、引用齐全。
        刻意保证不命中 RuleEngine.screen 的三条有罪推定规则：
        内容 >= 10 字符、来源非空、agent_opinion 也带引用。
        """
        self._case_seq += 1
        case_id = f"DDB-L1-{self._case_seq:04d}"
        nt = self._rng.choice(NEUTRAL_TOPICS)
        evidences: list[EvidenceDict] = []
        for idx in range(1, self._rng.randint(4, 6)):
            ev = self._make_evidence(
                case_id, idx, content=self._neutral_sentence(), category="neutral",
            )
            # 来源限定在已知谱系（避免 unknown），引用默认齐全（缺陷在此外注入）
            if ev.source == "unknown":
                ev.source = self._sample_source()
            if not ev.reference_ids:
                ev.reference_ids = self._make_reference_ids(missing=False)
            evidences.append(ev)
        case = DdbCase(
            case_id=case_id,
            topic=nt["topic"],
            polarity="no_contradiction",
            category="neutral",
            evidence_type="clean",
            defect_flags=[],
            evidences=evidences,
            ground_truth={"has_true_contradiction": False, "defect_only": False},
        )
        self._selfcheck_clean_case(case)
        return case

    def _make_defect_only_case(self) -> DdbCase:
        """
        生成"仅形式缺陷"无矛盾案例（defect_only=True，H2a 假阳性核心测集）：
        内容全部中性句（无任何矛盾关键词），但至少注入一种命中
        RuleEngine.screen 有罪推定的形式缺陷，且缺陷证据的内容本身无矛盾：

        - short_content：某条证据内容替换为 <10 字符短句（含争议主题词但无矛盾词）
        - missing_reference：某条证据改为 agent_opinion 且清空引用
        - unknown_source：先在一条中性句证据上记 unknown_source 缺陷计数，
          再把来源替换为谱系外的 "unknown"（screen() 不拦截、评分器记 0.2 低分）。
          这是与规格的适配点：源码 screen() 只在来源为 **空** 时报缺陷，
          为严守"无矛盾案例不得含任何矛盾关键词"，无法用空来源表达该缺陷，
          故以谱系外来源表达"来源异常"的形式缺陷语义。
        """
        self._case_seq += 1
        case_id = f"DDB-L1-{self._case_seq:04d}"
        nt = self._rng.choice(NEUTRAL_TOPICS)
        evidences: list[EvidenceDict] = []
        for idx in range(1, self._rng.randint(4, 6)):
            ev = self._make_evidence(
                case_id, idx, content=self._neutral_sentence(), category="neutral",
            )
            if not ev.reference_ids:
                ev.reference_ids = self._make_reference_ids(missing=False)
            evidences.append(ev)

        # 至少保证一种形式缺陷；三种缺陷各自以独立概率叠加注入
        defect_flags: list[str] = []
        roll = self._rng.random()
        inject_short = roll < 0.55
        inject_missing = not inject_short or self._rng.random() < 0.45
        inject_unknown = self._rng.random() < 0.30

        victim_pool = list(evidences)
        self._rng.shuffle(victim_pool)

        if inject_missing:
            victim = victim_pool.pop(0)
            victim.source = "agent_opinion"
            victim.reference_ids = []
            defect_flags.append("missing_reference")
        if inject_short and victim_pool:
            victim = victim_pool.pop(0)
            # 短缺陷句里带议题主体词但保证无任何矛盾关键词（模板已自检过 subject 安全）
            short_text = f"{nt['subject']}数据暂缺，待复核。"
            all_keywords = [kw for pair in self._pairs for kw in pair]
            if any(kw in short_text for kw in all_keywords):
                short_text = "数据暂缺，待复核。"
            victim.content = short_text[:9] if len(short_text) >= 10 else short_text
            # 截断仅用于确保 <10 字符，两种文本 strip 后都远小于 10
            if len(victim.content.strip()) >= 10:
                victim.content = "数据暂缺。"
            defect_flags.append("short_content")
        if inject_unknown and victim_pool:
            victim = victim_pool.pop(0)
            victim.source = "unknown"
            # 保留其引用与中性内容：形式缺陷 = 谱系外来源
            defect_flags.append("unknown_source")

        # 兜底：极端随机组合下若一个缺陷都没注入，强制注入 missing_reference
        if not defect_flags:
            victim = evidences[0]
            victim.source = "agent_opinion"
            victim.reference_ids = []
            defect_flags.append("missing_reference")

        evidence_type = "short_and_missing_reference" if (
            "short_content" in defect_flags and "missing_reference" in defect_flags
        ) else defect_flags[0]

        case = DdbCase(
            case_id=case_id,
            topic=nt["topic"],
            polarity="no_contradiction",
            category="neutral",
            evidence_type=evidence_type,
            defect_flags=defect_flags,
            evidences=evidences,
            ground_truth={"has_true_contradiction": False, "defect_only": True},
        )
        self._selfcheck_clean_case(case)
        return case

    # ------------------------------------------------------------------
    # 自检（生成期即时校验，防止基准泄漏）
    # ------------------------------------------------------------------

    def _selfcheck_contradiction_case(self, case: DdbCase) -> None:
        """
        有矛盾案例自检：必须存在一对 **不同来源** 的证据，其内容双向命中
        同一组矛盾关键词（与源码 _find_contradictions/_check_pair_contradiction
        的判定口径逐条对齐）。
        """
        evs = case.evidences
        found = False
        for i, ea in enumerate(evs):
            for eb in evs[i + 1:]:
                if ea.source == eb.source:
                    continue
                ca, cb = ea.content.lower(), eb.content.lower()
                for kw_a, kw_b in self._pairs:
                    if (kw_a in ca and kw_b in cb) or (kw_b in ca and kw_a in cb):
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if not found:
            raise AssertionError(f"案例 {case.case_id} 未注入跨来源矛盾对，生成器逻辑异常")

    def _selfcheck_clean_case(self, case: DdbCase) -> None:
        """
        无矛盾案例自检：所有证据内容不得出现任何矛盾对的 **任一侧** 关键词
        （严于源码的双向命中口径——保证无矛盾案例彻底不沾染争议语义关键词）。
        """
        all_keywords = [kw for pair in self._pairs for kw in pair]
        for ev in case.evidences:
            hit = [kw for kw in all_keywords if kw in ev.content]
            if hit:
                raise AssertionError(
                    f"无矛盾案例 {case.case_id} 证据 {ev.evidence_id} 含矛盾关键词 {hit}"
                )

    # ------------------------------------------------------------------
    # 批量生成
    # ------------------------------------------------------------------

    def generate(
        self,
        num_with_contradiction: int = 300,
        num_no_contradiction: int = 300,
    ) -> list[dict]:
        """
        批量生成 DDB-L1 基准案例。

        采样策略：
        - 8 个矛盾类别轮转（round-robin）后再随机打乱，保证类别均衡；
        - 有矛盾案例的形式缺陷类型按 [clean×2, short, missing, both] 比例循环，
          约 3/5 带形式缺陷（模拟真实噪声，同时召回子集仍由矛盾对保证）；
        - 无矛盾案例前一半为 defect_only=True（形式缺陷测假阳性），
          后一半为完全干净案例。

        Args:
            num_with_contradiction: 有矛盾案例数
            num_no_contradiction: 无矛盾案例数

        Returns:
            案例字典列表（schema 见模块 docstring）
        """
        cases: list[DdbCase] = []

        # ---- 有矛盾案例 ----
        categories_cycle = CONTRADICTION_CATEGORIES * (
            num_with_contradiction // len(CONTRADICTION_CATEGORIES) + 1
        )
        defect_cycle: list[Optional[str]] = [
            None, None, "short_content", "missing_reference",
            "short_and_missing_reference",
        ]
        # 记录类别随机选择被 _make_contradiction_case 内部消耗，为保证类别
        # round-robin 精确生效，这里显式覆盖 cat 采样：按循环序指定类别。
        for i in range(num_with_contradiction):
            forced_cat = categories_cycle[i]
            form_defect = defect_cycle[i % len(defect_cycle)]
            case = self._make_contradiction_case(forced_cat, form_defect)
            cases.append(case)

        # ---- 无矛盾案例：一半 defect_only ----
        num_defect = num_no_contradiction // 2
        for _ in range(num_defect):
            cases.append(self._make_defect_only_case())
        for _ in range(num_no_contradiction - num_defect):
            cases.append(self._make_clean_neutral_case())

        return [c.to_dict() for c in cases]

    # ------------------------------------------------------------------
    # 全局完整性校验
    # ------------------------------------------------------------------

    @staticmethod
    def validate(cases: list[dict]) -> None:
        """
        对生成结果做最终全量校验（测试与 main 均可复用）：
        1. case_id 全局唯一；
        2. 有矛盾案例必含跨来源矛盾对（用源码真实关键词表复检）；
        3. 无矛盾案例所有证据必不含任何矛盾关键词侧；
        4. defect_only 案例必带至少一个 defect_flag；
        5. 所有证据 strip 后非空。
        """
        pairs = list(AsymmetricRubric.CONTRADICTION_KEYWORD_PAIRS)
        all_keywords = [kw for pair in pairs for kw in pair]
        seen_ids: set[str] = set()
        for case in cases:
            cid = case["case_id"]
            if cid in seen_ids:
                raise AssertionError(f"case_id 重复: {cid}")
            seen_ids.add(cid)
            evs = case["evidences"]
            if case["polarity"] == "with_contradiction":
                found = any(
                    ea["source"] != eb["source"]
                    and any(
                        (a in ea["content"] and b in eb["content"])
                        or (b in ea["content"] and a in eb["content"])
                        for a, b in pairs
                    )
                    for i, ea in enumerate(evs) for eb in evs[i + 1:]
                )
                if not found:
                    raise AssertionError(f"{cid}: 有矛盾案例缺少跨来源矛盾对")
            else:
                for ev in evs:
                    if any(kw in ev["content"] for kw in all_keywords):
                        raise AssertionError(
                            f"{cid}/{ev['evidence_id']}: 无矛盾案例证据含矛盾关键词侧"
                        )
                if case["ground_truth"]["defect_only"] and not case["defect_flags"]:
                    raise AssertionError(f"{cid}: defect_only 案例缺少 defect_flags")
            for ev in evs:
                if not ev["content"] or not ev["content"].strip():
                    raise AssertionError(f"{cid}/{ev['evidence_id']}: 证据内容为空")


# ============================================================================
# 入口
# ============================================================================

def generate(
    num_with_contradiction: int = 300,
    num_no_contradiction: int = 300,
    seed: int = 42,
) -> list[dict]:
    """
    模块级便捷入口：以指定规模与种子生成 DDB-L1 基准。

    Args:
        num_with_contradiction: 有矛盾案例数（默认 300）
        num_no_contradiction: 无矛盾案例数（默认 300，其中一半 defect_only）
        seed: 随机种子（默认 42）

    Returns:
        案例字典列表（已通过 DdbL1Generator.validate 全量校验）
    """
    gen = DdbL1Generator(seed=seed)
    cases = gen.generate(
        num_with_contradiction=num_with_contradiction,
        num_no_contradiction=num_no_contradiction,
    )
    DdbL1Generator.validate(cases)
    return cases


def main() -> None:
    """
    命令行入口：生成默认 600 例并写入 tests/ddb/l1_cases.json。

    运行方式：cd backend && python -m tests.ddb.l1_generator
    """
    out_path = Path(__file__).resolve().parent / "l1_cases.json"
    cases = generate()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    n_with = sum(1 for c in cases if c["polarity"] == "with_contradiction")
    n_defect = sum(1 for c in cases if c["ground_truth"]["defect_only"])
    print(
        f"DDB-L1 生成完成：共 {len(cases)} 例"
        f"（有矛盾 {n_with} / 无矛盾 {len(cases) - n_with}，其中 defect_only {n_defect}）"
        f" → {out_path}"
    )


if __name__ == "__main__":
    main()
