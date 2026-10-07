"""
RubricEngine - 非对称举证审查 rubric 引擎

P0-2 实现：把审查从"有罪推定（内容过短直接拒）"改为
"非对称举证（找到矛盾才定罪）+ 按证据位置加权 + drill before disposition"。

v0.3 扩展（论文大纲 §4.4 规划项落地，吸收 Kumar et al. 2026 IMPACT 设计）：
1. Aspect 约束：矛盾词对条件化为 (aspect, 词对) 三元组，跨切面伪冲突
   （如"合成可行性差" vs "靶点验证充分"）不再定罪；UNKNOWN 通配兜底，
   门禁只降 FPR 不降召回（ABP 保守方向）。
2. ValidityGate（CVG 等价物）：候选矛盾对需通过对象同一性轻量校验
   （同一靶点/分子/位点），三态语义：SAME/UNKNOWN 放行定罪、
   DIFFERENT 拦截、AMBIGUOUS 降为待举证（pending_pairs）。
3. 矛盾强度分级：intensity ∈ {1 轻微/2 中度/3 对抗}，纯函数规则可审计，
   喂给 §4.7 升级条款 5（intensity=3 强制人工复核）做分级分诊。

三项机制各有独立开关（enable_aspect / enable_validity_gate /
enable_intensity），任一关闭即精确回退 v0.2 对应行为（§6.4 消融自变量）；
子串极性修复（"不存在"含"存在"陷阱）始终生效——它修复的是正确性缺陷，
不属于可选机制。

核心组件：
1. AsymmetricRubric：非对称举证 rubric，只有找到矛盾证据才定罪
2. EvidencePositionWeighter：按证据在回答中的位置加权
3. DrillBeforeDisposition：先收集证据再下结论，避免过早判定

设计哲学（来自 Claude Science "追溯而非重算"）：
- 追溯而非重算：基于已有证据追溯矛盾，而非重新计算"正确答案"
- 产出物而非答案：审查产出物（证据矛盾对）而非直接给答案
- 非对称举证：指控方（审查者）需举证矛盾，被控方（Agent）默认无罪

数据流：
    consensus_court.RuleEngine.screen() → 标记问题（不下结论）
    → AsymmetricRubric.evaluate() → 矛盾对流水线（P1 词对极性 → P2 aspect
      存活 → P3 ValidityGate → P4 强度分级 → P5 组装）
    → DrillBeforeDisposition → 收集所有证据后下结论
    → EvidencePositionWeighter → 按位置加权最终置信度
"""
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from app.core.virtual_lab.consensus_court import Evidence, EvidenceScore

logger = logging.getLogger(__name__)


# ============================================================================
# 严重等级枚举（P0-2 统一定义）
# ============================================================================

class Severity(str, Enum):
    """审查问题严重等级"""

    CRITICAL = "CRITICAL"  # 严重：确认的矛盾/幻觉，必须拒绝
    WARNING = "WARNING"    # 警告：潜在问题，需人工复核
    INFO = "INFO"          # 提示：轻微问题，记录但不影响通过
    PENDING = "PENDING"    # 待举证：非对称举证标记，默认通过但需补充证据


class Aspect(str, Enum):
    """
    语义切面（aspect）——药物发现证据的语义域（v0.3 Aspect 约束）。

    矛盾关键词对只在同一 aspect 内激活；UNKNOWN 是兜底切面，
    表示"词典无法判定该证据属于哪个语义域"（非"无切面"），
    匹配语义为通配（任一方 UNKNOWN → aspect 判据退化为恒真）。
    与论文大纲 §4.4（Kumar et al., arXiv 2605.10171 aspect 条件化）对齐。
    """

    ACTIVITY_DIRECTION = "活性方向"       # 激动/拮抗/激酶活性/耐药/结合
    EXPRESSION_REGULATION = "表达调控"    # 上调/下调/高表达/低表达/转录
    POTENCY = "效价"                      # ORR/PFS/OS/疗效端点/临床应答
    DRUGGABILITY = "成药性"               # 溶解度/PK/偶联工艺/制剂稳定性
    SAFETY = "安全性"                     # 毒性/irAE/DLT/耐受性
    STRUCTURAL_CONFIDENCE = "结构可信度"  # AF3/对接/冷冻电镜/姿态精度
    MECHANISM = "机制"                    # 通路方向/依赖关系/旁观者效应
    UNKNOWN = "未知切面"                  # 兜底：词典零命中


class GateDecision(str, Enum):
    """
    ValidityGate 三态判定（v0.3 有效性门禁，CVG 等价物）。

    UNKNOWN（无对象信号）与 AMBIGUOUS（有对象但指代冲突）严格区分：
    前者放行定罪，后者降为待举证不定罪。
    """

    SAME_OBJECT = "SAME_OBJECT"              # 同一对象（同靶点/分子/位点）
    DIFFERENT_OBJECT = "DIFFERENT_OBJECT"    # 证实为不同对象 → 拦截不定罪
    AMBIGUOUS = "AMBIGUOUS"                  # 指代冲突 → 降为待举证
    UNKNOWN = "UNKNOWN"                      # 至少一方无对象信号 → 放行


# ============================================================================
# 数据类
# ============================================================================

@dataclass
class GateResult:
    """
    ValidityGate 门禁完整产出（v0.3，随矛盾对留痕供审计）。

    Attributes:
        decision: 三态判定结果
        object_a / object_b: 双方抽取到的规范对象键列表（供 gate_detail 追溯）
        matched_keys: 判定 SAME 时的共享键依据
        detail: 人读理由（审计日志与裁决 payload 用）
    """

    decision: GateDecision = GateDecision.UNKNOWN
    object_a: list[str] = field(default_factory=list)
    object_b: list[str] = field(default_factory=list)
    matched_keys: list[str] = field(default_factory=list)
    detail: str = ""


@dataclass
class ContradictionPair:
    """
    矛盾证据对

    当两条证据在同一主题上给出相互矛盾的结论时，记录为矛盾对。
    这是非对称举证中"定罪"的唯一依据。

    v0.3 扩展字段全部带默认值并追加在末尾：v0.2 的四参位置构造
    （evidence_a_id, evidence_b_id, topic, contradiction_desc）保持可用，
    既有调用方与测试零改动。

    Attributes:
        evidence_a_id: 证据 A 的 ID
        evidence_b_id: 证据 B 的 ID
        topic: 矛盾主题
        contradiction_desc: 矛盾描述（v0.3 为强度 rationale 模板产物，
            仍保证包含 topic 与两侧关键词——既有断言依赖）
        severity: 严重等级（CRITICAL 表示确认矛盾；强度是正交维度）
        intensity: 矛盾强度（v0.3）：1 轻微（程度分歧）/2 中度（方向存疑）/
            3 对抗（方向直接相反）；enable_intensity=False 时恒为 2
        matched_aspect: 词对存活切面集合的规范代表（开关关闭时为 None）
        shared_aspects: 全部存活切面（审计用）
        gate_detail: ValidityGate 完整产出（门禁关闭时为 None）
        matched_keyword_pair: 命中的词对两侧词（v0.3.1：desc 模板已不再
            嵌入两侧词，故结构化保留供审计、测试与下游摘要使用；
            开关全关的经典路径不填此字段，保持 v0.2 行为）
    """

    evidence_a_id: str
    evidence_b_id: str
    topic: str
    contradiction_desc: str
    severity: Severity = Severity.CRITICAL
    intensity: int = 2
    matched_aspect: Optional[Aspect] = None
    shared_aspects: list[Aspect] = field(default_factory=list)
    gate_detail: Optional[GateResult] = None
    matched_keyword_pair: Optional[tuple[str, str]] = None


@dataclass
class RubricVerdict:
    """
    非对称举证 rubric 裁决结果

    Attributes:
        passed: 是否通过审查（True=默认通过，False=找到矛盾定罪）
        contradictions: 找到的矛盾证据对列表（空表示无矛盾）
        pending_issues: 待举证问题列表（非对称举证标记，不影响通过）
        confidence: 置信度（0-1，受证据位置加权影响）
        rationale: 裁决理由
        pending_pairs: ValidityGate 判 AMBIGUOUS 而降级的矛盾候选对
            （v0.3：不进 contradictions、不影响 passed，供离线 FPR/
            AMBIGUOUS 率统计与人工复核）
    """

    passed: bool
    contradictions: list[ContradictionPair] = field(default_factory=list)
    pending_issues: list[str] = field(default_factory=list)
    confidence: float = 0.0
    rationale: str = ""
    pending_pairs: list[ContradictionPair] = field(default_factory=list)


# ============================================================================
# v0.3 机制基础设施：词对三元组 / 极性拆分 / Aspect 推断 / ValidityGate / 强度分级
# ============================================================================

#: 矛盾词对的语义极性（强度分级基线分的唯一来源）
POLARITY_DIRECTIONAL = "directional"  # 方向对立（激活/抑制）→ 基线 2
POLARITY_DEGREE = "degree"            # 程度分歧 → 基线 1

#: 经典（v0.2）矛盾描述模板——开关全关或词对未激活新机制时逐字使用，
#: 保证既有断言（desc 含两侧关键词与 topic）零改动通过。
CLASSIC_DESC_TEMPLATE = (
    "证据 {a_id} 含 '{ka}'，证据 {b_id} 含 '{kb}'，在主题 '{topic}' 上存在矛盾"
)


@dataclass(frozen=True)
class KeywordPairSpec:
    """
    矛盾词对的 v0.3 元数据（K 从二元词对重构为 (aspect, 词对) 三元组）。

    Attributes:
        keyword_a: 词对 A 侧（肯定侧）
        keyword_b: 词对 B 侧（A 的否定/对立形）
        negated_b: keyword_b 是否为 keyword_a 的否定形（决定分侧算法是否
            剥离该重叠出现；"不存在"→"存在" 即此类）
        aspects: 该词对合法的语义切面集合（词对在这些切面内可激活）
        polarity: 语义极性（directional/degree），强度分级基线分来源
        hard_constraint: 命中即 CRITICAL/intensity=3 的硬约束标记
            （词对路径 S1 短路预留，当前全部 False；AF2 类系统级禁令
            已通过 HARD_CONSTRAINT_PATTERNS 前置扫描独立通道激活，
            见 §4.4 实现注记 v0.4.7）
    """

    keyword_a: str
    keyword_b: str
    negated_b: bool
    aspects: frozenset[Aspect]
    polarity: str
    hard_constraint: bool = False


def _all_aspects() -> frozenset[Aspect]:
    """全部切面集合（通用谓词词对使用，等价"恒允许激活"）。"""
    return frozenset(Aspect)


#: v0.3 词对规格表：词对内容与顺序和 v0.2 CONTRADICTION_KEYWORD_PAIRS
#: 完全一致（l1_generator 的 8 元组解包与现有参数化测试依赖该顺序）。
KEYWORD_PAIR_SPECS: tuple[KeywordPairSpec, ...] = (
    # "有效/无效"：全切面通用谓词（任何语义域都谈有效性）
    KeywordPairSpec("有效", "无效", True, _all_aspects(), POLARITY_DIRECTIONAL),
    KeywordPairSpec("成功", "失败", False,
                    frozenset({Aspect.DRUGGABILITY, Aspect.POTENCY, Aspect.SAFETY,
                               Aspect.UNKNOWN}), POLARITY_DIRECTIONAL),
    KeywordPairSpec("阳性", "阴性", False,
                    frozenset({Aspect.POTENCY, Aspect.ACTIVITY_DIRECTION,
                               Aspect.SAFETY, Aspect.UNKNOWN}), POLARITY_DIRECTIONAL),
    KeywordPairSpec("激活", "抑制", False,
                    frozenset({Aspect.ACTIVITY_DIRECTION, Aspect.EXPRESSION_REGULATION,
                               Aspect.MECHANISM}), POLARITY_DIRECTIONAL),
    KeywordPairSpec("存在", "不存在", True,
                    frozenset({Aspect.STRUCTURAL_CONFIDENCE, Aspect.DRUGGABILITY,
                               Aspect.MECHANISM, Aspect.SAFETY,
                               Aspect.UNKNOWN}), POLARITY_DIRECTIONAL),
    KeywordPairSpec("高表达", "低表达", False,
                    frozenset({Aspect.EXPRESSION_REGULATION}), POLARITY_DIRECTIONAL),
    KeywordPairSpec("促进", "抑制", False,
                    frozenset({Aspect.EXPRESSION_REGULATION, Aspect.MECHANISM}),
                    POLARITY_DIRECTIONAL),
    KeywordPairSpec("上调", "下调", False,
                    frozenset({Aspect.EXPRESSION_REGULATION}), POLARITY_DIRECTIONAL),
)

#: 否定前缀表（子串极性修复用）："不/未/无/非" 开头的词出现不计入肯定侧。
#: "抗"（抗体语境）为 v0.4 预留，v0.3 不启用（"抗 PD-1" 不是 "PD-1" 的否定）。
NEGATION_PREFIXES = ("不", "未", "无", "非")

#: 强度分级：来源可信度差加成阈值（SOURCE_WEIGHTS 口径，|Δ| ≥ 0.30 时 +1）
SOURCE_GAP_INTENSITY_THRESHOLD = 0.30

#: 结构预测工具禁令（v0.4.8 修正）：运行时工具层已把 AlphaFold 系列
#: （AF2 / AF3）全部系统级禁用，合规降级链为 OpenDDE → Chai1 → Boltz
#: （见 result_interpreter.py L33-34 工具降级链）。证据仲裁层据此对
#: AlphaFold2 与 AlphaFold3 引用**双禁**——任一命中即 CRITICAL / intensity=3。
#:
#: 四条模式（alphafold2/af2/alphafold3/af3）共用统一规则名
#: "structure-tool-ban"：这样一条证据无论命中 AF2 还是 AF3（或两者），
#: _scan_hard_constraints 最多产出一条 CRITICAL（去重见
#: _HARD_CONSTRAINT_FALSE_POSITIVE_EXCLUSIONS 同族去重 + 扫描器 rule_name
#: 跨族去重），避免同一 AlphaFold 引用因多变体名重复计入多条自相矛盾对。
#:
#: 注意：Aspect.STRUCTURAL_CONFIDENCE 触发词典与 OBJECT_ALIASES 仍保留
#: AF3/AlphaFold3 作为【中性切面键/对象键】（用于切面推断与同一性判定，
#: 不触发禁令），与本禁令表方向不同、互不影响——禁令只看 HARD_CONSTRAINT_PATTERNS。
HARD_CONSTRAINT_PATTERNS: list[tuple[str, str, str]] = [
    # (pattern_lower, rule_name, description)
    ("alphafold2", "structure-tool-ban", "检测到 AlphaFold2 引用，违反结构预测工具禁令（合规降级 OpenDDE→Chai1→Boltz）"),
    ("af2",        "structure-tool-ban", "检测到 AF2 引用，违反结构预测工具禁令（合规降级 OpenDDE→Chai1→Boltz）"),
    ("alphafold3", "structure-tool-ban", "检测到 AlphaFold3 引用，违反结构预测工具禁令（合规降级 OpenDDE→Chai1→Boltz）"),
    ("af3",        "structure-tool-ban", "检测到 AF3 引用，违反结构预测工具禁令（合规降级 OpenDDE→Chai1→Boltz）"),
]

#: 硬约束模式误匹配排除词（**同族变体去重**）：简称模式命中后，若同族
#: 全称也出现在文本中则跳过简称——全称模式已代表该变体，避免同一工具
#: 全称/简称各算一次。跨族去重（AF2 与 AF3 同现）由 _scan_hard_constraints
#: 的 rule_name 集合保证（同一禁令族 structure-tool-ban 只产出一条）。
#: 例："AlphaFold2" → alphafold2 命中、af2 因排除词 alphafold2 跳过 → 1 条；
#:     "AlphaFold3 与 AF2 对比" → alphafold3 命中、af2 命中（其同族全称
#:     alphafold2 不在场）→ 但二者 rule_name 同为 structure-tool-ban，
#:     扫描器按 rule_name 去重 → 仅 1 条族级 CRITICAL。
_HARD_CONSTRAINT_FALSE_POSITIVE_EXCLUSIONS: dict[str, list[str]] = {
    # 简称模式在同族全称在场时跳过（"alphafold" ⊂ 两个全称，统一去重）
    "af2": ["alphafold"],
    "af3": ["alphafold"],
}


def _scan_hard_constraints(
    content: str,
) -> list[tuple[str, str, str]]:
    """
    扫描证据内容中的硬约束模式命中。

    匹配口径：对 content 做 lowercase 后逐模式做子串匹配。命中后做两层去重：
      ① 同族变体去重：简称（af2/af3）在同族全称（alphafold2/alphafold3）
         在场时跳过（见 _HARD_CONSTRAINT_FALSE_POSITIVE_EXCLUSIONS）；
      ② 跨族禁令去重：同一 rule_name（structure-tool-ban）只产出一条——
         AF2 与 AF3 同属"结构预测工具禁令族"，单条证据无论含 AF2、AF3
         或两者都只算 1 条 CRITICAL（结构预测工具违规是单一性质）。

    Args:
        content: 证据内容（未预处理，函数内部做 lower()）

    Returns:
        命中列表，每项 (pattern, rule_name, description)；无命中返回空列表
    """
    hits: list[tuple[str, str, str]] = []
    hit_rules: set[str] = set()
    lower = content.lower()
    for pattern, rule_name, desc in HARD_CONSTRAINT_PATTERNS:
        if pattern not in lower:
            continue
        # ① 同族变体去重：简称在同族全称在场时跳过
        exclusions = _HARD_CONSTRAINT_FALSE_POSITIVE_EXCLUSIONS.get(pattern, [])
        if any(excl in lower for excl in exclusions):
            continue
        # ② 跨族禁令去重：同一禁令族（rule_name）只产出一条 CRITICAL
        if rule_name in hit_rules:
            continue
        hit_rules.add(rule_name)
        hits.append((pattern, rule_name, desc))
    return hits


def _polarity_sides(text: str, spec: KeywordPairSpec) -> tuple[set[str], set[str]]:
    """
    把文本按词对拆成（肯定侧命中词集合，否定侧命中词集合）——v0.3 子串修复核心。

    修复目标：v0.2 的 "存在" in "不存在" 子串误配（两侧均写"不存在"时
    经肯定侧"存在"误命中）。规则：对肯定侧词 ka 的每次出现，若该位置
    同时是 keyword_b（negated_b 形，如"不存在"）或任意否定前缀+ka（如
    "未激活"）的对齐起点，则该出现归否定侧，不重复计入肯定侧。

    Args:
        text: 证据内容（调用方已 lower()）
        spec: 词对规格

    Returns:
        (肯定侧词集合, 否定侧词集合)，集合元素固定为 (ka, kb) 中的词，
        空集表示该侧零命中
    """
    affirmative: set[str] = set()
    negated: set[str] = set()

    # kb 命中全部记为否定侧（kb 语义即 ka 的对立面）
    if spec.keyword_b in text:
        negated.add(spec.keyword_b)

    # ka 出现逐位判定：被否定前缀或 negated_b 覆盖的出现不算肯定侧
    start = 0
    ka, kb = spec.keyword_a, spec.keyword_b
    while (pos := text.find(ka, start)) != -1:
        neg_covered = False
        for prefix in NEGATION_PREFIXES:
            p_start = pos - len(prefix)
            if p_start >= 0 and text[p_start:pos] == prefix:
                neg_covered = True
                break
        if not neg_covered and spec.negated_b and kb.startswith(ka):
            # ka 是 kb 的前缀（存在⊂不存在）：kb 的对齐起点即否定起点
            kb_at = pos - (len(kb) - len(ka))
            if kb_at >= 0 and text[kb_at:kb_at + len(kb)] == kb:
                neg_covered = True
        if neg_covered:
            # 被否定覆盖的出现记否定侧词：negated_b 形记 kb（"不存在"）；
            # 非 negated_b 词对（如"未激活"）没有专用否定词，记 ka 本身，
            # 由 _pair_hit_by_polarity 经 neg 集合非空完成否定侧判定
            negated.add(kb if spec.negated_b else ka)
        else:
            affirmative.add(ka)
        start = pos + 1

    return affirmative, negated


def _pair_hit_by_polarity(a: str, b: str, spec: KeywordPairSpec) -> bool:
    """
    单向极性词对命中判定（v0.3，替代 v0.2 双向包含）。

    否定侧语义（子串修复核心）：否定集合非空 ⟺ 该侧存在否定形命中——
    kb 直接出现，或 ka 的出现全部/部分被否定前缀、negated_b 对齐覆盖。
    "不存在"字面包含"存在"，但后者被覆盖后不再计入肯定侧。

    hit ⟺ (肯定(a,ka) ∧ 否定(b)) ∨ (肯定(b,ka) ∧ 否定(a))
    双向语义保留：A 正 B 负、A 负 B 正都算命中；但同侧（同正/同负）不算。

    Args:
        a: 证据 A 内容（lower 后）
        b: 证据 B 内容（lower 后）
        spec: 词对规格

    Returns:
        True 表示词对通道命中
    """
    pos_a, neg_a = _polarity_sides(a, spec)
    pos_b, neg_b = _polarity_sides(b, spec)
    return (spec.keyword_a in pos_a and bool(neg_b)) or \
        (spec.keyword_a in pos_b and bool(neg_a))


#: Aspect 触发词典（术语键；ASCII 键词边界匹配、中文键子串匹配）。
#: 覆盖 DDB-L2 全部金标句的术语/现象词——tests/ddb/test_l2_v03_locks.py
#: 的 L2L-012 锁死该覆盖，任何缺口即红线。
ASPECT_TRIGGER_LEXICON: dict[Aspect, tuple[str, ...]] = {
    Aspect.ACTIVITY_DIRECTION: (
        # 中文术语
        "激动", "拮抗", "抑制剂", "激活剂", "敏感性", "耐药", "酶活", "激酶",
        "磷酸化", "信号通路", "结合口袋", "共价", "细胞毒", "杀伤", "载荷",
        # 靶点/分子/突变术语（L2 主题词全覆盖）
        "EGFR", "EAI045", "osimertinib", "阿法替尼", "afatinib",
        "KRAS", "sotorasib", "adagrasib", "G12C", "Y96D", "C797S", "T790M",
        "L858R", "IC50", "pIC50", "Ki", "Kd",
    ),
    Aspect.EXPRESSION_REGULATION: (
        "高表达", "低表达", "上调", "下调", "转录", "表达", "沉默", "过表达",
        "生物标志物",
        "p53", "TP53", "PFT", "APR-246", "eprenetapopt",
        "HER2", "ERBB2", "CD276", "B7-H3", "c-MET", "CYP3A4", "p21",
        "mRNA", "qPCR",
    ),
    Aspect.POTENCY: (
        "疗效", "有效性", "ORR", "缓解", "无进展", "总生存", "临床获益",
        "疾病稳定", "端点", "应答", "响应", "获益率", "生存",
        "PD-1", "PD-L1", "pembrolizumab", "atezolizumab", "durvalumab",
        "FLAURA", "PFS", "OS", "HR",
    ),
    Aspect.DRUGGABILITY: (
        "溶解度", "透膜性", "膜通透", "口服", "生物利用度", "稳定性",
        "处方", "偶联", "工艺", "制剂", "药代", "半衰期", "批次", "纯化",
        "合成", "路线", "收率", "ADC", "T-DXd", "T-DM1", "DXd", "MMAE",
        "DM1", "DAR", "PEG",
    ),
    Aspect.SAFETY: (
        "毒性", "不良事件", "心脏毒性", "hERG", "肝毒性", "肺毒性", "ILD",
        "剂量限制", "耐受", "细胞因子风暴", "安全性", "风险", "致死",
        "CAR-T", "urelumab", "ATG-101", "irAE", "DLT", "TNF", "IFN",
    ),
    Aspect.STRUCTURAL_CONFIDENCE: (
        "结构预测", "姿态", "构象", "冷冻电镜", "晶体", "对接", "同源建模",
        "骨架", "密度图",
        "AF3", "AlphaFold3", "AlphaFold", "AF-Multimer", "RMSD", "GPCR",
        "分子动力学",
    ),
    Aspect.MECHANISM: (
        "机制", "依赖", "交联", "旁观者效应", "通路", "协同", "靶点验证",
        "遗传学证据", "必需", "选择性优势", "内化", "FcγR", "Fc",
        "4-1BB", "CD137", "bystander", "Cys124",
    ),
}

#: 科学对象别名表（canonical → 变体键，均 ASCII/中英术语）。
#: ValidityGate 的同一性判据：同 canonical 即同对象。
OBJECT_ALIASES: dict[str, tuple[str, ...]] = {
    "EGFR": ("EGFR", "ERBB1", "EAI045", "osimertinib", "阿法替尼", "afatinib",
             "C797S", "T790M", "L858R"),
    "KRAS": ("KRAS", "G12C", "Y96D", "sotorasib", "adagrasib"),
    "TP53": ("p53", "TP53", "PFT", "APR-246", "eprenetapopt", "Cys124"),
    "HER2": ("HER2", "ERBB2"),
    "CD276": ("CD276", "B7-H3"),
    "PDCD1_AXIS": ("PD-1", "PD-L1", "PD1", "PDL1", "pembrolizumab",
                   "atezolizumab", "durvalumab"),
    "AF3": ("AF3", "AlphaFold3", "AlphaFold", "AF-Multimer"),
    "DXd": ("DXd", "T-DXd", "T-DM1", "DS-8201a"),
    "41BB": ("4-1BB", "CD137", "urelumab", "ATG-101"),
    "EZH2": ("EZH2", "tazemetostat", "SHR2554", "西达本胺"),
}

#: 现象词典（现象词 → 宿主分子键）：L2 存在性争议大量讨论"抽象效应"
#: 而非分子本身；宿主键用于现象同名但语境不同时交叉佐证。
PHENOMENON_LEXICON: dict[str, tuple[str, ...]] = {
    "旁观者效应": ("DXd",),
    "bystander": ("DXd",),
    "FcγR 交联": ("41BB",),
    "选择性优势": ("PDCD1_AXIS",),
    "内化速率": ("CD276",),
    "耐药机制": ("KRAS",),
}


@dataclass(frozen=True)
class ScientificObject:
    """
    证据针对的科学对象（ValidityGate 的同一性判断单位）。

    Attributes:
        kind: TARGET/MOLECULE（词典专名）| PHENOMENON（抽象效应）
        raw: 抽取到的原文片段
        canonical_key: 词典命中→规范名（别名归一）；未命中→原文小写
    """

    kind: str
    raw: str
    canonical_key: str


class AspectInferrer:
    """
    从 Evidence 推断 AspectSet（非空集合）——v0.3 Aspect 约束推断器。

    纯词典、零 LLM、确定性。三级瀑布：
    1. 显式字段：metadata["aspects"]（复数优先）/ metadata["aspect"]，
       值可解析为 Aspect 成员（中文名或枚举名）则采纳，非法值跳过；
    2. 词典推断：对 content + topic 扫描 ASPECT_TRIGGER_LEXICON，
       ASCII 键词边界匹配、中文键子串匹配，命中切面全部入集合；
    3. 兜底：集合为空 → {UNKNOWN}（保守公理：推断器永不返回空集）。
    """

    def infer(self, ev: Evidence, topic: str = "") -> frozenset[Aspect]:
        """
        推断证据的语义切面集合（永不返回空集）。

        Args:
            ev: 证据条目
            topic: 争议主题（词典扫描域的一部分）

        Returns:
            非空 frozenset[Aspect]；词典零命中时为 {Aspect.UNKNOWN}
        """
        # 第一级：metadata 显式切面（复数键优先于单数键）
        explicit = self._from_metadata(ev)
        if explicit:
            return explicit

        # 第二级：词典扫描（content + topic 联合语料）
        text = f"{ev.content or ''} {topic}"
        hits: set[Aspect] = set()
        for aspect, keys in ASPECT_TRIGGER_LEXICON.items():
            for key in keys:
                if self._lexicon_hit(key, text):
                    hits.add(aspect)
                    break
        if hits:
            return frozenset(hits)

        # 第三级：兜底 UNKNOWN（通配语义见 _pair_alive）
        return frozenset({Aspect.UNKNOWN})

    @staticmethod
    def _from_metadata(ev: Evidence) -> Optional[frozenset[Aspect]]:
        """
        读取 metadata 显式切面字段（复数键 "aspects" 优先，单数 "aspect" 次之）。

        解析规则：元素能按 Aspect 枚举名（ACTIVITY_DIRECTION）或中文名
        （"活性方向"）解析则采纳；非法元素跳过。全部非法 → 返回 None
        继续下一级（不抛异常——基准接入前的容错面）。

        Args:
            ev: 证据条目

        Returns:
            解析出的切面集合；无可用字段时 None
        """
        md = ev.metadata or {}
        raw = md.get("aspects")
        if raw is None:
            raw = md.get("aspect")
        if raw is None:
            return None
        if isinstance(raw, (str, Aspect)):
            raw = [raw]
        parsed: set[Aspect] = set()
        for item in raw:
            if isinstance(item, Aspect):
                parsed.add(item)
                continue
            if not isinstance(item, str):
                continue
            try:
                parsed.add(Aspect[item.upper()])
            except KeyError:
                pass
            try:
                parsed.add(Aspect(item))
            except ValueError:
                pass
        return frozenset(parsed) if parsed else None

    @staticmethod
    def _lexicon_hit(key: str, text: str) -> bool:
        """
        词典键匹配：ASCII 键按词边界（防 "IC50" 误中 "C50"），中文键子串。

        Args:
            key: 词典键（原样大小写）
            text: 待匹配语料

        Returns:
            True 表示命中
        """
        if key.isascii():
            # 大小写不敏感 + ASCII 词边界；"-" 不视作词内字符（B7-H3 整键匹配）
            pattern = r"(?<![A-Za-z0-9])" + re.escape(key) + r"(?![A-Za-z0-9])"
            return re.search(pattern, text, flags=re.IGNORECASE) is not None
        return key in text


class ObjectExtractor:
    """
    从 Evidence 抽取科学对象列表——ValidityGate 的对象抽取器（v0.3）。

    三级瀑布（全部命中均入列表，不做唯一化裁决——同一性由键集相交判定）：
    1. metadata 显式字段：target / molecule / locus（经别名表归一）；
    2. 术语词典：OBJECT_ALIASES 变体键扫描（词边界/子串规则同 Aspect）；
    3. 现象词典：PHENOMENON_LEXICON（现象键 + 宿主分子键一并入列）；
    4. 全空 → []（→ Gate 判 UNKNOWN 放行）。
    """

    def extract(self, ev: Evidence, topic: str = "") -> list[ScientificObject]:
        """
        抽取证据针对的科学对象（可能多个，全部保留供键集相交判定）。

        Args:
            ev: 证据条目
            topic: 争议主题（扫描域的一部分）

        Returns:
            ScientificObject 列表（可为空=无对象信号）
        """
        text = f"{ev.content or ''} {topic}"
        objs: list[ScientificObject] = []

        # 第一级：metadata 显式对象字段（别名归一后采纳）
        md = ev.metadata or {}
        for field_key in ("target", "molecule", "locus"):
            value = md.get(field_key)
            if isinstance(value, str) and value.strip():
                raw = value.strip()
                canonical = self._canonicalize(raw) or raw.lower()
                objs.append(ScientificObject("MOLECULE", raw, canonical))

        # 第二级：别名词典（大小写不敏感，命中变体归一到 canonical）
        lowered = text.lower()
        for canonical, variants in OBJECT_ALIASES.items():
            for variant in variants:
                if self._variant_hit(variant, text, lowered):
                    objs.append(ScientificObject("MOLECULE", variant, canonical))
                    break  # 同 canonical 只记一次，避免同族变体重复计数

        # 第三级：现象词典（现象键命中 → 现象对象 + 宿主分子键对象）
        plowered = lowered
        for phenomenon, hosts in PHENOMENON_LEXICON.items():
            if phenomenon.lower() in plowered:
                objs.append(ScientificObject("PHENOMENON", phenomenon, phenomenon))
                for host in hosts:
                    objs.append(ScientificObject("HOST", host, host))

        return objs

    @staticmethod
    def _canonicalize(raw: str) -> Optional[str]:
        """把原文对象名经别名表归一为 canonical（大小写不敏感）。"""
        raw_l = raw.lower()
        for canonical, variants in OBJECT_ALIASES.items():
            for variant in variants:
                if variant.lower() == raw_l:
                    return canonical
        return None

    @staticmethod
    def _variant_hit(variant: str, text: str, lowered: str) -> bool:
        """词典变体匹配：ASCII 变体词边界（大小写不敏感）、中文变体子串。"""
        if variant.isascii():
            pattern = r"(?<![A-Za-z0-9])" + re.escape(variant) + r"(?![A-Za-z0-9])"
            return re.search(pattern, text, flags=re.IGNORECASE) is not None
        return variant in lowered


class ValidityGate:
    """
    有效性门禁（v0.3，CVG 等价物）：候选矛盾对的对象同一性轻量校验。

    判定语义（确定性、零 LLM）：
    - 任一方无对象信号 → UNKNOWN：放行（ABP 保守方向，门禁只降 FPR 不降召回）
    - 键集相交（含同族/现象通道） → SAME_OBJECT：放行
    - 双方均含词典专名且键集完全不相交 → DIFFERENT_OBJECT：拦截不定罪
    - 其余（仅泛指/现象词冲突） → AMBIGUOUS：降为待举证（pending_pairs）
    """

    def __init__(self, extractor: Optional[ObjectExtractor] = None):
        """
        Args:
            extractor: 对象抽取器（None 时用默认 ObjectExtractor，消融注入点）
        """
        self._extractor = extractor or ObjectExtractor()

    def check(self, ev_a: Evidence, ev_b: Evidence, topic: str = "") -> GateResult:
        """
        校验候选矛盾对是否针对同一科学对象。

        Args:
            ev_a: 证据 A
            ev_b: 证据 B
            topic: 争议主题（参与对象扫描）

        Returns:
            GateResult（三态 + 双方键 + 依据 + 人读理由）
        """
        objs_a = self._extractor.extract(ev_a, topic)
        objs_b = self._extractor.extract(ev_b, topic)
        keys_a = {o.canonical_key for o in objs_a}
        keys_b = {o.canonical_key for o in objs_b}

        # 1) 任一方无对象信号：UNKNOWN 放行（信息缺失≠对象不同）
        if not keys_a or not keys_b:
            return GateResult(
                decision=GateDecision.UNKNOWN,
                object_a=sorted(keys_a), object_b=sorted(keys_b),
                detail="至少一方无对象信号，门禁放行（ABP 保守方向）",
            )

        # 2) 键集相交：SAME（同族别名天然经 canonical 归一后相交）
        shared = keys_a & keys_b
        if shared:
            return GateResult(
                decision=GateDecision.SAME_OBJECT,
                object_a=sorted(keys_a), object_b=sorted(keys_b),
                matched_keys=sorted(shared),
                detail=f"双方共享对象键 {sorted(shared)}，判同一对象",
            )

        # 3) 双方均有词典专名（MOLECULE 且非泛指归一名）且不相交：DIFFERENT 拦截
        #    （同族关系如 PD-1/PD-L1 已在 OBJECT_ALIASES 归一，不会走到这里）
        spec_a = {o.canonical_key for o in objs_a if o.kind == "MOLECULE"}
        spec_b = {o.canonical_key for o in objs_b if o.kind == "MOLECULE"}
        if spec_a and spec_b:
            return GateResult(
                decision=GateDecision.DIFFERENT_OBJECT,
                object_a=sorted(keys_a), object_b=sorted(keys_b),
                detail=f"双方为不同对象专名（{sorted(spec_a)} vs {sorted(spec_b)}），拦截定罪",
            )

        # 4) 仅泛指/现象词冲突：AMBIGUOUS 降为待举证
        return GateResult(
            decision=GateDecision.AMBIGUOUS,
            object_a=sorted(keys_a), object_b=sorted(keys_b),
            detail="对象指代不明（无词典专名佐证同一性），降为待举证",
        )


def _grade_intensity(
    spec: KeywordPairSpec,
    source_a: str,
    source_b: str,
    gate: Optional[GateResult],
) -> int:
    """
    矛盾强度分级（v0.3，纯函数，输入全部可审计）。

    步骤（每步均有大纲 §4.4 依据）：
    S1 硬约束短路：spec.hard_constraint → 直接 3（CRITICAL 即 3 级）
    S2 极性基线：directional=2（方向存疑），degree=1（程度分歧）
    S3 来源可信度差加成：|w_a − w_b| ≥ 0.30 → +1（高可信对立更严重）
       注：权重口径必须用 SOURCE_WEIGHTS（来源谱系），不用 overall_score
       ——后者混入时效/相关性会破坏确定性与复现性。
    S4 门禁佐证加成：SAME_OBJECT 且 base==2 → 3（同一对象上的方向对立
       = 方向直接相反）；degree 基线不加成
    S5 截断到 [1, 3]

    Args:
        spec: 命中的词对规格
        source_a / source_b: 双方证据来源（SOURCE_WEIGHTS 键）
        gate: 门禁结论（门禁关闭时为 None）

    Returns:
        intensity ∈ {1, 2, 3}
    """
    # S1 硬约束
    if spec.hard_constraint:
        return 3

    # S2 极性基线
    base = 2 if spec.polarity == POLARITY_DIRECTIONAL else 1
    intensity = base

    # S3 来源可信度差（与 EvidenceScorer.SOURCE_WEIGHTS 同口径兜底 0.2）
    from app.core.virtual_lab.consensus_court import EvidenceScorer

    w_a = EvidenceScorer.SOURCE_WEIGHTS.get(source_a.lower(), 0.2)
    w_b = EvidenceScorer.SOURCE_WEIGHTS.get(source_b.lower(), 0.2)
    if abs(w_a - w_b) >= SOURCE_GAP_INTENSITY_THRESHOLD:
        intensity += 1

    # S4 同对象佐证（仅方向基线加成）
    if gate is not None and gate.decision == GateDecision.SAME_OBJECT and base == 2:
        intensity = 3

    return max(1, min(3, intensity))


# ============================================================================
# 非对称举证 rubric
# ============================================================================

class AsymmetricRubric:
    """
    非对称举证 rubric - "找到矛盾才定罪"

    核心原则：
    1. 被控方（Agent 输出）默认无罪（passed=True）
    2. 指控方（审查者）需找到矛盾证据对才能定罪
    3. "内容过短"等问题标记为 PENDING（待举证），不直接拒绝
    4. 只有当找到相互矛盾的证据时，才判定为不通过

    与"有罪推定"的区别：
    - 有罪推定：内容过短 → 直接拒绝
    - 非对称举证：内容过短 → 标记 PENDING → 寻找矛盾 → 无矛盾则通过

    v0.3 三开关（§6.4 消融自变量）：
    - enable_aspect：关闭 → 跳过切面判据（该机制不参与判定）
    - enable_validity_gate：关闭 → 跳过门禁、pending_pairs 恒空
    - enable_intensity：关闭 → intensity 恒 2、desc 用经典模板
    三开关全关 = v0.2 行为等价（子串极性修复除外——它始终生效，
    修复的是"不存在"含"存在"的正确性缺陷，见 _polarity_sides）。
    """

    # 矛盾检测的关键词对（同一主题的相反结论）。
    # v0.3 兼容锁：本类属性必须保持"8 个 (keyword_a, keyword_b) 二元组"
    # 的形态与顺序（l1_generator._build_all_keyword_table 解包、
    # test_rubric_engine 参数化均依赖），故不做成动态 property。
    CONTRADICTION_KEYWORD_PAIRS = [
        ("有效", "无效"),
        ("成功", "失败"),
        ("阳性", "阴性"),
        ("激活", "抑制"),
        ("存在", "不存在"),
        ("高表达", "低表达"),
        ("促进", "抑制"),
        ("上调", "下调"),
    ]

    def __init__(
        self,
        enable_aspect: bool = True,
        enable_validity_gate: bool = True,
        enable_intensity: bool = True,
        aspect_inferrer: Optional[AspectInferrer] = None,
        object_extractor: Optional[ObjectExtractor] = None,
    ) -> None:
        """
        Args:
            enable_aspect: 是否启用 Aspect 条件化（机制一）
            enable_validity_gate: 是否启用有效性门禁（机制二）
            enable_intensity: 是否启用强度分级（机制三）
            aspect_inferrer: 切面推断器（消融注入点，None 用默认词典版）
            object_extractor: 对象抽取器（消融注入点，None 用默认词典版）
        """
        self._enable_aspect = enable_aspect
        self._enable_validity_gate = enable_validity_gate
        self._enable_intensity = enable_intensity
        self._inferrer = aspect_inferrer or AspectInferrer()
        self._gate = ValidityGate(object_extractor) if enable_validity_gate else None

    def evaluate(
        self,
        evidence_scores: list[EvidenceScore],
        evidence_map: dict[str, Evidence],
        topic: str = "",
    ) -> RubricVerdict:
        """
        评估证据集，寻找矛盾

        Args:
            evidence_scores: 证据评分列表
            evidence_map: evidence_id -> Evidence 映射
            topic: 争议主题（用于矛盾描述）

        Returns:
            RubricVerdict 裁决结果
        """
        # 收集待举证问题（非对称举证标记，不影响通过）
        pending_issues = self._collect_pending_issues(evidence_scores, evidence_map)

        # 寻找矛盾证据对（v0.3：门降级候选一并返回）
        contradictions, pending_pairs = self._find_contradictions(
            evidence_scores, evidence_map, topic
        )
        # 门禁降级的候选对同步进入 pending_issues（人读文本，供复核）
        for pair in pending_pairs:
            gate = pair.gate_detail
            pending_issues.append(
                f"矛盾候选 {pair.evidence_a_id}↔{pair.evidence_b_id}："
                f"对象指代不明（{gate.detail}），降为待举证" if gate
                else f"矛盾候选 {pair.evidence_a_id}↔{pair.evidence_b_id}：对象指代不明，降为待举证"
            )

        # 非对称举证：只有找到矛盾才定罪
        if contradictions:
            passed = False
            critical_count = sum(1 for c in contradictions if c.severity == Severity.CRITICAL)
            rationale = (
                f"非对称举证：找到 {len(contradictions)} 个矛盾证据对"
                f"（{critical_count} 个 CRITICAL），判定为不通过。"
            )
            # v0.3：强度分布摘要（仅启用强度机制时追加，不影响原句式）
            if self._enable_intensity:
                dist: dict[int, int] = {}
                for c in contradictions:
                    dist[c.intensity] = dist.get(c.intensity, 0) + 1
                dist_text = " / ".join(
                    f"{k}×{dist[k]}" for k in sorted(dist, reverse=True)
                )
                rationale += f"（强度分布 {dist_text}）"
            confidence = 0.9  # 找到矛盾时置信度高
        else:
            passed = True
            rationale = (
                f"非对称举证：未找到矛盾证据对，{len(pending_issues)} 个问题标记为待举证，"
                f"默认通过。"
            )
            # 无矛盾时置信度基于证据数量和质量
            confidence = self._compute_confidence(evidence_scores)

        return RubricVerdict(
            passed=passed,
            contradictions=contradictions,
            pending_issues=pending_issues,
            confidence=confidence,
            rationale=rationale,
            pending_pairs=pending_pairs,
        )

    def _collect_pending_issues(
        self,
        evidence_scores: list[EvidenceScore],
        evidence_map: dict[str, Evidence],
    ) -> list[str]:
        """
        收集待举证问题（非对称举证标记）

        这些问题不直接导致拒绝，而是标记为"待举证"，
        需要补充证据但默认通过。

        Args:
            evidence_scores: 证据评分列表
            evidence_map: 证据映射

        Returns:
            待举证问题列表
        """
        pending = []
        for score in evidence_scores:
            ev = evidence_map.get(score.evidence_id)
            if not ev:
                continue

            # 内容过短 → 标记为待举证（非对称举证：不直接拒绝）
            if not ev.content or len(ev.content.strip()) < 10:
                pending.append(
                    f"证据 {score.evidence_id} 内容过短，待补充详细论证"
                )

            # 来源为空 → 标记为待举证
            if not ev.source:
                pending.append(
                    f"证据 {score.evidence_id} 来源为空，待补充数据来源"
                )

            # Agent 观点缺少引用 → 标记为待举证
            if ev.source.lower() == "agent_opinion" and not ev.reference_ids:
                pending.append(
                    f"证据 {score.evidence_id} 为 Agent 观点但缺少引用，待补充可验证引用"
                )

            # 证据评分中的问题也收集
            for issue in score.issues:
                pending.append(f"证据 {score.evidence_id}: {issue}")

        return pending

    def _find_contradictions(
        self,
        evidence_scores: list[EvidenceScore],
        evidence_map: dict[str, Evidence],
        topic: str,
    ) -> tuple[list[ContradictionPair], list[ContradictionPair]]:
        """
        寻找矛盾证据对（v0.3：返回定罪矛盾对 + 门禁降级候选对）

        通过关键词对检测同一主题下的相反结论。
        同一来源不对比（防止 Agent 自相矛盾假阳性）。

        Args:
            evidence_scores: 证据评分列表
            evidence_map: 证据映射
            topic: 争议主题

        Returns:
            (contradictions, pending_pairs)：
            - contradictions: 定罪矛盾对（进 RubricVerdict.contradictions）
            - pending_pairs: ValidityGate 判 AMBIGUOUS 而降级的候选对
              （门禁关闭时恒为空）
        """
        contradictions: list[ContradictionPair] = []
        pending_pairs: list[ContradictionPair] = []

        # 只检查有内容的证据
        valid_evidence = [
            (score, evidence_map.get(score.evidence_id))
            for score in evidence_scores
            if score.evidence_id in evidence_map
            and evidence_map[score.evidence_id].content
        ]

        # ---------- P0 硬约束前置扫描（v0.4.8 双禁令） ----------
        # 独立于词对流水线（P1-P5）：单条证据命中硬约束模式即构成
        # CRITICAL 定罪矛盾对（intensity=3），无需对立词对匹配。
        # 与 RuleEngine.screen()/screen_asymmetric() 中结构预测工具禁令
        # （AlphaFold2/3 双禁，合规降级 OpenDDE→Chai1→Boltz）同源——
        # 初筛标记 CRITICAL 后由本扫描转为可审计的矛盾对记录，
        # 使 S1 硬约束短路通道在 rubric 评估路径上同样可达。
        for _score, ev in valid_evidence:
            if ev is None:
                continue
            for _pat, rule_name, desc in _scan_hard_constraints(ev.content):
                contradictions.append(
                    ContradictionPair(
                        evidence_a_id=ev.evidence_id,
                        evidence_b_id=ev.evidence_id,
                        topic=topic,
                        contradiction_desc=(
                            f"硬约束短路[{rule_name}]：证据 {ev.evidence_id} "
                            f"{desc}，intensity=3"
                        ),
                        severity=Severity.CRITICAL,
                        intensity=3,
                        matched_keyword_pair=None,
                    )
                )

        # 两两对比寻找矛盾
        for i, (score_a, ev_a) in enumerate(valid_evidence):
            for score_b, ev_b in valid_evidence[i + 1:]:
                # 同一来源不对比（不会自相矛盾）
                if ev_a.source == ev_b.source and ev_a.evidence_id != ev_b.evidence_id:
                    continue

                contradiction, ambiguous = self._check_pair_contradiction(
                    ev_a, ev_b, topic
                )
                if contradiction:
                    contradictions.append(contradiction)
                elif ambiguous:
                    pending_pairs.append(ambiguous)

        return contradictions, pending_pairs

    def _check_pair_contradiction(
        self,
        ev_a: Evidence,
        ev_b: Evidence,
        topic: str,
    ) -> tuple[Optional[ContradictionPair], Optional[ContradictionPair]]:
        """
        检查两条证据是否矛盾——v0.3 五段流水线

        P1 词对极性匹配（子串修复始终生效：修复正确性缺陷，非可选机制）
        P2 [enable_aspect] 词对存活判据 _pair_alive——跨切面伪冲突在此拦截
        P3 [enable_validity_gate] ValidityGate 三态：SAME/UNKNOWN 放行定罪；
           DIFFERENT 拦截不定罪；AMBIGUOUS 降级候选对（第二返回值）
        P4 [enable_intensity] 强度分级 + v0.3 rationale 模板
           （关闭时 desc 逐字用 CLASSIC_DESC_TEMPLATE，等价 v0.2）
        P5 ContradictionPair 组装（同证据对首个存活词对即返回，防重复记录）

        Args:
            ev_a: 证据 A
            ev_b: 证据 B
            topic: 争议主题

        Returns:
            (定罪矛盾对 | None, AMBIGUOUS 降级候选对 | None)
        """
        content_a = ev_a.content.lower()
        content_b = ev_b.content.lower()

        # 切面推断仅在 aspect 机制启用时进行（关闭时完全不发生，含异常面）
        aspects_a: Optional[frozenset[Aspect]] = None
        aspects_b: Optional[frozenset[Aspect]] = None
        if self._enable_aspect:
            aspects_a = self._inferrer.infer(ev_a, topic)
            aspects_b = self._inferrer.infer(ev_b, topic)

        gate: Optional[GateResult] = None
        gate_checked = False

        for spec in KEYWORD_PAIR_SPECS:
            # ---------- P1 词对极性命中（始终启用） ----------
            if not _pair_hit_by_polarity(content_a, content_b, spec):
                continue

            # ---------- P2 aspect 存活判据 ----------
            shared_aspects: list[Aspect] = []
            matched_aspect: Optional[Aspect] = None
            if self._enable_aspect:
                assert aspects_a is not None and aspects_b is not None
                alive, shared = self._pair_alive(spec, aspects_a, aspects_b)
                if not alive:
                    # 跨切面伪冲突：该词对不构成矛盾，继续尝试下一词对；
                    # 全部词对不存活才判无矛盾
                    continue
                shared_aspects = sorted(shared, key=lambda a: a.value)
                matched_aspect = shared_aspects[0] if shared_aspects else None

            # ---------- P3 ValidityGate（同证据对只判一次） ----------
            if self._enable_validity_gate and self._gate is not None:
                if not gate_checked:
                    gate = self._gate.check(ev_a, ev_b, topic)
                    gate_checked = True
                if gate.decision == GateDecision.DIFFERENT_OBJECT:
                    # 证实不同对象：非对称举证下不构成可定罪矛盾，
                    # 拦截且不进 pending_pairs（不是"待举证"，是"非对抗"）
                    return None, None
                if gate.decision == GateDecision.AMBIGUOUS:
                    # 指代不明 → 降为待举证候选（不定罪、不影响 passed）
                    desc = (
                        f"矛盾候选：证据 {ev_a.evidence_id} 与 {ev_b.evidence_id} "
                        f"在主题 '{topic}' 上词对（{spec.keyword_a}/{spec.keyword_b}）对立，"
                        f"但对象指代不明，降为待举证（{gate.detail}）"
                    )
                    ambiguous_pair = ContradictionPair(
                        evidence_a_id=ev_a.evidence_id,
                        evidence_b_id=ev_b.evidence_id,
                        topic=topic,
                        contradiction_desc=desc,
                        severity=Severity.PENDING,
                        intensity=2,
                        matched_aspect=matched_aspect,
                        shared_aspects=list(shared_aspects),
                        gate_detail=gate,
                        matched_keyword_pair=(spec.keyword_a, spec.keyword_b),
                    )
                    return None, ambiguous_pair
                # SAME_OBJECT / UNKNOWN：继续定罪通道

            # ---------- P4 强度分级 + desc 模板 ----------
            if self._enable_intensity:
                intensity = _grade_intensity(
                    spec, ev_a.source or "unknown", ev_b.source or "unknown", gate
                )
                desc = self._render_desc(spec, ev_a, ev_b, topic, intensity, gate)
            else:
                # 经典路径：desc 逐字使用 v0.2 模板（等价性断言依赖）
                intensity = 2
                desc = CLASSIC_DESC_TEMPLATE.format(
                    a_id=ev_a.evidence_id, b_id=ev_b.evidence_id,
                    ka=spec.keyword_a, kb=spec.keyword_b, topic=topic,
                )

            # ---------- P5 组装（同证据对取首个存活词对，避免重复记录） ----------
            # 结构化词对字段仅在任一 v0.3 机制激活时填入：三开关全关的
            # 经典路径保持 v0.2 形态（该字段为 None），等价性断言依赖。
            structural = (
                self._enable_aspect
                or self._enable_validity_gate
                or self._enable_intensity
            )
            return ContradictionPair(
                evidence_a_id=ev_a.evidence_id,
                evidence_b_id=ev_b.evidence_id,
                topic=topic,
                contradiction_desc=desc,
                severity=Severity.CRITICAL,
                intensity=intensity,
                matched_aspect=matched_aspect,
                shared_aspects=list(shared_aspects),
                gate_detail=gate,
                matched_keyword_pair=(
                    (spec.keyword_a, spec.keyword_b) if structural else None
                ),
            ), None

        return None, None

    @staticmethod
    def _pair_alive(
        spec: KeywordPairSpec,
        aspects_a: frozenset[Aspect],
        aspects_b: frozenset[Aspect],
    ) -> tuple[bool, set[Aspect]]:
        """
        Aspect 存活判据（大纲形式化：Aspect(ei) ∩ Aspect(ej) ≠ ∅ ∧ K 在该切面合法）。

        UNKNOWN 通配语义：任一方切面集合 ⊆ {UNKNOWN}（词典零命中）时，
        aspect 判据对该证据对退化为恒真（放行）。理由：UNKNOWN 是"推断器
        无知识"而非"证据无切面"，若按空交集拦截会把词典覆盖缺口变成系统性
        漏检——ABP 的保守方向是宁放行不漏检（门禁只降 FPR 不降召回）。
        真正跨切面的伪冲突双方必然各有词典命中，不受通配影响。

        Args:
            spec: 词对规格
            aspects_a: 证据 A 的切面集合（非空）
            aspects_b: 证据 B 的切面集合（非空）

        Returns:
            (是否存活, 存活切面集合)；通配且无交集时存活切面=词对合法切面兜底
        """
        shared = aspects_a & aspects_b
        unknown_a = aspects_a <= {Aspect.UNKNOWN}
        unknown_b = aspects_b <= {Aspect.UNKNOWN}
        if unknown_a or unknown_b:
            # 通配放行：词对在该证据对上按 v0.2 语义激活
            fallback = shared if shared else set(spec.aspects)
            return True, fallback
        if not shared:
            # 双方各有明确切面且完全不相交 → 跨切面伪冲突，不存活
            return False, set()
        alive = spec.aspects & shared
        if not alive:
            return False, set()
        return True, alive

    @staticmethod
    def _render_desc(
        spec: KeywordPairSpec,
        ev_a: Evidence,
        ev_b: Evidence,
        topic: str,
        intensity: int,
        gate: Optional[GateResult],
    ) -> str:
        """
        渲染 v0.3 强度 rationale 模板。

        兼容性：经典模板不含两侧关键词会破坏既有 desc 关键词断言，
        因此 v0.3 模板同样嵌入 ka/kb 与 topic 子串，仅追加
        切面/强度/门禁审计信息（前缀与 v0.2 完全一致）。

        Args:
            spec: 命中词对
            ev_a / ev_b: 双方证据
            topic: 争议主题
            intensity: 已计算的强度等级
            gate: 门禁结论（可 None）

        Returns:
            矛盾描述文本
        """
        level_zh = {1: "程度分歧（轻微）", 2: "方向存疑（中度）", 3: "方向直接相反（对抗）"}
        gate_zh = gate.detail if gate is not None else "门禁未启用"
        return (
            f"证据 {ev_a.evidence_id} 含 '{spec.keyword_a}'，"
            f"证据 {ev_b.evidence_id} 含 '{spec.keyword_b}'，"
            f"在主题 '{topic}' 上存在矛盾"
            f"——强度 {intensity}/3（{level_zh[intensity]}）；门禁：{gate_zh}"
        )

    def _compute_confidence(self, evidence_scores: list[EvidenceScore]) -> float:
        """
        计算无矛盾时的置信度

        基于证据数量和平均评分计算。

        Args:
            evidence_scores: 证据评分列表

        Returns:
            置信度（0-1）
        """
        if not evidence_scores:
            return 0.3  # 无证据时低置信度

        avg_score = sum(s.overall_score for s in evidence_scores) / len(evidence_scores)
        # 证据越多置信度越高（但不超过 0.9）
        evidence_bonus = min(0.1, len(evidence_scores) * 0.02)
        return round(min(0.9, avg_score + evidence_bonus), 3)


# ============================================================================
# 证据位置加权器
# ============================================================================

class EvidencePositionWeighter:
    """
    证据位置加权器

    按证据在回答中的位置加权：
    - 开头（0.0-0.33）：声明/背景，权重 0.8（可能被后续推翻）
    - 中间（0.33-0.67）：论证/数据，权重 1.0（最可靠）
    - 结尾（0.67-1.0）：结论/总结，权重 0.9（可能过度概括）

    使用场景：
        当 Agent 回答中包含多个声明时，中间的论证比开头的声明更可靠。
        审查时按位置加权置信度。
    """

    # 位置区间权重
    POSITION_WEIGHTS = {
        "opening": 0.8,    # 开头：声明/背景
        "middle": 1.0,     # 中间：论证/数据（最可靠）
        "closing": 0.9,    # 结尾：结论/总结
    }

    def weight(self, position_ratio: float) -> float:
        """
        根据证据在回答中的位置比例计算权重

        Args:
            position_ratio: 位置比例（0.0=开头，1.0=结尾）

        Returns:
            位置权重（0.8-1.0）
        """
        if not 0.0 <= position_ratio <= 1.0:
            logger.warning("位置比例 %.2f 超出 [0, 1] 范围，使用默认权重 1.0", position_ratio)
            return 1.0

        if position_ratio < 0.33:
            return self.POSITION_WEIGHTS["opening"]
        elif position_ratio < 0.67:
            return self.POSITION_WEIGHTS["middle"]
        else:
            return self.POSITION_WEIGHTS["closing"]

    def weight_by_section(self, section: str) -> float:
        """
        根据段落名称获取权重

        Args:
            section: 段落名称（opening/middle/closing）

        Returns:
            位置权重
        """
        return self.POSITION_WEIGHTS.get(section, 1.0)

    def adjust_confidence(
        self,
        base_confidence: float,
        position_ratio: float,
    ) -> float:
        """
        根据位置调整置信度

        Args:
            base_confidence: 基础置信度
            position_ratio: 位置比例

        Returns:
            调整后的置信度
        """
        position_weight = self.weight(position_ratio)
        adjusted = base_confidence * position_weight
        return round(max(0.0, min(1.0, adjusted)), 3)


# ============================================================================
# 先举证后裁决
# ============================================================================

class DrillBeforeDisposition:
    """
    先举证后裁决

    核心原则：先收集所有证据再下结论，避免过早判定。

    使用流程：
    1. collect(evidence)：逐条收集证据
    2. disposition()：基于全部收集的证据下结论

    与直接判定的区别：
    - 直接判定：看到第一条矛盾就下结论
    - 先举证后裁决：收集所有证据，统计矛盾数量，基于整体情况下结论
    """

    def __init__(self):
        # 收集的证据列表
        self._collected_evidence: list[Evidence] = []
        # 收集的证据评分列表
        self._collected_scores: list[EvidenceScore] = []
        # 是否已完成收集
        self._collection_complete: bool = False

    def collect(self, evidence: Evidence, score: Optional[EvidenceScore] = None) -> None:
        """
        收集一条证据

        Args:
            evidence: 证据条目
            score: 证据评分（可选，稍后可补充）
        """
        if self._collection_complete:
            logger.warning("收集已完成，仍尝试添加证据 %s", evidence.evidence_id)
            self._collection_complete = False  # 重新开放收集

        self._collected_evidence.append(evidence)
        if score:
            self._collected_scores.append(score)

    def collect_batch(
        self,
        evidence_list: list[Evidence],
        scores: Optional[list[EvidenceScore]] = None,
    ) -> None:
        """
        批量收集证据

        Args:
            evidence_list: 证据列表
            scores: 对应的评分列表（可选）
        """
        if scores and len(scores) != len(evidence_list):
            logger.warning("证据列表长度 %d 与评分列表长度 %d 不匹配", len(evidence_list), len(scores))

        for i, ev in enumerate(evidence_list):
            score = scores[i] if scores and i < len(scores) else None
            self.collect(ev, score)

    def complete_collection(self) -> None:
        """
        标记收集完成

        调用此方法后，disposition() 才能下结论。
        """
        self._collection_complete = True
        logger.info(
            "证据收集完成：共 %d 条证据，%d 条评分",
            len(self._collected_evidence),
            len(self._collected_scores),
        )

    def disposition(
        self,
        topic: str = "",
        rubric: Optional[AsymmetricRubric] = None,
    ) -> RubricVerdict:
        """
        基于收集的全部证据下结论

        Args:
            topic: 争议主题
            rubric: 非对称举证 rubric（None 时创建新实例）

        Returns:
            RubricVerdict 裁决结果

        Raises:
            RuntimeError: 收集未完成时调用
        """
        if not self._collection_complete:
            raise RuntimeError(
                "收集未完成，请先调用 complete_collection() 再下结论"
            )

        rubric = rubric or AsymmetricRubric()

        # 构建证据映射
        evidence_map = {ev.evidence_id: ev for ev in self._collected_evidence}

        # 如果没有评分，创建默认评分
        if not self._collected_scores:
            from app.core.virtual_lab.consensus_court import EvidenceScorer
            scorer = EvidenceScorer()
            self._collected_scores = [
                scorer.score(ev, topic) for ev in self._collected_evidence
            ]

        return rubric.evaluate(self._collected_scores, evidence_map, topic)

    def reset(self) -> None:
        """重置收集器，开始新的收集周期"""
        self._collected_evidence.clear()
        self._collected_scores.clear()
        self._collection_complete = False

    @property
    def evidence_count(self) -> int:
        """已收集的证据数量"""
        return len(self._collected_evidence)

    @property
    def is_complete(self) -> bool:
        """收集是否完成"""
        return self._collection_complete


# ============================================================================
# 便捷函数
# ============================================================================

def create_default_rubric_engine() -> tuple[AsymmetricRubric, EvidencePositionWeighter, DrillBeforeDisposition]:
    """
    创建默认的 rubric 引擎三件套

    Returns:
        (AsymmetricRubric, EvidencePositionWeighter, DrillBeforeDisposition)
    """
    return AsymmetricRubric(), EvidencePositionWeighter(), DrillBeforeDisposition()
