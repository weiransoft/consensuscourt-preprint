"""
DDB-L1.5 负例对照基准生成器（Dispute Benchmark - Level 1.5）

用途（论文大纲 §6.1 层 1.5 / §5.3 定理 1-A 的 P(S) 估计载体）：
    为 v0.3 三项机制（Aspect 约束 / ValidityGate / 强度分级）提供
    **专算假阳性率** 的负例对照集——对标 RevCI（Kumar et al. 2026）
    448 负例驱动 FPR 评测的设计。每个案例由构造保证 **无真矛盾**
    （ground_truth.has_true_contradiction = False），但刻意在 v0.2 的
    关键词通道上"看起来像矛盾"，使 v0.2 基线（三开关全关）必然误定罪，
    而 v0.3 对应机制应当拦截。四类负例与机制一阶效应一一对应：

    - cross_aspect（跨切面伪冲突）→ Aspect 约束拦截
      两侧证据真实同现某词对的两个关键词（一侧肯定、一侧否定形），
      但语义切面完全不相交（"放大工艺不存在障碍" 成药性 vs
      "靶点验证的机制依据" 机制——"存在/不存在"词对合法切面
      {结构可信度,成药性,安全性,UNKNOWN} 与切面交集无重叠），
      构成真语言冲突、假科学矛盾——aspect 约束的纯测量面。
    - degree_only（同方向程度分歧）→ Aspect 约束拦截
      同一科学问题、同一语义切面内，一方肯定、一方否定
      （"机制存在" vs "机制不存在"），v0.3 词表全为 directional
      极性，degree 拦截面不存在，故本类以 **aspect 切面错位** 拦截：
      双方同属 机制 切面，而 存在/不存在 词对合法切面不含 机制，
      _pair_alive 判死（机制层"存在性之争"是程度/证据强度分歧，
      非方向对立——切面错位拦截符合保守公理）。
    - different_object_gate（同切面异对象伪冲突）→ ValidityGate 拦截
      两侧证据同属一个切面、各挂 **词典专名且键集不相交** 的对象
      （EGFR vs KRAS），v0.2 必然定罪，v0.3 门禁判 DIFFERENT_OBJECT 拦截。
    - ambiguous_reference（对象指代不明）→ ValidityGate 降级待举证
      双方仅含 **现象词（PHENOMENON）无词典专名**，词对极性真实命中，
      v0.3 门禁判 AMBIGUOUS 降为 pending_pairs（不定罪、不拦截，待举证）。

schema（案例 dict，与 L1 兼容 + L1.5 扩展字段）：
    case_id:        str   "DDB-L15-0001" 形式全局唯一编号
    topic:          str   药物发现议题短语
    polarity:       str   恒为 "no_contradiction"（L1.5 全部是负例）
    category:       str   cross_aspect / degree_only / different_object_gate
                          / ambiguous_reference
    evidence_type:  str   恒为 "clean"（不注入形式缺陷——假阳性来源
                          是伪冲突而非缺陷，与 L1 的 defect_only 维度正交）
    defect_flags:   list  恒为空列表（同上）
    evidence_count: int   证据条数（扩展审计用）
    evidences:      list[Evidence dict]，字段与 L1 一致：
                          evidence_id / source / content / role /
                          timestamp(ISO 或 null) / reference_ids / metadata
                          （metadata 仅记 category，**严禁** 写 aspects/
                          target 等显式机制字段——负例必须考验词典推断路径）
    ground_truth:   {
        has_true_contradiction: False  恒 False（构造保证）
        defect_only:            False  恒 False（无形式缺陷）
        mechanism:              str    期望拦截机制 aspect / validity_gate
        expected_full:          str    全开配置期望结果
                                      no_conviction / pending_ambiguous
        expected_v02:           str    v0.2 基线期望结果，恒 convicted
                                      （四类构造均刻意命中 v0.2 词对通道）
    }

一致性约束（与源码严格对齐，严禁修改核心模块）：
    - 词对唯一权威来源：rubric_engine.KEYWORD_PAIR_SPECS（aspect/polarity/
      negated_b 均从规格表读取，杜绝双源漂移）
    - 机制拦截路径断言以实际机制实现为准（_pair_alive 通配语义、
      ValidityGate 四态判定），生成期即时自检：
      * 负例内容必须命中其目标词对通道（否则该负例无鉴别力，报错）
      * cross_aspect / degree_only：双方切面集合（按源码 AspectInferrer
        复算）必须使目标词对 _pair_alive=False（aspect 拦截面真实存在）
      * different_object_gate：双方对象键集（按源码 ObjectExtractor 复算）
        均含 MOLECULE 专名且不相交（DIFFERENT 拦截面真实存在）
      * ambiguous_reference：双方键集非空但无 MOLECULE 专名且不相交
        （AMBIGUOUS 面真实存在）

运行：
    cd backend && python -m tests.ddb.l15_generator
    → 生成 tests/ddb/l15_cases.json（UTF-8, ensure_ascii=False, indent=2）
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

# backend 根目录入 sys.path（python -m 在 backend 下运行；直跑脚本时兜底）
_BACKEND_ROOT = str(Path(__file__).resolve().parents[2])
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

# 导入核心模块需要环境变量占位（仅导入期校验，生成逻辑不联网）
os.environ.setdefault("SCIMINER_API_KEY", "ddb-l15-placeholder")
os.environ.setdefault("DB_PASSWORD", "ddb-l15-placeholder")

from app.core.virtual_lab.consensus_court import Evidence
from app.core.virtual_lab.rubric_engine import (
    Aspect,
    AspectInferrer,
    KEYWORD_PAIR_SPECS,
    KeywordPairSpec,
    ObjectExtractor,
    POLARITY_DIRECTIONAL,
    ValidityGate,
    _pair_hit_by_polarity,
)

# ---------------------------------------------------------------------------
# 采样池（来源/角色与 l1_generator 同源风格；来源权重沿用 L1 谱系权重）
# ---------------------------------------------------------------------------

#: 来源谱系池（键=来源，值=采样权重，与 L1 一致）
SOURCE_POOL: dict[str, float] = {
    "pubmed": 22.0,
    "chembl": 16.0,
    "uniprot": 14.0,
    "drugbank": 12.0,
    "experiment": 12.0,
    "llm": 14.0,
    "agent_opinion": 10.0,
}

#: 角色池（v0.4.13 对齐：论文 13 角色全覆盖 + 虚拟实验室平台扩展角色，
#: 与 consensus_court 虚拟实验室角色命名及 PIAgentArbiter.ROLE_WEIGHTS 一致；
#: data-specialist（论文名）= data-analyst（平台实现名））
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
]

#: 关键词 → 词对规格索引（keyword_a/keyword_b 双向登记，权威只读）
_SPEC_BY_KEYWORD: dict[str, KeywordPairSpec] = {}
for _spec in KEYWORD_PAIR_SPECS:
    _SPEC_BY_KEYWORD.setdefault(_spec.keyword_a, _spec)
    _SPEC_BY_KEYWORD.setdefault(_spec.keyword_b, _spec)


def _spec_of(keyword: str) -> KeywordPairSpec:
    """
    按关键词取词对规格。

    Args:
        keyword: 词对任一侧关键词

    Returns:
        KeywordPairSpec

    Raises:
        KeyError: 关键词不在权威规格表中（词表演进后模板未同步）
    """
    if keyword not in _SPEC_BY_KEYWORD:
        raise KeyError(f"关键词 '{keyword}' 不在 KEYWORD_PAIR_SPECS 权威词表中")
    return _SPEC_BY_KEYWORD[keyword]


# ---------------------------------------------------------------------------
# 模板池（人工设计、可审计；负例语义全部内嵌在句面，不依赖 metadata）
#
# 极性命中验证纪律（v0.4.3 修正）：
#   _pair_hit_by_polarity 为对称命中——(肯定(a)∧否定(b)) ∨
#   (肯定(b)∧否定(a))，同侧（同正/同负）不算命中。cross_aspect
#   模板必须让目标词对在两侧真实分居肯定/否定，同时双方语义切面
#   交集内没有任何命中词对存活（_pair_alive 判死），拦截面才成立。
#   所有模板的命中面以 _selfcheck 复算（_pair_hit_by_polarity
#   / _pair_alive / ValidityGate）强制验证，构造即验证。
# ---------------------------------------------------------------------------

#: 跨切面伪冲突模板（aspect 拦截面）：(topic, A 句, B 句, 目标词对关键词)
#: aspect 拦截面的复算纪律（_selfcheck_aspect，与源码 _check_pair_contradiction
#: 流水线同构）：词对极性命中 → 存活词对进门禁（门禁必须 UNKNOWN）→
#: v0.3 全开复算判无罪。两条构造纪律：
#:   a) 词对真实分居（_pair_hit_by_polarity 对称命中）——一侧含肯定形
#:      （ka 未被否定前缀/negated_b 覆盖的出现），另一侧含否定形
#:      （kb 直接出现，或 ka 被 不/未/无/非 覆盖）。双方全否定句
#:      （"未见…偏低…低位"堆叠）= 同侧，不算命中；
#:   b) 目标词对极性命中；其余**全部词对要么不命中、要么被 aspect 判死**
#:      （源码 P2 判死是词对级 continue，任何存活词对到达 P3 门禁——
#:      门禁 UNKNOWN/SAME 即定罪、DIFFERENT 即拦截，均破坏本类
#:      "aspect 独占拦截"的纯度）。推论：句面必须**零对象词典信号**
#:      （对象键集相交 → SAME 定罪；专名不相交 → DIFFERENT 拦截），
#:      且目标词对在 aspect 判死（交集 ∩ 合法集 = ∅）。
#:      词对合法性是**切面级**判定（spec.aspects 与交集求交），
#:      与"双方切面各自含合法成员"是两回事——交集必须整体非法。
#:      可行词对与隔离切面（对照 KEYWORD_PAIR_SPECS 合法切面 + 实测）：
#:      阳性（{效价,活性方向,安全性,UNKNOWN}，隔离切面 {成药性,机制,
#:      表达调控,结构可信度}）、
#:      激活（{活性方向,表达调控,机制}，隔离切面 {成药性,安全性}——
#:      交集必须整体落在隔离集内，双方各踩一类即可）。
#:      **单切面词对禁用**（上调/下调、高表达/低表达、存在、有效）：
#:      前两组合法集={表达调控}，而 ka/kb（上调/下调/高表达/低表达）
#:      全部是表达调控词典触发键——_pair_hit_by_polarity 要求双方句
#:      各含 ka/kb 之一（极性分居），则表达调控必然同入双方切面
#:      → 交集必含表达调控 → 合法集∩交集={表达调控}≠∅ → 词对
#:      永远存活，**aspect 拦截面结构性不存在**（静态探针实测钉死，
#:      四种 ka/kb 组合全部存活；探针已清理，结论固化于本注释）；
#:      存在/有效 合法集近全域，同样禁用。
#:      **UNKNOWN 通配（_pair_alive）**：任一方切面全 UNKNOWN → 恒真
#:      存活，aspect 拦截面消失——双方句面+topic 必须各有词典命中。
#:      **Topic 参与双方切面推断域**（AspectInferrer 扫 content+topic
#:      联合语料）：topic 中的术语（如"激酶检测"的"激酶"）会同时注入
#:      双方切面，复活任何合法交集中的命中词对——topic 严禁含切面
#:      触发术语（连"检测/复核"类后缀也带上主词，须整体避开词典键）。
#:      **否定侧必须显式落地**：_pair_hit_by_polarity 要求肯定侧
#:      ka∈pos（_polarity_sides 只登记 ka/kb 两个词），否定侧要求
#:      neg 集合非空（kb 直接出现，或 ka 出现被 不/未/无/非 **紧邻**
#:      前缀覆盖——"未见 X 上调"间隔形不生效，必须"不上调"紧邻形）。
#:      "未见异常/未见漂移"这类与词对无关的否定短语**不进
#:      neg 集合**——对侧没有 ka 的否定覆盖 = neg 为空 = 同侧不命中。
#:      否定侧必须有 ka 的否定形（如"不激活"）或 kb 直接出现。
#:      **否定句极性纪律**：否定句（未见/未/不）若同时含其它词对的
#:      肯定形关键词，会把该词对拖成反向命中（neg 侧已有否定形）——
#:      否定句必须**零肯定形词对关键词**。
#:      **单切面词对铁律**（上调/下调、高表达/低表达 合法集=
#:      {表达调控}）：ka/kb 全部是表达调控词典触发键，极性命中
#:      要求双方各含其一 → 表达调控必入双方切面 → 交集必含
#:      合法切面 → **本类无可行构造**，只能换词对（见上段禁用清单）。
#:      **Topic 注入对称性**：AspectInferrer 对双方都用 content+topic
#:      联合语料——topic 中的任何词典键都会**同时注入双方切面**。
#:      对单切面词对，这正是隔离切面的注入通道（topic 含隔离键 →
#:      交集=隔离切面）；对多切面词对，topic 严禁含任何合法切面键。
#:   句面纪律：中文键子串匹配——"受"会误命中"必需"、裸"偶联"误命中
#:   成药性、**"上调/下调/高表达/低表达"本身是表达调控触发键**（ka
#:   出现即把该切面注入所在侧——单切面词对的隔离句必须**零 ka/kb**，
#:   否定覆盖形"不上调"同样注入表达调控！）、"抑制/杀伤"是活性方向
#:   独立键（句面出现即把交集拖入活性方向）、"稳定/化合物/配方"误命中
#:   成药性、"达标/放行/合格"踩"反应"尾部子串→效价键、"重复"踩
#:   "应答"→效价键、"标志物"踩"生物标志物"→表达调控键；ASCII 键词
#:   边界大小写不敏感（"ic50"可命中"IC50"键）。目标词对的否定形与
#:   肯定形**严禁同时出现在同一句**
#:   （会先命中其它词对）。
#:   改句必须复算切面集与 hit/alive 矩阵，并逐句打印全词典命中键清单
#:   （切面集合相同 ≠ 命中键相同——topic 注入的键要靠清单才能暴露）。
CROSS_ASPECT_TEMPLATES: list[tuple[str, str, str, str]] = [
    # 阳性（{效价,活性方向,安全性,UNKNOWN}，切 表达调控）：双方切面
    # 交集={表达调控}（A 句"筛选"踩"苗头"、B 句"转录"），词对在表达调控
    # 非法 → P2 判死；零对象信号 → 门禁 UNKNOWN。
    (
        "候选药 WQ-066 的方向之争（筛选判读与转录组复核）",
        "高通量筛选判为阳性，苗头化合物的剂量曲线单调，构效关系清晰",
        "同一苗头的转录组复核判为阴性，差异基因的富集未越过显著性阈值，"
        "关联有待复核",
        "阳性",
    ),
    # 阳性（切 成药性+机制）：A 句=工艺句（成药性键 偶联/纯化/DAR）+
    # 肯定形"阳性"（pos_a），B 句=机制句（遗传学证据/通路/必需）+
    # 否定形"阴性"（neg_b）——极性分居 (正_a ∧ 负_b)。交集
    # {成药性,机制}（**双方都要踩齐两切面**——A≠B 时交集为空同样判死，
    # 但"交集非空且整体非法"才是标准拦截面；topic 用零词典词）。
    # 阳性词对合法集 {效价,活性方向,安全性,UNKNOWN} 与交集整体无交
    # → P2 判死。"达标/放行/合格"踩"反应"尾部子串→效价键、"重复"
    # 踩"应答"→效价键——会把交集拖入效价复活阳性词对，一律用
    # "可控/可信/可复现"替代。
    (
        "候选分子 WQ-207 的方向之争（工艺放大与科学依据）",
        "偶联工艺复核判为阳性，遗传学证据不足，DAR 分布收窄、纯化收率可信",
        "通路必需性结论判为阴性，冻干工艺收率有待复现，科学依据不足",
        "阳性",
    ),
    # 激活（切 成药性+安全性）：A 句=成药性句 + 肯定形"激活"（pos_a），
    # B 句=成药性+安全性句 + ka 否定覆盖"不激活"（"激活"被"不"紧邻
    # 覆盖 → neg_b；否定前缀必须**紧邻** ka，"X未见激活"类间隔形不
    # 生效；"抑制"是活性方向键严禁入句，故否定侧只能走 ka 覆盖）——
    # 极性分居 (正_a ∧ 负_b)。交集={成药性,安全性}
    # （"冻干/处方/纯化/制剂"双方、"安全性"双方），与 激活 词对合法集
    # {活性方向,表达调控,机制} 整体无交 → P2 判死。句面严禁
    # 激酶/磷酸化/酶活/通路/协同/表达/靶点（合法切面键——注入即复活）；
    # "药物/候选物"零对象信号而"化合物"命中 DXd。
    (
        "候选分子 WQ-101 的方向之争（冻干处方与心脏安全性）",
        "冻干处方筛选确认辅料的激活效应可控，纯化工艺参数全部可控",
        "同一制剂的 hERG 复核显示心肌细胞不激活，"
        "制剂安全性与冻干处方评估意见均为风险可控",
        "激活",
    ),
    # 激活（切 成药性+安全性）：同 cross#2 构造模式复用——A 句=
    # 成药性句 + 肯定形"激活"（pos_a），B 句=成药性+安全性句 + ka
    # 否定覆盖"不激活"（neg_b），极性分居 (正_a ∧ 负_b)。交集
    # ={成药性,安全性}，与 激活 词对合法集 {活性方向,表达调控,机制}
    # 整体无交 → P2 判死；零对象信号 → 门禁 UNKNOWN。
    (
        "候选抗体 WQ-118 的心脏安全性之争（冻干处方与 hERG 复核）",
        "冻干处方筛选确认辅料的激活效应可控，纯化工艺参数全部可控",
        "同一抗体制剂的 hERG 复核显示心肌细胞不激活，"
        "制剂安全性与冻干处方评估意见均为风险可控",
        "激活",
    ),
    # 激活（切 成药性）：A 句=成药性句 + 肯定形"激活"（pos_a），
    # B 句=成药性句 + ka 否定覆盖"不激活"（neg_b）——交集={成药性}
    # （"合成/路线/收率/给药方案"双方对称，topic 零词典键），与
    # 激活 词对合法集 {活性方向,表达调控,机制} 整体无交 → P2 判死。
    # 激活 否定侧只能走 ka 覆盖（"抑制"是活性方向键严禁入句，
    # 否则 激活/抑制 词对在活性方向复活——cross#3 教训）。
    (
        "候选肽 WQ-450 的方向之争（合成路线与给药方案）",
        "合成路线复核确认片段的激活效应可控，纯化工艺参数一致",
        "同一给药方案的读数复核显示该片段不激活，合成路线与工艺收率一致",
        "激活",
    ),
]

#: 同方向程度分歧模板（aspect 拦截面）——**暂空池**。
#: v0.3 词表全为 directional 极性，degree 拦截面不存在（模块 docstring
#: 已记录：本类以切面错位构造）。但现有 8 词对均无法稳定构造：
#:   - 阳性/阴性 为真对立词对，一侧肯定形、一侧否定形即命中，
#:     不存在"同一存在性命题双方同判阳性"的 degree 语义；
#:   - 存在/不存在（negated_b）合法切面 {结构可信度,成药性,机制,
#:     安全性,UNKNOWN} 覆盖过宽，六切面下交集极易含合法切面而复活。
#: 待词表演进引入 degree 极性词对（如 高表达/表达偏低）或新增
#: 零合法切面词对后填充；_selfcheck_aspect 对空池无操作，
#: 填充模板后自动被强制校验。四类语义（含 degree_only 的
#: ground_truth 定义）保留在模块 docstring 与 DDB schema 中。
#: 中文词典键为子串匹配——"受"会误命中"必需"、"转录本"会误命中"转录"、
#: 裸"偶联"会误命中成药性、**"抑制"是独立键**（命中即复活
#: 激活/抑制 词对，安全句严禁出现"抑制"）；英文词按 ASCII 词边界
#: （大小写不敏感），全大写的中文拼音缩写式写法（IC50）可规避误命中。
#: 改句后必须复算切面集与 hit/alive 矩阵。
DEGREE_TEMPLATES: list[tuple[str, str, str, str]] = []

#: 同切面异对象模板（ValidityGate DIFFERENT 拦截面）：
#: 双方句子各挂 OBJECT_ALIASES 专名且键集不相交；v0.3 流水线顺序是
#: P2 aspect 存活 → P3 门禁——aspect 先杀词对则门禁测不到，而强度分级
#: （同一对象上的方向对立强度更高）恰与门禁互补，故本类**故意让目标
#: 词对在 aspect 存活**，门禁作为唯一拦截面。切面纪律：双方句子只能命中
#: 词对合法切面内的词（合法集外词会把交集拖出合法集，词对被 aspect
#: 判死、门禁效应归零）——激活/抑制 词对合法集
#: {活性方向,表达调控,机制}，句面出现 安全性/效价/结构可信度/成药性
#: 触发词即违规，改句必须复算。
#: topic 严禁含任何词典专名（topic 参与双方对象扫描域，注入第二专名
#: 会污染键集判定）。
DIFFERENT_OBJECT_TEMPLATES: list[tuple[str, str, str, str]] = [
    (
        "双靶点共抑制策略中两条通路的激酶方向之争",
        "EGFR L858R 激酶活性实验显示候选分子对该突变体呈直接抑制，"
        "磷酸化水平同步下降",
        "KRAS G12C 结合口袋实验显示 sotorasib 共价结合位点被占据，"
        "GTPase 活性检测呈激活状态",
        "抑制",
    ),
    (
        "两个独立项目的生物标志物质控复核（队列基线之争）",
        "HER2 在入组队列中呈高表达，IHC 3+ 占比与既往文献一致，质控结论通过",
        "CD276（B7-H3）在配对队列中呈低表达，流式复核结果集中，"
        "另一项目的标志物基线数据独立完整",
        "高表达",
    ),
    (
        "两个独立项目的激酶方向争议（共价位点占据之争）",
        "EGFR C797S 位点的共价结合实验显示激酶活性受到抑制，"
        "结合动力学读数稳定",
        "KRAS Y96D 位点的变构口袋实验显示 GTPase 活性维持激活状态，"
        "两个项目的结论各自独立成立",
        "抑制",
    ),
]

#: 对象指代不明模板（ValidityGate AMBIGUOUS 降级面 = 源码第 4 分支）：
#: 进入 AMBIGUOUS 的充要条件是双方对象键集均非空、不相交、且无 MOLECULE
#: 专名（kind==MOLECULE 为空）。对象抽取只认两级词典：OBJECT_ALIASES
#: 专名（=MOLECULE，禁用）与 PHENOMENON_LEXICON 现象词（=PHENOMENON +
#: 宿主键，HOST 不计 MOLECULE）——故 AMBIGUOUS 的唯一构造面是
#: **双侧词典现象词且键集不相交**（现象键与宿主键均不同；同现象词
#: 或共宿主会把键集拉成交集 → SAME 放行；零词典句键集为空 → UNKNOWN）。
#: 切面侧现象词天然带出 机制/活性方向 等切面，**正是因此**
#: 目标词对的合法集必须覆盖现象词带出的切面（存活或 UNKNOWN 通配
#: 均可），否则 aspect 在 P2 先杀词对、门禁 P3 根本测不到——
#: 阳性/阴性 合法集 {效价,活性方向,安全性,UNKNOWN} 与 机制 交集
#: 为空，本类**禁用 阳性 目标词**；存在/不存在（合法集含机制）
#: 为可行目标词。
#: 句内其它词严禁命中 OBJECT_ALIASES 或 ASPECT_TRIGGER_LEXICON
#: （如"批次"是成药性切面词、"剂量限制"踩 剂量/限制 子串）。
#: 否定句纪律：bystander 是机制键——否定句严禁其英文形，宿主词
#: 同理（同一化合物→同一供体群）。
AMBIGUOUS_TEMPLATES: list[tuple[str, str, str, str]] = [
    # 存在（旁观者效应 vs FcγR 交联）：现象词带出 机制，
    # 存在/不存在 合法集含机制 → 存活进门禁 → AMBIGUOUS 降级。
    (
        "复核的读数讨论",
        "复核读数提示主要清除途径以原型排泄为主，旁观者效应的贡献存在，"
        "定量下限以上的信号可检出",
        "同一供体群在不同给药的复核提示 FcγR 交联依赖的贡献不存在，"
        "代偿清除比例低于定量下限",
        "存在",
    ),
    # 存在（耐药机制 vs 选择性优势）：A 句现象词带出 活性方向
    # （耐药键），存在 词对合法集含机制——A 句切面={机制,活性方向}；
    # B 句必须同样带 机制（交集含机制才存活），但严禁带 活性方向
    # （选择性优势键本身非活性方向键，OK）。bystander/交联 严禁入
    # A 句（否定句纪律）。
    (
        "联合策略的读数讨论",
        "联合给药的复核读数提示耐药机制的驱动成分存在，单药臂的信号可检出",
        "同一方案在选择性窗口的机制复核提示选择性优势的贡献不存在，"
        "单药臂的读数低于定量下限",
        "存在",
    ),
    # 存在（内化速率 vs 选择性优势）：双方现象词键集不相交；
    # A 句带 内化（机制），B 句必须也带 机制（加"机制复核"），
    # 交集={机制} 存活 → AMBIGUOUS。
    (
        "构效关系的读数讨论",
        "构效系列的复核读数提示内化速率相关的贡献存在，"
        "高亲和力分子的信号可检出",
        "同一配体系列在选择性窗口的机制复核提示选择性优势的贡献不存在，"
        "低亲和力分子的读数低于定量下限",
        "存在",
    ),
]

#: L1.5 的 neutral 干扰证据模板（句内严禁任何词对任一侧关键词，
#: 自检强制校验；与 L1 NEUTRAL 纪律一致）。
#: 对象词典纪律：干扰证据参与全对扫描，若句子命中 OBJECT_ALIASES
#: 专名或 PHENOMENON_LEXICON 现象词（化合物→DXd、实验动物→EGFR
#: 等宽子串陷阱），会给无关证据对引入对象信号、改变门禁态；
#: L15_NEUTRAL_ZERO_OBJECT 为强制标记——池内句子必须**全部零对象
#: 词典命中**（_neutral_evidence 逐句强制校验，ObjectExtractor
#: 权威扫描域含 topic，但 topic 属案例核心证据的上下文，
#: 干扰句本身按零 topic 扫描）。
L15_NEUTRAL_ZERO_OBJECT = True
L15_NEUTRAL_SENTENCES: list[str] = [
    "该项目的台账经过双人复核，样品溯源记录完整",
    "筛选平台的加样精度校准在验收标准之内，板间效应可控",
    "组内复核意见与项目周会记录保持一致，待补充的原始数据已列入下周计划",
    "动物设施的环境监测数据处于规定范围，福利审查通过",
    "文库的质检指标满足建库要求，重复性达到放行限度",
    "知情同意流程符合伦理批件要求，随访计划按期执行",
]


# ============================================================================
# 数据类（与 DdbCase 同构 + L1.5 审计字段，便于消融脚本无损复用）
# ============================================================================

@dataclass
class EvidenceDict:
    """
    单条证据的可序列化字典（字段与 consensus_court.Evidence 一一对应）。

    Attributes:
        evidence_id: 证据全局唯一 ID（案例内唯一）
        source: 来源谱系键
        content: 证据正文（模板句；负例语义的全部陷阱已内嵌其中）
        role: 提交证据的虚拟实验室角色
        timestamp: ISO 8601 字符串或 None（制造 recency 梯度）
        reference_ids: 引用 ID 列表（恒非空——L1.5 不注入形式缺陷）
        metadata: 仅含 category 审计字段（严禁机制显式字段）
    """

    evidence_id: str
    source: str
    content: str
    role: str
    timestamp: Optional[str]
    reference_ids: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class DdbL15Case:
    """
    DDB-L1.5 单个负例对照案例。

    Attributes:
        case_id: 案例编号（DDB-L15-0001 形式）
        topic: 药物发现议题短语
        polarity: 恒为 "no_contradiction"
        category: 负例类别（cross_aspect/degree_only/different_object_gate/
            ambiguous_reference）
        evidence_type: 恒为 "clean"
        defect_flags: 恒为空列表
        evidence_count: 证据条数（含干扰证据）
        evidences: 证据条目列表
        ground_truth: 真值标签（has_true_contradiction=False + 期望行为三元组）
    """

    case_id: str
    topic: str
    polarity: str
    category: str
    evidence_type: str
    evidence_count: int
    defect_flags: list[str]
    evidences: list[EvidenceDict]
    ground_truth: dict

    def to_dict(self) -> dict:
        """转为可 JSON 序列化的普通字典。"""
        return asdict(self)


def _to_evidence_obj(ev: EvidenceDict) -> Evidence:
    """
    把 EvidenceDict 还原为 consensus_court.Evidence（自检用）。

    Args:
        ev: 证据字典对象

    Returns:
        consensus_court.Evidence 实例
    """
    timestamp = None
    if ev.timestamp:
        timestamp = datetime.fromisoformat(ev.timestamp)
    return Evidence(
        evidence_id=ev.evidence_id,
        source=ev.source,
        content=ev.content,
        role=ev.role,
        timestamp=timestamp,
        reference_ids=list(ev.reference_ids),
        metadata=dict(ev.metadata or {}),
    )


def _spec_for_template(keyword: str) -> KeywordPairSpec:
    """
    取模板目标词对规格并校验其极性为 directional（L1.5 拦截面语义前提）。

    degree 极性词对在 v0.3 规格表中不存在（全部 directional），
    若词表演进引入 degree 词对，本生成器模板需重设计，提前报错。

    Args:
        keyword: 模板的目标词对关键侧词

    Returns:
        KeywordPairSpec
    """
    spec = _spec_of(keyword)
    if spec.polarity != POLARITY_DIRECTIONAL:
        raise RuntimeError(
            f"词对 {spec.keyword_a}/{spec.keyword_b} 极性为 degree，"
            f"与 L1.5 模板设计前提（directional）不符"
        )
    return spec


def _case_aspect_hits(content: str, topic: str) -> list[tuple[str, str]]:
    """
    逐句复算切面词典命中键清单（(切面名, 命中键)，含 topic 联合语料）。

    诊断专用：切面集合相同 ≠ 命中键相同——topic 注入的键只有靠
    键级清单才能暴露（AspectInferrer 每切面 break 只报首键）。

    Args:
        content: 证据正文
        topic: 争议主题

    Returns:
        (切面中文名, 命中词典键) 列表
    """
    from app.core.virtual_lab.rubric_engine import ASPECT_TRIGGER_LEXICON

    text = f"{content} {topic}"
    hits: list[tuple[str, str]] = []
    for asp, keys in ASPECT_TRIGGER_LEXICON.items():
        for key in keys:
            if AspectInferrer._lexicon_hit(key, text):
                hits.append((asp.value, key))
    return hits


class DdbL15Generator:
    """
    DDB-L1.5 负例对照基准生成器。

    模板决定负例语义（人工设计、可审计），随机仅决定：来源/角色/
    时间戳/干扰证据数量与位置。所有生成期断言直接调用源码机制组件
    （AspectInferrer / ObjectExtractor / _pair_hit_by_polarity /
    ValidityGate / AsymmetricRubric._pair_alive），确保"构造即验证"，
    机制演进（词典/规格表变更）会让坏样本立即暴露而非悄然泄漏。
    """

    def __init__(self, seed: int = 42):
        """
        Args:
            seed: 随机种子（写入报告，保证可复现）
        """
        self._rng = random.Random(seed)
        self._inferrer = AspectInferrer()
        self._extractor = ObjectExtractor()
        self._gate = ValidityGate(self._extractor)
        self._case_seq = 0

    def _sample_source(self) -> str:
        """按权重从来源谱系采样一个来源键。"""
        return self._rng.choices(
            population=list(SOURCE_POOL.keys()),
            weights=list(SOURCE_POOL.values()),
            k=1,
        )[0]

    def _sample_source_pair(self) -> tuple[str, str]:
        """采样两个 **不同** 来源（同一来源不对比，负例陷阱必须跨来源）。"""
        return tuple(self._rng.sample(list(SOURCE_POOL.keys()), 2))

    def _sample_timestamp(self) -> Optional[str]:
        """采样证据时间戳：约 1/3 为 None，其余落在距今 0-6 年内。"""
        if self._rng.random() < 0.33:
            return None
        age_days = self._rng.randint(0, 2190)
        ts = datetime.utcnow() - timedelta(days=age_days)
        return ts.isoformat()

    def _make_reference_ids(self) -> list[str]:
        """生成非空引用 ID 列表（L1.5 严禁形式缺陷，引用必须齐全）。"""
        prefix = self._rng.choice(["PMID", "CHEMBL", "UNI", "DRUGBANK", "EXP"])
        return [f"{prefix}-{self._rng.randint(100000, 9999999)}"]

    def _make_evidence(
        self,
        case_id: str,
        index: int,
        content: str,
        category: str,
        source: Optional[str] = None,
    ) -> EvidenceDict:
        """
        组装单条证据（自动采样来源/角色/时间戳，引用恒非空）。

        Args:
            case_id: 所属案例编号
            index: 案例内序号
            content: 证据正文（模板句）
            category: 负例类别（写入 metadata 供审计）
            source: 指定来源；None 时按权重采样

        Returns:
            EvidenceDict
        """
        if len(content.strip()) < 10:
            raise AssertionError(f"L1.5 证据内容异常偏短（<10 字符）: {content[:40]}")
        return EvidenceDict(
            evidence_id=f"{case_id}-EV{index:02d}",
            source=source or self._sample_source(),
            content=content,
            role=self._rng.choice(ROLE_POOL),
            timestamp=self._sample_timestamp(),
            reference_ids=self._make_reference_ids(),
            metadata={"category": category},
        )

    def _neutral_evidence(self, case_id: str, index: int, category: str) -> EvidenceDict:
        """
        生成一条中性干扰证据（句内严禁任何词对任一侧关键词，
        且零对象词典命中——L15_NEUTRAL_ZERO_OBJECT 纪律）。

        Args:
            case_id: 所属案例编号
            index: 案例内序号
            category: 负例类别

        Returns:
            EvidenceDict

        Raises:
            AssertionError: 句面出现词对任一侧关键词，或
                L15_NEUTRAL_ZERO_OBJECT 开启时句面命中对象词典
                （OBJECT_ALIASES 专名 / PHENOMENON_LEXICON 现象词）。
        """
        sentence = self._rng.choice(L15_NEUTRAL_SENTENCES)
        for spec in KEYWORD_PAIR_SPECS:
            for kw in (spec.keyword_a, spec.keyword_b):
                if kw in sentence:
                    raise AssertionError(
                        f"中性句出现词对关键词 {kw}，违反 L1.5 纪律: {sentence[:40]}"
                    )
        if L15_NEUTRAL_ZERO_OBJECT:
            objs = self._extractor.extract(
                Evidence(
                    evidence_id=f"{case_id}-EV{index:02d}",
                    source="",
                    content=sentence,
                    role="",
                ),
                "",
            )
            if objs:
                raise AssertionError(
                    f"中性句命中对象词典 {[(o.kind, o.canonical_key) for o in objs]}"
                    f"，违反 L15_NEUTRAL_ZERO_OBJECT 纪律: {sentence[:40]}"
                )
        return self._make_evidence(case_id, index, sentence, category)

    def _build_case_evidences(
        self,
        case_id: str,
        category: str,
        content_a: str,
        content_b: str,
    ) -> list[EvidenceDict]:
        """
        组装案例证据集：两条核心证据（跨来源）+ 1-2 条中性干扰证据。

        干扰证据插入随机位置，避免"核心证据总在开头"的位置偏置。

        Args:
            case_id: 案例编号
            category: 负例类别
            content_a: 核心证据 A 正文
            content_b: 核心证据 B 正文

        Returns:
            证据字典列表（含干扰，顺序随机）
        """
        source_a, source_b = self._sample_source_pair()
        core = [
            self._make_evidence(case_id, 1, content_a, category, source=source_a),
            self._make_evidence(case_id, 2, content_b, category, source=source_b),
        ]
        distractors = [
            self._neutral_evidence(case_id, i, category)
            for i in range(3, self._rng.randint(4, 6))
        ]
        evidences = core + distractors
        self._rng.shuffle(evidences)
        return evidences

    def _make_cross_aspect_case(self, tpl: tuple[str, str, str, str]) -> DdbL15Case:
        """
        构造跨切面伪冲突负例（aspect 拦截面）。

        Args:
            tpl: (topic, A 句, B 句, 目标词对关键侧词)

        Returns:
            DdbL15Case（category=cross_aspect）
        """
        topic, content_a, content_b, keyword = tpl
        spec = _spec_for_template(keyword)
        self._case_seq += 1
        case_id = f"DDB-L15-{self._case_seq:04d}"
        evidences = self._build_case_evidences(case_id, "cross_aspect", content_a, content_b)
        try:
            case = DdbL15Case(
                case_id=case_id,
                topic=topic,
                polarity="no_contradiction",
                category="cross_aspect",
                evidence_type="clean",
                evidence_count=len(evidences),
                defect_flags=[],
                evidences=evidences,
                ground_truth={
                    "has_true_contradiction": False,
                    "defect_only": False,
                    "mechanism": "aspect",
                    "expected_full": "no_conviction",
                    "expected_v02": "convicted",
                },
            )
            self._selfcheck_aspect(case, spec)
        except AssertionError as exc:
            # 自检失败时附诊断上下文：句面/topic 注入的词典键清单——
            # 切面集合相同 ≠ 命中键相同，靠键级清单才能定位误伤词
            raise AssertionError(
                f"{exc}\n  模板: topic={topic!r}\n  A={content_a!r}\n  B={content_b!r}\n"
                f"  句面A={_case_aspect_hits(content_a, '')}\n"
                f"  句面B={_case_aspect_hits(content_b, '')}\n"
                f"  含topic A={_case_aspect_hits(content_a, topic)}\n"
                f"  含topic B={_case_aspect_hits(content_b, topic)}"
            ) from exc
        return case

    def _make_degree_case(self, tpl: tuple[str, str, str, str]) -> DdbL15Case:
        """
        构造同方向程度分歧负例（aspect 拦截面）。

        Args:
            tpl: (topic, A 句, B 句, 目标词对关键侧词)

        Returns:
            DdbL15Case（category=degree_only）
        """
        topic, content_a, content_b, keyword = tpl
        spec = _spec_for_template(keyword)
        self._case_seq += 1
        case_id = f"DDB-L15-{self._case_seq:04d}"
        evidences = self._build_case_evidences(case_id, "degree_only", content_a, content_b)
        case = DdbL15Case(
            case_id=case_id,
            topic=topic,
            polarity="no_contradiction",
            category="degree_only",
            evidence_type="clean",
            evidence_count=len(evidences),
            defect_flags=[],
            evidences=evidences,
            ground_truth={
                "has_true_contradiction": False,
                "defect_only": False,
                "mechanism": "aspect",
                "expected_full": "no_conviction",
                "expected_v02": "convicted",
            },
        )
        self._selfcheck_aspect(case, spec)
        return case

    def _make_different_object_case(self, tpl: tuple[str, str, str, str]) -> DdbL15Case:
        """
        构造同切面异对象伪冲突负例（ValidityGate DIFFERENT 拦截面）。

        Args:
            tpl: (topic, A 句, B 句, 目标词对关键侧词)

        Returns:
            DdbL15Case（category=different_object_gate）
        """
        topic, content_a, content_b, keyword = tpl
        spec = _spec_for_template(keyword)
        self._case_seq += 1
        case_id = f"DDB-L15-{self._case_seq:04d}"
        evidences = self._build_case_evidences(case_id, "different_object_gate", content_a, content_b)
        case = DdbL15Case(
            case_id=case_id,
            topic=topic,
            polarity="no_contradiction",
            category="different_object_gate",
            evidence_type="clean",
            evidence_count=len(evidences),
            defect_flags=[],
            evidences=evidences,
            ground_truth={
                "has_true_contradiction": False,
                "defect_only": False,
                "mechanism": "validity_gate",
                "expected_full": "no_conviction",
                "expected_v02": "convicted",
            },
        )
        self._selfcheck_gate(case, spec, expect="DIFFERENT")
        return case

    def _make_ambiguous_case(self, tpl: tuple[str, str, str, str]) -> DdbL15Case:
        """
        构造对象指代不明负例（ValidityGate AMBIGUOUS 降级面）。

        Args:
            tpl: (topic, A 句, B 句, 目标词对关键侧词)

        Returns:
            DdbL15Case（category=ambiguous_reference）
        """
        topic, content_a, content_b, keyword = tpl
        spec = _spec_for_template(keyword)
        self._case_seq += 1
        case_id = f"DDB-L15-{self._case_seq:04d}"
        evidences = self._build_case_evidences(case_id, "ambiguous_reference", content_a, content_b)
        case = DdbL15Case(
            case_id=case_id,
            topic=topic,
            polarity="no_contradiction",
            category="ambiguous_reference",
            evidence_type="clean",
            evidence_count=len(evidences),
            defect_flags=[],
            evidences=evidences,
            ground_truth={
                "has_true_contradiction": False,
                "defect_only": False,
                "mechanism": "validity_gate",
                "expected_full": "pending_ambiguous",
                "expected_v02": "convicted",
            },
        )
        self._selfcheck_gate(case, spec, expect="AMBIGUOUS")
        return case

    def _core_pair(self, case: DdbL15Case) -> tuple[Evidence, Evidence]:
        """
        从案例证据集中取回两条核心证据（按 evidence_id 序号 EV01/EV02）。

        Args:
            case: 待检查案例

        Returns:
            (核心证据 A, 核心证据 B)
        """
        by_id = {ev.evidence_id: _to_evidence_obj(ev) for ev in case.evidences}
        return by_id[f"{case.case_id}-EV01"], by_id[f"{case.case_id}-EV02"]

    def _pair_verdict(self, case: DdbL15Case) -> tuple[bool, bool]:
        """
        按源码 P1→P3 流水线顺序复算核心证据对的 (convicted, ambiguous)。

        与 _check_pair_contradiction 循环语义严格同构：词对按规格表
        顺序遍历——极性命中 → aspect 判死则 continue 下一词对 →
        存活词对进门禁：DIFFERENT 整对拦截（无罪）、AMBIGUOUS 整对
        降级（无罪）、SAME/UNKNOWN 定罪。注意 aspect 判死是**词对级**
        continue 而非整对放行——后续存活词对仍可定罪（自检曾在此
        语义上误判，复算必须逐词对推进）。

        Args:
            case: 待检查案例

        Returns:
            (convicted, ambiguous)：v0.3 全开配置下是否定罪 / 是否降待举证
        """
        from app.core.virtual_lab.rubric_engine import AsymmetricRubric, GateDecision

        ev_a, ev_b = self._core_pair(case)
        ca, cb = ev_a.content.lower(), ev_b.content.lower()
        aspects_a = self._inferrer.infer(ev_a, case.topic)
        aspects_b = self._inferrer.infer(ev_b, case.topic)
        gate = self._gate.check(ev_a, ev_b, case.topic)
        for s in KEYWORD_PAIR_SPECS:
            if not _pair_hit_by_polarity(ca, cb, s):
                continue
            alive, _shared = AsymmetricRubric._pair_alive(s, aspects_a, aspects_b)
            if not alive:
                continue
            # 存活词对到达门禁（P3）：门禁结论对整对生效
            if gate.decision == GateDecision.DIFFERENT_OBJECT:
                return False, False
            if gate.decision == GateDecision.AMBIGUOUS:
                return False, True
            # SAME_OBJECT / UNKNOWN → 定罪通道（P4/P5 必然产出矛盾对）
            return True, False
        return False, False

    def _selfcheck_aspect(self, case: DdbL15Case, spec: KeywordPairSpec) -> None:
        """
        aspect 类负例自检（与源码 _check_pair_contradiction 流水线同构）：
        1. 核心证据对必须命中**某个**词对极性通道（负例对 v0.2 的鉴别力）；
        2. 按 P1→P3 顺序复算（_pair_verdict），v0.3 全开必须判无罪——
           拦截来源必须是 aspect（存活词对为空 → 无罪），门禁不得参与
           （门禁复算必须 UNKNOWN，本类的拦截面归 aspect 独占）；
        3. 参与判定的核心证据切面不得全 UNKNOWN（词典必须覆盖负例句面
           术语，否则通配放行使拦截面消失）。

        Args:
            case: 待检查案例
            spec: 模板声明的目标词对规格（用于报告，实际判定遍历全词对）

        Raises:
            AssertionError: 任一断言失败（模板/词典/源码演进导致拦截面丢失）
        """
        ev_a, ev_b = self._core_pair(case)
        ca, cb = ev_a.content.lower(), ev_b.content.lower()
        aspects_a = self._inferrer.infer(ev_a, case.topic)
        aspects_b = self._inferrer.infer(ev_b, case.topic)

        hit_specs = [s for s in KEYWORD_PAIR_SPECS if _pair_hit_by_polarity(ca, cb, s)]
        if not hit_specs:
            raise AssertionError(
                f"{case.case_id}: 核心证据对未命中任何词对极性通道"
                f"（模板声明目标 {spec.keyword_a}/{spec.keyword_b}），负例无鉴别力"
            )
        convicted, ambiguous = self._pair_verdict(case)
        if convicted:
            from app.core.virtual_lab.rubric_engine import AsymmetricRubric

            alive_names = [
                (s.keyword_a, s.keyword_b) for s in hit_specs
                if AsymmetricRubric._pair_alive(s, aspects_a, aspects_b)[0]
            ]
            raise AssertionError(
                f"{case.case_id}: v0.3 全开复算仍定罪，存活词对 {alive_names}"
                f"（切面 {sorted(a.value for a in aspects_a)} vs "
                f"{sorted(b.value for b in aspects_b)}）——aspect 拦截面丢失"
            )
        if ambiguous:
            raise AssertionError(
                f"{case.case_id}: v0.3 全开复算降为待举证——aspect 类案例的"
                f"门禁必须 UNKNOWN（对象信号缺失），当前门禁参与判定"
            )
        gate = self._gate.check(ev_a, ev_b, case.topic)
        if gate.decision.value != "UNKNOWN":
            raise AssertionError(
                f"{case.case_id}: aspect 类案例门禁复算为 {gate.decision.value}，"
                f"期望 UNKNOWN——拦截面归属漂移（aspect 案例不得触发门禁）"
            )
        # 至少一条核心证据有明确切面（全 UNKNOWN 时通配放行，拦截面不存在）
        if aspects_a <= {Aspect.UNKNOWN} and aspects_b <= {Aspect.UNKNOWN}:
            raise AssertionError(
                f"{case.case_id}: 双方核心证据切面推断全 UNKNOWN，"
                f"Aspect 词典对模板句未覆盖（通配会放行，负例失效）"
            )

    def _selfcheck_gate(self, case: DdbL15Case, spec: KeywordPairSpec, expect: str) -> None:
        """
        门禁类负例自检：
        1. 核心证据对必须命中目标词对极性通道；
        2. aspect 不得提前杀掉目标词对——两种合法形态：
           * DIFFERENT（异对象）：词对必须存活（双方各有明确切面且交集
             内词对合法），否则 aspect 先拦截、门禁效应测不到；
           * AMBIGUOUS（指代不明）：双方对象键集均非空且不相交（否则
              SAME 放行或 UNKNOWN 缺信号），且无任何 MOLECULE 专名佐证；
              切面侧不得把目标词对判死（通配或存活均可）；
        3. ValidityGate 复算必须给出期望态（DIFFERENT / AMBIGUOUS）。

        Args:
            case: 待检查案例
            spec: 目标词对规格
            expect: 期望门禁态（DIFFERENT / AMBIGUOUS）

        Raises:
            AssertionError: 任一断言失败
        """
        from app.core.virtual_lab.rubric_engine import AsymmetricRubric, GateDecision

        ev_a, ev_b = self._core_pair(case)
        if not _pair_hit_by_polarity(ev_a.content.lower(), ev_b.content.lower(), spec):
            raise AssertionError(
                f"{case.case_id}: 核心证据对未命中目标词对极性通道，负例无鉴别力"
            )
        convicted, ambiguous = self._pair_verdict(case)
        expect_convicted = expect == "SAME" or expect == "UNKNOWN"
        expect_ambiguous = expect == "AMBIGUOUS"
        if convicted != expect_convicted or ambiguous != expect_ambiguous:
            raise AssertionError(
                f"{case.case_id}: v0.3 全开复算 (convicted={convicted},"
                f" ambiguous={ambiguous})，期望 (convicted={expect_convicted},"
                f" ambiguous={expect_ambiguous})——拦截面漂移或词对被 aspect 先杀"
            )
        gate = self._gate.check(ev_a, ev_b, case.topic)
        expect_decision = {
            "SAME": GateDecision.SAME_OBJECT,
            "UNKNOWN": GateDecision.UNKNOWN,
            "DIFFERENT": GateDecision.DIFFERENT_OBJECT,
            "AMBIGUOUS": GateDecision.AMBIGUOUS,
        }[expect]
        if gate.decision != expect_decision:
            raise AssertionError(
                f"{case.case_id}: 门禁复算为 {gate.decision.value}，期望 "
                f"{expect_decision.value}（对象抽取器/词典演进导致拦截面漂移）"
            )

    # ------------------------------------------------------------------
    # 批量生成
    # ------------------------------------------------------------------

    def generate(self, per_template: int = 8) -> list[dict]:
        """
        批量生成 DDB-L1.5 负例基准。

        采样策略：每条模板轮转 per_template 次（同模板不同来源/角色/
        时间戳/干扰组合），四类模板轮转顺序交错后打乱，保证类别均衡。

        Args:
            per_template: 每条模板的重复采样次数（默认 8）

        Returns:
            案例字典列表（schema 见模块 docstring）
        """
        plan: list[tuple[str, tuple[str, str, str, str]]] = []
        for rep in range(per_template):
            for tpl in CROSS_ASPECT_TEMPLATES:
                plan.append(("cross_aspect", tpl))
            for tpl in DEGREE_TEMPLATES:
                plan.append(("degree_only", tpl))
            for tpl in DIFFERENT_OBJECT_TEMPLATES:
                plan.append(("different_object_gate", tpl))
            for tpl in AMBIGUOUS_TEMPLATES:
                plan.append(("ambiguous_reference", tpl))
        self._rng.shuffle(plan)

        cases: list[DdbL15Case] = []
        for category, tpl in plan:
            if category == "cross_aspect":
                cases.append(self._make_cross_aspect_case(tpl))
            elif category == "degree_only":
                cases.append(self._make_degree_case(tpl))
            elif category == "different_object_gate":
                cases.append(self._make_different_object_case(tpl))
            else:
                cases.append(self._make_ambiguous_case(tpl))
        return [c.to_dict() for c in cases]

    @staticmethod
    def validate(cases: list[dict]) -> None:
        """
        对生成结果做最终全量校验（测试与 main 均可复用）：
        1. case_id 全局唯一；
        2. 全部案例 polarity=no_contradiction、has_true_contradiction=False；
        3. 期望标签齐全（mechanism / expected_full / expected_v02）；
        4. 证据引用恒非空、metadata 严禁机制显式字段（防绕过词典路径）；
        5. 类别覆盖：cross_aspect / different_object_gate /
           ambiguous_reference 必须非空（aspect 与门禁两类拦截面）。
           degree_only 当前为暂空池（v0.3 词表全 directional，
           词对语义无法构造稳定拦截面，见模板池注释），故**豁免
           非空校验**——词表演进后填充模板即自动恢复覆盖。

        Args:
            cases: 案例字典列表

        Raises:
            AssertionError: 任一完整性约束被破坏
        """
        seen: set[str] = set()
        categories: set[str] = set()
        for case in cases:
            cid = case["case_id"]
            if cid in seen:
                raise AssertionError(f"case_id 重复: {cid}")
            seen.add(cid)
            categories.add(case["category"])
            if case["polarity"] != "no_contradiction":
                raise AssertionError(f"{cid}: L1.5 案例 polarity 异常")
            gt = case["ground_truth"]
            if gt["has_true_contradiction"]:
                raise AssertionError(f"{cid}: 负例案例真值标签被污染")
            for key in ("mechanism", "expected_full", "expected_v02"):
                if key not in gt:
                    raise AssertionError(f"{cid}: ground_truth 缺少 {key}")
            for ev in case["evidences"]:
                if not ev["reference_ids"]:
                    raise AssertionError(f"{cid}/{ev['evidence_id']}: 引用为空")
                md = ev.get("metadata") or {}
                if {"aspect", "aspects", "target", "molecule", "locus"} & set(md):
                    raise AssertionError(
                        f"{cid}/{ev['evidence_id']}: metadata 含机制显式字段，"
                        f"负例必须考验词典推断路径"
                    )
        required = {"cross_aspect", "different_object_gate", "ambiguous_reference"}
        missing = required - categories
        if missing:
            raise AssertionError(f"负例类别缺失: {missing}")


def generate(per_template: int = 8, seed: int = 42) -> list[dict]:
    """
    模块级便捷入口：以指定规模与种子生成 DDB-L1.5 负例基准。

    Args:
        per_template: 每条模板重复次数（默认 8 → 12 模板 × 8 = 96 例，
            degree_only 暂空池不计）
        seed: 随机种子（默认 42）

    Returns:
        案例字典列表（已通过 DdbL15Generator.validate 全量校验）
    """
    gen = DdbL15Generator(seed=seed)
    cases = gen.generate(per_template=per_template)
    DdbL15Generator.validate(cases)
    return cases


def main() -> None:
    """
    命令行入口：生成默认 120 例并写入 tests/ddb/l15_cases.json。

    运行方式：cd backend && python -m tests.ddb.l15_generator
    """
    out_path = Path(__file__).resolve().parent / "l15_cases.json"
    cases = generate()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    dist: dict = {}
    for c in cases:
        dist[c["category"]] = dist.get(c["category"], 0) + 1
    print(f"DDB-L1.5 负例基准生成完成：共 {len(cases)} 例，类别分布 {dist} → {out_path}")


if __name__ == "__main__":
    main()
