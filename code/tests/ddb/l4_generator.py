"""
DDB-L4 强度标注层基准生成器（Dispute Benchmark - Level 4）

用途（需求文档 design/l4-intensity-annotation-spec.md）：
    为 rubric_engine._grade_intensity（前端强度分级器，大纲 §4.4）建立
    可复核的标注金标层，覆盖其判定表全部可达输入分支：
        S1 硬约束短路（v0.3 前端不可达，gold 计算器仍同构实现，spec D4）
        S2 极性基线（directional=2 / degree=1；v0.3 词对全 directional，D3）
        S3 来源可信度差加成（|Δw| ≥ 0.30 → +1，SOURCE_WEIGHTS 口径）
        S4 门禁 SAME_OBJECT 佐证（base==2 → 3）
    每例携带 expected_intensity 自动金标 + gold_derivation 推导留痕 +
    expert_adjudication="pending" 双轨占位（大纲 §6.1 专家标注协议）。

格点设计（spec §2.2 / §6 D6）：
    8 词对 × {gap, nogap} × {unknown, same} = 32 格点模板，
    每模板 rounds 轮转（每轮换一组来源对 / 对象词，Δ 桶不变）。
    类别（category）= {gap,nogap} × {unknown,same} 四桶，
    gold 与类别一一绑定（gap_unknown=3 / gap_same=3 / nogap_unknown=2 /
    nogap_same=3），生成期自检强制该绑定。

模板语法（渲染期全部断言，不允许静默失配）：
    {TP}   目标词对肯定侧 ka（如"抑制"）
    {TPN}  目标词对否定侧（negated_b 形取 kb；否则写"不"+ka，
           否定前缀紧邻 ka，_polarity_sides 判负）
    {OBJ}  对象词典专名（same 类双方共享同一 canonical；
           unknown 类仅注入【零信号专名】→ 双方 ObjectExtractor 零信号
           → 门禁 UNKNOWN，且零 Aspect 隐式注入）
    {FILL} 填充池短语（池内逐短语自检：零切面词典键 / 零对象词典键 /
           零词对关键词 / 零否定前缀）

构造纪律（违反即生成期 AssertionError）：
    1. 词对唯一：A∪B∪topic 联合语料中，除目标词对外不得有任何
       KEYWORD_PAIR_SPECS 词对极性命中；目标词对极性命中恰好 1 次——
       保证任意开关配置下 gold_pair 输出强度 == gold
       （_check_pair_contradiction 首词对即返回 + 全枚举唯一命中）。
    2. 切面隔离（aspect 拦截面必须不存在）：联合语料（去 {OBJ}）按
       ASPECT_TRIGGER_LEXICON 中文键推断双方切面（ASCII 零键纪律另行
       断言），要求【双方切面集合相等】且与目标词对合法集相交——
       相等保证 unknown 轮（零信号专名）双方恒为同一非空集合，
       词对恒存活；same 轮的词典专名附加切面经断言 2-c 验证不破坏
       相交性。
    3. 门禁可控：unknown 类对象词零 OBJECT_ALIASES/PHENOMENON_LEXICON/
       ASPECT ASCII 键信号；same 类对象词归一为单一 canonical 且
       双侧纯集相等。
    4. 极性分居：ka 肯定出现在含 {TP} 的一侧、否定命中在另一侧。
    5. 来源对：双方 source 必须不同键（_find_contradictions 同源
       不对比纪律），且 |SOURCE_WEIGHTS 差| ≥/< 0.30 与 Δ 桶严格一致。
    6. 零否定前缀："不/未/无/非"字【只允许】出现在 {TPN} 渲染形
       （"不"+ka）中——防"无效/不上调"等词对成员被否定前缀拆出
       意外极性（_polarity_sides 的否定覆盖规则会改写极性归属）。
    7. 词对成员零自由出现：ka/kb 字只能出现在 {TP}/{TPN} 渲染位；
       模板静态段不得含任何词对成员字。

运行（落盘 tests/ddb/l4_cases.json）：
    backend$ .venv/bin/python -m tests.ddb.l4_generator [--rounds 4] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# 导入前置：backend 根入 sys.path + 环境变量占位（必须在 import app 之前）
# ---------------------------------------------------------------------------
_BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

import os

os.environ.setdefault("SCIMINER_API_KEY", "l4-generator-placeholder")
os.environ.setdefault("DB_PASSWORD", "l4-generator-placeholder")

from app.core.virtual_lab.consensus_court import (  # noqa: E402
    Evidence,
    EvidenceScorer,
)
from app.core.virtual_lab.rubric_engine import (  # noqa: E402
    ASPECT_TRIGGER_LEXICON,
    KEYWORD_PAIR_SPECS,
    NEGATION_PREFIXES,
    OBJECT_ALIASES,
    PHENOMENON_LEXICON,
    POLARITY_DEGREE,
    POLARITY_DIRECTIONAL,
    SOURCE_GAP_INTENSITY_THRESHOLD,
    Aspect,
    AsymmetricRubric,
    ObjectExtractor,
    _pair_hit_by_polarity,
    _polarity_sides,
)

#: 生成器版本（写入案例 metadata，审计可追溯）
L4_GENERATOR_VERSION = "v1.0"

#: 默认随机种子与轮转数（spec §6 D6：32 格点 × 4 轮转 = 128 例）
L4_DEFAULT_SEED = 42
L4_DEFAULT_ROUNDS = 4

#: 来源差加成阈值（与 rubric_engine.SOURCE_GAP_INTENSITY_THRESHOLD 同源）
GAP_THRESHOLD = SOURCE_GAP_INTENSITY_THRESHOLD

#: 来源对池（按 Δ 桶预分组；(source_a, source_b) 双方键必须不同——
#: _find_contradictions 同源不对比纪律；键 ∈ SOURCE_WEIGHTS）
GAP_SOURCE_PAIRS: list[tuple[str, str]] = [
    ("pubmed", "agent_opinion"),        # Δ = 0.9-0.4 = 0.50 ≥ 0.30
    ("chembl", "llm"),                  # Δ = 0.85-0.5 = 0.35 ≥ 0.30
    ("uniprot", "llm"),                 # Δ = 0.85-0.5 = 0.35 ≥ 0.30
    ("experiment", "agent_opinion"),    # Δ = 0.75-0.4 = 0.35 ≥ 0.30
]
NOGAP_SOURCE_PAIRS: list[tuple[str, str]] = [
    ("pubmed", "chembl"),               # Δ = 0.05 < 0.30
    ("chembl", "drugbank"),             # Δ = 0.05 < 0.30
    ("experiment", "drugbank"),         # Δ = 0.05 < 0.30
    ("chembl", "uniprot"),              # Δ = 0.00 < 0.30
]

#: 同对象类共享专名池（canonical 各异，均 OBJECT_ALIASES 在册专名；
#: 每轮取一个，双侧对称注入 → 键集相等 → SAME_OBJECT）。
#:
#: 安全约束（_static_same_round_check 强制）：池中每个专名在全部
#: TEMPLATES（含 topic、filler）渲染后必须使全部词对存活——专名注入
#: 的切面不得把任何非 UNKNOWN 词对的完整语料切面推出合法集。
#: 入选条件：注入切面 ∈ {安全性} ∪ {零注入}，因为：
#:   · 安全性 在 成功/失败、存在/不存在、有效/无效 合法集内均包含；
#:   · 零注入 → full = 句子基线，与无专名时等价，不引入新切面。
#: 排除：T-DM1（成药性⊄阳性/阴性合法集）、pembrolizumab（效价⊄
#: 存在/不存在合法集）、AlphaFold（结构可信度⊄成功/失败合法集）、
#: APR-246（表达调控⊄成功/失败合法集）、sotorasib（活性方向⊄
#: 成功/失败合法集）。
SAME_OBJECT_POOL: list[str] = [
    "urelumab",        # canonical=41BB  注入{安全性}（安全：在全部词对合法集内）
    "tazemetostat",    # canonical=EZH2  零注入
    "ERBB1",           # canonical=EGFR  零注入
    "ATG-101",         # canonical=41BB  注入{安全性}（与 urelumab 同 canonical 不同变体）
]

#: 零信号专名池（unknown 类使用：WQ 编号不得命中任何词典键——
#: OBJECT_ALIASES 变体、PHENOMENON 键、ASPECT ASCII 键全部避开，
#: 保证 ObjectExtractor 零信号 → 门禁 UNKNOWN，且零切面注入）
UNKNOWN_OBJECT_POOL: list[str] = [
    "WQ-118", "WQ-450", "WQ-512", "WQ-633", "WQ-707", "WQ-808",
]

#: 专名切面注入复算（D2-EX：same 轮专名注入的切面【只扩大】双方
#: 切面集合（full = base ∪ injection），不改变交集——因为双方
#: 注入同一专名、同一切面集。但 _pair_alive 判据是
#: spec.aspects & (full_a ∩ full_b) ≠ ∅，交集 = base_a ∩ base_b
#: ∪ injection（仅当 base 有命中时）；若 base 零命中 → 双方均
#: {UNKNOWN}，full = {UNKNOWN} ∪ injection，交集 = injection。
#: 存活条件：spec.aspects 必须含 UNKNOWN（通配）或含 injection
#: 中至少一个切面。零注入专名（tazemetostat/ERBB1）依赖 UNKNOWN
#: 通配存活——该前提由 _static_same_round_check 静态自检保证。
#: 详见 SAME_OBJECT_POOL 注释与 _assert_discipline D2。
def _object_aspect_injection(token: str) -> frozenset:
    """专名在 ASPECT 词典全量键（中文子串+ASCII 词边界）下的命中集。"""
    hit: set = set()
    for aspect, keys in ASPECT_TRIGGER_LEXICON.items():
        for key in keys:
            if _regex_hit(key, token):
                hit.add(aspect)
    return frozenset(hit)

#: 填充短语池：中文短语，逐短语静态自检（干净池，模块加载即校验）：
#: 零切面词典中文键、零词对成员字、零否定前缀、零 ASCII 词典键。
#: 池设计原则：全部是"读数/复核/意见"类中性叙述词。
FILLER_POOL: list[str] = [
    "读数复核确认", "读数复核提示", "复核读数显示", "复核结果提示",
    "读数复核结论为", "复核意见显示", "复核显示", "读数意见提示",
    "复核结果确认", "复核意见提示", "复核结论提示", "读数意见确认",
    "复核结论显示",
]

# ============================================================================
# 模板表：32 格点（8 词对 × 4 模板/词对；Δ 桶 × 门禁桶在 generate 阶段轮转）
#
# 模板字段：(topic, sideA, sideB, keyword_pair)
# sideA/sideB 语法见模块 docstring；约定 ka 肯定侧在含 {TP} 的一侧。
#
# 切面安全设计（纪律 2）：全部模板联合语料（渲染后、去 {OBJ}）的
# 词典命中【全部落在目标词对合法集内】，且双方命中集合相等 →
# aspect_on/off 词对恒存活，隔离出"仅 intensity/gate 开关影响输出"
# 的干净变量（spec §2.3）。填充短语刻意【零词典键】（干净池），
# 切面全部由句子静态段自含（如 高表达 句自含"表达/转录"∈表达调控）。
# ============================================================================

TEMPLATES: list[tuple[str, str, str, tuple[str, str]]] = [
    # ------------------------------------------------------------------
    # 有效/无效（negated_b=True，合法集=全切面通用谓词；
    # 句子零切面词典键 → 双方恒 {UNKNOWN}，通配放行）
    # ------------------------------------------------------------------
    (
        "候选分子临床前评价的读数复核",
        "{OBJ}的{FILL}，{FILL}该候选{TP}",
        "{OBJ}的{FILL}，{FILL}该候选{TPN}",
        ("有效", "无效"),
    ),
    (
        "联合给药评价的读数复核",
        "{OBJ}的{FILL}，{FILL}联合给药{TP}",
        "{OBJ}的{FILL}，{FILL}联合给药{TPN}",
        ("有效", "无效"),
    ),
    # ------------------------------------------------------------------
    # 成功/失败（negated_b=False，否定侧"不成功"；合法集含
    # 成药性/疗效/安全性，句子自含对应切面词且双方相等）
    # ------------------------------------------------------------------
    (
        "制备复核的读数",
        "{OBJ}的偶联工艺{FILL}，验证结论{TP}",
        "{OBJ}的偶联工艺{FILL}，验证结论{TPN}",
        ("成功", "失败"),
    ),
    (
        "终点达成判定的读数复核",
        "{OBJ}的终点达成{FILL}，判定结论{TP}",
        "{OBJ}的终点达成{FILL}，判定结论{TPN}",
        ("成功", "失败"),
    ),
    # ------------------------------------------------------------------
    # 阳性/阴性（合法集含 疗效/活性方向/安全性，句子自含对应词）
    # ------------------------------------------------------------------
    (
        "药敏试验判定的读数复核",
        "{OBJ}的药敏试验{FILL}，判定结果{TP}",
        "{OBJ}的药敏试验{FILL}，判定结果{TPN}",
        ("阳性", "阴性"),
    ),
    (
        "安全试验判定的读数复核",
        "{OBJ}的安全性试验{FILL}，判定结果{TP}",
        "{OBJ}的安全性试验{FILL}，判定结果{TPN}",
        ("阳性", "阴性"),
    ),
    # ------------------------------------------------------------------
    # 激活/抑制（合法集含 活性方向/表达调控/机制，否定侧"不激活"；
    # 严禁在句内出现 kb"抑制"以外的其它词对成员）
    # ------------------------------------------------------------------
    (
        "位点复核",
        "{OBJ}的磷酸化检测{FILL}，激酶通路被{TP}",
        "{OBJ}的磷酸化检测{FILL}，激酶通路{TPN}",
        ("激活", "抑制"),
    ),
    (
        "基因激活读数复核",
        "{OBJ}的转录读数{FILL}，报告基因被{TP}",
        "{OBJ}的转录读数{FILL}，报告基因{TPN}",
        ("激活", "抑制"),
    ),
    # ------------------------------------------------------------------
    # 存在/不存在（negated_b=True，否定侧直接 kb；合法集含
    # 结构可信度/成药性/机制/安全性，句子自含对应词）
    # ------------------------------------------------------------------
    (
        "结合位点判定的读数复核",
        "{OBJ}的结合位点{FILL}，该位点{TP}",
        "{OBJ}的结合位点{FILL}，该位点{TPN}",
        ("存在", "不存在"),
    ),
    (
        "旁路活性判定的读数复核",
        "{OBJ}的旁路活性{FILL}，该活性{TP}",
        "{OBJ}的旁路活性{FILL}，该活性{TPN}",
        ("存在", "不存在"),
    ),
    # ------------------------------------------------------------------
    # 高表达/低表达（合法集={表达调控}单切面；句子自含 表达/转录。
    # 注意：单切面词对在强度层【可用】——L1.5 的禁用定理只约束负例
    # 拦截面；L4 正例要求词对恒存活，双方恒含表达调控 → 恒相交存活）
    # ------------------------------------------------------------------
    (
        "标志物读数复核",
        "{OBJ}的生物标志物{TP}，读数复核在定量下限之上",
        "{OBJ}的生物标志物{TPN}，读数复核在定量下限之下",
        ("高表达", "低表达"),
    ),
    (
        "水平读数复核",
        "{OBJ}的靶基因{TP}，转录读数复核在定量下限之上",
        "{OBJ}的靶基因{TPN}，转录读数复核在定量下限之下",
        ("高表达", "低表达"),
    ),
    # ------------------------------------------------------------------
    # 促进/抑制（合法集含 表达调控/机制，否定侧"不促进"）
    # ------------------------------------------------------------------
    (
        "代谢转化读数复核",
        "{OBJ}的代谢通路{FILL}，该分子{TP}其代谢",
        "{OBJ}的代谢通路{FILL}，该分子{TPN}其代谢",
        ("促进", "抑制"),
    ),
    (
        "基因读数复核",
        "{OBJ}的靶基因读数{FILL}，该分子{TP}其表达",
        "{OBJ}的靶基因读数{FILL}，该分子{TPN}其表达",
        ("促进", "抑制"),
    ),
    # ------------------------------------------------------------------
    # 上调/下调（合法集={表达调控}单切面，句子自含 转录/表达）
    # ------------------------------------------------------------------
    (
        "组学读数复核",
        "{OBJ}的转录组{FILL}，靶基因{TP}",
        "{OBJ}的转录组{FILL}，靶基因{TPN}",
        ("上调", "下调"),
    ),
    (
        "谱型读数复核",
        "{OBJ}的表达谱{FILL}，靶基因{TP}",
        "{OBJ}的表达谱{FILL}，靶基因{TPN}",
        ("上调", "下调"),
    ),
]

#: Δ 桶 × 门禁桶（generate 阶段轮转序）
GAP_BUCKETS: tuple[str, ...] = ("gap", "nogap")
GATE_BUCKETS: tuple[str, ...] = ("unknown", "same")

#: 类别 → gold 绑定（spec §2.2 完全枚举：directional 基线 2，
#: gap → S3 +1 → 3；same → S4 提升至 3；nogap_unknown → 恒 2）
CATEGORY_GOLD: dict[str, int] = {
    "gap_unknown": 3,
    "gap_same": 3,
    "nogap_unknown": 2,
    "nogap_same": 3,
}


# ============================================================================
# 词典键池（自检基础设施）
# ============================================================================

def _chinese_aspect_keys() -> list[str]:
    """ASPECT_TRIGGER_LEXICON 全部中文键（子串匹配语义）。"""
    keys: list[str] = []
    for group in ASPECT_TRIGGER_LEXICON.values():
        keys.extend(k for k in group if not k.isascii())
    return keys


def _all_aspect_keys() -> list[str]:
    """ASPECT_TRIGGER_LEXICON 全部键（含 ASCII，词边界语义另判）。"""
    keys: list[str] = []
    for group in ASPECT_TRIGGER_LEXICON.values():
        keys.extend(group)
    return keys


def _object_alias_variants() -> list[str]:
    """OBJECT_ALIASES 全部变体键。"""
    variants: list[str] = []
    for group in OBJECT_ALIASES.values():
        variants.extend(group)
    return variants


def _all_pair_members() -> set[str]:
    """全部词对成员字（KEYWORD_PAIR_SPECS 的 ka ∪ kb）。"""
    members: set[str] = set()
    for s in KEYWORD_PAIR_SPECS:
        members.add(s.keyword_a)
        members.add(s.keyword_b)
    return members


def _regex_hit(key: str, text: str) -> bool:
    """词典键匹配（与前端 _lexicon_hit/_variant_hit 同规则）。"""
    if key.isascii():
        pattern = r"(?<![A-Za-z0-9])" + re.escape(key) + r"(?![A-Za-z0-9])"
        return re.search(pattern, text, flags=re.IGNORECASE) is not None
    return key in text


def _spec_of(pair: tuple[str, str]):
    """按 (ka, kb) 取词对规格。"""
    return next(
        s for s in KEYWORD_PAIR_SPECS if (s.keyword_a, s.keyword_b) == pair
    )


def _static_pool_check() -> None:
    """
    干净池静态自检（模块加载即执行）：
    填充短语必须零切面词典中文键 / 零词对成员字 / 零否定前缀字 /
    零 ASCII 词典键 / 零对象变体 / 零现象键——保证填充只注入语境、
    不注入任何机制信号。
    """
    zh_keys = _chinese_aspect_keys()
    members = _all_pair_members()
    for phrase in FILLER_POOL:
        for key in zh_keys:
            if key in phrase:
                raise AssertionError(
                    f"填充短语 {phrase!r} 命中切面词典中文键 {key!r}"
                )
        for m in members:
            if m in phrase:
                raise AssertionError(
                    f"填充短语 {phrase!r} 含词对成员字 {m!r}"
                )
        for p in NEGATION_PREFIXES:
            if p in phrase:
                raise AssertionError(
                    f"填充短语 {phrase!r} 含否定前缀字 {p!r}"
                )
        for key in _all_aspect_keys():
            if key.isascii() and _regex_hit(key, phrase):
                raise AssertionError(
                    f"填充短语 {phrase!r} 命中 ASCII 切面键 {key!r}"
                )
        for v in _object_alias_variants():
            if _regex_hit(v, phrase):
                raise AssertionError(
                    f"填充短语 {phrase!r} 命中对象变体键 {v!r}"
                )
        for ph in PHENOMENON_LEXICON:
            if _regex_hit(ph, phrase):
                raise AssertionError(
                    f"填充短语 {phrase!r} 命中现象词典键 {ph!r}"
                )
    # 零信号专名静态自检：不得命中任何 ASCII 词典键/对象变体/现象键
    for token in UNKNOWN_OBJECT_POOL:
        for key in _all_aspect_keys():
            if key.isascii() and _regex_hit(key, token):
                raise AssertionError(
                    f"零信号专名 {token!r} 命中 ASCII 切面键 {key!r}"
                )
        for v in _object_alias_variants():
            if v.isascii() and _regex_hit(v, token):
                raise AssertionError(
                    f"零信号专名 {token!r} 命中对象变体键 {v!r}"
                )


_static_pool_check()


def _full_aspect_hit(text: str) -> frozenset:
    """
    全量词典键切面推断（_infer_aspects_full 内联副本，静态自检专用）。
    """
    hit: set = set()
    for aspect, keys in ASPECT_TRIGGER_LEXICON.items():
        for key in keys:
            if _regex_hit(key, text):
                hit.add(aspect)
    return frozenset(hit) if hit else frozenset({Aspect.UNKNOWN})


def _static_same_round_check() -> None:
    """
    same 轮专名切面存活静态自检（模块加载即执行）：

    验证 SAME_OBJECT_POOL 中每个专名在全部 TEMPLATES 渲染后（含
    topic、filler）均使全部词对存活（_pair_alive 判据）。generate
    循环为 for tpl → for r → obj=SAME_POOL[r%len]，故每个对象都会
    被分配到每个模板，必须对全部模板安全。

    判据：spec.aspects & full ≠ ∅
      - full = _full_aspect_hit(A 侧渲染 + topic)（前端等价扫描域）
      - 两侧渲染域对称（同专名 → 同切面集），仅校验 A 侧
      - UNKNOWN ∈ spec.aspects → 通配恒存活，跳过
    """
    for obj_word in SAME_OBJECT_POOL:
        for tpl_idx, (topic, side_a, side_b, pair) in enumerate(TEMPLATES):
            spec = _spec_of(pair)
            # UNKNOWN 通配：合法集含 UNKNOWN → 任何切面集均存活
            if Aspect.UNKNOWN in spec.aspects:
                continue
            # 渲染（与 _fill 等价，{FILL}→""；扫描域含 topic，
            # 与 _infer_aspects_full(content, topic) 前端等价）
            neg_form = spec.keyword_b if spec.negated_b else "不" + spec.keyword_a
            rendered = (
                side_a.replace("{TPN}", neg_form)
                .replace("{TP}", pair[0])
                .replace("{OBJ}", obj_word)
                .replace("{FILL}", "")
            )
            full = _full_aspect_hit(f"{rendered} {topic}")
            if not (spec.aspects & full):
                raise AssertionError(
                    f"same 轮专名切面不存活: tpl[{tpl_idx}] [{topic}] "
                    f"{pair} spec={sorted(s.value for s in spec.aspects)} "
                    f"full={sorted(a.value for a in full)} obj={obj_word}"
                )


_static_same_round_check()


# ============================================================================
# gold 判定表计算器（与 _grade_intensity 同构的独立重算，spec D1）
# ============================================================================

def gold_intensity(
    keyword_pair: tuple[str, str],
    source_a: str,
    source_b: str,
    gate_decision: str,
) -> tuple[int, str]:
    """
    独立重算案例金标强度（不导入前端函数——双向锁：前端逻辑与
    本计算器不同步修改时，评测期差分即暴露漂移）。

    步骤与 rubric_engine._grade_intensity 一一对应：
        S1 硬约束短路 → 3
        S2 极性基线：directional=2 / degree=1
        S3 来源差加成：|w_a − w_b| ≥ 0.30 → +1（SOURCE_WEIGHTS 口径，
           未知键兜底 0.2，与前端一致）
        S4 门禁佐证：SAME_OBJECT 且 base==2 → 3（degree 基线不加成）
        S5 截断 [1,3]

    Args:
        keyword_pair: 目标词对 (keyword_a, keyword_b)
        source_a / source_b: 双方证据来源键
        gate_decision: "SAME_OBJECT" | "UNKNOWN"（gold 期门禁预期态）

    Returns:
        (gold_intensity, gold_derivation 推导字符串留痕)
    """
    spec = _spec_of(keyword_pair)
    steps: list[str] = []

    # S1 硬约束
    if spec.hard_constraint:
        return 3, "S1 硬约束短路 → 3"
    steps.append("S1 无硬约束")

    # S2 极性基线
    base = 2 if spec.polarity == POLARITY_DIRECTIONAL else 1
    steps.append(f"S2 基线={base}({spec.polarity})")
    intensity = base

    # S3 来源可信度差（与前端同口径兜底 0.2）
    w_a = EvidenceScorer.SOURCE_WEIGHTS.get(source_a.lower(), 0.2)
    w_b = EvidenceScorer.SOURCE_WEIGHTS.get(source_b.lower(), 0.2)
    delta = abs(w_a - w_b)
    if delta >= GAP_THRESHOLD:
        intensity += 1
        steps.append(f"S3 +1(Δ={delta:.2f}≥{GAP_THRESHOLD:.2f})")
    else:
        steps.append(f"S3 无加成(Δ={delta:.2f}<{GAP_THRESHOLD:.2f})")

    # S4 门禁佐证
    if gate_decision == "SAME_OBJECT" and base == 2:
        intensity = 3
        steps.append("S4 SAME_OBJECT 佐证 → 3")
    elif gate_decision == "SAME_OBJECT":
        steps.append("S4 SAME_OBJECT 但 degree 基线不加成")

    final = max(1, min(3, intensity))
    steps.append(f"截断 → {final}")
    return final, ", ".join(steps)


# ============================================================================
# 生成器
# ============================================================================

class L4Generator:
    """
    DDB-L4 强度标注层生成器（构造即验证：模板渲染、纪律自检、
    gold 独立重算全部在 generate() 内强制执行，任何违例即中断；
    集合级违例一次汇总报出，便于批量修模板）。
    """

    def __init__(self, seed: int = L4_DEFAULT_SEED):
        """
        Args:
            seed: 轮转采样种子（写入每例 metadata，可复现）
        """
        import random

        self._seed = seed
        self._rng = random.Random(seed)
        self._extractor = ObjectExtractor()

    # ---------------- 渲染 ----------------

    @staticmethod
    def _neg_form(pair: tuple[str, str]) -> str:
        """
        目标词对的否定侧字面形。

        negated_b 词对（有效/无效、存在/不存在）直接用 kb；
        其余词对写"不"+ka——否定前缀紧邻 ka，_polarity_sides 判负。
        纪律：严禁用 kb 作否定侧（kb 可能与其它词对 ka 同形，
        如 抑制 ∈ {激活/抑制, 促进/抑制}，造成跨词对极性污染）。
        """
        spec = _spec_of(pair)
        return spec.keyword_b if spec.negated_b else "不" + spec.keyword_a

    @staticmethod
    def _fill(side_tpl: str, pair: tuple[str, str], obj_word: str,
              fill_start: int, keep_obj: bool = True) -> str:
        """
        把模板槽渲染为证据句内容。

        Args:
            side_tpl: 单侧模板
            pair: 目标词对
            obj_word: 对象专名词
            fill_start: 本侧首个 {FILL} 的池起始下标（同例双侧起始错开，
                避免两侧句子完全对称；keep_obj=False 时删除 {OBJ}，
                用于切面隔离校验的"去专名"重渲染）
            keep_obj: 是否保留对象专名

        Returns:
            渲染后的证据句
        """
        text = side_tpl.replace("{TPN}", L4Generator._neg_form(pair))
        text = text.replace("{TP}", pair[0])
        text = text.replace("{OBJ}", obj_word if keep_obj else "")
        parts = text.split("{FILL}")
        out = parts[0]
        for i, seg in enumerate(parts[1:]):
            out += FILLER_POOL[(fill_start + i) % len(FILLER_POOL)] + seg
        return out

    def _make_evidence(self, suffix: str, source: str, content: str) -> dict:
        """
        构造 L4 证据字典（schema 与 L1/L1.5 对齐；metadata 严禁写
        aspects/target 等显式机制字段——金标必须考验词典推断路径）。

        Args:
            suffix: 证据侧标（"A"/"B"）
            source: 来源键（SOURCE_WEIGHTS 在册）
            content: 渲染后的证据句

        Returns:
            证据字典
        """
        ts = datetime.utcnow() - timedelta(
            days=self._rng.randint(1, 365), minutes=self._rng.randint(0, 1439)
        )
        return {
            "evidence_id": f"EV-{suffix}",
            "source": source,
            "content": content,
            "role": self._rng.choice(
                ["assay-scientist", "reviewer", "data-scientist"]
            ),
            "timestamp": ts.isoformat(),
            "reference_ids": [f"L4-REF-{suffix}-{self._rng.randint(1000, 9999)}"],
            "metadata": {},
        }

    def _render(
        self,
        tpl: tuple[str, str, str, tuple[str, str]],
        round_no: int,
        gap_bucket: str,
        gate_bucket: str,
    ) -> dict:
        """
        渲染一个案例（模板 + 轮转 + 桶 → 完整 case dict + 纪律自检）。

        Args:
            tpl: 模板四元组 (topic, sideA, sideB, keyword_pair)
            round_no: 轮转序号（决定来源对 / 对象词 / 填充短语采样）
            gap_bucket: "gap" | "nogap"（来源差桶）
            gate_bucket: "unknown" | "same"（门禁预期态桶）

        Returns:
            完整 L4 案例字典

        Raises:
            AssertionError: 任何构造纪律违例
        """
        topic, side_a_tpl, side_b_tpl, pair = tpl

        # 1) 来源对采样（4 元池按轮转序整轮覆盖）
        pair_pool = GAP_SOURCE_PAIRS if gap_bucket == "gap" else NOGAP_SOURCE_PAIRS
        source_a, source_b = pair_pool[round_no % len(pair_pool)]
        w_a = EvidenceScorer.SOURCE_WEIGHTS[source_a]
        w_b = EvidenceScorer.SOURCE_WEIGHTS[source_b]
        delta = abs(w_a - w_b)
        bucket_ok = (delta >= GAP_THRESHOLD) if gap_bucket == "gap" else (
            delta < GAP_THRESHOLD
        )
        if not bucket_ok:
            raise AssertionError(
                f"来源对 Δ={delta:.2f} 与桶 {gap_bucket} 不符: {source_a}/{source_b}"
            )

        # 2) 对象词采样
        if gate_bucket == "same":
            obj_word = SAME_OBJECT_POOL[round_no % len(SAME_OBJECT_POOL)]
        else:
            obj_word = UNKNOWN_OBJECT_POOL[round_no % len(UNKNOWN_OBJECT_POOL)]

        # 3) 渲染（双侧起始下标错开 3——同一轮内两侧填充短语不同）
        content_a = self._fill(side_a_tpl, pair, obj_word, round_no * 2)
        content_b = self._fill(side_b_tpl, pair, obj_word, round_no * 2 + 3)

        # 4) gold 独立重算 + 类别绑定校验
        category = f"{gap_bucket}_{gate_bucket}"
        gate_decision = "SAME_OBJECT" if gate_bucket == "same" else "UNKNOWN"
        gold, derivation = gold_intensity(pair, source_a, source_b, gate_decision)
        if gold != CATEGORY_GOLD[category]:
            raise AssertionError(
                f"gold 与类别绑定破坏: {category} gold={gold} "
                f"期望 {CATEGORY_GOLD[category]}（判定表或模板漂移）"
            )

        case = {
            "case_id": "",  # 由 generate() 统一编号
            "topic": topic,
            "polarity": "with_contradiction",
            "category": category,
            "evidence_type": "l4_intensity",
            "defect_flags": [],
            "dispute_side": "A",
            "evidences": [
                self._make_evidence("A", source_a, content_a),
                self._make_evidence("B", source_b, content_b),
            ],
            "ground_truth": {
                "has_true_contradiction": True,
                "gold_pair": ["EV-A", "EV-B"],
                "gold_intensity": gold,
                "gold_derivation": derivation,
                "target_pair_spec": [pair[0], pair[1]],
                "source_delta": round(delta, 4),
                "gate_expectation": gate_decision,
                "expert_adjudication": "pending",
            },
            "metadata": {
                "generator_version": L4_GENERATOR_VERSION,
                "seed": self._seed,
                "round": round_no,
                "gap_bucket": gap_bucket,
                "gate_bucket": gate_bucket,
            },
        }

        # 5) 纪律自检（全部通过才返回；side_tpls 供 _assert_discipline
        #    做去专名重渲染的切面隔离校验，避免模板反查歧义）
        self._assert_discipline(
            case, pair, gap_bucket, gate_bucket, obj_word,
            (side_a_tpl, side_b_tpl),
        )
        return case

    # ---------------- 纪律自检 ----------------

    def _assert_discipline(
        self,
        case: dict,
        pair: tuple[str, str],
        gap_bucket: str,
        gate_bucket: str,
        obj_word: str,
        side_tpls: tuple[str, str],
    ) -> None:
        """
        案例级纪律自检（模块 docstring 纪律 1-7 全量执行）。

        Raises:
            AssertionError: 任何不变量违例
        """
        ev_a, ev_b = case["evidences"]
        topic = case["topic"]
        spec = _spec_of(pair)

        # D5 来源纪律：双方键不同、均在册、Δ 桶正确
        if ev_a["source"] == ev_b["source"]:
            raise AssertionError(f"双方来源同键（同源不对比纪律）: {topic}")
        for s in (ev_a["source"], ev_b["source"]):
            if s not in EvidenceScorer.SOURCE_WEIGHTS:
                raise AssertionError(f"来源键不在 SOURCE_WEIGHTS: {s}")
        delta = abs(
            EvidenceScorer.SOURCE_WEIGHTS[ev_a["source"]]
            - EvidenceScorer.SOURCE_WEIGHTS[ev_b["source"]]
        )
        if gap_bucket == "gap" and delta < GAP_THRESHOLD:
            raise AssertionError(f"gap 桶但 Δ={delta:.2f} < 阈值: {topic}")
        if gap_bucket == "nogap" and delta >= GAP_THRESHOLD:
            raise AssertionError(f"nogap 桶但 Δ={delta:.2f} ≥ 阈值: {topic}")

        # D6/D7 否定前缀与词对成员出现计数纪律
        #   （渲染级动态校验见 D6 / D7-b，模板静态段零成员字
        #    由干净池自检 + 词对唯一性枚举联合保证）
        ka, kb = pair[0], pair[1]

        # D6 零否定前缀纪律：否定前缀字只允许出现在否定侧字面形
        #   （"不"+ka 形紧邻 ka；negated_b 形 kb 内部前缀字豁免——
        #    豁免逻辑封装在 _check_negation_discipline）
        _check_negation_discipline(ev_a["content"], ev_b["content"], ka, kb, spec)

        # D1 词对极性命中枚举：恰好 1 个词对命中 == 目标词对
        hits = [
            (s.keyword_a, s.keyword_b)
            for s in KEYWORD_PAIR_SPECS
            if _pair_hit_by_polarity(
                ev_a["content"].lower(), ev_b["content"].lower(), s
            )
        ]
        if hits != [pair]:
            raise AssertionError(
                f"词对极性命中异常 [{topic}]: 命中={hits} 期望=[{pair}]"
            )

        # D4 极性分居 + 肯定侧唯一
        pos_a, neg_a = _polarity_sides(ev_a["content"].lower(), spec)
        pos_b, neg_b = _polarity_sides(ev_b["content"].lower(), spec)
        sided = (
            (ka in pos_a and bool(neg_b) and not neg_a and ka not in pos_b)
            or (ka in pos_b and bool(neg_a) and not neg_b and ka not in pos_a)
        )
        if not sided:
            raise AssertionError(
                f"目标词对极性未严格分居: [{topic}] pos_a={pos_a} neg_a={neg_a} "
                f"pos_b={pos_b} neg_b={neg_b}"
            )

        # D3 门禁桶纪律（扫描域=完整语料 content+topic，前端等价）：
        #   unknown 轮：双方对象抽取必须为空（Gate → UNKNOWN 放行定罪）；
        #   same 轮：双方键集必须【相等且 = {canon(obj_word)}】
        #   （句子段零对象信号 + 专名对称注入 → SAME_OBJECT 佐证确定）。
        topic_full = topic
        objs_a = self._extractor.extract(
            Evidence(evidence_id="a", source="", content=ev_a["content"],
                     role="", metadata={}), topic_full)
        objs_b = self._extractor.extract(
            Evidence(evidence_id="b", source="", content=ev_b["content"],
                     role="", metadata={}), topic_full)
        keys_a = {o.canonical_key for o in objs_a}
        keys_b = {o.canonical_key for o in objs_b}
        if gate_bucket == "same":
            canon = _canonicalize(obj_word)
            if canon is None or keys_a != {canon} or keys_b != {canon}:
                raise AssertionError(
                    f"same 类对象词须归一为单一 canonical 且双侧纯集: "
                    f"[{topic}] {obj_word} → {keys_a} / {keys_b}"
                )
        else:
            if objs_a or objs_b:
                raise AssertionError(
                    f"unknown 类零信号破坏: [{topic}] "
                    f"{[(o.kind, o.canonical_key) for o in objs_a + objs_b]}"
                )

        # D2 切面隔离（aspect 拦截面必须不存在；词对对双方切面
        # 【完整语料】恒存活）：
        #   断言分两层：
        #   (a) 去专名基线（unknown 轮的精确面）：双方切面集合相等
        #       （= 模板静态段命中的精确集合，词对恒存活，隔离干净）；
        #   (b) 完整语料存活：对全部 8 个词对规格逐一复算 _pair_alive
        #       （含专名、topic），词对必须存活——same 轮专名带来的
        #       额外切面允许双方【不相等】（如 EGFR 词对使双方各含
        #       活性方向/表达调控 + 句子命中集），但目标词对交集内
        #       必须仍有合法切面（DIFFERENT 由 D3 单独钉死）。
        rnd = case["metadata"]["round"]
        content_a_noo = self._fill(
            side_tpls[0], pair, obj_word, rnd * 2, keep_obj=False)
        content_b_noo = self._fill(
            side_tpls[1], pair, obj_word, rnd * 2 + 3,
            keep_obj=False)
        base_a = _infer_aspects(content_a_noo, topic)
        base_b = _infer_aspects(content_b_noo, topic)
        if base_a != base_b:
            raise AssertionError(
                f"去专名切面基线不相等（拦截面可能存在）: [{topic}] "
                f"{sorted(a.value for a in base_a)} vs "
                f"{sorted(b.value for b in base_b)}"
            )
        if not (spec.aspects & base_a):
            raise AssertionError(
                f"去专名基线与词对合法集无交（aspect 拦截面存在）: [{topic}] "
                f"aspects={sorted(a.value for a in base_a)} "
                f"spec={sorted(s.value for s in spec.aspects)}"
            )
        # (b) 完整语料（含专名/topic，前端真实扫描域）目标词对存活复算
        full_a = _infer_aspects_full(ev_a["content"], topic)
        full_b = _infer_aspects_full(ev_b["content"], topic)
        alive_full, shared_full = AsymmetricRubric._pair_alive(
            spec, full_a, full_b)
        if not alive_full:
            raise AssertionError(
                f"词对对完整语料不存活（引擎判死）: [{topic}] "
                f"A={sorted(x.value for x in full_a)} "
                f"B={sorted(x.value for x in full_b)}"
            )

        # D7-b 渲染级词对成员出现计数：
        #   ka 在肯定侧句恰 1 次（{TP}）；在否定侧句出现次数 =
        #   negated_b ? 0（kb 形不含 ka，除 negated_b 包含形）: 1（"不"+ka）
        _check_member_occurrence(ev_a["content"], ev_b["content"], ka, kb,
                                 spec, pos_a, pos_b)

        # D8 topic 词典零键纪律（中文/ASCII/对象/现象全池）
        for key in _chinese_aspect_keys():
            if key in topic:
                raise AssertionError(f"topic 命中切面词典中文键 {key!r}: {topic}")
        for key in _all_aspect_keys():
            if key.isascii() and _regex_hit(key, topic):
                raise AssertionError(f"topic 命中 ASCII 切面键 {key!r}: {topic}")
        for v in _object_alias_variants():
            if _regex_hit(v, topic):
                raise AssertionError(f"topic 命中对象变体键 {v!r}: {topic}")
        for ph in PHENOMENON_LEXICON:
            if _regex_hit(ph, topic):
                raise AssertionError(f"topic 命中现象词典键 {ph!r}: {topic}")


# ---------------- 纪律辅助（模块级纯函数，自检复用） ----------------

def _find_all(text: str, sub: str) -> list[int]:
    """返回 sub 在 text 中全部出现起点。"""
    out: list[int] = []
    start = 0
    while (pos := text.find(sub, start)) != -1:
        out.append(pos)
        start = pos + 1
    return out


def _canonicalize(raw: str) -> Optional[str]:
    """原文对象名经别名表归一（与前端 ObjectExtractor 同规则）。"""
    raw_l = raw.lower()
    for canonical, variants in OBJECT_ALIASES.items():
        for variant in variants:
            if variant.lower() == raw_l:
                return canonical
    return None


def _infer_aspects(content: str, topic: str) -> frozenset[Aspect]:
    """
    中文词典键切面推断（前端 AspectInferrer 词典通道的子集复算：
    L4 模板保证 ASCII 零键——WQ 专名在 unknown 重渲染中已删除、
    填充池零 ASCII——故只需中文键通道即可复算双方切面）。

    Args:
        content: 去专名后的证据句
        topic: 争议主题（联合语料）

    Returns:
        切面集合（零命中 → {UNKNOWN}，与前端通配语义一致）
    """
    corpus = f"{content} {topic}"
    hit: set[Aspect] = set()
    for aspect, keys in ASPECT_TRIGGER_LEXICON.items():
        for key in keys:
            if not key.isascii() and key in corpus:
                hit.add(aspect)
    return frozenset(hit) if hit else frozenset({Aspect.UNKNOWN})


def _infer_aspects_full(content: str, topic: str) -> frozenset[Aspect]:
    """
    完整语料切面推断（前端 AspectInferrer 词典通道全量复算：
    中文键子串 + ASCII 键词边界——same 轮专名的隐式切面注入
    经此进入校验域，与前端 infer(content, topic) 扫描域等价）。

    Args:
        content: 完整证据句（含专名）
        topic: 争议主题（联合语料）

    Returns:
        切面集合（零命中 → {UNKNOWN}，与前端通配语义一致）
    """
    corpus = f"{content} {topic}"
    hit: set[Aspect] = set()
    for aspect, keys in ASPECT_TRIGGER_LEXICON.items():
        for key in keys:
            if _regex_hit(key, corpus):
                hit.add(aspect)
    return frozenset(hit) if hit else frozenset({Aspect.UNKNOWN})


def _check_negation_discipline(
    content_a: str, content_b: str, ka: str, kb: str, spec
) -> None:
    """
    D6 否定前缀纪律（豁免逻辑版；negated_b 形 kb 的内部前缀字
    豁免——"不存在"整体记否定侧）。

    Args:
        content_a / content_b: 双方证据句
        ka / kb: 目标词对成员
        spec: 目标词对规格
    """
    for content in (content_a, content_b):
        for p in NEGATION_PREFIXES:
            for pos in _find_all(content, p):
                if spec.negated_b and content[pos:pos + len(kb)] == kb:
                    continue  # (b) kb 整体豁免（"不存在"的"不"）
                tail = content[pos + len(p):]
                if tail.startswith(ka):
                    continue  # (a) "不"+ka 预期构造
                raise AssertionError(
                    f"否定前缀 {p!r} 出现在非预期位置（意外极性风险）: "
                    f"{content[:60]}"
                )


def _check_member_occurrence(
    content_a: str, content_b: str, ka: str, kb: str, spec,
    pos_a: set[str], pos_b: set[str],
) -> None:
    """
    D7 词对成员出现计数纪律。

    目标词对在联合语料的字面出现次数必须与极性分居构造严格一致：
    - 肯定侧句：ka 恰 1 次、kb 0 次；
    - 否定侧句：negated_b 形 → kb 恰 1 次、ka 0 次；
      "不"+ka 形 → ka 恰 1 次（必被否定覆盖）、kb 0 次。

    Args:
        content_a / content_b: 双方证据句
        ka / kb: 目标词对成员
        spec: 目标词对规格
        pos_a / pos_b: 双方肯定侧命中集（_polarity_sides 输出，用于
            判定哪侧为肯定侧）
    """
    a_affirm = ka in pos_a
    side_pos = content_a if a_affirm else content_b
    side_neg = content_b if a_affirm else content_a

    if _find_all(side_pos, ka).__len__() != 1:
        raise AssertionError(f"肯定侧 ka 出现次数≠1: {side_pos[:60]}")
    if _find_all(side_pos, kb):
        raise AssertionError(f"肯定侧出现 kb: {side_pos[:60]}")

    neg_count_ka = len(_find_all(side_neg, ka))
    neg_count_kb = len(_find_all(side_neg, kb))
    if spec.negated_b:
        # negated_b 形：kb 独立出现恰 1 次（neg 侧唯一），且不得有
        # 任何脱离 kb 覆盖的裸 ka（两种子形态的覆盖判据同 _polarity_sides）
        if neg_count_kb != 1:
            raise AssertionError(
                f"negated_b 否定侧 kb 独立出现应恰 1 次，实际 kb×{neg_count_kb} "
                f"ka×{neg_count_ka}: {side_neg[:60]}"
            )
        bare_ka = 0
        for pos in _find_all(side_neg, ka):
            covered = False
            # 形态一：kb 以 ka 为前缀（"无效"⊃"效"? 否——"无效"前缀字是
            #   "无"；"不存在"前缀字是"不"？均不成立时走形态二）
            if side_neg[pos:pos + len(kb)] == kb:
                covered = True
            # 形态二：ka 为 kb 的后缀（"存在"⊂"不存在"）——kb 对齐起点
            #   = pos - (len(kb) - len(ka))（与 _polarity_sides L300 同式）
            kb_at = pos - (len(kb) - len(ka))
            if kb_at >= 0 and side_neg[kb_at:kb_at + len(kb)] == kb:
                covered = True
            if not covered:
                bare_ka += 1
        if bare_ka != 0:
            raise AssertionError(
                f"negated_b 否定侧 kb 之外出现裸 ka×{bare_ka}（正负混合极性）: "
                f"{side_neg[:60]}"
            )
    else:
        if neg_count_ka != 1 or neg_count_kb != 0:
            raise AssertionError(
                f"否定侧应为 (不+ka)×1/kb×0，实际 ka×{neg_count_ka} "
                f"kb×{neg_count_kb}: {side_neg[:60]}"
            )


# ============================================================================
# 主流程
# ============================================================================

def generate(rounds: int = L4_DEFAULT_ROUNDS, seed: int = L4_DEFAULT_SEED) -> list[dict]:
    """
    生成整份 L4 基准（全部模板 × Δ 桶 × 门禁桶 × rounds 轮转）。

    Args:
        rounds: 每格点轮转数（默认 4，来源对池 4 元恰好整轮覆盖）
        seed: 随机种子

    Returns:
        案例字典列表（已按纪律自检 + gold 判定表校验）

    Raises:
        AssertionError: 任何纪律违例（集合级汇总一次报出）
    """
    gen = L4Generator(seed=seed)
    cases: list[dict] = []
    seq = 0
    violations: list[str] = []
    for tpl in TEMPLATES:
        for gap_bucket in GAP_BUCKETS:
            for gate_bucket in GATE_BUCKETS:
                for r in range(rounds):
                    seq += 1
                    try:
                        case = gen._render(tpl, r, gap_bucket, gate_bucket)
                    except AssertionError as exc:
                        violations.append(
                            f"[{tpl[3]} {gap_bucket}/{gate_bucket} r{r}] {exc}"
                        )
                        continue
                    case["case_id"] = f"DDB-L4-{seq:04d}"
                    cases.append(case)
    if violations:
        raise AssertionError(
            f"L4 模板纪律违例（{len(violations)} 条）:\n"
            + "\n".join(violations[:20])
        )

    # ---------------- 集合级校验 ----------------
    cats = {c["category"] for c in cases}
    if cats != set(CATEGORY_GOLD):
        raise AssertionError(f"类别不齐: {sorted(cats)}")
    per_cat = {cat: [c for c in cases if c["category"] == cat]
               for cat in CATEGORY_GOLD}
    for cat, items in per_cat.items():
        if len(items) != len(TEMPLATES) * rounds:
            raise AssertionError(
                f"{cat} 类例数 {len(items)} ≠ 模板数×轮转 {len(TEMPLATES) * rounds}"
            )
        if any(c["ground_truth"]["gold_intensity"] != CATEGORY_GOLD[cat]
               for c in items):
            raise AssertionError(f"{cat} 类 gold 绑定破坏")
    covered = {tuple(c["ground_truth"]["target_pair_spec"]) for c in cases}
    expect = {(s.keyword_a, s.keyword_b) for s in KEYWORD_PAIR_SPECS}
    if covered != expect:
        raise AssertionError(f"词对覆盖不齐: 缺 {sorted(expect - covered)}")

    # v0.3 极性现实锁（D3）：词对表全 directional → degree 基线前端可达、
    # 案例集不可达；未来接入 degree 词对时放开该断言并按 spec 补模板。
    if any(s.polarity != POLARITY_DIRECTIONAL for s in KEYWORD_PAIR_SPECS):
        raise AssertionError(
            "词对表已出现 degree 极性词对——L4 生成器需按 spec D3 补模板"
        )
    _ = POLARITY_DEGREE  # 常量引用保留（声明 degree 桶当前不可达的锚点）
    return cases


def main() -> int:
    """落盘 tests/ddb/l4_cases.json。"""
    parser = argparse.ArgumentParser(description="DDB-L4 强度标注层生成器")
    parser.add_argument("--rounds", type=int, default=L4_DEFAULT_ROUNDS,
                        help="每格点轮转数（默认 4 → 128 例）")
    parser.add_argument("--seed", type=int, default=L4_DEFAULT_SEED,
                        help="随机种子（默认 42）")
    parser.add_argument("--out", type=str, default="",
                        help="输出路径（默认 tests/ddb/l4_cases.json）")
    args = parser.parse_args()

    cases = generate(rounds=args.rounds, seed=args.seed)
    out_path = Path(args.out) if args.out else (
        Path(__file__).resolve().parent / "l4_cases.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    by_cat: dict[str, int] = {}
    for c in cases:
        by_cat[c["category"]] = by_cat.get(c["category"], 0) + 1
    print(f"L4 基准生成并校验通过：{len(cases)} 例 → {out_path}")
    print(f"类别分布: {by_cat}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
