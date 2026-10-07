#!/usr/bin/env bash
# v0.3 三机制回归套件运行脚本（Aspect 约束 / ValidityGate / 强度分级）
#
# 用法：
#   bash backend/tests/scripts/run_v03_regression.sh
#
# 测试目标：
# 1. 运行 v0.3 专项测试：
#    - test_rubric_engine_v03.py   （U1-U5：aspect/极性/门控/强度/开关差分，30 例）
#    - test_l2_v03_locks.py        （U6 升级挂钩 + U7 结构锁，16 例）
#    - tests/ddb/test_l15_generator.py （U8-U10：L1.5 完整性/FPR=0/全关差分）
#    - tests/ddb/test_l4_generator.py  （U11-U12：L4 完整性/强度一致性/消融冒烟）
#    - test_baseline_arbiter.py        （U13.0：§6.2 三对照基线仲裁器）
#    - test_baseline_comparison_experiment.py（U13：§6.2 端到端对照实验）
#    - test_court_ablation_pipeline.py（U14：§6.4 法院组件消融 + hold_court parity）
#    - test_robustness_experiments.py （U15：§6.5 顺序置换 + n 扫描鲁棒性，
#      v0.4.12 起含实验 C 流式即判 + 实验 D 来源标签噪声）
#    - test_advantage_decomposition.py（U16：v0.4.12 §5.3/§6.5 定理 1-A
#      优势分解 P(S)/P(M\D)/q_K/κ + 恒等式闭合）
#    - test_role_weights_v0413.py（U17：v0.4.13 §6.5 十三角色权重
#      对齐锁 + fallback 判别 + 新画像可加载 + 校准核心函数）
#    - test_network_failure_experiment.py（U18：v0.4.14 §6.5 网络失败
#      区分实验——注入器/案例生成/二态基线映射/格点执行/红线检查器）
#    - test_baseline_sensitivity.py（U19：v0.4.16 §6.2 基线阈值敏感性
#      扫描——决策原语重放确定性/49 格点结构锁/RL1 互锁/FPR 单调性/
#      Pareto 前沿正确性与论文坐标锁/报告落盘锁/检查器反测试）
# 2. 运行既有回归测试（零回归验证）：
#    - test_rubric_engine.py                （v0.2 基线）
#    - test_virtual_lab_consensus_court.py
#    - test_court_verdict_persistence.py
#    - tests/ddb/test_l2_cases.py / test_pilot_h2a.py / test_pilot_h3_network.py
# 3. 运行实验管线（确定性生成 + 零 LLM，全部红线自动断言）：
#    - run_ablation_v03.py：60×2 例 6 个 rubric 配置（保守公理；--l15/--l4
#      时附加 L1.5/L4 红线；v0.4.5 位置加权折扣；--court 时附加法院
#      组件消融 court_full/no_abp/no_reviewer/no_boss）
#    - run_baseline_comparison.py：§6.2 ConsensusCourt vs 三基线
#      FPR/召回/Acc 对照（Tab.3 冒烟数据；v0.4.16 起率值块带 Wilson 95% CI）
#    - run_robustness_experiments.py：§6.5 顺序置换 1000 次（定理 2）
#      + n=3/5/10/20 扫描（定理 1-B 计数标度律）；v0.4.12 起追加
#      实验 C 流式即判对照（k=1/k=2 顺序置换）与实验 D 来源标签
#      ±20% 噪声（零构陷/召回保持/置信度漂移红线）
#    - run_advantage_decomposition.py：v0.4.12 §5.3/§6.5 定理 1-A
#      优势分解标定（L1 负例 + L1.5 共 148 例双配置四格分解、恒等式
#      精确闭合；零假设 m=20/300 案标定 q_K 与相邻对相依 κ>1）
#    - run_role_weight_calibration.py：v0.4.13 §6.5 十三角色权重校准
#      （单角色 ±20% 扰动翻转率红线 0.05 + ×0.1 阳性对照 + 覆盖率红线，
#      报告 tests/ddb/role_weight_calibration.json）
#    - run_network_failure_experiment.py：v0.4.14 §6.5 网络失败区分实验
#      （三态+重试+缓存 vs 二态直拒 × p∈{0,0.3,0.5,1.0} 假阳性曲线，
#      六条预注册红线，报告 tests/ddb/network_failure_report.json，
#      Fig.4 数据源）
#    - run_baseline_sensitivity.py：v0.4.16 §6.2 基线阈值敏感性扫描
#      （(0.3..0.9)² 共 49 格点决策原语重放 Pareto 前沿 + RL1 主对比
#      互锁/RL2 FPR 单调/RL3 法庭阈值无关三红线，报告
#      tests/ddb/baseline_sensitivity_report.json，论文 Pareto 图
#      （fig:pareto）数据源）
#
# 退出码：
#   0 = 全部通过（含消融红线）
#   1 = v0.3 专项测试失败
#   2 = 既有回归测试失败
#   3 = 消融冒烟红线违例或运行异常
#
# 设计说明：
# - 与 run_baseline.sh 同构（多段退出码汇总），保证 CI 可接入
# - 消融冒烟走 tests/scripts/run_ablation_v03.py（确定性生成 + 零 LLM）

# 注意：不使用 set -e，需要手动捕获各步骤退出码并生成汇总报告

# ============================================================================
# 配置
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BACKEND_ROOT="${PROJECT_ROOT}/backend"

# Python 解释器（优先 .venv-bio311（Python 3.11，支持 typing.ParamSpec，
# 数据库层 app/database.py 依赖 3.10+ 特性），其次 .venv-bio，其次 .venv，
# 最后系统 python3）
if [[ -f "${PROJECT_ROOT}/.venv-bio311/bin/python" ]]; then
    PYTHON="${PROJECT_ROOT}/.venv-bio311/bin/python"
elif [[ -f "${PROJECT_ROOT}/.venv-bio/bin/python" ]]; then
    PYTHON="${PROJECT_ROOT}/.venv-bio/bin/python"
elif [[ -f "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    PYTHON="${PROJECT_ROOT}/.venv/bin/python"
else
    PYTHON="python3"
fi

# 消融冒烟规模（每极性案例数；主实验可改为 300）
ABLATION_CASES=60

# L1.5 负例对照基准开关（大纲 v0.4.3 §6.1 层 1.5）：1=启用附加评估，
# 0=跳过（回退 v0.3 冒烟原行为）。启用后消融红线追加：负例集
# full_v03 FPR=0 + 分机制一阶效应（aspect/gate 拦截面归属纯度）。
ABLATION_L15=1

# L4 强度标注基准开关（spec §2）：1=启用附加评估（强度判定表全分支
# 可达性 + 全开一致性 + 强度一阶效应红线），0=跳过。
ABLATION_L4=1

# 法院组件消融开关（v0.4.11 §6.4）：1=启用 --court，附加
# court_full / no_abp / no_reviewer / no_boss 四模式的定罪层
# 与升级层红线（ABP 必要性 + 评审/BOSS 升级信号单调性 + parity 守卫）。
ABLATION_COURT=1

# scorer 级消融开关（v0.4.15 §6.4 第 4/8 行）：1=启用 --scorer-ablation，
# 附加来源等权（no_source_pedigree）/ 时效衰减关闭（no_recency）两格点
# 的 FPR/召回单调性 + 通过案例置信度效应方向红线（生产代码零改动，
# 评分输出实验侧后处理）。
ABLATION_SCORER=1

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log()  { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $*"; }
err()  { echo -e "${RED}[$(date '+%H:%M:%S')] ERROR:${NC} $*" >&2; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARN:${NC} $*"; }
info() { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $*"; }

V03_RC=0
REGRESSION_RC=0
ABLATION_RC=0

# ============================================================================
# Step 1: v0.3 专项测试（U1-U7）
# ============================================================================
log "Step 1: v0.3 专项测试（U1-U5 单元 + U6/U7 集成与结构锁）..."
echo ""

cd "$BACKEND_ROOT"
# 单次运行同时显示输出并捕获 pytest 退出码（先写日志文件再 cat，避免
# 管道吞码，也避免 run_baseline.sh 双跑模式的耗时与偶发不一致）
V03_LOG="$(mktemp)"
"$PYTHON" -m pytest \
    tests/test_rubric_engine_v03.py \
    tests/ddb/test_l2_v03_locks.py \
    tests/ddb/test_l15_generator.py \
    tests/ddb/test_l4_generator.py \
    tests/test_baseline_arbiter.py \
    tests/test_baseline_comparison_experiment.py \
    tests/test_court_ablation_pipeline.py \
    tests/test_robustness_experiments.py \
    tests/test_advantage_decomposition.py \
    tests/test_role_weights_v0413.py \
    tests/test_network_failure_experiment.py \
    tests/test_baseline_sensitivity.py \
    tests/test_scorer_ablation_v0415.py \
    tests/test_case_study_v0415.py \
    -q --tb=short > "$V03_LOG" 2>&1
v03_rc=$?
while IFS= read -r line; do
    echo "  $line"
done < "$V03_LOG"
rm -f "$V03_LOG"

if [[ $v03_rc -eq 0 ]]; then
    log "  v0.3 专项测试: 全部通过"
else
    err "  v0.3 专项测试: 失败 (exit=${v03_rc})"
    V03_RC=1
fi

echo ""

# ============================================================================
# Step 2: 既有回归测试（零回归验证）
# ============================================================================
log "Step 2: 既有回归测试（零回归）..."
echo ""

REG_LOG="$(mktemp)"
"$PYTHON" -m pytest \
    tests/test_rubric_engine.py \
    tests/test_virtual_lab_consensus_court.py \
    tests/test_court_verdict_persistence.py \
    tests/ddb/test_l2_cases.py \
    tests/ddb/test_pilot_h2a.py \
    tests/ddb/test_pilot_h3_network.py \
    -q --tb=short > "$REG_LOG" 2>&1
regression_rc=$?
while IFS= read -r line; do
    echo "  $line"
done < "$REG_LOG"
rm -f "$REG_LOG"

if [[ $regression_rc -eq 0 ]]; then
    log "  既有回归测试: 全部通过（零回归）"
else
    err "  既有回归测试: 失败 (exit=${regression_rc})"
    REGRESSION_RC=2
fi

echo ""

# ============================================================================
# Step 3: v0.3 消融冒烟实验（保守公理红线检查）
# ============================================================================
L15_NOTE=""
if [[ "$ABLATION_L15" == "1" ]]; then
    L15_NOTE="，L1.5=启用"
fi
L4_NOTE=""
if [[ "$ABLATION_L4" == "1" ]]; then
    L4_NOTE="，L4=启用"
fi
COURT_NOTE=""
if [[ "$ABLATION_COURT" == "1" ]]; then
    COURT_NOTE="，法院组件消融=启用"
fi
SCORER_NOTE=""
if [[ "$ABLATION_SCORER" == "1" ]]; then
    SCORER_NOTE="，scorer 消融=启用"
fi
log "Step 3: 实验管线（消融 ${ABLATION_CASES}×2 例 / 6 rubric 配置${L15_NOTE}${L4_NOTE}${COURT_NOTE}${SCORER_NOTE} + §6.2 对照 + §6.5 鲁棒性 + §5.3 优势分解 + §6.5 权重校准 + §6.5 网络失败 + §6.5 案例研究）..."
echo ""

ABL_LOG="$(mktemp)"
ABL_FLAGS=()
if [[ "$ABLATION_L15" == "1" ]]; then
    ABL_FLAGS+=(--l15)
fi
if [[ "$ABLATION_L4" == "1" ]]; then
    ABL_FLAGS+=(--l4)
fi
if [[ "$ABLATION_COURT" == "1" ]]; then
    ABL_FLAGS+=(--court)
fi
if [[ "$ABLATION_SCORER" == "1" ]]; then
    ABL_FLAGS+=(--scorer-ablation)
fi
"$PYTHON" tests/scripts/run_ablation_v03.py --cases "$ABLATION_CASES" "${ABL_FLAGS[@]}" > "$ABL_LOG" 2>&1
ablation_rc=$?
while IFS= read -r line; do
    echo "  $line"
done < "$ABL_LOG"
rm -f "$ABL_LOG"

if [[ $ablation_rc -eq 0 ]]; then
    log "  消融冒烟: 通过（FPR 不升 / 召回不降 / L1.5 / L4 / 法院组件红线满足）"
elif [[ $ablation_rc -eq 1 ]]; then
    err "  消融冒烟: 红线违例（见上方明细）"
    ABLATION_RC=3
else
    err "  消融冒烟: 运行异常 (exit=${ablation_rc})"
    ABLATION_RC=3
fi

echo ""

# ----------------------------------------------------------------------------
# Step 3b: §6.2 对照基线端到端对比（ConsensusCourt vs Majority/Elo/SingleJudge）
# ----------------------------------------------------------------------------
log "Step 3b: §6.2 对照基线端到端对比实验..."
echo ""

CMP_LOG="$(mktemp)"
"$PYTHON" tests/scripts/run_baseline_comparison.py --cases "$ABLATION_CASES" > "$CMP_LOG" 2>&1
compare_rc=$?
while IFS= read -r line; do
    echo "  $line"
done < "$CMP_LOG"
rm -f "$CMP_LOG"

if [[ $compare_rc -eq 0 ]]; then
    log "  §6.2 对照实验: 通过（完整法庭严格优于三基线）"
else
    err "  §6.2 对照实验: 失败 (exit=${compare_rc})"
    ABLATION_RC=3
fi

echo ""

# ----------------------------------------------------------------------------
# Step 3b-2: §6.2 基线阈值敏感性扫描（v0.4.16：49 格点决策原语重放 Pareto）
# ----------------------------------------------------------------------------
log "Step 3b-2: §6.2 基线阈值敏感性扫描（49 格点 Pareto + RL1/RL2/RL3 红线）..."
echo ""

SENS_LOG="$(mktemp)"
"$PYTHON" tests/scripts/run_baseline_sensitivity.py --cases "$ABLATION_CASES" > "$SENS_LOG" 2>&1
sens_rc=$?
while IFS= read -r line; do
    echo "  $line"
done < "$SENS_LOG"
rm -f "$SENS_LOG"

if [[ $sens_rc -eq 0 ]]; then
    log "  §6.2 敏感性扫描: 通过（三红线满足，基线可达画像完整落盘）"
else
    err "  §6.2 敏感性扫描: 失败 (exit=${sens_rc})"
    ABLATION_RC=3
fi

echo ""

# ----------------------------------------------------------------------------
# Step 3c: §6.5 鲁棒性专项（顺序置换 1000 次 + n=3/5/10/20 标度扫描
#           + v0.4.12 实验 C 流式即判对照 + 实验 D 来源标签噪声）
# ----------------------------------------------------------------------------
log "Step 3c: §6.5 鲁棒性专项实验（定理 2 + 定理 1-B + 流式对照 + 来源噪声）..."
echo ""

ROB_LOG="$(mktemp)"
"$PYTHON" tests/scripts/run_robustness_experiments.py --permutations 1000 --trials 300 > "$ROB_LOG" 2>&1
robust_rc=$?
while IFS= read -r line; do
    echo "  $line"
done < "$ROB_LOG"
rm -f "$ROB_LOG"

if [[ $robust_rc -eq 0 ]]; then
    log "  §6.5 鲁棒性实验: 通过（顺序零分歧 + 计数标度律 + 流式顺序敏感对照 + 噪声零构陷）"
else
    err "  §6.5 鲁棒性实验: 失败 (exit=${robust_rc})"
    ABLATION_RC=3
fi

echo ""

# ----------------------------------------------------------------------------
# Step 3d: v0.4.12 §5.3/§6.5 定理 1-A 优势分解标定
#           （P(S)/P(M\D) 双配置四格分解 + q_K/κ 零假设相依标定）
# ----------------------------------------------------------------------------
log "Step 3d: §5.3 定理 1-A 优势分解标定实验（恒等式闭合 + κ>1）..."
echo ""

DEC_LOG="$(mktemp)"
"$PYTHON" tests/scripts/run_advantage_decomposition.py --cases 60 --null-trials 300 > "$DEC_LOG" 2>&1
decomp_rc=$?
while IFS= read -r line; do
    echo "  $line"
done < "$DEC_LOG"
rm -f "$DEC_LOG"

if [[ $decomp_rc -eq 0 ]]; then
    log "  §5.3 优势分解实验: 通过（恒等式精确闭合 + ABP 优势可测 + κ 正相依）"
else
    err "  §5.3 优势分解实验: 失败 (exit=${decomp_rc})"
    ABLATION_RC=3
fi

echo ""

# ----------------------------------------------------------------------------
# Step 3e: v0.4.13 §6.5 十三角色权重校准（单角色 ±20% 扰动翻转率
#           + 阳性对照 + 覆盖率 + 翻转率红线）
# ----------------------------------------------------------------------------
log "Step 3e: §6.5 十三角色权重校准实验（±20% 扰动翻转率 + 阳性对照）..."
echo ""

CAL_LOG="$(mktemp)"
"$PYTHON" tests/scripts/run_role_weight_calibration.py --cases 60 > "$CAL_LOG" 2>&1
calib_rc=$?
while IFS= read -r line; do
    echo "  $line"
done < "$CAL_LOG"
rm -f "$CAL_LOG"

if [[ $calib_rc -eq 0 ]]; then
    log "  §6.5 权重校准实验: 通过（±20% 翻转率红线满足 + 阳性对照有效 + 覆盖率达标）"
else
    err "  §6.5 权重校准实验: 失败 (exit=${calib_rc})"
    ABLATION_RC=3
fi

echo ""

# ----------------------------------------------------------------------------
# Step 3f: v0.4.14 §6.5 网络失败区分实验（P0-2 核心，Fig.4 数据源）
#           （三态+重试+缓存 vs 二态直拒 × p∈{0,0.3,0.5,1.0} 假阳性曲线
#           + 六条预注册红线）
# ----------------------------------------------------------------------------
log "Step 3f: §6.5 网络失败区分实验（三态 vs 二态直拒假阳性曲线）..."
echo ""

NET_LOG="$(mktemp)"
"$PYTHON" tests/scripts/run_network_failure_experiment.py > "$NET_LOG" 2>&1
network_rc=$?
while IFS= read -r line; do
    echo "  $line"
done < "$NET_LOG"
rm -f "$NET_LOG"

if [[ $network_rc -eq 0 ]]; then
    log "  §6.5 网络失败区分实验: 通过（三态 FPR=0 全格点 + 二态污染曲线 + 六红线满足）"
else
    err "  §6.5 网络失败区分实验: 失败 (exit=${network_rc})"
    ABLATION_RC=3
fi

echo ""

# ----------------------------------------------------------------------------
# Step 3g: v0.4.15 §6.5 端到端案例研究（DDB-L2 落地版，案例研究章节数据源）
#           （精选僵局型 L2-0001 + 升级型 L2-0005 走生产 run_court_headless
#           完整管线：七环节时间线 / ABP 收敛点 / 裁决链 / 金标对检测，
#           结构锁 + 双跑确定性 + L2 H2b 锁一致性自检）
# ----------------------------------------------------------------------------
log "Step 3g: §6.5 端到端案例研究（DDB-L2 僵局型 + 升级型）..."
echo ""

CS_LOG="$(mktemp)"
"$PYTHON" tests/scripts/run_case_study.py > "$CS_LOG" 2>&1
casestudy_rc=$?
while IFS= read -r line; do
    echo "  $line"
done < "$CS_LOG"
rm -f "$CS_LOG"

if [[ $casestudy_rc -eq 0 ]]; then
    log "  §6.5 端到端案例研究: 通过（结构锁 + 确定性 + L2 锁对齐）"
else
    err "  §6.5 端到端案例研究: 失败 (exit=${casestudy_rc})"
    ABLATION_RC=3
fi

echo ""

# ============================================================================
# 汇总报告
# ============================================================================
echo "=============================================="
log "v0.3 回归套件汇总"
echo "=============================================="
echo ""

if [[ $V03_RC -eq 0 ]]; then
    log "  v0.3 专项测试: 通过"
else
    err "  v0.3 专项测试: 失败"
fi

if [[ $REGRESSION_RC -eq 0 ]]; then
    log "  既有回归测试: 通过（零回归）"
else
    err "  既有回归测试: 失败"
fi

if [[ $ABLATION_RC -eq 0 ]]; then
    log "  实验管线（消融/§6.2 对照/§6.5 鲁棒性/§5.3 优势分解/§6.5 权重校准/§6.5 网络失败/§6.5 案例研究）: 通过"
else
    err "  实验管线（消融/§6.2 对照/§6.5 鲁棒性/§5.3 优势分解/§6.5 权重校准/§6.5 网络失败/§6.5 案例研究）: 失败"
fi

echo ""

# 综合退出码（取最高严重度）
if [[ $V03_RC -ne 0 ]]; then
    exit 1
elif [[ $REGRESSION_RC -ne 0 ]]; then
    exit 2
elif [[ $ABLATION_RC -ne 0 ]]; then
    exit 3
else
    log "v0.3 回归套件全部通过！"
    exit 0
fi
