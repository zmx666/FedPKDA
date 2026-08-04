#!/usr/bin/env bash
# ============================================================================
# FedPKDA 批量实验脚本
#
# 运行前请先激活环境 (脚本内调用的是未限定路径的 python):
#   conda activate robocup
#
# 用法:
#   ./run_experiments.sh                       # 依次运行 EXPERIMENTS 里定义的实验
#   RUN_TAG=20260804 ./run_experiments.sh      # 给所有实验的 goal 追加后缀,
#                                              # 同一组参数重复跑也不会覆盖旧结果
#   ./run_experiments.sh --summary-only        # 跳过训练, 只重新生成 results/experiment_summary.csv
#
# 自动保存机制 (依据代码, 见 system/flcore/servers/serverbase.py 的 save_results):
#   main.py 每完成一次运行, 自动把精度/损失曲线写入
#       results/{dataset}_{algorithm}_{goal}_{run_idx}.h5   (含 rs_test_acc 等)
#   该文件以 'w' 模式写入 => 同名文件会被直接覆盖。
#   因此本脚本的防覆盖规则是: 每次实验的 -go (goal) 唯一
#   (不同 dataset / algorithm / goal 组合即可, 不必全局唯一)。
#   开跑前先做冲突预检, 发现目标 .h5 已存在就整体中止, 绝不覆盖旧数据。
#   日志通过 tee 保留为 results/{实验名}.log, 同样可供后续解析。
#
# 说明:
#   - 训练必须在 system/ 目录下运行 (数据 ../dataset, 结果 ../results 都是相对路径)。
#   - system/models/{dataset}/{algorithm}_server.pt 是模型检查点, 同一 (dataset, algorithm)
#     会被覆盖; 实验数据以 .h5 / .log 为准。
#   - 参数串最后会自动补 -go {goal} (argparse 重复参数取最后一个值, 不会报错)。
#
# 自定义实验: 在 EXPERIMENTS 里加一行, 格式为
#     "实验名 | 传给 python -u main.py 的参数"
#   实验名同时用作日志文件名和默认 goal (除非参数里显式写了 -go)。
#   实验名不要包含空格 / | ; goal 尽量避免纯数字。
# ============================================================================

set -uo pipefail

# --- 路径: 脚本放在仓库根目录, 训练在 system/ 下运行 -----------------------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSTEM_DIR="$ROOT/system"
RESULTS_DIR="$ROOT/results"
mkdir -p "$RESULTS_DIR"

# --- 实验列表 (默认提供 8 组示例, 按需增删/注释) ---------------------------
# 格式: "实验名 | main.py 参数"
EXPERIMENTS=(
    # ---- NSLKDD (dnn), 对应原 run.sh 的两条命令 ----
    # 这两条已跑过, results/ 里已有同名 .h5, 默认注释避免触发防覆盖保护; 想重跑请取消注释
    "nslkdd_fedpkda_dnn| -algo Fedpkda -data NSLKDD -m dnn -nc 20 -nb 2 -gr 60 -lr 0.001 -mo 0.9 -lbs 16 -jr 0.6 -ls 5"
    "nslkdd_fedavg_dnn| -algo FedAvg -data NSLKDD -m dnn -nc 20 -nb 2 -gr 60 -lr 0.001 -mo 0.9 -lbs 16 -jr 0.6 -ls 5"

    # ---- NSLKDD: Fedpkda 参数敏感性 (lr 与 join_ratio) ----
    "nslkdd_fedavg_dnn_lr5e-3| -algo FedAvg -data NSLKDD -m dnn -nc 20 -nb 2 -gr 60 -lr 0.005 -mo 0.9 -lbs 16 -jr 0.6 -ls 5"
    "nslkdd_fedpkda_dnn_lr5e-3| -algo Fedpkda -data NSLKDD -m dnn -nc 20 -nb 2 -gr 60 -lr 0.005 -mo 0.9 -lbs 16 -jr 0.6 -ls 5"

    "nslkdd_fedavg_dnn_jr0.2| -algo FedAvg -data NSLKDD -m dnn -nc 20 -nb 2 -gr 60 -lr 0.001 -mo 0.9 -lbs 16 -jr 0.2 -ls 5"
    "nslkdd_fedpkda_dnn_jr0.2| -algo Fedpkda -data NSLKDD -m dnn -nc 20 -nb 2 -gr 60 -lr 0.001 -mo 0.9 -lbs 16 -jr 0.2 -ls 5"

    # ---- Flowers102 (cnn, README 默认数据集, 共 35 个 client) ----
    "flowers_fedpkda_cnn| -algo Fedpkda -data Flowers102 -m cnn -nc 35 -nb 102 -gr 40 -lr 0.005 -mo 0.1 -lbs 16 -jr 0.2 -ls 5"
    "flowers_fedavg_cnn| -algo FedAvg -data Flowers102 -m cnn -nc 35 -nb 102 -gr 40 -lr 0.005 -mo 0.1 -lbs 16 -jr 0.2 -ls 5"

    # ---- Cifar100 (cnn), Fedpkda 原型对齐权重不同取值 ----
    "cifar100_fedpkda_lp0.3_gp0.5| -algo Fedpkda -data Cifar100 -m cnn -nc 20 -nb 100 -gr 40 --local_par 0.3 --global_par 0.5"
    "cifar100_fedpkda_lp0.6_gp0.3| -algo Fedpkda -data Cifar100 -m cnn -nc 20 -nb 100 -gr 40 --local_par 0.6 --global_par 0.3"
)

# --- 工具函数 ---------------------------------------------------------------
# pick: 从参数串里取某个 flag 的值, 支持 "-f v" 与 "-f=v" 两种写法, flag 可给多个别名
pick() {
    local args="$1"
    shift
    local tok prev="" alias
    for tok in $args; do
        case "$tok" in
        *=*)
            for alias in "$@"; do
                if [ "${tok%%=*}" = "$alias" ]; then
                    echo "${tok#*=}"
                    return 0
                fi
            done
            ;;
        *)
            for alias in "$@"; do
                if [ "$prev" = "$alias" ]; then
                    echo "$tok"
                    return 0
                fi
            done
            prev="$tok"
            ;;
        esac
    done
    return 1
}

count_exps() {
    local entry n=0
    for entry in "${EXPERIMENTS[@]}"; do
        [[ "$entry" =~ ^[[:space:]]*# ]] && continue
        [ -n "$entry" ] || continue
        n=$((n + 1))
    done
    echo "$n"
}

# --- 冲突预检: 目标 .h5 已存在则中止 ----------------------------------------
preflight() {
    local bad=0 entry name args dataset algo goal times i path
    for entry in "${EXPERIMENTS[@]}"; do
        [[ "$entry" =~ ^[[:space:]]*# ]] && continue
        [ -n "$entry" ] || continue
        name="${entry%%|*}"
        args="${entry#*|}"
        dataset="$(pick "$args" -data --dataset)"
        dataset="${dataset:-Flowers102}"
        algo="$(pick "$args" -algo --algorithm)"
        algo="${algo:-Fedpkda}"
        goal="$(pick "$args" -go --goal)"
        goal="${goal:-$name}"
        [ -n "${RUN_TAG:-}" ] && goal="${goal}_${RUN_TAG}"
        times="$(pick "$args" -t --times)"
        times="${times:-1}"
        for ((i = 0; i < times; i++)); do
            path="$RESULTS_DIR/${dataset}_${algo}_${goal}_${i}.h5"
            if [ -f "$path" ]; then
                echo "  [冲突] 已存在, 会被覆盖: $path"
                bad=1
            fi
        done
    done
    if [ "$bad" -eq 1 ]; then
        echo ""
        echo "!!! 检测到目标 .h5 已存在, 直接运行会覆盖旧数据, 已中止(未运行任何实验)。"
        echo "    解决办法: 改 EXPERIMENTS 里的实验名 / -go, 或 RUN_TAG=xxx ./run_experiments.sh,"
        echo "    或先把旧的 .h5 / .log 归档到别处。"
        exit 1
    fi
}

# --- 逐个运行 ---------------------------------------------------------------
PASSED=0
FAILED=0
TOTAL=0

run_all() {
    local entry name args goal logname status
    for entry in "${EXPERIMENTS[@]}"; do
        [[ "$entry" =~ ^[[:space:]]*# ]] && continue
        [ -n "$entry" ] || continue
        name="${entry%%|*}"
        args="${entry#*|}"
        goal="$(pick "$args" -go --goal)"
        goal="${goal:-$name}"
        logname="$name"
        if [ -n "${RUN_TAG:-}" ]; then
            goal="${goal}_${RUN_TAG}"
            logname="${name}_${RUN_TAG}"
        fi
        TOTAL=$((TOTAL + 1))
        echo ""
        echo "=================================================================="
        echo "[$(date '+%F %T')] 开始实验: $name"
        echo "  命令: cd system && python -u main.py $args -go $goal | tee $RESULTS_DIR/$logname.log"
        echo "  预期产物: $RESULTS_DIR/${goal}.h5  (dataset_algo 前缀自动拼上)"
        echo "=================================================================="
        (
            cd "$SYSTEM_DIR" || exit 1
            python -u main.py $args -go "$goal" | tee "$RESULTS_DIR/$logname.log"
        )
        status=$?
        if [ "$status" -eq 0 ]; then
            PASSED=$((PASSED + 1))
            echo "[OK]  完成: $name (goal=$goal)"
        else
            FAILED=$((FAILED + 1))
            echo "[FAIL]失败: $name, 退出码=$status, 详见 $RESULTS_DIR/$logname.log"
        fi
    done
}

# --- 汇总: 重新解析所有 .h5 生成 CSV ----------------------------------------
summary() {
    echo ""
    echo "==== 生成汇总表 results/experiment_summary.csv 与曲线表 results/curves.csv ===="
    (cd "$SYSTEM_DIR" && python summarize_results.py) ||
        echo "[warn] 汇总生成失败: 请确认 system/ 下 python 可用且已安装 h5py / numpy"
    (cd "$SYSTEM_DIR" && python load_results.py) ||
        echo "[warn] 曲线表生成失败: 请确认已安装 pandas"
}

main() {
    if [ "${1:-}" = "--summary-only" ]; then
        summary
        exit 0
    fi
    echo "FedPKDA 批量实验脚本"
    echo "  仓库:     $ROOT"
    echo "  实验数:   $(count_exps)"
    [ -n "${RUN_TAG:-}" ] && echo "  RUN_TAG:  ${RUN_TAG} (追加到 goal, 防覆盖)"
    echo ""
    echo "---- 冲突预检 ----"
    preflight
    echo "预检通过, 开始运行..."
    run_all
    summary
    echo ""
    echo "==== 汇总: 共 $TOTAL, 成功 $PASSED, 失败 $FAILED ===="
    echo "    每条实验的曲线: results/*.h5 (可用 summarize_results.py 解析)"
    echo "    每条实验的日志: results/*.log"
    echo "    汇总表:         results/experiment_summary.csv"
    [ "$FAILED" -gt 0 ] && exit 1
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    main "$@"
fi
