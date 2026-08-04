#!/usr/bin/env python
"""
扫描 ../results 下所有 .h5 实验文件 (main.py 自动保存机制的产物),
汇总成 results/experiment_summary.csv 并打印控制台表格。

文件名格式 (见 system/flcore/servers/serverbase.py 的 save_results):
    {dataset}_{algorithm}_{goal}_{run_idx}.h5
其中 run_idx 由 main.py 的 -t/--times 控制, 默认为 0。

用法:
    cd system && python summarize_results.py

每次运行都会基于当前所有 .h5 重新生成 CSV (幂等, 不会累积重复行)。
"""
import csv
import glob
import os

import h5py
import numpy as np

RESULTS_DIR = os.path.join("..", "results")
CSV_PATH = os.path.join(RESULTS_DIR, "experiment_summary.csv")

CSV_HEADER = [
    "dataset", "algorithm", "goal", "run", "rounds",
    "best_acc_pct", "best_round", "last5_acc_pct", "final_acc_pct",
    "final_train_loss", "best_acc_std_pct",
]


def parse_stem(stem):
    """从文件名 stem 解析 (dataset, algorithm, goal, run_idx)。

    文件名总是以 _{run_idx} 结尾且 run_idx 是整数, 所以取最后一个数字段作 run_idx,
    goal 为中间所有段 (goal 内部允许包含下划线)。
    """
    parts = stem.split("_")
    dataset = parts[0]
    algorithm = parts[1] if len(parts) > 1 else ""
    if len(parts) > 2 and parts[-1].isdigit():
        run = int(parts[-1])
        goal = "_".join(parts[2:-1])
    else:
        run = 0
        goal = "_".join(parts[2:])
    return dataset, algorithm, goal, run


def load_metrics(path):
    with h5py.File(path, "r") as hf:
        acc = np.array(hf["rs_test_acc"])
        auc = np.array(hf["rs_test_auc"]) if "rs_test_auc" in hf else None
        train_loss = np.array(hf["rs_train_loss"]) if "rs_train_loss" in hf else None
    return acc, auc, train_loss


def summarize():
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.h5")))
    rows = []       # 每个 (实验, run) 一行
    groups = {}     # (dataset, algorithm, goal) -> [best_acc, ...] 用于多次运行的聚合

    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        try:
            acc, _, train_loss = load_metrics(f)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] 跳过无法读取的文件 {f}: {e}")
            continue
        if acc.size == 0:
            print(f"[warn] 跳过空结果 {f}")
            continue

        dataset, algorithm, goal, run = parse_stem(stem)
        rounds = int(acc.size)
        best_acc = float(acc.max())
        best_round = int(acc.argmax())
        last5 = float(acc[-5:].mean()) if rounds >= 5 else float(acc.mean())
        final_acc = float(acc[-1])
        final_train_loss = (
            float(train_loss[-1])
            if train_loss is not None and train_loss.size
            else ""
        )

        rows.append({
            "dataset": dataset, "algorithm": algorithm, "goal": goal, "run": run,
            "rounds": rounds, "best_acc_pct": best_acc * 100.0, "best_round": best_round,
            "last5_acc_pct": last5 * 100.0, "final_acc_pct": final_acc * 100.0,
            "final_train_loss": final_train_loss, "best_acc_std_pct": "",
        })
        groups.setdefault((dataset, algorithm, goal), []).append(best_acc)

    # 多次运行 (-t N) 的聚合行, 追加 best_acc 的均值/标准差
    for (dataset, algorithm, goal), bests in sorted(groups.items()):
        if len(bests) > 1:
            rows.append({
                "dataset": dataset, "algorithm": algorithm, "goal": goal, "run": "ALL",
                "rounds": "", "best_acc_pct": float(np.mean(bests)) * 100.0,
                "best_round": "", "last5_acc_pct": "", "final_acc_pct": "",
                "final_train_loss": "", "best_acc_std_pct": float(np.std(bests)) * 100.0,
            })

    def sort_key(r):
        return (r["dataset"], r["algorithm"], r["goal"],
                r["run"] == "ALL", 0 if r["run"] == "ALL" else int(r["run"]))

    rows.sort(key=sort_key)

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: _fmt(v) for k, v in r.items()})

    print(f"✅ 共解析 {len(files)} 个 .h5 → {CSV_PATH}")
    if not rows:
        print("   (results 下暂无实验结果文件)")
        return

    # 控制台表格
    head = f"{'dataset':<10} {'algo':<8} {'goal':<28} {'run':<5} {'rounds':<7} {'best%':>8} {'@r':>4} {'last5%':>8} {'std%':>7}"
    print(head)
    print("-" * len(head))
    for r in rows:
        run = r["run"] if r["run"] == "ALL" else f"{r['run']}"
        print(
            f"{r['dataset']:<10} {r['algorithm']:<8} {r['goal']:<28} {run:<5} "
            f"{_fmt(r['rounds']):<7} {_fmt(r['best_acc_pct']):>8} {_fmt(r['best_round']):>4} "
            f"{_fmt(r['last5_acc_pct']):>8} {_fmt(r['best_acc_std_pct']):>7}"
        )


def _fmt(v, nd=4):
    if v == "" or v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


if __name__ == "__main__":
    summarize()
