#!/usr/bin/env python
"""
生成简单实验报告 + 「轮数 - acc - std」图像, 全部输出到 output/ 目录。

服务器无显示环境: 强制使用 matplotlib Agg 后端, 图像直接保存, 不弹窗。

用法:
    cd system && python make_report.py

产物 (均在 output/):
    report.md            简单实验报告 (数据驱动的观察 + 每数据集汇总表)
    acc_by_round.png     按数据集分 3 个子图, 每子图画出各实验的
                         轮数-精度曲线, 附 ±std 误差带
    experiment_summary.csv / curves.csv  汇总表与曲线表副本

std 说明: 当前每组实验均为单次运行 (run=0), 不存在 run 间方差;
图中误差带取自训练日志里每轮打印的「Std Test Accurancy」(各客户端
测试精度分布的标准差)。若以后用 -t N 跑多次, 将优先使用 run 间标准差。
"""
import csv
import os
import re
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")  # 服务器无显示, 直接保存
import matplotlib.pyplot as plt
import numpy as np

from load_results import RESULTS_DIR, load_results

OUTPUT_DIR = os.path.join("..", "output")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "experiment_summary.csv")
REPORT_PATH = os.path.join(OUTPUT_DIR, "report.md")
FIG_PATH = os.path.join(OUTPUT_DIR, "acc_by_round.png")

STD_LOG_PATTERN = re.compile(r"^\s*Std Test Accurancy:\s*([0-9.]+)")


def parse_log_std(goal):
    """从 results/{goal}.log 解析每轮的客户端精度 std; 解析不到返回 None。"""
    path = os.path.join(RESULTS_DIR, f"{goal}.log")
    if not os.path.exists(path):
        return None
    vals = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = STD_LOG_PATTERN.match(line)
            if m:
                vals.append(float(m.group(1)))
    return np.array(vals) if vals else None


def build_curves(df):
    """按 (dataset, goal) 聚合出逐轮 mean/std acc。

    run > 1 时优先用 run 间标准差, 否则退回日志里的客户端 std。
    返回 OrderedDict: (dataset, goal) -> (rounds, mean, std|None)
    """
    curves = OrderedDict()
    for (dataset, goal), g in df.groupby(["dataset", "goal"], dropna=False):
        runs = sorted(g["run"].unique())
        n_rounds = g["round"].nunique()
        acc = np.full((len(runs), n_rounds), np.nan)
        for j, run in enumerate(runs):
            sub = g[g["run"] == run].sort_values("round")
            acc[j, : len(sub)] = sub["test_acc"].values
        mean = np.nanmean(acc, axis=0)
        run_std = np.nanstd(acc, axis=0) if len(runs) > 1 else None
        if run_std is not None and np.nanmax(run_std) > 0:
            std = run_std
        else:
            log_std = parse_log_std(goal)
            std = log_std if log_std is not None and log_std.size == n_rounds else None
        curves[(dataset, goal)] = (np.arange(n_rounds), mean, std)
    return curves


def plot_curves(curves):
    datasets = list(OrderedDict.fromkeys(ds for (ds, _) in curves))
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.2 * len(datasets), 5), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        for (gds, goal), (rounds, mean, std) in curves.items():
            if gds != ds:
                continue
            y = mean * 100.0
            ax.plot(rounds, y, label=goal, linewidth=1.6)
            if std is not None:
                lo = np.clip((mean - std) * 100.0, 0, 100)
                hi = np.clip((mean + std) * 100.0, 0, 100)
                ax.fill_between(rounds, lo, hi, alpha=0.22)
        ax.set_title(ds, fontsize=13)
        ax.set_xlabel("Round")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel("Test Accuracy (%)")
    fig.suptitle("Round vs Test Accuracy (mean ± std)", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)
    return FIG_PATH


def read_summary():
    """读 experiment_summary.csv 的逐 run 行, 返回按 (dataset, goal) 索引的 dict。"""
    out = {}
    with open(SUMMARY_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["run"] == "ALL":
                continue
            key = (row["dataset"], row["goal"])
            out[key] = {k: v for k, v in row.items()}
    return out


def write_report(curves, summary):
    lines = []
    A = lines.append
    A("# FedPKDA 实验报告\n")
    A(f"- 实验组数: {len(curves)} 组, 覆盖数据集: {', '.join(sorted(set(ds for ds, _ in curves)))}")
    A("- 数据来源: `results/*.h5` (每轮 test accuracy 曲线, 由 `main.py` 自动保存)")
    A("- 每组实验均为单次运行 (run=0); 图像中误差带为训练日志里每轮"
      "「Std Test Accurancy」(各客户端测试精度分布的标准差), 非 run 间方差\n")

    datasets = list(OrderedDict.fromkeys(ds for (ds, _) in curves))
    for ds in datasets:
        keys = [k for k in curves if k[0] == ds]
        A(f"## {ds}\n")
        A("| goal | best acc (%) | best round | last-5 avg (%) | final acc (%) |")
        A("| --- | ---: | ---: | ---: | ---: |")
        rows = []
        for key in keys:
            s = summary.get(key)
            if s:
                rows.append((key[1], s["best_acc_pct"], s["best_round"],
                             s["last5_acc_pct"], s["final_acc_pct"]))
        rows.sort(key=lambda r: -float(r[1]))
        for goal, best, br, last5, fin in rows:
            A(f"| {goal} | {float(best):.2f} | {br} | {float(last5):.2f} | {float(fin):.2f} |")
        if rows:
            best_goal, best_acc, _, _, _ = rows[0]
            gap = rows[0][1] if len(rows) == 1 else float(rows[0][1]) - float(rows[1][1])
            A(f"\n- 该数据集最优: `{best_goal}` (best acc {float(best_acc):.2f}%)"
              f"{f', 领先第二名 {gap:.2f}pp' if len(rows) > 1 else ''}\n")
        A("")

    A("## 关键观察\n")
    obs = []
    # Flowers102: Fedpkda vs FedAvg 同条件对比
    for ds in ("Flowers102", "NSLKDD"):
        pkda = [float(s["best_acc_pct"]) for k, s in summary.items() if k[0] == ds and "fedpkda" in k[1]]
        avg = [float(s["best_acc_pct"]) for k, s in summary.items() if k[0] == ds and "fedavg" in k[1]]
        if pkda and avg:
            obs.append(f"- **{ds}**: Fedpkda 最优 {max(pkda):.2f}% vs FedAvg 最优 {max(avg):.2f}%,"
                       f" 差距 {max(pkda) - max(avg):.2f}pp。")
    # NSLKDD: jr 与 lr 敏感性
    if any(k[0] == "NSLKDD" and k[1].endswith("_jr0.2") for k in summary):
        fpkda_jr = float(summary[("NSLKDD", "nslkdd_fedpkda_dnn_jr0.2")]["best_acc_pct"])
        favg_jr = float(summary[("NSLKDD", "nslkdd_fedavg_dnn_jr0.2")]["best_acc_pct"])
        favg_jr_final = float(summary[("NSLKDD", "nslkdd_fedavg_dnn_jr0.2")]["final_acc_pct"])
        obs.append(f"- **NSLKDD join_ratio 敏感性**: jr=0.2 时 Fedpkda 仍保持 {fpkda_jr:.2f}%,"
                   f" 而 FedAvg 掉到 {favg_jr:.2f}% (最终轮仅 {favg_jr_final:.2f}%),"
                   " 说明 Fedpkda 在客户端稀疏参与时更稳健。")
    if any(k[0] == "NSLKDD" and k[1].endswith("_lr5e-3") for k in summary):
        fpkda_lr = float(summary[("NSLKDD", "nslkdd_fedpkda_dnn_lr5e-3")]["best_acc_pct"])
        fpkda_base = float(summary[("NSLKDD", "nslkdd_fedpkda_dnn")]["best_acc_pct"])
        obs.append(f"- **NSLKDD 学习率**: lr=5e-3 比 lr=1e-3 略优 (Fedpkda {fpkda_lr:.2f}% vs {fpkda_base:.2f}%)。")
    if any(k[0] == "Cifar100" for k in summary):
        best = max(((k, float(s["best_acc_pct"])) for k, s in summary.items() if k[0] == "Cifar100"),
                   key=lambda t: t[1])
        obs.append(f"- **Cifar100**: 两个原型对齐权重设置 (lp0.3/gp0.5 与 lp0.6/gp0.3) 精度接近,"
                   f" 最优 {best[1]:.2f}%, 对齐权重敏感性不高。")
    A("\n".join(obs) if obs else "- (暂无自动生成的观察)")
    A("")

    A("## 图像\n")
    A("- `acc_by_round.png`: 轮数-精度曲线 (mean ± std), 按数据集分子图。\n")
    A("## 复现\n")
    A("1. 训练: 编辑 `run_experiments.sh` 的 `EXPERIMENTS` 后 `./run_experiments.sh`;")
    A("2. 汇总: `cd system && python summarize_results.py`;")
    A("3. 本报告: `cd system && python make_report.py`。")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return REPORT_PATH


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_results()
    curves = build_curves(df)
    if not curves:
        print("results/ 下没有可用的实验曲线")
        return
    fig = plot_curves(curves)
    summary = read_summary()
    report = write_report(curves, summary)
    # 附上 CSV 副本, 输出目录自包含
    for name in ("experiment_summary.csv", "curves.csv"):
        src = os.path.join(RESULTS_DIR, name)
        if os.path.exists(src):
            with open(src, "rb") as a, open(os.path.join(OUTPUT_DIR, name), "wb") as b:
                b.write(a.read())
    print(f"✅ 图像: {fig}")
    print(f"✅ 报告: {report}")
    print(f"✅ 输出目录: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
