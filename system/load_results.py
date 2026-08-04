#!/usr/bin/env python
"""
把所有实验的 .h5 曲线导出为适合 pandas 直接加载的长表:
    results/curves.csv
    列: dataset, algorithm, goal, run, round, test_acc, test_auc, train_loss
    每行 = 某次实验 (run) 在第 round 轮的一个精度/损失点。

用法:
    cd system && python load_results.py      # 生成 ../results/curves.csv
    # 或作为模块导入 (Jupyter 里直接用):
    from load_results import load_results
    df = load_results()                      # 返回 pandas DataFrame

背景: serverbase.save_results() 用 h5py 原生写 HDF5, 不是 pandas HDFStore 格式,
所以 pd.read_hdf() 读不了这些文件; 本脚本就是标准做法:
h5py 读出 numpy 数组 → 长表 DataFrame → CSV。
画图时按 goal / algorithm 分组即可, 例如:
    import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
    df = pd.read_csv("../results/curves.csv")
    sns.lineplot(data=df, x="round", y="test_acc", hue="goal")
    plt.show()
"""
import glob
import os
import sys

import numpy as np

from summarize_results import RESULTS_DIR, parse_stem, load_metrics

CSV_PATH = os.path.join(RESULTS_DIR, "curves.csv")

COLUMNS = ["dataset", "algorithm", "goal", "run", "round",
           "test_acc", "test_auc", "train_loss"]

try:
    import pandas as pd
except ImportError:  # pandas 只在需要 DataFrame/CSV 时用, h5 读取本身只依赖 h5py+numpy
    pd = None


def load_results(results_dir=RESULTS_DIR):
    """返回长表 DataFrame, 每行 = (实验, run, round) 的一个精度/损失点。"""
    rows = []
    for f in sorted(glob.glob(os.path.join(results_dir, "*.h5"))):
        stem = os.path.splitext(os.path.basename(f))[0]
        try:
            acc, auc, tloss = load_metrics(f)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] 跳过 {f}: {e}")
            continue
        if acc.size == 0:
            print(f"[warn] 跳过空结果 {f}")
            continue

        dataset, algorithm, goal, run = parse_stem(stem)
        n = int(acc.size)
        auc_arr = auc if auc is not None else np.full(n, np.nan)
        loss_arr = tloss if tloss is not None else np.full(n, np.nan)

        for r in range(n):
            rows.append((
                dataset, algorithm, goal, run, r,
                float(acc[r]),
                float(auc_arr[r]) if auc_arr.size > r else np.nan,
                float(loss_arr[r]) if loss_arr.size > r else np.nan,
            ))

    if pd is None:
        sys.exit("需要 pandas: pip install pandas (h5 读取本身只用 h5py+numpy)")
    return pd.DataFrame(rows, columns=COLUMNS)


def main():
    df = load_results()
    df.to_csv(CSV_PATH, index=False)
    print(f"✅ 导出 {len(df)} 行 → {CSV_PATH}")
    if len(df):
        exps = (df.groupby(["dataset", "algorithm", "goal"], dropna=False)
                  .size().reset_index(name="points"))
        print(f"   共 {len(exps)} 组实验, 每组曲线点数:")
        print(exps.to_string(index=False))
        print("\n示例 (前 5 行):")
        print(df.head().to_string(index=False))
        print(f"\n画图示例: pd.read_csv('{CSV_PATH}') 后按 goal 分组画 x=round, y=test_acc")


if __name__ == "__main__":
    main()
