# FedPKDA 实验报告

- 实验组数: 10 组, 覆盖数据集: Cifar100, Flowers102, NSLKDD
- 数据来源: `results/*.h5` (每轮 test accuracy 曲线, 由 `main.py` 自动保存)
- 每组实验均为单次运行 (run=0); 图像中误差带为训练日志里每轮「Std Test Accurancy」(各客户端测试精度分布的标准差), 非 run 间方差

## Cifar100

| goal | best acc (%) | best round | last-5 avg (%) | final acc (%) |
| --- | ---: | ---: | ---: | ---: |
| cifar100_fedpkda_lp0.3_gp0.5 | 45.29 | 40 | 44.75 | 45.29 |
| cifar100_fedpkda_lp0.6_gp0.3 | 45.24 | 40 | 44.75 | 45.24 |

- 该数据集最优: `cifar100_fedpkda_lp0.3_gp0.5` (best acc 45.29%), 领先第二名 0.05pp

## Flowers102

| goal | best acc (%) | best round | last-5 avg (%) | final acc (%) |
| --- | ---: | ---: | ---: | ---: |
| flowers_fedpkda_cnn | 36.65 | 40 | 35.53 | 36.65 |
| flowers_fedavg_cnn | 30.15 | 40 | 28.55 | 30.15 |

- 该数据集最优: `flowers_fedpkda_cnn` (best acc 36.65%), 领先第二名 6.50pp

## NSLKDD

| goal | best acc (%) | best round | last-5 avg (%) | final acc (%) |
| --- | ---: | ---: | ---: | ---: |
| nslkdd_fedpkda_dnn_lr5e-3 | 99.75 | 58 | 99.75 | 99.75 |
| nslkdd_fedpkda_dnn | 99.68 | 57 | 99.68 | 99.68 |
| nslkdd_fedpkda_dnn_jr0.2 | 99.68 | 43 | 99.67 | 99.67 |
| nslkdd_fedavg_dnn_lr5e-3 | 99.47 | 46 | 96.61 | 97.68 |
| nslkdd_fedavg_dnn | 99.35 | 36 | 95.18 | 96.64 |
| nslkdd_fedavg_dnn_jr0.2 | 93.72 | 17 | 67.36 | 55.02 |

- 该数据集最优: `nslkdd_fedpkda_dnn_lr5e-3` (best acc 99.75%), 领先第二名 0.06pp

## 关键观察

- **Flowers102**: Fedpkda 最优 36.65% vs FedAvg 最优 30.15%, 差距 6.50pp。
- **NSLKDD**: Fedpkda 最优 99.75% vs FedAvg 最优 99.47%, 差距 0.28pp。
- **NSLKDD join_ratio 敏感性**: jr=0.2 时 Fedpkda 仍保持 99.68%, 而 FedAvg 掉到 93.72% (最终轮仅 55.02%), 说明 Fedpkda 在客户端稀疏参与时更稳健。
- **NSLKDD 学习率**: lr=5e-3 比 lr=1e-3 略优 (Fedpkda 99.75% vs 99.68%)。
- **Cifar100**: 两个原型对齐权重设置 (lp0.3/gp0.5 与 lp0.6/gp0.3) 精度接近, 最优 45.29%, 对齐权重敏感性不高。

## 图像

- `acc_by_round.png`: 轮数-精度曲线 (mean ± std), 按数据集分子图。

## 复现

1. 训练: 编辑 `run_experiments.sh` 的 `EXPERIMENTS` 后 `./run_experiments.sh`;
2. 汇总: `cd system && python summarize_results.py`;
3. 本报告: `cd system && python make_report.py`。
