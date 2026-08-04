# FedPKDA 实验报告

- 实验组数: 15 组, 覆盖数据集: Cifar100, Flowers102, NSLKDD
- 数据来源: `results/*.h5` (每轮 test accuracy 曲线, 由 `main.py` 自动保存)
- 每组实验均为单次运行 (run=0); 图像中误差带为训练日志里每轮「Std Test Accurancy」(各客户端测试精度分布的标准差), 非 run 间方差

## Cifar100

| goal | best acc (%) | best round | last-5 avg (%) | final acc (%) |
| --- | ---: | ---: | ---: | ---: |
| cifar100_fedpkda_lp0.3_gp0.5 | 45.29 | 40 | 44.75 | 45.29 |
| cifar100_fedpkda_lp0.6_gp0.3 | 45.24 | 40 | 44.75 | 45.24 |
| cifar100_fedavg_cnn | 27.87 | 34 | 27.48 | 27.43 |

- 该数据集最优: `cifar100_fedpkda_lp0.3_gp0.5` (best acc 45.29%), 领先第二名 0.05pp


## Flowers102

| goal | best acc (%) | best round | last-5 avg (%) | final acc (%) |
| --- | ---: | ---: | ---: | ---: |
| flowers_fedpkda_cnn | 36.65 | 40 | 35.53 | 36.65 |
| flowers_fedpkda_cnn_jr0.1 | 33.30 | 40 | 32.90 | 33.30 |
| flowers_fedavg_cnn | 30.15 | 40 | 28.55 | 30.15 |
| flowers_fedavg_cnn_jr0.1 | 25.25 | 36 | 23.65 | 23.80 |

- 该数据集最优: `flowers_fedpkda_cnn` (best acc 36.65%), 领先第二名 3.34pp


## NSLKDD

| goal | best acc (%) | best round | last-5 avg (%) | final acc (%) |
| --- | ---: | ---: | ---: | ---: |
| nslkdd_fedpkda_dnn_lr5e-3 | 99.75 | 58 | 99.75 | 99.75 |
| nslkdd_fedpkda_dnn | 99.68 | 57 | 99.68 | 99.68 |
| nslkdd_fedpkda_dnn_jr0.2 | 99.68 | 43 | 99.67 | 99.67 |
| nslkdd_fedpkda_dnn_jr0.1 | 99.67 | 43 | 99.66 | 99.66 |
| nslkdd_fedavg_dnn_lr5e-3 | 99.47 | 46 | 96.61 | 97.68 |
| nslkdd_fedavg_dnn | 99.35 | 36 | 95.18 | 96.64 |
| nslkdd_fedavg_dnn_jr0.1 | 94.09 | 55 | 55.31 | 53.39 |
| nslkdd_fedavg_dnn_jr0.2 | 93.72 | 17 | 67.36 | 55.02 |

- 该数据集最优: `nslkdd_fedpkda_dnn_lr5e-3` (best acc 99.75%), 领先第二名 0.06pp


## 关键观察

- **Cifar100** 方法对比: Fedpkda 最优 45.29% vs FedAvg 最优 27.87%, 差距 +17.42pp。
- **Flowers102** 方法对比: Fedpkda 最优 36.65% vs FedAvg 最优 30.15%, 差距 +6.50pp。
- **NSLKDD** 方法对比: Fedpkda 最优 99.75% vs FedAvg 最优 99.47%, 差距 +0.28pp。
- **Flowers102 稀疏参与敏感性** (jr=0.1): Fedpkda 33.30% vs FedAvg 25.25%, 差距 +8.05pp。
- **NSLKDD 稀疏参与敏感性** (jr=0.1): Fedpkda 99.67% vs FedAvg 94.09%, 差距 +5.58pp。
- **Cifar100 原型对齐权重**: lp/gp 两种设置精度接近, 最优 45.29%, 权重敏感性不高。
- **NSLKDD 学习率**: lr=5e-3 比 lr=1e-3 略优 (Fedpkda 99.75% vs 99.68%)。

## 图像

- `fig_Cifar100_accuracy.png`
- `fig_Flowers102_accuracy.png`
- `fig_NSLKDD_accuracy.png`

绘图约定: 蓝色=FedAvg, 红色=Fedpkda; 同色不同线型为参数变体(实线为基准配置); 误差带为 ±std。

## 复现

1. 训练: 编辑 `run_experiments.sh` 的 `EXPERIMENTS` 后 `./run_experiments.sh`;
2. 汇总: `cd system && python summarize_results.py`;
3. 本报告: `cd system && python make_report.py`。