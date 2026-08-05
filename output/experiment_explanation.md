# FedPKDA 实验说明与复现文档

> 本文档描述 `output/` 目录下所有结果/图表/报告的**复现步骤**，以及实验配置、自动保存机制与注意事项。
> 对应代码仓库：FedPKDA（AAAI 2026，Prototype Knowledge Distillation for Federated Learning）。

---

## 1. 当前实验概览

- 共 **15 组实验**，覆盖 3 个数据集：`Cifar100`、`Flowers102`、`NSLKDD`
- 两种方法：**FedAvg**（基线）与 **Fedpkda**（原型知识蒸馏）
- 每组实验均为单次运行（run=0）；结果曲线保存在 `results/*.h5`
- 当前 output 产物：

| 文件 | 说明 |
| --- | --- |
| `report.md` | 实验报告：每数据集汇总表 + 关键观察（自动生成） |
| `fig_{dataset}_accuracy.png` ×3 | 每数据集一张「轮数-精度（mean ± std）」图（自动生成） |
| `experiment_summary.csv` | 15 组实验的 best/last-5/final 精度汇总表 |
| `curves.csv` | 全部实验逐轮曲线长表（pandas 可直接加载画图） |
| `batch_run.log` | 最近一次批量训练的运行记录 |

---

## 2. 环境准备

```bash
# 创建并激活环境（已有 robocup 环境可直接激活）
conda create -n robocup python=3.12 -y
conda activate robocup
pip install -r requirements.txt
# 画图/数据分析额外依赖（服务器无显示，仅需要库本身）
pip install matplotlib pandas seaborn

# 验证
python -c "import torch, h5py, numpy, pandas, matplotlib; print('OK')"
```

> 注意：仓库所有脚本内部调用未限定路径的 `python`，**运行前必须先激活环境**。

---

## 3. 数据准备

三个数据集均已生成好（`dataset/{数据集}/train|test/*.npz`，按客户端分片）。如需重新生成：

```bash
cd dataset
python generate_cifar100.py      # Cifar100, 20 个 client
python generate_nslkdd.py        # NSLKDD, 20 个 client
python generate_Flowers102.py    # Flowers102, 35 个 client
```

---

## 4. 实验配置（15 组全量清单）

### 批次 1（已完成，数据在 results/，10 组）

| 数据集 | 方法 | goal（h5 标识） | 关键参数 |
| --- | --- | --- | --- |
| Cifar100 | Fedpkda | `cifar100_fedpkda_lp0.3_gp0.5` | cnn, 40轮, lr5e-3, jr0.2, `--local_par 0.3 --global_par 0.5` |
| Cifar100 | Fedpkda | `cifar100_fedpkda_lp0.6_gp0.3` | cnn, 40轮, lr5e-3, jr0.2, `--local_par 0.6 --global_par 0.3` |
| Flowers102 | Fedpkda | `flowers_fedpkda_cnn` | cnn, 40轮, lr5e-3, jr0.2 |
| Flowers102 | FedAvg | `flowers_fedavg_cnn` | cnn, 40轮, lr5e-3, jr0.2 |
| NSLKDD | Fedpkda | `nslkdd_fedpkda_dnn` | dnn, 60轮, lr1e-3, jr0.6 |
| NSLKDD | FedAvg | `nslkdd_fedavg_dnn` | dnn, 60轮, lr1e-3, jr0.6 |
| NSLKDD | Fedpkda | `nslkdd_fedpkda_dnn_lr5e-3` | dnn, 60轮, lr5e-3, jr0.6 |
| NSLKDD | FedAvg | `nslkdd_fedavg_dnn_lr5e-3` | dnn, 60轮, lr5e-3, jr0.6 |
| NSLKDD | Fedpkda | `nslkdd_fedpkda_dnn_jr0.2` | dnn, 60轮, lr1e-3, jr0.2 |
| NSLKDD | FedAvg | `nslkdd_fedavg_dnn_jr0.2` | dnn, 60轮, lr1e-3, jr0.2 |

### 批次 2（区分度补充，5 组）

| 数据集 | 方法 | goal | 关键参数 |
| --- | --- | --- | --- |
| Cifar100 | FedAvg | `cifar100_fedavg_cnn` | cnn, 40轮, lr5e-3, jr0.2（Fedpkda 基线对照） |
| NSLKDD | Fedpkda | `nslkdd_fedpkda_dnn_jr0.1` | dnn, 60轮, lr1e-3, **jr0.1**（极端稀疏） |
| NSLKDD | FedAvg | `nslkdd_fedavg_dnn_jr0.1` | dnn, 60轮, lr1e-3, **jr0.1** |
| Flowers102 | Fedpkda | `flowers_fedpkda_cnn_jr0.1` | cnn, 40轮, lr5e-3, **jr0.1** |
| Flowers102 | FedAvg | `flowers_fedavg_cnn_jr0.1` | cnn, 40轮, lr5e-3, **jr0.1** |

固定公共参数：`-nc` 见上表、`-nb`（类别数 100/102/2）、`-lbs 16`、`-ls 5`、`-mo 0.1/0.9`（按原 run.sh 约定）。

---

## 5. 复现步骤（完整流程）

### Step 0 激活环境

```bash
conda activate robocup
```

### Step 1 训练

编辑根目录 `run_experiments.sh` 的 `EXPERIMENTS` 列表，格式为 `"实验名 | main.py 参数"`：

- **要重跑某组**：取消该行注释；已跑过的组保持注释（防覆盖预检会拦截同名重跑）
- **要完全重跑全部 15 组**：先归档/移动旧结果，再取消全部注释

```bash
# 归档旧数据（可选，防覆盖预检要求目标 .h5 不存在）
mkdir -p results/archive && mv results/*.h5 results/*.log results/archive/

# 一键批量训练（会自动 tee 日志到 results/{实验名}.log）
./run_experiments.sh

# 也可以手动跑单条（等价于 run.sh 的命令）
cd system
python -u main.py -algo Fedpkda -data NSLKDD -m dnn -nc 20 -nb 2 -gr 60 \
    -lr 0.001 -mo 0.9 -lbs 16 -jr 0.1 -ls 5 -go nslkdd_fedpkda_dnn_jr0.1 \
    | tee ../results/nslkdd_fedpkda_dnn_jr0.1.log
```

脚本特性：

- **开跑前冲突预检**：目标 `results/{dataset}_{algo}_{goal}_0.h5` 已存在则整体中止，绝不覆盖旧数据
- `RUN_TAG=xxx ./run_experiments.sh`：给所有 goal 追加后缀，同一组参数可重复跑而不覆盖
- 全部成功后退出码 0；任一组失败则退出码 1（`pipefail` 保证 tee 不掩盖 python 失败）

### Step 2 汇总

```bash
cd system
python summarize_results.py   # 解析全部 .h5 → results/experiment_summary.csv（best/last-5/final）
python load_results.py        # 导出 results/curves.csv（逐轮长表, 供 pandas 画图）
```

### Step 3 生成报告与图像

```bash
cd system
python make_report.py         # → output/report.md + output/fig_{dataset}_accuracy.png + CSV 副本
```

或一次性：`./run_experiments.sh --summary-only`（跳过训练，只刷新汇总/曲线表）。

---

## 6. 自动保存机制（重要）

依据代码 `system/flcore/servers/serverbase.py` 的 `save_results()`：

1. 每次训练结束自动写 `results/{dataset}_{algorithm}_{goal}_{run_idx}.h5`
   - 内容：`rs_test_acc` / `rs_test_auc` / `rs_train_loss`（每轮一条）
   - `run_idx` 由 `-t/--times` 控制，默认 0
2. 文件以 **'w' 模式写入 → 同名直接覆盖**。因此防覆盖的唯一规则：**每次实验的 `-go`(goal) 唯一**（`dataset/algorithm/goal` 组合唯一即可）
3. 日志由 `tee` 保留为 `results/{实验名}.log`

解析工具：

- `summarize_results.py` — 汇总统计 → `experiment_summary.csv`
- `load_results.py` — 逐轮曲线长表 → `curves.csv`（列：`dataset, algorithm, goal, run, round, test_acc, test_auc, train_loss`）

> h5 是 h5py 原生格式，**不能**直接用 `pd.read_hdf`；用 `load_results.py` 或 `h5py` 读取后转 DataFrame 即可。

---

## 7. 图像说明

- 每数据集一张图（`fig_Cifar100_accuracy.png` / `fig_Flowers102_accuracy.png` / `fig_NSLKDD_accuracy.png`）
- **每图仅两种颜色**：蓝 = FedAvg，红 = Fedpkda；同方法的不同参数变体共享颜色、用线型区分（实线=基准配置，虚线/点线=变体）
- 误差带为 ±std：当前为单次运行，std 取自训练日志每轮的「Std Test Accurancy」（各客户端精度分布标准差）；若用 `-t N` 跑多次，脚本自动改用 run 间标准差
- 使用 matplotlib **Agg 后端**，直接保存 PNG，不弹窗（服务器无显示环境）

---

## 8. 当前结果摘要（自动生成于 report.md）

- **Cifar100**：Fedpkda 45.29% vs FedAvg 27.87%（+17.42pp）
- **Flowers102**：Fedpkda 36.65% vs FedAvg 30.15%（+6.50pp）；jr=0.1 时 +8.05pp
- **NSLKDD**：Fedpkda 99.75% vs FedAvg 99.47%（+0.28pp）；jr=0.1 时 FedAvg 崩至 94.09%（最终轮 53.39%），Fedpkda 保持 99.67%
- 结论：Fedpkda 在客户端稀疏参与（低 jr）场景优势显著，且对原型对齐权重（lp/gp）不敏感

---

## 9. 注意事项

1. **std 语义**：单次运行的误差带是"客户端间"标准差，非"多次运行间"方差；论文级 mean±std 请用 `-t 3~5` 重跑
2. **检查点覆盖**：`system/models/{dataset}/{algorithm}_server.pt` 为模型检查点，同一 (dataset, algorithm) 会被覆盖；实验数据以 `.h5` / `.log` 为准
3. **goal 命名**：避免纯数字（文件名解析按末尾整数识别 run 序号）；建议把关键参数编码进 goal（如 `_lr5e-3`、`_jr0.1`）
4. **中文字体**：matplotlib 默认 DejaVu Sans 无 CJK 字形，图内文字用英文，报告正文为中文 Markdown
5. **防覆盖**：重跑同名实验前，要么改 goal、要么 `RUN_TAG`、要么先归档旧 `.h5`/`.log`
