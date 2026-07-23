# FedPKDA 代码分析 — 平面化数据流

本文以平面方式追踪数据的每一次流转，以及每个步骤发生的具体操作。

---

## 1. 数据准备阶段

### 1.1 原始数据集加载

```
flowers-102 (torchvision.datasets.Flowers102)
  │
  ├── split="train"  → 6552 张图片
  ├── split="val"    → 818 张图片
  └── split="test"   → 614 张图片
      原始 3 个 split 被完全合并成一个 pool。
```

**操作**：`generate_Flowers102.py` 调用 `load_data()` 三次，每次用 `ToTensor()` + `Resize(64,64)` + `Normalize(0.5, 0.5)` 变换。所有图片拼成 `dataset_image` (numpy array)，标签拼成 `dataset_label`。

### 1.2 数据划分为客户端

```
dataset_image (N, 3, 64, 64)
dataset_label (N,)
  │
  ▼
separate_data(
    niid=True, balance=True, partition="dir",
    alpha=0.3, num_clients=35
)
  │
  ├── 用 Dirichlet(α=0.3) 分布生成 35 个客户端的标签比例
  │   └── 每个客户端得到非均衡的类分布
  ├── 保证每个客户端至少有 least_samples 个样本
  │
  ▼
X[35个客户端]: 每个客户端一个 numpy 数组 (图片)
y[35个客户端]: 每个客户端一个 numpy 数组 (标签)
```

### 1.3 训练/测试拆分

```
X[i], y[i]  (第 i 个客户端)
  │
  ▼
train_test_split(train_size=0.75)
  │
  ├── train_data[i] = {'x': X_train, 'y': y_train}
  └── test_data[i]  = {'x': X_test,  'y': y_test}
```

### 1.4 保存到磁盘

```
train_data[i]  →  dataset/Flowers102/train/{i}.npz   (35 个文件)
test_data[i]   →  dataset/Flowers102/test/{i}.npz    (35 个文件)
config.json    →  记录 num_clients, alpha, partition, balance 等
```

**注意**：Flowers102 生成了 35 个客户端的数据，但默认训练只用前 20 个（`-nc 20`）。

---

## 2. 程序入口 — `main.py`

```
用户执行:
  python main.py -algo Fedpkda
                 -data Flowers102
                 -m cnn
                 -nc 20
                 -gr 40
                 ...

  │
  ▼
解析命令行参数 → args 对象
  │
  ▼
根据 args.model 创建模型:
  "cnn" + "Flowers102" → FedAvgCNN(in_features=3, num_classes=102, dim=10816)
  结构:
    conv1: Conv2d(3→32, 5×5) + ReLU + MaxPool(2×2)
    conv2: Conv2d(32→64, 5×5) + ReLU + MaxPool(2×2)
    fc1:   Linear(10816→512) + ReLU
    fc:    Linear(512→102)     ← 分类头

  │
  ▼
根据 args.algorithm 选择算法:
  "Fedpkda" → 创建 Fedpkda 服务器
  "FedAvg"  → 创建 FedAvg 服务器

  │
  ▼
server.train()
```

---

## 3. 服务器初始化 — `__init__`

```python
# serverbase.py (父类)
self.global_model = copy.deepcopy(args.model)
self.clients = []
self.selected_clients = []
self.rs_test_acc = []      # 记录每轮的测试准确率
self.rs_train_loss = []    # 记录每轮的训练损失
self.rs_test_loss = []     # 记录每轮的测试损失
```

### FedAvg 特有的初始化

```
set_slow_clients()
  ├── 按 train_slow_rate 随机选部分客户端标记为"慢训练"
  └── 按 send_slow_rate 随机选部分客户端标记为"慢发送"

set_clients(clientAVG)
  └── 对每个 i in 0..19:
        ├── 读 train/{i}.npz 和 test/{i}.npz
        ├── 创建 clientAVG(args, id=i, ...)
        └── 存入 self.clients
```

### Fedpkda 特有的初始化

```
set_clients(clientpkda)
  └── 对每个 i in 0..19:
        ├── 读取数据 → 创建 clientpkda(args, id=i, ...)
        └── 使用 clientpkda 特有的属性:
              - self.fim_trace_history = []   # 存储 FIM trace
              - self.learn_scale = Parameter(5.0)
              - self.local_par = args.local_par   # 局部原型对齐权重
              - self.global_par = args.global_par  # 全局原型对齐权重
              - self.func = Func()                 # 激活函数
```

---

## 4. 客户端初始化 — `Client.__init__`

```
每个客户端创建时:

self.model = copy.deepcopy(args.model)   # 深拷贝全局模型
self.loss = CrossEntropyLoss()
self.optimizer = SGD(self.model.parameters(), lr=args.local_learning_rate)
self.learning_rate_scheduler = ExponentialLR(gamma=0.99)

self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
self.send_time_cost  = {'num_rounds': 0, 'total_cost': 0.0}
```

---

## 5. 训练循环主体

### 5.1 FedAvg 训练循环 — `serveravg.py`

```
for round i in range(0, 41):   # global_rounds=40 → 0..40

  step 1: 选择客户端
    select_clients()
      └── 从 20 个中随机选 4 个 (join_ratio=0.2)

  step 2: 评估（每 eval_gap=1 轮）
    evaluate()
      ├── test_metrics(): 对所有 20 个客户端评估
      │     └── 每个客户端: model.eval() → 遍历 test DataLoader
      │         ├── 累计正确数, 总样本数, loss
      │         └── 返回 (test_acc, test_num, auc=0, losses)
      ├── train_metrics(): 对所有 20 个客户端评估
      │     └── 每个客户端: model.eval() → 遍历 train DataLoader
      │         └── 返回 (losses, train_num)
      ├── 计算加权平均的 test_acc, train_loss, test_loss
      ├── 追加到 self.rs_test_acc, self.rs_train_loss, self.rs_test_loss
      └── 打印结果

  step 3: 发送模型
    send_models()
      └── 对所有 20 个客户端:
            client.set_parameters(self.global_model)
            └── 逐个参数复制: old_param.data = new_param.data.clone()

  step 4: 被选中的客户端本地训练
    for client in self.selected_clients:  (4 个)
      client.train()
        ├── 加载 train DataLoader (train/{id}.npz)
        ├── model.train()
        ├── for epoch in range(5):    (local_epochs=5)
        │     for batch (x, y) in loader:
        │       ├── x → device, y → device
        │       ├── output = model(x)
        │       ├── loss = CrossEntropyLoss(output, y)
        │       ├── optimizer.zero_grad()
        │       ├── loss.backward()
        │       └── optimizer.step()
        └── 如果 learning_rate_decay=True，衰减学习率

  step 5: 接收模型
    receive_models()
      ├── 从 4 个选中客户端中随机丢弃 client_drop_rate 比例
      ├── 对剩余活跃客户端:
      │     ├── 检查时间成本是否 < time_threthold
      │     └── 记录 client.id, client.train_samples (作为权重), client.model
      └── 计算权重归一化: w_i = sample_i / total_samples

  step 6: 聚合参数
    aggregate_parameters()
      ├── global_model = copy.deepcopy(uploaded_models[0])
      ├── global_model.param.data.zero_()
      └── for w, client_model in zip(weights, models):
            server_param.data += client_param.data * w
            # FedAvg 加权平均

  step 7: 自动早停检查
    if auto_break and check_done():
      └── 如果最佳准确率超过 top_cnt 轮未更新 → 停止

结束循环后:
  ├── 打印 Best accuracy: max(self.rs_test_acc)
  ├── 打印平均每轮时间
  ├── 计算通用化准确率: avg_generalization_metrics()
  ├── 保存结果到 h5: ../results/Flowers102_FedAvg_test_0.h5
  └── 保存模型: system/models/Flowers102/FedAvg_server.pt
```

### 5.2 Fedpkda 训练循环 — `serverpkda.py`

```
for round i in range(0, 41):

  step 1: 选择客户端
    self.selected_clients = select_clients()     # 4 个
    self.alled_clients = self.all_clients()      # 所有 20 个

  step 2: 评估（每 eval_gap=1 轮）
    与 FedAvg 相同

  step 3: 计算全局原型
    self.global_prototypes = compute_global_prototypes()
      ├── 对每个客户端 c in self.clients (20 个):
      │     local_noisy = c.get_noisy_local_prototypes()
      │       ├── model.eval()，遍历 train DataLoader
      │       ├── 对每个 batch:
      │       │     proto = model.base(x)         # 特征提取 (无分类头)
      │       │     proto = clamp(proto, -1, 1)
      │       │     按分类标签收集到 local_prototypes[label]
      │       └── 对每个类:
      │             ├── 计算均值: mean_proto
      │             ├── 添加 Laplace 噪声
      │             └── 存入 noisy_local_prototypes[label]
      │
      ├── 收集所有客户端的有噪声原型
      │     client_prototypes = [20 个客户端 × 102 个类]
      │
      └── 对每个类 k in 0..101:
            ├── 收集所有客户端中该类非 None 的原型
            ├── 如果 >= 3 个原型:
            │     ├── KMeans(n_clusters=1)  → cluster_center
            │     ├── 计算协方差矩阵 + 正则化
            │     ├── 计算 Mahalanobis 距离
            │     ├── 用距离倒数作为权重（越远权重越小）
            │     └── 加权平均 → global_prototypes[k]
            └── 否则: global_prototypes[k] = None

  step 4: 发送模型 + 全局原型 (仅对选中的客户端)
    send_selected_models(selected_ids, i)
      └── 对每个选中的客户端 (4 个):
            ├── progress = epoch / global_rounds
            └── client.set_parameters(
                  self.global_model,
                  self.global_prototypes,
                  progress,
                  epoch
                )

  step 5: 所有 20 个客户端各自训练
    for client in alled_clients:
      client.train(is_selected=(client.id in selected_ids))

    # 对于选中的客户端 (4 个):
    阶段 A — 本地 SGD 训练:
      └── 同 FedAvg: 5 epochs 的 SGD

    阶段 B — 计算 FIM trace:
      ├── model.eval()
      ├── 遍历 train DataLoader
      │     output = model(x)
      │     nll = -log_softmax(output)[label]
      │     grads = grad(nll, model.parameters())   # 对所有权重求梯度
      │     fim_trace_sum += sum(g^2 for all grads)  # Fisher 信息矩阵的迹
      └── 追加到 self.fim_trace_history

    阶段 C — 原型对齐 (在 set_parameters 中完成):
      ├── model.load_state_dict(global_model)
      ├── 获取局部原型: get_local_prototypes()
      │     ├── model.eval()，遍历 train DataLoader
      │     └── 收集每个类的特征均值 (不加噪声)
      │
      ├── 计算动态权重:
      │     ├── activation = Func(progress * learn_scale)
      │     │     Func: softplus(x) - softplus(-x)
      │     │           ────────────────────────  × scale
      │     │           softplus(x) + softplus(-x) + eps
      │     ├── lambda_PA = 0.1 + local_par * (1 - activation)
      │     └── lambda_FA = 0.1 + global_par * activation
      │
      ├── 原型对齐优化 (1 epoch):
      │     for batch (x, y):
      │       global_proto = model.base(x)
      │       loss = 0
      │       for each unique label in batch:
      │         if local_prototypes[label] exists:
      │           loss += lambda_PA * MSE(global_proto, local_proto[label])
      │         if global_prototypes[label] exists:
      │           loss += lambda_FA * MSE(global_proto, global_proto[label])
      │       alignment_optimizer.step()
      │
      └── 将优化后的 base 参数复制回 client model

    # 对于未选中的客户端 (16 个):
      ├── model.eval()
      └── 只计算 FIM trace (同阶段 B)

  step 6: 接收模型
    同 FedAvg

  step 7: Fisher 加权聚合 (替代 FedAvg 的样本数加权)
    aggregate_wrt_fisher()
      ├── 对每个上传的客户端:
      │     weight = FIM_trace[client] / sum(all FIM_traces)
      └── global_model = sum(weight_i * model_i)

  step 8: 自动早停检查
    同 FedAvg

结束循环后:
  同 FedAvg, 但文件名不同:
  ├── ../results/Flowers102_Fedpkda_test_0.h5
  └── system/models/Flowers102/Fedpkda_server.pt
```

---

## 6. 评估阶段

### 6.1 标准评估 (`evaluate` → `test_metrics`)

```
for each client in self.clients:
  model.eval()
  for batch (x, y) in test DataLoader:
    output = model(x)
    correct += argmax(output) == y
    total += len(y)
    loss += CrossEntropyLoss(output, y)

返回: (total_correct, total_samples, auc=0, losses)

服务器做加权平均:
  test_acc = sum(correct) / sum(samples)
  追加到 self.rs_test_acc
```

### 6.2 通用化度量 (`avg_generalization_metrics`)

```
收集所有客户端的 test DataLoader (20 个 loader)

for each client A in self.clients:
  for each loader (属于客户端 B):
    用客户端 A 的 model 评估客户端 B 的 test data
    acc = correct / total

  客户端 A 的通用化准确率 = 平均(在所有 loader 上的准确率)

返回 = 平均(所有客户端的通用化准确率)
```

### 6.3 结果保存 (`save_results`)

```
h5 文件 (e.g., ../results/Flowers102_Fedpkda_test_0.h5):
  ├── rs_test_acc:   每轮的测试准确率 (list of 41 floats)
  ├── rs_test_auc:   每轮的 AUC (全 0)
  └── rs_train_loss: 每轮的平均训练损失 (list of 41 floats)
```

### 6.4 最终打印 (`result_utils.average_data`)

```
读取 h5 文件 → 找到最佳准确率 → 计算 mean 和 std
```

---

## 7. 数据流全景图

### FedAvg 数据流

```
┌─────────────────────────────────────────────────────────┐
│                       每轮迭代 (0..40)                   │
│                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │ 选择客户端 │───→│ 发送全局  │───→│ 客户端本地 │         │
│  │ (随机4/20) │    │ 模型参数  │    │ SGD 训练  │         │
│  └──────────┘    └──────────┘    └─────┬────┘          │
│                                        │                │
│  ┌──────────┐    ┌──────────┐          │                │
│  │打印评估结果│←───│ 全局评估  │    ┌─────▼────┐         │
│  │ rs_test  │    │(所有客户端)│    │ 上传模型  │         │
│  │ _acc追加  │    └──────────┘    │ 参数+权重 │         │
│  └──────────┘                     └─────┬────┘          │
│                                         │                │
│  ┌──────────┐    ┌──────────┐          │                │
│  │ 早停检查  │←───│ 加权平均  │◄─────────┘                │
│  │ auto_break│    │聚合参数   │                          │
│  └──────────┘    └──────────┘                          │
└─────────────────────────────────────────────────────────┘
```

### Fedpkda 数据流 (红色为 FedAvg 没有的步骤)

```
┌──────────────────────────────────────────────────────────────┐
│                        每轮迭代 (0..40)                       │
│                                                              │
│  ┌──────────┐    ┌──────────────────┐    ┌───────────────┐  │
│  │ 选择客户端 │───→│ 🔴 计算全局原型   │───→│ 🔴 发送模型 +  │  │
│  │ (随机4/20) │    │ (K-Means +       │    │   全局原型    │  │
│  └──────────┘    │  Mahalanobis加权)  │    └───────┬───────┘  │
│                  └──────────────────┘            │          │
│                                   ┌──────────────▼───────┐  │
│                                   │ 🔴 所有客户端分别训练  │  │
│                                   │                      │  │
│                                   │ 选中客户端:           │  │
│                                   │  ├─ SGD 训练 (5 epoch)│  │
│                                   │  ├─ 🔴 FIM trace 计算 │  │
│                                   │  └─ 🔴 原型对齐优化    │  │
│                                   │     (local+global MSE)│  │
│                                   │                      │  │
│                                   │ 未选中客户端:         │  │
│                                   │  └─ 🔴 FIM trace 计算 │  │
│                                   └──────────┬───────────┘  │
│  ┌──────────┐    ┌──────────┐               │              │
│  │打印评估结果│←───│ 全局评估  │    ┌─────────▼──────┐       │
│  │ rs_test  │    │(所有客户端)│    │ 🔴 Fisher加权聚合│       │
│  │ _acc追加  │    └──────────┘    │ (FIM trace 作为  │       │
│  └──────────┘                     │  权重, 非样本数) │       │
│                                  └────────┬─────────┘       │
│  ┌──────────┐    ┌──────────┐             │                 │
│  │ 早停检查  │←───│ 加权平均  │◄────────────┘                 │
│  │ auto_break│    │聚合参数   │                               │
│  └──────────┘    └──────────┘                               │
└──────────────────────────────────────────────────────────────┘
```

---

## 8. 关键差异点总结

| 步骤 | FedAvg | FedPKDA |
| --- | --- | --- |
| **客户端选择** | 只选中的 4 个参与 | 选中 4 个参与完整训练 + 对齐；全部 20 个计算 FIM trace |
| **发送内容** | 仅模型参数 | 模型参数 + 全局原型 (102 个类的特征向量) |
| **客户端训练** | 5 epoch SGD | 5 epoch SGD + FIM trace + 1 epoch 原型对齐 |
| **原型计算** | 无 | local → noisy → global (K-Means + Mahalanobis) |
| **聚合方式** | 按样本数加权平均 | 按 FIM trace 加权平均 |
| **每轮计算量** | 低 (~1.1s) | 高 (~12.5s)，主要开销在原型相关操作 |

---

## 9. 文件/类依赖关系

```
main.py
  ├── flcore/servers/serveravg.py  → FedAvg(Server)
  │     └── flcore/clients/clientavg.py → clientAVG(Client)
  │           └── flcore/clients/clientbase.py → Client (基类)
  │
  ├── flcore/servers/serverpkda.py → Fedpkda(Server)
  │     └── flcore/clients/clientpkda.py → clientpkda(Client)
  │           ├── flcore/clients/clientbase.py → Client (基类)
  │           └── flcore/utils/func.py → Func
  │
  ├── flcore/utils/data_utils.py   → read_client_data()
  ├── utils/result_utils.py        → average_data()
  └── flcore/trainmodel/models.py  → FedAvgCNN, BaseHeadSplit

dataset/generate_Flowers102.py
  └── utils/dataset_utils.py → separate_data(), split_data(), save_file()
```
