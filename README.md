# FedPKDA

AAAI 2026

Prototype Knowledge Distillation for Federated Learning.

## Quick Start

- clone this repo

```bash
git clone --depth=1 https://github.com/zmx666/FedPKDA.git
cd FedPKDA
```

- install the dependences

```bash
# If you have conda env, recommand to use it
# conda create -n FedPKDA python=3.12 (Optional)
# conda activate FedPKDA              (Optional)
pip install -r requirements.txt
```

- download the dataset first

```bash
cd dataset && python generate_Flowers102.py
# or you can run this => python generate_cifar100.py  
# It depends on what dataset you would like to use, and Flowers102 is default dataset
```

- run the train process with default configuration

```bash
cd system
python main.py
# Or you can choose recored the std out at the same time
# python -u main.py | tee ../results/FedAvg.log
```

---

## Configurable Parameters

All parameters are set via command-line flags when running `system/main.py`.

### Algorithm Control

| Flag | Long | Type | Default | Description |
| ------ | ------ | ------ | --------- | ------------- |
| `-algo` | `--algorithm` | str | `Fedpkda` | Federated learning algorithm (`FedAvg` or `Fedpkda`) |
| `-m` | `--model` | str | `cnn` | Model architecture (`mlr` or `cnn`) |
| `-go` | `--goal` | str | `test` | Goal / experiment label for this run |

### Data & Task

| Flag | Long | Type | Default | Description |
| ------ | ------ | ------ | --------- | ------------- |
| `-data` | `--dataset` | str | `Flowers102` | Dataset name (e.g. `Cifar10`, `mnist`, `Flowers102`, `omniglot`, `Digit5`) |
| `-nb` | `--num_classes` | int | `102` | Number of classification classes |
| `-ml` | `--max_len` | int | `200` | Maximum sequence length (for text models) |

### Training Loop

| Flag | Long | Type | Default | Description |
| ------ | ------ | ------ | --------- | ------------- |
| `-gr` | `--global_rounds` | int | `40` | Number of global communication rounds |
| `-ls` | `--local_epochs` | int | `5` | Number of local training epochs per round |
| `-lbs` | `--batch_size` | int | `16` | Local batch size |
| `-pls` | `--plocal_epochs` | int | `1` | Number of local epochs for prototype learning |
| `-bnpc` | `--batch_num_per_client` | int | `2` | Number of batches per client per round |
| `-jr` | `--join_ratio` | float | `0.2` | Ratio of clients selected per round |
| `-rjr` | `--random_join_ratio` | bool | `False` | Whether to randomly vary the join ratio each round |
| `-nc` | `--num_clients` | int | `20` | Total number of clients |
| `-eg` | `--eval_gap` | int | `1` | Rounds gap between evaluations |
| `-ab` | `--auto_break` | bool | `False` | Whether to stop training early based on convergence |
| `-nnc` | `--num_new_clients` | int | `0` | Number of new clients added after training starts |
| `-fte` | `--fine_tuning_epoch` | int | `0` | Fine-tuning epochs after training |

### Optimization

| Flag | Long | Type | Default | Description |
| ------ | ------ | ------ | --------- | ------------- |
| `-lr` | `--local_learning_rate` | float | `0.005` | Local learning rate |
| `-ld` | `--learning_rate_decay` | bool | `False` | Whether to apply learning rate decay |
| `-ldg` | `--learning_rate_decay_gamma` | float | `0.99` | Decay factor for learning rate scheduling |
| `-mo` | `--momentum` | float | `0.1` | SGD momentum |
| `-slr` | `--server_learning_rate` | float | `1.0` | Server-side learning rate (for model aggregation) |
| `-tau` | `--tau` | float | `1.0` | Temperature parameter for knowledge distillation |
| `-bt` | `--beta` | float | `1.0` | Beta parameter (prototype balancing) |
| `-klw` | `--kl_weight` | float | `0.0` | Weight for KL divergence loss |

### FedPKDA-Specific

| Flag | Long | Type | Default | Description |
| ------ | ------ | ------ | --------- | ------------- |
| `--local_par` | — | float | `0.3` | Weight for local prototype alignment |
| `--global_par` | — | float | `0.5` | Weight for global prototype alignment |
| `-fd` | `--feature_dim` | int | `512` | Feature embedding dimension for prototypes |

### Hardware & Device

| Flag | Long | Type | Default | Description |
|------|------|------|---------|-------------|
| `-dev` | `--device` | str | `cuda` | Device to use (`cpu` or `cuda`) |
| `-did` | `--device_id` | str | `0` | CUDA device ID |

### Privacy

| Flag | Long | Type | Default | Description |
|------|------|------|---------|-------------|
| `-dp` | `--privacy` | bool | `False` | Enable differential privacy |
| `-dps` | `--dp_sigma` | float | `0.0` | Noise multiplier for differential privacy |

### Practical / Simulation

| Flag | Long | Type | Default | Description |
| ------ | ------ | ------ | --------- | ------------- |
| `-cdr` | `--client_drop_rate` | float | `0.0` | Rate of clients that train but drop out before sending updates |
| `-tsr` | `--train_slow_rate` | float | `0.0` | Rate of clients that train slowly (simulating stragglers) |
| `-ssr` | `--send_slow_rate` | float | `0.0` | Rate of clients that send model updates slowly |
| `-ts` | `--time_select` | bool | `False` | Whether to group and select clients based on time cost |
| `-tth` | `--time_threthold` | float | `10000` | Threshold (seconds) for dropping slow clients |

### Experiment Management

| Flag | Long | Type | Default | Description |
| ------ | ------ | ------ | --------- | ------------- |
| `-t` | `--times` | int | `1` | Number of repeated experiment runs |
| `-pv` | `--prev` | int | `0` | Starting index for experiment runs (for continuing runs) |
| `-sfn` | `--save_folder_name` | str | `items` | Folder name for saving results |

### Notes

- The commented-out flag `--fine_tuning_epochs` (original `-fte` with default 10) has been superseded by `--fine_tuning_epoch` (`-fte`) with default 0.
- When `cuda` is requested but unavailable, the system automatically falls back to `cpu`.
- The `--device_id` is overridden to `0` inside `main.py` immediately after parsing.
