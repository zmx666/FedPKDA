"""
Extract experiment results from .h5 files and export as JSON for Streamlit.

Usage:
    conda run -n robocup python extract_results.py

Output:
    ../results/experiment_data.json  -- all curves in one file for Streamlit
"""
import json
import h5py
import numpy as np
import os

EXPERIMENTS = [
    {
        "key": "fedavg_cnn",
        "label": "FedAvg + CNN (5.6M)",
        "color": "#94a3b8",
        "file": "Cifar100_FedAvg_feavg_cnn_0",
    },
    {
        "key": "fedpkda_cnn",
        "label": "FedPKDA + CNN (5.6M)",
        "color": "#3b82f6",
        "file": "Cifar100_Fedpkda_fepkda_cnn_0",
    },
    {
        "key": "fedavg_resnet18",
        "label": "FedAvg + ResNet-18 (11.2M)",
        "color": "#94a3b8",
        "file": "Cifar100_FedAvg_feavg_resnet18_0",
    },
    {
        "key": "fedpkda_resnet18",
        "label": "FedPKDA + ResNet-18 (11.2M)",
        "color": "#f59e0b",
        "file": "Cifar100_Fedpkda_pkda_resnet18_0",
    },
]


def extract():
    output = {"experiments": [], "meta": {}}

    for exp in EXPERIMENTS:
        path = os.path.join("../results", exp["file"] + ".h5")
        with h5py.File(path, "r") as hf:
            acc = np.array(hf.get("rs_test_acc")).tolist()
            loss = np.array(hf.get("rs_train_loss")).tolist()
            best_acc = max(acc)
            best_round = acc.index(best_acc)
            last5_avg = np.mean(acc[-5:]).item() if len(acc) >= 5 else np.mean(acc)

        output["experiments"].append(
            {
                "key": exp["key"],
                "label": exp["label"],
                "color": exp["color"],
                "test_acc": [round(v * 100, 2) for v in acc],  # convert to %
                "train_loss": [round(v, 4) for v in loss],
                "best_acc": round(best_acc * 100, 2),
                "best_round": best_round,
                "last5_avg": round(last5_avg * 100, 2),
                "num_rounds": len(acc),
            }
        )

    output["meta"] = {
        "dataset": "CIFAR-100",
        "num_classes": 100,
        "num_clients": 20,
        "global_rounds": len(output["experiments"][0]["test_acc"]) - 1 if output["experiments"] else 0,
    }

    out_path = "../results/experiment_data.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"✅ Extracted {len(output['experiments'])} experiments → {out_path}")
    for exp in output["experiments"]:
        print(
            f"   {exp['label']:30s}  best={exp['best_acc']:5.2f}% @ round {exp['best_round']:2d}  last5={exp['last5_avg']:5.2f}%"
        )


if __name__ == "__main__":
    extract()
