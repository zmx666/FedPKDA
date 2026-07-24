"""Inspect HDF5 result files and print summary."""
import h5py
import numpy as np

files = [
    'Cifar100_FedAvg_feavg_cnn_0',
    'Cifar100_Fedpkda_fepkda_cnn_0',
    'Cifar100_Fedpkda_pkda_resnet18_0',
    'Cifar100_FedAvg_feavg_resnet18_0',
]

for f in files:
    path = f'../results/{f}.h5'
    with h5py.File(path, 'r') as hf:
        acc = np.array(hf.get('rs_test_acc'))
        loss = np.array(hf.get('rs_train_loss'))
        auc = hf.get('rs_test_auc')
        if auc is not None:
            auc = np.array(auc)
    
    best_idx = np.argmax(acc)
    tail_mean = np.mean(acc[-5:]) if len(acc) >= 5 else np.mean(acc)
    best_pct = acc[best_idx] * 100
    tail_pct = tail_mean * 100
    print(f'{f}')
    print(f'  test_acc: {len(acc)} rounds, best={best_pct:.2f}% @ round {best_idx}')
    print(f'  last-5 avg: {tail_pct:.2f}%')
    print(f'  train_loss: {len(loss)} rounds')
    if auc is not None:
        print(f'  test_auc:  {len(auc)} rounds')
    print()
