import sys, torch
sys.path.insert(0, '.')
from flcore.trainmodel.models import FedAvgCNN
from flcore.trainmodel.resnet import resnet18, resnet10, resnet8

for name, fn, kwargs in [
    ('FedAvgCNN', FedAvgCNN, {'in_features': 3, 'num_classes': 102, 'dim': 10816}),
    ('resnet8', resnet8, {'num_classes': 102}),
    ('resnet10', resnet10, {'num_classes': 102}),
    ('resnet18', resnet18, {'num_classes': 102}),
]:
    m = fn(**kwargs)
    n = sum(p.numel() for p in m.parameters())
    print(f'{name:20s}: {n:>8,} params')
