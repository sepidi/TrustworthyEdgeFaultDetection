"""1D CNN models on spectral input (N, 1, F).

wdcnn    : WDCNN-style 5-conv network (Zhang et al., Sensors 2017), wide first kernel.
resnet1d : lightweight 1D ResNet (4 stages, 1 BasicBlock each), ~0.7M params.
Both end in global average pooling + linear head so structured pruning and ONNX
export work without shape bookkeeping.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def conv_bn(cin, cout, k, s=1, p=None):
    p = k // 2 if p is None else p
    return nn.Sequential(nn.Conv1d(cin, cout, k, s, p, bias=False), nn.BatchNorm1d(cout), nn.ReLU(inplace=True))


class WDCNN(nn.Module):
    def __init__(self, n_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            conv_bn(1, 16, 64, 8, 28), nn.MaxPool1d(2),
            conv_bn(16, 32, 3), nn.MaxPool1d(2),
            conv_bn(32, 64, 3), nn.MaxPool1d(2),
            conv_bn(64, 64, 3), nn.MaxPool1d(2),
            conv_bn(64, 64, 3), nn.MaxPool1d(2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        return self.fc(torch.flatten(self.pool(self.features(x)), 1))


class BasicBlock1d(nn.Module):
    def __init__(self, cin, cout, stride, k=5):
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, k, stride, k // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, k, 1, k // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout))

    def forward(self, x):
        idt = x if self.down is None else self.down(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + idt)


class ResNet1D(nn.Module):
    def __init__(self, n_classes: int = 3, widths=(32, 64, 128, 256)):
        super().__init__()
        self.stem = nn.Sequential(conv_bn(1, widths[0], 15, 2), nn.MaxPool1d(3, 2, 1))
        layers, cin = [], widths[0]
        for i, w in enumerate(widths):
            layers.append(BasicBlock1d(cin, w, 1 if i == 0 else 2))
            cin = w
        self.layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(cin, n_classes)

    def forward(self, x):
        return self.fc(torch.flatten(self.pool(self.layers(self.stem(x))), 1))


def build_model(arch: str, n_classes: int) -> nn.Module:
    if arch == "wdcnn":
        return WDCNN(n_classes)
    if arch == "resnet1d":
        return ResNet1D(n_classes)
    raise ValueError(f"Unknown arch {arch!r}")


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
