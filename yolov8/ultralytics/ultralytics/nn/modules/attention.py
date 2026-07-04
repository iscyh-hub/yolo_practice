# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Custom attention modules for YOLOv8 blood-cell detection experiments."""

import math

import torch
import torch.nn as nn

from .block import C2f
from .conv import CBAM


class SE(nn.Module):
    """Squeeze-and-Excitation channel attention module.

    Args:
        c1 (int): Number of input channels.
        reduction (int): Channel reduction ratio for the fully-connected bottleneck.
    """

    def __init__(self, c1, reduction=16):
        """Initialize SE module."""
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c1, c1 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c1 // reduction, c1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Apply channel attention."""
        b, c, _, _ = x.shape
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ECA(nn.Module):
    """Efficient Channel Attention module using a 1D convolution.

    Args:
        c1 (int): Number of input channels.
        gamma (int): Gamma parameter for adaptive kernel size.
        b (int): Bias parameter for adaptive kernel size.
    """

    def __init__(self, c1, gamma=2, b=1):
        """Initialize ECA module."""
        super().__init__()
        kernel_size = int(abs((math.log(c1, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Apply efficient channel attention."""
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


class C2f_CBAM(C2f):
    """C2f block followed by a CBAM attention tail.

    Inheriting from C2f keeps the original weight keys (cv1, cv2, m) unchanged,
    so pretrained YOLOv8 weights can be transferred safely.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize C2f_CBAM module."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.cbam = CBAM(c2)

    def forward(self, x):
        """Forward pass with CBAM attention."""
        return self.cbam(super().forward(x))


class C2f_SE(C2f):
    """C2f block followed by an SE attention tail."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize C2f_SE module."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.se = SE(c2)

    def forward(self, x):
        """Forward pass with SE attention."""
        return self.se(super().forward(x))


class C2f_ECA(C2f):
    """C2f block followed by an ECA attention tail."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize C2f_ECA module."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.eca = ECA(c2)

    def forward(self, x):
        """Forward pass with ECA attention."""
        return self.eca(super().forward(x))
