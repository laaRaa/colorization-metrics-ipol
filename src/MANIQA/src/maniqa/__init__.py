"""Multi-dimension Attention Network for No-Reference Image Quality Assessment.

This is a python module adapted from the official PyTorch implementation of [Yang et al.](https://github.com/IIGROUP/MANIQA).
"""

from . import inference, models, timm, utils

__all__ = ["inference", "models", "timm", "utils"]
