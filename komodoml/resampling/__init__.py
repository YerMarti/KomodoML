"""Resampling strategies module for KomodoML."""

from .base import ResamplingStrategy
from .kfold import KFoldFit, StratifiedKFoldFit, LeaveOneOutFit
from .bootstrap import BootstrapFit

__all__ = [
    "ResamplingStrategy",
    "KFoldFit",
    "StratifiedKFoldFit",
    "LeaveOneOutFit",
    "BootstrapFit"
]

for cls in [ResamplingStrategy, KFoldFit, StratifiedKFoldFit, LeaveOneOutFit, BootstrapFit]:
    cls.__module__ = "komodoml.resampling"