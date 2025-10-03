"""Models module for KomodoML."""

from .decision_tree import DecisionTreeClf, DecisionTreeReg

__all__ = [
    "DecisionTreeClf",
    "DecisionTreeReg"
]

for cls in [DecisionTreeClf, DecisionTreeReg]:
    cls.__module__ = "komodoml.models"