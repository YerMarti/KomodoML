"""Base module for KomodoML."""

from .base_model import BaseModel
from .wrapper import Wrapper

__all__ = [
    "BaseModel",
    "Wrapper",
]

for cls in [BaseModel, Wrapper]:
    cls.__module__ = "komodoml.base"