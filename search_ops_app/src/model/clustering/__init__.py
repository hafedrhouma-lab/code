"""module init"""
from src.model.interfaces import IModel
from .clustering_quantile import QuantileModel

__all__ = ["IModel", "QuantileModel"]
