"""
Models module.

This module contains various model implementations including
classification, regression, and baseline models.
"""

# Base model
from dendritic_modeling.models.base import BaseModel

# Classification models
from dendritic_modeling.models.classifier import Classifier

# Recurrent classification models
from dendritic_modeling.models.recurrent_classifier import RecurrentClassifier

# Regression models
from dendritic_modeling.models.regressor import Regressor

__all__ = [
    # Base
    "BaseModel",
    # Classification
    "Classifier",
    # Recurrent classification
    "RecurrentClassifier",
    # Regression
    "Regressor",
]
