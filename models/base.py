"""
Base interface for Anomaly Detection Models in Behavioral Fraud Experiments.
Standardizes scoring so that higher scores ALWAYS indicate higher anomaly/fraud risk.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseAnomalyDetector(ABC):
    """
    Standard interface for all evaluated anomaly detection algorithms.
    """
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray = None) -> "BaseAnomalyDetector":
        """Fit model on training features (unsupervised or semi-supervised)."""
        pass

    @abstractmethod
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly score for each sample.
        Convention: HIGHER score = MORE ANOMALOUS (higher fraud likelihood).
        """
        pass

    def predict(self, X: np.ndarray, percentile_threshold: float = 98.0) -> np.ndarray:
        """
        Predict binary labels (1 = anomaly, 0 = normal) based on top percentile alert budget.
        """
        scores = self.score_samples(X)
        threshold = np.percentile(scores, percentile_threshold)
        return (scores >= threshold).astype(int)
