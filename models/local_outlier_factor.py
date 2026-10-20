"""
Local Outlier Factor (LOF) Detector for Behavioral Fraud Detection.
Evaluates local density relative to k-nearest neighbors to detect density-based outliers.
"""

import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from .base import BaseAnomalyDetector


class LocalOutlierFactorDetector(BaseAnomalyDetector):
    def __init__(
        self,
        n_neighbors: int = 25,
        contamination: float = 0.02,
    ):
        super().__init__(name="Local Outlier Factor (LOF)")
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True,  # Enables score_samples on unseen test samples
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> "LocalOutlierFactorDetector":
        self.model.fit(X)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        # Invert score so that higher = more anomalous
        # score_samples returns negative outlier factor
        return -self.model.score_samples(X)
