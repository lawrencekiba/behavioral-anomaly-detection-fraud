"""
Isolation Forest Detector for Behavioral Fraud Detection.
Partitions feature space using random orthogonal hyperplanes to isolate points.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from .base import BaseAnomalyDetector


class IsolationForestDetector(BaseAnomalyDetector):
    def __init__(
        self,
        n_estimators: int = 150,
        max_samples: str = "auto",
        contamination: float = 0.02,
        random_state: int = 42,
    ):
        super().__init__(name="Isolation Forest")
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> "IsolationForestDetector":
        self.model.fit(X)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        # Invert score so that higher = more anomalous
        # sklearn score_samples returns negative anomaly score (-1 to 0)
        return -self.model.score_samples(X)
