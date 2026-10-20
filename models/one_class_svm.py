"""
One-Class Support Vector Machine (OC-SVM) Detector for Behavioral Fraud.
Learns a maximal margin hypersphere / hyperplane separating normal points from origin in kernel space.
"""

import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler
from .base import BaseAnomalyDetector


class OneClassSVMDetector(BaseAnomalyDetector):
    def __init__(
        self,
        kernel: str = "rbf",
        gamma: str = "scale",
        nu: float = 0.03,  # Upper bound on training errors & lower bound on SVs
    ):
        super().__init__(name="One-Class SVM")
        self.scaler = RobustScaler()
        self.model = OneClassSVM(
            kernel=kernel,
            gamma=gamma,
            nu=nu,
        )

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> "OneClassSVMDetector":
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        # score_samples returns signed distance to separating hyperplane:
        # positive for inliers, negative for outliers.
        # Invert so higher = more anomalous.
        return -self.model.score_samples(X_scaled)
