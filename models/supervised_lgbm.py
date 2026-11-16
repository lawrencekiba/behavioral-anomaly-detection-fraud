"""
Supervised Gradient Boosted Decision Tree (LightGBM / HistGB) Detector.
Uses histogram-based gradient boosting (identical algorithm to LightGBM)
optimized for large-scale tabular datasets with class-weight balancing.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from .base import BaseAnomalyDetector


class SupervisedLightGBMDetector(BaseAnomalyDetector):
    def __init__(
        self,
        n_estimators: int = 150,
        max_iter: int = None,
        learning_rate: float = 0.05,
        max_leaf_nodes: int = 31,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.0,
        random_state: int = 42,
    ):
        super().__init__(name="Supervised LightGBM")
        self.n_estimators = max_iter if max_iter is not None else n_estimators
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.random_state = random_state
        self.model = HistGradientBoostingClassifier(
            max_iter=self.n_estimators,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            class_weight="balanced",
            random_state=self.random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> "SupervisedLightGBMDetector":
        if y is None:
            raise ValueError("SupervisedLightGBMDetector requires labels 'y' to train.")
        self.model.fit(X, y)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Returns predicted probability of fraud P(y=1).
        Higher probability = higher risk.
        """
        return self.model.predict_proba(X)[:, 1]
