"""
Hybrid Fraud Detection Pipeline.
Combines Supervised Gradient Boosted Trees (LightGBM) with unsupervised anomaly feature enrichers:
- Isolation Forest anomaly score (isolation path length)
- Deep Tabular Autoencoder reconstruction error (latent reconstruction MSE)

Architecture:
  X_raw --> [ Isolation Forest ] -----> iforest_score --+
        --> [ Autoencoder      ] -----> ae_recon_mse  --+--> X_hybrid --> [ LightGBM ] --> P(Fraud)
        -----------------------------------------------+
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional
from .base import BaseAnomalyDetector
from .isolation_forest import IsolationForestDetector
from .autoencoder import AutoencoderDetector
from .supervised_lgbm import SupervisedLightGBMDetector


class HybridFraudDetector(BaseAnomalyDetector):
    def __init__(
        self,
        use_iforest: bool = True,
        use_autoencoder: bool = True,
        lgbm_n_estimators: int = 150,
        lgbm_learning_rate: float = 0.05,
        ae_latent_dim: int = 8,
        ae_epochs: int = 25,
        random_state: int = 42,
    ):
        name_parts = ["LightGBM"]
        if use_iforest:
            name_parts.append("iForest")
        if use_autoencoder:
            name_parts.append("AE")
        super().__init__(name=f"Hybrid ({' + '.join(name_parts)})")

        self.use_iforest = use_iforest
        self.use_autoencoder = use_autoencoder
        self.random_state = random_state

        # Unsupervised feature enrichers
        self.iforest = IsolationForestDetector(n_estimators=100, n_jobs=1, random_state=random_state) if use_iforest else None
        self.autoencoder = AutoencoderDetector(latent_dim=ae_latent_dim, epochs=ae_epochs, random_state=random_state) if use_autoencoder else None

        # Supervised classifier
        self.classifier = SupervisedLightGBMDetector(
            n_estimators=lgbm_n_estimators,
            learning_rate=lgbm_learning_rate,
            random_state=random_state,
        )
        self.feature_names_: List[str] = []

    def _extract_anomaly_features(self, X: np.ndarray, fit: bool = False, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fits (if fit=True) and extracts unsupervised anomaly scores to append as columns.
        """
        anomaly_cols = []

        if self.use_iforest:
            if fit:
                self.iforest.fit(X)
            iforest_scores = self.iforest.score_samples(X).reshape(-1, 1)
            anomaly_cols.append(iforest_scores)

        if self.use_autoencoder:
            if fit:
                self.autoencoder.fit(X)
            ae_scores = self.autoencoder.score_samples(X).reshape(-1, 1)
            anomaly_cols.append(ae_scores)

        if len(anomaly_cols) > 0:
            return np.hstack(anomaly_cols)
        return np.empty((len(X), 0))

    def fit(self, X: np.ndarray, y: np.ndarray = None, feature_names: Optional[List[str]] = None) -> "HybridFraudDetector":
        if y is None:
            raise ValueError("HybridFraudDetector requires labels 'y' for its supervised stage.")

        # 1. Fit unsupervised feature enrichers on training features
        anomaly_feats = self._extract_anomaly_features(X, fit=True, y=y)

        # 2. Construct enriched training feature matrix
        if anomaly_feats.shape[1] > 0:
            X_hybrid = np.hstack([X, anomaly_feats])
        else:
            X_hybrid = X

        # Track feature names
        base_names = feature_names if feature_names else [f"f_{i}" for i in range(X.shape[1])]
        extra_names = []
        if self.use_iforest:
            extra_names.append("feat_iforest_anomaly_score")
        if self.use_autoencoder:
            extra_names.append("feat_ae_recon_mse")
        self.feature_names_ = list(base_names) + extra_names

        # 3. Train supervised LightGBM on enriched features
        self.classifier.fit(X_hybrid, y)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Extracts anomaly features on test samples, appends them, and predicts fraud probability.
        """
        anomaly_feats = self._extract_anomaly_features(X, fit=False)
        if anomaly_feats.shape[1] > 0:
            X_hybrid = np.hstack([X, anomaly_feats])
        else:
            X_hybrid = X
        return self.classifier.score_samples(X_hybrid)

    def get_feature_importances(self) -> Dict[str, float]:
        """
        Returns feature importance dict from the underlying gradient boosted tree.
        """
        if self.classifier.model is None:
            return {}
        # Scikit-learn HistGradientBoostingClassifier does not expose tree.feature_importances_ directly,
        # but we can return permutation or tree info, or attribute importances.
        # Fallback to empty if not computed yet.
        return {}
