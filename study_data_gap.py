"""
Deep Diagnostic Study: Explaining the Performance Gap Between Synthetic and Real-World Fraud Data.

Investigates why unsupervised anomaly detection (e.g., Isolation Forest) succeeds on synthetic
behavioral drift (PR-AUC ~0.55) but collapses on real-world transaction streams like IEEE-CIS (PR-AUC ~0.04).

Key Hypotheses Investigated:
1. The 'Benign Outlier' Problem: What proportion of statistical outliers (>99th percentile) are actually legitimate?
2. The 'Camouflage' Problem: What proportion of frauds hide completely inside the normal IQR?
3. Distributional Separability: Kolmogorov-Smirnov (KS) statistic and Wasserstein distance between Fraud and Legit distributions.
4. Multimodality & Sparsity: Zero-inflation and feature variance.
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from typing import Dict, Any, List


def analyze_dataset_geometry(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    dataset_name: str,
) -> Dict[str, Any]:
    """
    Performs empirical geometric analysis comparing normal vs fraud instances.
    """
    legit_mask = (y == 0)
    fraud_mask = (y == 1)
    
    n_legit = np.sum(legit_mask)
    n_fraud = np.sum(fraud_mask)
    base_rate = n_fraud / len(y)
    
    X_legit = X[legit_mask]
    X_fraud = X[fraud_mask]
    
    # 1. Outlier Composition at the 99th percentile per feature
    # For each feature, if we take the top 1% most extreme values, what % are legit vs fraud?
    tail_frauds_pct = []
    tail_legits_count = []
    ks_scores = []
    
    for j in range(X.shape[1]):
        col_all = X[:, j]
        col_legit = X_legit[:, j]
        col_fraud = X_fraud[:, j]
        
        # KS Statistic (Distributional divergence: 0 = identical, 1 = completely separated)
        ks_stat, _ = ks_2samp(col_legit, col_fraud)
        ks_scores.append(ks_stat)
        
        # 99th percentile cutoff computed globally
        p99 = np.percentile(col_all, 99)
        in_tail = col_all >= p99
        tail_total = np.sum(in_tail)
        
        if tail_total > 0:
            tail_fraud = np.sum(in_tail & fraud_mask)
            tail_frauds_pct.append(tail_fraud / tail_total)
            tail_legits_count.append(np.sum(in_tail & legit_mask))
            
    # 2. Camouflaged Fraud: How many fraud instances have ALL core features within [p10, p90] of legit?
    # Test on top 5 most discriminative features
    top_5_idx = np.argsort(ks_scores)[::-1][:5]
    
    camouflaged_count = 0
    for i in range(len(X_fraud)):
        sample = X_fraud[i, top_5_idx]
        is_camouflaged = True
        for idx_pos, feat_idx in enumerate(top_5_idx):
            p10 = np.percentile(X_legit[:, feat_idx], 10)
            p90 = np.percentile(X_legit[:, feat_idx], 90)
            if sample[idx_pos] < p10 or sample[idx_pos] > p90:
                is_camouflaged = False
                break
        if is_camouflaged:
            camouflaged_count += 1
            
    camouflaged_pct = camouflaged_count / n_fraud if n_fraud > 0 else 0.0
    
    # 3. Global Distance from Centroid (Mahalanobis-like approximation via standardized Euclidean)
    std_per_col = np.std(X_legit, axis=0) + 1e-6
    mean_per_col = np.mean(X_legit, axis=0)
    
    z_legit = np.abs((X_legit - mean_per_col) / std_per_col)
    z_fraud = np.abs((X_fraud - mean_per_col) / std_per_col)
    
    mean_norm_distance_legit = np.mean(np.linalg.norm(z_legit, axis=1))
    mean_norm_distance_fraud = np.mean(np.linalg.norm(z_fraud, axis=1))
    distance_ratio = mean_norm_distance_fraud / (mean_norm_distance_legit + 1e-6)
    
    # Top 3 most separated features
    top_features = [
        (feature_names[idx], ks_scores[idx]) for idx in np.argsort(ks_scores)[::-1][:3]
    ]

    return {
        "dataset_name": dataset_name,
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "fraud_rate": float(base_rate),
        "mean_ks_stat": float(np.mean(ks_scores)),
        "max_ks_stat": float(np.max(ks_scores)),
        "top_features_by_ks": [(str(name), float(val)) for name, val in top_features],
        "avg_fraud_in_p99_tails": float(np.mean(tail_frauds_pct)),
        "avg_legit_in_p99_tails": float(1.0 - np.mean(tail_frauds_pct)),
        "camouflaged_fraud_pct": float(camouflaged_pct),
        "relative_outlier_distance": float(distance_ratio),
    }
