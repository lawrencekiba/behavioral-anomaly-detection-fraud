"""
Unified Evaluation Suite for Behavioral Anomaly Detection in Fraud.

Computes:
- PR-AUC (Average Precision): Key metric under extreme imbalance.
- ROC-AUC: Global ranking discriminability.
- Recall @ Top k% (e.g., k=0.5%, 1.0%, 2.0% alert capacity).
- Precision @ Top k%.
- Training duration (s) and Inference latency (ms / 1k samples).
- Precision-Recall comparison plots.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from models.base import BaseAnomalyDetector


def evaluate_detector(
    detector: BaseAnomalyDetector,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alert_budgets: List[float] = [0.005, 0.01, 0.02],  # 0.5%, 1%, 2%
) -> Dict[str, Any]:
    """
    Fits and rigorously evaluates an anomaly detector on test data.
    """
    # 1. Measure Training Time
    t0 = time.perf_counter()
    detector.fit(X_train)
    train_time_sec = time.perf_counter() - t0

    # 2. Measure Inference Latency
    t1 = time.perf_counter()
    scores = detector.score_samples(X_test)
    inference_time_sec = time.perf_counter() - t1
    latency_ms_per_1k = (inference_time_sec / len(X_test)) * 1000.0 * 1000.0

    # 3. Global Discriminative Metrics
    pr_auc = average_precision_score(y_test, scores)
    roc_auc = roc_auc_score(y_test, scores)

    # 4. Operational Alert Budget Metrics (Top-k%)
    alert_metrics = {}
    total_positives = np.sum(y_test)
    n_samples = len(y_test)

    for k in alert_budgets:
        cutoff_rank = max(1, int(n_samples * k))
        threshold = np.partition(scores, -cutoff_rank)[-cutoff_rank]
        y_pred = (scores >= threshold).astype(int)
        
        # True Positives in top k
        tp = np.sum((y_pred == 1) & (y_test == 1))
        flagged = np.sum(y_pred == 1)
        
        recall_at_k = tp / total_positives if total_positives > 0 else 0.0
        precision_at_k = tp / flagged if flagged > 0 else 0.0
        
        k_pct_label = f"{k*100:.1f}%"
        alert_metrics[f"recall@{k_pct_label}"] = recall_at_k
        alert_metrics[f"precision@{k_pct_label}"] = precision_at_k

    # Compute PR curve data for plotting
    precision, recall, thresholds = precision_recall_curve(y_test, scores)

    return {
        "model_name": detector.name,
        "train_time_sec": train_time_sec,
        "latency_ms_per_1k": latency_ms_per_1k,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        **alert_metrics,
        "scores": scores,
        "pr_curve": (precision, recall),
    }


def generate_benchmark_plots(
    results: List[Dict[str, Any]],
    y_test: np.ndarray,
    dataset_title: str,
    output_path: str,
):
    """
    Generates comparison plots:
    1. Precision-Recall Curves for all candidate algorithms.
    2. Operational Trade-off: Recall @ Alert Budget (0.5%, 1%, 2%).
    3. Latency vs. PR-AUC Scatter.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    palette = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"]

    # Plot 1: Precision-Recall Curve
    ax_pr = axes[0]
    base_rate = np.mean(y_test)
    ax_pr.axhline(base_rate, color="gray", linestyle="--", label=f"Random ({base_rate:.2%})")
    
    for i, res in enumerate(results):
        precision, recall = res["pr_curve"]
        color = palette[i % len(palette)]
        ax_pr.plot(recall, precision, label=f"{res['model_name']} (PR-AUC: {res['pr_auc']:.3f})", color=color, lw=2)
        
    ax_pr.set_title(f"Precision-Recall Curve\n({dataset_title})", fontsize=12, fontweight="bold")
    ax_pr.set_xlabel("Recall (Detection Rate)", fontsize=11)
    ax_pr.set_ylabel("Precision (Alert Accuracy)", fontsize=11)
    ax_pr.legend(loc="upper right", fontsize=9)
    ax_pr.grid(True, alpha=0.3)

    # Plot 2: Recall @ Operational Alert Capacity (0.5%, 1%, 2%)
    ax_cap = axes[1]
    budgets = ["0.5%", "1.0%", "2.0%"]
    x = np.arange(len(budgets))
    width = 0.2

    for i, res in enumerate(results):
        recalls = [res.get(f"recall@{b}", 0.0) for b in budgets]
        color = palette[i % len(palette)]
        ax_cap.bar(x + i * width, recalls, width, label=res["model_name"], color=color, alpha=0.85)

    ax_cap.set_title("Detection Rate @ Operational Alert Capacity", fontsize=12, fontweight="bold")
    ax_cap.set_xlabel("Fraud Ops Alert Budget (% of traffic reviewed)", fontsize=11)
    ax_cap.set_ylabel("Recall of True Fraud Caught", fontsize=11)
    ax_cap.set_xticks(x + width * (len(results) - 1) / 2)
    ax_cap.set_xticklabels(budgets)
    ax_cap.legend(loc="upper left", fontsize=9)
    ax_cap.grid(True, alpha=0.3)

    # Plot 3: PR-AUC vs. Inference Latency (Efficiency Frontier)
    ax_eff = axes[2]
    for i, res in enumerate(results):
        color = palette[i % len(palette)]
        ax_eff.scatter(
            res["latency_ms_per_1k"],
            res["pr_auc"],
            color=color,
            s=120,
            label=res["model_name"],
            edgecolor="black",
            zorder=5,
        )
        ax_eff.annotate(
            res["model_name"],
            (res["latency_ms_per_1k"], res["pr_auc"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

    ax_eff.set_title("Detection Quality vs. Inference Latency", fontsize=12, fontweight="bold")
    ax_eff.set_xlabel("Latency (ms per 1,000 txns) [Log Scale]", fontsize=11)
    ax_eff.set_ylabel("PR-AUC (Higher is Better)", fontsize=11)
    ax_eff.set_xscale("log")
    ax_eff.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved benchmark figure to {output_path}")
