"""
Master Hybrid Benchmark: Supervised + Unsupervised Anomaly Detection for Fraud.

Evaluates a 4-way ablation across both Synthetic and IEEE-CIS datasets:
1. Pure Unsupervised (Isolation Forest)
2. Pure Supervised (LightGBM on Raw Features)
3. Hybrid Model A (LightGBM + Isolation Forest Score)
4. Hybrid Model B (LightGBM + Isolation Forest + Autoencoder Reconstruction MSE)

Also measures:
- Permutation feature importance on test set (PR-AUC impact).
- Precision-Recall curves.
- Operational Alert Budget Metrics (Recall @ 0.5%, 1.0%, 2.0%).
- Inference Latency per 1k transactions.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

sys.path.append(os.path.dirname(__file__))

from data.synthetic_generator import generate_behavioral_data
from data.ieee_cis_loader import fetch_or_load_ieee_sample, prepare_ieee_behavioral_features
from models import (
    IsolationForestDetector,
    SupervisedLightGBMDetector,
    HybridFraudDetector,
)
from evaluator import evaluate_detector


def print_comparison_table(title: str, results: list):
    print(f"\n=========================================================================================")
    print(f"  {title}")
    print(f"=========================================================================================")
    headers = ["Model Architecture", "Paradigm", "PR-AUC", "ROC-AUC", "Recall@0.5%", "Recall@1.0%", "Recall@2.0%", "Latency(ms/1k)"]
    print(f"| {' | '.join(headers)} |")
    print(f"| {' | '.join(['---' for _ in headers])} |")
    for r in results:
        mname = r["model_name"]
        if "Isolation Forest" == mname:
            paradigm = "Unsupervised"
        elif "Supervised LightGBM" in mname:
            paradigm = "Supervised"
        else:
            paradigm = "Hybrid"
        print(f"| **{mname}** | {paradigm} | {r['pr_auc']:.4f} | {r['roc_auc']:.4f} | {r['recall@0.5%']:.2%} | {r['recall@1.0%']:.2%} | {r['recall@2.0%']:.2%} | {r['latency_ms_per_1k']:.1f}ms |")
    print(f"=========================================================================================\n")


def plot_hybrid_benchmark(
    results_synthetic: list,
    results_ieee: list,
    output_path: str,
):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    palette = ["#999999", "#2b5c8f", "#e66101", "#2ca02c"]

    # 1. Synthetic PR Curves
    ax0 = axes[0]
    for i, r in enumerate(results_synthetic):
        p, rec = r["pr_curve"]
        ax0.plot(rec, p, label=f"{r['model_name']} (PR-AUC: {r['pr_auc']:.4f})", lw=2.2, color=palette[i % len(palette)])
    ax0.set_title("Synthetic Behavioral Dataset\n4-Way Paradigm Comparison", fontsize=13, fontweight="bold")
    ax0.set_xlabel("Recall (Fraud Detection Rate)", fontsize=11)
    ax0.set_ylabel("Precision (True Fraud / Flagged)", fontsize=11)
    ax0.legend(loc="upper right", fontsize=9.5)
    ax0.grid(True, alpha=0.3)

    # 2. IEEE-CIS PR Curves
    ax1 = axes[1]
    for i, r in enumerate(results_ieee):
        p, rec = r["pr_curve"]
        ax1.plot(rec, p, label=f"{r['model_name']} (PR-AUC: {r['pr_auc']:.4f})", lw=2.2, color=palette[i % len(palette)])
    ax1.set_title("IEEE-CIS Real-World Dataset\n4-Way Paradigm Comparison", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Recall (Fraud Detection Rate)", fontsize=11)
    ax1.set_ylabel("Precision (True Fraud / Flagged)", fontsize=11)
    ax1.legend(loc="upper right", fontsize=9.5)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved hybrid benchmark plots to {output_path}")


def analyze_hybrid_feature_importance(
    hybrid_model: HybridFraudDetector,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    base_feature_names: list,
) -> pd.DataFrame:
    """
    Evaluates how much the supervised tree actually depends on the engineered anomaly scores.
    """
    # Build enriched test matrix manually for permutation evaluation
    anomaly_feats_train = hybrid_model._extract_anomaly_features(X_train, fit=False)
    X_tr_hybrid = np.hstack([X_train, anomaly_feats_train])
    
    anomaly_feats_test = hybrid_model._extract_anomaly_features(X_test, fit=False)
    X_te_hybrid = np.hstack([X_test, anomaly_feats_test])
    
    extra_names = []
    if hybrid_model.use_iforest:
        extra_names.append("iForest_Score")
    if hybrid_model.use_autoencoder:
        extra_names.append("Autoencoder_MSE")
    all_names = list(base_feature_names) + extra_names
    
    # Train a bare classifier on the precomputed hybrid matrix to inspect permutation importance
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(class_weight="balanced", max_iter=150, random_state=42)
    clf.fit(X_tr_hybrid, y_train)
    
    # Permutation importance on test set scored by average precision
    perm = permutation_importance(clf, X_te_hybrid, y_test, scoring="average_precision", n_repeats=5, random_state=42)
    
    df_imp = pd.DataFrame({
        "feature": all_names,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    
    return df_imp


def run_experiment(dataset_name: str, X: np.ndarray, y: np.ndarray, feature_names: list):
    print(f"\n>>> RUNNING ABLATION ON: {dataset_name.upper()} ({len(X)} samples, {X.shape[1]} features, {np.mean(y):.2%} fraud)")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=42
    )

    models = [
        # 1. Pure Unsupervised
        IsolationForestDetector(n_estimators=100, n_jobs=1, random_state=42),
        # 2. Pure Supervised
        SupervisedLightGBMDetector(n_estimators=150, learning_rate=0.05, random_state=42),
        # 3. Hybrid: LightGBM + iForest
        HybridFraudDetector(use_iforest=True, use_autoencoder=False, lgbm_n_estimators=150, random_state=42),
        # 4. Hybrid: LightGBM + iForest + Autoencoder
        HybridFraudDetector(use_iforest=True, use_autoencoder=True, lgbm_n_estimators=150, ae_epochs=20, random_state=42),
    ]

    results = []
    full_hybrid = None

    for m in models:
        print(f"  Training & Evaluating: {m.name}...")
        res = evaluate_detector(m, X_train, X_test, y_test, y_train=y_train)
        results.append(res)
        if m.name == "Hybrid (LightGBM + iForest + AE)":
            full_hybrid = m

    print_comparison_table(f"ABLATION RESULTS: {dataset_name.upper()}", results)

    # Analyze feature importances of the hybrid model
    if full_hybrid is not None:
        print(f"  Computing permutation importance of anomaly features on {dataset_name}...")
        df_imp = analyze_hybrid_feature_importance(
            full_hybrid, X_train, y_train, X_test, y_test, feature_names
        )
        print(f"\n  Top 7 Most Important Features (Test PR-AUC Impact):")
        for rank, row in df_imp.head(7).iterrows():
            marker = " *** ANOMALY FEATURE ***" if ("iForest" in row["feature"] or "Autoencoder" in row["feature"]) else ""
            print(f"    #{rank+1}: {row['feature']:<26} (Impact: +{row['importance_mean']:.4f} ± {row['importance_std']:.4f}){marker}")
            
        # Report anomaly features ranks
        for anom in ["iForest_Score", "Autoencoder_MSE"]:
            match = df_imp[df_imp["feature"] == anom]
            if not match.empty:
                idx = match.index[0]
                val = match["importance_mean"].values[0]
                print(f"    -> Anomaly Feature [{anom}] Ranked #{idx+1} out of {len(df_imp)} features (Impact: {val:+.4f})")

    return results


def main():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    # 1. Synthetic Dataset
    _, df_syn = generate_behavioral_data(n_users=2000, n_transactions=30000, fraud_rate=0.018, random_state=42)
    syn_feature_cols = [
        "amount_ratio_to_median", "amount_zscore", "circular_hour_deviation",
        "travel_speed_kmh", "is_new_device", "is_new_cat",
        "velocity_1h_proxy", "velocity_24h_proxy", "time_since_last_txn_hours",
    ]
    X_syn = np.asarray(df_syn[syn_feature_cols].values, dtype=np.float32)
    y_syn = np.asarray(df_syn["is_fraud"].values, dtype=np.int32)
    syn_results = run_experiment("Synthetic Behavioral Dataset", X_syn, y_syn, syn_feature_cols)

    # 2. IEEE-CIS Real-World Dataset
    df_ieee_raw = fetch_or_load_ieee_sample(n_rows_approx=35000)
    X_ieee_df, y_ieee_s = prepare_ieee_behavioral_features(df_ieee_raw, max_samples=25000, random_state=42)
    X_ieee = np.asarray(X_ieee_df.values, dtype=np.float32)
    y_ieee = np.asarray(y_ieee_s.values, dtype=np.int32)
    ieee_feature_cols = list(X_ieee_df.columns)
    ieee_results = run_experiment("IEEE-CIS Real-World Dataset", X_ieee, y_ieee, ieee_feature_cols)

    # 3. Plots
    plot_path = os.path.join(results_dir, "hybrid_benchmark.png")
    plot_hybrid_benchmark(syn_results, ieee_results, plot_path)

    # 4. JSON Summary
    def clean_res(res_list):
        return [
            {k: v for k, v in r.items() if k not in ["scores", "pr_curve"]}
            for r in res_list
        ]

    summary = {
        "synthetic": clean_res(syn_results),
        "ieee_cis": clean_res(ieee_results),
    }
    summary_path = os.path.join(results_dir, "hybrid_benchmark_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print(f"\nAll Hybrid Benchmarks Complete! Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
