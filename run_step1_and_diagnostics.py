"""
Step 1 Benchmark & Data Gap Diagnostics.

1. Performs statistical and geometric diagnosis of Synthetic vs IEEE-CIS datasets
   to identify why unsupervised anomaly detection collapses in real-world data.
2. Evaluates Supervised LightGBM alongside Isolation Forest on identical train/test splits.
3. Generates comparative performance curves and a diagnostic summary.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(__file__))

from data.synthetic_generator import generate_behavioral_data
from data.ieee_cis_loader import fetch_or_load_ieee_sample, prepare_ieee_behavioral_features
from models import IsolationForestDetector, SupervisedLightGBMDetector
from evaluator import evaluate_detector
from study_data_gap import analyze_dataset_geometry


def plot_step1_comparison(
    results_synthetic: list,
    results_ieee: list,
    output_path: str,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Synthetic PR Curves
    ax0 = axes[0]
    for r in results_synthetic:
        p, rec = r["pr_curve"]
        ax0.plot(rec, p, label=f"{r['model_name']} (PR-AUC: {r['pr_auc']:.4f})", lw=2.5)
    ax0.set_title("Synthetic Behavioral Dataset\nPrecision-Recall Curve", fontsize=12, fontweight="bold")
    ax0.set_xlabel("Recall", fontsize=11)
    ax0.set_ylabel("Precision", fontsize=11)
    ax0.legend(loc="upper right", fontsize=10)
    ax0.grid(True, alpha=0.3)
    
    # Right: IEEE-CIS PR Curves
    ax1 = axes[1]
    for r in results_ieee:
        p, rec = r["pr_curve"]
        ax1.plot(rec, p, label=f"{r['model_name']} (PR-AUC: {r['pr_auc']:.4f})", lw=2.5)
    ax1.set_title("IEEE-CIS Real-World Dataset\nPrecision-Recall Curve", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Recall", fontsize=11)
    ax1.set_ylabel("Precision", fontsize=11)
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved Step 1 comparison plot to {output_path}")


def print_table(title: str, results: list):
    print(f"\n=======================================================")
    print(f"  {title}")
    print(f"=======================================================")
    headers = ["Model", "Type", "PR-AUC", "ROC-AUC", "Recall@0.5%", "Recall@1.0%", "Recall@2.0%", "Train(s)", "Latency(ms/1k)"]
    print(f"| {' | '.join(headers)} |")
    print(f"| {' | '.join(['---' for _ in headers])} |")
    for r in results:
        mtype = "Supervised" if "LightGBM" in r["model_name"] else "Unsupervised"
        print(f"| **{r['model_name']}** | {mtype} | {r['pr_auc']:.4f} | {r['roc_auc']:.4f} | {r['recall@0.5%']:.2%} | {r['recall@1.0%']:.2%} | {r['recall@2.0%']:.2%} | {r['train_time_sec']:.2f}s | {r['latency_ms_per_1k']:.1f}ms |")
    print(f"=======================================================\n")


def main():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print(">>> PART 1: DATA GEOMETRY & STATISTICAL GAP DIAGNOSIS")
    print("=" * 70)
    
    # 1. Load Synthetic Data
    _, df_syn = generate_behavioral_data(n_users=2000, n_transactions=30000, fraud_rate=0.018, random_state=42)
    syn_feature_cols = [
        "amount_ratio_to_median", "amount_zscore", "circular_hour_deviation",
        "travel_speed_kmh", "is_new_device", "is_new_cat",
        "velocity_1h_proxy", "velocity_24h_proxy", "time_since_last_txn_hours",
    ]
    X_syn = np.asarray(df_syn[syn_feature_cols].values, dtype=np.float32)
    y_syn = np.asarray(df_syn["is_fraud"].values, dtype=np.int32)
    
    # 2. Load IEEE-CIS Data
    df_ieee_raw = fetch_or_load_ieee_sample(n_rows_approx=35000)
    X_ieee_df, y_ieee_s = prepare_ieee_behavioral_features(df_ieee_raw, max_samples=25000, random_state=42)
    X_ieee = np.asarray(X_ieee_df.values, dtype=np.float32)
    y_ieee = np.asarray(y_ieee_s.values, dtype=np.int32)
    ieee_feature_cols = list(X_ieee_df.columns)
    
    # Run Diagnostics
    diag_syn = analyze_dataset_geometry(X_syn, y_syn, syn_feature_cols, "Synthetic Dataset")
    diag_ieee = analyze_dataset_geometry(X_ieee, y_ieee, ieee_feature_cols, "IEEE-CIS Real-World")
    
    print(f"\n[DIAGNOSTIC COMPARISON]")
    print(f"Metric                                    | Synthetic Data       | IEEE-CIS Real-World")
    print(f"------------------------------------------|----------------------|--------------------")
    print(f"Number of Features                        | {diag_syn['n_features']:<20} | {diag_ieee['n_features']}")
    print(f"Base Fraud Rate                           | {diag_syn['fraud_rate']:<20.2%} | {diag_ieee['fraud_rate']:.2%}")
    print(f"Mean KS Separability (0 to 1)             | {diag_syn['mean_ks_stat']:<20.4f} | {diag_ieee['mean_ks_stat']:.4f}")
    print(f"Max Single Feature KS                     | {diag_syn['max_ks_stat']:<20.4f} | {diag_ieee['max_ks_stat']:.4f}")
    print(f"Fraud inside Top 1% Tails (Geometric Out) | {diag_syn['avg_fraud_in_p99_tails']:<20.2%} | {diag_ieee['avg_fraud_in_p99_tails']:.2%}")
    print(f"Benign Users in Top 1% Tails (False Pos)  | {diag_syn['avg_legit_in_p99_tails']:<20.2%} | {diag_ieee['avg_legit_in_p99_tails']:.2%}")
    print(f"Camouflaged Fraud (% hiding inside IQR)   | {diag_syn['camouflaged_fraud_pct']:<20.2%} | {diag_ieee['camouflaged_fraud_pct']:.2%}")
    print(f"Fraud/Legit Distance Ratio from Center    | {diag_syn['relative_outlier_distance']:<20.2f}x| {diag_ieee['relative_outlier_distance']:.2f}x")
    print(f"Top 3 Separating Features                 | {diag_syn['top_features_by_ks'][0][0][:18]:<20} | {diag_ieee['top_features_by_ks'][0][0][:18]}")
    
    # Save diagnostic summary
    with open(os.path.join(results_dir, "data_gap_diagnostics.json"), "w") as f:
        json.dump({"synthetic": diag_syn, "ieee_cis": diag_ieee}, f, indent=2, default=float)

    print("\n" + "=" * 70)
    print(">>> PART 2: STEP 1 - SUPERVISED LIGHTGBM VS ISOLATION FOREST")
    print("=" * 70)
    
    # Synthetic split
    X_syn_train, X_syn_test, y_syn_train, y_syn_test = train_test_split(
        X_syn, y_syn, test_size=0.40, stratify=y_syn, random_state=42
    )
    
    # IEEE split
    X_ieee_train, X_ieee_test, y_ieee_train, y_ieee_test = train_test_split(
        X_ieee, y_ieee, test_size=0.40, stratify=y_ieee, random_state=42
    )
    
    # Models to compare
    syn_models = [
        IsolationForestDetector(n_estimators=150, contamination=0.02, random_state=42),
        SupervisedLightGBMDetector(n_estimators=150, learning_rate=0.05, random_state=42),
    ]
    
    results_syn = []
    for m in syn_models:
        print(f"Evaluating {m.name} on Synthetic...")
        res = evaluate_detector(m, X_syn_train, X_syn_test, y_syn_test, y_train=y_syn_train)
        results_syn.append(res)
        
    print_table("SYNTHETIC BENCHMARK: SUPERVISED VS UNSUPERVISED", results_syn)
    
    ieee_models = [
        IsolationForestDetector(n_estimators=150, contamination=0.03, random_state=42),
        SupervisedLightGBMDetector(n_estimators=150, learning_rate=0.05, random_state=42),
    ]
    
    results_ieee = []
    for m in ieee_models:
        print(f"Evaluating {m.name} on IEEE-CIS...")
        res = evaluate_detector(m, X_ieee_train, X_ieee_test, y_ieee_test, y_train=y_ieee_train)
        results_ieee.append(res)
        
    print_table("IEEE-CIS REAL-WORLD: SUPERVISED VS UNSUPERVISED", results_ieee)
    
    # Plot Step 1 comparison curves
    plot_step1_comparison(
        results_syn,
        results_ieee,
        os.path.join(results_dir, "step1_supervised_vs_iforest.png"),
    )
    
    # Save combined results
    step1_summary = {
        "synthetic": [
            {k: v for k, v in r.items() if k not in ["scores", "pr_curve"]} for r in results_syn
        ],
        "ieee_cis": [
            {k: v for k, v in r.items() if k not in ["scores", "pr_curve"]} for r in results_ieee
        ],
    }
    with open(os.path.join(results_dir, "step1_benchmark_summary.json"), "w") as f:
        json.dump(step1_summary, f, indent=2, default=float)
        
    print("\nStep 1 & Diagnostics complete!")


if __name__ == "__main__":
    main()
