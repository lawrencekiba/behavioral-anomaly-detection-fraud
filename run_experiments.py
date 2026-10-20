"""
Master Experiment Runner: Behavioral Anomaly Detection for Fraud.

Runs comparative benchmarks of:
1. Isolation Forest
2. Local Outlier Factor (LOF)
3. One-Class SVM (OC-SVM)
4. Deep Tabular Autoencoder (AE)

Across two distinct datasets:
- Experiment 1: Synthetic Multi-Cohort Behavioral Deviation Dataset
- Experiment 2: IEEE-CIS Real-World Transaction & Behavioral Velocity Dataset
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add local path
sys.path.append(os.path.dirname(__file__))

from data.synthetic_generator import generate_behavioral_data
from data.ieee_cis_loader import fetch_or_load_ieee_sample, prepare_ieee_behavioral_features
from models import (
    IsolationForestDetector,
    LocalOutlierFactorDetector,
    OneClassSVMDetector,
    AutoencoderDetector,
)
from evaluator import evaluate_detector, generate_benchmark_plots


def print_results_table(dataset_name: str, results: list):
    """Prints a clean ASCII markdown comparison table."""
    print(f"\n=======================================================")
    print(f"  BENCHMARK RESULTS: {dataset_name.upper()}")
    print(f"=======================================================")
    headers = [
        "Model",
        "PR-AUC",
        "ROC-AUC",
        "Recall@0.5%",
        "Recall@1.0%",
        "Recall@2.0%",
        "Train (s)",
        "Latency (ms/1k)",
    ]
    header_line = f"| {' | '.join(headers)} |"
    sep_line = f"| {' | '.join(['---' for _ in headers])} |"
    print(header_line)
    print(sep_line)
    
    for r in results:
        row = [
            f"**{r['model_name']}**",
            f"{r['pr_auc']:.4f}",
            f"{r['roc_auc']:.4f}",
            f"{r['recall@0.5%']:.2%}",
            f"{r['recall@1.0%']:.2%}",
            f"{r['recall@2.0%']:.2%}",
            f"{r['train_time_sec']:.2f}s",
            f"{r['latency_ms_per_1k']:.1f}ms",
        ]
        print(f"| {' | '.join(row)} |")
    print(f"=======================================================\n")


def run_synthetic_experiment(results_dir: str):
    print("\n" + "=" * 60)
    print(">>> RUNNING EXPERIMENT 1: SYNTHETIC BEHAVIORAL DEVIATION")
    print("=" * 60)
    
    # 1. Generate data
    _, df_features = generate_behavioral_data(
        n_users=2000,
        n_transactions=30000,
        fraud_rate=0.018,
        random_state=42,
    )
    
    feature_cols = [
        "amount_ratio_to_median",
        "amount_zscore",
        "circular_hour_deviation",
        "travel_speed_kmh",
        "is_new_device",
        "is_new_cat",
        "velocity_1h_proxy",
        "velocity_24h_proxy",
        "time_since_last_txn_hours",
    ]
    
    X = np.asarray(df_features[feature_cols].values, dtype=np.float32)
    y = np.asarray(df_features["is_fraud"].values, dtype=np.int32)
    fraud_types = np.asarray(df_features["fraud_type"].tolist())
    
    # Stratified split: 60% train, 40% test
    X_train, X_test, y_train, y_test, ft_train, ft_test = train_test_split(
        X, y, fraud_types, test_size=0.40, stratify=y, random_state=42
    )
    
    print(f"Synthetic dataset: {len(X_train)} train samples, {len(X_test)} test samples.")
    print(f"Test positive fraud cases: {np.sum(y_test)} ({np.mean(y_test):.2%})")
    
    # Initialize models
    models = [
        IsolationForestDetector(n_estimators=150, contamination=0.02, random_state=42),
        LocalOutlierFactorDetector(n_neighbors=30, contamination=0.02),
        OneClassSVMDetector(kernel="rbf", nu=0.02),
        AutoencoderDetector(latent_dim=4, hidden_dim=24, epochs=25, batch_size=128, random_state=42),
    ]
    
    results = []
    trained_ae = None
    
    for model in models:
        print(f"Evaluating {model.name}...")
        res = evaluate_detector(model, X_train, X_test, y_test)
        results.append(res)
        if isinstance(model, AutoencoderDetector):
            trained_ae = model
            
    print_results_table("Synthetic Behavioral Deviation", results)
    
    # Generate visual figure
    plot_path = os.path.join(results_dir, "synthetic_benchmark.png")
    generate_benchmark_plots(results, y_test, "Synthetic Behavioral Deviation", plot_path)
    
    # Deep Dive: Autoencoder Feature Attribution across Fraud Types
    if trained_ae is not None:
        print("\n--- AUTOENCODER ROOT-CAUSE FEATURE ATTRIBUTION ---")
        ae_errors = trained_ae.explain_sample_errors(X_test)
        for fmode in ["ato", "card_testing", "bust_out"]:
            mask = (ft_test == fmode)
            if np.sum(mask) > 0:
                mean_feat_err = np.mean(ae_errors[mask], axis=0)
                # Normalize to percentage of total error
                pct_contrib = (mean_feat_err / np.sum(mean_feat_err)) * 100
                top_3_idx = np.argsort(pct_contrib)[::-1][:3]
                top_drivers = ", ".join([f"{feature_cols[idx]} ({pct_contrib[idx]:.1f}%)" for idx in top_3_idx])
                print(f"Archetype [{fmode.upper()}] top error drivers: {top_drivers}")
                
    return results


def run_ieee_cis_experiment(results_dir: str):
    print("\n" + "=" * 60)
    print(">>> RUNNING EXPERIMENT 2: IEEE-CIS REAL-WORLD BEHAVIORAL BENCHMARK")
    print("=" * 60)
    
    df_raw = fetch_or_load_ieee_sample(n_rows_approx=35000)
    X_df, y_s = prepare_ieee_behavioral_features(df_raw, max_samples=25000, random_state=42)
    
    X = np.asarray(X_df.values, dtype=np.float32)
    y = np.asarray(y_s.values, dtype=np.int32)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=42
    )
    
    print(f"IEEE-CIS dataset: {len(X_train)} train, {len(X_test)} test, {X.shape[1]} behavioral features.")
    print(f"Test positive fraud cases: {np.sum(y_test)} ({np.mean(y_test):.2%})")
    
    # Initialize models
    models = [
        IsolationForestDetector(n_estimators=150, contamination=0.03, random_state=42),
        LocalOutlierFactorDetector(n_neighbors=30, contamination=0.03),
        OneClassSVMDetector(kernel="rbf", nu=0.03),
        AutoencoderDetector(latent_dim=8, hidden_dim=48, epochs=25, batch_size=128, random_state=42),
    ]
    
    results = []
    for model in models:
        print(f"Evaluating {model.name}...")
        res = evaluate_detector(model, X_train, X_test, y_test)
        results.append(res)
        
    print_results_table("IEEE-CIS Real-World Fraud", results)
    
    plot_path = os.path.join(results_dir, "ieee_cis_benchmark.png")
    generate_benchmark_plots(results, y_test, "IEEE-CIS Real-World Benchmark", plot_path)
    
    return results


def main():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    syn_results = run_synthetic_experiment(results_dir)
    ieee_results = run_ieee_cis_experiment(results_dir)
    
    # Serialize results to clean JSON report
    def clean_res(res_list):
        return [
            {k: v for k, v in r.items() if k not in ["scores", "pr_curve"]}
            for r in res_list
        ]
        
    summary = {
        "synthetic": clean_res(syn_results),
        "ieee_cis": clean_res(ieee_results),
    }
    
    summary_path = os.path.join(results_dir, "benchmark_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"\nAll experiments complete! Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
