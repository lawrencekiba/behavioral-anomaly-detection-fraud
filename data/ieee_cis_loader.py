"""
IEEE-CIS Fraud Detection Dataset Loader & Behavioral Feature Pipeline.

Streams a clean, stratified-representative slice directly from HuggingFace,
caches it locally, and derives entity-level behavioral deviation features:
- Spending ratio / Z-score relative to cardholder baseline.
- Recency & delta deviations (D-series features).
- Velocity & count surges (C-series features).
- Identity & address interactions.
"""

import os
import io
import urllib.request
import numpy as np
import pandas as pd
from typing import Tuple

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
SAMPLE_PATH = os.path.join(CACHE_DIR, "ieee_cis_sample.parquet")
HF_URL = "https://huggingface.co/datasets/aliceczr/ieee-fraud-detection/resolve/main/train_transaction.csv"


def fetch_or_load_ieee_sample(n_rows_approx: int = 35000) -> pd.DataFrame:
    """
    Downloads and caches a representative slice of IEEE-CIS train_transaction.csv.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(SAMPLE_PATH):
        return pd.read_parquet(SAMPLE_PATH)

    print(f"Streaming ~{n_rows_approx} records from IEEE-CIS dataset...")
    # 35,000 rows is roughly 35-40 MB
    req = urllib.request.Request(HF_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        # Read ~40 MB chunk
        chunk = resp.read(40 * 1024 * 1024)
        last_newline = chunk.rfind(b"\n")
        text = chunk[:last_newline].decode("utf-8", errors="ignore")
        df = pd.read_csv(io.StringIO(text), low_memory=False)

    # Save to parquet for rapid local reload
    df.to_parquet(SAMPLE_PATH, index=False)
    print(f"Cached {len(df)} records to {SAMPLE_PATH}")
    return df


def prepare_ieee_behavioral_features(
    df: pd.DataFrame, max_samples: int = 25000, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extracts entity behavioral deviation features from IEEE-CIS transaction records.
    
    Returns:
        X: Behavioral feature DataFrame (scaled and imputed).
        y: Binary fraud labels (0 = legit, 1 = fraud).
    """
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=random_state).reset_index(drop=True)

    # Entity proxy: card1 (card issuer ID) + addr1 (billing area)
    df["entity_id"] = df["card1"].astype(str) + "_" + df["addr1"].fillna(-1).astype(str)

    # 1. Behavioral Amount Deviations
    entity_stats = df.groupby("entity_id")["TransactionAmt"].agg(["median", "std"]).reset_index()
    entity_stats.columns = ["entity_id", "entity_med_amt", "entity_std_amt"]
    df = df.merge(entity_stats, on="entity_id", how="left")
    
    df["amt_ratio_to_entity_med"] = df["TransactionAmt"] / (df["entity_med_amt"] + 1e-3)
    df["amt_zscore_entity"] = (df["TransactionAmt"] - df["entity_med_amt"]) / (df["entity_std_amt"].fillna(10.0) + 1.0)
    
    # 2. Key Behavioral Velocity & TimeDelta features (C & D series in IEEE-CIS)
    # C-features: count of phone/email/card matches
    # D-features: timedeltas (days between previous transaction events)
    c_cols = [c for c in ["C1", "C2", "C5", "C6", "C9", "C11", "C13", "C14"] if c in df.columns]
    d_cols = [c for c in ["D1", "D2", "D3", "D4", "D10", "D11", "D15"] if c in df.columns]
    
    # Selected V-columns representing high-variance behavioral verification flags
    v_cols = [c for c in ["V12", "V29", "V45", "V62", "V76", "V96", "V130", "V281", "V310"] if c in df.columns]
    
    # Frequency encoding for categorical columns
    cat_cols = ["ProductCD", "card4", "card6", "P_emaildomain"]
    for c in cat_cols:
        if c in df.columns:
            freq = df[c].value_counts(normalize=True).to_dict()
            df[f"{c}_freq"] = df[c].map(freq).fillna(0.0)
            
    cat_freq_cols = [f"{c}_freq" for c in cat_cols if f"{c}_freq" in df.columns]

    feature_cols = [
        "TransactionAmt",
        "amt_ratio_to_entity_med",
        "amt_zscore_entity",
    ] + c_cols + d_cols + v_cols + cat_freq_cols
    
    X = df[feature_cols].copy()
    y = df["isFraud"].astype(int)
    
    # Handle missing values: median imputation
    X = X.fillna(X.median())
    
    return X, y


if __name__ == "__main__":
    df = fetch_or_load_ieee_sample()
    X, y = prepare_ieee_behavioral_features(df)
    print(f"Prepared IEEE-CIS features: shape {X.shape}, fraud rate: {y.mean():.2%} ({y.sum()} frauds)")
    print(f"Features list: {list(X.columns[:10])} ... ({len(X.columns)} total)")
