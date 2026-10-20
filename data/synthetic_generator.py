"""
Synthetic Behavioral Deviation Dataset Generator for Fraud Detection.

Simulates multiple user cohorts with unique historical baselines and generates
both legitimate behavioral variations and characteristic behavioral fraud vectors:
- Account Takeover (ATO): sudden velocity burst, uncharacteristic amount, new device, impossible travel speed.
- Card Testing / Bot Activity: micro-transactions, abnormal velocity, digital goods.
- Bust-Out / Escalation: progressive deviation from historical mean up to limit maxing.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any


def generate_behavioral_data(
    n_users: int = 2500,
    n_transactions: int = 40000,
    fraud_rate: float = 0.015,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates synthetic transaction logs and extracts entity-level behavioral deviation features.
    
    Returns:
        df_raw: Full transaction log.
        df_features: Standardized behavioral deviation feature matrix with target 'is_fraud'.
    """
    rng = np.random.default_rng(random_state)
    
    # 1. Define User Behavioral Baselines
    user_ids = np.arange(1, n_users + 1)
    
    # User spending baselines: log-normal distribution
    user_median_amount = rng.lognormal(mean=3.5, sigma=0.7, size=n_users)  # ~ $15 to $150 typical
    user_std_amount = user_median_amount * rng.uniform(0.2, 0.5, size=n_users)
    
    # User hourly activity patterns: peak hour (0-23)
    user_peak_hour = rng.integers(8, 22, size=n_users)
    
    # Typical daily transaction velocity (e.g. 0.5 to 3.0 txns/day)
    user_daily_velocity = rng.gamma(shape=2.0, scale=0.8, size=n_users)
    
    # Primary user location (lat, lon) within a bounded continent/region
    user_home_lat = rng.uniform(25.0, 48.0, size=n_users)
    user_home_lon = rng.uniform(-120.0, -75.0, size=n_users)
    
    # Primary device pool (1-2 devices per user)
    user_devices = {uid: [f"dev_{uid}_1", f"dev_{uid}_2"] if rng.random() > 0.4 else [f"dev_{uid}_1"] 
                    for uid in user_ids}
    
    # Baseline merchant categories (from a pool of 10)
    user_preferred_cats = {uid: rng.choice(10, size=rng.integers(2, 5), replace=False).tolist()
                           for uid in user_ids}
    
    user_profiles: Dict[int, Dict[str, Any]] = {}
    for i, uid in enumerate(user_ids):
        user_profiles[uid] = {
            "median_amount": user_median_amount[i],
            "std_amount": max(user_std_amount[i], 2.0),
            "peak_hour": user_peak_hour[i],
            "daily_velocity": user_daily_velocity[i],
            "home_lat": user_home_lat[i],
            "home_lon": user_home_lon[i],
            "devices": user_devices[uid],
            "categories": user_preferred_cats[uid],
        }

    # 2. Generate Transactions over a 30-day timeline (in hours: 0 to 720)
    records = []
    
    # Target number of frauds
    n_fraud = int(n_transactions * fraud_rate)
    n_legit = n_transactions - n_fraud
    
    # Assign transactions randomly weighted by user activity
    weights = user_daily_velocity / user_daily_velocity.sum()
    assigned_users_legit = rng.choice(user_ids, size=n_legit, p=weights)
    
    for uid in assigned_users_legit:
        prof = user_profiles[uid]
        
        # Timestamp around user's peak hour with normal noise
        day = rng.uniform(0, 30)
        hour = (prof["peak_hour"] + rng.normal(0, 3.0)) % 24
        ts_hours = day * 24 + hour
        
        # Legitimate amount: log-normal around user median
        amount = max(1.0, rng.normal(prof["median_amount"], prof["std_amount"]))
        
        # Device: 96% from known devices, 4% benign device change (new phone, browser update)
        if rng.random() < 0.96:
            device = rng.choice(prof["devices"])
            is_new_device = 0
        else:
            device = f"dev_{uid}_new"
            is_new_device = 1
            
        # Location: around home (within ~30 km)
        lat = prof["home_lat"] + rng.normal(0, 0.15)
        lon = prof["home_lon"] + rng.normal(0, 0.15)
        
        # Category: 90% preferred, 10% new
        if rng.random() < 0.90:
            cat = rng.choice(prof["categories"])
            is_new_cat = 0
        else:
            cat = rng.integers(0, 10)
            is_new_cat = 1 if cat not in prof["categories"] else 0
            
        records.append({
            "user_id": uid,
            "timestamp_hours": ts_hours,
            "amount": amount,
            "is_new_device": is_new_device,
            "lat": lat,
            "lon": lon,
            "category": cat,
            "is_new_cat": is_new_cat,
            "is_fraud": 0,
            "fraud_type": "none",
        })
        
    # Generate Fraudulent Transactions with behavioral deviations
    fraud_uids = rng.choice(user_ids, size=n_fraud, replace=True)
    for uid in fraud_uids:
        prof = user_profiles[uid]
        day = rng.uniform(5, 30)
        fraud_mode = rng.choice(["ato", "card_testing", "bust_out"], p=[0.45, 0.35, 0.20])
        
        if fraud_mode == "ato":
            # Account Takeover: sudden spike at off-hours, new device, distant location, high amount
            hour = (prof["peak_hour"] + 12 + rng.normal(0, 1.5)) % 24  # opposite of typical hour
            ts_hours = day * 24 + hour
            amount = prof["median_amount"] * rng.uniform(4.0, 12.0)  # 4x to 12x usual
            is_new_device = 1
            # Geo jump: 1000+ km away
            lat = prof["home_lat"] + rng.choice([-1, 1]) * rng.uniform(8.0, 15.0)
            lon = prof["home_lon"] + rng.choice([-1, 1]) * rng.uniform(10.0, 25.0)
            cat = rng.integers(0, 10)
            is_new_cat = 1
            
        elif fraud_mode == "card_testing":
            # Micro-transaction velocity test
            hour = rng.uniform(0, 24)
            ts_hours = day * 24 + hour
            amount = rng.uniform(0.50, 4.00)  # tiny amount
            is_new_device = 1 if rng.random() > 0.3 else 0
            lat = prof["home_lat"] + rng.normal(0, 1.0)
            lon = prof["home_lon"] + rng.normal(0, 1.0)
            cat = 9  # e.g., digital services / subscriptions
            is_new_cat = 1 if 9 not in prof["categories"] else 0
            
        else:  # bust_out
            # Significant escalation on existing or new device
            hour = (prof["peak_hour"] + rng.normal(0, 2.0)) % 24
            ts_hours = day * 24 + hour
            amount = prof["median_amount"] * rng.uniform(6.0, 18.0)
            is_new_device = 1 if rng.random() > 0.5 else 0
            lat = prof["home_lat"] + rng.normal(0, 0.5)
            lon = prof["home_lon"] + rng.normal(0, 0.5)
            cat = rng.integers(0, 10)
            is_new_cat = 1 if cat not in prof["categories"] else 0
            
        records.append({
            "user_id": uid,
            "timestamp_hours": ts_hours,
            "amount": amount,
            "is_new_device": is_new_device,
            "lat": lat,
            "lon": lon,
            "category": cat,
            "is_new_cat": is_new_cat,
            "is_fraud": 1,
            "fraud_type": fraud_mode,
        })
        
    df_raw = pd.DataFrame(records).sort_values("timestamp_hours").reset_index(drop=True)
    
    # 3. Compute Behavioral Deviation Features per Entity
    # We compute rolling / baseline comparison metrics
    features = []
    
    # Pre-group by user for fast rolling lookup
    user_groups = df_raw.groupby("user_id")
    
    for uid, group in user_groups:
        prof = user_profiles[uid]
        med = prof["median_amount"]
        std = prof["std_amount"]
        peak = prof["peak_hour"]
        
        # Differences in time
        time_diff = group["timestamp_hours"].diff().fillna(24.0).clip(lower=0.01)
        
        # Velocity proxy: exponential rolling count in past 1h and past 24h
        # Using vectorized approximations based on time difference
        velocity_1h = 1.0 / (time_diff.clip(lower=0.05))
        velocity_24h = (1.0 / (time_diff.clip(lower=0.5))) * 2.0
        
        # Geo distance from previous transaction (Haversine approx in km)
        dlat = (group["lat"].diff().fillna(0.0)) * 111.0
        dlon = (group["lon"].diff().fillna(0.0)) * 85.0
        geo_dist_km = np.sqrt(dlat**2 + dlon**2)
        travel_speed_kmh = (geo_dist_km / time_diff).clip(upper=1500.0)
        
        # Amount deviations
        amount_ratio = group["amount"] / med
        amount_zscore = (group["amount"] - med) / std
        
        # Temporal deviation (angular / circular distance from peak hour)
        hours = group["timestamp_hours"] % 24
        hour_diff = np.abs(hours - peak)
        circular_hour_diff = np.minimum(hour_diff, 24.0 - hour_diff)
        
        group_feat = pd.DataFrame({
            "amount_ratio_to_median": amount_ratio,
            "amount_zscore": amount_zscore,
            "circular_hour_deviation": circular_hour_diff,
            "travel_speed_kmh": travel_speed_kmh,
            "is_new_device": group["is_new_device"],
            "is_new_cat": group["is_new_cat"],
            "velocity_1h_proxy": velocity_1h,
            "velocity_24h_proxy": velocity_24h,
            "time_since_last_txn_hours": time_diff,
            "is_fraud": group["is_fraud"],
            "fraud_type": group["fraud_type"],
            "user_id": group["user_id"],
        }, index=group.index)
        
        features.append(group_feat)
        
    df_features = pd.concat(features).sort_index()
    return df_raw, df_features


if __name__ == "__main__":
    raw, feat = generate_behavioral_data(n_transactions=10000, fraud_rate=0.02)
    print(f"Generated {len(feat)} rows. Fraud count: {feat['is_fraud'].sum()} ({feat['is_fraud'].mean():.2%})")
    print("\nFeature Summary:")
    print(feat.drop(columns=['fraud_type', 'user_id']).describe().T[['mean', 'std', 'min', '50%', 'max']])
