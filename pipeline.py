import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler  # Fix 1: Missing import added


MODEL_PATH = "isolation_forest.pkl"
SCALER_PATH = "scaler.pkl"


def create_features(logs_df, users_df):

    df = logs_df.merge(
        users_df,
        on="user_id",
        how="left"
    )

    user_features = pd.DataFrame(
        index=df["user_id"].unique()
    )

    user_features.index.name = "user_id"

    user_features["access_count"] = (
        df.groupby("user_id").size()
    )

    user_features["high_sensitivity"] = (
        df.groupby("user_id")["resource_sensitivity"]
        .apply(lambda x: (x == "high").sum())
    )

    user_features["failed_logins"] = (
        df.groupby("user_id")["status"]
        .apply(lambda x: (x == "failure").sum())
    )

    user_features["export_count"] = (
        df.groupby("user_id")["action"]
        .apply(lambda x: (x == "export_data").sum())
    )

    user_features["night_access"] = (
        df.groupby("user_id")["time_classification"]
        .apply(lambda x: (x == "night").sum())
    )

    user_features["weekend_access"] = (
        df.groupby("user_id")["time_classification"]
        .apply(lambda x: (x == "weekend").sum())
    )

    user_features["unique_resources"] = (
        df.groupby("user_id")["resource"]
        .nunique()
    )

    user_features = user_features.reset_index()

    profile_features = df.groupby("user_id").agg({
        "department": "first",
        "job_title": "first",
        "privilege_level": "first",
        "days_inactive": "first"
    }).reset_index()

    user_features = user_features.merge(
        profile_features,
        on="user_id",
        how="left"
    )

    privilege_map = {
        "user": 1,
        "power-user": 2,
        "admin": 3,
        "service-account": 4,
        "executive": 2,
        "contractor": 1
    }

    user_features["privilege_score"] = (
        user_features["privilege_level"]
        .astype(str)
        .str.lower()
        .map(privilege_map)
        .fillna(1)
    )

    # Fix 2: Guard against division by zero before computing resource_diversity
    user_features["resource_diversity"] = (
        user_features["unique_resources"]
        /
        user_features["access_count"].replace(0, 1)
    )

    user_typical_time = (
        df.groupby("user_id")["time_classification"]
        .agg(lambda x: x.mode()[0])
    )

    df["typical_time"] = (
        df["user_id"].map(user_typical_time)
    )

    df["time_deviation_flag"] = (
        df["time_classification"]
        !=
        df["typical_time"]
    ).astype(int)

    time_deviation = (
        df.groupby("user_id")["time_deviation_flag"]
        .sum()
    )

    user_features["time_deviation"] = (
        user_features["user_id"]
        .map(time_deviation)
        .fillna(0)
    )

    return user_features


def run_detection(user_features):

    X = user_features[
        [
            "access_count",
            "high_sensitivity",
            "failed_logins",
            "export_count",
            "night_access",
            "weekend_access",
            "unique_resources",
            "days_inactive",
            "privilege_score",
            "time_deviation",
            "resource_diversity"  # Fix 3: Added missing feature to model input
        ]
    ]

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)

    scaler = joblib.load(SCALER_PATH)

    model = joblib.load(MODEL_PATH)

    X_scaled = scaler.transform(X)

    user_features["anomaly"] = (
        model.predict(X_scaled)
    )

    user_features["is_anomaly"] = (
        user_features["anomaly"] == -1
    ).astype(int)

    user_features["anomaly_score"] = (
        -model.decision_function(X_scaled)
    )

    return user_features


def get_risk_level(score):

    if score >= 75:
        return "CRITICAL"

    elif score >= 55:
        return "HIGH"

    elif score >= 35:
        return "MEDIUM"

    return "LOW"


def generate_reason(row):

    reasons = []

    if row["high_sensitivity"] >= 8:
        reasons.append(
            "High sensitive resource access"
        )

    if row["export_count"] >= 4:
        reasons.append(
            "Large export activity"
        )

    if row["failed_logins"] >= 2:
        reasons.append(
            "Multiple failed logins"
        )

    if row["night_access"] >= 4:
        reasons.append(
            "Frequent night access"
        )

    if row["time_deviation"] >= 2:
        reasons.append(
            "Access outside normal schedule"
        )

    if row["unique_resources"] >= 7:
        reasons.append(
            "Accessed many different resources"
        )

    if row["is_anomaly"] == 1:
        reasons.append(
            "ML anomaly detected"
        )

    if len(reasons) == 0:
        return "Normal behavior"

    return ", ".join(reasons)


def calculate_risk(user_features):

    user_features["risk_score_raw"] = (
          user_features["high_sensitivity"] * 4
        + user_features["failed_logins"] * 10
        + user_features["export_count"] * 6
        + user_features["night_access"] * 3
        + user_features["weekend_access"] * 2
        + user_features["unique_resources"] * 2
        + user_features["time_deviation"] * 4
        # Fix 4: privilege_score only contributes when combined with anomaly,
        # to avoid penalising users purely for their role (e.g. admins).
        + user_features["privilege_score"] * user_features["is_anomaly"] * 3
        + user_features["is_anomaly"] * 25
    )

    risk_scaler = MinMaxScaler(
        feature_range=(0, 100)
    )

    user_features["risk_score"] = (
        risk_scaler.fit_transform(
            user_features[["risk_score_raw"]]
        )
    )

    user_features["risk_level"] = (
        user_features["risk_score"]
        .apply(get_risk_level)
    )

    user_features["reason"] = (
        user_features.apply(
            generate_reason,
            axis=1
        )
    )

    return user_features


def run_pipeline(logs_df, users_df):

    user_features = create_features(
        logs_df,
        users_df
    )

    user_features = run_detection(
        user_features
    )

    user_features = calculate_risk(
        user_features
    )

    return user_features[
        [
            "user_id",
            "department",
            "job_title",
            "privilege_level",
            "risk_score",
            "risk_level",
            "reason",
            "anomaly_score",
            "access_count",
            "high_sensitivity",
            "failed_logins",
            "export_count"
        ]
    ].sort_values(
        "risk_score",
        ascending=False
    )
