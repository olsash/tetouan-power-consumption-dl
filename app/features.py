"""Feature engineering used by the deployed app.

This is the same logic as notebook 02, so the model receives inputs
in exactly the same format it was trained on.
"""
import numpy as np
import pandas as pd

WEATHER = ["Temperature", "Humidity", "WindSpeed", "GeneralDiffuseFlows", "DiffuseFlows"]
TARGETS = ["Zone1", "Zone2", "Zone3"]
CYCLICAL = ["time_sin", "time_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend"]
LAG_TYPES = ["lag1d", "lag7d", "mean1d"]


def load_raw(csv_path):
    """Load the original dataset and clean the column names."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.replace(r"\s+", " ", regex=True).str.strip()
    df = df.rename(columns={
        "Wind Speed": "WindSpeed",
        "general diffuse flows": "GeneralDiffuseFlows",
        "diffuse flows": "DiffuseFlows",
        "Zone 1 Power Consumption": "Zone1",
        "Zone 2 Power Consumption": "Zone2",
        "Zone 3 Power Consumption": "Zone3",
    })
    df["DateTime"] = pd.to_datetime(df["DateTime"], format="%m/%d/%Y %H:%M")
    return df.set_index("DateTime").sort_index()


def build_features(df):
    """Add cyclical time features and day-ahead lag features (unscaled)."""
    df = df.copy()
    minute_of_day = df.index.hour * 60 + df.index.minute
    df["time_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
    df["time_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)
    df["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * (df.index.month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df.index.month - 1) / 12)
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    for z in TARGETS:
        df[f"{z}_lag1d"] = df[z].shift(144)
        df[f"{z}_lag7d"] = df[z].shift(1008)
        df[f"{z}_mean1d"] = df[z].shift(144).rolling(144).mean()
    return df.dropna()


def scale_features(df, features, feature_scaler, target_scaler):
    """Scale features exactly like in training (lag features use the target scaler)."""
    out = df[features].copy()
    out[WEATHER + CYCLICAL] = feature_scaler.transform(df[WEATHER + CYCLICAL])
    for t in LAG_TYPES:
        cols = [f"{z}_{t}" for z in TARGETS]
        out[cols] = target_scaler.transform(df[cols].set_axis(TARGETS, axis=1))
    return out.astype("float32")
