"""Flask web app for day-ahead power consumption forecasting in Tetouan city.

Run from the project root:
    python app/app.py
Then open http://127.0.0.1:5000 in the browser.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from features import TARGETS, WEATHER, build_features, load_raw, scale_features

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_FILE = BASE_DIR / "data" / "Tetuan_City_power_consumption.csv"

# ---------------------------------------------------------------- load once at startup
with open(MODEL_DIR / "best_model.json") as f:
    INFO = json.load(f)

MODEL_NAME = INFO["best_model"]
FEATURES = INFO["features"]
WINDOW = INFO["window"]
IS_SEQUENCE = INFO["is_sequence_model"]

feature_scaler = joblib.load(MODEL_DIR / "feature_scaler.pkl")
target_scaler = joblib.load(MODEL_DIR / "target_scaler.pkl")

model_path = MODEL_DIR / INFO["model_file"]
if model_path.suffix == ".keras":
    from tensorflow import keras
    model = keras.models.load_model(model_path)
else:
    model = joblib.load(model_path)

# Historical data: provides the lag features (yesterday / last week) for any forecast date
history = build_features(load_raw(DATA_FILE))
history_scaled = scale_features(history, FEATURES, feature_scaler, target_scaler)

# The first day after dropping NaN cannot be forecast by sequence models (needs 6 h of history)
MIN_DATE = (history.index.min() + pd.Timedelta(days=1)).date()
MAX_DATE = history.index.max().date()

print(f"Loaded model: {MODEL_NAME} ({model_path.name}) | forecast dates {MIN_DATE} to {MAX_DATE}")

app = Flask(__name__)


# ---------------------------------------------------------------- prediction helpers
def predict_rows(positions, scaled_values):
    """Predict the 3 zones (in KW) for the given row positions of the history table."""
    if IS_SEQUENCE:
        X = np.stack([scaled_values[p - WINDOW + 1: p + 1] for p in positions])
        pred = model.predict(X, verbose=0)
    else:
        X = scaled_values[positions]
        pred = model.predict(X)
    return target_scaler.inverse_transform(pred)


def forecast_day(date):
    """Forecast all 144 ten-minute steps of one day and compare with the actual values."""
    mask = history.index.normalize() == pd.Timestamp(date)
    positions = np.flatnonzero(mask)
    pred = predict_rows(positions, history_scaled.values)
    actual = history.loc[mask, TARGETS].values
    mae = np.abs(actual - pred).mean(axis=0)
    mape = (np.abs(actual - pred) / actual).mean(axis=0) * 100
    return {
        "date": str(date),
        "times": history.index[mask].strftime("%H:%M").tolist(),
        "predicted": {z: pred[:, i].round(1).tolist() for i, z in enumerate(TARGETS)},
        "actual": {z: actual[:, i].round(1).tolist() for i, z in enumerate(TARGETS)},
        "total_predicted_mwh": {z: round(pred[:, i].sum() / 6 / 1000, 2) for i, z in enumerate(TARGETS)},
        "mae": {z: round(float(mae[i]), 1) for i, z in enumerate(TARGETS)},
        "mape": {z: round(float(mape[i]), 2) for i, z in enumerate(TARGETS)},
    }


def forecast_point(timestamp, weather):
    """Forecast one 10-minute step, using weather values given by the user."""
    ts = pd.Timestamp(timestamp).floor("10min")
    if ts not in history.index:
        raise ValueError("Date/time outside the available range")
    row = history.loc[[ts]].copy()
    for col, value in weather.items():
        row[col] = float(value)
    row_scaled = scale_features(row, FEATURES, feature_scaler, target_scaler).values[0]

    pos = history.index.get_loc(ts)
    values = history_scaled.values.copy()
    values[pos] = row_scaled                 # replace the recorded weather with the user's values
    pred = predict_rows([pos], values)[0]
    return {
        "datetime": ts.strftime("%Y-%m-%d %H:%M"),
        "weather": {c: float(row[c].iloc[0]) for c in WEATHER},
        "predicted": {z: round(float(pred[i]), 1) for i, z in enumerate(TARGETS)},
        "actual": {z: round(float(history.loc[ts, z]), 1) for z in TARGETS},
    }


def check_date(value):
    date = pd.Timestamp(value).date()
    if not (MIN_DATE <= date <= MAX_DATE):
        raise ValueError(f"Date must be between {MIN_DATE} and {MAX_DATE}")
    return date


# ---------------------------------------------------------------- routes
@app.route("/")
def index():
    return render_template("index.html", model_name=MODEL_NAME, metrics=INFO["test_metrics"],
                           min_date=MIN_DATE, max_date=MAX_DATE, weather=WEATHER)


@app.route("/api/forecast_day")
def api_forecast_day():
    try:
        date = check_date(request.args.get("date", str(MAX_DATE)))
        return jsonify(forecast_day(date))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json(force=True)
        check_date(data["datetime"])
        weather = {c: data[c] for c in WEATHER if data.get(c) not in (None, "")}
        return jsonify(forecast_point(data["datetime"], weather))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/weather")
def api_weather():
    """Return the recorded weather for a timestamp (used to pre-fill the form)."""
    try:
        check_date(request.args["datetime"])
        ts = pd.Timestamp(request.args["datetime"]).floor("10min")
        return jsonify({c: round(float(history.loc[ts, c]), 3) for c in WEATHER})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME})


if __name__ == "__main__":
    from waitress import serve
    print("Serving on http://127.0.0.1:5000")
    serve(app, host="127.0.0.1", port=5000)