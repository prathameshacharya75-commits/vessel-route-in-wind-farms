"""
Vessel Route Optimization - ML Model Training
Pipeline: Data Cleaning → Feature Engineering → Model Training → Save
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "vesselroute.xlsx")
MODELS_PATH = os.path.join(os.path.dirname(__file__), "models")

# ─────────────────────────────────────────────
# STEP 1 · DATA CLEANING
# ─────────────────────────────────────────────
def load_and_clean(path):
    print("\n[1/4] DATA CLEANING")
    df = pd.read_excel(path)
    print(f"  Raw rows: {len(df)}")

    # Fix column names
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("longtiude", "longitude")   # fix typo
        .str.replace("timestap", "timestamp")    # fix typo
        .str.replace("wave_heights", "wave_height")
        .str.replace("wind_speed", "wind_speed")  # normalise
    )
    # Rename travel_time & trip_id cleanly
    df = df.rename(columns={
        "travel_time": "travel_time_hrs",
        "trip_id":    "trip_id",
        "turbine_name": "turbine_id",
    })
    # handle column if still old name
    if "travel time" in df.columns:
        df = df.rename(columns={"travel time": "travel_time_hrs"})
    if "turbine name" in df.columns:
        df = df.rename(columns={"turbine name": "turbine_id"})
    if "wind speed" in df.columns:
        df = df.rename(columns={"wind speed": "wind_speed"})

    print(f"  Columns fixed: {df.columns.tolist()}")

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"  Duplicates removed: {before - len(df)}")

    # Check missing values
    mv = df.isnull().sum()
    print(f"  Missing values:\n{mv[mv>0] if mv.any() else '  None – clean dataset!'}")

    # Convert timestamp (handles datetime objects, strings, or Excel serial numbers)
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        sample = df["timestamp"].dropna().iloc[0] if not df["timestamp"].dropna().empty else None
        if isinstance(sample, (int, float)):
            df["timestamp"] = pd.to_datetime(df["timestamp"], origin="1899-12-30", unit="D", errors="coerce")
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    print(f"  Timestamp range: {df['timestamp'].min()} → {df['timestamp'].max()}")

    print(f"  Clean rows: {len(df)}")
    return df


# ─────────────────────────────────────────────
# STEP 2 · FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(df):
    print("\n[2/4] FEATURE ENGINEERING")

    # Time features
    df["hour"]       = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"]      = df["timestamp"].dt.month

    # Weather severity score (0-5)
    weather_map = {"Clear": 0, "Cloudy": 1, "Overcast": 2, "Fog": 3, "Rain": 4, "Storm": 5}
    df["weather_severity"] = df["weather"].map(weather_map).fillna(2)

    # Effective speed (speed adjusted for wave height)
    df["effective_speed"] = df["vessel_speed"] / (1 + df["wave_height"] * 0.1)

    # Distance per unit speed ratio
    df["dist_speed_ratio"] = df["distance_km"] / (df["vessel_speed"] + 1e-5)

    # Wind penalty
    df["wind_penalty"] = df["wind_speed"] / (df["vessel_speed"] + 1e-5)

    # Encode categoricals
    le_vessel  = LabelEncoder()
    le_port    = LabelEncoder()
    le_turbine = LabelEncoder()
    le_weather = LabelEncoder()
    le_tech    = LabelEncoder()

    df["vessel_enc"]  = le_vessel.fit_transform(df["vessel"])
    df["port_enc"]    = le_port.fit_transform(df["port"])
    df["turbine_enc"] = le_turbine.fit_transform(df["turbine_id"])
    df["weather_enc"] = le_weather.fit_transform(df["weather"])
    df["tech_enc"]    = le_tech.fit_transform(df["tech"])

    encoders = {
        "vessel":  le_vessel,
        "port":    le_port,
        "turbine": le_turbine,
        "weather": le_weather,
        "tech":    le_tech,
    }

    features = [
        "distance_km", "vessel_speed", "wave_height", "wind_speed",
        "vessel_enc", "port_enc", "turbine_enc", "weather_enc", "tech_enc",
        "weather_severity", "effective_speed", "dist_speed_ratio", "wind_penalty",
        "hour", "day_of_week", "month"
    ]

    print(f"  Features created: {len(features)}")
    return df, features, encoders


# ─────────────────────────────────────────────
# STEP 3 · TRAIN ML MODEL
# ─────────────────────────────────────────────
def train(df, features):
    print("\n[3/4] ML MODEL TRAINING")

    X = df[features]
    y = df["travel_time_hrs"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.08,
        max_depth=5,
        min_samples_split=5,
        random_state=42,
        subsample=0.8
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)

    print(f"  MAE : {mae:.3f} hrs")
    print(f"  RMSE: {rmse:.3f} hrs")
    print(f"  R²  : {r2:.4f}")

    # Feature importance
    imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("  Top features:")
    for feat, val in imp.head(5).items():
        print(f"    {feat}: {val:.3f}")

    return model, {"mae": mae, "rmse": rmse, "r2": r2}


# ─────────────────────────────────────────────
# STEP 4 · SAVE MODEL + ARTIFACTS
# ─────────────────────────────────────────────
def save_artifacts(model, encoders, features, df, metrics):
    print("\n[4/4] SAVING ARTIFACTS")
    os.makedirs(MODELS_PATH, exist_ok=True)

    with open(f"{MODELS_PATH}/model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(f"{MODELS_PATH}/encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)

    with open(f"{MODELS_PATH}/features.pkl", "wb") as f:
        pickle.dump(features, f)

    with open(f"{MODELS_PATH}/metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)

    # Save unique values for UI dropdowns
    meta = {
        "vessels":  sorted(df["vessel"].unique().tolist()),
        "ports":    sorted(df["port"].unique().tolist()),
        "turbines": sorted(df["turbine_id"].unique().tolist()),
        "weathers": sorted(df["weather"].unique().tolist()),
        "techs":    sorted(df["tech"].unique().tolist()),
        "turbine_coords": df.groupby("turbine_id")[["longitude", "latitude"]].first().to_dict(),
        "port_coords": {
            "Buckie Port":       {"lat": 57.6769, "lon": -2.9669},
            "Wick Harbour":      {"lat": 58.4432, "lon": -3.0887},
            "Peterhead Port":    {"lat": 57.5050, "lon": -1.7822},
            "Fraserburgh Port":  {"lat": 57.6920, "lon": -2.0050},
        },
        "vessel_speeds": df.groupby("vessel")["vessel_speed"].mean().to_dict(),
    }
    with open(f"{MODELS_PATH}/meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    print(f"  Saved to {MODELS_PATH}/")
    print("  ✓ model.pkl  encoders.pkl  features.pkl  meta.pkl  metrics.pkl")


if __name__ == "__main__":
    df = load_and_clean(DATA_PATH)
    df, features, encoders = engineer_features(df)
    model, metrics = train(df, features)
    save_artifacts(model, encoders, features, df, metrics)
    print("\n✅ Training complete – run app.py to start the dashboard.\n")
