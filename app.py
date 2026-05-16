"""
Vessel Route Optimization – Flask API Backend
Run: python app.py
"""

import os, pickle, json, math
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from aco_optimizer import ACORoutePlanner

app = Flask(__name__)
BASE = os.path.dirname(__file__)
MODELS = os.path.join(BASE, "models")

# ── Load artifacts ──────────────────────────────────────
def load_artifacts():
    with open(f"{MODELS}/model.pkl",    "rb") as f: model    = pickle.load(f)
    with open(f"{MODELS}/encoders.pkl", "rb") as f: encoders = pickle.load(f)
    with open(f"{MODELS}/features.pkl", "rb") as f: features = pickle.load(f)
    with open(f"{MODELS}/meta.pkl",     "rb") as f: meta     = pickle.load(f)
    with open(f"{MODELS}/metrics.pkl",  "rb") as f: metrics  = pickle.load(f)
    return model, encoders, features, meta, metrics

try:
    MODEL, ENCODERS, FEATURES, META, METRICS = load_artifacts()
    print("✓ ML artifacts loaded")
except Exception as e:
    print(f"⚠ Could not load model ({e}). Run train_model.py first.")
    MODEL = ENCODERS = FEATURES = META = METRICS = None


# ── Helpers ─────────────────────────────────────────────
WEATHER_SEVERITY = {"Clear": 0, "Cloudy": 1, "Overcast": 2, "Fog": 3, "Rain": 4, "Storm": 5}

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    la1,lo1,la2,lo2 = map(math.radians,[lat1,lon1,lat2,lon2])
    dlat = la2-la1; dlon = lo2-lo1
    a = math.sin(dlat/2)**2 + math.cos(la1)*math.cos(la2)*math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(a))

def predict_travel_time(turbine_id, port, vessel, weather, wave_height, wind_speed, vessel_speed, hour=12, dow=0, month=6):
    if MODEL is None:
        return None, "Model not loaded – run train_model.py first"

    enc = ENCODERS
    try:
        vessel_enc  = enc["vessel"].transform([vessel])[0]
        port_enc    = enc["port"].transform([port])[0]
        turbine_enc = enc["turbine"].transform([turbine_id])[0]
        weather_enc = enc["weather"].transform([weather])[0]
        tech_enc    = enc["tech"].transform(["T001"])[0]   # default tech
    except Exception as e:
        return None, f"Encoding error: {e}"

    # Port coords
    pc = META["port_coords"].get(port, {"lat": 57.69, "lon": -2.51})
    # Turbine coords
    tc_lat = META["turbine_coords"]["latitude"].get(turbine_id, 58.2)
    tc_lon = META["turbine_coords"]["longitude"].get(turbine_id, -2.9)

    dist_km = haversine_km(pc["lat"], pc["lon"], tc_lat, tc_lon)
    ws = float(weather_severity := WEATHER_SEVERITY.get(weather, 2))
    eff_speed    = vessel_speed / (1 + wave_height * 0.1)
    dist_ratio   = dist_km / (vessel_speed + 1e-5)
    wind_penalty = wind_speed  / (vessel_speed + 1e-5)

    row = {
        "distance_km":      dist_km,
        "vessel_speed":     vessel_speed,
        "wave_height":      wave_height,
        "wind_speed":       wind_speed,
        "vessel_enc":       vessel_enc,
        "port_enc":         port_enc,
        "turbine_enc":      turbine_enc,
        "weather_enc":      weather_enc,
        "tech_enc":         tech_enc,
        "weather_severity": ws,
        "effective_speed":  eff_speed,
        "dist_speed_ratio": dist_ratio,
        "wind_penalty":     wind_penalty,
        "hour":             hour,
        "day_of_week":      dow,
        "month":            month,
    }

    X = pd.DataFrame([row])[FEATURES]
    pred = float(MODEL.predict(X)[0])
    return max(0.1, round(pred, 2)), None


# ── Routes ──────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/meta")
def api_meta():
    if META is None:
        return jsonify({"error": "Model not trained yet"}), 503
    return jsonify({
        "vessels":        META["vessels"],
        "ports":          META["ports"],
        "turbines":       META["turbines"],
        "weathers":       META["weathers"],
        "turbine_coords": {
            k: {"lat": META["turbine_coords"]["latitude"][k],
                "lon": META["turbine_coords"]["longitude"][k]}
            for k in META["turbines"]
        },
        "port_coords": META["port_coords"],
        "metrics":    METRICS if METRICS else {},
    })

@app.route("/api/optimise", methods=["POST"])
def api_optimise():
    if MODEL is None:
        return jsonify({"error": "Model not trained. Run train_model.py first."}), 503

    data = request.json
    port         = data.get("port", "Buckie Port")
    turbine_list = data.get("turbines", [])
    vessel       = data.get("vessel", "SEACAT RAINBOW")
    weather      = data.get("weather", "Clear")
    wave_height  = float(data.get("wave_height", 1.0))
    wind_speed   = float(data.get("wind_speed", 10.0))
    vessel_speed = float(META["vessel_speeds"].get(vessel, 15.0))

    if not turbine_list:
        return jsonify({"error": "Select at least one turbine"}), 400

    # Predict travel time to each turbine
    predictions = {}
    errors = []
    for tid in turbine_list:
        t, err = predict_travel_time(tid, port, vessel, weather, wave_height, wind_speed, vessel_speed)
        if err:
            errors.append(err)
        else:
            tc_lat = META["turbine_coords"]["latitude"][tid]
            tc_lon = META["turbine_coords"]["longitude"][tid]
            predictions[tid] = {"time": t, "lat": tc_lat, "lon": tc_lon}

    if errors:
        return jsonify({"error": errors[0]}), 400

    # Build location list for ACO (port first)
    pc = META["port_coords"][port]
    locations = [{"id": f"PORT:{port}", "lat": pc["lat"], "lon": pc["lon"], "pred_time": 0.0}]
    for tid, info in predictions.items():
        locations.append({"id": tid, "lat": info["lat"], "lon": info["lon"], "pred_time": info["time"]})

    aco = ACORoutePlanner(n_ants=50, n_iterations=150)
    result = aco.optimise(locations)

    # Build response
    route_details = []
    total_dist = 0.0
    prev_lat, prev_lon = pc["lat"], pc["lon"]

    for rid in result["route"]:
        if rid.startswith("PORT:"):
            lat, lon = pc["lat"], pc["lon"]
            name = port
            ptime = 0.0
        else:
            lat  = META["turbine_coords"]["latitude"][rid]
            lon  = META["turbine_coords"]["longitude"][rid]
            name = rid
            ptime = predictions[rid]["time"]

        seg_dist = haversine_km(prev_lat, prev_lon, lat, lon)
        total_dist += seg_dist

        route_details.append({
            "id": rid, "name": name, "lat": lat, "lon": lon,
            "pred_time_hrs": ptime,
            "segment_dist_km": round(seg_dist, 2),
        })
        prev_lat, prev_lon = lat, lon

    # Cost estimate (simple: hours × 5000 USD/day ÷ 24)
    cost_per_hr = 5000 / 24
    total_cost = round(result["total_time_hrs"] * cost_per_hr, 0)

    return jsonify({
        "route":            result["route"],
        "route_details":    route_details,
        "total_time_hrs":   result["total_time_hrs"],
        "total_dist_km":    round(total_dist, 2),
        "total_cost_usd":   total_cost,
        "predictions":      {k: v["time"] for k, v in predictions.items()},
        "convergence":      result["convergence"],
        "port":             port,
        "port_coords":      pc,
        "vessel":           vessel,
        "weather":          weather,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5050)
