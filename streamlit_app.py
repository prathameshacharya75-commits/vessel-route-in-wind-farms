"""
Vessel Route Optimization – Streamlit App
Deploy to Streamlit Cloud with: streamlit run streamlit_app.py
"""

import os, pickle, json, math
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from aco_optimizer import ACORoutePlanner

# ── Page config ─────────────────────────────────────────
st.set_page_config(
    page_title="Vessel Route Optimizer",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Hide Streamlit chrome to give the custom UI full space
st.markdown("""
<style>
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 0 !important; max-width: 100% !important; }
    iframe { border: none !important; }
</style>
""", unsafe_allow_html=True)

BASE   = os.path.dirname(__file__)
MODELS = os.path.join(BASE, "models")

WEATHER_SEVERITY = {"Clear": 0, "Cloudy": 1, "Overcast": 2, "Fog": 3, "Rain": 4, "Storm": 5}


# ── Load artifacts (cached) ──────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        with open(f"{MODELS}/model.pkl",    "rb") as f: model    = pickle.load(f)
        with open(f"{MODELS}/encoders.pkl", "rb") as f: encoders = pickle.load(f)
        with open(f"{MODELS}/features.pkl", "rb") as f: features = pickle.load(f)
        with open(f"{MODELS}/meta.pkl",     "rb") as f: meta     = pickle.load(f)
        with open(f"{MODELS}/metrics.pkl",  "rb") as f: metrics  = pickle.load(f)
        return model, encoders, features, meta, metrics, None
    except Exception as e:
        return None, None, None, None, None, str(e)

MODEL, ENCODERS, FEATURES, META, METRICS, LOAD_ERROR = load_artifacts()


# ── Helpers ─────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    la1, lo1, la2, lo2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = la2 - la1; dlon = lo2 - lo1
    a = math.sin(dlat/2)**2 + math.cos(la1)*math.cos(la2)*math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def predict_travel_time(turbine_id, port, vessel, weather, wave_height, wind_speed, vessel_speed, hour=12, dow=0, month=6):
    enc = ENCODERS
    try:
        vessel_enc  = enc["vessel"].transform([vessel])[0]
        port_enc    = enc["port"].transform([port])[0]
        turbine_enc = enc["turbine"].transform([turbine_id])[0]
        weather_enc = enc["weather"].transform([weather])[0]
        tech_enc    = enc["tech"].transform(["T001"])[0]
    except Exception as e:
        return None, f"Encoding error: {e}"

    pc     = META["port_coords"].get(port, {"lat": 57.69, "lon": -2.51})
    tc_lat = META["turbine_coords"]["latitude"].get(turbine_id, 58.2)
    tc_lon = META["turbine_coords"]["longitude"].get(turbine_id, -2.9)

    dist_km      = haversine_km(pc["lat"], pc["lon"], tc_lat, tc_lon)
    ws           = float(WEATHER_SEVERITY.get(weather, 2))
    eff_speed    = vessel_speed / (1 + wave_height * 0.1)
    dist_ratio   = dist_km / (vessel_speed + 1e-5)
    wind_penalty = wind_speed / (vessel_speed + 1e-5)

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

    X    = pd.DataFrame([row])[FEATURES]
    pred = float(MODEL.predict(X)[0])
    return max(0.1, round(pred, 2)), None


def run_optimise(port, turbine_list, vessel, weather, wave_height, wind_speed):
    vessel_speed = float(META["vessel_speeds"].get(vessel, 15.0))

    predictions = {}
    for tid in turbine_list:
        t, err = predict_travel_time(tid, port, vessel, weather, wave_height, wind_speed, vessel_speed)
        if err:
            return {"error": err}
        tc_lat = META["turbine_coords"]["latitude"][tid]
        tc_lon = META["turbine_coords"]["longitude"][tid]
        predictions[tid] = {"time": t, "lat": tc_lat, "lon": tc_lon}

    pc = META["port_coords"][port]
    locations = [{"id": f"PORT:{port}", "lat": pc["lat"], "lon": pc["lon"], "pred_time": 0.0}]
    for tid, info in predictions.items():
        locations.append({"id": tid, "lat": info["lat"], "lon": info["lon"], "pred_time": info["time"]})

    aco    = ACORoutePlanner(n_ants=50, n_iterations=150)
    result = aco.optimise(locations)

    route_details = []
    total_dist    = 0.0
    prev_lat, prev_lon = pc["lat"], pc["lon"]

    for rid in result["route"]:
        if rid.startswith("PORT:"):
            lat, lon  = pc["lat"], pc["lon"]
            name      = port
            ptime     = 0.0
        else:
            lat   = META["turbine_coords"]["latitude"][rid]
            lon   = META["turbine_coords"]["longitude"][rid]
            name  = rid
            ptime = predictions[rid]["time"]

        seg_dist    = haversine_km(prev_lat, prev_lon, lat, lon)
        total_dist += seg_dist
        route_details.append({
            "id": rid, "name": name, "lat": lat, "lon": lon,
            "pred_time_hrs":    ptime,
            "segment_dist_km":  round(seg_dist, 2),
        })
        prev_lat, prev_lon = lat, lon

    cost_per_hr = 5000 / 24
    total_cost  = round(result["total_time_hrs"] * cost_per_hr, 0)

    return {
        "route":          result["route"],
        "route_details":  route_details,
        "total_time_hrs": result["total_time_hrs"],
        "total_dist_km":  round(total_dist, 2),
        "total_cost_usd": total_cost,
        "predictions":    {k: v["time"] for k, v in predictions.items()},
        "convergence":    result["convergence"],
        "port":           port,
        "port_coords":    pc,
        "vessel":         vessel,
        "weather":        weather,
    }


# ── Build meta JSON for injection ───────────────────────
def build_meta_json():
    turbine_coords = {
        k: {
            "lat": META["turbine_coords"]["latitude"][k],
            "lon": META["turbine_coords"]["longitude"][k],
        }
        for k in META["turbines"]
    }
    return json.dumps({
        "vessels":        META["vessels"],
        "ports":          META["ports"],
        "turbines":       META["turbines"],
        "weathers":       META["weathers"],
        "turbine_coords": turbine_coords,
        "port_coords":    META["port_coords"],
        "metrics": {
            "mae": float(METRICS["mae"]) if METRICS else 0,
            "r2":  float(METRICS["r2"])  if METRICS else 0,
        },
    })


# ── Handle optimise request via query params ─────────────
query_params = st.query_params

optimise_result_json = "null"
if "optimise" in query_params:
    try:
        payload = json.loads(query_params["optimise"])
        result  = run_optimise(
            port         = payload["port"],
            turbine_list = payload["turbines"],
            vessel       = payload["vessel"],
            weather      = payload["weather"],
            wave_height  = float(payload["wave_height"]),
            wind_speed   = float(payload["wind_speed"]),
        )
        optimise_result_json = json.dumps(result)
    except Exception as e:
        optimise_result_json = json.dumps({"error": str(e)})
    # Clear the param after processing
    st.query_params.clear()


# ── Read the HTML template and inject data ───────────────
html_path = os.path.join(BASE, "templates", "index.html")
with open(html_path, "r") as f:
    html_content = f.read()

meta_json  = build_meta_json() if MODEL else "null"
load_error = json.dumps(LOAD_ERROR or "")

# Inject a <script> block before </head> that:
#  1. Patches fetch() so /api/meta and /api/optimise are handled locally
#  2. Triggers fetchMeta() automatically with the pre-loaded data
injection = f"""
<script>
// ── Streamlit shim: intercept fetch calls to Flask API endpoints ──
const _INJECTED_META   = {meta_json};
const _OPTIMISE_RESULT = {optimise_result_json};
const _LOAD_ERROR      = {load_error};

// Streamlit communication helper
function _sendToStreamlit(payload) {{
    // Use window.parent postMessage to communicate with Streamlit
    window.parent.postMessage({{
        type: "streamlit:setComponentValue",
        value: JSON.stringify(payload)
    }}, "*");
    // Also update the URL query param to trigger a Streamlit rerun
    const url = new URL(window.parent.location.href);
    url.searchParams.set("optimise", JSON.stringify(payload));
    window.parent.history.pushState({{}}, "", url);
    window.parent.location.href = url;
}}

const _origFetch = window.fetch.bind(window);
window.fetch = async function(url, opts) {{
    if (url === "/api/meta" || url.endsWith("/api/meta")) {{
        if (!_INJECTED_META) {{
            return new Response(JSON.stringify({{error: _LOAD_ERROR || "Model not loaded"}}), {{
                status: 503,
                headers: {{"Content-Type": "application/json"}}
            }});
        }}
        return new Response(JSON.stringify(_INJECTED_META), {{
            status: 200,
            headers: {{"Content-Type": "application/json"}}
        }});
    }}

    if (url === "/api/optimise" || url.endsWith("/api/optimise")) {{
        if (!_INJECTED_META) {{
            return new Response(JSON.stringify({{error: "Model not loaded"}}), {{
                status: 503,
                headers: {{"Content-Type": "application/json"}}
            }});
        }}

        const body = JSON.parse(opts.body);

        // If we already have a pre-computed result for this request, return it
        if (_OPTIMISE_RESULT && _OPTIMISE_RESULT !== null) {{
            const result = _OPTIMISE_RESULT;
            // Clear so next call goes through
            window._OPTIMISE_RESULT = null;
            return new Response(JSON.stringify(result), {{
                status: result.error ? 400 : 200,
                headers: {{"Content-Type": "application/json"}}
            }});
        }}

        // Trigger Streamlit rerun with optimise params
        // Show a loading message while we wait for page reload
        return new Response(JSON.stringify({{
            _pending: true,
            _params: body
        }}), {{
            status: 200,
            headers: {{"Content-Type": "application/json"}}
        }});
    }}

    return _origFetch(url, opts);
}};

// Patch runOptimisation to use Streamlit for computation
window._streamlitOptimise = async function(payload) {{
    const enc = encodeURIComponent(JSON.stringify(payload));
    const baseUrl = window.parent.location.href.split("?")[0];
    window.parent.location.href = baseUrl + "?optimise=" + enc;
}};
</script>
"""

# Also patch runOptimisation in the JS to use Streamlit when result is pending
patch_js = """
// Override to handle pending Streamlit computation
const _origRunOptimisation = window.runOptimisation;
window.runOptimisation = async function() {
    const port    = document.getElementById("sel-port").value;
    const vessel  = document.getElementById("sel-vessel").value;
    const weather = document.getElementById("sel-weather").value;
    const wave    = parseFloat(document.getElementById("sl-wave").value);
    const wind    = parseFloat(document.getElementById("sl-wind").value);
    const turbines = [...document.querySelectorAll("#turbine-grid input:checked")].map(c => c.value);

    if (!port || !vessel || turbines.length === 0) {
        showToast("Please select port, vessel, and at least one turbine", "error"); return;
    }

    document.getElementById("loading").classList.add("show");
    document.getElementById("btn-optimise").disabled = true;

    try {
        const res = await fetch("/api/optimise", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({port, vessel, weather, wave_height: wave, wind_speed: wind, turbines})
        });
        const data = await res.json();

        if (data._pending) {
            // Redirect to Streamlit for server-side computation
            window._streamlitOptimise(data._params);
            return;
        }

        if (!res.ok || data.error) throw new Error(data.error || "Optimisation failed");
        renderResults(data);
        showToast("✓ Optimisation complete – best route found!", "success");
    } catch(e) {
        showToast("Error: " + e.message, "error");
    } finally {
        document.getElementById("loading").classList.remove("show");
        document.getElementById("btn-optimise").disabled = false;
    }
};
"""

# Inject before </head>
html_content = html_content.replace("</head>", injection + "</head>", 1)

# Inject override after the existing runOptimisation definition (at end of body)
html_content = html_content.replace("</body>", f"<script>{patch_js}</script>\n</body>", 1)

# If we have a pre-computed optimise result, inject JS to auto-render it
if optimise_result_json != "null":
    auto_render = f"""
<script>
window.addEventListener("load", function() {{
    const result = {optimise_result_json};
    if (result && !result.error) {{
        renderResults(result);
        showToast("✓ Optimisation complete – best route found!", "success");
    }} else if (result && result.error) {{
        showToast("Error: " + result.error, "error");
    }}
}});
</script>
"""
    html_content = html_content.replace("</body>", auto_render + "</body>", 1)

# ── Render ───────────────────────────────────────────────
components.html(html_content, height=900, scrolling=False)
