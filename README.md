# ⚓ Vessel Route Optimizer — ML + ACO Dashboard

A full machine-learning project for wind farm vessel route optimization.

## 🧠 Pipeline
```
Historical Dataset (vesselroute.xlsx)
        ↓  Data Cleaning  (fix columns, remove dupes, convert timestamps)
        ↓  Feature Engineering  (weather severity, effective speed, distance ratios)
        ↓  ML Model Training  (Gradient Boosting Regressor)
        ↓  Predict Travel Time Matrix
        ↓  ACO Optimization  (50 ants · 150 iterations)
        ↓  Best Route → Map + Dashboard
```

## 📁 Project Structure
```
vessel_route_ml/
├── data/
│   └── vesselroute.xlsx          ← your dataset
├── models/                        ← auto-created after training
│   ├── model.pkl
│   ├── encoders.pkl
│   ├── features.pkl
│   ├── meta.pkl
│   └── metrics.pkl
├── templates/
│   └── index.html                 ← dashboard UI
├── train_model.py                 ← STEP 1: train ML model
├── aco_optimizer.py               ← ACO engine
├── app.py                         ← STEP 2: Flask web server
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the ML model
```bash
python train_model.py
```
Expected output:
```
[1/4] DATA CLEANING
  Raw rows: 3000
  Clean rows: 3000

[2/4] FEATURE ENGINEERING
  Features created: 16

[3/4] ML MODEL TRAINING
  MAE : 0.35 hrs
  RMSE: 0.48 hrs
  R²  : 0.9xxx

[4/4] SAVING ARTIFACTS
  ✓ model.pkl  encoders.pkl  features.pkl  meta.pkl  metrics.pkl

✅ Training complete – run app.py to start the dashboard.
```

### 3. Start the dashboard
```bash
python app.py
```
Then open: **http://localhost:5050**

## 🖥️ Dashboard Usage

**Inputs:**
- Select **Departure Port** (Buckie Port, Wick Harbour, Peterhead Port, Fraserburgh Port)
- Choose **Vessel** (SEACAT RAINBOW, LARGO, NJORD ODIN)
- Set **Weather** condition
- Adjust **Wave Height** and **Wind Speed** sliders
- Tick the **Turbines** you want to visit
- Enter optional **Manifest ID**
- Click **Run ML + ACO Optimisation**

**Outputs:**
- 🗺️ Interactive map — turbine locations + animated route
- ⏱ Predicted travel time per turbine (ML)
- 📍 Optimised visit order (ACO)
- 💰 Estimated trip cost (USD)
- 📈 ACO convergence curve

## 🔬 Model Details

| Property        | Value                         |
|----------------|-------------------------------|
| Algorithm      | Gradient Boosting Regressor   |
| Features       | 16 (distance, weather, speed…)|
| Training rows  | 3,000                         |
| Target         | Travel time (hours)           |
| ACO ants       | 50                            |
| ACO iterations | 150                           |

## 📊 Dataset Columns (cleaned)
| Column           | Description                     |
|------------------|---------------------------------|
| turbine_id       | Turbine identifier              |
| longitude/latitude | Turbine GPS position          |
| vessel           | Vessel name                     |
| distance_km      | Port-to-turbine distance        |
| wave_height      | Wave height in metres           |
| wind_speed       | Wind speed in knots             |
| vessel_speed     | Vessel speed (knots)            |
| port             | Departure port                  |
| weather          | Weather condition                |
| travel_time_hrs  | Actual travel time (target)     |
| timestamp        | Trip datetime                   |
