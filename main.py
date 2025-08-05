"""
pm10_forecaster.py
Your solution for PM10 forecasting.
The script should produce a valid output.json file from the provided data.json and landuse.pbf files.
"""
import argparse
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import osmium
import joblib
from geopy.distance import geodesic

class LanduseHandler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.landuse_ways = []
        self.landuse_relations = []

    def way(self, w):
        if 'landuse' in w.tags:
            self.landuse_ways.append({
                "type": "way",
                "id": w.id,
                "landuse": w.tags['landuse'],
                "tags": dict(w.tags),
                "node_refs": [node.ref for node in w.nodes]
            })

    def relation(self, r):
        if 'landuse' in r.tags:
            self.landuse_relations.append({
                "type": "relation",
                "id": r.id,
                "landuse": r.tags['landuse'],
                "tags": dict(r.tags),
                "members": [(m.ref, m.role, m.type) for m in r.members]
            })

def parse_metar_weather(weather_record):
    """Parse METAR-style weather data into features."""
    features = {
        "temperature": 0.0,
        "wind_direction": 0.0,
        "wind_speed": 0.0,
        "dew_point": 0.0,
        "visibility": 0.0,
        "sea_level_pressure": 0.0
    }
    if not weather_record:
        return features
    try:
        tmp = weather_record.get("tmp", "0,0")
        features["temperature"] = float(tmp.replace(",", ".")) if tmp else 0.0
        wnd = weather_record.get("wnd", "0,0,N,0,0")
        wnd_parts = wnd.split(",")
        features["wind_direction"] = float(wnd_parts[0]) if wnd_parts[0].isdigit() else 0.0
        features["wind_speed"] = float(wnd_parts[3].replace(",", ".")) if len(wnd_parts) > 3 else 0.0
        dew = weather_record.get("dew", "0,0")
        features["dew_point"] = float(dew.replace(",", ".")) if dew else 0.0
        vis = weather_record.get("vis", "0,0")
        features["visibility"] = float(vis.split(",")[0]) if vis else 0.0
        slp = weather_record.get("slp", "0,0")
        features["sea_level_pressure"] = float(slp.replace(",", ".")) if slp else 0.0
    except Exception as e:
        print(f"[WARNING] Error parsing weather: {e}")
    return features

def extract_landuse_features(landuse_data, target_lat, target_lon):
    """Extract land-use features based on proximity to target location."""
    features = {"urban_count": 0, "industrial_count": 0, "forest_count": 0}
    if not landuse_data:
        return features
    for way in landuse_data.get("ways", []):
        landuse_type = way.get("landuse", "")
        if landuse_type == "residential":
            features["urban_count"] += 1
        elif landuse_type == "industrial":
            features["industrial_count"] += 1
        elif landuse_type == "forest":
            features["forest_count"] += 1
    return features

def get_station_distances(target_lat, target_lon):
    """Return distances to Krakow stations from target location."""
    stations = {
        "MpKrakAlKras": (50.057678, 19.926189),
        "MpKrakBujaka": (50.010575, 19.949189),
        "MpKrakBulwar": (50.069308, 20.053492),
        "MpKrakOsPias": (50.098508, 20.018269),
        "MpKrakSwoszo": (49.991442, 19.936792),
        "MpKrakWadow": (50.100569, 20.122561),
        "MpKrakZloRog": (50.081197, 19.895358)
    }
    distances = {}
    for code, (lat, lon) in stations.items():
        distances[f"dist_{code}"] = geodesic((target_lat, target_lon), (lat, lon)).km
    return distances

def predict_pm10(base_time, history, landuse_data, target_lat, target_lon, weather_data, hours=24):
    """Generate 24-hour PM10 forecast using XGBoost model."""
    try:
        model = joblib.load("models/xgboost_pm10_model.joblib")
        scaler = joblib.load("data/xgboost_data/scaler.joblib")
        scaler_y = joblib.load("data/xgboost_data/scaler_y.joblib")
        with open("data/xgboost_data/feature_names.json", "r") as f:
            feature_names = json.load(f)["feature_names"]
    except Exception as e:
        raise ValueError(f"Failed to load model, scalers, or feature names: {e}")

    # Convert history to DataFrame
    history_df = pd.DataFrame(history)
    if not history_df.empty:
        history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])
        history_df["pm10"] = history_df["pm10"].astype(float)
    else:
        history_df = pd.DataFrame(columns=["timestamp", "pm10"])

    forecast_list = []
    for h in range(hours):
        forecast_time = base_time + timedelta(hours=h)
        # Define features in the exact order used during training
        features = [
            0.0,  # temperature (placeholder, use pm10 lag1 for now)
            0.0,  # wind_speed (placeholder, use pm10 mean for now)
            1 if forecast_time.month == base_time.month else 0,  # month_dummy
            0.0,  # wind_dir (placeholder, use weather if available)
            0.0,  # cloud_amount (placeholder)
            0.0,  # station_code (placeholder)
            1 if forecast_time.day == base_time.day else 0,  # day_dummy
            forecast_time.year,  # year
            1 if forecast_time.hour == base_time.hour else 0,  # hour_dummy
            0.0,  # sea_level_pressure (placeholder)
            forecast_time.weekday(),  # dayofweek
            target_lon,  # longitude
            target_lat,  # latitude
            forecast_time.day,  # day
            forecast_time.hour,  # hour
            forecast_time.isocalendar()[1],  # week
            forecast_time.month,  # month
            forecast_time.timetuple().tm_yday  # dayofyear
        ]
        # Update placeholders with available data
        recent_history = history_df[history_df["timestamp"] <= forecast_time]
        if not recent_history.empty:
            features[0] = recent_history["pm10"].iloc[-1] if len(recent_history) >= 1 else 0.0  # temperature
            features[1] = recent_history["pm10"].mean() if len(recent_history) >= 1 else 0.0  # wind_speed
        weather_features = parse_metar_weather([w for w in weather_data if w.get("date") and pd.to_datetime(w["date"]) <= forecast_time][-1] if weather_data else {})
        features[3] = weather_features["wind_direction"]  # wind_dir
        features[9] = weather_features["sea_level_pressure"]  # sea_level_pressure
        features[4] = weather_features.get("cloud_amount", 0.0)  # cloud_amount

        # Create feature vector with explicit order
        X = pd.DataFrame([features], columns=["temperature", "wind_speed", "month_dummy", "wind_dir", "cloud_amount", 
                                             "station_code", "day_dummy", "year", "hour_dummy", "sea_level_pressure", 
                                             "dayofweek", "longitude", "latitude", "day", "hour", "week", "month", "dayofyear"])
        # Scale features
        X_scaled = scaler.transform(X)
        # Predict
        pm10_pred_scaled = model.predict(X_scaled)[0]
        # Inverse transform prediction
        pm10_pred = scaler_y.inverse_transform([[pm10_pred_scaled]])[0][0]
        pm10_pred = max(0, float(pm10_pred))  # Ensure non-negative

        # Format output
        ts = forecast_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        forecast_list.append({
            "timestamp": ts,
            "pm10_pred": round(pm10_pred, 1)
        })

    if len(forecast_list) != 24:
        raise ValueError(f"Forecast must contain exactly 24 hours, got {len(forecast_list)}")
    return forecast_list

def generate_output(data, landuse_data=None, forecast_hours=24):
    predictions = []
    if landuse_data:
        total = len(landuse_data["ways"]) + len(landuse_data["relations"])
        print(f"[INFO] Landuse objects loaded: {total}")

    for case in data["cases"]:
        case_id = case["case_id"]
        target = case.get("target")
        if not target or "prediction_start_time" not in target:
            raise ValueError(f"Case '{case_id}' is missing 'prediction_start_time' in target.")

        # Parse 'prediction_start_time' into a datetime object; raise on parse failure
        try:
            base_forecast_start = datetime.fromisoformat(target["prediction_start_time"].replace("Z", "+00:00"))
        except Exception as e:
            raise ValueError(f"Invalid prediction_start_time for case '{case_id}': {e}")
        longitude = target.get("longitude")
        latitude = target.get("latitude")
        if longitude is None or latitude is None:
            raise ValueError(f"Case '{case_id}' target must include both 'longitude' and 'latitude'.")
        stations = case.get("stations", [])
        weather_data = case.get("weather", [])
        print(f"[DEBUG] Generating for case: {case_id}, target: ({latitude}, {longitude}), start: {base_forecast_start.isoformat()}")
        print(f"[DEBUG] Available stations: {len(stations)}")
        all_history = []
        for station in stations:
            station_code = station["station_code"]
            history = station.get("history", [])
            all_history.extend(history)
            print(f"  [INFO] Station {station_code}: {len(history)} points")
        forecast_list = predict_pm10(
            base_time=base_forecast_start,
            history=all_history,
            landuse_data=landuse_data,
            target_lat=latitude,
            target_lon=longitude,
            weather_data=weather_data,
            hours=forecast_hours
        )
        predictions.append({
            "case_id": case_id,
            "forecast": forecast_list
        })
    return {"predictions": predictions}

def main():
    parser = argparse.ArgumentParser(description="Generate PM10 forecasts.")
    parser.add_argument("--data-file", required=True, help="Path to input data.json")
    parser.add_argument("--landuse-pbf", required=False, help="Path to landuse.pbf")
    parser.add_argument("--output-file", required=True, help="Path to write output.json")
    args = parser.parse_args()

    with open(args.data_file, "r") as f:
        data = json.load(f)

    landuse_data = None
    if args.landuse_pbf:
        print(f"Reading landuse data from: {args.landuse_pbf}")
        handler = LanduseHandler()
        handler.apply_file(args.landuse_pbf)
        print(f"Found {len(handler.landuse_ways)} landuse ways.")
        print(f"Found {len(handler.landuse_relations)} landuse relations.")
        landuse_data = {
            "ways": handler.landuse_ways,
            "relations": handler.landuse_relations
        }

    output = generate_output(data, landuse_data=landuse_data)
    with open(args.output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Read input from: {args.data_file}")
    if args.landuse_pbf:
        print(f"Land use PBF provided at: {args.landuse_pbf}")
    print(f"Wrote forecasts to: {args.output_file}")

if __name__ == "__main__":
    # Simulate arguments for local testing if not provided
    import sys
    if len(sys.argv) < 4:
        sys.argv = ["main.py", "--data-file", "test/data.json", "--landuse-pbf", "test/landuse.pbf", "--output-file", "test/output.json"]
    main()