"""
pm10_forecaster.py

This script reads a JSON input file containing a list of cases and, for each case,
generates a synthetic hourly PM10 forecast at the case’s target location.

Input JSON schema:
{
  "cases": [
    {
      "case_id":        string,
      "stations": [     # list of station objects
        {
          "station_code": string,
          "longitude":    float,
          "latitude":     float,
          "history": [    # list of hourly observations
            {
              "timestamp": str (ISO8601, e.g. "2019-01-01T00:00:00"),
              "pm10":       float
            },
            ...
          ]
        },
        ...
      ],
      "target": {
        "longitude":               float,
        "latitude":                float,
        "prediction_start_time":   str (ISO8601)
      },
      "weather": [ ... ]  # optional array of METAR‐style records
    },
    ...
  ]
}

Usage:
    python pm10_forecaster.py --data-file data.json [--landuse-pbf landuse.pbf] --output-file output.json
"""

import argparse  # For parsing command-line arguments
import json      # For reading and writing JSON files
import random    # For generating random numbers (placeholder for real predictions)
from datetime import datetime, timedelta  # For handling dates and times
import osmium    # For reading OpenStreetMap .pbf files


class LanduseHandler(osmium.SimpleHandler):
    """
    Osmium handler to collect landuse ways and relations from a .pbf file.
    Each object with a 'landuse' tag is stored in a list.
    """
    def __init__(self):
        super().__init__()
        self.landuse_ways = []
        self.landuse_relations = []

    def way(self, w):
        # Called for each way in the .pbf; store if it has a 'landuse' tag
        if 'landuse' in w.tags:
            self.landuse_ways.append({
                "type": "way",
                "id": w.id,
                "landuse": w.tags['landuse'],
                "tags": dict(w.tags),
                # We only store node IDs (refs) here; lat/lon can be resolved later if needed
                "node_refs": [node.ref for node in w.nodes]
            })

    def relation(self, r):
        # Called for each relation in the .pbf; store if it has a 'landuse' tag
        if 'landuse' in r.tags:
            self.landuse_relations.append({
                "type": "relation",
                "id": r.id,
                "landuse": r.tags['landuse'],
                "tags": dict(r.tags),
                # Store member references; for further spatial analysis if needed
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
        features = {}
        # Temporal features
        features["hour"] = forecast_time.hour
        features["day_of_week"] = forecast_time.weekday()
        features["month"] = forecast_time.month
        # Historical PM10 features
        recent_history = history_df[history_df["timestamp"] <= forecast_time]
        if not recent_history.empty:
            features["pm10_lag1"] = recent_history["pm10"].iloc[-1] if len(recent_history) >= 1 else 0.0
            features["pm10_lag2"] = recent_history["pm10"].iloc[-2] if len(recent_history) >= 2 else 0.0
            features["pm10_lag3"] = recent_history["pm10"].iloc[-3] if len(recent_history) >= 3 else 0.0
            features["pm10_mean"] = recent_history["pm10"].mean()
        else:
            features["pm10_lag1"] = 0.0
            features["pm10_lag2"] = 0.0
            features["pm10_lag3"] = 0.0
            features["pm10_mean"] = 0.0
        # Weather features
        relevant_weather = [w for w in weather_data if w.get("date") and
                          pd.to_datetime(w["date"]) <= forecast_time]
        weather_features = parse_metar_weather(relevant_weather[-1] if relevant_weather else {})
        features.update(weather_features)
        # Land-use features
        landuse_features = extract_landuse_features(landuse_data, target_lat, target_lon)
        features.update(landuse_features)
        # Spatial features
        distance_features = get_station_distances(target_lat, target_lon)
        features.update(distance_features)

        # Create feature vector
        X = pd.DataFrame([features])
        # Add missing features with default values
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0.0
        X = X[feature_names]  # Ensure correct order
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
    """
    Generates synthetic PM10 forecasts for each case’s target location.
    - Uses 'prediction_start_time' from each case's target as the base timestamp.
    - Raises an exception if 'prediction_start_time', 'longitude', or 'latitude' are missing/invalid.
    - Optionally prints info about loaded landuse objects.
    - Calls predict_pm10() to obtain the hourly forecast list.
    """
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
            base_forecast_start = datetime.fromisoformat(target["prediction_start_time"])
        except Exception as e:
            raise ValueError(f"Invalid prediction_start_time for case '{case_id}': {e}")

        # Ensure both longitude and latitude are present
        longitude = target.get("longitude")
        latitude = target.get("latitude")
        if longitude is None or latitude is None:
            raise ValueError(f"Case '{case_id}' target must include both 'longitude' and 'latitude'.")

        stations = case.get("stations", [])
        print(f"[DEBUG] Generating for case: {case_id}, "
              f"target: ({latitude}, {longitude}), "
              f"start: {base_forecast_start.isoformat()}")
        print(f"[DEBUG] Available stations: {len(stations)}")

        # (Optional) Gather history from all stations for potential future use
        all_history = []
        for station in stations:
            station_code = station["station_code"]
            history = station.get("history", [])
            all_history.extend(history)
            print(f"  [INFO] Station {station_code}: {len(history)} history points")

        # Call the separate prediction function
        forecast_list = predict_pm10(
            base_time=base_forecast_start,
            history=all_history,
            landuse_data=landuse_data,
            hours=forecast_hours
        )

        predictions.append({
            "case_id": case_id,
            "forecast": forecast_list
        })

    return {"predictions": predictions}


def main():
    parser = argparse.ArgumentParser(description="Generate random PM10 forecasts.")
    parser.add_argument("--data-file", required=True, help="Path to input data.json")
    parser.add_argument("--landuse-pbf", required=False, help="Path to landuse.pbf")
    parser.add_argument("--output-file", required=True, help="Path to write output.json")
    args = parser.parse_args()

    # Read the input JSON file containing cases, stations, and target definitions
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

    # Generate synthetic forecasts (random) for each case’s target
    output = generate_output(data, landuse_data=landuse_data)

    # Write the generated forecasts to the specified output JSON file
    with open(args.output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Read input from: {args.data_file}")
    if args.landuse_pbf:
        print(f"Land use PBF provided at: {args.landuse_pbf}")
    print(f"Wrote forecasts to: {args.output_file}")


if __name__ == "__main__":
    main()