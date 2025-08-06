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
import joblib
import pandas as pd


scaler_X = joblib.load("/app/models/scaler_xgb_X.joblib")

MODEL = joblib.load("/app/models/xgboost_pm10_model.joblib")

with open("/app/data/test/data.json", "r") as f:
    data = json.load(f)

# If you use landuse data:
landuse_path = "/app/data/test/landuse.pbf"

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


def parse_weather_from_history(history):
    """
    Extract weather info (temperature, wind_dir, wind_speed, etc.) from the most recent entry in history.
    """
    if not history:
        return {
            "wind_dir": 0.0,
            "wind_speed": 0.0,
            "temperature": 0.0,
            "sea_level_pressure": 0.0,
            "cloud_amount": 0.0,
        }

    last = history[-1]
    return {
        "wind_dir": last.get("wind_dir", 0.0),
        "wind_speed": last.get("wind_speed", 0.0),
        "temperature": last.get("temperature", 0.0),
        "sea_level_pressure": last.get("sea_level_pressure", 0.0),
        "cloud_amount": last.get("cloud_amount", 0.0),
    }

def predict_pm10(base_time, history, landuse_data, hours=24):
    """
    Predict 24-hour PM10 using XGBoost and 18 engineered features.
    """
    forecast_list = []

    # Ensure history is sorted by timestamp
    history = sorted(history, key=lambda x: x["timestamp"])
    last_pm10 = history[-1]["pm10"] if history else 0.0
    weather_feats = parse_weather_from_history(history)

    # Get lat/lon from last record (or landuse_data if needed)
    latitude = history[-1].get("latitude", 0.0)
    longitude = history[-1].get("longitude", 0.0)

    for h in range(hours):
        ts = base_time + timedelta(hours=h)

        # Build feature dictionary
        feature_dict = {
            "year": ts.year,
            "dayofyear": ts.timetuple().tm_yday,
            "hour": ts.hour,
            "week": ts.isocalendar()[1],
            "month": ts.month,
            "dayofweek": ts.weekday(),
            "day": ts.day,
            "month_dummy": ts.month,
            "hour_dummy": ts.hour,
            "day_dummy": ts.day,
            "station_code": 1,              # Replace with proper encoding if needed
            "latitude": latitude,
            "longitude": longitude,
            **weather_feats
        }

        # Convert to DataFrame and scale
        df = pd.DataFrame([feature_dict])
        X_scaled = scaler_X.transform(df)

        # Predict
        y_pred = MODEL.predict(X_scaled)[0]

        forecast_list.append({
            "timestamp": ts.strftime("%Y-%m-%dT%H:%MZ"),
            "pm10_pred": float(y_pred)
        })

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

        # Exit if forecast is empty
        if not forecast_list:
            print(f"[ERROR] Prediction failed or empty for case '{case_id}'", file=sys.stderr)
            sys.exit(1)

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