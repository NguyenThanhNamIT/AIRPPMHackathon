FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY models/xgboost_pm10_model.joblib ./models/
COPY data/xgboost_data/scaler.joblib ./data/xgboost_data/
COPY data/xgboost_data/scaler_y.joblib ./data/xgboost_data/
COPY data/xgboost_data/feature_names.json ./data/xgboost_data/

ENTRYPOINT ["python", "main.py"]
CMD ["--data-file", "/data/data.json", "--landuse-pbf", "/data/landuse.pbf", "--output-file", "/data/output.json"]