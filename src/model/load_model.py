import joblib
import pandas as pd


# Load dữ liệu đầu vào
df = pd.read_csv("D:/AIRPPMHackathon/data/xgboost_data/xgb_test_scaled.csv")

# Loại bỏ cột 'target' nếu có
if 'target' in df.columns:
    X = df.drop(columns=['target'])
else:
    X = df

# Load model
model = joblib.load("D:/AIRPPMHackathon/models/xgboost_pm10_model.joblib")

# Dự đoán
y_pred = model.predict(X)

print(y_pred)
