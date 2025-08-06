import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,  accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

train_df = pd.read_csv("../../data/xgboost_data/xgb_train_scaled.csv")
test_df = pd.read_csv("../../data/xgboost_data/xgb_test_scaled.csv")

target_column = "target" 
X_train = train_df.drop(columns=[target_column])
y_train = train_df[target_column]

X_test = test_df.drop(columns=[target_column])
y_test = test_df[target_column]

model = XGBRegressor(
      n_estimators=1000,
            early_stopping_rounds=50,
            objective='reg:squarederror',
            eval_metric=['rmse'],
            learning_rate=0.1,
            reg_alpha=0.5,
            reg_lambda=1.0,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,

            random_state=2025
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE:  {mae:.3f}")
print(f"MSE: {mse:.3f}")
print(f"R²:   {r2:.3f}")

y_test_pred = y_pred
model_name = "XGBoost"
target_name = "PM10"

plt.figure(figsize=(14, 5))


subset_idx = slice(0, min(500, len(y_test)))
plt.plot(y_test.iloc[subset_idx].values, label='True', color='blue', alpha=0.7)
plt.plot(y_test_pred[subset_idx], label='Predicted', color='red', alpha=0.7)


plt.legend()
plt.title(f'{model_name} - Time Series Prediction')
plt.xlabel('Time Index')
plt.ylabel(target_name)
plt.grid(True)
plt.tight_layout()
plt.show()

