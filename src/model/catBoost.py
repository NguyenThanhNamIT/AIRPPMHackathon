import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_csv("../../data/xgboost_data/xgb_train_scaled.csv")
test_df = pd.read_csv("../../data/xgboost_data/xgb_test_scaled.csv")

target_column = "target"

X_train = train_df.drop(columns=[target_column])
y_train = train_df[target_column]

X_test = test_df.drop(columns=[target_column])
y_test = test_df[target_column]

catboost_model = CatBoostRegressor(
   loss_function='RMSE',
    learning_rate=0.1,
    depth=6,
    l2_leaf_reg=0.1,
    random_seed=2025,
    iterations=7000,
    early_stopping_rounds=1000,
    subsample=1.0,
    rsm=1.0,
    min_data_in_leaf=1,
    verbose=100
)
catboost_model.fit(X_train, y_train, eval_set=(X_test, y_test))

catboost_model.fit(X_train, y_train)

y_pred = catboost_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE:  {mae:.3f}")
print(f"MSE: {mse:.3f}")
print(f"R²:   {r2:.3f}")

y_test_pred = y_pred
model_name = "CatBoost"
subset_idx = slice(0, min(500, len(y_test)))

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

axes[0].plot(y_test.iloc[subset_idx].values, label='True', color='blue', alpha=0.7)
axes[0].plot(y_test_pred[subset_idx], label='Predicted', color='red', alpha=0.7)
axes[0].legend()
axes[0].set_title(f'{model_name} - Time Series Prediction')
axes[0].set_xlabel('Time Index')
axes[0].set_ylabel('PM10')

axes[1].scatter(y_test, y_test_pred, alpha=0.6, color='green')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1].set_xlabel('Actual PM10')
axes[1].set_ylabel('Predicted PM10')
axes[1].set_title(f'{model_name} - Actual vs Predicted (R²={r2:.3f})')

plt.tight_layout()
plt.show()
