import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,  accuracy_score

train_df = pd.read_csv("../../data/xgboost_data/xgb_train_scaled.csv")
test_df = pd.read_csv("../../data/xgboost_data/xgb_test_scaled.csv")

target_column = "target" 
X_train = train_df.drop(columns=[target_column])
y_train = train_df[target_column]

X_test = test_df.drop(columns=[target_column])
y_test = test_df[target_column]

model = XGBRegressor(
      n_estimators=5000,
            early_stopping_rounds=1000,
            objective='reg:squarederror',
            eval_metric=['rmse'],
            learning_rate=0.1,
            reg_alpha=0,
            reg_lambda=0.1,
            min_child_weight=0,
            max_depth=6,
            subsample=1,
            colsample_bytree=1,
            random_state=2025
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE:  {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²:   {r2:.3f}")

