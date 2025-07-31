import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import joblib

final_data = pd.read_csv('final_pm10_dataset.csv')
final_data['DateTime'] = pd.to_datetime(final_data['DateTime'])

final_data['year'] = final_data['DateTime'].dt.year
final_data['month'] = final_data['DateTime'].dt.month
final_data['day'] = final_data['DateTime'].dt.day
final_data['hour'] = final_data['DateTime'].dt.hour
final_data['dayofweek'] = final_data['DateTime'].dt.dayofweek
final_data['dayofyear'] = final_data['DateTime'].dt.dayofyear

le_station = LabelEncoder()
final_data['station_encoded'] = le_station.fit_transform(final_data['station_code'])

categorical_cols = []
for col in final_data.columns:
    if final_data[col].dtype == 'object' and col not in ['DateTime', 'station_code']:
        categorical_cols.append(col)
        le = LabelEncoder()
        final_data[f'{col}_encoded'] = le.fit_transform(final_data[col].astype(str))

feature_cols = [
    'latitude', 'longitude', 'station_encoded',
    'year', 'month', 'day', 'hour', 'dayofweek', 'dayofyear'
]

numeric_cols = final_data.select_dtypes(include=[np.number]).columns.tolist()
weather_features = [col for col in numeric_cols if col not in 
                   ['pm10', 'latitude', 'longitude', 'station_encoded', 
                    'year', 'month', 'day', 'hour', 'dayofweek', 'dayofyear']]
feature_cols.extend(weather_features)
feature_cols = list(set(feature_cols))
feature_cols = [col for col in feature_cols if col in final_data.columns]

X = final_data[feature_cols].copy()
y = final_data['pm10'].copy()
X = X.fillna(X.median())
y = y.fillna(y.median())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=final_data['station_code']
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

xgb_model = XGBRegressor(
    objective='reg:squarederror',
    max_depth=6,
    learning_rate=0.1,
    n_estimators=1000,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='rmse' 
)

xgb_model.fit(X_train_scaled, y_train)


y_train_pred = xgb_model.predict(X_train_scaled)
y_test_pred = xgb_model.predict(X_test_scaled)

def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

train_rmse, train_mae, train_r2 = evaluate_model(y_train, y_train_pred)
test_rmse, test_mae, test_r2 = evaluate_model(y_test, y_test_pred)

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes[0, 0].barh(feature_importance.head(10)['feature'], feature_importance.head(10)['importance'])
axes[0, 0].set_title('Top 10 Feature Importance')
axes[0, 0].set_xlabel('Importance')

axes[0, 1].scatter(y_test, y_test_pred, alpha=0.5)
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 1].set_xlabel('Actual PM10')
axes[0, 1].set_ylabel('Predicted PM10')
axes[0, 1].set_title(f'Test Set: Actual vs Predicted (R² = {test_r2:.3f})')
print(mean_squared_error)

residuals = y_test - y_test_pred
axes[1, 0].scatter(y_test_pred, residuals, alpha=0.5)
axes[1, 0].axhline(y=0, color='r', linestyle='--')
axes[1, 0].set_xlabel('Predicted PM10')
axes[1, 0].set_ylabel('Residuals')
axes[1, 0].set_title('Residual Plot')

if hasattr(xgb_model, 'evals_result_'):
    eval_result = xgb_model.evals_result_
    x_axis = range(len(eval_result['validation_0']['rmse']))
    axes[1, 1].plot(x_axis, eval_result['validation_0']['rmse'], label='Train')
    axes[1, 1].plot(x_axis, eval_result['validation_1']['rmse'], label='Test')
    axes[1, 1].set_xlabel('Epochs')
    axes[1, 1].set_ylabel('RMSE')
    axes[1, 1].set_title('Training History')
    axes[1, 1].legend()

plt.tight_layout()
plt.show()

test_results = pd.DataFrame({
    'actual': y_test,
    'predicted': y_test_pred,
    'station': final_data.loc[y_test.index, 'station_code']
})

station_perf_df = test_results.groupby('station').apply(
    lambda df: pd.Series({
        'rmse': np.sqrt(mean_squared_error(df['actual'], df['predicted'])),
        'mae': mean_absolute_error(df['actual'], df['predicted']),
        'r2': r2_score(df['actual'], df['predicted']),
        'samples': len(df)
    })
).reset_index()

joblib.dump(xgb_model, 'xgboost_pm10_model.joblib')
joblib.dump(scaler, 'scaler_pm10.joblib')
joblib.dump(le_station, 'station_encoder.joblib')
feature_importance.to_csv('feature_importance.csv', index=False)
station_perf_df.to_csv('station_performance.csv', index=False)
mae = mean_absolute_error(y_test, y_test_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
