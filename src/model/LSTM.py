import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

df = pd.read_csv('final_pm10_dataset.csv')
df['DateTime'] = pd.to_datetime(df['DateTime'])
df.sort_values(['station_code', 'DateTime'], inplace=True)

df['year'] = df['DateTime'].dt.year
df['month'] = df['DateTime'].dt.month
df['day'] = df['DateTime'].dt.day
df['hour'] = df['DateTime'].dt.hour
df['dayofweek'] = df['DateTime'].dt.dayofweek
df['dayofyear'] = df['DateTime'].dt.dayofyear


le_station = LabelEncoder()
df['station_encoded'] = le_station.fit_transform(df['station_code'])


feature_cols = [
    'latitude', 'longitude', 'station_encoded',
    'year', 'month', 'day', 'hour', 'dayofweek', 'dayofyear'
]
weather_features = [col for col in df.select_dtypes(include=np.number).columns 
                    if col not in ['pm10'] + feature_cols]
feature_cols += weather_features
feature_cols = list(set(feature_cols))  

df = df[feature_cols + ['pm10']].copy()
df = df.fillna(df.median())


scaler = StandardScaler()
scaled = scaler.fit_transform(df[feature_cols])
df_scaled = pd.DataFrame(scaled, columns=feature_cols)
df_scaled['pm10'] = df['pm10'].values


def create_sequences(data, seq_len=24):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data.iloc[i:i+seq_len][feature_cols].values)
        y.append(data.iloc[i+seq_len]['pm10'])
    return np.array(X), np.array(y)

X, y = create_sequences(df_scaled)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = Sequential([
    LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=False),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])


history = model.fit(X_train, y_train, 
                    epochs=20, 
                    batch_size=64, 
                    validation_split=0.2,
                    verbose=1)


y_pred = model.predict(X_test).flatten()

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.2f}")

plt.figure(figsize=(14, 5))
plt.plot(y_test[:200], label='Actual')
plt.plot(y_pred[:200], label='Predicted')
plt.legend()
plt.title('PM10 Prediction (first 200 samples)')
plt.show()
