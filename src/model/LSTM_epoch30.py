import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os

# Tắt oneDNN và ẩn warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# Load the preprocessed LSTM data
X_train_lstm = np.load("../../data/lstm_data/X_train_lstm.npy")
X_test_lstm = np.load("../../data/lstm_data/X_test_lstm.npy")
y_train_lstm = np.load("../../data/lstm_data/y_train_lstm.npy")
y_test_lstm = np.load("../../data/lstm_data/y_test_lstm.npy")
print("Data loaded successfully. Shapes:", X_train_lstm.shape, X_test_lstm.shape)

# Define the LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# Get input shape for the model
input_shape = (X_train_lstm.shape[1], X_train_lstm.shape[2])
model = build_lstm_model(input_shape)

# Define callback for best checkpoint only
checkpoint = ModelCheckpoint(
    filepath='../models/lstm_model_epoch30.h5',  # Lưu mô hình tốt nhất
    monitor='val_loss',
    save_best_only=True,  # Chỉ lưu mô hình có val_loss thấp nhất
    save_freq='epoch',
    verbose=1
)

# Train the model (without EarlyStopping)
print("Starting training...")
history = model.fit(
    X_train_lstm, y_train_lstm,
    epochs=30,  # Chạy đủ 30 epoch
    batch_size=32,
    validation_split=0.2,
    verbose=1,
    callbacks=[checkpoint]
)
print("Training completed.")
print(f"Final val_loss: {history.history['val_loss'][-1]}")

# Evaluate the model
y_pred_lstm = model.predict(X_test_lstm)

# Calculate metrics
mse = mean_squared_error(y_test_lstm, y_pred_lstm)
mae = mean_absolute_error(y_test_lstm, y_pred_lstm)
r2 = r2_score(y_test_lstm, y_pred_lstm)
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R²): {r2:.4f}")

# Visualize training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Visualize predictions vs actual values
plt.figure(figsize=(12, 6))
plt.plot(y_test_lstm[:100], label='Actual PM10')
plt.plot(y_pred_lstm[:100], label='Predicted PM10')
plt.title('LSTM Predictions vs Actual Values')
plt.xlabel('Time Step')
plt.ylabel('Scaled PM10')
plt.legend()
plt.show()

# Save the final model
model.save("../models/lstm_model.h5")
print("Final model saved successfully at ../models/lstm_model.h5")