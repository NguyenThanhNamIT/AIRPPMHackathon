import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Tắt oneDNN và ẩn warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# Đường dẫn checkpoint (mô hình tốt nhất)
checkpoint_path = "../../models/lstm_model_epoch30.h5"
print(f"Checking path: {os.path.abspath(checkpoint_path)}")

# Kiểm tra file tồn tại
if not os.path.exists(checkpoint_path):
    print(f"Error: File {checkpoint_path} does not exist. Please rerun training with ModelCheckpoint.")
else:
    try:
        # Tải mô hình mà không compile
        loaded_model = load_model(checkpoint_path, compile=False)
        print(f"Model loaded successfully from {checkpoint_path}")
        loaded_model.summary()

        # Biên dịch lại thủ công
        loaded_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

        # Load dữ liệu kiểm tra (điều chỉnh đường dẫn)
        X_test_lstm = np.load("../../data/lstm_data/X_test_lstm.npy")
        y_test_lstm = np.load("../../data/lstm_data/y_test_lstm.npy")
        print("Data loaded successfully. Shapes:", X_test_lstm.shape, y_test_lstm.shape)

        # Dự đoán
        y_pred = loaded_model.predict(X_test_lstm)
        print("Prediction shape:", y_pred.shape)

        # Đánh giá
        mse = mean_squared_error(y_test_lstm, y_pred)
        mae = mean_absolute_error(y_test_lstm, y_pred)
        r2 = r2_score(y_test_lstm, y_pred)
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"R-squared (R²): {r2:.4f}")

        # Visualize Prediction vs Actual Values
        plt.figure(figsize=(12, 6))
        plt.plot(y_test_lstm[:100], label='Actual PM10')
        plt.plot(y_pred[:100], label='Predicted PM10')
        plt.title('LSTM Predictions vs Actual Values')
        plt.xlabel('Time Step')
        plt.ylabel('Scaled PM10')
        plt.legend()
        plt.show()

        # Note: Training history (loss, val_loss) is not available in checkpoint
        print("Note: Training history (e.g., loss, val_loss) is not saved in the checkpoint file.")
        print("To visualize training history, rerun LSTM.py and save history to a file (e.g., using pickle).")

    except Exception as e:
        print(f"Error loading model: {str(e)}")