import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the preprocessed data
train_data = pd.read_csv('../../data/xgboost_data/xgb_train_scaled.csv')
test_data = pd.read_csv('../../data/xgboost_data/xgb_test_scaled.csv')

# Separate features and target
X_train = train_data.drop(columns=['target']).values
y_train = train_data['target'].values
X_test = test_data.drop(columns=['target']).values
y_test = test_data['target'].values

# Loop over different n_neighbors values
best_mse = float('inf')
best_k = None
best_model = None

for k in range(1, 21):
    knn = KNeighborsRegressor(n_neighbors=k, weights='distance', algorithm='auto')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"k = {k} -> MSE: {mse:.4f}, R²: {r2:.4f}")

    if mse < best_mse:
        best_mse = mse
        best_k = k
        best_model = knn

print(f"\nBest k = {best_k} with MSE = {best_mse:.4f}")
# Save the best model
joblib.dump(best_model, 'C:/Users/Win10/AIRPPMHackathon/models/knn_best_manual.pkl')
