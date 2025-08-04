import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import joblib

# Load the preprocessed data
train_data = pd.read_csv('../../data/xgboost_data/xgb_train_scaled.csv')
test_data = pd.read_csv('../../data/xgboost_data/xgb_test_scaled.csv')

# Separate features and target
X_train = train_data.drop(columns=['target']).values
y_train = train_data['target'].values
X_test = test_data.drop(columns=['target']).values
y_test = test_data['target'].values

# Define hyperparameter grid
param_grid = {
    'n_neighbors': [3, 5, 7, 10],
    'weights': ['uniform', 'distance']
}

# Set up GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(
    estimator=KNeighborsRegressor(metric='euclidean'),
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

# Fit the model
grid_search.fit(X_train, y_train)

# Best estimator after search
best_knn = grid_search.best_estimator_

# Print best hyperparameters and score
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV MSE: {-grid_search.best_score_:.4f}")

# Evaluate the best model on the test set
y_pred = best_knn.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test Set Performance:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")

# Save the best model
joblib.dump(best_knn, 'C:/Users/Win10/AIRPPMHackathon/models/knn_model.pkl')
