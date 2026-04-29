from preprocess import load_data, prepare_data
from train import train_model
from evaluate import evaluate_model
from plot import plot_predictions
import numpy as np
import joblib

# Load data
df = load_data("data/train.csv")

# Prepare data
X_train, X_test, y_train, y_test, scaler, feature_cols = prepare_data(df)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# Train + compare models
model = train_model(X_train, y_train, X_test, y_test, scaler, feature_cols)

# Final evaluation on best model
y_pred = model.predict(X_test)


y_actual = np.expm1(y_test)
y_pred_actual = np.expm1(y_pred)

rmse = evaluate_model(y_test, y_pred)
print("Final RMSE (best model):", rmse)

# Plot results
plot_predictions(y_actual, y_pred_actual)