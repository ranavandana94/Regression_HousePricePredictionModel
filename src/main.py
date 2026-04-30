from preprocess import load_data, prepare_data
from train import train_model
from evaluate import evaluate_model
from plot import plot_predictions
import numpy as np
import joblib

# Load data
df = load_data("data/train.csv")

# Prepare data
(
    X_train_scaled,
    X_test_scaled,
    X_train_raw,
    X_test_raw,
    y_train,
    y_test,
    scaler,
    feature_cols
) = prepare_data(df)

print("X_train_scaled shape:", X_train_scaled.shape)
print("y_train shape:", y_train.shape)
print("X_test_scaled shape:", X_test_scaled.shape)
print("y_test shape:", y_test.shape)

# Train + compare models
model = train_model(
    X_train_scaled,
    y_train,
    X_test_scaled,
    y_test,
    scaler,
    feature_cols,
    X_train_raw,
    X_test_raw
)

# -----------------------------
#  Use correct data for prediction
# -----------------------------
model_name = type(model).__name__

if model_name == "LinearRegression":
    X_eval = X_test_scaled
else:
    X_eval = X_test_raw

# Final evaluation on best model
y_pred = model.predict(X_eval)


y_actual = np.expm1(y_test)
y_pred_actual = np.expm1(y_pred)

rmse = evaluate_model(y_test, y_pred)
print("Final RMSE (best model):", rmse)

# Plot results
plot_predictions(y_actual, y_pred_actual)