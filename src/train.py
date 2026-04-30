from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import numpy as np
import joblib


def train_model(
    X_train_scaled,
    y_train,
    X_test_scaled,
    y_test,
    scaler,
    feature_cols,
    X_train_raw,
    X_test_raw
):

    best_model = None
    best_score = float("inf")

    # -----------------------------
    # Linear Regression (scaled)
    # -----------------------------
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    preds = lr.predict(X_test_scaled)

    rmse = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(preds)))
    print(f"Linear RMSE: {rmse}")

    best_model = lr
    best_score = rmse

    # -----------------------------
    # Random Forest (raw)
    # -----------------------------
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_raw, y_train)
    preds = rf.predict(X_test_raw)

    rmse = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(preds)))
    print(f"RandomForest RMSE: {rmse}")

    if rmse < best_score:
        best_model = rf
        best_score = rmse

    # -----------------------------
    # XGBoost (raw + tuning)
    # -----------------------------
    print("\nTraining XGBoost with tuning...")

    xgb_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    param_dist = {
        "n_estimators": [800, 1200, 1500],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.01, 0.03],
        "subsample": [0.7, 0.8],
        "colsample_bytree": [0.7, 0.8],
        "gamma": [0, 0.1],
        "reg_alpha": [0, 0.1],
        "reg_lambda": [1, 5],
    }

    search = RandomizedSearchCV(
        xgb_model,
        param_distributions=param_dist,
        n_iter=25,
        scoring="neg_root_mean_squared_error",
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    # ✅ USE RAW DATA
    search.fit(X_train_raw, y_train)

    xgb_best = search.best_estimator_

    preds = xgb_best.predict(X_test_raw)
    rmse = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(preds)))

    print(f"XGBoost RMSE: {rmse}")
    print("Best Params:", search.best_params_)

    if rmse < best_score:
        best_model = xgb_best
        best_score = rmse

    print("\n✅ Best Model Selected:", type(best_model).__name__)
    print("✅ Best RMSE:", best_score)

    # -----------------------------
    # Save artifacts
    # -----------------------------
    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(feature_cols, "models/feature_cols.pkl")

    return best_model