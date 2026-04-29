from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib


def train_model(X_train, y_train, X_test, y_test, scaler, feature_cols):
    models = {
        "Linear": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42)
    }

    best_model = None
    best_score = float("inf")

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        preds_actual = np.expm1(preds)
        y_actual = np.expm1(y_test)

        rmse = np.sqrt(mean_squared_error(y_actual, preds_actual))
        print(f"{name} RMSE: {rmse}")

        if rmse < best_score:
            best_score = rmse
            best_model = model

    # Save best model
    
    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(feature_cols, "models/feature_cols.pkl")

    return best_model