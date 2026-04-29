import pandas as pd
import numpy as np

def predict_new(model, scaler, feature_dict, feature_cols):
    # Create DataFrame with correct columns
    df = pd.DataFrame([feature_dict])

    # Ensure same order as training
    df = df.reindex(columns=feature_cols, fill_value=0)

    # Scale
    df_scaled = scaler.transform(df)

    # Convert back to DataFrame (this fixes the warning)
    df_scaled = pd.DataFrame(df_scaled, columns=feature_cols)

    # Predict
    pred_log = model.predict(df_scaled)

    # Convert from log to actual price
    pred = np.expm1(pred_log)

    return pred[0]