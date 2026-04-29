import pandas as pd
import numpy as np

def predict_new(model, scaler, feature_dict, feature_cols):
    # Create DataFrame with correct columns
    df = pd.DataFrame([feature_dict])

   
    df = df.reindex(columns=feature_cols, fill_value=0)

    # Scale
    df_scaled = scaler.transform(df)

    # Convert back to DataFrame 
    df_scaled = pd.DataFrame(df_scaled, columns=feature_cols)

    # Predict
    pred_log = model.predict(df_scaled)

    
    pred = np.expm1(pred_log)

    return pred[0]