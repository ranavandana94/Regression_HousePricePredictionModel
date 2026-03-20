import pandas as pd

def predict_new(model, scaler, feature_dict, feature_cols):
    """
    feature_dict: dict with keys = feature names
    feature_cols: list of all columns used for training (including one-hot)
    """
    # Create DataFrame with zeros for all features
    features = pd.DataFrame(0, index=[0], columns=feature_cols)

    # Fill in values from feature_dict
    for key, value in feature_dict.items():
        if key in features.columns:
            features.at[0, key] = value

    # Scale and predict
    features_scaled = scaler.transform(features)
    price = model.predict(features_scaled)
    return price[0]