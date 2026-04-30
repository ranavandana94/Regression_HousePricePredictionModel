import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from src.preprocess import load_data

# -----------------------------
# PATH SETUP 
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_artifact(path, required=True):
    """Safely load a .pkl file"""
    full_path = os.path.join(BASE_DIR, path)
    if not os.path.exists(full_path):
        if required:
            st.error(f"Missing file: {path}")
            st.stop()
        else:
            return None
    return joblib.load(full_path)

# -----------------------------
# LOAD ARTIFACTS
# -----------------------------
model = load_artifact("models/best_model.pkl")
feature_cols = load_artifact("models/feature_cols.pkl")
scaler = load_artifact("models/scaler.pkl", required=False)  # optional

# -----------------------------
# LOAD DATA
# -----------------------------
df = load_data(os.path.join(BASE_DIR, "data/train.csv"))

# -----------------------------
# UI
# -----------------------------
st.title("🏠 House Price Prediction App")
st.write("Provide house details to estimate price")

# -----------------------------
# USER INPUTS
# -----------------------------
overall_qual = st.slider("Overall Quality", 1, 10, 5)
gr_liv_area = st.number_input("Living Area (sq ft)", 500, 5000, 1500)
garage_cars = st.slider("Garage Capacity", 0, 4, 2)
total_bsmt_sf = st.number_input("Basement Area (sq ft)", 0, 3000, 800)
full_bath = st.slider("Full Bathrooms", 0, 4, 2)
year_built = st.number_input("Year Built", 1900, 2025, 2000)

# -----------------------------
# BUILD INPUT
# -----------------------------
input_dict = {
    "OverallQual": overall_qual,
    "GrLivArea": gr_liv_area,
    "GarageCars": garage_cars,
    "TotalBsmtSF": total_bsmt_sf,
    "FullBath": full_bath,
    "YearBuilt": year_built,
}

input_df = pd.DataFrame([input_dict])
input_df = input_df.reindex(columns=feature_cols, fill_value=0)

# -----------------------------
# SCALING 
# -----------------------------
model_name = type(model).__name__
if model_name == "LinearRegression":
    input_processed = scaler.transform(input_df)
    input_processed = pd.DataFrame(input_processed, columns=feature_cols)
else:
    input_processed = input_df

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict Price"):
    pred_log = model.predict(input_processed)
    pred_price = np.expm1(pred_log)

    st.success(f"Estimated House Price: ${pred_price[0]:,.2f}")

# -----------------------------
# DEBUG 
# -----------------------------
with st.expander("Debug Info"):
    st.write("Working directory:", os.getcwd())
    models_path = os.path.join(BASE_DIR, "models")
    if os.path.exists(models_path):
        st.write("Models folder contents:", os.listdir(models_path))
    else:
        st.write("Models folder not found")