import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_data(path="data/house_data.csv"):
    return pd.read_csv(path)


def preprocess_dataframe(df):
    # Handle missing values
    df = df.dropna()

    
    if "SqFt" in df.columns and "Price" in df.columns:
        df["price_per_sqft"] = df["Price"] / df["SqFt"]

    return df


def build_pipeline(X):
    # Identify column types
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    # Remove target leakage if present
    if "Price" in numeric_features:
        numeric_features.remove("Price")

    # Pipelines
    numeric_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("onehot", pd.get_dummies)  
    ])


    return numeric_features, categorical_features, numeric_pipeline


def prepare_data(df):
    
    df = df.fillna(0)

    # Target
    y = np.log1p(df["SalePrice"])
    X = df.drop("SalePrice", axis=1)
    X["TotalSF"] = X["TotalBsmtSF"] + X["GrLivArea"]
    X["HouseAge"] = 2025 - X["YearBuilt"]
    X["TotalBathrooms"] = X["FullBath"] + (0.5 * X.get("HalfBath", 0))
    X["GarageScore"] = X["GarageCars"] * X.get("GarageArea", 0)

    
    X = pd.get_dummies(X, drop_first=True)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Keep raw versions (for XGBoost)
    X_train_raw = X_train.copy()
    X_test_raw = X_test.copy()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    feature_cols = X.columns.tolist()

    return (
    X_train_scaled,
    X_test_scaled,
    X_train_raw,
    X_test_raw,
    y_train,
    y_test,
    scaler,
    feature_cols
)