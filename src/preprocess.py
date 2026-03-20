import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(path ="data/house_data.csv"):
    return pd.read_csv(path)


def prepare_data(df):
    # Convert categorical feature 'Neighborhood' to one-hot
    df = pd.get_dummies(df, columns=['Neighborhood'], drop_first=True)

    # Features and target 
    feature_cols = ['SqFt', 'Bedrooms', 'Bathrooms', 'Offers', 'Brick'] + \
                   [c for c in df.columns if 'Neighborhood_' in c]
    X = df[feature_cols]
    y = df['Price']  

    # Split into train/test
    X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.3, random_state= 42)

    # Reset indices to align X and y
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Scale features 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols
