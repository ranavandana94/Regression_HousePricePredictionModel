from preprocess import load_data, prepare_data 
from train import train_model
from evaluate import evaluate_model
from predict import predict_new
from plot import plot_predictions

df = load_data("data/house_data.csv")

X_train, X_test, y_train, y_test, scaler, feature_cols = prepare_data(df)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

model = train_model(X_train, y_train)
y_pred = model.predict(X_test)
rmse = evaluate_model(y_test, y_pred)
print("RMSE:", rmse)

# Plot actual vs predicted
plot_predictions(y_test, y_pred)

# Example: predict new house
feature_dict = {
    'SqFt': 2000,
    'Bedrooms': 3,
    'Bathrooms': 2,
    'Offers': 1,
    'Brick': 0,
    'Neighborhood_1': 0,  # set 1 for relevant neighborhood
    'Neighborhood_2': 1
}
new_price = predict_new(model, scaler, feature_dict, feature_cols)
print("Predicted price:", new_price)