from sklearn.metrics import mean_squared_error
import numpy as np

def evaluate_model(y_test, y_pred):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse