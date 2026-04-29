import numpy as np
from sklearn.metrics import mean_squared_error


def evaluate_model(y_test, y_pred):
    
    y_actual = np.expm1(y_test)
    y_pred_actual = np.expm1(y_pred)

    rmse = np.sqrt(mean_squared_error(y_actual, y_pred_actual))
    return rmse