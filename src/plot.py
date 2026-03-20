import matplotlib.pyplot as plt

def plot_predictions(y_test, y_pred):
    plt.figure(figsize=(8,5))
    plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual')
    plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted')
    plt.xlabel('Sample')
    plt.ylabel('Price')
    plt.title('Actual vs Predicted House Prices')
    plt.legend()
    plt.show()