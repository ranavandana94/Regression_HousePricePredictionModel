# House Price Prediction

This is a **simple machine learning project** that predicts house prices using **Linear Regression**.  
The model uses the following features:

- `SqFt` — Size of the house in square feet  
- `Bedrooms` — Number of bedrooms  
- `Bathrooms` — Number of bathrooms  

The target variable is the **Price** of the house.

---

## Folder Structure
house-price-prediction/
│
├── data/
│ └── house_data.csv # CSV file with house data
│
├── src/
│ ├── init.py # marks src as a package
│ ├── preprocess.py # data loading, splitting, scaling
│ ├── train.py # model training
│ ├── evaluate.py # RMSE evaluation
│ ├── predict.py # predict new house prices
│ └── main.py # pipeline execution
│
├── requirements.txt # Python dependencies
└── README.md # project instructions


---

## Features

- Modular Python code:
  - `preprocess.py` — data preprocessing, train/test split, scaling  
  - `train.py` — Linear Regression model training  
  - `evaluate.py` — calculate RMSE  
  - `predict.py` — predict price for new houses  
  - `main.py` — runs the full pipeline  
- Ready to run with a **small CSV dataset**.  
- Outputs **RMSE** and predicts price for new houses.

---

## Installation & Setup

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd house-price-prediction


2. **Create a virtual environment**
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows


3. Install dependencies
pip install -r requirements.txt


## How to Run
python src/main.py

Output includes:

Shapes of train/test data

RMSE of the model

Predicted price for an example house (2000 sqft, 3 bedrooms, 2 bathrooms)

Example Output
X_train shape: (6, 3)
y_train shape: (6,)
X_test shape: (3, 3)
y_test shape: (3,)
RMSE: 25946.3
Predicted price for 2000 sqft, 3 bed, 2 bath: 130500.0