# House Price Prediction using Linear Regression

## Objective

The objective of this project is to build a Linear Regression model that predicts house prices based on:

- Living Area (GrLivArea)
- Number of Bedrooms (BedroomAbvGr)
- Number of Bathrooms (FullBath)

---

## Dataset

**Source:** Kaggle - House Prices: Advanced Regression Techniques

Dataset File:

- train.csv

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Joblib

---

## Features Used

| Feature | Description |
|---------|-------------|
| GrLivArea | Above ground living area (sq ft) |
| BedroomAbvGr | Number of bedrooms |
| FullBath | Number of bathrooms |

Target Variable:

- SalePrice

---

## Machine Learning Workflow

1. Load Dataset
2. Explore Dataset
3. Feature Selection
4. Train-Test Split
5. Train Linear Regression Model
6. Make Predictions
7. Evaluate Model
8. Visualize Results
9. Save Trained Model

---

## Model Performance

| Metric | Value |
|---------|---------|
| MAE | 35788.06 |
| RMSE | 52975.72 |
| R² Score | 0.6341 |

---

## Project Structure

```
Task-01
│
├── data
│   └── train.csv
│
├── models
│   └── house_price_model.pkl
│
├── outputs
│   ├── actual_vs_predicted.png
│   └── residual_plot.png
│
├── src
│   └── train.py
│
├── README.md
└── requirements.txt
```

---

## Author

**Abhinav Kumar**

B.Tech Computer Science Engineering

Machine Learning Enthusiast
