# ==========================================
# House Price Prediction using Linear Regression
# Task-01 - Prodigy Infotech
# Author: Abhinav Kumar
# ==========================================

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load Dataset

df = pd.read_csv("Task-01/data/train.csv")


# Explore Dataset


print("First 5 Rows of Dataset:\n")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())


# Feature Selection

# Independent Variables (Features)
X = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]

# Dependent Variable (Target)
y = df['SalePrice']

print("\nSelected Features:")
print(X.head())

print("\nTarget Variable:")
print(y.head())


# Split Dataset into Training and Testing


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

print("\nTraining Target Shape:")
print(y_train.shape)

print("\nTesting Target Shape:")
print(y_test.shape)

# Train Linear Regression Model

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel Trained Successfully!")

# Make Predictions

y_pred = model.predict(X_test)

print("\nFirst 10 Predicted Prices:")
print(y_pred[:10])

# Model Evaluation

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("-" * 40)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# Actual vs Predicted Plot

plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred, alpha=0.7)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color='red',
    linewidth=2
)

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")

plt.savefig("Task-01/outputs/actual_vs_predicted.png")

plt.show()

# Residual Plot

residuals = y_test - y_pred

plt.figure(figsize=(8,6))

plt.scatter(y_pred, residuals, alpha=0.7)

plt.axhline(y=0, color='red', linestyle='--')

plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("Residual Plot")

plt.savefig("Task-01/outputs/residual_plot.png")

plt.show()

# Save Model

joblib.dump(model, "Task-01/models/house_price_model.pkl")

print("\nModel saved successfully!")