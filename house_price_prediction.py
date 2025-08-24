# house_price_prediction.py
# House Price Prediction using California Housing Dataset
# Author: Prachi Nayak
# Date: 2025-08-24

# -----------------------
# Step 1: Libraries Import
# -----------------------
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------
# Step 2: Load Dataset
# -----------------------
california = fetch_california_housing()
data = pd.DataFrame(california.data, columns=california.feature_names)
data['MedHouseVal'] = california.target  # Target variable

print("First 5 rows of dataset:")
print(data.head())

# -----------------------
# Step 3: Explore Data
# -----------------------
print("\nDataset info:")
print(data.info())

print("\nDataset statistics:")
print(data.describe())

print("\nCheck for missing values:")
print(data.isnull().sum())

# Plot target distribution
sns.histplot(data['MedHouseVal'], kde=True, bins=50)
plt.title("House Price Distribution")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# -----------------------
# Step 4: Data Preprocessing
# -----------------------
X = data.drop('MedHouseVal', axis=1)
y = data['MedHouseVal']

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------
# Step 5: Train Model
# -----------------------
model = LinearRegression()
model.fit(X_train_scaled, y_train)
print("\nModel training complete!")

# -----------------------
# Step 6: Make Predictions
# -----------------------
y_pred = model.predict(X_test_scaled)

# -----------------------
# Step 7: Evaluate Model
# -----------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error (MSE): {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# -----------------------
# Step 8: Plot Predicted vs Actual
# -----------------------
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()

