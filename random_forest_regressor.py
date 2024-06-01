import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
import joblib

# Load and preprocess the dataset
data = pd.read_csv("final_potato_rainfall_data_cleaned.csv")
data['date'] = pd.to_datetime(data['date'], format='%d-%m-%Y')
data_sorted = data.sort_values(by=['state', 'date'])
data_sorted['year'] = data_sorted['date'].dt.year
data_sorted['month'] = data_sorted['date'].dt.month
data_sorted['rainfall_lag_1'] = data_sorted.groupby('state')['rainfall'].shift(1)
data_sorted['rainfall_lag_2'] = data_sorted.groupby('state')['rainfall'].shift(2)
data_sorted['rainfall_lag_3'] = data_sorted.groupby('state')['rainfall'].shift(3)
data_sorted['rolling_avg_3_months'] = data_sorted.groupby('state')['rainfall'].rolling(window=3).mean().reset_index(level=0, drop=True)
data_cleaned = data_sorted.dropna()

# Encode categorical data
column_transformer = ColumnTransformer([
    ('state_encoder', OneHotEncoder(), ['state']),
], remainder='passthrough')
X = data_cleaned[['state', 'rainfall', 'year', 'month', 'rainfall_lag_1', 'rainfall_lag_2', 'rainfall_lag_3', 'rolling_avg_3_months']]
X_transformed = column_transformer.fit_transform(X)
y = data_cleaned['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Train a Decision Tree Regressor
dt_regressor = DecisionTreeRegressor(random_state=42)
dt_regressor.fit(X_train, y_train)

# Save the trained Decision Tree model
joblib.dump(dt_regressor, 'dt_regressor.pkl')

# Load the model (simulating reuse of a trained model)
loaded_dt_regressor = joblib.load('dt_regressor.pkl')

# Random Forest Regressor initialized with parameters inspired by the Decision Tree
rf_regressor = RandomForestRegressor(n_estimators=100, max_depth=loaded_dt_regressor.tree_.max_depth, random_state=42)
rf_regressor.fit(X_train, y_train)

# Evaluate the Random Forest model
rf_score = rf_regressor.score(X_test, y_test)
print(f'Random Forest R^2 score: {rf_score}')
import matplotlib.pyplot as plt
y_pred = rf_regressor.predict(X_test)

# Generate a scatter plot of actual vs predicted prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3, color='blue', label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Ideal')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Potato Prices')
plt.legend()
plt.show()
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

# Histogram of actual prices
plt.hist(y_test, bins=30, alpha=0.7, label='Actual Prices', color='blue')

# Histogram of predicted prices
plt.hist(y_pred, bins=30, alpha=0.7, label='Predicted Prices', color='green')

plt.xlabel('Potato Prices')
plt.ylabel('Frequency')
plt.title('Histogram of Actual and Predicted Potato Prices')
plt.legend()
plt.show()
