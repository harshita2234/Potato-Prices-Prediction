import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Load the dataset
data = pd.read_csv("final_potato_rainfall_data_cleaned.csv")
data['date'] = pd.to_datetime(data['date'], format='%d-%m-%Y')
label_encoder = LabelEncoder()
data['state'] = label_encoder.fit_transform(data['state'])
data.drop_duplicates(inplace=True)

# Aggregate data by date
daily_data = data.groupby('date').agg({'rainfall':'mean', 'price':'mean'}).reset_index()

# Normalize the data
scaler = MinMaxScaler()
daily_data[['rainfall', 'price']] = scaler.fit_transform(daily_data[['rainfall', 'price']])

# Function to create sequences
def create_sequences(data, sequence_length):
    xs = []
    ys = []
    for i in range(len(data)-sequence_length):
        x = data[['rainfall', 'price']].iloc[i:(i+sequence_length)].values
        y = data['price'].iloc[i+sequence_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Create sequences
sequence_length = 30
X, y = create_sequences(daily_data, sequence_length)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print("Test Loss:", loss)

# Predict and compare with actual data
predictions = model.predict(X_test)
print(predictions)  # Add your comparison or visualization here


# Predict on the test set
predictions = model.predict(X_test)


# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(predictions, label='Predicted Prices')
plt.plot(y_test, label='Actual Prices')
plt.title('Potato Prices Prediction vs Actual')
plt.xlabel('Time')
plt.ylabel('Normalized Price')
plt.legend()
plt.show()
# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, predictions))

# Calculate MAE
mae = mean_absolute_error(y_test, predictions)

# Calculate R-squared
r2 = r2_score(y_test, predictions)

print("RMSE:", rmse)
print("MAE:", mae)
print("R-squared:", r2)
import matplotlib.pyplot as plt
import seaborn as sns

# Scatter plot of actual vs predicted prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.5)
plt.title('Actual vs Predicted Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Line for perfect prediction
plt.grid(True)
plt.show()
plt.figure(figsize=(10, 6))
sns.histplot(y_test, color="blue", label='Actual Prices', kde=True, stat="density", linewidth=0)
sns.histplot(predictions.ravel(), color="green", label='Predicted Prices', kde=True, stat="density", linewidth=0)
plt.title('Histogram of Actual vs Predicted Prices')
plt.xlabel('Price')
plt.legend()
plt.show()
