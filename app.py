# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
df_hourly = pd.read_csv('C:\\Users\\ademr\\Desktop\\hourly_water_consumption.csv')

# Features (Temperature, Production Output, Hour) and Target (Water Usage)
X = df_hourly[['Temperature (°C)', 'Production Output (Units)', 'Hour']]
y = df_hourly['Water Usage (Liters)']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set for evaluation
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Streamlit UI
st.title('Water Consumption Prediction & Leak Detection')

# Display the evaluation metrics
st.write(f"Model Performance Metrics:")
st.write(f"Mean Absolute Error (MAE): {mae:.2f} Liters")
st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f} Liters")

# User inputs for prediction
temperature = st.number_input('Temperature (°C)', min_value=10.0, max_value=40.0, value=25.0, step=0.1)
production_output = st.number_input('Production Output (Units)', min_value=50, max_value=200, value=120)
hour = st.number_input('Hour of the day (0-23)', min_value=0, max_value=23, value=12)

# Make a new prediction based on user input
input_data = pd.DataFrame({'Temperature (°C)': [temperature], 'Production Output (Units)': [production_output], 'Hour': [hour]})
predicted_usage = model.predict(input_data)[0]

st.write(f"Predicted Water Usage: {predicted_usage:.2f} Liters")

# Actual Flow input for leak detection
actual_flow = st.number_input('Actual Water Flow (Liters)', min_value=100.0, max_value=1000.0, value=predicted_usage)

# Leak detection function
def detect_leak(predicted, actual, threshold=10):
    return abs(predicted - actual) > threshold  # Return True if difference exceeds threshold

# Apply leak detection
if detect_leak(predicted_usage, actual_flow):
    st.error('Leak Detected!')
else:
    st.success('No Leak Detected.')

# Run the Streamlit app with "streamlit run app_name.py"
