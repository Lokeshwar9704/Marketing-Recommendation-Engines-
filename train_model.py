import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the AEP hourly dataset
data = pd.read_csv('AEP_hourly.csv')

# Check if the dataset contains any missing values and drop them (if necessary)
data = data.dropna()

# Features and target
# Assuming the dataset contains a "value" column that represents electricity consumption
# and "Datetime" or other features that can be used for prediction
X = data.drop(columns="AEP_MW")  # Drop the target column
y = data["AEP_MW"]  # Target variable

# Convert datetime column to numerical values if applicable (e.g., using Unix timestamp)
if 'Datetime' in X.columns:
    X['Datetime'] = pd.to_datetime(X['Datetime']).astype(int) / 10**9  # Convert to seconds since epoch

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RandomForestRegressor model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
