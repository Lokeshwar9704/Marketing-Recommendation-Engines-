import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# Load the data
data = pd.read_csv('supermarket_sales.csv')

# Drop unnecessary columns
data_cleaned = data.drop(columns=["Invoice ID", "Date", "Time", "gross margin percentage"])

# Encode categorical features
label_encoders = {}
for column in ["Branch", "City", "Customer type", "Gender", "Product line", "Payment"]:
    le = LabelEncoder()
    data_cleaned[column] = le.fit_transform(data_cleaned[column])
    label_encoders[column] = le

# Features and target
X = data_cleaned.drop(columns="Rating")
y = data_cleaned["Rating"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
