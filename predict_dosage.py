import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# =====================================================================
# 1. DATA COLLECTION & PREPROCESSING
# =====================================================================
# Load the 1000-instance dataset configured in the File widget
data_path = "Synthetic water chemical dosage data.csv"
df = pd.read_csv(data_path)

# Assign structural data roles (Features vs. Target Variable)
X = df[['pH', 'temperature', 'turbidity']] # Inputs
y = df['dosage']                           # Continuous Target

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =====================================================================
# 2. MACHINE LEARNING MODEL (RANDOM FOREST REGRESSOR)
# =====================================================================
# Initialize the ensemble learning model matching the Orange widget setup
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the historical treatment data
rf_model.fit(X_train, y_train)

# =====================================================================
# 3. PREDICTIONS & ACCURACY METRICS
# =====================================================================
# Generate predictions for the test partition
predictions = rf_model.predict(X_test)

# Calculate system errors (equivalent to the Orange Test & Score evaluation)
rmse = mean_squared_error(y_test, predictions, squared=False)
r2 = r2_score(y_test, predictions)

# Display real-time execution outputs
print("--- AI Model Evaluation Complete ---")
print(f"Dataset Scale: {len(df)} instances successfully processed.")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f} mg/L")
print(f"Model R-squared Accuracy (R² Score): {r2:.4f}")
