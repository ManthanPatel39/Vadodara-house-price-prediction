import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import StandardScaler 

# --- 1. Load Data ---
DATA_FILE_NAME = 'vadodara_house_price_dataset_new.csv'
try:
    df = pd.read_csv(DATA_FILE_NAME) 
except FileNotFoundError:
    print(f"ERROR: Data file '{DATA_FILE_NAME}' not found. Check the filename and path.")
    exit()

# Separate features (X) and target price (y)
X = df.drop('price', axis=1)
y = df['price']

# --- 2. Data Preprocessing (Encoding and Imputation) ---

# A. One-Hot Encoding: Convert all text/categorical columns (like Location) to dummy variables.
X_encoded = pd.get_dummies(X, drop_first=True) 
X = X_encoded

# B. Imputation: Fills any remaining missing values (NaNs) with the median.
X = X.fillna(X.median()) 

# --- 3. Scaling, Training, and Saving ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # Scale the clean, encoded data

# Save the fitted scaler (Needed for app.py)
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file) 
print("scaler.pkl saved.")

# Training
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LinearRegression() 
model.fit(X_train, y_train)

# --- 4. Save Model and Column Lists (CRITICAL STEP) ---

# Save the new, compatible model file
with open('vadodara_house_model.pkl', 'wb') as file:
    pickle.dump(model, file) 
print("vadodara_house_model.pkl saved.")

# Save the final feature column names (Needed for input alignment)
column_names = list(X.columns)
with open('columns.pkl', 'wb') as file:
    pickle.dump(column_names, file)
print("columns.pkl saved.")

# 🛑 CRITICAL FIX: Extract and save the list of location names for app.py's dropdown.
# This assumes your location features are prefixed with 'location_'.
location_names = [col.replace('location_', '') for col in column_names if col.startswith('location_')]
with open('locations.pkl', 'wb') as file:
    pickle.dump(location_names, file)
print("locations.pkl saved.")


print("\nTraining and saving process complete. You can now run app.py.")