import pandas as pd
from sklearn.preprocessing import RobustScaler
import os

# Load the split data
X_train = pd.read_csv('data/processed_data/X_train.csv')
X_test = pd.read_csv('data/processed_data/X_test.csv')

# Initialize the scaler
scaler = RobustScaler()

# Fit on training data ONLY, then transform both
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # ⚠️ Only transform, don't fit!

# Convert back to DataFrames with original column names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Save to CSV
X_train_scaled.to_csv('data/processed_data/X_train_scaled.csv', index=False)
X_test_scaled.to_csv('data/processed_data/X_test_scaled.csv', index=False)

print("Normalization completed")