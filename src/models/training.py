import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestRegressor

# Load scaled data
X_train = pd.read_csv('data/processed_data/X_train_scaled.csv')
y_train = pd.read_csv('data/processed_data/y_train.csv')

# Load best parameters from GridSearch
with open('models/best_params.pkl', 'rb') as f:
    best_params = pickle.load(f)

print(f"Loaded best parameters: {best_params}")

# Initialize model with best parameters
model = RandomForestRegressor(**best_params, random_state=42)

# Train the model
model.fit(X_train, y_train.values.ravel())

# Save trained model
with open('models/trained_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved to models/trained_model.pkl")