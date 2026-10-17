import pandas as pd
import pickle
import json
import os
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

# Load scaled test data
X_test = pd.read_csv('data/processed_data/X_test_scaled.csv')
y_test = pd.read_csv('data/processed_data/y_test.csv')

# Load trained model
with open('models/trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

print("Model loaded successfully")

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print metrics
print(f"\nModel Evaluation:")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R2:   {r2:.4f}")

# Save metrics to scores.json
os.makedirs('metrics', exist_ok=True)
scores = {
    'MSE': round(mse, 4),
    'RMSE': round(rmse, 4),
    'MAE': round(mae, 4),
    'R2': round(r2, 4)
}
with open('metrics/scores.json', 'w') as f:
    json.dump(scores, f, indent=4)

print("\nMetrics saved to metrics/scores.json")

# Save predictions
predictions = pd.DataFrame({
    'y_test': y_test.values.ravel(),
    'y_pred': y_pred
})
predictions.to_csv('data/predictions.csv', index=False)

print("Predictions saved to data/predictions.csv")