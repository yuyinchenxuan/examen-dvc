import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Load scaled data
X_train = pd.read_csv('data/processed_data/X_train_scaled.csv')
y_train = pd.read_csv('data/processed_data/y_train.csv')

# Define model
model = RandomForestRegressor(random_state=42)

# Define parameters to test
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# GridSearch with cross-validation
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,  # Use all available cores
    verbose=2
)

# Fit
grid_search.fit(X_train, y_train.values.ravel())

# Results
print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best score (MAE): {-grid_search.best_score_:.4f}")

# Save best parameters as .pkl
os.makedirs('models', exist_ok=True)
with open('models/best_params.pkl', 'wb') as f:
    pickle.dump(grid_search.best_params_, f)

print("\nBest parameters saved to models/best_params.pkl")