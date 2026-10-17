import pandas as pd
import pickle
import os
import mlflow
import dagshub
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor

# DagHub MLflow setup
mlflow.set_tracking_uri("https://dagshub.com/yuyinchenxuan/examen-dvc.mlflow")
os.environ['MLFLOW_TRACKING_USERNAME'] = 'yuyinchenxuan'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '5e74af8de8bac0418b22d7ee9f1d70af4ba1449d'

# Load scaled data
X_train = pd.read_csv('data/processed_data/X_train_scaled.csv')
y_train = pd.read_csv('data/processed_data/y_train.csv')

# Load best parameters from GridSearch
with open('models/best_params.pkl', 'rb') as f:
    best_params = pickle.load(f)

print(f"Loaded best parameters: {best_params}")

with mlflow.start_run():
    # Log parameters
    mlflow.log_params(best_params)

    # Train model
    model = RandomForestRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train.values.ravel())

    # Log model to MLflow
    mlflow.sklearn.log_model(
    model, 
    name = "trained_model",
    skops_trusted_types=["sklearn.tree._tree.Tree"]
)

    # Save model locally too
    os.makedirs('models', exist_ok=True)
    with open('models/trained_model.pkl', 'wb') as f:
        pickle.dump(model, f)

print("Model trained, saved to models/trained_model.pkl, and logged to MLflow")