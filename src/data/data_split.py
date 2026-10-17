import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load data
df = pd.read_csv('data/raw_data/raw.csv')

# Extract time features from date column
df['date'] = pd.to_datetime(df['date'])
df['hour'] = df['date'].dt.hour
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek

# Drop original date column
df = df.drop(columns=['date'])

# Split features and target
X = df.drop(columns=['silica_concentrate'])  
y = df['silica_concentrate']                 

# Split into train and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create output directory if it doesn't exist
os.makedirs('data/processed_data', exist_ok=True)

# Save to CSV
X_train.to_csv('data/processed_data/X_train.csv', index=False)
X_test.to_csv('data/processed_data/X_test.csv', index=False)
y_train.to_csv('data/processed_data/y_train.csv', index=False)
y_test.to_csv('data/processed_data/y_test.csv', index=False)

print("Data split completed")