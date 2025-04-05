"""
Simulated Features Fraud Detection Model.
This script loads simulated transaction data, applies feature engineering
and predicts fraud using a random forest classifier.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import joblib
import os
from geopy.distance import geodesic
import json

# --- Configuration ---
MODEL_DIR = Path(__file__).parent
DATA_DIR = MODEL_DIR.parent / "data" / "simulation_output"
MODELS_OUTPUT_DIR = MODEL_DIR / "trained_models"
RESULTS_DIR = DATA_DIR

# Ensure directories exist
MODELS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Input/output files
DEFAULT_INPUT_FILE = DATA_DIR / "simulated_account_transactions.csv"
ACCOUNT_DETAILS_FILE = DATA_DIR / "simulated_account_details.json"
MODEL_FILE = MODELS_OUTPUT_DIR / "random_forest_model.pkl"
SCALER_FILE = MODELS_OUTPUT_DIR / "rf_standard_scaler.pkl"
RESULTS_FILE = RESULTS_DIR / "simulated_fraud_predictions.csv"

# Model parameters
N_ESTIMATORS = 100
MAX_DEPTH = 10
MIN_SAMPLES_SPLIT = 2
RANDOM_STATE = 42
# ---

def load_data(input_file):
    """Load and preprocess transaction data."""
    print(f"Loading data from {input_file}...")
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} transactions.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        sys.exit(1)

def load_account_details():
    """Load account details for additional context."""
    try:
        with open(ACCOUNT_DETAILS_FILE, 'r') as f:
            account_details = json.load(f)
        return account_details
    except Exception as e:
        print(f"Warning: Could not load account details: {e}")
        return {}

def engineer_features(df, account_details=None):
    """Create fraud detection features from the transaction data."""
    print("Engineering features...")
    features = df.copy()
    
    # Convert timestamp to datetime if it exists
    if 'trans_date_trans_time' in features.columns:
        features['trans_datetime'] = pd.to_datetime(features['trans_date_trans_time'], errors='coerce')
        
        # Extract features from datetime
        features['hour'] = features['trans_datetime'].dt.hour
        features['day_of_week'] = features['trans_datetime'].dt.dayofweek
        features['is_weekend'] = features['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        features['is_night'] = features['hour'].apply(lambda x: 1 if (x < 6 or x >= 22) else 0)
    
    # Calculate distance from home location if coordinates are available
    if account_details and 'lat' in account_details and 'long' in account_details:
        home_lat = account_details.get('lat')
        home_lon = account_details.get('long')
        
        # Calculate distance if merchant coordinates exist
        if 'merch_lat' in features.columns and 'merch_long' in features.columns:
            # Function to calculate distance safely
            def calc_distance(row):
                try:
                    if pd.isna(row['merch_lat']) or pd.isna(row['merch_long']):
                        return np.nan
                    return geodesic((home_lat, home_lon), 
                                   (row['merch_lat'], row['merch_long'])).km
                except:
                    return np.nan
            
            features['distance_from_home'] = features.apply(calc_distance, axis=1)
            
            # Create distance bin features
            features['dist_bin_close'] = features['distance_from_home'].apply(
                lambda x: 1 if (not pd.isna(x) and x < 10) else 0)
            features['dist_bin_medium'] = features['distance_from_home'].apply(
                lambda x: 1 if (not pd.isna(x) and 10 <= x < 100) else 0)
            features['dist_bin_far'] = features['distance_from_home'].apply(
                lambda x: 1 if (not pd.isna(x) and x >= 100) else 0)
    
    # Transaction amount features
    if 'amt' in features.columns:
        # Log transform amount (common for financial data)
        features['log_amount'] = np.log1p(features['amt'])
        
        # Binned amount features
        features['amount_bin_small'] = features['amt'].apply(lambda x: 1 if x < 10 else 0)
        features['amount_bin_medium'] = features['amt'].apply(lambda x: 1 if 10 <= x < 100 else 0)
        features['amount_bin_large'] = features['amt'].apply(lambda x: 1 if 100 <= x < 1000 else 0)
        features['amount_bin_xlarge'] = features['amt'].apply(lambda x: 1 if x >= 1000 else 0)
    
    # Add merchant flags (if any suspicious merchant patterns are known)
    if 'merchant' in features.columns:
        # Flag merchants with 'fraud' in the name (based on the sample data)
        features['suspicious_merchant'] = features['merchant'].str.contains('fraud', case=False).astype(int)
    
    # Add category flags (if category is available)
    if 'category' in features.columns:
        # One-hot encode the category
        category_dummies = pd.get_dummies(features['category'], prefix='cat')
        features = pd.concat([features, category_dummies], axis=1)
    
    # Drop non-numeric columns and those we don't need for modeling
    cols_to_drop = ['trans_datetime', 'trans_date_trans_time', 'merchant', 'category', 
                   'first', 'last', 'street', 'city', 'state', 'zip', 'job',
                   'cc_num', 'trans_num', 'unix_time', 'dob']
    
    # Only drop columns that actually exist
    cols_to_drop = [col for col in cols_to_drop if col in features.columns]
    features.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    # Fill missing values with median
    for col in features.columns:
        if features[col].dtype in [np.float64, np.int64]:
            features[col] = features[col].fillna(features[col].median())
    
    print(f"Engineered {len(features.columns)} features.")
    return features

def train_model(X, y):
    """Train a random forest classifier."""
    print("Training random forest classifier...")
    # Initialize and train model
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        random_state=RANDOM_STATE,
        class_weight='balanced'  # Handle class imbalance
    )
    model.fit(X, y)
    
    # Display feature importances
    features = X.columns
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nTop 10 most important features:")
    for i in range(min(10, len(features))):
        print(f"{features[indices[i]]}: {importances[indices[i]]:.4f}")
    
    # Save the model
    joblib.dump(model, MODEL_FILE)
    
    return model

def predict_fraud(model, X, transactions_df):
    """Use the model to predict fraud and return results."""
    print("Predicting fraud...")
    
    # Predict class (0 = not fraud, 1 = fraud)
    y_pred = model.predict(X)
    
    # Get probability estimates - handle the case where there might only be one class
    if len(model.classes_) > 1:
        y_proba = model.predict_proba(X)[:, 1]  # Probability of fraud class
    else:
        # If only one class was found in training data, use a default probability
        # This happens when all training examples have the same class
        print("Warning: Model only has one class. Using decision function for probabilities.")
        try:
            # Try to use decision function if available
            decision_scores = model.decision_function(X)
            y_proba = 1 / (1 + np.exp(-decision_scores))  # Sigmoid to convert to probabilities
        except:
            print("Decision function not available, using prediction as probability")
            y_proba = y_pred.astype(float)
    
    # Create results dataframe
    results_df = transactions_df.copy()
    results_df['fraud_probability'] = y_proba
    results_df['predicted_fraud'] = y_pred
    
    # Count predicted fraud cases
    fraud_count = y_pred.sum()
    print(f"Detected {fraud_count} potential fraud cases ({fraud_count/len(y_pred):.2%})")
    
    return results_df

def main():
    """Main execution function."""
    print("\n=== Starting Simulated Features Fraud Detection ===\n")
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        input_file = Path(sys.argv[1])
    else:
        input_file = DEFAULT_INPUT_FILE
    
    # Load data
    transactions_df = load_data(input_file)
    account_details = load_account_details()
    
    # Engineer features
    features_df = engineer_features(transactions_df, account_details)
    
    # Check if the dataset includes the fraud label
    has_fraud_label = 'is_fraud' in transactions_df.columns
    
    if has_fraud_label:
        print("Found fraud labels in data, will train a supervised model.")
        # Prepare data for supervised learning
        y = transactions_df['is_fraud'].astype(int)
        X = features_df.select_dtypes(include=['number'])
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        joblib.dump(scaler, SCALER_FILE)
        
        # Train the model
        model = train_model(X_scaled, y)
        
        # Make predictions
        results_df = predict_fraud(model, X_scaled, transactions_df)
        
        # Calculate accuracy if labels exist
        accuracy = (results_df['predicted_fraud'] == results_df['is_fraud']).mean()
        print(f"Model accuracy: {accuracy:.2%}")
    else:
        print("No fraud labels found. Will load pretrained model if available.")
        X = features_df.select_dtypes(include=['number'])
        
        # Try to load model and scaler
        try:
            model = joblib.load(MODEL_FILE)
            scaler = joblib.load(SCALER_FILE)
            X_scaled = scaler.transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
            
            # Make predictions
            results_df = predict_fraud(model, X_scaled, transactions_df)
        except FileNotFoundError:
            print("Error: No pretrained model found and no labels available for training.")
            print("Please provide labeled data for initial training.")
            sys.exit(1)
    
    # Save results
    results_df.to_csv(RESULTS_FILE, index=False)
    print(f"\nResults saved to {RESULTS_FILE}\n")
    
    # Print top potential fraud transactions
    print("\nTop 5 highest fraud probability transactions:")
    top_fraud = results_df.sort_values('fraud_probability', ascending=False).head(5)
    for idx, row in top_fraud.iterrows():
        print(f"  Transaction {row.get('trans_num', idx)}")
        print(f"    Date: {row.get('trans_date_trans_time', 'N/A')}")
        print(f"    Amount: ${row.get('amt', 'N/A')}")
        print(f"    Merchant: {row.get('merchant', 'N/A')}")
        print(f"    Fraud Probability: {row['fraud_probability']:.4f}")
        if 'is_fraud' in row:
            print(f"    Original Fraud Label: {row['is_fraud']}")
        print()
    
    print("=== Simulated Features Fraud Detection Complete ===")

if __name__ == "__main__":
    main()
