# models/predict_simulated_fraud.py
"""
Simulated Features Fraud Detection Model (Random Forest).

** IMPORTANT NOTE ON TRAINING DATA **
This script currently trains a Random Forest model *directly on the
simulated transaction data generated during the current pipeline run*.
This is done for demonstration purposes within the project structure.

**LIMITATIONS:**
- The training set is very small (only transactions from one simulated user).
- The model will likely overfit to this specific user's patterns.
- Performance will not generalize well to other users or real-world data.

**RECOMMENDATION:**
For a robust, real-world system, this model should be trained offline on a
large, diverse dataset (like the full fraudTrain.csv or a commercial dataset)
and then loaded here only for prediction (`model.predict`, `model.predict_proba`).
Acquiring such a dataset often requires funding or access agreements.

This script serves as a placeholder showing how a supervised model could fit
into the pipeline, assuming the necessary features are available.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import joblib # Using joblib for consistency with PCA script
from sklearn.model_selection import train_test_split # Example if splitting needed
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from geopy.distance import geodesic # For distance feature

# Import configuration
try:
    import config
except ModuleNotFoundError:
    print("Error: config.py not found. Make sure it's in the project root.")
    sys.exit(1)

def calculate_distance(row):
    """Calculate distance if coordinates are present."""
    if pd.isna(row.get('merch_lat')) or pd.isna(row.get('merch_long')) or pd.isna(row.get('lat')) or pd.isna(row.get('long')):
        return np.nan
    try:
        return geodesic((row['lat'], row['long']), (row['merch_lat'], row['merch_long'])).km
    except ValueError:
        return np.nan

def preprocess_and_feature_engineer(df):
    """Prepare data and create features for Random Forest."""
    print("Preprocessing data for Random Forest...")
    df_processed = df.copy()

    # Feature Engineering Examples
    # 1. Time features
    try:
        df_processed['trans_datetime'] = pd.to_datetime(df_processed['trans_date_trans_time'])
        df_processed['hour'] = df_processed['trans_datetime'].dt.hour
        df_processed['day_of_week'] = df_processed['trans_datetime'].dt.dayofweek
    except Exception as e:
        print(f"Warning: Could not process time features: {e}")
        df_processed['hour'] = 0 # Default if error
        df_processed['day_of_week'] = 0

    # 2. Distance feature (if coordinates exist)
    if all(col in df.columns for col in ['lat', 'long', 'merch_lat', 'merch_long']):
         df_processed['distance_km'] = df_processed.apply(calculate_distance, axis=1).fillna(0) # Fill NaNs
    else:
         print("Warning: Coordinate columns missing, cannot calculate distance feature.")
         df_processed['distance_km'] = 0 # Default if missing

    # 3. Handle Categorical Features (e.g., 'category')
    if 'category' in df_processed.columns:
        df_processed['category'] = df_processed['category'].fillna('unknown').astype(str)
        # We'll use OneHotEncoder in the pipeline later
        categorical_features = ['category']
    else:
        categorical_features = []

    # Define numeric features to scale
    numeric_features = ['amt', 'hour', 'day_of_week', 'distance_km']
    # Filter based on actual columns present
    numeric_features = [f for f in numeric_features if f in df_processed.columns]

    # Select features for model + target
    model_features = numeric_features + categorical_features
    if not model_features:
        print("Error: No features selected for Random Forest model.", file=sys.stderr)
        sys.exit(1)

    print(f"   Using numeric features: {numeric_features}")
    print(f"   Using categorical features: {categorical_features}")

    if config.RF_TARGET not in df_processed.columns:
        print(f"Error: Target column '{config.RF_TARGET}' not found in data.", file=sys.stderr)
        sys.exit(1)

    X = df_processed[model_features]
    y = df_processed[config.RF_TARGET]

    # --- Preprocessor Definition ---
    # Create transformers for numeric and categorical data
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore')) # Ignore categories not seen in training
    ])

    # Combine transformers using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Keep other columns if any (shouldn't be if model_features is accurate)
    )

    return X, y, preprocessor, df_processed[['trans_num', 'is_fraud']] # Pass IDs/target back

def train_rf_model(X, y, preprocessor):
    """Train the Random Forest model."""
    print("Training Random Forest model (on current simulation data)...")
    print("WARNING: Training on small, single-user simulated data. Model will likely overfit.")

    # Create the full pipeline: preprocess -> classify
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=config.RF_N_ESTIMATORS,
            max_depth=config.RF_MAX_DEPTH,
            random_state=42,
            class_weight='balanced', # Good for imbalanced datasets
            n_jobs=-1 # Use all available cores
        ))
    ])

    # --- Model Training ---
    # NOTE: In a real scenario, you'd likely split data (train/test)
    #       Here, we train on the whole small simulated set for demonstration.
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # model_pipeline.fit(X_train, y_train)
    # print("Sample Classification Report (on test split of simulated data):")
    # y_pred_test = model_pipeline.predict(X_test)
    # print(classification_report(y_test, y_pred_test))

    # Fit on the entire available simulated data
    model_pipeline.fit(X, y)

    # --- Save Model ---
    print(f"Saving trained RF model pipeline to {config.RF_MODEL_FILE}...")
    try:
        joblib.dump(model_pipeline, config.RF_MODEL_FILE)
        print("   Model saved.")
    except Exception as e:
        print(f"Error saving RF model: {e}", file=sys.stderr)
        # Continue without saving if error occurs, prediction will fail later if load is expected

    return model_pipeline

def predict_with_rf(model_pipeline, X, original_data_ids):
    """Make predictions using the trained RF model."""
    print("Making predictions with Random Forest...")
    y_pred_proba = model_pipeline.predict_proba(X)[:, 1] # Probability of class 1 (fraud)
    y_pred = (y_pred_proba >= config.RF_PREDICTION_THRESHOLD).astype(int)

    results_df = original_data_ids.copy()
    results_df['fraud_probability'] = y_pred_proba
    results_df['predicted_fraud'] = y_pred

    print(f"   Predicted {results_df['predicted_fraud'].sum()} frauds out of {len(results_df)}.")
    return results_df

def main():
    print("\n=== Starting Simulated Features Fraud Detection (Random Forest) ===")
    print("WARNING: Model is trained on the small, current simulation run - see script notes.")

    # --- Load Data ---
    print(f"Loading transactions from {config.TRANSACTIONS_FILE}...")
    try:
        transactions_df = pd.read_csv(config.TRANSACTIONS_FILE)
        if transactions_df.empty:
             print("Error: Input transaction file is empty.", file=sys.stderr)
             sys.exit(1)
        print(f"   Loaded {len(transactions_df)} transactions.")
    except FileNotFoundError:
        print(f"Error: Transactions file not found at {config.TRANSACTIONS_FILE}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Preprocess and Feature Engineer ---
    X, y, preprocessor, original_data_ids = preprocess_and_feature_engineer(transactions_df)

    # --- Train Model ---
    # In this Option B setup, we train every time.
    # In Option A, you would load a pre-trained model here instead.
    # try:
    #     print(f"Loading pre-trained RF model from {config.RF_MODEL_FILE}...")
    #     model_pipeline = joblib.load(config.RF_MODEL_FILE)
    # except FileNotFoundError:
    #     print("Error: Pre-trained RF model not found. Cannot proceed with prediction only.", file=sys.stderr)
    #     sys.exit(1)
    # except Exception as e:
    #     print(f"Error loading RF model: {e}", file=sys.stderr)
    #     sys.exit(1)
    model_pipeline = train_rf_model(X, y, preprocessor)


    # --- Predict ---
    results_df = predict_with_rf(model_pipeline, X, original_data_ids)

    # --- Save Results ---
    print(f"\nSaving RF predictions to {config.RF_PREDICTIONS_FILE}...")
    try:
        results_df.to_csv(config.RF_PREDICTIONS_FILE, index=False, float_format='%.6f')
        print("   Predictions saved.")
    except Exception as e:
        print(f"Error saving RF predictions: {e}", file=sys.stderr)

    print("=== Simulated Features Fraud Detection Complete ===")

if __name__ == "__main__":
    main()