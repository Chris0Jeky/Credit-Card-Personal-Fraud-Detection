# models/predict_pca_fraud.py
"""
PCA-based Anomaly Detection Model using Isolation Forest.

** IMPORTANT NOTE ON TRAINING DATA **
This script currently fits the StandardScaler, PCA, and Isolation Forest model
*directly on the simulated transaction data generated during the current
pipeline run*. This is done for demonstration purposes (Option B).

**LIMITATIONS:**
- The training set is very small and specific to one simulated user.
- The definition of "anomaly" learned by the model will be based solely on
  this limited data, potentially misidentifying normal variations as anomalies
  or missing true anomalies that fall within the simulated pattern.
- Performance will not generalize well to other users or real-world data.

**RECOMMENDATION:**
For a more robust system, these components (Scaler, PCA, IF Model) could be
trained offline on a larger, more diverse dataset (potentially the anonymized
PCA dataset mentioned in the project description if features align, or another
suitable dataset). The trained components would then be loaded here for
transforming and predicting on the new simulated data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import joblib
import os

# Import configuration
try:
    import config
except ModuleNotFoundError:
    print("Error: config.py not found. Make sure it's in the project root.")
    sys.exit(1)

# --- Configuration from config.py ---
INPUT_FILE = config.TRANSACTIONS_FILE # Use raw transactions as input
MODEL_FILE = config.PCA_MODEL_FILE
PCA_FILE = config.PCA_TRANSFORMER_FILE
SCALER_FILE = config.PCA_SCALER_FILE
RESULTS_FILE = config.PCA_PREDICTIONS_FILE
N_COMPONENTS = config.PCA_N_COMPONENTS
CONTAMINATION = config.PCA_CONTAMINATION
NUMERIC_FEATURES = config.PCA_NUMERIC_FEATURES # Use features defined in config
# ---

def load_data(input_file):
    """Load transaction data."""
    print(f"Loading data from {input_file}...")
    try:
        df = pd.read_csv(input_file)
        if df.empty:
             print("Error: Input transaction file is empty.", file=sys.stderr)
             sys.exit(1)
        print(f"Loaded {len(df)} transactions.")
        # Ensure trans_num and is_fraud are present for output consistency
        if 'trans_num' not in df.columns:
            print("Warning: 'trans_num' missing from input, adding index as placeholder.")
            df['trans_num'] = df.index.astype(str)
        if 'is_fraud' not in df.columns:
            print("Warning: 'is_fraud' missing from input, adding placeholder column with 0.")
            df['is_fraud'] = 0
        return df
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        sys.exit(1)

def preprocess_data(df):
    """Prepare data for PCA and anomaly detection."""
    print("Preprocessing data for PCA...")

    # Select only numeric features that exist in the dataset AND are in config
    features_to_use = [f for f in NUMERIC_FEATURES if f in df.columns]
    if not features_to_use:
        print("Error: No usable numeric features (defined in config.PCA_NUMERIC_FEATURES) found in the dataset!", file=sys.stderr)
        sys.exit(1)

    print(f"Using features: {features_to_use}")
    X = df[features_to_use].copy()

    # Handle missing values using median imputation
    for col in X.columns:
        if X[col].isnull().any():
            col_median = X[col].median()
            X[col] = X[col].fillna(col_median)
            print(f"   Filled NaNs in '{col}' with median ({col_median:.2f}).")

    # Convert columns to numeric, coercing errors and filling resultant NaNs
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        if X[col].isnull().any():
            col_median = X[col].median() # Recalculate median after coercion
            X[col] = X[col].fillna(col_median)
            print(f"   Filled NaNs in '{col}' (post-coercion) with median ({col_median:.2f}).")

    # Check if any column is still non-numeric (e.g., all NaNs initially)
    if X.isnull().any().any():
        print("Error: Data contains NaNs even after imputation. Check input data.", file=sys.stderr)
        print(X.isnull().sum())
        sys.exit(1)

    return X, df[['trans_num', 'is_fraud']] # Return original IDs/target

def fit_or_load_transformers(X, force_train=False):
    """Fit or load PCA and StandardScaler (Fits on current data in Option B)."""
    # In Option B, we always train on the current data unless models exist from a previous failed run
    # A 'force_train' might be useful for debugging but default is to train if files don't exist.

    # Determine if we need to train
    train_needed = force_train or not (PCA_FILE.exists() and SCALER_FILE.exists())

    if not train_needed:
        print("Loading existing transformers (Scaler, PCA)...")
        try:
            scaler = joblib.load(SCALER_FILE)
            pca = joblib.load(PCA_FILE)
            print("   Transformers loaded successfully.")
            return scaler, pca, False # Indicate transformers were loaded
        except Exception as e:
            print(f"Warning: Error loading existing transformers: {e}. Training new ones.")
            train_needed = True

    print("Training new transformers (Scaler, PCA) on current simulation data...")
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    # Adjust n_components if X has fewer features or samples than requested
    effective_n_components = min(N_COMPONENTS, X.shape[1], X.shape[0])
    if effective_n_components < N_COMPONENTS:
         print(f"Warning: Requested {N_COMPONENTS} PCA components, but data only supports {effective_n_components}.")
    pca = PCA(n_components=effective_n_components, random_state=42)
    pca.fit(X_scaled)

    # Save transformers
    try:
        joblib.dump(scaler, SCALER_FILE)
        joblib.dump(pca, PCA_FILE)
        print("   Saved new Scaler and PCA transformers.")
    except Exception as e:
        print(f"Error saving transformers: {e}", file=sys.stderr)


    # Print variance explained
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"PCA with {effective_n_components} components explains {explained_var:.2%} of variance")

    return scaler, pca, True # Indicate transformers were trained

def fit_or_load_model(X_pca, force_train=False):
    """Fit or load the anomaly detection model (Fits on current data in Option B)."""
     # Determine if we need to train
    train_needed = force_train or not MODEL_FILE.exists()

    if not train_needed:
        print("Loading existing Isolation Forest model...")
        try:
            model = joblib.load(MODEL_FILE)
            print("   Model loaded successfully.")
            return model
        except Exception as e:
            print(f"Warning: Error loading existing model: {e}. Training a new one.")
            train_needed = True

    print("Training new Isolation Forest model on current simulation data (PCA transformed)...")
    # Initialize and train the model
    # Adjust contamination based on data size? For very small N, maybe lower it?
    effective_contamination = CONTAMINATION
    if X_pca.shape[0] < 50: # Arbitrary small number
        effective_contamination = max(0.01, 1 / X_pca.shape[0]) # Ensure at least 1 anomaly if possible
        print(f"   Adjusting contamination to {effective_contamination:.3f} due to small sample size ({X_pca.shape[0]}).")

    model = IsolationForest(
        contamination=effective_contamination,
        random_state=42,
        n_estimators=100, # Default is usually fine
        max_samples='auto'
    )
    model.fit(X_pca)

    # Save the model
    try:
        joblib.dump(model, MODEL_FILE)
        print("   Saved new Isolation Forest model.")
    except Exception as e:
        print(f"Error saving Isolation Forest model: {e}", file=sys.stderr)

    return model

def predict_anomalies(model, X_pca, original_data_ids):
    """Use the model to predict anomalies and return results."""
    print("Predicting anomalies using Isolation Forest...")

    # Get anomaly scores (-1 for anomalies, 1 for normal)
    y_pred = model.predict(X_pca)

    # Get decision function scores (lower = more anomalous)
    anomaly_scores = model.decision_function(X_pca)

    # Create results dataframe - Start with original IDs and ground truth
    results_df = original_data_ids.copy()
    results_df['anomaly'] = np.where(y_pred == -1, 1, 0)
    results_df['anomaly_score'] = anomaly_scores

    # Count anomalies
    anomaly_count = results_df['anomaly'].sum()
    if len(y_pred) > 0:
        print(f"Detected {anomaly_count} anomalies ({anomaly_count/len(y_pred):.2%})")
    else:
        print("Detected 0 anomalies (empty input).")

    return results_df

def main():
    """Main execution function."""
    print("\n=== Starting PCA-based Anomaly Detection (Isolation Forest) ===")
    print("NOTE: Transformers and Model are trained on the current simulation run.")

    # Load and preprocess data
    transactions_df = load_data(INPUT_FILE)
    X, original_data_ids = preprocess_data(transactions_df) # Get IDs/target back

    # Fit/load transformers and transform data
    scaler, pca, _ = fit_or_load_transformers(X) # Ignore 'trained' flag for now
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)

    # Fit/load and apply model
    model = fit_or_load_model(X_pca)
    results_df = predict_anomalies(model, X_pca, original_data_ids) # Pass IDs/target

    # Merge results back with original transaction details for context (Optional but good)
    final_results_df = pd.merge(transactions_df, results_df, on=['trans_num', 'is_fraud'], how='left')

    # Save results
    final_results_df.to_csv(RESULTS_FILE, index=False, float_format='%.6f')
    print(f"\nResults saved to {RESULTS_FILE}\n")

    # Print top anomalies with transaction info
    print("Top 5 most anomalous transactions (lowest score):")
    # Use final_results_df which has original columns
    top_anomalies = final_results_df.sort_values('anomaly_score').head(5)
    if not top_anomalies.empty:
        # Select columns dynamically in case some are missing
        cols_to_print = ['trans_num', 'trans_date_trans_time', 'amt', 'merchant', 'anomaly_score', 'is_fraud']
        cols_present = [col for col in cols_to_print if col in top_anomalies.columns]
        print(top_anomalies[cols_present].to_string(index=False))
        # for idx, row in top_anomalies.iterrows():
        #     print(f"  Transaction {row.get('trans_num', 'N/A')}")
        #     print(f"    Date: {row.get('trans_date_trans_time', 'N/A')}")
        #     print(f"    Amount: ${row.get('amt', -1):.2f}")
        #     print(f"    Merchant: {row.get('merchant', 'N/A')}")
        #     print(f"    Anomaly Score: {row['anomaly_score']:.4f}")
        #     print(f"    Original Fraud Label: {row.get('is_fraud', 'N/A')}")
        #     print()
    else:
        print("   No transactions to display.")

    print("\n=== PCA-based Anomaly Detection Complete ===")

if __name__ == "__main__":
    main()