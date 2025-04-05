# Credit Card Fraud Detection System

A personal fraud detection system for monitoring credit card transactions and identifying potentially fraudulent activity.

## Project Overview

This project provides a pipeline for:

1. Simulating realistic credit card transactions
2. Applying rule-based checks to flag suspicious transactions
3. Running machine learning models to identify potential fraud
4. Checking merchant legitimacy

## Components

### Data Simulation

- `simulate_entities.py`: Generates simulated account details and transactions using realistic transaction patterns from a real-world dataset.

### Rule-Based Detection

- `check_rules.py`: Applies multiple rule-based checks to transactions, including:
  - Location checks (transactions far from home)
  - Amount checks (high-value transactions)
  - Velocity checks (too many transactions in short period)
  - Account creation vs. transaction time checks

### Machine Learning Models

- `predict_pca_fraud.py`: Anomaly detection using PCA and Isolation Forest
- `predict_simulated_fraud.py`: Feature engineering and Random Forest classification

### Merchant Verification

- `merchant_checker.py`: Checks merchant names against a list of legitimate businesses
  - Supports exact and fuzzy matching
  - Detects suspicious patterns in merchant names

### Workflow Orchestration

- `runner.py`: Orchestrates the entire pipeline, running each component sequentially and handling errors

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Required packages:
  - pandas
  - faker
  - geopy
  - scikit-learn (for ML models)
  - thefuzz (optional, for fuzzy matching)

### Installation

1. Clone the repository
2. Install dependencies: `pip install pandas faker geopy scikit-learn thefuzz`

### Running the Pipeline

```
python runner.py
```

This will:
1. Generate simulated account data
2. Check transactions against fraud rules
3. Run ML models for fraud detection (if dependencies are installed)
4. Verify sample merchants against a whitelist

## Results and Output

Simulation output is saved to `data/simulation_output/` directory:
- `simulated_account_details.json`: Account information
- `simulated_account_transactions.csv`: Transaction data
- `simulated_transactions_with_flags.csv`: Transactions with rule-based flags
- `pca_fraud_predictions.csv`: ML model 1 output
- `simulated_fraud_predictions.csv`: ML model 2 output

## Future Enhancements

- Advanced feature engineering for better fraud detection
- Web interface for viewing flagged transactions
- Real-time transaction monitoring
- Email/SMS alerts for suspicious activity
- Integration with real banking APIs
