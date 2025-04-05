# Credit Card Fraud Detection System

A personal fraud detection system for monitoring credit card transactions and identifying potentially fraudulent activity.

## Project Overview

This project provides a pipeline for:

1. Simulating realistic credit card transactions
2. Applying rule-based checks to flag suspicious transactions
3. Running machine learning models to identify potential fraud
4. Checking merchant legitimacy
5. Analyzing and visualizing transaction patterns

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

### Analysis & Visualization

- `analyze_flags.py`: Analyzes rule violations and transaction patterns
- `visualize_fraud.py`: Generates data visualizations, including:
  - Transaction amount distributions
  - Rule violation statistics
  - Time-based patterns
  - Distance patterns
  - ML model results

### Workflow Orchestration

- `runner.py`: Orchestrates the entire pipeline, running each component sequentially
  - Support for command-line options to control execution
  - Handles errors and dependencies gracefully

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Required packages (install using `pip install -r requirements.txt`):
  - pandas
  - numpy
  - faker
  - geopy
  - scikit-learn (for ML models)
  - matplotlib (for visualization)
  - seaborn (for visualization)
  - thefuzz (for fuzzy matching)
  - python-Levenshtein (for faster fuzzy matching)

### Running the Pipeline

```bash
# Run the full workflow with default settings
python runner.py

# Only simulate transactions
python runner.py --simulate-only --transactions 100

# Run only the ML models
python runner.py --ml-only

# Check a specific merchant
python runner.py --check-merchant "Amazon"

# Analyze rule flags
python runner.py --analyze-flags --detailed-analysis

# Generate visualizations
python runner.py --visualize --viz-format png
```

## Results and Output

Simulation output is saved to the `data/simulation_output/` directory:

- `simulated_account_details.json`: Account information
- `simulated_account_transactions.csv`: Transaction data
- `simulated_transactions_with_flags.csv`: Transactions with rule-based flags
- `pca_fraud_predictions.csv`: ML model 1 output
- `simulated_fraud_predictions.csv`: ML model 2 output
- `flag_analysis.json`: Analysis of rule violations
- `visualizations/`: Directory containing charts and graphs

## Future Enhancements

- Real-time transaction monitoring
- Web dashboard for reviewing flagged transactions
- SMS/Email alerts for suspicious activity
- Integration with real banking APIs
- Interactive data exploration tools
- Additional ML models for fraud detection
