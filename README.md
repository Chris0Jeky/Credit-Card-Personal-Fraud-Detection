# Credit Card Fraud Detection System

A personal fraud detection system proof-of-concept for monitoring credit card transactions and identifying potentially fraudulent activity using a combination of rules and machine learning.

## Project Overview

This project provides a pipeline for:

1.  **Simulating** realistic credit card transactions for a single user account, sampling patterns from a real-world dataset (`fraudTrain.csv`).
2.  Applying **rule-based checks** (location, amount, velocity, etc.) to flag suspicious transactions.
3.  Running **machine learning models** (PCA/Isolation Forest, Random Forest) to identify potential fraud based on transaction features.
4.  **Integrating** signals from rules and ML models into a combined risk score.
5.  **Evaluating** the performance of each detection component against ground truth labels (`is_fraud` flag from the source dataset).
6.  Checking **merchant legitimacy** against a predefined list.
7.  **Analyzing** and **visualizing** transaction patterns, rule violations, ML predictions, evaluation metrics, and integrated risk scores.

## ML Model Training Approach (Important Note)

The current implementation trains the Machine Learning models (PCA/Isolation Forest, Random Forest) **directly on the small set of simulated transactions generated during each pipeline run**.

**Why this approach?**

*   **Demonstration:** It allows the ML components to run within the self-contained pipeline using the generated data, demonstrating the *concept* of applying ML.
*   **Proof-of-Concept:** It fulfills the requirement of having ML models integrated without needing external, pre-trained models for this stage.
*   **Potential for Adaptation (Conceptual):** One could imagine such a system adapting over time to a user's specific spending patterns, although this requires more sophisticated online learning techniques not implemented here.

**Critical Limitations:**

*   **Small, Biased Training Data:** Training on only one user's simulated transactions (e.g., ~75 transactions by default) is insufficient for building robust, generalizable models.
*   **Overfitting:** The models will likely overfit to the specific patterns in the small simulation run and will not perform well on different users or real-world data.
*   **Not Realistic:** Production fraud detection models are trained on massive, diverse datasets encompassing millions of transactions from many users over extended periods.

**Path Forward:**

> For this system to be genuinely effective, the ML models **must be trained offline on a large, representative dataset**. Acquiring or accessing such datasets (like the full `fraudTrain.csv` or commercial alternatives) often requires **significant funding or institutional access**. Once trained, the saved models would be loaded by the prediction scripts (`predict_pca_fraud.py`, `predict_simulated_fraud.py`) for inference only.

Therefore, the current ML results should be viewed as illustrative of the *pipeline's capability* rather than indicative of real-world fraud detection performance. The evaluation metrics generated reflect performance *only on this specific, limited simulation*.

## Components

### Configuration

*   `config.py`: Central configuration for file paths, parameters, thresholds, and model settings.

### Workflow Orchestration

*   `runner.py`: Orchestrates the entire pipeline, running components sequentially. Supports command-line options to control execution flow and parameters.

### Data Simulation

*   `scripts/simulate_entities.py`: Generates simulated account details and transactions using patterns sampled from `fraudTrain.csv`.

### Rule-Based Detection

*   `scripts/check_rules.py`: Applies rule-based checks (location, amount, velocity, account age) and adds `rule_flags` and a binary `rule_triggered` column to the transactions.

### Machine Learning Models (Training on Simulated Data)

*   `models/predict_pca_fraud.py`: Unsupervised anomaly detection using StandardScaler, PCA, and Isolation Forest. *Fits transformers and model on current simulation run.* Outputs `anomaly` flag and `anomaly_score`.
*   `models/predict_simulated_fraud.py`: Supervised classification using Random Forest. Performs basic feature engineering. *Trains model on current simulation run*. Outputs `predicted_fraud` flag and `fraud_probability`.

### Results Integration

*   `scripts/integrate_results.py`: Combines signals from `rule_triggered`, `anomaly_score`, and `fraud_probability` into a weighted `risk_score` and assigns a `risk_level` (Low, Medium, High).

### Performance Evaluation

*   `scripts/evaluate_performance.py`: Compares predictions (Rules, PCA/IF, RF, Integrated) against the `is_fraud` ground truth. Calculates metrics like Precision, Recall, F1-Score, Accuracy, and AUPRC. Outputs `evaluation_report.json`. Optionally plots PR curves.

### Merchant Verification

*   `scripts/merchant_checker.py`: Checks merchant names against `data/legit_companies.txt` using exact and fuzzy matching.

### Analysis & Visualization

*   `scripts/analyze_flags.py`: Analyzes rule violations and transaction patterns (focused on rules). Outputs `flag_analysis.json`.
*   `scripts/visualize_fraud.py`: Generates data visualizations (distributions, time/distance patterns, ML results, evaluation metrics, integrated scores) and creates `visualization_report.html`.

## Getting Started

### Prerequisites

*   Python 3.8+
*   Required packages (install using `pip install -r requirements.txt`):
    *   pandas
    *   numpy
    *   faker
    *   geopy
    *   scikit-learn
    *   matplotlib
    *   seaborn
    *   thefuzz
    *   python-Levenshtein (recommended for faster fuzzy matching)
    *   # PyYAML (if using YAML for config instead of .py)

### Running the Pipeline

```bash
# Ensure config.py is present in the root directory

# Run the full workflow with default settings (75 transactions)
python runner.py --full-run

# Run only simulation (e.g., 100 transactions)
python runner.py --simulate-only --transactions 100

# Run only ML predictions (uses existing simulated data)
python runner.py --ml-only

# Run only evaluation (uses existing predictions) with PR plots
python runner.py --evaluate-only --plot-eval

# Run only integration
python runner.py --integrate-only

# Generate visualizations only (uses existing results)
python runner.py --visualize-only --viz-format svg

# Check a specific merchant (runs independently)
python runner.py --check-merchant "Starbucks"