import subprocess
import sys
import argparse
from pathlib import Path
import time
import os

# --- Script Paths ---
# Use Path objects for better cross-platform compatibility
ROOT_DIR = Path(__file__).parent # Assumes runner.py is in the project root

SCRIPTS_DIR = ROOT_DIR / "scripts"

SIMULATE_SCRIPT = SCRIPTS_DIR / "simulate_entities.py"
CHECK_RULES_SCRIPT = SCRIPTS_DIR / "check_rules.py"
MERCHANT_CHECKER_SCRIPT = SCRIPTS_DIR / "merchant_checker.py"

# Add paths to future model prediction scripts here
MODEL_PCA_PREDICT_SCRIPT = ROOT_DIR / "models" / "predict_pca_fraud.py"
MODEL_SIMULATED_PREDICT_SCRIPT = ROOT_DIR / "models" / "predict_simulated_fraud.py"
# ---

def run_script(script_path, args=[], script_desc="script"):
    """Runs a python script using subprocess and checks for errors."""
    if not script_path.is_file():
        print(f"Error: {script_desc} not found at {script_path}", file=sys.stderr)
        return False

    command = [sys.executable, str(script_path)] + args
    print(f"\n>>> Running {script_desc}: {' '.join(command)}")
    start_time = time.time()
    # Use text=True for automatic encoding/decoding
    # Use capture_output=False to see the script's output in real-time
    result = subprocess.run(command, text=True, capture_output=False,
                            check=False)  # check=False allows us to handle errors manually
    end_time = time.time()
    print(f"<<< Finished {script_desc} in {end_time - start_time:.2f} seconds. Exit code: {result.returncode}")

    if result.returncode != 0:
        print(f"Error running {script_path.name}. Please check its output above.", file=sys.stderr)
        # If output was captured (capture_output=True), print stderr:
        # if result.stderr:
        #     print("--- Error Output ---", file=sys.stderr)
        #     print(result.stderr, file=sys.stderr)
        #     print("--- End Error Output ---", file=sys.stderr)
        return False
    return True

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Credit Card Fraud Detection Workflow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--simulate-only", 
        action="store_true", 
        help="Run only the simulation step"
    )
    parser.add_argument(
        "--check-only", 
        action="store_true", 
        help="Run only the rule checking step"
    )
    parser.add_argument(
        "--ml-only", 
        action="store_true", 
        help="Run only the ML model steps"
    )
    parser.add_argument(
        "--transactions", 
        type=int, 
        default=75, 
        help="Number of transactions to simulate"
    )
    parser.add_argument(
        "--check-merchant", 
        type=str,
        help="Check a specific merchant name against the legitimate list"
    )
    return parser.parse_args()

def main():
    """Orchestrates the simulation and checking process."""
    # Parse arguments
    args = parse_arguments()
    
    # Set environment variable for number of transactions
    os.environ["NUM_TRANSACTIONS"] = str(args.transactions)
    
    print("=============================================")
    print("=== Starting Fraud Detection Simulation Workflow ===")
    print("=============================================")
    workflow_start_time = time.time()

    # Determine which steps to run
    run_simulate = not (args.check_only or args.ml_only)
    run_check = not (args.simulate_only or args.ml_only)
    run_ml = not (args.simulate_only or args.check_only)
    
    # If no specific step is selected, run all steps
    if not (run_simulate or run_check or run_ml):
        run_simulate = run_check = run_ml = True

    # Step 1: Simulate account and transactions
    if run_simulate:
        print("\n--- Step 1: Generating Simulated Data ---")
        print(f"Simulating {args.transactions} transactions...")
        if not run_script(SIMULATE_SCRIPT, script_desc="Simulation Script"):
            print("\nWorkflow aborted due to error in simulation.", file=sys.stderr)
            sys.exit(1)
    
    # Step 2: Check rules on simulated data
    if run_check:
        print("\n--- Step 2: Applying Rule-Based Checks ---")
        if not run_script(CHECK_RULES_SCRIPT, script_desc="Rule Checking Script"):
            print("\nWorkflow continued, but errors occurred during rule checking.", file=sys.stderr)
            # Decide whether to stop or continue if rules fail
            # sys.exit(1)
    
    # Machine Learning Steps
    if run_ml:
        # Step 3: Run ML Model 1 (PCA) Prediction
        print("\n--- Step 3: Running ML Model 1 (PCA) Prediction ---")
        transactions_file = Path("./data/simulation_output/simulated_account_transactions.csv")
        if transactions_file.exists():
            # Run the PCA-based anomaly detection model
            try:
                run_script(MODEL_PCA_PREDICT_SCRIPT, script_desc="PCA-based Fraud Detection")
            except Exception as e:
                print(f"  Warning: Could not run PCA model: {e}.")
                print("  You may need to install scikit-learn with: pip install scikit-learn")
        else:
            print(f"   (Skipping - Simulated transactions file not found at {transactions_file})")
            if args.ml_only:
                print("   Run the simulation step first or provide transaction data.")

        # Step 4: Run ML Model 2 (Simulated Features) Prediction
        print("\n--- Step 4: Running ML Model 2 (Full Features) Prediction ---")
        if transactions_file.exists():
            # Run the simulated features-based prediction model
            try:
                run_script(MODEL_SIMULATED_PREDICT_SCRIPT, script_desc="Simulated Features Fraud Detection")
            except Exception as e:
                print(f"  Warning: Could not run Simulated Features model: {e}.")
                print("  You may need to install scikit-learn with: pip install scikit-learn")
        else:
            print(f"   (Skipping - Simulated transactions file not found at {transactions_file})")


    # Step 5: Run Merchant Checker
    # If a specific merchant was provided via command line
    if args.check_merchant:
        print(f"\n--- Step 5: Checking Specific Merchant: '{args.check_merchant}' ---")
        run_script(MERCHANT_CHECKER_SCRIPT, [args.check_merchant], script_desc=f"Merchant Check: {args.check_merchant}")
    # Otherwise run the standard examples
    else:
        print("\n--- Step 5 (Optional): Running Merchant Checker Example ---")
        # Example: Check a known legit name and a potentially fake one
        # Make sure 'Your Bank Inc' is in your data/legit_companies.txt for this to work
        run_script(MERCHANT_CHECKER_SCRIPT, ["Your Bank Inc"], script_desc="Merchant Checker (Legit)")
        run_script(MERCHANT_CHECKER_SCRIPT, ["fraud_Definitely_Fake_LLC"], script_desc="Merchant Checker (Fake)")
        
        # Check merchants from actual transactions if available
        transactions_file = Path("./data/simulation_output/simulated_account_transactions.csv")
        if transactions_file.exists() and pd.__version__ != "2.0.0":
            try:
                import pandas as pd
                print("\n--- Checking Random Merchants from Simulated Transactions ---")
                # Load transactions and sample a few merchants
                trans_df = pd.read_csv(transactions_file)
                if 'merchant' in trans_df.columns:
                    # Get up to 3 random merchants
                    sample_merchants = trans_df['merchant'].sample(min(3, len(trans_df))).tolist()
                    for merchant in sample_merchants:
                        run_script(MERCHANT_CHECKER_SCRIPT, [merchant], 
                                 script_desc=f"Transaction Merchant: {merchant}")
            except Exception as e:
                print(f"Could not analyze transaction merchants: {e}")
    # --- End Merchant Checking ---

    workflow_end_time = time.time()
    print("\n=============================================")
    print(f"=== Workflow Finished in {workflow_end_time - workflow_start_time:.2f} seconds ===")
    print("=============================================")

if __name__ == "__main__":
    main()
