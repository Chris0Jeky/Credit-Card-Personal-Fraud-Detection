import subprocess
import sys
from pathlib import Path
import time

# --- Script Paths ---
# Use Path objects for better cross-platform compatibility
ROOT_DIR = Path(__file__).parent # Assumes runner.py is in the project root

SCRIPTS_DIR = ROOT_DIR / "scripts"

SIMULATE_SCRIPT = SCRIPTS_DIR / "simulate_entities.py"
CHECK_RULES_SCRIPT = SCRIPTS_DIR / "check_rules.py"
MERCHANT_CHECKER_SCRIPT = SCRIPTS_DIR / "merchant_checker.py"

# Add paths to future model prediction scripts here
# MODEL_PCA_PREDICT_SCRIPT = ROOT_DIR / "models" / "predict_pca_fraud.py"
# MODEL_SIMULATED_PREDICT_SCRIPT = ROOT_DIR / "models" / "predict_simulated_fraud.py"
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

def main():
    """Orchestrates the simulation and checking process."""
    print("=============================================")
    print("=== Starting Fraud Detection Simulation Workflow ===")
    print("=============================================")
    workflow_start_time = time.time()

    # Step 1: Simulate account and transactions
    print("\n--- Step 1: Generating Simulated Data ---")
    if not run_script(SIMULATE_SCRIPT, script_desc="Simulation Script"):
        print("\nWorkflow aborted due to error in simulation.", file=sys.stderr)
        sys.exit(1)

    # Step 2: Check rules on simulated data
    print("\n--- Step 2: Applying Rule-Based Checks ---")
    if not run_script(CHECK_RULES_SCRIPT, script_desc="Rule Checking Script"):
        print("\nWorkflow continued, but errors occurred during rule checking.", file=sys.stderr)
        # Decide whether to stop or continue if rules fail
        # sys.exit(1)

    # --- Placeholder for Future Steps ---

    # Step 3 (Future): Run ML Model 1 (PCA) Prediction
    # print("\n--- Step 3 (Future): Running ML Model 1 (PCA) Prediction ---")
    # if Path("./data/simulation_output/simulated_account_transactions.csv").exists():
         # You'd need a script that loads the simulated data, preprocesses it like the PCA data,
         # loads the PCA model, makes predictions, and saves/prints them.
         # run_script(MODEL_PCA_PREDICT_SCRIPT, ["./data/simulation_output/simulated_account_transactions.csv"])
    #    print("   (Skipping - Prediction script not implemented yet)")
    # else:
    #    print("   (Skipping - Simulated transactions file not found)")


    # Step 4 (Future): Run ML Model 2 (Simulated Features) Prediction
    # print("\n--- Step 4 (Future): Running ML Model 2 (Full Features) Prediction ---")
    # if Path("./data/simulation_output/simulated_account_transactions.csv").exists():
        # This script would load the simulated data, apply feature engineering matching
        # how Model 2 was trained, load Model 2, predict, and save/print.
        # run_script(MODEL_SIMULATED_PREDICT_SCRIPT, ["./data/simulation_output/simulated_account_transactions.csv"])
    #    print("   (Skipping - Prediction script not implemented yet)")
    # else:
    #    print("   (Skipping - Simulated transactions file not found)")


    # Step 5 (Optional): Run Merchant Checker on a sample
    print("\n--- Step 5 (Optional): Running Merchant Checker Example ---")
    # Example: Check a known legit name and a potentially fake one
    # Make sure 'Your Bank Inc' is in your data/legit_companies.txt for this to work
    run_script(MERCHANT_CHECKER_SCRIPT, ["Your Bank Inc"], script_desc="Merchant Checker (Legit)")
    run_script(MERCHANT_CHECKER_SCRIPT, ["fraud_Definitely_Fake_LLC"], script_desc="Merchant Checker (Fake)")
    # --- End Placeholder ---

    
