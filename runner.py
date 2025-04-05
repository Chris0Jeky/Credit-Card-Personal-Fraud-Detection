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