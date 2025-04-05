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

    