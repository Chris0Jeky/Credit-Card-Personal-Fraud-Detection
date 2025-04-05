import subprocess
import sys
from pathlib import Path
import time

# --- Script Paths ---
# Use Path objects for better cross-platform compatibility
ROOT_DIR = Path(__file__).parent # Assumes runner.py is in the project root
SIMULATE_SCRIPT = ROOT_DIR / "simulate_entities.py"
CHECK_RULES_SCRIPT = ROOT_DIR / "check_rules.py"
MERCHANT_CHECKER_SCRIPT = ROOT_DIR / "merchant_checker.py"