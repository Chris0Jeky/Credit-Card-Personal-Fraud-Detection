# runner.py

import subprocess
import sys
import argparse
from pathlib import Path
import time
import os

# Import configuration AFTER ensuring config.py exists
try:
    import config
except ModuleNotFoundError:
    print("CRITICAL ERROR: config.py not found in the project root.", file=sys.stderr)
    print("Please create config.py before running the workflow.", file=sys.stderr)
    sys.exit(1)

# --- Script Paths from Config ---
SIMULATE_SCRIPT = config.SIMULATE_SCRIPT
CHECK_RULES_SCRIPT = config.CHECK_RULES_SCRIPT
MERCHANT_CHECKER_SCRIPT = config.MERCHANT_CHECKER_SCRIPT
ANALYZE_FLAGS_SCRIPT = config.ANALYZE_FLAGS_SCRIPT
VISUALIZE_SCRIPT = config.VISUALIZE_SCRIPT
EVALUATE_SCRIPT = config.EVALUATE_SCRIPT
INTEGRATE_SCRIPT = config.INTEGRATE_SCRIPT
MODEL_PCA_PREDICT_SCRIPT = config.MODEL_PCA_PREDICT_SCRIPT
MODEL_RF_PREDICT_SCRIPT = config.MODEL_RF_PREDICT_SCRIPT
# ---

def run_script(script_path, args=[], script_desc="script"):
    """Runs a python script using subprocess and checks for errors."""
    if not script_path.is_file():
        print(f"Error: {script_desc} not found at {script_path}", file=sys.stderr)
        return False

    command = [sys.executable, str(script_path)] + args
    print(f"\n>>> Running {script_desc}: {' '.join(map(str, command))}") # Ensure args are strings
    start_time = time.time()
    # Use text=True for automatic encoding/decoding
    # Use capture_output=False to see the script's output in real-time
    result = subprocess.run(command, text=True, capture_output=False, check=False)
    end_time = time.time()
    print(f"<<< Finished {script_desc} in {end_time - start_time:.2f} seconds. Exit code: {result.returncode}")

    if result.returncode != 0:
        print(f"Error running {script_path.name}. Check output above.", file=sys.stderr)
        # Consider adding more specific error handling if needed
        return False
    return True

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Credit Card Fraud Detection Workflow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Execution control
    parser.add_argument("--simulate-only", action="store_true", help="Run only the simulation step")
    parser.add_argument("--check-only", action="store_true", help="Run only rule checking")
    parser.add_argument("--ml-only", action="store_true", help="Run only ML model predictions (PCA & RF)")
    parser.add_argument("--integrate-only", action="store_true", help="Run only results integration")
    parser.add_argument("--evaluate-only", action="store_true", help="Run only performance evaluation")
    parser.add_argument("--analyze-only", action="store_true", help="Run only flag analysis")
    parser.add_argument("--visualize-only", action="store_true", help="Run only visualization")
    parser.add_argument("--full-run", action="store_true", help="Run all steps (default if no specific step chosen)")

    # Simulation parameters
    parser.add_argument(
        "--transactions", type=int, default=config.DEFAULT_NUM_TRANSACTIONS,
        help="Number of transactions to simulate"
    )

    # Merchant check
    parser.add_argument(
        "--check-merchant", type=str,
        help="Check a specific merchant name against the legitimate list (runs independently)"
    )

    # Analysis & Visualization parameters
    parser.add_argument(
        "--detailed-analysis", action="store_true",
        help="Include detailed statistics in flag analysis (if analyze step runs)"
    )
    parser.add_argument(
        "--viz-format", type=str, default=config.DEFAULT_VIZ_FORMAT,
        choices=["png", "svg", "pdf", "jpg"],
        help="Format for visualization files (if visualize step runs)"
    )
    parser.add_argument(
        "--plot-eval", action="store_true",
        help="Generate PR curve plots during evaluation (if evaluate step runs)"
    )

    return parser.parse_args()

def main():
    """Orchestrates the simulation, checking, ML, integration, evaluation, analysis, and viz process."""
    args = parse_arguments()

    # Override config transaction number if provided via command line
    num_transactions = args.transactions
    print(f"--- Workflow Configuration ---")
    print(f"Number of Transactions to Simulate: {num_transactions}")
    print(f"Visualization Format: {args.viz_format}")
    print(f"Plot Evaluation Curves: {args.plot_eval}")
    print(f"Detailed Analysis: {args.detailed_analysis}")
    # We'll pass num_transactions as an argument to the simulation script now

    # --- Special Case: Check Merchant ---
    if args.check_merchant:
        print(f"\n--- Running Standalone Merchant Check: '{args.check_merchant}' ---")
        run_script(MERCHANT_CHECKER_SCRIPT, [args.check_merchant], script_desc="Merchant Check")
        sys.exit(0) # Exit after merchant check

    # --- Determine Workflow Steps ---
    run_all = args.full_run or not any([
        args.simulate_only, args.check_only, args.ml_only, args.integrate_only,
        args.evaluate_only, args.analyze_only, args.visualize_only
    ])

    run_simulate = run_all or args.simulate_only
    run_check = run_all or args.check_only
    run_ml = run_all or args.ml_only
    run_integrate = run_all or args.integrate_only
    run_evaluate = run_all or args.evaluate_only
    run_analyze = run_all or args.analyze_only # Flag analysis focuses on rules now
    run_visualize = run_all or args.visualize_only

    print("\n--- Workflow Steps Activated ---")
    print(f"Simulate Data: {run_simulate}")
    print(f"Check Rules: {run_check}")
    print(f"Run ML Models (PCA, RF): {run_ml}")
    print(f"Integrate Results: {run_integrate}")
    print(f"Evaluate Performance: {run_evaluate}")
    print(f"Analyze Rule Flags: {run_analyze}")
    print(f"Generate Visualizations: {run_visualize}")
    print("=" * 30)

    workflow_start_time = time.time()
    pipeline_successful = True

    # --- Execute Pipeline Steps ---

    # Step 1: Simulate Data
    if run_simulate:
        print("\n--- Step 1: Generating Simulated Data ---")
        sim_args = ["--transactions", str(num_transactions)] # Pass count as arg
        if not run_script(SIMULATE_SCRIPT, sim_args, script_desc="Simulation"):
            pipeline_successful = False
            print("\nWorkflow aborted due to critical error in simulation.", file=sys.stderr)
            sys.exit(1) # Simulation is fundamental

    # Step 2: Check Rules
    if run_check and pipeline_successful:
        print("\n--- Step 2: Applying Rule-Based Checks ---")
        if not run_script(CHECK_RULES_SCRIPT, script_desc="Rule Checking"):
            pipeline_successful = False
            print("\nWarning: Errors occurred during rule checking. Subsequent steps might fail.", file=sys.stderr)
            # Decide whether to stop: Let's continue but warn

    # Step 3: Run ML Models
    if run_ml and pipeline_successful:
        print("\n--- Step 3: Running ML Models ---")
        # Check if input file exists (should be created by simulation)
        if not config.TRANSACTIONS_FILE.exists():
             print(f"Error: Input file {config.TRANSACTIONS_FILE.name} not found. Cannot run ML models.", file=sys.stderr)
             pipeline_successful = False
        else:
            # Run ML Model 1 (PCA/IF)
            print("\n   --- Running ML Model 1 (PCA / Isolation Forest) ---")
            if not run_script(MODEL_PCA_PREDICT_SCRIPT, script_desc="PCA/IF Prediction"):
                print("Warning: PCA/IF model script failed. Integration/Evaluation might be incomplete.", file=sys.stderr)
                # Don't necessarily stop the whole pipeline

            # Run ML Model 2 (RF)
            print("\n   --- Running ML Model 2 (Random Forest) ---")
            if not run_script(MODEL_RF_PREDICT_SCRIPT, script_desc="Random Forest Prediction"):
                 print("Warning: Random Forest model script failed. Integration/Evaluation might be incomplete.", file=sys.stderr)
                 # Don't necessarily stop

    # Step 4: Integrate Results
    if run_integrate and pipeline_successful:
        print("\n--- Step 4: Integrating Rule and ML Results ---")
         # Check if inputs exist (rules + *at least one* ML model output)
        rule_file_ok = config.FLAGGED_TRANSACTIONS_FILE.exists()
        pca_file_ok = config.PCA_PREDICTIONS_FILE.exists()
        rf_file_ok = config.RF_PREDICTIONS_FILE.exists()
        if not rule_file_ok:
             print(f"Error: Rule results file {config.FLAGGED_TRANSACTIONS_FILE.name} not found. Cannot integrate.", file=sys.stderr)
             pipeline_successful = False
        elif not (pca_file_ok and rf_file_ok): # Require both for standard integration
             print(f"Warning: One or both ML prediction files missing. Integration may be incomplete.", file=sys.stderr)
             # Attempt to run anyway, integrate_results.py should handle missing files gracefully
             if not run_script(INTEGRATE_SCRIPT, script_desc="Results Integration"):
                 pipeline_successful = False
                 print("Warning: Results integration failed.", file=sys.stderr)
        elif not run_script(INTEGRATE_SCRIPT, script_desc="Results Integration"):
             pipeline_successful = False
             print("Warning: Results integration failed.", file=sys.stderr)


    # Step 5: Evaluate Performance
    if run_evaluate and pipeline_successful:
        print("\n--- Step 5: Evaluating Performance ---")
         # Check if inputs exist
        rule_file_ok = config.FLAGGED_TRANSACTIONS_FILE.exists()
        pca_file_ok = config.PCA_PREDICTIONS_FILE.exists()
        rf_file_ok = config.RF_PREDICTIONS_FILE.exists()
        if not (rule_file_ok and pca_file_ok and rf_file_ok):
            print(f"Warning: One or more required files for evaluation missing. Evaluation may be incomplete.", file=sys.stderr)
            # Proceed if possible, evaluate_performance.py should handle missing components
        eval_args = []
        if args.plot_eval:
            eval_args.append("--plot")
        if not run_script(EVALUATE_SCRIPT, eval_args, script_desc="Performance Evaluation"):
             pipeline_successful = False # Evaluation failure might be significant
             print("Warning: Performance evaluation failed.", file=sys.stderr)

    # Step 6: Analyze Rule Flags
    if run_analyze and pipeline_successful:
        print("\n--- Step 6: Analyzing Rule Flags ---")
        if not config.FLAGGED_TRANSACTIONS_FILE.exists():
             print(f"Error: Flagged transactions file {config.FLAGGED_TRANSACTIONS_FILE.name} not found. Cannot analyze.", file=sys.stderr)
        else:
            analysis_args = []
            if args.detailed_analysis:
                analysis_args.append("--detailed")
            if not run_script(ANALYZE_FLAGS_SCRIPT, analysis_args, script_desc="Flag Analysis"):
                 print("Warning: Flag analysis script failed.", file=sys.stderr)
                 # Usually not critical to stop pipeline

    # Step 7: Generate Visualizations
    if run_visualize and pipeline_successful:
        print("\n--- Step 7: Generating Data Visualizations ---")
        # Check if *any* data files exist to visualize
        data_files_exist = any([
            config.TRANSACTIONS_FILE.exists(),
            config.FLAGGED_TRANSACTIONS_FILE.exists(),
            config.PCA_PREDICTIONS_FILE.exists(),
            config.RF_PREDICTIONS_FILE.exists(),
            config.INTEGRATED_ASSESSMENT_FILE.exists(),
            config.EVALUATION_REPORT_FILE.exists()
        ])
        if not data_files_exist:
            print("Warning: No data files found to visualize. Skipping visualization.", file=sys.stderr)
        else:
            viz_args = ["--format", args.viz_format]
            # Pass paths to existing files dynamically
            if config.TRANSACTIONS_FILE.exists(): viz_args.extend(["--transactions", str(config.TRANSACTIONS_FILE)])
            if config.FLAGGED_TRANSACTIONS_FILE.exists(): viz_args.extend(["--flagged", str(config.FLAGGED_TRANSACTIONS_FILE)])
            if config.PCA_PREDICTIONS_FILE.exists(): viz_args.extend(["--pca", str(config.PCA_PREDICTIONS_FILE)])
            if config.RF_PREDICTIONS_FILE.exists(): viz_args.extend(["--simulated", str(config.RF_PREDICTIONS_FILE)]) # Keep arg name consistent with README
            if config.INTEGRATED_ASSESSMENT_FILE.exists(): viz_args.extend(["--integrated", str(config.INTEGRATED_ASSESSMENT_FILE)])
            if config.EVALUATION_REPORT_FILE.exists(): viz_args.extend(["--evaluation", str(config.EVALUATION_REPORT_FILE)])

            if not run_script(VISUALIZE_SCRIPT, viz_args, script_desc="Data Visualization"):
                print("\nWarning: Visualization script failed.", file=sys.stderr)
            else:
                print(f"\nVisualization report created in {config.VISUALIZATIONS_DIR}")

    # --- Optional: Merchant Check Examples (if not checking specific merchant) ---
    if not args.check_merchant and run_all: # Only run examples on a full default run
        print("\n--- Step 8 (Optional): Running Merchant Checker Examples ---")
        # Example: Check a known legit name and a potentially fake one
        run_script(MERCHANT_CHECKER_SCRIPT, ["Amazon"], script_desc="Merchant Check (Example Legit)")
        run_script(MERCHANT_CHECKER_SCRIPT, ["fraud_Definitely_Fake_LLC"], script_desc="Merchant Check (Example Fake)")
        # Check merchants from actual transactions if available
        if config.TRANSACTIONS_FILE.exists():
            try:
                import pandas as pd # Local import to avoid dependency if only checking merchant
                print("\n--- Checking Random Merchants from Simulated Transactions ---")
                trans_df = pd.read_csv(config.TRANSACTIONS_FILE)
                if 'merchant' in trans_df.columns and not trans_df.empty:
                    sample_merchants = trans_df['merchant'].dropna().sample(min(3, len(trans_df['merchant'].dropna()))).tolist()
                    for merchant in sample_merchants:
                        run_script(MERCHANT_CHECKER_SCRIPT, [merchant],
                                     script_desc=f"Transaction Merchant: {merchant}")
            except ImportError:
                 print("   (Skipping transaction merchant check: pandas not found)")
            except Exception as e:
                 print(f"   (Skipping transaction merchant check: error - {e})")

    # --- Workflow Summary ---
    workflow_end_time = time.time()
    print("\n=============================================")
    status = "Finished Successfully" if pipeline_successful else "Finished with ERRORS"
    print(f"=== Workflow {status} in {workflow_end_time - workflow_start_time:.2f} seconds ===")
    print("=============================================")

    if not pipeline_successful:
        sys.exit(1) # Exit with error code if any critical step failed

if __name__ == "__main__":
    main()