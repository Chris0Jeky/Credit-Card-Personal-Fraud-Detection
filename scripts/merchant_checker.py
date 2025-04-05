import argparse
import sys
from pathlib import Path

# Simple fuzzy matching (optional, requires 'pip install thefuzz')
# from thefuzz import process

LEGIT_COMPANIES_FILE = Path(__file__).parent.parent / "data" / "legit_companies.txt"
# Adjust path relative to where you place the script if needed

def load_legit_companies(filepath: Path) -> set:
    """Loads legitimate company names from a file into a set."""
    if not filepath.is_file():
        print(f"Error: Legit companies file not found at {filepath}", file=sys.stderr)
        return set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # Read lines, strip whitespace, ignore empty lines
            companies = {line.strip() for line in f if line.strip()}
        print(f"Loaded {len(companies)} legitimate company names.")
        return companies
    except Exception as e:
        print(f"Error loading file {filepath}: {e}", file=sys.stderr)
        return set()

def check_merchant(merchant_name: str, legit_companies: set) -> bool:
    """
    Checks if the merchant name is in the set of legitimate companies.
    Starts with a simple exact match.
    """
    # Basic check: Exact match (case-insensitive)
    return merchant_name.strip().lower() in {c.lower() for c in legit_companies}

    # --- Optional: Fuzzy Matching Example ---
    # if not legit_companies:
    #     return False # Or handle error appropriately
    # # Find the best match above a certain threshold (e.g., 85)
    # best_match = process.extractOne(merchant_name, legit_companies, score_cutoff=85)
    # return best_match is not None
    # --- End Optional ---


def main():
    """Main function to parse arguments and perform the check."""
    parser = argparse.ArgumentParser(description="Check if a merchant name exists in a list of known legitimate companies.")
    parser.add_argument("merchant_name", type=str, help="The name of the merchant to check.")
    # Add other arguments if needed (e.g., path to company list, matching threshold)

    args = parser.parse_args()

    # Load the list of legit companies
    legit_company_set = load_legit_companies(LEGIT_COMPANIES_FILE)

    if not legit_company_set:
        print("Could not proceed without a list of legitimate companies.", file=sys.stderr)
        sys.exit(1)

    # Perform the check
    is_legit = check_merchant(args.merchant_name, legit_company_set)

    # Output the result
    if is_legit:
        print(f"Result: Merchant '{args.merchant_name}' appears to be on the known legitimate list.")
    else:
        print(f"Result: Merchant '{args.merchant_name}' NOT found on the known legitimate list.")
        # You could add fuzzy match suggestions here if using that option

if __name__ == "__main__":
    main()