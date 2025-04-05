import argparse
import sys
from pathlib import Path

# Simple fuzzy matching 
try:
    from thefuzz import process
    FUZZY_MATCHING_AVAILABLE = True
except ImportError:
    print("Note: thefuzz library not available. Install with 'pip install thefuzz' for fuzzy matching.")
    FUZZY_MATCHING_AVAILABLE = False

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

def check_merchant(merchant_name: str, legit_companies: set) -> tuple:
    """
    Checks if the merchant name is in the set of legitimate companies.
    Returns a tuple of (is_legit, match_details).
    """
    merchant_name = merchant_name.strip()
    
    # Basic check: Exact match (case-insensitive)
    if merchant_name.lower() in {c.lower() for c in legit_companies}:
        return True, f"Exact match found for '{merchant_name}'"

    # Check for fraud prefix in merchant name
    if merchant_name.lower().startswith('fraud_'):
        return False, f"Suspicious prefix 'fraud_' found in merchant name"
    
    # Fuzzy matching if available
    if FUZZY_MATCHING_AVAILABLE and legit_companies:
        # Find the best match above a certain threshold (e.g., 85)
        threshold = 85  # Minimum similarity score (0-100)
        best_match = process.extractOne(merchant_name, legit_companies, score_cutoff=threshold)
        
        if best_match:
            return True, f"Fuzzy match found: '{best_match[0]}' with {best_match[1]}% similarity"
            
    return False, "No match found in legitimate companies list"


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
    is_legit, details = check_merchant(args.merchant_name, legit_company_set)

    # Output the result
    if is_legit:
        print(f"Result: Merchant '{args.merchant_name}' appears to be LEGITIMATE")
        print(f"Details: {details}")
    else:
        print(f"Result: Merchant '{args.merchant_name}' appears to be SUSPICIOUS")
        print(f"Details: {details}")
        
        # If fuzzy matching is available but no match was found, suggest closest matches
        if FUZZY_MATCHING_AVAILABLE and "No match found" in details:
            print("\nSuggestions for closest matches in our database:")
            suggestions = process.extract(args.merchant_name, legit_company_set, limit=3)
            for name, score in suggestions:
                print(f"  '{name}' (similarity: {score}%)")

if __name__ == "__main__":
    main()