import json
from pathlib import Path

def extract_single_values(input_path, output_path):
    """
    Extracts only single-value parameters from PySAM JSON,
    removing all arrays entirely.
    """
    with open(input_path, 'r') as f:
        data = json.load(f)

    def process_item(item):
        if isinstance(item, dict):
            return {k: process_item(v) for k, v in item.items() if process_item(v) is not None}
        elif isinstance(item, list):
            return None  # Remove all arrays
        return item

    simplified = process_item(data)

    with open(output_path, 'w') as f:
        json.dump(simplified, f, indent=2)

    orig_size = Path(input_path).stat().st_size / 1024
    new_size = Path(output_path).stat().st_size / 1024
    print(f"Single-value config saved to {output_path}")
    print(f"Original size: {orig_size:.1f} KB")
    print(f"New size: {new_size:.1f} KB")
    print(f"Reduction: {(1 - new_size / orig_size) * 100:.1f}%")

# Usage:
input_json = "All_commercial_updated.json"
output_json = "pysam_single_values.json"

extract_single_values(input_json, output_json)