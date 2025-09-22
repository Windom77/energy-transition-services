#!/usr/bin/env python3
"""
Parquet to CSV Converter Script
Converts PySAM parquet output files to CSV for analysis
"""

import pandas as pd
import sys
import os
from pathlib import Path
import argparse


def convert_parquet_to_csv(parquet_file, output_file=None, show_preview=True):
    """
    Convert a parquet file to CSV

    Args:
        parquet_file (str): Path to input parquet file
        output_file (str): Path to output CSV file (optional)
        show_preview (bool): Whether to show data preview
    """
    try:
        # Read the parquet file
        print(f"📂 Reading parquet file: {parquet_file}")
        df = pd.read_parquet(parquet_file)

        # Show basic info
        print(f"✅ Loaded successfully!")
        print(f"   📊 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"   📅 Columns: {list(df.columns)}")

        # Show preview if requested
        if show_preview:
            print(f"\n📋 DATA PREVIEW:")
            print(f"   First 5 rows:")
            pd.set_option('display.max_columns', 10)
            pd.set_option('display.width', 120)
            print(df.head())

            # Show basic statistics for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                print(f"\n📈 BASIC STATISTICS (first few numeric columns):")
                print(df[numeric_cols[:5]].describe())

        # Generate output filename if not provided
        if output_file is None:
            parquet_path = Path(parquet_file)
            output_file = parquet_path.with_suffix('.csv')

        # Write to CSV
        print(f"\n💾 Writing to CSV: {output_file}")
        df.to_csv(output_file, index=False)
        print(f"✅ Conversion complete!")

        return df

    except Exception as e:
        print(f"❌ Error converting file: {e}")
        return None


def list_parquet_files(directory):
    """List all parquet files in a directory"""
    parquet_files = list(Path(directory).glob("*.parquet"))
    if parquet_files:
        print(f"📁 Found {len(parquet_files)} parquet files in {directory}:")
        for i, file in enumerate(parquet_files, 1):
            print(f"   {i}. {file.name}")
        return parquet_files
    else:
        print(f"❌ No parquet files found in {directory}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Convert PySAM parquet files to CSV")
    parser.add_argument("input", nargs="?", help="Input parquet file path")
    parser.add_argument("-o", "--output", help="Output CSV file path")
    parser.add_argument("-d", "--directory", help="List parquet files in directory")
    parser.add_argument("--no-preview", action="store_true", help="Skip data preview")

    args = parser.parse_args()

    # If directory specified, list files
    if args.directory:
        list_parquet_files(args.directory)
        return

    # If no input file specified, try to find parquet files in current directory
    if not args.input:
        print("🔍 No input file specified. Looking for parquet files...")

        # Check for results directory
        results_dir = Path("results")
        if results_dir.exists():
            parquet_files = list_parquet_files(results_dir)
        else:
            parquet_files = list_parquet_files(".")

        if not parquet_files:
            print("❌ No parquet files found. Please specify an input file.")
            print("Usage: python parquet_to_csv.py <input_file.parquet>")
            return

        # If only one file, use it automatically
        if len(parquet_files) == 1:
            args.input = str(parquet_files[0])
            print(f"📋 Using: {args.input}")
        else:
            # Let user choose
            try:
                choice = int(input(f"\nSelect file (1-{len(parquet_files)}): ")) - 1
                if 0 <= choice < len(parquet_files):
                    args.input = str(parquet_files[choice])
                else:
                    print("❌ Invalid selection")
                    return
            except (ValueError, KeyboardInterrupt):
                print("\n❌ Selection cancelled")
                return

    # Convert the file
    if args.input:
        df = convert_parquet_to_csv(
            args.input,
            args.output,
            show_preview=not args.no_preview
        )

        # Special handling for battery SOC data
        if df is not None and 'batt_SOC_year1' in df.columns:
            print(f"\n🔋 BATTERY SOC ANALYSIS:")
            soc_data = df['batt_SOC_year1']
            print(f"   📊 SOC Statistics:")
            print(f"      Min: {soc_data.min():.3f}%")
            print(f"      Max: {soc_data.max():.3f}%")
            print(f"      Mean: {soc_data.mean():.3f}%")
            print(f"      Std: {soc_data.std():.3f}%")
            print(f"   🎯 Expected Range: 15-95% (typical Li-ion)")

            if soc_data.max() < 10:
                print(f"   🚨 WARNING: Max SOC is {soc_data.max():.1f}% - Battery configuration issue!")
            elif soc_data.mean() < 20:
                print(f"   ⚠️  LOW: Average SOC is {soc_data.mean():.1f}% - Check dispatch strategy")
            else:
                print(f"   ✅ SOC values appear normal")


if __name__ == "__main__":
    main()


# Alternative: Simple interactive version
def simple_convert():
    """Simple interactive conversion"""
    print("🔄 PySAM Parquet to CSV Converter")
    print("=" * 40)

    # Get input file
    input_file = input("📂 Enter parquet file path (or press Enter to browse current directory): ").strip()

    if not input_file:
        # List available files
        parquet_files = list(Path(".").glob("*.parquet"))
        if not parquet_files:
            results_dir = Path("results")
            if results_dir.exists():
                parquet_files = list(results_dir.glob("*.parquet"))

        if parquet_files:
            print("\n📁 Available parquet files:")
            for i, file in enumerate(parquet_files, 1):
                print(f"   {i}. {file}")

            try:
                choice = int(input(f"\nSelect file (1-{len(parquet_files)}): ")) - 1
                input_file = str(parquet_files[choice])
            except (ValueError, IndexError):
                print("❌ Invalid selection")
                return
        else:
            print("❌ No parquet files found")
            return

    # Convert
    convert_parquet_to_csv(input_file)

# Uncomment this line to run the simple interactive version instead:
# simple_convert()