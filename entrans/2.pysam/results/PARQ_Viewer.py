#!/usr/bin/env python3
"""
Parquet File Explorer for PySAM Results
Analyze and visualize battery/solar simulation data
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


def load_parquet_info(file_path):
    """Load parquet file and return basic info"""
    try:
        df = pd.read_parquet(file_path)
        print(f"\n📊 File: {Path(file_path).name}")
        print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 ** 2:.1f} MB")
        return df
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None


def show_columns(df):
    """Display all column names in organized format"""
    print(f"\n📋 Available Columns ({len(df.columns)}):")
    print("=" * 50)

    # Group columns by category for better readability
    battery_cols = [col for col in df.columns if 'batt' in col.lower()]
    solar_cols = [col for col in df.columns if
                  any(x in col.lower() for x in ['poa', 'dc', 'ac', 'inv', 'solar', 'subarray'])]
    grid_cols = [col for col in df.columns if any(x in col.lower() for x in ['grid', 'load', 'utility'])]
    weather_cols = [col for col in df.columns if
                    any(x in col.lower() for x in ['temp', 'wind', 'sun', 'weather', 'gh', 'dn', 'df'])]
    other_cols = [col for col in df.columns if col not in battery_cols + solar_cols + grid_cols + weather_cols]

    categories = [
        ("🔋 Battery", battery_cols),
        ("☀️ Solar/PV", solar_cols),
        ("⚡ Grid/Load", grid_cols),
        ("🌤️ Weather", weather_cols),
        ("📈 Other", other_cols)
    ]

    for category, cols in categories:
        if cols:
            print(f"\n{category} ({len(cols)}):")
            for i, col in enumerate(cols, 1):
                print(f"  {i:2d}. {col}")


def get_summary_stats(df):
    """Generate summary statistics for all numeric columns"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) == 0:
        print("No numeric columns found")
        return None

    print(f"\n📊 Summary Statistics for {len(numeric_cols)} numeric columns:")
    print("=" * 80)

    stats = df[numeric_cols].describe()

    # Add additional useful stats
    additional_stats = pd.DataFrame({
        'non_zero_count': (df[numeric_cols] != 0).sum(),
        'zero_count': (df[numeric_cols] == 0).sum(),
        'missing_count': df[numeric_cols].isnull().sum()
    }).T

    combined_stats = pd.concat([stats, additional_stats])

    # Display in chunks to avoid overwhelming output
    cols_per_chunk = 5
    for i in range(0, len(numeric_cols), cols_per_chunk):
        chunk_cols = numeric_cols[i:i + cols_per_chunk]
        print(f"\nColumns {i + 1}-{min(i + cols_per_chunk, len(numeric_cols))}:")
        print(combined_stats[chunk_cols].round(3))
        print("-" * 80)

    return combined_stats


def plot_column(df, column_name):
    """Create visualization for selected column"""
    if column_name not in df.columns:
        print(f"❌ Column '{column_name}' not found")
        return

    data = df[column_name].dropna()

    if len(data) == 0:
        print(f"❌ No data available for '{column_name}'")
        return

    # Determine if data is time series based on index
    is_time_series = len(data) > 100  # Assume time series if many data points

    plt.style.use('seaborn-v0_8' if hasattr(plt.style, 'seaborn-v0_8') else 'default')

    if is_time_series:
        # Time series plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Analysis: {column_name}', fontsize=16, fontweight='bold')

        # Line plot
        axes[0, 0].plot(data.values, linewidth=0.8, alpha=0.8)
        axes[0, 0].set_title('Time Series')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].grid(True, alpha=0.3)

        # Histogram
        axes[0, 1].hist(data, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Distribution')
        axes[0, 1].set_xlabel('Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)

        # Box plot
        axes[1, 0].boxplot(data, vert=True)
        axes[1, 0].set_title('Box Plot')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].grid(True, alpha=0.3)

        # Rolling statistics (if enough data)
        if len(data) > 24:
            window = min(24, len(data) // 10)  # Daily window for hourly data
            rolling_mean = data.rolling(window=window, center=True).mean()
            rolling_std = data.rolling(window=window, center=True).std()

            axes[1, 1].plot(data.values, alpha=0.3, label='Original', linewidth=0.5)
            axes[1, 1].plot(rolling_mean.values, label=f'{window}-period Moving Avg', linewidth=2)
            axes[1, 1].fill_between(range(len(data)),
                                    (rolling_mean - rolling_std).values,
                                    (rolling_mean + rolling_std).values,
                                    alpha=0.2, label='±1 Std Dev')
            axes[1, 1].set_title(f'Rolling Statistics (Window: {window})')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Not enough data\nfor rolling stats',
                            ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Rolling Statistics')

    else:
        # Simple plot for smaller datasets
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Analysis: {column_name}', fontsize=16, fontweight='bold')

        # Line/bar plot
        if len(data) <= 50:
            axes[0].bar(range(len(data)), data.values, alpha=0.7)
        else:
            axes[0].plot(data.values, marker='o', linewidth=1, markersize=3)
        axes[0].set_title('Data Plot')
        axes[0].set_xlabel('Index')
        axes[0].set_ylabel('Value')
        axes[0].grid(True, alpha=0.3)

        # Histogram
        axes[1].hist(data, bins=min(30, len(data.unique())), alpha=0.7, edgecolor='black')
        axes[1].set_title('Distribution')
        axes[1].set_xlabel('Value')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print summary statistics for the column
    print(f"\n📊 Statistics for '{column_name}':")
    print(f"  Count: {len(data):,}")
    print(f"  Mean: {data.mean():.3f}")
    print(f"  Std: {data.std():.3f}")
    print(f"  Min: {data.min():.3f}")
    print(f"  Max: {data.max():.3f}")
    print(f"  Non-zero values: {(data != 0).sum():,} ({(data != 0).mean() * 100:.1f}%)")


def main():
    """Main interactive function"""
    print("🔍 Parquet File Explorer for PySAM Results")
    print("=" * 50)

    # Get file path
    while True:
        file_path = input("\n📁 Enter parquet file path (or 'quit' to exit): ").strip()

        if file_path.lower() == 'quit':
            print("👋 Goodbye!")
            return

        if not Path(file_path).exists():
            print(f"❌ File not found: {file_path}")
            continue

        # Load file
        df = load_parquet_info(file_path)
        if df is None:
            continue

        while True:
            print("\n🎯 What would you like to do?")
            print("1. Show all columns")
            print("2. Show summary statistics")
            print("3. Plot a specific column")
            print("4. Load different file")
            print("5. Quit")

            choice = input("\nEnter choice (1-5): ").strip()

            if choice == '1':
                show_columns(df)

            elif choice == '2':
                get_summary_stats(df)

            elif choice == '3':
                column_name = input("\n📊 Enter column name to plot: ").strip()
                plot_column(df, column_name)

            elif choice == '4':
                break

            elif choice == '5':
                print("👋 Goodbye!")
                return

            else:
                print("❌ Invalid choice. Please enter 1-5.")


if __name__ == "__main__":
    main()