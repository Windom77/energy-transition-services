from pathlib import Path
import pandas as pd
import os


def check_results_structure():
    """Check what result files exist and their structure"""

    print("🔍 RESULTS DIAGNOSTIC CHECK")
    print("=" * 50)

    # Check current working directory
    print(f"📁 Current working directory: {os.getcwd()}")

    # Define paths
    main_results_dir = Path("2.pysam/results")
    standalone_results_dir = Path("4.testing/results")

    print(f"\n📂 Main results directory: {main_results_dir.absolute()}")
    print(f"   Exists: {main_results_dir.exists()}")

    print(f"\n📂 Standalone results directory: {standalone_results_dir.absolute()}")
    print(f"   Exists: {standalone_results_dir.exists()}")

    # Check main results
    print(f"\n🔍 MAIN RESULTS DIRECTORY CONTENTS:")
    if main_results_dir.exists():
        files = list(main_results_dir.glob("*"))
        if files:
            for file in sorted(files):
                print(f"   📄 {file.name} ({file.stat().st_size} bytes)")

                # Check file structure for key files
                if file.name == "scalar_results.csv":
                    try:
                        df = pd.read_csv(file, index_col=0)
                        print(f"      → {len(df)} rows, columns: {list(df.columns)}")

                        # Show a few sample metrics
                        battery_metrics = [idx for idx in df.index if 'batt' in str(idx).lower()]
                        if battery_metrics:
                            print(f"      → Battery metrics found: {len(battery_metrics)}")
                            for metric in battery_metrics[:3]:  # Show first 3
                                print(f"         • {metric}: {df.loc[metric, 'Value']}")
                        else:
                            print(f"      → No battery metrics found")

                    except Exception as e:
                        print(f"      → Error reading CSV: {e}")
        else:
            print("   (Empty directory)")
    else:
        print("   (Directory does not exist)")

    # Check standalone results
    print(f"\n🔍 STANDALONE RESULTS DIRECTORY CONTENTS:")
    if standalone_results_dir.exists():
        files = list(standalone_results_dir.glob("*"))
        if files:
            for file in sorted(files):
                print(f"   📄 {file.name} ({file.stat().st_size} bytes)")

                # Check file structure for key files
                if file.name == "scalar_results.csv":
                    try:
                        df = pd.read_csv(file, index_col=0)
                        print(f"      → {len(df)} rows, columns: {list(df.columns)}")

                        # Show a few sample metrics
                        battery_metrics = [idx for idx in df.index if 'batt' in str(idx).lower()]
                        if battery_metrics:
                            print(f"      → Battery metrics found: {len(battery_metrics)}")
                            for metric in battery_metrics[:3]:  # Show first 3
                                print(f"         • {metric}: {df.loc[metric, 'Value']}")
                        else:
                            print(f"      → No battery metrics found")

                    except Exception as e:
                        print(f"      → Error reading CSV: {e}")
        else:
            print("   (Empty directory)")
    else:
        print("   (Directory does not exist)")

    # Try to find common battery metrics
    print(f"\n🔋 BATTERY METRICS COMPARISON:")
    try:
        main_scalar_path = main_results_dir / "scalar_results.csv"
        standalone_scalar_path = standalone_results_dir / "scalar_results.csv"

        if main_scalar_path.exists() and standalone_scalar_path.exists():
            main_df = pd.read_csv(main_scalar_path, index_col=0)
            standalone_df = pd.read_csv(standalone_scalar_path, index_col=0)

            # Find battery metrics in both
            main_battery = [idx for idx in main_df.index if 'batt' in str(idx).lower()]
            standalone_battery = [idx for idx in standalone_df.index if 'batt' in str(idx).lower()]

            common_battery = set(main_battery) & set(standalone_battery)

            print(f"   Main model battery metrics: {len(main_battery)}")
            print(f"   Standalone battery metrics: {len(standalone_battery)}")
            print(f"   Common battery metrics: {len(common_battery)}")

            if common_battery:
                print(f"   Common metrics:")
                for metric in sorted(list(common_battery))[:5]:  # Show first 5
                    main_val = main_df.loc[metric, 'Value']
                    standalone_val = standalone_df.loc[metric, 'Value']
                    print(f"      • {metric}: Main={main_val}, Standalone={standalone_val}")
            else:
                print(f"   ⚠️ No common battery metrics found!")
                print(f"   Main sample metrics: {main_df.index[:5].tolist()}")
                print(f"   Standalone sample metrics: {standalone_df.index[:5].tolist()}")
        else:
            print("   ⚠️ Cannot compare - scalar files missing")

    except Exception as e:
        print(f"   ❌ Error during comparison: {e}")

    # Check time series files
    print(f"\n📈 TIME SERIES FILES:")

    # Main monthly
    main_monthly = main_results_dir / "monthly_timeseries.parquet"
    if main_monthly.exists():
        try:
            df = pd.read_parquet(main_monthly)
            battery_cols = [col for col in df.columns if 'batt' in col.lower()]
            print(f"   Main monthly: {len(df)} rows, {len(df.columns)} cols, {len(battery_cols)} battery cols")
        except Exception as e:
            print(f"   Main monthly: Error reading - {e}")
    else:
        print(f"   Main monthly: Not found")

    # Standalone monthly
    standalone_monthly = standalone_results_dir / "monthly_timeseries.csv"
    if standalone_monthly.exists():
        try:
            df = pd.read_csv(standalone_monthly)
            battery_cols = [col for col in df.columns if 'batt' in col.lower()]
            print(f"   Standalone monthly: {len(df)} rows, {len(df.columns)} cols, {len(battery_cols)} battery cols")
        except Exception as e:
            print(f"   Standalone monthly: Error reading - {e}")
    else:
        print(f"   Standalone monthly: Not found")

    print(f"\n✅ DIAGNOSTIC COMPLETE")

    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if not main_results_dir.exists():
        print("   1. Run main PySAM simulation: python 2.pysam/PySAM_main_v2.py")
    if not standalone_results_dir.exists():
        print("   2. Run standalone verification: python 4.testing/standalone_pysam_verification.py")
    if main_results_dir.exists() and standalone_results_dir.exists():
        print("   3. Both result directories exist - comparison should work")
        print("   4. Re-run comparison: python 4.testing/battery_comparison_analysis.py")


if __name__ == "__main__":
    check_results_structure()