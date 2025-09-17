import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import os


def find_project_root():
    """Find the project root directory regardless of where script is run from"""
    current_path = Path(__file__).resolve()

    # Look for project root indicators
    for parent in [current_path.parent] + list(current_path.parents):
        # Check for key project files/directories
        if (parent / 'main.py').exists() or (parent / 'config.py').exists():
            return parent
        if (parent / '2.pysam').exists() and (parent / '1.input').exists():
            return parent

    # Fallback: if running from 4.testing, go up one level
    if current_path.parent.name == '4.testing':
        return current_path.parent.parent

    # Last resort: current working directory
    return Path.cwd()


class PathAwareBatteryComparisonAnalyzer:
    """Analyze differences in battery outputs with automatic path detection"""

    def __init__(self):
        self.project_root = find_project_root()
        self.main_results_dir = self.project_root / "2.pysam" / "results"
        self.standalone_results_dir = self.project_root / "4.testing" / "results"
        self.analysis_results_dir = self.project_root / "4.testing" / "analysis"

        # Create analysis directory
        self.analysis_results_dir.mkdir(parents=True, exist_ok=True)

        # Debug path information
        print(f"🔍 PATH DETECTION:")
        print(f"   Project root: {self.project_root}")
        print(f"   Main results: {self.main_results_dir}")
        print(f"   Standalone results: {self.standalone_results_dir}")
        print(f"   Analysis output: {self.analysis_results_dir}")
        print(f"   Current working dir: {Path.cwd()}")

    def run_analysis(self):
        """Run complete battery comparison analysis"""
        print(f"\n🔋 BATTERY OUTPUT COMPARISON ANALYSIS")
        print(f"=" * 60)

        try:
            # Load data
            main_scalar, main_monthly, main_annual = self._load_main_results()
            standalone_scalar, standalone_monthly, standalone_annual = self._load_standalone_results()

            # Compare scalar metrics
            self._compare_scalar_metrics(main_scalar, standalone_scalar)

            # Compare time series
            if main_monthly is not None and standalone_monthly is not None:
                self._compare_monthly_series(main_monthly, standalone_monthly)

            if main_annual is not None and standalone_annual is not None:
                self._compare_annual_series(main_annual, standalone_annual)

            # Generate summary report
            self._generate_summary_report()

            print(f"\n✅ ANALYSIS COMPLETED")
            print(f"📁 Results saved to: {self.analysis_results_dir}")

        except Exception as e:
            print(f"❌ ANALYSIS FAILED: {e}")
            import traceback
            traceback.print_exc()

    def _load_main_results(self):
        """Load main model results"""
        print(f"\n📂 Loading main model results from: {self.main_results_dir}")

        scalar_path = self.main_results_dir / "scalar_results.csv"
        monthly_path = self.main_results_dir / "monthly_timeseries.parquet"
        annual_path = self.main_results_dir / "annual_timeseries.parquet"

        # Load scalar results
        scalar_df = None
        if scalar_path.exists():
            try:
                scalar_df = pd.read_csv(scalar_path, index_col=0)
                print(f"   ✅ Loaded main scalar results: {len(scalar_df)} metrics")
            except Exception as e:
                print(f"   ❌ Error loading main scalar results: {e}")
        else:
            print(f"   ⚠️ Main scalar results not found at: {scalar_path}")

        # Load monthly results
        monthly_df = None
        if monthly_path.exists():
            try:
                monthly_df = pd.read_parquet(monthly_path)
                print(f"   ✅ Loaded main monthly results: {len(monthly_df.columns)} series")
            except Exception as e:
                print(f"   ❌ Error loading main monthly results: {e}")
        else:
            print(f"   ⚠️ Main monthly results not found at: {monthly_path}")

        # Load annual results
        annual_df = None
        if annual_path.exists():
            try:
                annual_df = pd.read_parquet(annual_path)
                print(f"   ✅ Loaded main annual results: {len(annual_df.columns)} series")
            except Exception as e:
                print(f"   ❌ Error loading main annual results: {e}")
        else:
            print(f"   ⚠️ Main annual results not found at: {annual_path}")

        return scalar_df, monthly_df, annual_df

    def _load_standalone_results(self):
        """Load standalone model results"""
        print(f"\n📂 Loading standalone model results from: {self.standalone_results_dir}")

        scalar_path = self.standalone_results_dir / "scalar_results.csv"
        monthly_path = self.standalone_results_dir / "monthly_timeseries.csv"
        annual_path = self.standalone_results_dir / "annual_timeseries.csv"

        # Load scalar results
        scalar_df = None
        if scalar_path.exists():
            try:
                scalar_df = pd.read_csv(scalar_path, index_col=0)
                print(f"   ✅ Loaded standalone scalar results: {len(scalar_df)} metrics")
            except Exception as e:
                print(f"   ❌ Error loading standalone scalar results: {e}")
        else:
            print(f"   ⚠️ Standalone scalar results not found at: {scalar_path}")

        # Load monthly results
        monthly_df = None
        if monthly_path.exists():
            try:
                monthly_df = pd.read_csv(monthly_path, index_col=0)
                print(f"   ✅ Loaded standalone monthly results: {len(monthly_df.columns)} series")
            except Exception as e:
                print(f"   ❌ Error loading standalone monthly results: {e}")
        else:
            print(f"   ⚠️ Standalone monthly results not found at: {monthly_path}")

        # Load annual results
        annual_df = None
        if annual_path.exists():
            try:
                annual_df = pd.read_csv(annual_path, index_col=0)
                print(f"   ✅ Loaded standalone annual results: {len(annual_df.columns)} series")
            except Exception as e:
                print(f"   ❌ Error loading standalone annual results: {e}")
        else:
            print(f"   ⚠️ Standalone annual results not found at: {annual_path}")

        return scalar_df, monthly_df, annual_df

    def _compare_scalar_metrics(self, main_df, standalone_df):
        """Compare scalar battery metrics"""
        print("\n🔋 COMPARING SCALAR BATTERY METRICS:")

        if main_df is None or standalone_df is None:
            print("   ⚠️ Cannot compare - missing data")
            if main_df is None:
                print("      Main model scalar data missing")
            if standalone_df is None:
                print("      Standalone model scalar data missing")
            return

        # Define battery-related metrics to compare
        battery_metrics = [
            'annual_energy',
            'batt_annual_charge_energy',
            'batt_annual_discharge_energy',
            'average_battery_roundtrip_efficiency',
            'batt_bank_installed_capacity',
            'batt_power_charge_max_kwac',
            'batt_power_discharge_max_kwac',
            'average_battery_conversion_efficiency',
            'batt_system_charge_percent',
            'batt_grid_charge_percent'
        ]

        # Find available metrics
        common_metrics = set(main_df.index) & set(standalone_df.index)
        available_battery_metrics = [m for m in battery_metrics if m in common_metrics]

        # Also find any metrics with 'batt' in the name
        additional_battery_metrics = [m for m in common_metrics if 'batt' in str(m).lower()]
        all_battery_metrics = list(set(available_battery_metrics + additional_battery_metrics))

        if not all_battery_metrics:
            print("   ⚠️ No common battery metrics found")
            print(f"   Main model has {len(main_df)} metrics")
            print(f"   Standalone model has {len(standalone_df)} metrics")
            print(f"   Common metrics: {len(common_metrics)}")

            # Show sample metrics from each
            print(f"   Sample main metrics: {list(main_df.index[:5])}")
            print(f"   Sample standalone metrics: {list(standalone_df.index[:5])}")
            return

        print(f"   Found {len(all_battery_metrics)} battery-related metrics to compare")

        # Create comparison DataFrame
        comparison_data = []

        for metric in all_battery_metrics:
            main_val = main_df.loc[metric, 'Value']
            standalone_val = standalone_df.loc[metric, 'Value']

            # Calculate difference
            try:
                if isinstance(main_val, (int, float)) and isinstance(standalone_val, (int, float)):
                    diff = main_val - standalone_val
                    pct_diff = (diff / standalone_val * 100) if standalone_val != 0 else 'N/A'
                else:
                    diff = 'N/A'
                    pct_diff = 'N/A'
            except:
                diff = 'N/A'
                pct_diff = 'N/A'

            comparison_data.append({
                'Metric': metric,
                'Main_Model': main_val,
                'Standalone_Model': standalone_val,
                'Absolute_Difference': diff,
                'Percentage_Difference': pct_diff
            })

            # Print key differences
            if isinstance(pct_diff, (int, float)):
                if abs(pct_diff) > 5:  # >5% difference
                    print(f"   🚨 {metric}: {pct_diff:.1f}% difference")
                elif abs(pct_diff) > 1:  # >1% difference
                    print(f"   ⚠️ {metric}: {pct_diff:.1f}% difference")
                elif pct_diff != 0:
                    print(f"   ✅ {metric}: {pct_diff:.2f}% difference")
                else:
                    print(f"   ✅ {metric}: Perfect match")

        # Save comparison
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(self.analysis_results_dir / "scalar_battery_comparison.csv", index=False)
        print(f"   💾 Saved scalar comparison: {len(comparison_data)} metrics")

    def _compare_monthly_series(self, main_df, standalone_df):
        """Compare monthly time series (simplified for now)"""
        print("\n📅 COMPARING MONTHLY TIME SERIES:")
        print("   (Monthly comparison implementation - checking data structure)")

        print(f"   Main monthly shape: {main_df.shape}")
        print(f"   Standalone monthly shape: {standalone_df.shape}")

        # Find battery columns
        main_battery_cols = [col for col in main_df.columns if 'batt' in str(col).lower()]
        standalone_battery_cols = [col for col in standalone_df.columns if 'batt' in str(col).lower()]

        print(f"   Main battery columns: {len(main_battery_cols)}")
        print(f"   Standalone battery columns: {len(standalone_battery_cols)}")

        if main_battery_cols:
            print(f"   Sample main battery columns: {main_battery_cols[:3]}")
        if standalone_battery_cols:
            print(f"   Sample standalone battery columns: {standalone_battery_cols[:3]}")

    def _compare_annual_series(self, main_df, standalone_df):
        """Compare annual time series (simplified for now)"""
        print("\n📈 COMPARING ANNUAL TIME SERIES:")
        print("   (Annual comparison implementation - checking data structure)")

        print(f"   Main annual shape: {main_df.shape}")
        print(f"   Standalone annual shape: {standalone_df.shape}")

    def _generate_summary_report(self):
        """Generate a summary report of the comparison"""
        print("\n📋 GENERATING SUMMARY REPORT:")

        report = {
            'analysis_info': {
                'timestamp': datetime.now().isoformat(),
                'purpose': 'Compare battery outputs between main PySAM model and standalone verification',
                'project_root': str(self.project_root),
                'main_results_path': str(self.main_results_dir),
                'standalone_results_path': str(self.standalone_results_dir)
            },
            'path_detection': {
                'script_location': str(Path(__file__).resolve()),
                'working_directory': str(Path.cwd()),
                'project_root_found': str(self.project_root)
            },
            'files_analyzed': {
                'scalar_comparison': 'scalar_battery_comparison.csv'
            },
            'key_findings': [],
            'recommendations': []
        }

        # Check for comparison files and add findings
        scalar_comp_path = self.analysis_results_dir / "scalar_battery_comparison.csv"
        if scalar_comp_path.exists():
            try:
                scalar_df = pd.read_csv(scalar_comp_path)

                # Find significant differences - handle non-numeric values
                numeric_mask = pd.to_numeric(scalar_df['Percentage_Difference'], errors='coerce').notna()
                if numeric_mask.any():
                    numeric_diffs = scalar_df[numeric_mask].copy()
                    numeric_diffs['Percentage_Difference'] = pd.to_numeric(numeric_diffs['Percentage_Difference'])

                    significant_diffs = numeric_diffs[numeric_diffs['Percentage_Difference'].abs() > 5]

                    if not significant_diffs.empty:
                        report['key_findings'].append(
                            f"Found {len(significant_diffs)} scalar metrics with >5% difference"
                        )

                        # List the most significant differences
                        for _, row in significant_diffs.head(5).iterrows():
                            report['key_findings'].append(
                                f"  - {row['Metric']}: {row['Percentage_Difference']:.1f}% difference"
                            )
                    else:
                        report['key_findings'].append("No significant scalar differences found (all <5%)")
                else:
                    report['key_findings'].append("No numeric comparisons possible")

            except Exception as e:
                report['key_findings'].append(f"Error analyzing scalar comparison: {e}")
        else:
            report['key_findings'].append("No scalar comparison file generated")

        # Add recommendations
        report['recommendations'].extend([
            "1. Review any metrics with >5% difference for potential issues",
            "2. Check dispatch strategy settings if battery behavior differs significantly",
            "3. Verify weather file consistency between models",
            "4. Compare module execution order and parameters",
            "5. Check for any custom processing affecting battery calculations"
        ])

        # Save report
        with open(self.analysis_results_dir / "battery_comparison_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print("   📄 Generated summary report")

        # Print key findings to console
        print(f"\n🔍 KEY FINDINGS:")
        for finding in report['key_findings']:
            print(f"   {finding}")

        print(f"\n💡 RECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"   {rec}")


# ========== MAIN EXECUTION ==========

def main():
    """Main execution function"""
    try:
        analyzer = PathAwareBatteryComparisonAnalyzer()
        analyzer.run_analysis()

    except Exception as e:
        print(f"💥 ANALYSIS FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()