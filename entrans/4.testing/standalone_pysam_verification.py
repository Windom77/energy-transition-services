"""
Standalone PySAM Verification Script
Basic PySAM execution using the same config as main model but with only standard outputs.
This is for verifying battery behavior against a clean baseline without custom processing.

Place in: 4.testing/standalone_pysam_verification.py
Run from project root: python 4.testing/standalone_pysam_verification.py
"""

import json
import logging
import os
import sys
import importlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import PySAM.PySSC as pssc


# ========== CONFIGURATION ==========

class TestConfig:
    """Configuration paths for testing"""

    def __init__(self):
        # Determine project root
        current_path = Path(__file__).resolve()
        self.project_root = None

        # Look for project root indicators
        for parent in current_path.parents:
            if (parent / 'main.py').exists() or (parent / 'config.py').exists():
                self.project_root = parent
                break

        if self.project_root is None:
            # Fallback: go up from 4.testing
            self.project_root = current_path.parent.parent

        # Set paths
        self.input_config = self.project_root / "1.input" / "json_updated" / "All_commercial_updated.json"
        self.weather_dir = self.project_root / "data" / "weather"
        self.results_dir = Path(__file__).parent / "results"

        print(f"📁 Project Root: {self.project_root}")
        print(f"📁 Input Config: {self.input_config}")
        print(f"📁 Weather Dir: {self.weather_dir}")
        print(f"📁 Results Dir: {self.results_dir}")


# ========== STANDALONE PYSAM RUNNER ==========

class StandalonePySAMRunner:
    """Bare-bones PySAM runner for verification"""

    def __init__(self):
        self.config = TestConfig()
        self.config_dict = {}
        self.modules = {}
        self.module_order = []

    def run_verification(self):
        """Run complete verification"""
        try:
            print(f"\n🔧 STANDALONE PYSAM VERIFICATION")
            print(f"=" * 50)

            # Setup
            self._setup_logging()
            self._load_config()
            self._validate_weather_file()

            # Execute PySAM
            self._initialize_modules()
            self._execute_modules()

            # Export results
            self._export_results()

            print(f"\n✅ VERIFICATION COMPLETED SUCCESSFULLY")
            print(f"📁 Results saved to: {self.config.results_dir}")

        except Exception as e:
            print(f"❌ VERIFICATION FAILED: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _setup_logging(self):
        """Setup basic logging"""
        self.config.results_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.results_dir / 'verification.log'),
                logging.StreamHandler()
            ]
        )

        logging.info("🚀 Starting Standalone PySAM Verification")

    def _load_config(self):
        """Load configuration from JSON"""
        if not self.config.input_config.exists():
            raise FileNotFoundError(f"Config file not found: {self.config.input_config}")

        with open(self.config.input_config, 'r') as f:
            self.config_dict = json.load(f)

        print(f"✅ Loaded config: {len(self.config_dict)} parameters")

        # Extract module order
        compute_modules = sorted(
            [(int(k.split('_')[-1]), v) for k, v in self.config_dict.items()
             if k.startswith("compute_module_")],
            key=lambda x: x[0]
        )

        self.module_order = [name for _, name in compute_modules]
        print(f"📋 Modules to execute: {self.module_order}")

        # Log key parameters for verification
        self._log_key_parameters()

    def _log_key_parameters(self):
        """Log key parameters for verification"""
        print(f"\n🔍 KEY PARAMETERS FOR VERIFICATION:")

        # System parameters
        system_capacity = self.config_dict.get('system_capacity', 'N/A')
        print(f"   System Capacity: {system_capacity} kW")

        # Battery parameters
        en_batt = self.config_dict.get('en_batt', 0)
        batt_capacity = self.config_dict.get('batt_computed_bank_capacity', 0)
        batt_power = self.config_dict.get('batt_power_discharge_max_kwac', 0)
        batt_dispatch = self.config_dict.get('batt_dispatch_choice', 'N/A')

        print(f"   Battery Enabled: {en_batt}")
        print(f"   Battery Capacity: {batt_capacity} kWh")
        print(f"   Battery Power: {batt_power} kW")
        print(f"   Battery Dispatch: {batt_dispatch}")

        # Financial parameters
        analysis_period = self.config_dict.get('analysis_period', 'N/A')
        discount_rate = self.config_dict.get('real_discount_rate', 'N/A')

        print(f"   Analysis Period: {analysis_period} years")
        print(f"   Discount Rate: {discount_rate}")

        # Weather file
        weather_file = self.config_dict.get('solar_resource_file', 'N/A')
        print(f"   Weather File: {weather_file}")

    def _validate_weather_file(self):
        """Validate and fix weather file path"""
        weather_file = self.config_dict.get('solar_resource_file')

        if not weather_file:
            print("⚠️ No weather file specified")
            return

        weather_path = Path(weather_file)

        # If absolute path exists, use it
        if weather_path.exists():
            print(f"✅ Weather file found: {weather_path}")
            return

        # Try in weather directory
        weather_filename = weather_path.name
        weather_dir_path = self.config.weather_dir / weather_filename

        if weather_dir_path.exists():
            self.config_dict['solar_resource_file'] = str(weather_dir_path)
            print(f"✅ Weather file found in directory: {weather_dir_path}")
        else:
            print(f"❌ Weather file not found: {weather_file}")
            print(f"   Tried: {weather_path}")
            print(f"   Tried: {weather_dir_path}")

    def _initialize_modules(self):
        """Initialize PySAM modules"""
        print(f"\n🔧 INITIALIZING PYSAM MODULES:")

        for i, mod_name in enumerate(self.module_order):
            try:
                print(f"   Initializing {mod_name}...")

                # Convert config to SSC table for this module
                mod_data = pssc.dict_to_ssc_table(self.config_dict, mod_name)

                # Normalize module name for import
                pysam_mod_name = mod_name[0].upper() + mod_name[1:]

                # Import PySAM module
                pysam_module = importlib.import_module(f'PySAM.{pysam_mod_name}')

                # Create module instance
                if i == 0:
                    # First module
                    module_instance = pysam_module.wrap(mod_data)
                else:
                    # Chain from previous module
                    prev_mod_name = self.module_order[i - 1]
                    module_instance = pysam_module.from_existing(self.modules[prev_mod_name])
                    module_instance.assign(pysam_module.wrap(mod_data).export())

                self.modules[mod_name] = module_instance
                print(f"   ✅ {mod_name} initialized")

            except Exception as e:
                print(f"   ❌ {mod_name} failed: {e}")
                raise

        print(f"✅ All {len(self.modules)} modules initialized")

    def _execute_modules(self):
        """Execute PySAM modules"""
        print(f"\n🚀 EXECUTING PYSAM MODULES:")

        for mod_name in self.module_order:
            try:
                print(f"   Executing {mod_name}...")
                self.modules[mod_name].execute()
                print(f"   ✅ {mod_name} executed")

            except Exception as e:
                print(f"   ❌ {mod_name} execution failed: {e}")
                raise

        print(f"✅ All modules executed successfully")

    def _export_results(self):
        """Export standard PySAM results"""
        print(f"\n💾 EXPORTING RESULTS:")

        # Change to results directory
        original_cwd = os.getcwd()
        os.chdir(str(self.config.results_dir))

        try:
            # Export scalar results
            self._export_scalar_results()

            # Export time series
            self._export_time_series()

            # Export metadata
            self._export_metadata()

        finally:
            os.chdir(original_cwd)

    def _export_scalar_results(self):
        """Export scalar results to CSV"""
        print("   📊 Exporting scalar results...")

        scalar_results = {}

        # Collect outputs from all modules
        for mod_name, module in self.modules.items():
            if hasattr(module, 'Outputs'):
                outputs = self._collect_outputs_recursive(module.Outputs, f"{mod_name}_")

                # Only keep scalar values
                for key, value in outputs.items():
                    if not isinstance(value, (list, tuple, np.ndarray)):
                        scalar_results[key] = value

        # Convert to DataFrame and export
        if scalar_results:
            df = pd.DataFrame.from_dict(scalar_results, orient='index', columns=['Value'])
            df.index.name = 'Metric'
            df.to_csv('scalar_results.csv')
            print(f"   ✅ Exported {len(scalar_results)} scalar metrics")
        else:
            print("   ⚠️ No scalar results found")

    def _export_time_series(self):
        """Export time series to CSV files"""
        print("   📈 Exporting time series...")

        time_series = {}

        # Collect time series from all modules
        for mod_name, module in self.modules.items():
            if hasattr(module, 'Outputs'):
                outputs = self._collect_outputs_recursive(module.Outputs, f"{mod_name}_")

                # Only keep array values
                for key, value in outputs.items():
                    if isinstance(value, (list, tuple, np.ndarray)) and len(value) > 1:
                        time_series[key] = list(value)

        if not time_series:
            print("   ⚠️ No time series found")
            return

        # Group by length and export
        length_groups = {}
        for key, values in time_series.items():
            length = len(values)
            if length not in length_groups:
                length_groups[length] = {}
            length_groups[length][key] = values

        # Export each group
        for length, group_data in length_groups.items():
            # Determine frequency
            if length == 12:
                filename = 'monthly_timeseries.csv'
                period_name = 'Month'
            elif length == 26 or length == 25:
                filename = 'annual_timeseries.csv'
                period_name = 'Year'
            elif length == 8760:
                filename = 'hourly_timeseries.csv'
                period_name = 'Hour'
            else:
                filename = f'timeseries_{length}_periods.csv'
                period_name = 'Period'

            # Create DataFrame
            df = pd.DataFrame({
                period_name: range(1, length + 1),
                **group_data
            })

            df.to_csv(filename, index=False)
            print(f"   ✅ Exported {filename}: {len(group_data)} series, {length} periods")

    def _export_metadata(self):
        """Export metadata about the verification run"""
        print("   📋 Exporting metadata...")

        metadata = {
            'verification_info': {
                'timestamp': datetime.now().isoformat(),
                'config_file': str(self.config.input_config),
                'weather_file': self.config_dict.get('solar_resource_file', 'N/A'),
                'modules_executed': self.module_order,
                'purpose': 'Standalone verification of PySAM outputs without custom processing'
            },
            'system_config': {
                'system_capacity_kw': self.config_dict.get('system_capacity', 'N/A'),
                'battery_enabled': bool(self.config_dict.get('en_batt', 0)),
                'battery_capacity_kwh': self.config_dict.get('batt_computed_bank_capacity', 0),
                'battery_power_kw': self.config_dict.get('batt_power_discharge_max_kwac', 0),
                'battery_dispatch_choice': self.config_dict.get('batt_dispatch_choice', 'N/A'),
                'analysis_period_years': self.config_dict.get('analysis_period', 'N/A')
            },
            'financial_config': {
                'discount_rate': self.config_dict.get('real_discount_rate', 'N/A'),
                'inflation_rate': self.config_dict.get('inflation_rate', 'N/A'),
                'loan_rate': self.config_dict.get('loan_rate', 'N/A'),
                'debt_fraction': self.config_dict.get('debt_fraction', 'N/A')
            }
        }

        with open('verification_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        print("   ✅ Exported verification metadata")

    def _collect_outputs_recursive(self, obj, prefix=""):
        """Recursively collect outputs from PySAM module"""
        outputs = {}

        for attr in dir(obj):
            if attr.startswith('_'):
                continue

            try:
                val = getattr(obj, attr)

                # If it's another object with attributes, recurse
                if hasattr(val, '__dict__') and not isinstance(val, (list, tuple, np.ndarray)):
                    outputs.update(self._collect_outputs_recursive(val, f"{prefix}{attr}_"))
                else:
                    # Store the value
                    key = f"{prefix}{attr}"

                    # Convert numpy arrays to lists
                    if isinstance(val, np.ndarray):
                        outputs[key] = val.tolist()
                    elif isinstance(val, (list, tuple)):
                        # Handle tuples and nested structures
                        if val and isinstance(val[0], tuple):
                            outputs[key] = [v[0] if isinstance(v, tuple) else v for v in val]
                        else:
                            outputs[key] = list(val)
                    else:
                        outputs[key] = val

            except Exception:
                # Skip inaccessible attributes
                continue

        return outputs


# ========== BATTERY COMPARISON UTILITIES ==========

def compare_with_main_results():
    """Optional: Compare standalone results with main model results"""
    print(f"\n🔍 COMPARISON WITH MAIN RESULTS:")

    # Path to main results
    main_results_path = Path("2.pysam/results/scalar_results.csv")
    standalone_results_path = Path("4.testing/results/scalar_results.csv")

    if not main_results_path.exists():
        print("   ⚠️ Main results not found - run main simulation first")
        return

    if not standalone_results_path.exists():
        print("   ⚠️ Standalone results not found")
        return

    try:
        # Load both result sets
        main_df = pd.read_csv(main_results_path, index_col=0)
        standalone_df = pd.read_csv(standalone_results_path, index_col=0)

        # Find common metrics
        common_metrics = set(main_df.index) & set(standalone_df.index)

        if not common_metrics:
            print("   ⚠️ No common metrics found")
            return

        # Compare key metrics
        key_metrics = [
            'annual_energy', 'npv', 'lcoe_real', 'lcoe_nom', 'payback',
            'batt_annual_charge_energy', 'batt_annual_discharge_energy',
            'average_battery_roundtrip_efficiency'
        ]

        comparison_metrics = [m for m in key_metrics if m in common_metrics]

        print(f"   📊 Comparing {len(comparison_metrics)} key metrics:")

        for metric in comparison_metrics:
            main_val = main_df.loc[metric, 'Value']
            standalone_val = standalone_df.loc[metric, 'Value']

            # Calculate difference
            if isinstance(main_val, (int, float)) and isinstance(standalone_val, (int, float)):
                diff = main_val - standalone_val
                if standalone_val != 0:
                    pct_diff = (diff / standalone_val) * 100
                    print(f"   {metric}: Main={main_val:.2f}, Standalone={standalone_val:.2f}, Diff={pct_diff:.1f}%")
                else:
                    print(f"   {metric}: Main={main_val:.2f}, Standalone={standalone_val:.2f}")
            else:
                print(f"   {metric}: Main={main_val}, Standalone={standalone_val}")

        print(f"   ✅ Comparison completed")

    except Exception as e:
        print(f"   ❌ Comparison failed: {e}")


# ========== MAIN EXECUTION ==========

def main():
    """Main execution function"""
    try:
        # Run standalone verification
        runner = StandalonePySAMRunner()
        runner.run_verification()

        # Optional comparison
        compare_with_main_results()

    except Exception as e:
        print(f"💥 VERIFICATION FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()