#!/usr/bin/env python3
"""
Analyze the current JSON configuration to find capacity mismatches
FIXED: Proper path detection and error handling
"""

import json
import sys
from pathlib import Path


def find_json_file():
    """Intelligently find the JSON configuration file"""

    # Get the current script location
    script_path = Path(__file__).resolve()
    print(f"🔍 Script location: {script_path}")

    # Possible locations for the JSON file
    possible_paths = [
        # If running from project root
        Path.cwd() / "1.input" / "json_updated" / "All_commercial_updated.json",

        # If running from 1.input/json_updated directory
        Path.cwd() / "All_commercial_updated.json",

        # Relative to script location
        script_path.parent / "All_commercial_updated.json",
        script_path.parent.parent / "All_commercial_updated.json",

        # Navigate up from script to project root
        script_path.parent.parent.parent / "1.input" / "json_updated" / "All_commercial_updated.json",

        # Common project structure locations
        Path.home() / "PycharmProjects" / "EnTrans_v2" / "1.input" / "json_updated" / "All_commercial_updated.json"
    ]

    print(f"🔍 Searching for JSON file...")
    for i, path in enumerate(possible_paths, 1):
        print(f"   {i}. Checking: {path}")
        if path.exists():
            print(f"   ✅ FOUND: {path}")
            return path
        else:
            print(f"   ❌ Not found")

    return None


def analyze_pysam_config(json_path):
    """Analyze PySAM configuration for capacity consistency issues"""

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        print(f"\n🔍 ANALYZING PYSAM CONFIGURATION")
        print(f"File: {json_path}")
        print("=" * 80)

        # === CAPACITY ANALYSIS ===
        print(f"\n[1] CAPACITY CONFIGURATION:")

        system_capacity = data.get('system_capacity', 'Not found')
        nstrings = data.get('subarray1_nstrings', 'Not found')
        modules_per_string = data.get('subarray1_modules_per_string', 'Not found')

        print(f"   system_capacity: {system_capacity}")
        print(f"   subarray1_nstrings: {nstrings}")
        print(f"   subarray1_modules_per_string: {modules_per_string}")

        # Calculate expected capacity
        if isinstance(nstrings, (int, float)) and isinstance(modules_per_string, (int, float)):
            total_modules = nstrings * modules_per_string

            # Check for module power specification
            module_power = data.get('module_power', data.get('spe_power', 500))  # Default 500W
            calculated_dc_kw = (total_modules * module_power) / 1000

            print(f"\n   📊 CALCULATED FROM ARRAY:")
            print(f"   Total modules: {nstrings} × {modules_per_string} = {total_modules}")
            print(f"   Module power: {module_power}W")
            print(f"   Expected DC capacity: {calculated_dc_kw:.1f} kW")

            # Check consistency
            if isinstance(system_capacity, (int, float)):
                mismatch = abs(calculated_dc_kw - system_capacity)
                mismatch_pct = (mismatch / system_capacity * 100) if system_capacity > 0 else 0

                print(f"\n   🔍 CONSISTENCY CHECK:")
                print(f"   Array capacity: {calculated_dc_kw:.1f} kW")
                print(f"   system_capacity: {system_capacity:.1f} kW")
                print(f"   Mismatch: {mismatch:.1f} kW ({mismatch_pct:.1f}%)")

                if mismatch > 50:  # More than 50kW difference
                    print(f"   🚨 MAJOR MISMATCH: {mismatch:.1f} kW difference!")
                    print(f"   💡 This is likely causing the 81% loss issue")
                    print(f"   🔧 FIX: Set system_capacity = {calculated_dc_kw:.1f}")
                    return False, {
                        'issue': 'capacity_mismatch',
                        'current': system_capacity,
                        'should_be': calculated_dc_kw,
                        'mismatch_kw': mismatch
                    }
                elif mismatch > 5:
                    print(f"   ⚠️  Minor mismatch: {mismatch:.1f} kW")
                else:
                    print(f"   ✅ Good: Capacities match within {mismatch:.1f} kW")
            else:
                print(f"   ❌ system_capacity is not numeric: {system_capacity}")
                return False, {'issue': 'invalid_system_capacity', 'value': system_capacity}
        else:
            print(f"   ❌ Array configuration incomplete")
            return False, {'issue': 'incomplete_array_config'}

        # === INVERTER ANALYSIS ===
        print(f"\n[2] INVERTER CONFIGURATION:")

        inverter_model = data.get('inverter_model', 'Not found')
        print(f"   inverter_model: {inverter_model}")

        # Check for multiple inverter model configurations (conflicts)
        inverter_configs = {}
        for key, value in data.items():
            if key.startswith('inv_'):
                inverter_configs[key] = value

        # Group by model type
        snl_params = {k: v for k, v in inverter_configs.items() if 'snl' in k}
        cec_params = {k: v for k, v in inverter_configs.items() if 'cec' in k}
        ds_params = {k: v for k, v in inverter_configs.items() if 'ds' in k}
        pd_params = {k: v for k, v in inverter_configs.items() if 'pd' in k}

        print(f"   SNL model params: {len(snl_params)}")
        print(f"   CEC model params: {len(cec_params)}")
        print(f"   Datasheet params: {len(ds_params)}")
        print(f"   Part Load params: {len(pd_params)}")

        # Check for conflicts
        model_count = sum(1 for params in [snl_params, cec_params, ds_params, pd_params] if params)
        if model_count > 1:
            print(f"   🚨 INVERTER CONFLICT: {model_count} different inverter models configured!")
            print(f"   💡 This can cause calculation errors")
            return False, {'issue': 'inverter_conflict', 'model_count': model_count}
        elif model_count == 1:
            print(f"   ✅ Single inverter model configured")
        else:
            print(f"   ⚠️  No inverter parameters found")

        # === BATTERY ANALYSIS ===
        print(f"\n[3] BATTERY CONFIGURATION:")

        battery_capacity = data.get('batt_computed_bank_capacity', 0)
        battery_power = data.get('batt_power_discharge_max_kwac', 0)
        battery_enabled = data.get('en_batt', 0)

        print(f"   en_batt: {battery_enabled}")
        print(f"   batt_computed_bank_capacity: {battery_capacity} kWh")
        print(f"   batt_power_discharge_max_kwac: {battery_power} kW")

        if battery_capacity > 0:
            c_rate = battery_power / battery_capacity if battery_capacity > 0 else 0
            print(f"   C-rate: {c_rate:.2f}C")
            if c_rate > 2:
                print(f"   ⚠️  High C-rate: {c_rate:.2f}C")
            else:
                print(f"   ✅ Reasonable C-rate")

        # === FINANCIAL ANALYSIS ===
        print(f"\n[4] FINANCIAL CONFIGURATION:")

        total_cost = data.get('total_installed_cost', 'Not found')
        analysis_period = data.get('analysis_period', 'Not found')

        print(f"   total_installed_cost: {total_cost}")
        print(f"   analysis_period: {analysis_period}")

        if isinstance(total_cost, (int, float)) and isinstance(system_capacity, (int, float)) and system_capacity > 0:
            cost_per_kw = total_cost / system_capacity
            print(f"   Cost per kW: ${cost_per_kw:,.0f}/kW")

            if cost_per_kw > 10000:  # Very high cost per kW
                print(f"   🚨 Very high cost per kW: ${cost_per_kw:,.0f}/kW")
                print(f"   💡 May indicate capacity or cost calculation error")
            elif cost_per_kw < 500:  # Very low cost per kW
                print(f"   🚨 Very low cost per kW: ${cost_per_kw:,.0f}/kW")
                print(f"   💡 May indicate capacity or cost calculation error")
            else:
                print(f"   ✅ Reasonable cost per kW")

        # === SUMMARY ===
        print(f"\n" + "=" * 80)
        print(f"🎯 SUMMARY:")
        print(f"✅ No major configuration issues found")
        print(f"🔍 If 81% loss persists, it's likely in PySAM's internal calculations")
        print(f"=" * 80)

        return True, {'status': 'clean'}

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {'issue': 'analysis_error', 'error': str(e)}


def main():
    """Main function with improved path handling"""

    print(f"🔍 JSON CONFIGURATION ANALYZER")
    print(f"=" * 40)

    # Check for command line argument
    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1])
        if not json_path.exists():
            print(f"❌ Specified file not found: {json_path}")
            json_path = None
    else:
        json_path = None

    # If no valid path from command line, try to find it
    if json_path is None:
        json_path = find_json_file()

    if json_path is None:
        print(f"\n❌ Could not find All_commercial_updated.json file")
        print(f"💡 Please run this script from the project root directory")
        print(f"💡 Or specify the path: python {sys.argv[0]} <path_to_json>")

        # Show current directory for debugging
        print(f"\n🔍 Current directory: {Path.cwd()}")
        print(f"🔍 Script location: {Path(__file__).resolve()}")

        sys.exit(1)

    # Run analysis
    is_clean, details = analyze_pysam_config(json_path)

    print(f"\n" + "=" * 40)
    if is_clean:
        print(f"✅ CONFIGURATION ANALYSIS COMPLETE")
        print(f"🔍 Configuration appears clean")
        print(f"💡 If 81% loss persists, it's in PySAM internals")
    else:
        print(f"🚨 CONFIGURATION ISSUES FOUND")
        print(f"📋 Issue details: {details}")
        print(f"💡 Fix these issues before running PySAM")

    sys.exit(0 if is_clean else 1)


if __name__ == "__main__":
    main()