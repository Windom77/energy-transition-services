# dashboard_module.py - Enhanced with Variable Analysis Period Support
"""
Enhanced FCAS Results Dashboard Module for Flask Integration
Now supports variable analysis periods (not just 25 years)
"""

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import pandas as pd
import pyarrow.parquet as pq
import plotly.express as px
import plotly.graph_objects as go
import os
import ast
import json
import dash_bootstrap_components as dbc
from datetime import datetime
import numpy as np
from pathlib import Path
import traceback
import logging
import time

logger = logging.getLogger(__name__)

# Import config
try:
    from config import Config

except ImportError:
    # Fallback config if not available
    class Config:
        RESULTS_DIR = Path("2.pysam/results")
        OUTPUT_DIR = Path("3.output")
        INPUT_DIR = Path("1.input")
        DATA_DIR = Path("data")

# ==============================================================================
# ENHANCED CSS STYLING - Consolidated and streamlined
# ==============================================================================

ENHANCED_CSS = """
body {
    background: #f8f9fa;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.dashboard-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 2rem 0;
    margin-bottom: 2rem;
}

.metric-card {
    border: none;
    border-radius: 15px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.08);
    transition: all 0.3s ease;
    height: 100%;
}

.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 35px rgba(0,0,0,0.1);
}

.value-prop-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 15px;
    padding: 2rem;
    text-align: center;
    height: 180px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    box-shadow: 0 10px 30px rgba(0,0,0,0.15);
}

.performance-table {
    border-radius: 15px;
    overflow: hidden;
    box-shadow: 0 5px 15px rgba(0,0,0,0.08);
    background: white;
    margin-bottom: 2rem;
}

.performance-table .dash-table-container {
    border-radius: 0 0 15px 15px;
}

.performance-table .dash-spreadsheet-container {
    border-radius: 0 0 15px 15px;
}

/* Enhanced table header styling */
.performance-table .dash-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    font-weight: bold !important;
    padding: 15px 12px !important;
    text-align: center !important;
    border: none !important;
}

/* Improved table cell styling */
.performance-table .dash-cell {
    padding: 12px !important;
    font-family: 'Segoe UI', sans-serif !important;
    font-size: 14px !important;
    border-bottom: 1px solid #e5e7eb !important;
    border-right: 1px solid #e5e7eb !important;
}

/* Alternating row colors */
.performance-table .dash-cell.dash-cell--odd {
    background-color: #f9fafb !important;
}

.performance-table .dash-cell.dash-cell--even {
    background-color: white !important;
}

/* First column styling (metric names) */
.performance-table .dash-cell:first-child {
    font-weight: 600 !important;
    background-color: #f8fafc !important;
    border-right: 2px solid #d1d5db !important;
}

.section-title {
    font-weight: 600;
    color: #2d3748;
    border-bottom: 3px solid #667eea;
    padding-bottom: 0.5rem;
    margin-bottom: 1.5rem;
    margin-top: 2rem;
}

.waterfall-container {
    background: white;
    border-radius: 15px;
    padding: 1.5rem;
    box-shadow: 0 5px 15px rgba(0,0,0,0.08);
    margin-bottom: 2rem;
}

/* Status indicators */
.status-excellent { 
    background-color: #10b981; 
    color: white; 
    padding: 4px 8px; 
    border-radius: 12px; 
    font-size: 0.8em;
    font-weight: bold;
}
.status-good { 
    background-color: #3b82f6; 
    color: white; 
    padding: 4px 8px; 
    border-radius: 12px; 
    font-size: 0.8em;
    font-weight: bold;
}
.status-warning { 
    background-color: #f59e0b; 
    color: white; 
    padding: 4px 8px; 
    border-radius: 12px; 
    font-size: 0.8em;
    font-weight: bold;
}

/* Card styling improvements */
.card {
    border-radius: 15px !important;
    border: none !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06) !important;
}

.card-header {
    border-radius: 15px 15px 0 0 !important;
    border-bottom: 1px solid #e5e7eb !important;
    padding: 1.25rem 1.5rem !important;
}

.card-body {
    padding: 1.5rem !important;
}
"""

LANDSCAPE_TABLE_CSS = """
/* Enhanced landscape table styling */
.landscape-table-container {
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.landscape-table-container .dash-table-container {
    border-radius: 0;
}

.landscape-table-container .dash-spreadsheet-container {
    border-radius: 0;
}

/* FCAS table specific styling */
.fcas-landscape-table .dash-header {
    background: linear-gradient(135deg, #198754 0%, #20c997 100%) !important;
}

/* Cashflow table specific styling */
.cashflow-landscape-table .dash-header {
    background: linear-gradient(135deg, #0d6efd 0%, #6610f2 100%) !important;
}

/* Revenue table specific styling */
.revenue-landscape-table .dash-header {
    background: linear-gradient(135deg, #20c997 0%, #0dcaf0 100%) !important;
}

/* Cost table specific styling */
.cost-landscape-table .dash-header {
    background: linear-gradient(135deg, #dc3545 0%, #fd7e14 100%) !important;
}
"""


# ==============================================================================
# ANALYSIS PERIOD UTILITIES - NEW SECTION
# ==============================================================================

def get_analysis_period_from_metadata(data_sources):
    """Extract analysis period from metadata with fallback to 25 years"""
    try:
        metadata = data_sources.get('metadata', {})
        execution_info = metadata.get('execution', {})
        analysis_period = execution_info.get('analysis_period', 25)

        # Validate analysis period is reasonable
        if isinstance(analysis_period, (int, float)) and 1 <= analysis_period <= 50:
            return int(analysis_period)
        else:
            logger.warning(f"Invalid analysis period {analysis_period}, using 25 years")
            return 25
    except Exception as e:
        logger.warning(f"Could not extract analysis period: {e}, using 25 years")
        return 25


def get_period_labels(analysis_period):
    """Generate period labels for given analysis period"""
    year_labels = []
    max_periods = analysis_period + 1  # Include Year 0

    for i in range(max_periods):
        if i == 0:
            year_labels.append("Year 0")
        else:
            year_labels.append(f"Year {i}")

    return year_labels


def validate_dataframe_length(df, analysis_period, data_source_name=""):
    """Validate and truncate dataframe to analysis period length"""
    max_periods = analysis_period + 1  # Include Year 0

    if len(df) > max_periods:
        logger.info(f"Truncating {data_source_name} from {len(df)} to {max_periods} periods")
        return df.iloc[:max_periods]
    elif len(df) < max_periods:
        logger.warning(f"{data_source_name} has only {len(df)} periods, expected {max_periods}")

    return df


def get_dynamic_column_config(analysis_period, custom_layout):
    """Get column configuration with dynamic period headers"""
    base_config = {
        'showYear1': True,
        'show25Year': False,  # Will be updated below
        'year1Header': 'Year 1',
        'totalHeader': f'{analysis_period}-Year Total'
    }

    # Update from custom layout
    if custom_layout:
        column_config = custom_layout.get('columnConfig', {})
        base_config.update(column_config)

        # Handle legacy 'show25Year' key
        if 'show25Year' in column_config:
            base_config['showTotal'] = column_config['show25Year']

        # Update header if not specified
        if 'total25Header' in column_config and 'totalHeader' not in base_config:
            base_config['totalHeader'] = column_config['total25Header']

    # Set showTotal if not explicitly set
    if 'showTotal' not in base_config:
        base_config['showTotal'] = base_config.get('show25Year', False)

    return base_config


# ==============================================================================
# DATA LOADING AND PROCESSING FUNCTIONS - UPDATED
# ==============================================================================

def get_paths():
    """Get configured paths or fallback paths"""
    try:
        results_path = Config.RESULTS_DIR
        structure_path = Config.OUTPUT_DIR / "results_structure7.csv"
        project_config_path = Config.INPUT_DIR / "json_updated" / "All_commercial_updated.json"
        data_dir = Config.DATA_DIR
    except:
        # Fallback paths
        results_path = Path("2.pysam/results")
        structure_path = Path("3.output/results_structure7.csv")
        project_config_path = Path("1.input/json_updated/All_commercial_updated.json")
        data_dir = Path("data")

    return results_path, structure_path, project_config_path, data_dir


def safe_float_conversion(value, default=0.0):
    """Safely convert value to float, handling NaN, None, and invalid values"""
    if pd.isna(value) or value is None or value == '' or value == 'N/A':
        return default
    try:
        float_val = float(value)
        # Check if it's actually NaN (not just zero)
        if pd.isna(float_val) or not np.isfinite(float_val):
            return default
        return float_val
    except (ValueError, TypeError):
        return default


def safe_literal_eval(s):
    """Safely evaluate string literals with enhanced error handling"""
    if pd.isna(s) or s == '' or s is None:
        return {}
    try:
        clean_s = str(s).strip()
        return ast.literal_eval(clean_s)
    except (ValueError, SyntaxError, TypeError):
        try:
            json_s = clean_s.replace("'", '"')
            return json.loads(json_s)
        except:
            print(f"WARNING: Could not parse custom_layout: {s}")
            return {}


def safe_get_calculation(component):
    """Safely get calculation value handling NaN"""
    calc_value = component.get('calculation', '')
    if pd.isna(calc_value):
        return ''
    return str(calc_value).strip()


def format_metric_value(value, format_str):
    """Enhanced metric formatting with proper pipe-separated format handling"""
    if pd.isna(value) or value == '' or value is None or value == "N/A":
        return "N/A"

    if pd.isna(format_str) or format_str == '' or format_str is None:
        return str(value)

    try:
        num_value = float(value)

        # Enhanced format mappings with degree symbols
        format_mappings = {
            # Currency formats
            '${:.3f}/kWh': lambda x: f"${x:.3f}/kWh",
            '${:,.2f}': lambda x: f"${x:,.2f}",
            '${:,.0f}': lambda x: f"${x:,.0f}",

            # Power and energy formats
            '{:,.1f} kW': lambda x: f"{x:,.1f} kW",
            '{:.1f} kW': lambda x: f"{x:.1f} kW",
            '{:.1f} kWh': lambda x: f"{x:,.1f} kWh",
            '{:,.1f} kWh': lambda x: f"{x:,.1f} kWh",
            '{:,.0f} kWh': lambda x: f"{x:,.0f} kWh",
            '{:,.1f} kWh/kW': lambda x: f"{x:,.1f} kWh/kW",
            '{:,.0f} kWh/kW': lambda x: f"{x:,.0f} kWh/kW",

            # Angle formats with degree symbols
            '{:.0f}°': lambda x: f"{x:.0f}°",
            '{:.1f}°': lambda x: f"{x:.1f}°",

            # Count/number formats
            '{:.0f}': lambda x: f"{x:.0f}",
            '{:,.0f}': lambda x: f"{x:,.0f}",
            '{:.1f}': lambda x: f"{x:.1f}",
            '{:.2f}': lambda x: f"{x:.2f}",

            # Percentage formats
            '{:.2%}': lambda x: f"{x:.2%}" if x <= 1 else f"{x / 100:.2%}",
            '{:.1%}': lambda x: f"{x:.1%}" if x <= 1 else f"{x / 100:.1%}",
            '{:.0%}': lambda x: f"{x:.0%}" if x <= 1 else f"{x / 100:.0%}",
            '{:.1f}%': lambda x: f"{x:.1f}%",
            '{:.2f}%': lambda x: f"{x:.2f}%",

            # Time formats
            '{:.1f} years': lambda x: "No payback" if x <= 0 else f"{x:.1f} years",

            # Battery specific
            '{:.0f} cycles': lambda x: "No cycles" if x == 0 else f"{x:.0f} cycles"
        }

        if format_str in format_mappings:
            return format_mappings[format_str](num_value)
        else:
            try:
                return format_str.format(num_value)
            except:
                return str(value)

    except (ValueError, TypeError):
        return str(value)


def format_landscape_currency(value):
    """Format currency for landscape tables - comma separated, no decimals"""
    if pd.isna(value) or value is None:
        return "$0"

    try:
        num_value = float(value)
        if num_value == 0:
            return "$0"
        elif num_value < 0:
            return f"-${abs(num_value):,.0f}"
        else:
            return f"${num_value:,.0f}"
    except (ValueError, TypeError):
        return str(value)


def load_data_sources_from_csv(structure_df):
    """Enhanced data loading with dynamic analysis period support"""
    results_path, _, project_config_path, data_dir = get_paths()
    data_sources = {}

    # Get all unique data sources and handle pipe-separated values
    all_sources = set()
    for source_string in structure_df['data_source'].dropna().unique():
        if '|' in source_string:
            individual_sources = source_string.split('|')
            for source in individual_sources:
                source = source.strip()
                if source:
                    all_sources.add(source)
        else:
            source = source_string.strip()
            if source:
                all_sources.add(source)

    referenced_sources = list(all_sources)

    # Load metadata first to get analysis period
    metadata_path = results_path / 'metadata.json'
    analysis_period = 25  # Default fallback
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            data_sources['metadata'] = metadata
            analysis_period = metadata.get('execution', {}).get('analysis_period', 25)
            logger.info(f"Analysis period from metadata: {analysis_period} years")
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")

    # Load scalar results with FCAS support
    if 'scalar_results.csv' in referenced_sources:
        scalar_path = results_path / 'scalar_results.csv'
        if scalar_path.exists():
            try:
                df = pd.read_csv(scalar_path)
                scalar_dict = {}
                for _, row in df.iterrows():
                    key = row.iloc[0]
                    value = row.iloc[1]
                    if pd.notna(key) and pd.notna(value):
                        scalar_dict[key] = value

                # Prioritize correct NPV field
                if "Net present value" in scalar_dict:
                    scalar_dict["NPV (Real)"] = scalar_dict["Net present value"]

                data_sources['scalar_results.csv'] = scalar_dict

                # Log FCAS availability in scalar results
                fcas_keys = [k for k in scalar_dict.keys() if 'ancillary' in k.lower() or 'fcas' in k.lower()]
                if fcas_keys:
                    print(f"   FCAS keys: {fcas_keys[:3]}..." if len(fcas_keys) > 3 else f"   FCAS keys: {fcas_keys}")

            except Exception as e:
                print(f"Error loading scalar_results.csv: {e}")
                data_sources['scalar_results.csv'] = {}

    # Load project configuration
    if any(source in ['project_info', 'main_config'] for source in referenced_sources):
        if project_config_path.exists():
            try:
                with open(project_config_path, 'r') as f:
                    config = json.load(f)

                if 'project_info' in referenced_sources:
                    data_sources['project_info'] = config.get('project_info', {})

                if 'main_config' in referenced_sources:
                    main_config_data = {}
                    for section in ['financial_inputs', 'system_inputs', 'battery_inputs']:
                        if section in config:
                            main_config_data.update(config[section])
                    for key, value in config.items():
                        if not isinstance(value, dict):
                            main_config_data[key] = value
                    data_sources['main_config'] = main_config_data

            except Exception as e:
                print(f"Error loading configuration: {e}")

    # Load parquet files with dynamic period validation
    parquet_sources = [s for s in referenced_sources if s.endswith('.parquet')]
    for source in parquet_sources:
        file_path = results_path / source
        if file_path.exists():
            try:
                df = pq.read_table(file_path).to_pandas()

                # Dynamic validation and truncation for cashflow data
                if 'cashflow' in source:
                    df = validate_dataframe_length(df, analysis_period, source)
                    df = df.dropna(axis=1, how='all').ffill()

                # Log column information for cashflow files
                if 'cashflow' in source:
                    fcas_columns = [col for col in df.columns if
                                    'ancillary' in col.lower() or 'fcas' in col.lower()]
                    print(f"   FCAS-related columns: {len(fcas_columns)}")
                    if fcas_columns:
                        print(f"   FCAS columns: {fcas_columns[:3]}..." if len(
                            fcas_columns) > 3 else f"   FCAS columns: {fcas_columns}")

                data_sources[source] = df

            except Exception as e:
                print(f"Error loading {source}: {e}")

    # Load FCAS historical data
    fcas_historical_path = data_dir / 'fcas' / 'fcas_historical.csv'
    if 'fcas_historical.csv' in referenced_sources and fcas_historical_path.exists():
        try:
            df = pd.read_csv(fcas_historical_path)
            df = df.dropna(how='all').reset_index(drop=True)

            if df.columns[0].lower() in ['quarter', 'quarters', 'q', 'period']:
                df = df.rename(columns={df.columns[0]: 'Quarter'})
            elif 'Quarter' not in df.columns:
                quarters = []
                for i in range(len(df)):
                    year_offset = i // 4
                    quarter_num = (i % 4) + 1
                    base_year = 2016
                    quarters.append(f"Q{quarter_num} {base_year + year_offset}")
                df.insert(0, 'Quarter', quarters)

            data_sources['fcas_historical.csv'] = df

        except Exception as e:
            print(f"Error loading fcas_historical.csv: {e}")

    # Create calculated values with dynamic period support
    if 'calculated' in referenced_sources:
        data_sources['calculated'] = create_enhanced_calculated_values(data_sources, analysis_period)

    return data_sources


def create_enhanced_calculated_values(data_sources, analysis_period=25):
    """Enhanced calculated values with dynamic analysis period support"""
    calculated = {}

    project_info = data_sources.get('project_info', {})
    main_config = data_sources.get('main_config', {})
    scalar_data = data_sources.get('scalar_results.csv', {})
    metadata = data_sources.get('metadata', {})

    try:
        # FCAS calculation with dynamic period
        total_fcas = scalar_data.get('Total ancillary services revenue', 0)
        calculated['total_fcas_revenue'] = float(total_fcas) if total_fcas else 0
        calculated['fcas_revenue_annual'] = calculated[
                                                'total_fcas_revenue'] / analysis_period if analysis_period > 0 else 0

        # Get individual FCAS services
        for i in range(1, 5):
            service_key = f'Ancillary services {i} revenue'
            if service_key in scalar_data:
                calculated[f'fcas_service_{i}_revenue'] = float(scalar_data[service_key])
            else:
                calculated[f'fcas_service_{i}_revenue'] = 0

        # Availability check
        calculated['fcas_services_available'] = calculated['total_fcas_revenue'] > 0

        # Energy calculations with dynamic period
        annual_energy = safe_float_conversion(scalar_data.get('annual_energy', 0))
        energy_value = safe_float_conversion(scalar_data.get('Energy value', 0))
        calculated['annual_energy_production'] = annual_energy
        calculated['total_energy_revenue'] = energy_value
        calculated[
            'energy_revenue_annual'] = energy_value / analysis_period if energy_value > 0 and analysis_period > 0 else 0

        # Other revenue streams with dynamic period
        lrec_revenue = safe_float_conversion(scalar_data.get('Total LREC revenue', 0))
        calculated['total_lrec_revenue'] = lrec_revenue
        calculated[
            'lrec_revenue_annual'] = lrec_revenue / analysis_period if lrec_revenue > 0 and analysis_period > 0 else 0

        stc_revenue = safe_float_conversion(project_info.get('stc_value', 0))
        calculated['total_stc_revenue'] = stc_revenue

        # Service revenue comparison - only if FCAS exists
        service_revenues = []
        if calculated['fcas_services_available']:
            service_names = ['Fast Raise (6s)', 'Fast Lower (6s)', 'Slow Raise (60s)', 'Slow Lower (60s)']
            for i, name in enumerate(service_names, 1):
                revenue = calculated.get(f'fcas_service_{i}_revenue', 0)
                if revenue > 0:
                    service_revenues.append({'Service': name, 'Revenue': revenue})

        calculated['service_revenue_comparison'] = service_revenues

        # Revenue breakdown - only include sources with revenue > 0
        revenue_breakdown = []

        if calculated['total_energy_revenue'] > 0:
            revenue_breakdown.append({
                'Source': 'Energy Sales',
                'Revenue': calculated['total_energy_revenue'],
                'Type': 'Energy'
            })

        if calculated['fcas_services_available'] and calculated['total_fcas_revenue'] > 0:
            revenue_breakdown.append({
                'Source': 'FCAS Services',
                'Revenue': calculated['total_fcas_revenue'],
                'Type': 'FCAS'
            })

        if calculated['total_lrec_revenue'] > 0:
            revenue_breakdown.append({
                'Source': 'LREC Revenue',
                'Revenue': calculated['total_lrec_revenue'],
                'Type': 'Certificates'
            })

        if calculated['total_stc_revenue'] > 0:
            revenue_breakdown.append({
                'Source': 'STC Revenue',
                'Revenue': calculated['total_stc_revenue'],
                'Type': 'Certificates'
            })

        calculated['revenue_breakdown_all'] = revenue_breakdown

        # Add analysis period info to calculated values
        calculated['analysis_period'] = analysis_period

        logger.info(
            f"FCAS STATUS: {calculated['fcas_services_available']} (Revenue: ${calculated['total_fcas_revenue']:,.0f}, Period: {analysis_period} years)")

    except Exception as e:
        logger.error(f"Error creating calculated values: {e}")
        # Fallback with period info
        calculated.update({
            'total_fcas_revenue': 0,
            'fcas_revenue_annual': 0,
            'fcas_services_available': False,
            'service_revenue_comparison': [],
            'revenue_breakdown_all': [],
            'analysis_period': analysis_period
        })

    return calculated


def get_value_from_data_sources(data_sources, data_source, attribute, calculation=None):
    """Data retrieval with enhanced period awareness"""
    if pd.isna(data_source):
        data_source = ''
    else:
        data_source = str(data_source)

    # Handle multiple data sources
    if '|' in data_source:
        source_files = data_source.split('|')
        attributes = attribute.split('|') if '|' in attribute else [attribute]

        while len(source_files) < len(attributes):
            source_files.append(source_files[-1])

        for i, attr in enumerate(attributes):
            if i < len(source_files):
                source_file = source_files[i]
                if source_file in data_sources:
                    calc = None
                    if calculation and not pd.isna(calculation):
                        calc_list = str(calculation).split('|') if '|' in str(calculation) else [str(calculation)]
                        calc = calc_list[i] if i < len(calc_list) else ''

                    value = get_single_source_value(data_sources[source_file], attr, calc)

                    if value != "N/A" and value != "Data not available" and not pd.isna(value):
                        return value

        return "N/A"

    # Single data source
    if data_source not in data_sources:
        return "N/A"

    calc = None
    if calculation:
        calc_clean = safe_get_calculation({'calculation': calculation})
        if calc_clean:
            calc = calc_clean

    return get_single_source_value(data_sources[data_source], attribute, calc)


def get_dispatch_choice_description(dispatch_value):
    """Convert numeric dispatch choice to descriptive text"""
    dispatch_mapping = {
        0: "Peak Shaving",
        4: "Smart Energy Trading",
        5: "Self Consumption",
        '0': "Peak Shaving",
        '4': "Smart Energy Trading",
        '5': "Self Consumption"
    }

    if pd.isna(dispatch_value) or dispatch_value == '':
        return "Not Set"

    try:
        if isinstance(dispatch_value, str):
            dispatch_key = dispatch_value.strip()
        else:
            dispatch_key = int(float(dispatch_value))

        return dispatch_mapping.get(dispatch_key, f"Unknown ({dispatch_value})")
    except (ValueError, TypeError):
        return f"Invalid ({dispatch_value})"


def get_single_source_value(source_data, attribute, calculation=None):
    """Enhanced single source value retrieval with missing column protection"""
    # Handle calculations first
    if calculation and calculation.strip():
        return perform_calculation(source_data, attribute, calculation)

    # Handle nested attributes (e.g., "project_info.location")
    if '.' in attribute:
        parts = attribute.split('.')
        value = source_data

        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return "N/A"

        if pd.isna(value) or value == '':
            return "Data not available"
        return value

    # Standard retrieval for dict sources
    if isinstance(source_data, dict):
        if attribute in source_data:
            value = source_data[attribute]
            if pd.isna(value) or value == '':
                return "Data not available"

            # SPECIAL HANDLING: Convert dispatch choice to descriptive text
            if attribute == 'batt_dispatch_choice':
                return get_dispatch_choice_description(value)

            return value
        else:
            return "N/A"

    # Enhanced retrieval for DataFrame sources
    elif isinstance(source_data, pd.DataFrame):
        if attribute not in source_data.columns:
            is_fcas = 'ancillary' in attribute.lower() or 'fcas' in attribute.lower()
            attr_type = "FCAS" if is_fcas else "regular"
            return "N/A"

        if len(source_data) > 0:
            value = source_data[attribute].iloc[0]

            # Check if the value is NaN or invalid
            if pd.isna(value):
                return "Data not available"

            # SPECIAL HANDLING: Convert dispatch choice to descriptive text
            if attribute == 'batt_dispatch_choice':
                return get_dispatch_choice_description(value)

            return value
        else:
            return "N/A"

    return "N/A"


def perform_calculation(source_data, attribute, calculation):
    """FIXED calculation handling with dynamic period support"""
    try:
        calc_lower = calculation.lower().strip()

        # Handle index-based calculations (e.g., index_1 for Year 1)
        if calc_lower.startswith('index_'):
            try:
                index_num = int(calc_lower.replace('index_', ''))

                if isinstance(source_data, pd.DataFrame):
                    if attribute in source_data.columns and len(source_data) > index_num:
                        value = source_data[attribute].iloc[index_num]
                        return value
                    else:
                        return "Index out of range"
                else:
                    return "Not a DataFrame"
            except ValueError as e:
                return "Invalid index format"

        # Handle single attribute calculations
        if not '|' in attribute:
            if isinstance(source_data, dict):
                if attribute not in source_data:
                    return "N/A"
                value = source_data[attribute]
            elif isinstance(source_data, pd.DataFrame):
                if attribute not in source_data.columns:
                    return "N/A"
                series = source_data[attribute]
            else:
                return "N/A"

            # Single attribute calculations
            if calc_lower == 'sum':
                result = series.sum() if isinstance(source_data, pd.DataFrame) else value
                return result
            elif calc_lower == 'mean' or calc_lower == 'average':
                result = series.mean() if isinstance(source_data, pd.DataFrame) else value
                return result
            elif calc_lower == 'max':
                result = series.max() if isinstance(source_data, pd.DataFrame) else value
                return result
            elif calc_lower == 'min':
                result = series.min() if isinstance(source_data, pd.DataFrame) else value
                return result
            elif calc_lower == 'count':
                result = len(series) if isinstance(source_data, pd.DataFrame) else 1
                return result
            elif calc_lower.startswith('divide_by_'):
                divisor = float(calc_lower.replace('divide_by_', ''))
                base_value = series.sum() if isinstance(source_data, pd.DataFrame) else value
                result = base_value / divisor
                return result
            elif calc_lower.startswith('multiply_by_'):
                multiplier = float(calc_lower.replace('multiply_by_', ''))
                base_value = series.sum() if isinstance(source_data, pd.DataFrame) else value
                result = base_value * multiplier
                return result
            else:
                return "Invalid calculation"

        # Handle multi-attribute calculations
        else:
            attributes = attribute.split('|')

            if isinstance(source_data, dict):
                values = []
                for attr in attributes:
                    if attr in source_data:
                        values.append(float(source_data[attr]) if source_data[attr] not in [None, '', 'N/A'] else 0)
                    else:
                        values.append(0)
            elif isinstance(source_data, pd.DataFrame):
                values = []
                for attr in attributes:
                    if attr in source_data.columns:
                        if len(source_data) > 1:
                            values.append(source_data[attr].sum())
                        else:
                            values.append(source_data[attr].iloc[0] if len(source_data) > 0 else 0)
                    else:
                        values.append(0)
            else:
                return "N/A"

            # Multi-attribute calculations
            if calc_lower == 'sum':
                result = sum(values)
                return result
            elif calc_lower == 'subtract':
                result = values[0]
                for v in values[1:]:
                    result -= v
                return result
            elif calc_lower == 'multiply':
                result = values[0]
                for v in values[1:]:
                    result *= v
                return result
            elif calc_lower == 'divide':
                if len(values) >= 2 and values[1] != 0:
                    result = values[0] / values[1]
                    return result
                else:
                    return "Division error"
            elif calc_lower == 'percentage':
                if len(values) >= 2 and values[1] != 0:
                    result = (values[0] / values[1]) * 100
                    return result
                else:
                    return 0
            elif calc_lower == 'average' or calc_lower == 'mean':
                result = sum(values) / len(values) if values else 0
                return result
            elif calc_lower == 'max':
                result = max(values) if values else 0
                return result
            elif calc_lower == 'min':
                result = min(values) if values else 0
                return result
            else:
                return "Invalid calculation"

    except Exception as e:
        traceback.print_exc()
        return "Calculation error"


# ==============================================================================
# COMPONENT CREATION FUNCTIONS - UPDATED
# ==============================================================================

def get_metric_icon_and_color(component_title, attribute):
    """Optimized icon and color mapping"""
    text_to_check = f"{component_title} {attribute}".lower()

    icon_mappings = {
        'fcas': {'icon': 'fas fa-cogs', 'color': 'success'},
        'ancillary': {'icon': 'fas fa-cogs', 'color': 'success'},
        'arbitrage': {'icon': 'fas fa-chart-line', 'color': 'info'},
        'market': {'icon': 'fas fa-chart-bar', 'color': 'primary'},
        'cost': {'icon': 'fas fa-dollar-sign', 'color': 'warning'},
        'npv': {'icon': 'fas fa-coins', 'color': 'primary'},
        'revenue': {'icon': 'fas fa-money-bill-wave', 'color': 'success'},
        'payback': {'icon': 'fas fa-calendar-alt', 'color': 'info'},
        'lcoe': {'icon': 'fas fa-calculator', 'color': 'secondary'},
        'energy': {'icon': 'fas fa-bolt', 'color': 'warning'},
        'battery': {'icon': 'fas fa-battery-three-quarters', 'color': 'success'},
        'grid': {'icon': 'fas fa-plug', 'color': 'primary'},
        'export': {'icon': 'fas fa-arrow-up', 'color': 'info'},
        'rec': {'icon': 'fas fa-certificate', 'color': 'success'},
    }

    for keyword, styling in icon_mappings.items():
        if keyword in text_to_check:
            return styling

    return {'icon': 'fas fa-chart-simple', 'color': 'primary'}


def has_battery_attributes(attributes):
    """Safely check if any attributes are battery-related"""
    try:
        for attr in attributes:
            if attr is not None and isinstance(attr, str) and 'batt' in attr.lower():
                return True
        return False
    except Exception as e:
        print(f"DEBUG: Error checking battery attributes: {e}")
        return False


def apply_value_modifiers(df_melted, attributes, component, metric_titles):
    """Apply value modifiers to make specified attributes negative"""
    try:
        value_modifiers = component.get('value_modifiers', '')

        if pd.isna(value_modifiers) or value_modifiers == '' or value_modifiers is None:
            return df_melted

        value_modifiers = str(value_modifiers).strip()
        if not value_modifiers or value_modifiers.lower() == 'nan':
            return df_melted

        modifiers = value_modifiers.split('|') if '|' in value_modifiers else [value_modifiers]

        while len(modifiers) < len(attributes):
            modifiers.append('positive')

        for i, attr in enumerate(attributes):
            if pd.isna(attr) or attr is None:
                continue

            attr_str = str(attr).strip()
            if not attr_str:
                continue

            if attr_str in metric_titles:
                display_name = metric_titles[attr_str]
            else:
                display_name = attr_str
                if 'monthly_' in display_name:
                    display_name = display_name.replace('monthly_', '').replace('_', ' ').title()
                elif 'hourly_' in display_name:
                    display_name = display_name.replace('hourly_', '').replace('_', ' ').title()
                elif 'annual_' in display_name:
                    display_name = display_name.replace('annual_', '').replace('_', ' ').title()
                else:
                    display_name = display_name.replace('_', ' ').title()

            modifier = modifiers[i] if i < len(modifiers) else 'positive'

            if isinstance(modifier, str) and modifier.strip().lower() in ['negative', 'neg', '-']:
                mask = df_melted['Series'] == display_name
                matching_rows = mask.sum()

                if matching_rows > 0:
                    df_melted.loc[mask, 'Value'] = -abs(df_melted.loc[mask, 'Value'])

            return df_melted

    except Exception as e:
        print(f"ERROR in apply_value_modifiers: {str(e)}")
        traceback.print_exc()
        return df_melted


def create_metric_card_from_csv(component, data_sources):
    """Enhanced metric card with period awareness"""
    attribute = component['attribute']
    data_source = component['data_source']
    calculation = safe_get_calculation(component)

    # Get the value
    value = get_value_from_data_sources(data_sources, data_source, attribute, calculation)
    formatted_value = format_metric_value(value, component.get('display_format', ''))

    # Enhanced messaging for FCAS metrics
    if value == "N/A" and (
            'fcas' in component['component_title'].lower() or 'ancillary' in component['component_title'].lower()):
        formatted_value = "Not Available"

    styling = get_metric_icon_and_color(component['component_title'], attribute)

    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.I(className=f"{styling['icon']} fa-2x mb-3")
            ], className="text-center"),
            html.H3(formatted_value, className="text-center fw-bold mb-1"),
            html.P(component['component_title'], className="text-center mb-0"),
            html.P(component.get('tooltip', ''), className="text-center small mt-2 opacity-75") if component.get(
                'tooltip') else None
        ])
    ], className="h-100 shadow-sm border-0 metric-card")


def create_panel_component_from_csv(component, data_sources):
    """Panel component creation with period awareness"""
    attributes = component['attribute'].split('|') if '|' in component['attribute'] else [component['attribute']]
    data_source = component['data_source']

    custom_layout = safe_literal_eval(component.get('custom_layout', '{}'))
    metric_titles = custom_layout.get('metricTitles', {})

    panel_items = []
    for attribute in attributes:
        value = get_value_from_data_sources(data_sources, data_source, attribute)
        formatted_value = format_metric_value(value, component.get('display_format', ''))
        title = metric_titles.get(attribute, attribute.replace('_', ' ').title())
        panel_items.append({'title': title, 'value': formatted_value})

    panel_rows = []
    for item in panel_items:
        panel_rows.append(
            html.Div([
                html.Small(item['title'], className="text-muted mb-1 small fw-semibold"),
                html.Div(item['value'], className="fw-bold text-dark h6 mb-3")
            ], style={'margin-bottom': '1rem', 'border-bottom': '1px solid #eee', 'padding-bottom': '0.5rem'})
        )

    header_class = "bg-primary"
    if 'fcas' in component['component_title'].lower():
        header_class = "bg-success"
    elif 'arbitrage' in component['component_title'].lower():
        header_class = "bg-info"

    card = dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="fas fa-chart-bar me-2"),
                component['component_title']
            ], className="text-white mb-0")
        ], className=header_class),
        dbc.CardBody(panel_rows, className="bg-light")
    ])

    return html.Div(
        card,
        style={
            'width': '400px',
            'min-width': '400px',
            'flex': '0 0 400px',
            'margin-right': '20px',
            'margin-bottom': '1rem'
        }
    )


def create_pipe_separated_table(component, data_sources):
    """Enhanced table with dynamic analysis period support"""
    print(f"DEBUG: Processing table: {component.get('component_title', 'Unknown')}")

    # Get analysis period from data sources
    analysis_period = get_analysis_period_from_metadata(data_sources)

    data_file = component.get('data_source', '')

    # Handle multiple data sources
    if '|' in data_file:
        source_files = data_file.split('|')
    else:
        source_files = [data_file]

    # Handle pipe-separated attributes
    if '|' in component['attribute']:
        attributes = component['attribute'].split('|')
    else:
        attributes = [component['attribute']]

    # Ensure we have enough source files for all attributes
    while len(source_files) < len(attributes):
        source_files.append(source_files[-1])

    # Handle pipe-separated display formats properly
    display_formats = []
    if component.get('display_format') and '|' in component.get('display_format', ''):
        display_formats = component['display_format'].split('|')
    else:
        display_formats = [component.get('display_format', '')] * len(attributes)

    # Handle pipe-separated calculations with NaN safety
    calculations = []
    calc_value = safe_get_calculation(component)

    if calc_value and '|' in calc_value:
        calculations = calc_value.split('|')
    else:
        calculations = [calc_value] * len(attributes)

    # Ensure we have enough formats and calculations for all attributes
    while len(display_formats) < len(attributes):
        display_formats.append('')
    while len(calculations) < len(attributes):
        calculations.append('')

    # Get custom layout with dynamic period support
    custom_layout_raw = component.get('custom_layout', '{}')
    custom_layout = safe_literal_eval(custom_layout_raw)
    metric_titles = custom_layout.get('metricTitles', {})

    # Get dynamic column configuration
    column_config = get_dynamic_column_config(analysis_period, custom_layout)
    show_total = column_config.get('showTotal', False)

    print(f"DEBUG: Analysis period: {analysis_period}, show_total: {show_total}")

    # Create table data structure
    table_data = []

    for i, attr in enumerate(attributes):
        source_file = source_files[i] if i < len(source_files) else source_files[0]
        calculation = calculations[i] if i < len(calculations) else ''
        display_format = display_formats[i] if i < len(display_formats) else ''

        # Check if source file exists
        if source_file not in data_sources:
            print(f"Warning: Data source '{source_file}' not found for attribute '{attr}'")
            continue

        # Get Year 1 value with PROPER calculation handling
        year1_value = get_value_from_data_sources(data_sources, source_file, attr, calculation)

        # Get total value if requested (dynamic period)
        total_value = "N/A"
        if show_total:
            if source_file.endswith('.parquet'):
                total_value = get_value_from_data_sources(data_sources, source_file, attr, 'sum')
            elif year1_value != "N/A" and isinstance(year1_value, (int, float)):
                total_value = year1_value * analysis_period

        if year1_value != "N/A":
            # Get display name from custom titles
            display_name = metric_titles.get(attr, attr.replace('_', ' ').replace('.', ' ').title())

            # Format values with corresponding display format
            formatted_year1 = format_metric_value(year1_value, display_format)
            formatted_total = format_metric_value(total_value, display_format) if total_value != "N/A" else "N/A"

            row_data = {
                'Metric': display_name,
                column_config.get('year1Header', 'Year 1'): formatted_year1
            }

            # Add total column if requested (dynamic header)
            if show_total:
                row_data[column_config.get('totalHeader', f'{analysis_period}-Year Total')] = formatted_total

            table_data.append(row_data)

    if not table_data:
        return dbc.Alert("No data available for the specified attributes", color="info")

    # Create columns dynamically based on configuration
    columns = [{"name": "Metric", "id": "Metric"}]
    columns.append({
        "name": column_config.get('year1Header', 'Year 1'),
        "id": column_config.get('year1Header', 'Year 1')
    })

    if show_total:
        total_header = column_config.get('totalHeader', f'{analysis_period}-Year Total')
        columns.append({
            "name": total_header,
            "id": total_header
        })

    # Create the table
    return dbc.Card([
        dbc.CardHeader([
            html.H5(component['component_title'], className="mb-0 text-primary"),
            html.Small(f"Analysis period: {analysis_period} years", className="text-muted")
        ]),
        dbc.CardBody([
            dash_table.DataTable(
                columns=columns,
                data=table_data,
                style_table={
                    'overflowX': 'auto',
                    'borderRadius': '0 0 15px 15px'
                },
                style_cell={
                    'padding': '12px 15px',
                    'textAlign': 'left',
                    'fontFamily': 'Segoe UI',
                    'fontSize': '14px',
                    'border': 'none'
                },
                style_header={
                    'backgroundColor': '#667eea',
                    'color': 'white',
                    'fontWeight': 'bold',
                    'textAlign': 'center',
                    'padding': '15px 12px',
                    'border': 'none'
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': '#f9fafb'
                    },
                    {
                        'if': {'column_id': 'Metric'},
                        'fontWeight': '600',
                        'backgroundColor': '#f8fafc',
                        'borderRight': '2px solid #d1d5db'
                    },
                    {
                        'if': {'column_id': [column_config.get('year1Header', 'Year 1'),
                                             column_config.get('totalHeader', f'{analysis_period}-Year Total')]},
                        'textAlign': 'right',
                        'fontWeight': 'bold'
                    }
                ],
                style_cell_conditional=[
                                           {
                                               'if': {'column_id': 'Metric'},
                                               'minWidth': '200px',
                                               'width': '50%' if show_total else '60%'
                                           },
                                           {
                                               'if': {'column_id': column_config.get('year1Header', 'Year 1')},
                                               'minWidth': '120px',
                                               'width': '25%' if show_total else '40%'
                                           }
                                       ] + ([{
                    'if': {'column_id': column_config.get('totalHeader', f'{analysis_period}-Year Total')},
                    'minWidth': '150px',
                    'width': '25%'
                }] if show_total else [])
            )
        ])
    ], className="performance-table")


def create_generic_landscape_table(component, data_sources):
    """Enhanced landscape table with dynamic analysis period support"""
    # Get analysis period from data sources
    analysis_period = get_analysis_period_from_metadata(data_sources)

    data_file = component.get('data_source', '')

    # Handle pipe-separated data sources
    if '|' in data_file:
        source_files = data_file.split('|')
        selected_source = None
        for source_file in source_files:
            if source_file.strip() in data_sources:
                selected_source = source_file.strip()
                break

        if not selected_source:
            return dbc.Alert(f"No valid data sources found in: {data_file}", color="warning")
        data_file = selected_source

    if data_file not in data_sources:
        return dbc.Alert(f"Data file not found: {data_file}", color="warning")

    df = data_sources[data_file].copy()
    if not isinstance(df, pd.DataFrame):
        return dbc.Alert(f"Data source is not a DataFrame: {data_file}", color="warning")

    try:
        # Parse attributes
        if '|' in component['attribute']:
            requested_attributes = [attr.strip() for attr in component['attribute'].split('|')]
        else:
            requested_attributes = [component['attribute'].strip()]

        # Find which attributes exist in the dataframe
        available_attributes = [attr for attr in requested_attributes if attr in df.columns]

        if not available_attributes:
            return dbc.Alert(f"No data columns found for: {', '.join(requested_attributes)}", color="info")

        custom_layout = safe_literal_eval(component.get('custom_layout', '{}'))
        metric_titles = custom_layout.get('metricTitles', {})

        # Create landscape data with dynamic period limits
        landscape_data = []
        df_subset = df[available_attributes].copy()

        # Limit to analysis period (+ year 0)
        max_years = min(len(df_subset), analysis_period + 1)
        year_labels = get_period_labels(analysis_period)[:max_years]

        # Process each available attribute
        for attr in available_attributes:
            display_name = metric_titles.get(attr,
                                             attr.replace('cf_', '')
                                             .replace('ancillary_services_', 'FCAS ')
                                             .replace('_', ' ')
                                             .title())

            row_data = {'Metric': display_name}

            # Add data for each year (limited by analysis period)
            for i, year_label in enumerate(year_labels):
                if i < len(df_subset):
                    value = df_subset[attr].iloc[i]
                    formatted_value = format_landscape_currency(value)
                    row_data[year_label] = formatted_value
                else:
                    row_data[year_label] = "$0"

            landscape_data.append(row_data)

        if not landscape_data:
            return dbc.Alert("No valid data available for landscape table", color="info")

        # Create columns for table
        columns = [{"name": "Metric", "id": "Metric", "type": "text"}]
        for year_label in year_labels:
            columns.append({"name": year_label, "id": year_label, "type": "text"})

        # Ensure all rows have all columns
        for row in landscape_data:
            for col in columns:
                if col['id'] not in row:
                    row[col['id']] = "$0"

        # Header color based on component title
        def get_header_color_class(component_title):
            title_lower = component_title.lower()
            if 'fcas' in title_lower or 'ancillary' in title_lower:
                return "#198754", "bg-success"
            elif 'cashflow' in title_lower or 'cash flow' in title_lower:
                return "#0d6efd", "bg-primary"
            elif 'payback' in title_lower:
                return "#fd7e14", "bg-warning"
            elif 'revenue' in title_lower:
                return "#20c997", "bg-info"
            elif 'cost' in title_lower or 'expense' in title_lower:
                return "#dc3545", "bg-danger"
            else:
                return "#6c757d", "bg-secondary"

        header_color, header_class = get_header_color_class(component['component_title'])

        # Create the table card with period info
        return dbc.Card([
            dbc.CardHeader([
                html.Div([
                    html.H6([
                        html.I(className="fas fa-table me-2"),
                        component['component_title']
                    ], className="mb-1 text-white"),
                    html.Small([
                        f"Analysis period: {analysis_period} years • ",
                        f"Showing {len(available_attributes)} of {len(requested_attributes)} metrics",
                        " • Scroll horizontally to view all years" if len(year_labels) > 6 else ""
                    ], className="text-white opacity-75")
                ])
            ], className=header_class),
            dbc.CardBody([
                html.Div([
                    dash_table.DataTable(
                        columns=columns,
                        data=landscape_data,
                        fixed_columns={'headers': True, 'data': 1},
                        style_table={
                            'overflowX': 'auto',
                            'minWidth': '100%',
                            'maxHeight': '500px',
                            'overflowY': 'auto'
                        },
                        style_cell={
                            'padding': '8px 12px',
                            'textAlign': 'center',
                            'whiteSpace': 'normal',
                            'height': 'auto',
                            'fontFamily': 'Segoe UI',
                            'fontSize': '12px',
                            'minWidth': '80px',
                            'border': '1px solid #dee2e6'
                        },
                        style_cell_conditional=[
                            {
                                'if': {'column_id': 'Metric'},
                                'textAlign': 'left',
                                'fontWeight': 'bold',
                                'backgroundColor': '#f8f9fa',
                                'minWidth': '200px',
                                'width': '200px',
                                'maxWidth': '200px',
                                'borderRight': '2px solid #6c757d'
                            }
                        ],
                        style_header={
                            'backgroundColor': header_color,
                            'color': 'white',
                            'fontWeight': 'bold',
                            'textAlign': 'center',
                            'fontSize': '11px',
                            'padding': '10px 8px',
                            'border': '1px solid rgba(255,255,255,0.2)'
                        },
                        style_data_conditional=[
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': '#f8f9fa'
                            },
                            {
                                'if': {'row_index': 'even'},
                                'backgroundColor': 'white'
                            }
                        ],
                        export_format="xlsx",
                        export_headers="display",
                        filter_action="none",
                        sort_action="native"
                    )
                ], className="table-responsive")
            ], className="p-0")
        ], className="shadow-sm border-0")

    except Exception as e:
        print(f"ERROR: Exception in create_generic_landscape_table: {str(e)}")
        traceback.print_exc()
        return dbc.Alert([
            html.H6("Table Creation Error", className="alert-heading"),
            html.P(f"Failed to create landscape table: {component.get('component_title', 'Unknown')}"),
            html.Hr(),
            html.Small(f"Technical details: {str(e)}", className="text-muted")
        ], color="danger")


def create_graph_from_csv(component, data_sources):
    """Enhanced graph creation with dynamic analysis period support"""
    # Get analysis period from data sources
    analysis_period = get_analysis_period_from_metadata(data_sources)

    data_file = component.get('data_source', '')

    if data_file not in data_sources:
        return dbc.Alert(f"Data file not found: {data_file}", color="warning")

    source_data = data_sources[data_file]

    # Handle calculated data for service comparison charts
    if data_file == 'calculated' and component['attribute'] == 'service_revenue_comparison':
        service_data = source_data.get('service_revenue_comparison', [])
        if not service_data or len(service_data) == 0:
            return dbc.Alert("No FCAS service revenue data available", color="info")

        df = pd.DataFrame(service_data)
        fig = px.bar(df, x='Service', y='Revenue',
                     title=component['component_title'],
                     color_discrete_sequence=['#10b981'])

        fig.update_layout(
            xaxis_title="FCAS Service",
            yaxis_title="Revenue ($)",
            height=400
        )

        return dbc.Card([
            dbc.CardBody([
                dcc.Graph(figure=fig, style={'height': '400px'})
            ])
        ], className="shadow-sm border-0")

    # Handle revenue breakdown pie charts
    if data_file == 'calculated' and component['attribute'] == 'revenue_breakdown_all':
        revenue_data = source_data.get('revenue_breakdown_all', [])
        if not revenue_data or len(revenue_data) == 0:
            return dbc.Alert("No revenue breakdown data available", color="info")

        # Filter out zero or invalid revenues
        valid_revenue_data = []
        for item in revenue_data:
            revenue = safe_float_conversion(item.get('Revenue', 0))
            if revenue > 0:  # Only include positive revenues
                valid_revenue_data.append({
                    'Source': item.get('Source', 'Unknown'),
                    'Revenue': revenue,
                    'Type': item.get('Type', 'Other')
                })

        if not valid_revenue_data:
            return dbc.Alert("No valid revenue data available for chart", color="info")

        df = pd.DataFrame(valid_revenue_data)

        # Create pie chart with only valid data
        fig = px.pie(df, values='Revenue', names='Source',
                     title=component['component_title'],
                     color_discrete_sequence=px.colors.qualitative.Set3)

        fig.update_layout(height=400)

        return dbc.Card([
            dbc.CardBody([
                dcc.Graph(figure=fig, style={'height': '400px'})
            ])
        ], className="shadow-sm border-0")

    if not isinstance(source_data, pd.DataFrame):
        return dbc.Alert(f"Data source is not a DataFrame: {data_file}", color="warning")

    df = source_data.copy()
    attributes = component['attribute'].split('|') if '|' in component['attribute'] else [component['attribute']]

    # Check which attributes exist in the dataframe
    available_attributes = [attr for attr in attributes if attr in df.columns]

    if not available_attributes:
        return dbc.Alert(f"No data columns found for chart: {', '.join(attributes)}", color="info")

    # Continue with only available attributes
    attributes = available_attributes

    visual_props = safe_literal_eval(component.get('visual_properties', '{}'))
    custom_layout = safe_literal_eval(component.get('custom_layout', '{}'))
    metric_titles = custom_layout.pop('metricTitles', {}) if custom_layout else {}

    graph_type = component.get('graph_type', 'line')
    if pd.isna(graph_type):
        graph_type = 'line'
    else:
        graph_type = str(graph_type).strip().lower()

    # PIE CHART HANDLING - Enhanced with period awareness
    if graph_type == 'pie':
        # Get calculations for each attribute
        calculation = safe_get_calculation(component)
        if calculation and '|' in calculation:
            calculations = calculation.split('|')
        else:
            calculations = [calculation] * len(attributes)

        # Ensure equal lengths
        while len(calculations) < len(attributes):
            calculations.append('')

        # Collect pie chart data
        pie_values = []
        pie_labels = []

        for i, attr in enumerate(attributes):
            if attr in df.columns:
                calc = calculations[i] if i < len(calculations) else ''

                # Get value based on calculation
                if calc == 'index_1':
                    value = df[attr].iloc[1] if len(df) > 1 else (df[attr].iloc[0] if len(df) > 0 else 0)
                elif calc == 'sum':
                    value = df[attr].sum()
                elif calc == 'mean':
                    value = df[attr].mean()
                else:
                    # Default to first value if no calculation specified
                    value = df[attr].iloc[0] if len(df) > 0 else 0

                # Only include non-zero values
                if value > 0:
                    pie_values.append(value)

                    # Use custom metric titles for pie chart labels
                    if attr in metric_titles:
                        label = metric_titles[attr]
                    elif custom_layout.get('labels') and i < len(custom_layout['labels']):
                        label = custom_layout['labels'][i]
                    else:
                        # Generate label from attribute name
                        label = attr.replace('cf_ancillary_services_', 'Service ').replace('_revenue', '').replace(
                            '_', ' ').title()

                    pie_labels.append(label)

        if not pie_values:
            return dbc.Alert("No non-zero values found for pie chart", color="info")

        # Modern color palette
        modern_colors = [
            '#667eea', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
            '#06b6d4', '#84cc16', '#f97316', '#ec4899', '#6b7280'
        ]

        # Create enhanced pie chart
        fig = go.Figure()

        fig.add_trace(go.Pie(
            labels=pie_labels,
            values=pie_values,
            hole=custom_layout.get('hole', 0.4),
            hovertemplate='<b>%{label}</b><br>' +
                          'Value: <b>%{value:$,.0f}</b><br>' +
                          'Percentage: <b>%{percent}</b><br>' +
                          '<extra></extra>',
            textinfo='label+percent',
            textposition='auto',
            textfont=dict(size=12, color='white', family='Segoe UI'),
            marker=dict(
                colors=modern_colors[:len(pie_values)],
                line=dict(color='white', width=2)
            ),
            pull=[0.05 if i == 0 else 0 for i in range(len(pie_values))],
            rotation=90,
            sort=False
        ))

        # Enhanced layout styling
        fig.update_layout(
            title=dict(
                text=custom_layout.get('title', component['component_title']),
                x=0.5, y=0.95, xanchor='center', yanchor='top',
                font=dict(size=18, color='#2d3748', family='Segoe UI', weight='bold')
            ),
            showlegend=custom_layout.get('showlegend', True),
            legend=dict(
                orientation='v', yanchor='middle', y=0.5, xanchor='left', x=1.02,
                font=dict(size=11, color='#374151', family='Segoe UI'),
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='rgba(0,0,0,0.1)', borderwidth=1
            ),
            height=visual_props.get('height', 500),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Segoe UI", size=12, color='#374151'),
            margin=dict(l=20, r=120, t=60, b=20),
            annotations=[
                dict(
                    text=f'<b>Total</b><br>${sum(pie_values):,.0f}<br><small>{analysis_period} years</small>',
                    x=0.5, y=0.5, font_size=14, font_color='#374151',
                    font_family='Segoe UI', showarrow=False
                )
            ] if custom_layout.get('hole', 0.4) > 0 else []
        )

        return dbc.Card([
            dbc.CardHeader([
                html.H5([
                    html.I(className="fas fa-chart-pie me-2 text-primary"),
                    component['component_title']
                ], className="mb-0"),
                html.Small(f"Analysis period: {analysis_period} years", className="text-muted")
            ], className="bg-light border-0"),
            dbc.CardBody([
                dcc.Graph(
                    figure=fig,
                    style={'height': f'{visual_props.get("height", 500)}px'},
                    config={'displayModeBar': True, 'displaylogo': False}
                )
            ], className="p-3")
        ], className="shadow-sm border-0")

    # WATERFALL CHART HANDLING with dynamic period
    if graph_type == 'waterfall':
        calculation = safe_get_calculation(component)
        calculations = calculation.split('|') if calculation and '|' in calculation else [calculation] * len(attributes)

        while len(calculations) < len(attributes):
            calculations.append('')

        # Category mapping for better names
        category_mapping = {
            'cf_energy_value': 'Energy Revenue',
            'cf_payback_with_expenses': 'Project Cost',
            'cf_energy_purchases': 'Energy Purchases',
            'cf_operating_expenses': 'O&M Costs',
            'cf_insurance_expense': 'Insurance',
            'cf_battery_replacement_cost': 'Battery Replacement',
            'cf_fed_tax_savings': 'Tax Benefits',
            'cf_debt_payment_total': 'Total Debt Repayment',
            'cf_debt_payment_interest': 'Total Interest Paid',
            'cf_after_tax_cash_flow': 'After Tax Cashflow'
        }

        # Calculate values for waterfall
        waterfall_data = []
        for i, attr in enumerate(attributes):
            if attr in df.columns:
                if calculations[i] == 'sum':
                    value = df[attr].sum()
                else:
                    value = df[attr].iloc[0] if len(df) > 0 else 0

                # Use custom metric titles for waterfall categories
                if attr in metric_titles:
                    category_name = metric_titles[attr]
                else:
                    category_name = category_mapping.get(attr, attr.replace('cf_', '').replace('_', ' ').title())

                # Determine type based on attribute name and position
                if 'purchases' in attr or 'cost' in attr or 'expense' in attr or 'payment' in attr:
                    waterfall_type = 'negative'
                    value = -abs(value)
                else:
                    waterfall_type = 'positive'

                waterfall_data.append({
                    'category': category_name,
                    'value': value,
                    'type': waterfall_type
                })

        # Create waterfall chart
        if waterfall_data:
            categories = [item['category'] for item in waterfall_data]
            values = [item['value'] for item in waterfall_data]

            fig = go.Figure()
            fig.add_trace(go.Waterfall(
                name="Financial Flow",
                orientation="v",
                measure=["relative" if item['type'] != 'total' else "total" for item in waterfall_data],
                x=categories,
                y=values,
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                increasing={"marker": {"color": "#10b981"}},
                decreasing={"marker": {"color": "#ef4444"}},
                totals={"marker": {"color": "#3b82f6"}}
            ))

            fig.update_layout(
                title=f"{custom_layout.get('title', component['component_title'])} ({analysis_period} years)",
                title_x=0.5,
                height=visual_props.get('height', 500),
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Segoe UI", size=12),
                yaxis=dict(title="Value ($)", tickformat="$,.0f", gridcolor="#f0f0f0"),
                xaxis=dict(title="Financial Components", tickangle=-45)
            )

            return dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig, style={'height': f'{visual_props.get("height", 500)}px'})
                ])
            ], className="waterfall-container")
        else:
            return dbc.Alert("No waterfall data available", color="info")

    # Continue with existing chart types with period awareness
    try:
        # Better index handling based on data type
        primary_attr = attributes[0]
        df = df.reset_index()

        # Determine the appropriate x-axis column
        if 'index' in df.columns:
            x_col = 'index'
        elif len(df.columns) > 0:
            x_col = df.columns[0]
        else:
            return dbc.Alert("No suitable x-axis column found", color="warning")

        fig = None

        # Handle quarterly/historical data
        if 'Quarter' in df.columns or data_file.endswith('_historical.csv'):
            x_col = 'Quarter'

            if len(attributes) > 1:
                df_subset = df[attributes + ['Quarter']].copy()

                # Apply metric titles
                df_display = df_subset.copy()
                for attr in attributes:
                    if attr in metric_titles:
                        clean_name = metric_titles[attr]
                    else:
                        clean_name = attr.replace('AVG_', '').replace(' ($/MW)', '').replace('_', ' ').title()
                        clean_name = clean_name.replace('Lowerreg', 'Lower Reg').replace('Raisereg', 'Raise Reg')
                        clean_name = clean_name.replace('Lower5min', 'Lower 5min').replace('Raise5min', 'Raise 5min')
                        clean_name = clean_name.replace('Lower60sec', 'Lower 60s').replace('Raise60sec', 'Raise 60s')
                        clean_name = clean_name.replace('Lower6sec', 'Lower 6s').replace('Raise6sec', 'Raise 6s')
                    df_display = df_display.rename(columns={attr: clean_name})

                display_attributes = [metric_titles.get(attr,
                                                        attr.replace('AVG_', '').replace(' ($/MW)', '')
                                                        .replace('_', ' ').title()
                                                        .replace('Lowerreg', 'Lower Reg').replace('Raisereg',
                                                                                                  'Raise Reg')
                                                        .replace('Lower5min', 'Lower 5min').replace('Raise5min',
                                                                                                    'Raise 5min')
                                                        .replace('Lower60sec', 'Lower 60s').replace('Raise60sec',
                                                                                                    'Raise 60s')
                                                        .replace('Lower6sec', 'Lower 6s').replace('Raise6sec',
                                                                                                  'Raise 6s'))
                                      for attr in attributes]

                df_melted = df_display.melt(id_vars=['Quarter'],
                                            value_vars=display_attributes,
                                            var_name='Series',
                                            value_name='Value')

                df_melted = apply_value_modifiers(df_melted, attributes, component, metric_titles)

                fig = px.line(df_melted, x='Quarter', y='Value', color='Series',
                              title=component['component_title'],
                              color_discrete_sequence=px.colors.qualitative.Set3)

                fig.update_layout(
                    xaxis_title="Quarter",
                    yaxis_title="Price ($/MW)",
                    height=visual_props.get('height', 450),
                    xaxis={'tickangle': 45}
                )
            else:
                if graph_type == 'line':
                    fig = px.line(df, x=x_col, y=primary_attr, title=component['component_title'])
                elif graph_type == 'bar':
                    fig = px.bar(df, x=x_col, y=primary_attr, title=component['component_title'])
                else:
                    fig = px.line(df, x=x_col, y=primary_attr, title=component['component_title'])

                fig.update_layout(
                    xaxis_title="Quarter",
                    yaxis_title=primary_attr.replace('_', ' ').title(),
                    height=visual_props.get('height', 450)
                )

        # Handle monthly data with proper month labeling
        elif 'monthly' in data_file and len(df) == 12:
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            df['month_name'] = month_names[:len(df)]
            x_col = 'month_name'

            # Handle multiple attributes for ALL chart types
            if len(attributes) > 1:
                df_subset = df[attributes + ['month_name']].copy()

                # Use metric titles if available
                df_display = df_subset.copy()
                for attr in attributes:
                    if attr in metric_titles:
                        clean_name = metric_titles[attr]
                    else:
                        clean_name = attr.replace('monthly_', '').replace('_', ' ').title()
                    df_display = df_display.rename(columns={attr: clean_name})

                display_attributes = [metric_titles.get(attr, attr.replace('monthly_', '').replace('_', ' ').title())
                                      for attr in attributes]

                df_melted = df_display.melt(id_vars=['month_name'],
                                            value_vars=display_attributes,
                                            var_name='Series',
                                            value_name='Value')

                df_melted = apply_value_modifiers(df_melted, attributes, component, metric_titles)

                # Create charts based on graph_type
                if graph_type == 'stacked_bar':
                    fig = px.bar(df_melted, x='month_name', y='Value', color='Series',
                                 title=component['component_title'],
                                 color_discrete_sequence=px.colors.qualitative.Set3)
                    fig.update_layout(barmode='stack')
                elif graph_type == 'grouped_bar':
                    fig = px.bar(df_melted, x='month_name', y='Value', color='Series',
                                 title=component['component_title'],
                                 color_discrete_sequence=px.colors.qualitative.Set3,
                                 barmode='group')
                elif graph_type in ['area', 'stacked_area']:
                    should_stack = (graph_type == 'stacked_area')
                    fig = go.Figure()
                    colors = px.colors.qualitative.Set3

                    for i, display_attr in enumerate(display_attributes):
                        series_data = df_melted[df_melted['Series'] == display_attr]

                        if should_stack:
                            fill_mode = 'tonexty' if i > 0 else 'tozeroy'
                            opacity = 0.7
                            hovermode = 'closest'
                        else:
                            fill_mode = 'tozeroy'
                            opacity = 0.4
                            hovermode = 'x unified'

                        base_color = colors[i % len(colors)]
                        if base_color.startswith('#'):
                            r = int(base_color[1:3], 16)
                            g = int(base_color[3:5], 16)
                            b = int(base_color[5:7], 16)
                            fillcolor = f'rgba({r},{g},{b},{opacity})'
                        else:
                            fillcolor = base_color

                        fig.add_trace(go.Scatter(
                            x=series_data['month_name'],
                            y=series_data['Value'],
                            mode='lines',
                            fill=fill_mode,
                            name=display_attr,
                            line=dict(color=base_color, width=2),
                            fillcolor=fillcolor,
                            hovertemplate='<b>%{fullData.name}</b><br>' +
                                          '%{x}<br>' +
                                          'Value: %{y:$,.0f}<br>' +
                                          '<extra></extra>'
                        ))

                    fig.update_layout(
                        title=component['component_title'],
                        xaxis_title="Month",
                        yaxis_title="Energy Flow (kWh)" if has_battery_attributes(attributes) else "Value",
                        height=visual_props.get('height', 400),
                        hovermode=hovermode,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )

                    if has_battery_attributes(attributes):
                        fig.add_hline(y=0, line_dash="dash", line_color="gray",
                                      annotation_text="Zero Line", annotation_position="bottom right")

                elif graph_type == 'line':
                    fig = px.line(df_melted, x='month_name', y='Value', color='Series',
                                  title=component['component_title'],
                                  color_discrete_sequence=px.colors.qualitative.Set3)
                else:
                    fig = px.line(df_melted, x='month_name', y='Value', color='Series',
                                  title=component['component_title'],
                                  color_discrete_sequence=px.colors.qualitative.Set3)
            else:
                # Single attribute charts
                if graph_type == 'bar':
                    fig = px.bar(df, x=x_col, y=primary_attr,
                                 title=component['component_title'],
                                 color_discrete_sequence=['#10b981'])
                elif graph_type == 'line':
                    fig = px.line(df, x=x_col, y=primary_attr,
                                  title=component['component_title'])
                elif graph_type in ['area', 'stacked_area']:
                    fig = px.area(df, x=x_col, y=primary_attr,
                                  title=component['component_title'])
                else:
                    fig = px.bar(df, x=x_col, y=primary_attr,
                                 title=component['component_title'])

            # Update layout for monthly data
            if len(attributes) == 1 or graph_type in ['stacked_bar', 'grouped_bar', 'line']:
                fig.update_layout(
                    xaxis_title="Month",
                    xaxis={'categoryorder': 'array', 'categoryarray': month_names},
                    yaxis_title=component.get('display_format', '').replace('{', '').replace('}', '').replace(':', ''),
                    height=visual_props.get('height', 400)
                )

        elif 'hourly' in data_file and len(df) == 8760:
            # Handle hourly data (like batt_SOC_year1) with period validation
            df['hour'] = range(1, len(df) + 1)
            x_col = 'hour'

            # Handle multiple attributes for hourly data
            if len(attributes) > 1:
                df_subset = df[attributes + ['hour']].copy()

                # Apply metric titles
                df_display = df_subset.copy()
                for attr in attributes:
                    if attr in metric_titles:
                        clean_name = metric_titles[attr]
                    else:
                        clean_name = attr.replace('_', ' ').title()
                    df_display = df_display.rename(columns={attr: clean_name})

                display_attributes = [metric_titles.get(attr, attr.replace('_', ' ').title())
                                      for attr in attributes]

                df_melted = df_display.melt(id_vars=['hour'],
                                            value_vars=display_attributes,
                                            var_name='Series',
                                            value_name='Value')

                df_melted = apply_value_modifiers(df_melted, attributes, component, metric_titles)

                if graph_type in ['area', 'stacked_area']:
                    should_stack = (graph_type == 'stacked_area')
                    fig = go.Figure()
                    colors = px.colors.qualitative.Set3

                    for i, display_attr in enumerate(display_attributes):
                        series_data = df_melted[df_melted['Series'] == display_attr]

                        if should_stack:
                            fill_mode = 'tonexty' if i > 0 else 'tozeroy'
                            opacity = 0.7
                            hovermode = 'closest'
                        else:
                            fill_mode = 'tozeroy'
                            opacity = 0.4
                            hovermode = 'x unified'

                        base_color = colors[i % len(colors)]
                        if base_color.startswith('#'):
                            r = int(base_color[1:3], 16)
                            g = int(base_color[3:5], 16)
                            b = int(base_color[5:7], 16)
                            fillcolor = f'rgba({r},{g},{b},{opacity})'
                        else:
                            fillcolor = base_color

                        fig.add_trace(go.Scatter(
                            x=series_data['hour'],
                            y=series_data['Value'],
                            mode='lines',
                            fill=fill_mode,
                            name=display_attr,
                            line=dict(color=base_color, width=2),
                            fillcolor=fillcolor,
                            hovertemplate='<b>%{fullData.name}</b><br>Hour: %{x}<br>Value: %{y:$,.0f}<br><extra></extra>'
                        ))

                    fig.update_layout(
                        title=component['component_title'],
                        xaxis_title="Hour of Year",
                        yaxis_title="Value",
                        height=visual_props.get('height', 400),
                        hovermode=hovermode
                    )
                elif graph_type == 'line':
                    fig = px.line(df_melted, x='hour', y='Value', color='Series',
                                  title=component['component_title'])
                else:
                    fig = px.line(df_melted, x='hour', y='Value', color='Series',
                                  title=component['component_title'])
            else:
                # Single attribute
                if graph_type == 'line':
                    fig = px.line(df, x=x_col, y=primary_attr, title=component['component_title'])
                elif graph_type in ['area', 'stacked_area']:
                    fig = px.area(df, x=x_col, y=primary_attr, title=component['component_title'])
                else:
                    fig = px.line(df, x=x_col, y=primary_attr, title=component['component_title'])

            fig.update_layout(
                xaxis_title="Hour of Year",
                yaxis_title=primary_attr.replace('_', ' ').title() if len(attributes) == 1 else "Value",
                height=visual_props.get('height', 400)
            )

        elif 'annual' in data_file:
            # Handle annual data with dynamic period validation
            df = validate_dataframe_length(df, analysis_period, data_file)
            df['year'] = range(len(df))
            x_col = 'year'

            # Handle multiple attributes for annual data
            if len(attributes) > 1:
                df_subset = df[attributes + ['year']].copy()

                # Apply metric titles
                df_display = df_subset.copy()
                for attr in attributes:
                    if attr in metric_titles:
                        clean_name = metric_titles[attr]
                    else:
                        clean_name = attr.replace('_', ' ').title()
                    df_display = df_display.rename(columns={attr: clean_name})

                display_attributes = [metric_titles.get(attr, attr.replace('_', ' ').title())
                                      for attr in attributes]

                df_melted = df_display.melt(id_vars=['year'],
                                            value_vars=display_attributes,
                                            var_name='Series',
                                            value_name='Value')

                df_melted = apply_value_modifiers(df_melted, attributes, component, metric_titles)

                if graph_type in ['area', 'stacked_area']:
                    should_stack = (graph_type == 'stacked_area')
                    fig = go.Figure()
                    colors = px.colors.qualitative.Set3

                    for i, display_attr in enumerate(display_attributes):
                        series_data = df_melted[df_melted['Series'] == display_attr]

                        if should_stack:
                            fill_mode = 'tonexty' if i > 0 else 'tozeroy'
                            opacity = 0.7
                            hovermode = 'closest'
                        else:
                            fill_mode = 'tozeroy'
                            opacity = 0.4
                            hovermode = 'x unified'

                        base_color = colors[i % len(colors)]
                        if base_color.startswith('#'):
                            r = int(base_color[1:3], 16)
                            g = int(base_color[3:5], 16)
                            b = int(base_color[5:7], 16)
                            fillcolor = f'rgba({r},{g},{b},{opacity})'
                        else:
                            fillcolor = base_color

                        fig.add_trace(go.Scatter(
                            x=series_data['year'],
                            y=series_data['Value'],
                            mode='lines',
                            fill=fill_mode,
                            name=display_attr,
                            line=dict(color=base_color, width=2),
                            fillcolor=fillcolor,
                            hovertemplate='<b>%{fullData.name}</b><br>Year: %{x}<br>Value: %{y:$,.0f}<br><extra></extra>'
                        ))

                    fig.update_layout(
                        title=f"{component['component_title']} ({analysis_period} years)",
                        xaxis_title="Year",
                        yaxis_title="Value",
                        height=visual_props.get('height', 400),
                        hovermode=hovermode
                    )
                elif graph_type == 'line':
                    fig = px.line(df_melted, x='year', y='Value', color='Series',
                                  title=f"{component['component_title']} ({analysis_period} years)",
                                  color_discrete_sequence=px.colors.qualitative.Set3)
                elif graph_type == 'bar':
                    fig = px.bar(df_melted, x='year', y='Value', color='Series',
                                 title=f"{component['component_title']} ({analysis_period} years)")
                else:
                    fig = px.line(df_melted, x='year', y='Value', color='Series',
                                  title=f"{component['component_title']} ({analysis_period} years)")
            else:
                # Single attribute
                if graph_type == 'line':
                    fig = px.line(df, x=x_col, y=primary_attr,
                                  title=f"{component['component_title']} ({analysis_period} years)")
                elif graph_type == 'bar':
                    fig = px.bar(df, x=x_col, y=primary_attr,
                                 title=f"{component['component_title']} ({analysis_period} years)")
                elif graph_type in ['area', 'stacked_area']:
                    fig = px.area(df, x=x_col, y=primary_attr,
                                  title=f"{component['component_title']} ({analysis_period} years)")
                else:
                    fig = px.line(df, x=x_col, y=primary_attr,
                                  title=f"{component['component_title']} ({analysis_period} years)")

            fig.update_layout(
                xaxis_title="Year",
                yaxis_title=primary_attr.replace('_', ' ').title() if len(attributes) == 1 else "Value",
                height=visual_props.get('height', 400)
            )

        else:
            # Generic handling for other data types with period awareness
            if len(attributes) > 1:
                df_subset = df[attributes + [x_col]].copy()

                # Apply metric titles
                df_display = df_subset.copy()
                for attr in attributes:
                    if attr in metric_titles:
                        clean_name = metric_titles[attr]
                    else:
                        clean_name = attr.replace('_', ' ').title()
                    df_display = df_display.rename(columns={attr: clean_name})

                display_attributes = [metric_titles.get(attr, attr.replace('_', ' ').title())
                                      for attr in attributes]

                df_melted = df_display.melt(id_vars=[x_col],
                                            value_vars=display_attributes,
                                            var_name='Series',
                                            value_name='Value')

                df_melted = apply_value_modifiers(df_melted, attributes, component, metric_titles)

                if graph_type in ['area', 'stacked_area']:
                    should_stack = (graph_type == 'stacked_area')
                    fig = go.Figure()
                    colors = px.colors.qualitative.Set3

                    for i, display_attr in enumerate(display_attributes):
                        series_data = df_melted[df_melted['Series'] == display_attr]

                        if should_stack:
                            fill_mode = 'tonexty' if i > 0 else 'tozeroy'
                            opacity = 0.7
                            hovermode = 'closest'
                        else:
                            fill_mode = 'tozeroy'
                            opacity = 0.4
                            hovermode = 'x unified'

                        base_color = colors[i % len(colors)]
                        if base_color.startswith('#'):
                            r = int(base_color[1:3], 16)
                            g = int(base_color[3:5], 16)
                            b = int(base_color[5:7], 16)
                            fillcolor = f'rgba({r},{g},{b},{opacity})'
                        else:
                            fillcolor = base_color

                        fig.add_trace(go.Scatter(
                            x=series_data[x_col],
                            y=series_data['Value'],
                            mode='lines',
                            fill=fill_mode,
                            name=display_attr,
                            line=dict(color=base_color, width=2),
                            fillcolor=fillcolor,
                            hovertemplate='<b>%{fullData.name}</b><br>%{x}<br>Value: %{y:$,.0f}<br><extra></extra>'
                        ))

                    fig.update_layout(
                        title=component['component_title'],
                        xaxis_title="Period",
                        yaxis_title="Value",
                        height=visual_props.get('height', 400),
                        hovermode=hovermode
                    )
                elif graph_type == 'line':
                    fig = px.line(df_melted, x=x_col, y='Value', color='Series',
                                  title=component['component_title'])
                elif graph_type == 'bar':
                    fig = px.bar(df_melted, x=x_col, y='Value', color='Series',
                                 title=component['component_title'])
                else:
                    fig = px.line(df_melted, x=x_col, y='Value', color='Series',
                                  title=component['component_title'])
            else:
                # Single attribute
                if graph_type == 'line':
                    fig = px.line(df, x=x_col, y=primary_attr, title=component['component_title'])
                elif graph_type == 'bar':
                    fig = px.bar(df, x=x_col, y=primary_attr, title=component['component_title'])
                elif graph_type in ['area', 'stacked_area']:
                    fig = px.area(df, x=x_col, y=primary_attr, title=component['component_title'])
                else:
                    fig = px.line(df, x=x_col, y=primary_attr, title=component['component_title'])

        # Apply custom layout AFTER removing metricTitles
        if custom_layout:
            valid_layout_props = {k: v for k, v in custom_layout.items()
                                  if k not in ['metricTitles']}
            if valid_layout_props:
                fig.update_layout(**valid_layout_props)

        # Check if fig was created successfully
        if fig is None:
            print(f"ERROR: fig was not created for component: {component['component_title']}")
            return dbc.Alert(f"Chart creation failed: {component['component_title']}", color="danger")

        height = visual_props.get('height', 400)

        # Return with proper card wrapper and period info
        return dbc.Card([
            dbc.CardHeader([
                html.H5([
                    html.I(className="fas fa-chart-line me-2 text-primary"),
                    component['component_title']
                ], className="mb-0"),
                html.Small(f"Analysis period: {analysis_period} years", className="text-muted")
            ], className="bg-light border-0"),
            dbc.CardBody([
                dcc.Graph(
                    figure=fig,
                    style={'height': f'{height}px'},
                    config={
                        'displayModeBar': True,
                        'displaylogo': False,
                        'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                        'toImageButtonOptions': {
                            'format': 'png',
                            'filename': f'{component["component_title"]}_chart',
                            'height': height,
                            'width': 900,
                            'scale': 2
                        }
                    }
                )
            ], className="p-3")
        ], className="shadow-sm border-0")

    except Exception as e:
        print(f"Graph creation error details: {str(e)}")
        traceback.print_exc()
        return dbc.Alert(f"Graph creation error: {str(e)}", color="danger")


# ==============================================================================
# LAYOUT CREATION FUNCTIONS - UPDATED
# ==============================================================================

def create_data_driven_tabs(structure_df):
    """Create tabs from CSV structure"""
    tab_info = structure_df.groupby('tab_name').agg({
        'tab_order': 'min'
    }).reset_index().sort_values('tab_order', na_position='last')

    tabs = []
    tab_icons = {
        'executive summary': '📊',
        'financial': '💰',
        'energy': '⚡',
        'battery': '🔋',
        'cashflow': '💸',
        'market analysis': '📈',
        'system performance': '📊',
        'technical details': '🔧',
        'compliance': '✅'
    }

    for _, row in tab_info.iterrows():
        tab_name = row['tab_name']
        tab_key = tab_name.lower().replace(' ', '_')

        icon_emoji = '📊'
        for key, icon in tab_icons.items():
            if key in tab_name.lower():
                icon_emoji = icon
                break

        tab_label = f"{icon_emoji} {tab_name}"

        tab = dbc.Tab(
            label=tab_label,
            tab_id=tab_key,
            active_tab_style={"background-color": "#667eea", "color": "white"}
        )
        tabs.append(tab)

    return dbc.Tabs(
        tabs,
        id="main-tabs",
        active_tab=tabs[0].tab_id if tabs else "executive_summary",
        className="nav-tabs-custom mb-4"
    )


def create_enhanced_app_layout():
    """Create the complete enhanced layout with period awareness"""
    return html.Div([
        # Enhanced Header
        html.Div([
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.H1("EnTrans Results Dashboard", className="text-white mb-2"),
                        html.P("Enhanced FCAS Revenue Analysis & Energy Arbitrage Platform",
                               className="text-white mb-0 opacity-75")
                    ], width=8),
                    dbc.Col([
                        html.Div([
                            html.A([
                                html.I(className="fas fa-arrow-left me-2"),
                                "Back to Form"
                            ], href="/form-with-saved-values", className="btn btn-light me-2"),
                            dbc.Button([
                                html.I(className="fas fa-download me-2"),
                                "Export PDF"
                            ], color="light", size="sm", className="me-2"),
                            dbc.Button([
                                html.I(className="fas fa-share me-2"),
                                "Share"
                            ], color="light", outline=True, size="sm")
                        ], className="d-flex justify-content-end")
                    ], width=4)
                ])
            ])
        ], className="dashboard-header"),

        # Main Content
        dbc.Container([
            html.Div(id='dynamic-tabs'),
            html.Div(id='tab-content', className="mt-4")
        ], fluid=True),

        # Footer with period info
        html.Footer([
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.P("EnTrans Energy Solutions", className="mb-1 fw-bold"),
                        html.P("Enhanced FCAS Revenue Analysis Platform",
                               className="text-muted small mb-0")
                    ], width=6),
                    dbc.Col([
                        html.P(f"Generated: {datetime.now().strftime('%Y-%m-%d at %H:%M')}",
                               className="mb-1 text-end"),
                        html.P(id="analysis-period-info",
                               className="text-muted small mb-0 text-end")
                    ], width=6)
                ])
            ])
        ], className="mt-5 pt-4 border-top text-muted")
    ])


def load_data_with_metadata():
    """Enhanced data loading for dashboard with period info"""
    results_path, structure_path, project_config_path, data_dir = get_paths()

    structure_df = pd.DataFrame()
    data_sources = {}

    if structure_path.exists():
        structure_df = pd.read_csv(structure_path)
        data_sources = load_data_sources_from_csv(structure_df)

    return structure_df, data_sources


def setup_dashboard_callbacks(dash_app):
    """Enhanced dashboard callbacks with period awareness"""

    @dash_app.callback(
        Output('dynamic-tabs', 'children'),
        [Input('dynamic-tabs', 'id')]
    )
    def create_tabs(_):
        """Create navigation tabs dynamically"""
        try:
            structure_df, data_sources = load_data_with_metadata()

            if structure_df is None or (hasattr(structure_df, 'empty') and structure_df.empty):
                return html.Div("No dashboard structure available", className="mb-4")

            return create_data_driven_tabs(structure_df)

        except Exception as e:
            logger.error(f"ERROR in create_tabs callback: {e}")
            return html.Div("Error loading tabs", className="mb-4")

    @dash_app.callback(
        Output('tab-content', 'children'),
        [Input('main-tabs', 'active_tab')]
    )
    def render_tab_content(active_tab):
        """Render tab content with period awareness"""
        try:
            if not active_tab:
                return dbc.Alert("Loading...", color="info")

            structure_df, data_sources = load_data_with_metadata()

            if structure_df is None or (hasattr(structure_df, 'empty') and structure_df.empty):
                return dbc.Alert("No data available", color="warning")

            # Get analysis period for display
            analysis_period = get_analysis_period_from_metadata(data_sources)

            # Tab name mapping
            tab_name_mapping = {
                'executive_summary': 'Executive Summary',
                'financial_results': 'Financial Results',
                'energy_results': 'Energy Results',
                'battery_analysis': 'Battery Analysis',
                'grid_interaction': 'Grid Interaction',
                'cashflow': 'CashFlow',
                'market_analysis': 'Market Analysis',
                'system_performance': 'System Performance',
                'technical_details': 'Technical Details',
                'compliance': 'Compliance'
            }

            tab_name = tab_name_mapping.get(active_tab, active_tab.replace('_', ' ').title())
            content = create_data_driven_tab_content(tab_name, structure_df, data_sources)

            # Add period info banner if not default 25 years
            if analysis_period != 25:
                period_banner = dbc.Alert([
                    html.I(className="fas fa-info-circle me-2"),
                    f"This analysis uses a {analysis_period}-year period instead of the standard 25 years."
                ], color="info", className="mb-3")

                return html.Div([period_banner, content])

            return content

        except Exception as e:
            logger.error(f"ERROR in render_tab_content callback: {e}")
            return dbc.Alert(f"Error loading tab content: {str(e)}", color="danger")

    # Add callback for footer analysis period info
    @dash_app.callback(
        Output('analysis-period-info', 'children'),
        [Input('analysis-period-info', 'id')]
    )
    def update_analysis_period_info(_):
        """Update analysis period info in footer"""
        try:
            structure_df, data_sources = load_data_with_metadata()
            analysis_period = get_analysis_period_from_metadata(data_sources)
            return f"Analysis Period: {analysis_period} years"
        except:
            return "Analysis Period: 25 years (default)"

    logger.info("✅ Enhanced dashboard callbacks with period awareness setup complete")


def create_section_content_from_csv(components, data_sources):
    """Create section content with proper component type handling and period awareness"""
    content_rows = []
    current_row_components = []
    current_row_width = 0

    width_mapping = {'quarter': 3, 'third': 4, 'half': 6, 'full': 12}

    for _, component in components.iterrows():
        comp_width = width_mapping.get(component.get('width', 'full'), 12)
        component_type = str(component['component_type']).strip().lower()

        try:
            if component_type == 'metric_card':
                comp_element = create_metric_card_from_csv(component, data_sources)
                current_row_components.append(dbc.Col(comp_element, width=comp_width, className="mb-4"))

            elif component_type == 'panel':
                comp_element = create_panel_component_from_csv(component, data_sources)
                current_row_components.append(comp_element)

            elif component_type == 'graph':
                comp_element = create_graph_from_csv(component, data_sources)
                current_row_components.append(dbc.Col(comp_element, width=comp_width, className="mb-4"))

            elif component_type == 'table':
                graph_type = str(component.get('graph_type', '')).strip().lower()
                data_source = str(component.get('data_source', '')).strip()
                component_title = str(component.get('component_title', '')).strip().lower()

                use_landscape = False

                if 'landscape' in graph_type:
                    use_landscape = True
                elif (data_source == 'cashflow_timeseries.parquet' and
                      ('cashflow' in component_title or 'payback' in component_title)):
                    use_landscape = True
                elif ('|' in data_source and
                      'cashflow_timeseries.parquet' in data_source and
                      len(data_source.split('|')) > 3):
                    use_landscape = False
                else:
                    use_landscape = False

                if use_landscape:
                    comp_element = create_generic_landscape_table(component, data_sources)
                else:
                    comp_element = create_pipe_separated_table(component, data_sources)

                current_row_components.append(dbc.Col(comp_element, width=comp_width, className="mb-4"))

            else:
                print(f"WARNING: Unknown component type '{component_type}'")
                continue

        except Exception as e:
            print(f"ERROR: Failed to create component {component.get('component_id', 'unknown')}: {str(e)}")
            traceback.print_exc()
            error_component = dbc.Alert(
                f"Error loading component: {component.get('component_title', 'Unknown')}",
                color="danger"
            )
            current_row_components.append(dbc.Col(error_component, width=comp_width, className="mb-4"))

        current_row_width += comp_width

        if current_row_width >= 12:
            content_rows.append(dbc.Row(current_row_components, className="g-3"))
            current_row_components = []
            current_row_width = 0

    if current_row_components:
        content_rows.append(dbc.Row(current_row_components, className="g-3"))

    return html.Div(content_rows)


def create_data_driven_tab_content(tab_name, structure_df, data_sources):
    """Create tab content with enhanced period support"""
    tab_components = structure_df[structure_df['tab_name'] == tab_name].sort_values('order', na_position='last')

    if tab_components.empty:
        return dbc.Alert(f"No components defined for {tab_name}", color="info")

    sections = tab_components['section'].unique()
    panel_sections = []
    regular_sections = []

    for section in sections:
        if pd.isna(section):
            continue

        section_components = structure_df[
            (structure_df['tab_name'] == tab_name) &
            (structure_df['section'] == section)
            ].sort_values('order', na_position='last')

        has_panels = any(comp['component_type'] == 'panel' for _, comp in section_components.iterrows())

        if has_panels:
            panel_sections.append(section)
        else:
            regular_sections.append(section)

    if panel_sections:
        return create_wrapping_layout(tab_name, structure_df, data_sources, panel_sections, regular_sections)
    else:
        return create_standard_tab_layout(tab_name, structure_df, data_sources, sections)


def create_wrapping_layout(tab_name, structure_df, data_sources, panel_sections, regular_sections):
    """Create wrapping layout for tabs with panels"""
    panels = []
    for section in panel_sections:
        section_components = structure_df[
            (structure_df['tab_name'] == tab_name) &
            (structure_df['section'] == section)
            ].sort_values('order', na_position='last')

        for _, component in section_components.iterrows():
            if component['component_type'] == 'panel':
                panel = create_panel_component_from_csv(component, data_sources)
                panels.append(panel)

    other_content = []
    for section in regular_sections:
        section_components = structure_df[
            (structure_df['tab_name'] == tab_name) &
            (structure_df['section'] == section)
            ].sort_values('order', na_position='last')

        section_div = html.Div([
            html.H5([
                html.I(className="fas fa-layer-group me-2 text-primary"),
                section
            ], className="mb-3"),
            create_section_content_from_csv(section_components, data_sources)
        ], className="mb-4")

        other_content.append(section_div)

    return html.Div([
        html.Div(panels, style={
            'display': 'flex',
            'flex-direction': 'column',
            'width': '400px',
            'flex': '0 0 400px',
            'margin-right': '20px'
        }),
        html.Div(other_content, style={
            'flex': '1',
            'min-width': '0'
        })
    ], style={
        'display': 'flex',
        'width': '100%',
        'align-items': 'flex-start'
    })


def create_standard_tab_layout(tab_name, structure_df, data_sources, sections):
    """Create standard tab layout"""
    content_sections = []

    for section in sections:
        if pd.isna(section):
            continue

        section_components = structure_df[
            (structure_df['tab_name'] == tab_name) &
            (structure_df['section'] == section)
            ].sort_values('order', na_position='last')

        content_sections.append(
            html.H4([
                html.I(className="fas fa-layer-group me-2 text-primary"),
                section
            ], className="section-title mb-4")
        )

        section_content = create_section_content_from_csv(section_components, data_sources)
        content_sections.append(section_content)

    return html.Div(content_sections)


# ==============================================================================
# EXPORT FUNCTIONS FOR FLASK INTEGRATION
# ==============================================================================

__all__ = [
    'load_data_with_metadata',
    'create_enhanced_app_layout',
    'setup_dashboard_callbacks',
    'ENHANCED_CSS',
    'LANDSCAPE_TABLE_CSS',
    'get_analysis_period_from_metadata',
    'get_period_labels',
    'validate_dataframe_length',
    'get_dynamic_column_config'
]