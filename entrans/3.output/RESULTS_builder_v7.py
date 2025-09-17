import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import pandas as pd
import pyarrow.parquet as pq
import plotly.express as px
import plotly.graph_objects as go
import ast
import json
import dash_bootstrap_components as dbc
from datetime import datetime
from entrans.config import Config





# Initialize the Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config.suppress_callback_exceptions = True
app.title = "EnTrans Results Dashboard v6 - Enhanced FCAS"

# Set paths
RESULTS_PATH = Config.RESULTS_DIR
STRUCTURE_PATH = Config.OUTPUT_DIR / "results_structure7.csv"  # Updated structure file
PROJECT_CONFIG_PATH = Config.INPUT_DIR / "json_updated" / "All_commercial_updated.json"

# ==============================================================================
# OPTIMIZED CSS STYLING - Consolidated and streamlined
# ==============================================================================

# Enhanced CSS for better table presentation
# Enhanced CSS for better table presentation
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


def apply_value_modifiers(df_melted, attributes, component, metric_titles):
    """Apply value modifiers to make specified attributes negative - FINAL FIX FOR NaN"""
    try:
        print(f"DEBUG VALUE MODIFIERS: Starting apply_value_modifiers")
        print(f"DEBUG VALUE MODIFIERS: Component title: {component.get('component_title', 'Unknown')}")
        print(f"DEBUG VALUE MODIFIERS: Attributes: {attributes}")

        value_modifiers = component.get('value_modifiers', '')
        print(f"DEBUG VALUE MODIFIERS: Raw value_modifiers from component: '{value_modifiers}'")

        # FIXED: Handle NaN values properly
        if pd.isna(value_modifiers) or value_modifiers == '' or value_modifiers is None:
            print(f"DEBUG VALUE MODIFIERS: No valid value_modifiers found, returning unchanged")
            return df_melted

        # Convert to string if it's not already
        value_modifiers = str(value_modifiers).strip()
        if not value_modifiers or value_modifiers.lower() == 'nan':
            print(f"DEBUG VALUE MODIFIERS: Empty or NaN value_modifiers, returning unchanged")
            return df_melted

        modifiers = value_modifiers.split('|') if '|' in value_modifiers else [value_modifiers]
        print(f"DEBUG VALUE MODIFIERS: Split modifiers: {modifiers}")

        # Ensure we have enough modifiers for all attributes
        while len(modifiers) < len(attributes):
            modifiers.append('positive')  # Default to positive

        print(f"DEBUG VALUE MODIFIERS: Final modifiers (padded): {modifiers}")

        # Apply modifiers
        for i, attr in enumerate(attributes):
            print(f"DEBUG VALUE MODIFIERS: Processing attribute {i}: {attr}")

            # FIXED: Safer attribute handling - ensure attr is a string
            if pd.isna(attr) or attr is None:
                print(f"DEBUG VALUE MODIFIERS: Skipping invalid attribute: {attr}")
                continue

            attr_str = str(attr).strip()
            if not attr_str:
                print(f"DEBUG VALUE MODIFIERS: Skipping empty attribute")
                continue

            # Get the display name (same logic as in your graph creation)
            if attr_str in metric_titles:
                display_name = metric_titles[attr_str]
                print(f"DEBUG VALUE MODIFIERS: Using metric title: {display_name}")
            else:
                # FIXED: Safer string operations
                display_name = attr_str
                if 'monthly_' in display_name:
                    display_name = display_name.replace('monthly_', '').replace('_', ' ').title()
                elif 'hourly_' in display_name:
                    display_name = display_name.replace('hourly_', '').replace('_', ' ').title()
                elif 'annual_' in display_name:
                    display_name = display_name.replace('annual_', '').replace('_', ' ').title()
                else:
                    display_name = display_name.replace('_', ' ').title()
                print(f"DEBUG VALUE MODIFIERS: Generated display name: {display_name}")

            # Check if this attribute should be negative
            modifier = modifiers[i] if i < len(modifiers) else 'positive'
            print(f"DEBUG VALUE MODIFIERS: Modifier for {display_name}: '{modifier}'")

            # FIXED: Safer modifier checking
            if isinstance(modifier, str) and modifier.strip().lower() in ['negative', 'neg', '-']:
                print(f"DEBUG VALUE MODIFIERS: Applying NEGATIVE modifier to {display_name}")

                # Check if the series exists in the dataframe
                mask = df_melted['Series'] == display_name
                matching_rows = mask.sum()
                print(f"DEBUG VALUE MODIFIERS: Found {matching_rows} rows matching '{display_name}'")

                if matching_rows > 0:
                    # Apply negative transformation
                    df_melted.loc[mask, 'Value'] = -abs(df_melted.loc[mask, 'Value'])
                    print(f"DEBUG VALUE MODIFIERS: ✅ Successfully applied negative modifier to {display_name}")
                else:
                    print(f"DEBUG VALUE MODIFIERS: ❌ No rows found for series '{display_name}'")
            else:
                print(f"DEBUG VALUE MODIFIERS: Keeping {display_name} as positive")

        print(f"DEBUG VALUE MODIFIERS: Finished apply_value_modifiers successfully")
        return df_melted

    except Exception as e:
        print(f"ERROR in apply_value_modifiers: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return unchanged dataframe on error
        return df_melted


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


def safe_get_calculation(component):
    """Safely get calculation value handling NaN"""
    calc_value = component.get('calculation', '')
    if pd.isna(calc_value):
        return ''
    return str(calc_value).strip()


def create_pipe_separated_table(component, data_sources):
    """Fixed pipe-separated table with proper format handling"""
    print(f"DEBUG: Processing table: {component.get('component_title', 'Unknown')}")
    print(f"DEBUG: Data source: {component.get('data_source', 'Unknown')}")
    print(f"DEBUG: Attributes: {component.get('attribute', 'Unknown')}")

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

    # FIXED: Handle pipe-separated display formats properly
    display_formats = []
    if component.get('display_format') and '|' in component.get('display_format', ''):
        display_formats = component['display_format'].split('|')
        print(f"DEBUG: Split display formats: {display_formats}")
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

    # Get custom layout
    custom_layout_raw = component.get('custom_layout', '{}')
    print(f"DEBUG: Raw custom_layout value: {repr(custom_layout_raw)}")

    custom_layout = safe_literal_eval(custom_layout_raw)
    print(f"DEBUG: Parsed custom_layout: {custom_layout}")

    metric_titles = custom_layout.get('metricTitles', {})
    print(f"DEBUG: metric_titles: {metric_titles}")

    # Get column configuration
    column_config = custom_layout.get('columnConfig', {
        'showYear1': True,
        'show25Year': False,
        'year1Header': 'Year 1',
        'total25Header': '25-Year Total'
    })

    show_25_year = column_config.get('show25Year', False)
    if isinstance(show_25_year, str):
        show_25_year = show_25_year.lower() == 'true'

    print(f"DEBUG: column_config: {column_config}")
    print(f"DEBUG: show25Year value: {show_25_year} (type: {type(show_25_year)})")

    # Create table data structure
    table_data = []

    for i, attr in enumerate(attributes):
        source_file = source_files[i] if i < len(source_files) else source_files[0]
        calculation = calculations[i] if i < len(calculations) else ''
        display_format = display_formats[i] if i < len(display_formats) else ''

        print(f"DEBUG: Processing attribute {i}: {attr}")
        print(f"DEBUG: Source: {source_file}, Calculation: {calculation}, Format: '{display_format}'")

        # Check if source file exists
        if source_file not in data_sources:
            print(f"Warning: Data source '{source_file}' not found for attribute '{attr}'")
            continue

        # Get Year 1 value
        year1_value = get_value_from_data_sources(data_sources, source_file, attr, calculation)
        print(f"DEBUG: Year 1 value for {attr}: {year1_value}")

        # Get 25-year total if requested
        total25_value = "N/A"
        if show_25_year:
            print(f"DEBUG: Calculating 25-year value for {attr}")
            if source_file.endswith('.parquet'):
                total25_value = get_value_from_data_sources(data_sources, source_file, attr, 'sum')
                print(f"DEBUG: 25-year sum value: {total25_value}")
            elif year1_value != "N/A" and isinstance(year1_value, (int, float)):
                total25_value = year1_value * 25
                print(f"DEBUG: 25-year calculated value (x25): {total25_value}")

        if year1_value != "N/A":
            # Get display name from custom titles
            display_name = metric_titles.get(attr, attr.replace('_', ' ').replace('.', ' ').title())
            print(f"DEBUG: Display name for {attr}: {display_name}")

            # FIXED: Format values with corresponding display format
            formatted_year1 = format_metric_value(year1_value, display_format)
            formatted_total25 = format_metric_value(total25_value, display_format) if total25_value != "N/A" else "N/A"

            print(f"DEBUG: Formatted Year 1: {formatted_year1}")
            if show_25_year:
                print(f"DEBUG: Formatted 25-year: {formatted_total25}")

            row_data = {
                'Metric': display_name,
                column_config.get('year1Header', 'Year 1'): formatted_year1
            }

            # Add 25-year column if requested
            if show_25_year:
                row_data[column_config.get('total25Header', '25-Year Total')] = formatted_total25
                print(f"DEBUG: Added 25-year column with value: {formatted_total25}")

            table_data.append(row_data)

    print(f"DEBUG: Final table_data: {table_data}")

    if not table_data:
        return dbc.Alert("No data available for the specified attributes", color="info")

    # Create columns dynamically based on configuration
    columns = [{"name": "Metric", "id": "Metric"}]
    columns.append({
        "name": column_config.get('year1Header', 'Year 1'),
        "id": column_config.get('year1Header', 'Year 1')
    })

    if show_25_year:
        columns.append({
            "name": column_config.get('total25Header', '25-Year Total'),
            "id": column_config.get('total25Header', '25-Year Total')
        })
        print(f"DEBUG: Added 25-year column to columns list")

    print(f"DEBUG: Final columns: {columns}")

    # Create the table
    return dbc.Card([
        dbc.CardHeader([
            html.H5(component['component_title'], className="mb-0 text-primary"),
            html.Small(component.get('tooltip', 'Key performance metrics'), className="text-muted")
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
                                             column_config.get('total25Header', '25-Year Total')]},
                        'textAlign': 'right',
                        'fontWeight': 'bold'
                    }
                ],
                style_cell_conditional=[
                                           {
                                               'if': {'column_id': 'Metric'},
                                               'minWidth': '200px',
                                               'width': '50%' if show_25_year else '60%'
                                           },
                                           {
                                               'if': {'column_id': column_config.get('year1Header', 'Year 1')},
                                               'minWidth': '120px',
                                               'width': '25%' if show_25_year else '40%'
                                           }
                                       ] + ([{
                    'if': {'column_id': column_config.get('total25Header', '25-Year Total')},
                    'minWidth': '150px',
                    'width': '25%'
                }] if show_25_year else [])
            )
        ])
    ], className="performance-table")


# Updated format_metric_value function to handle more formats
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
            # Try direct string formatting
            try:
                return format_str.format(num_value)
            except:
                return str(value)

    except (ValueError, TypeError):
        return str(value)


def get_metric_icon_and_color(component_title, attribute):
    """Optimized icon and color mapping"""
    text_to_check = f"{component_title} {attribute}".lower()

    icon_mappings = {
        # FCAS and market specific
        'fcas': {'icon': 'fas fa-cogs', 'color': 'success'},
        'ancillary': {'icon': 'fas fa-cogs', 'color': 'success'},
        'arbitrage': {'icon': 'fas fa-chart-line', 'color': 'info'},
        'market': {'icon': 'fas fa-chart-bar', 'color': 'primary'},

        # Financial metrics
        'cost': {'icon': 'fas fa-dollar-sign', 'color': 'warning'},
        'npv': {'icon': 'fas fa-coins', 'color': 'primary'},
        'revenue': {'icon': 'fas fa-money-bill-wave', 'color': 'success'},
        'payback': {'icon': 'fas fa-calendar-alt', 'color': 'info'},
        'lcoe': {'icon': 'fas fa-calculator', 'color': 'secondary'},

        # Energy/Technical metrics
        'energy': {'icon': 'fas fa-bolt', 'color': 'warning'},
        'battery': {'icon': 'fas fa-battery-three-quarters', 'color': 'success'},
        'grid': {'icon': 'fas fa-plug', 'color': 'primary'},
        'export': {'icon': 'fas fa-arrow-up', 'color': 'info'},

        # Certificates
        'rec': {'icon': 'fas fa-certificate', 'color': 'success'},
    }

    for keyword, styling in icon_mappings.items():
        if keyword in text_to_check:
            return styling

    return {'icon': 'fas fa-chart-simple', 'color': 'primary'}


def safe_literal_eval(s):
    """Safely evaluate string literals with enhanced error handling"""
    if pd.isna(s) or s == '' or s is None:
        return {}
    try:
        # Clean the string first
        clean_s = str(s).strip()

        # Try ast.literal_eval first (handles proper Python dict syntax)
        return ast.literal_eval(clean_s)
    except (ValueError, SyntaxError, TypeError) as e:
        # If that fails, try to fix common JSON issues and use json.loads
        try:
            # Replace single quotes with double quotes for JSON
            json_s = clean_s.replace("'", '"')
            return json.loads(json_s)
        except:
            print(f"WARNING: Could not parse custom_layout: {s}")
            print(f"Error: {e}")
            return {}

# ==============================================================================
# ENHANCED DATA LOADING WITH FCAS SUPPORT
# ==============================================================================

def load_data_sources_from_csv(structure_df):
    """Enhanced data loading with FCAS metadata integration - FIXED for pipe-separated sources"""
    data_sources = {}

    # Get all unique data sources and handle pipe-separated values
    all_sources = set()
    for source_string in structure_df['data_source'].dropna().unique():
        if '|' in source_string:
            # Split pipe-separated sources and add each individual source
            individual_sources = source_string.split('|')
            for source in individual_sources:
                source = source.strip()
                if source:  # Only add non-empty sources
                    all_sources.add(source)
        else:
            # Single source
            source = source_string.strip()
            if source:
                all_sources.add(source)

    referenced_sources = list(all_sources)
    print(f"📂 Loading data sources: {referenced_sources}")

    # Load scalar results with FCAS support
    if 'scalar_results.csv' in referenced_sources:
        scalar_path = RESULTS_PATH / 'scalar_results.csv'
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
                print(f"✅ Loaded scalar_results.csv with {len(scalar_dict)} metrics")

            except Exception as e:
                print(f"❌ Error loading scalar_results.csv: {e}")
                data_sources['scalar_results.csv'] = {}

    # Load project configuration
    if any(source in ['project_info', 'main_config'] for source in referenced_sources):
        if PROJECT_CONFIG_PATH.exists():
            try:
                with open(PROJECT_CONFIG_PATH, 'r') as f:
                    config = json.load(f)

                if 'project_info' in referenced_sources:
                    data_sources['project_info'] = config.get('project_info', {})

                if 'main_config' in referenced_sources:
                    main_config_data = {}

                    # Merge all config sections
                    for section in ['financial_inputs', 'system_inputs', 'battery_inputs']:
                        if section in config:
                            main_config_data.update(config[section])

                    # Add top-level config
                    for key, value in config.items():
                        if not isinstance(value, dict):
                            main_config_data[key] = value

                    data_sources['main_config'] = main_config_data

                print(f"✅ Loaded configuration data")

            except Exception as e:
                print(f"❌ Error loading configuration: {e}")

    # Load parquet files
    parquet_sources = [s for s in referenced_sources if s.endswith('.parquet')]
    for source in parquet_sources:
        file_path = RESULTS_PATH / source
        if file_path.exists():
            try:
                df = pq.read_table(file_path).to_pandas()

                # Special handling for cashflow data
                if 'cashflow' in source and len(df) > 25:
                    df = df.iloc[:26]  # Limit to 25 years
                    df = df.dropna(axis=1, how='all').ffill()

                data_sources[source] = df
                print(f"✅ Loaded {source} ({len(df)} rows, {len(df.columns)} cols)")

            except Exception as e:
                print(f"❌ Error loading {source}: {e}")

    # Load FCAS historical data
    fcas_historical_path = Config.DATA_DIR / 'fcas' / 'fcas_historical.csv'
    if 'fcas_historical.csv' in referenced_sources and fcas_historical_path.exists():
        try:
            df = pd.read_csv(fcas_historical_path)

            # Clean and prepare the data
            df = df.dropna(how='all')  # Remove empty rows
            df = df.reset_index(drop=True)

            # Ensure we have quarter information - if first column is quarters, rename it
            if df.columns[0].lower() in ['quarter', 'quarters', 'q', 'period']:
                df = df.rename(columns={df.columns[0]: 'Quarter'})
            elif 'Quarter' not in df.columns:
                # If no quarter column, create one based on row index
                quarters = []
                for i in range(len(df)):
                    year_offset = i // 4
                    quarter_num = (i % 4) + 1
                    base_year = 2016  # Adjust this to your starting year
                    quarters.append(f"Q{quarter_num} {base_year + year_offset}")
                df.insert(0, 'Quarter', quarters)

            data_sources['fcas_historical.csv'] = df
            print(f"✅ Loaded fcas_historical.csv ({len(df)} rows, {len(df.columns)} cols)")

        except Exception as e:
            print(f"❌ Error loading fcas_historical.csv: {e}")




    # Load metadata for FCAS details
    metadata_path = RESULTS_PATH / 'metadata.json'
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            data_sources['metadata'] = metadata
            print(f"✅ Loaded metadata.json")
        except Exception as e:
            print(f"❌ Error loading metadata: {e}")

    # Create calculated values with enhanced FCAS support
    if 'calculated' in referenced_sources:
        data_sources['calculated'] = create_enhanced_calculated_values(data_sources)

    return data_sources


def create_enhanced_calculated_values(data_sources):
    """Enhanced calculated values with FCAS and arbitrage metrics"""
    calculated = {}

    project_info = data_sources.get('project_info', {})
    main_config = data_sources.get('main_config', {})
    scalar_data = data_sources.get('scalar_results.csv', {})
    metadata = data_sources.get('metadata', {})

    # Enhanced FCAS calculations
    try:
        # Get total FCAS revenue
        total_fcas = scalar_data.get('Total ancillary services revenue', 0)
        calculated['total_fcas_revenue'] = float(total_fcas) if total_fcas else 0

        # Calculate annual FCAS revenue (total / 25 years)
        calculated['fcas_revenue_annual'] = calculated['total_fcas_revenue'] / 25

        # Get individual FCAS services
        for i in range(1, 5):
            service_key = f'Ancillary services {i} revenue'
            if service_key in scalar_data:
                calculated[f'fcas_service_{i}_revenue'] = float(scalar_data[service_key])
            else:
                calculated[f'fcas_service_{i}_revenue'] = 0

        # FCAS performance metrics from metadata
        ancillary_services = metadata.get('ancillary_services', {})
        if ancillary_services.get('enabled', False):
            revenue_streams = ancillary_services.get('revenue_streams', [])

            if revenue_streams:
                # Calculate participation rate (active services / total services)
                active_streams = sum(1 for stream in revenue_streams if stream.get('active', False))
                calculated['fcas_participation_rate'] = (active_streams / len(revenue_streams)) * 100

                # Calculate revenue per MW capacity
                battery_power = float(main_config.get('batt_power_discharge_max_kwac', 1)) / 1000  # Convert to MW
                if battery_power > 0:
                    calculated['fcas_revenue_per_mwh'] = calculated['fcas_revenue_annual'] / battery_power
                else:
                    calculated['fcas_revenue_per_mwh'] = 0

                # Estimate capacity utilization (simplified)
                calculated['fcas_capacity_utilization'] = min(85.0, calculated['fcas_participation_rate'])
            else:
                calculated['fcas_participation_rate'] = 0
                calculated['fcas_revenue_per_mwh'] = 0
                calculated['fcas_capacity_utilization'] = 0

        # Market integration metrics
        battery_capacity = float(main_config.get('batt_computed_bank_capacity', 1))
        if battery_capacity < 1000:  # < 1MWh requires aggregation
            calculated['aggregation_efficiency'] = 80.0  # 80% efficiency via VPP
            calculated['revenue_share_factor'] = 0.75  # 75% revenue share
            calculated['market_access_rating'] = "VPP Aggregated"
        else:
            calculated['aggregation_efficiency'] = 100.0  # Direct participation
            calculated['revenue_share_factor'] = 1.0  # 100% revenue share
            calculated['market_access_rating'] = "Direct Access"

        # Service revenue comparison data for charts
        service_revenues = []
        service_names = ['Fast Raise (6s)', 'Fast Lower (6s)', 'Slow Raise (60s)', 'Slow Lower (60s)']
        for i, name in enumerate(service_names, 1):
            revenue = calculated.get(f'fcas_service_{i}_revenue', 0)
            service_revenues.append({'Service': name, 'Revenue': revenue})

        calculated['service_revenue_comparison'] = service_revenues

        # REC calculations (existing)
        stc_value = project_info.get('stc_value', 0)
        annual_lrec_value = project_info.get('annual_lrec_value', 0)
        calculated['calculated_lrec_total'] = float(annual_lrec_value) * 20 if annual_lrec_value else 0
        calculated['total_rec_value'] = float(stc_value) + calculated['calculated_lrec_total'] if stc_value else \
        calculated['calculated_lrec_total']

        # Additional calculated metrics
        calculated['project_location'] = f"Lat: {main_config.get('lat', 'N/A')}, Lon: {main_config.get('lon', 'N/A')}"

        print(f"✅ Created {len(calculated)} enhanced calculated values including FCAS metrics")

    except Exception as e:
        print(f"❌ Error creating calculated values: {e}")
        # Set default values
        calculated.update({
            'total_fcas_revenue': 0,
            'fcas_revenue_annual': 0,
            'fcas_participation_rate': 0,
            'fcas_revenue_per_mwh': 0,
            'fcas_capacity_utilization': 0,
            'aggregation_efficiency': 0,
            'revenue_share_factor': 0,
            'market_access_rating': "Unknown"
        })

    return calculated


def get_value_from_data_sources(data_sources, data_source, attribute, calculation=None):
    """Enhanced data retrieval with calculation support"""

    # Handle NaN values safely for data_source
    if pd.isna(data_source):
        data_source = ''
    else:
        data_source = str(data_source)

    # Handle multiple data sources
    if '|' in data_source:
        source_files = data_source.split('|')
        attributes = attribute.split('|') if '|' in attribute else [attribute]

        # Ensure equal lengths
        while len(source_files) < len(attributes):
            source_files.append(source_files[-1])

        # Get value from corresponding data source
        for i, attr in enumerate(attributes):
            if i < len(source_files):
                source_file = source_files[i]
                if source_file in data_sources:
                    # Handle calculation properly
                    calc = None
                    if calculation and not pd.isna(calculation):
                        calc_list = str(calculation).split('|') if '|' in str(calculation) else [str(calculation)]
                        calc = calc_list[i] if i < len(calc_list) else ''

                    return get_single_source_value(data_sources[source_file], attr, calc)
        return "N/A"

    # Single data source (original logic)
    if data_source not in data_sources:
        return "N/A"

    # Handle calculation safely using helper function
    calc = None
    if calculation:
        calc_clean = safe_get_calculation({'calculation': calculation})
        if calc_clean:
            calc = calc_clean

    return get_single_source_value(data_sources[data_source], attribute, calc)


def get_single_source_value(source_data, attribute, calculation=None):
    """Get value from single data source with nested attribute support"""

    if calculation and calculation.strip():
        return perform_calculation(source_data, attribute, calculation)

    # Handle nested attributes (e.g., "costs_breakdown.solar_cost")
    if '.' in attribute:
        parts = attribute.split('.')
        value = source_data

        print(f"DEBUG: Looking for nested attribute '{attribute}' in {type(source_data)}")

        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
                print(f"DEBUG: Found '{part}', current value: {value}")
            else:
                print(
                    f"DEBUG: Could not find '{part}' in {list(value.keys()) if isinstance(value, dict) else type(value)}")
                return "N/A"

        if pd.isna(value) or value == '':
            return "Data not available"
        return value

    # Standard retrieval logic (existing code continues...)
    if isinstance(source_data, dict):
        if attribute in source_data:
            value = source_data[attribute]
            if pd.isna(value) or value == '':
                return "Data not available"
            return value
        else:
            return "N/A"

    elif isinstance(source_data, pd.DataFrame):
        if attribute in source_data.columns:
            if len(source_data) > 0:
                return source_data[attribute].iloc[0]
        return "N/A"

    return "N/A"


def perform_calculation(source_data, attribute, calculation):
    """Perform calculations on data based on CSV instructions"""

    try:
        calc_lower = calculation.lower().strip()

        # Handle index-based calculations (e.g., index_1 for Year 1)
        if calc_lower.startswith('index_'):
            try:
                index_num = int(calc_lower.replace('index_', ''))
                if isinstance(source_data, pd.DataFrame):
                    if attribute in source_data.columns and len(source_data) > index_num:
                        return source_data[attribute].iloc[index_num]
                    else:
                        return "Index out of range"
                else:
                    return "Not a DataFrame"
            except ValueError:
                return "Invalid index format"

        # Handle single attribute calculations (existing code continues...)
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
                return series.sum() if isinstance(source_data, pd.DataFrame) else value
            elif calc_lower == 'mean' or calc_lower == 'average':
                return series.mean() if isinstance(source_data, pd.DataFrame) else value
            elif calc_lower == 'max':
                return series.max() if isinstance(source_data, pd.DataFrame) else value
            elif calc_lower == 'min':
                return series.min() if isinstance(source_data, pd.DataFrame) else value
            elif calc_lower == 'count':
                return len(series) if isinstance(source_data, pd.DataFrame) else 1
            elif calc_lower.startswith('divide_by_'):
                divisor = float(calc_lower.replace('divide_by_', ''))
                base_value = series.sum() if isinstance(source_data, pd.DataFrame) else value
                return base_value / divisor
            elif calc_lower.startswith('multiply_by_'):
                multiplier = float(calc_lower.replace('multiply_by_', ''))
                base_value = series.sum() if isinstance(source_data, pd.DataFrame) else value
                return base_value * multiplier
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
                        # Default to sum for timeseries, first value for scalars
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
                return sum(values)
            elif calc_lower == 'subtract':  # First - Second - Third...
                result = values[0]
                for v in values[1:]:
                    result -= v
                return result
            elif calc_lower == 'multiply':
                result = values[0]
                for v in values[1:]:
                    result *= v
                return result
            elif calc_lower == 'divide':  # First / Second
                if len(values) >= 2 and values[1] != 0:
                    return values[0] / values[1]
                else:
                    return "Division error"
            elif calc_lower == 'percentage':  # (First / Second) * 100
                if len(values) >= 2 and values[1] != 0:
                    return (values[0] / values[1]) * 100
                else:
                    return 0
            elif calc_lower == 'average' or calc_lower == 'mean':
                return sum(values) / len(values) if values else 0
            elif calc_lower == 'max':
                return max(values) if values else 0
            elif calc_lower == 'min':
                return min(values) if values else 0
            else:
                return "Invalid calculation"

    except Exception as e:
        print(f"Calculation error: {e}")
        return "Calculation error"


# ==============================================================================
# OPTIMIZED COMPONENT CREATION FUNCTIONS
# ==============================================================================

def create_metric_card_from_csv(component, data_sources):
    """Updated metric card creation with calculation support"""
    attribute = component['attribute']
    data_source = component['data_source']
    calculation = safe_get_calculation(component)
    if pd.isna(calculation):
        calculation = ''
    else:
        calculation = str(calculation)

    value = get_value_from_data_sources(data_sources, data_source, attribute, calculation)
    formatted_value = format_metric_value(value, component.get('display_format', ''))

    styling = get_metric_icon_and_color(component['component_title'], attribute)

    # Special styling for FCAS/arbitrage cards
    card_class = "h-100 shadow-sm border-0 metric-card"
    if 'fcas' in component['component_title'].lower() or 'ancillary' in component['component_title'].lower():
        card_class += " fcas-highlight"
    elif 'arbitrage' in component['component_title'].lower() or 'export' in component['component_title'].lower():
        card_class += " arbitrage-highlight"

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
    ], className=card_class)


def create_panel_component_from_csv(component, data_sources):
    """Optimized panel component creation"""
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

    # Determine card styling
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


def format_landscape_currency(value):
    """Format currency for landscape tables - comma separated, no decimals"""
    if pd.isna(value) or value is None:
        return "$0"

    try:
        num_value = float(value)
        if num_value == 0:
            return "$0"
        elif num_value < 0:
            # Handle negative values
            return f"-${abs(num_value):,.0f}"
        else:
            # Positive values with comma separation, no decimals
            return f"${num_value:,.0f}"
    except (ValueError, TypeError):
        return str(value)


# UPDATED create_generic_landscape_table function with fixed formatting:
def create_generic_landscape_table(component, data_sources):
    """Enhanced landscape table with comma-separated currency formatting"""
    data_file = component.get('data_source', '')

    # Handle multiple data sources
    if '|' in data_file:
        source_files = data_file.split('|')
        for source_file in source_files:
            if source_file in data_sources:
                data_file = source_file
                break
        else:
            return dbc.Alert(f"No valid data sources found in: {data_file}", color="warning")

    if data_file not in data_sources:
        return dbc.Alert(f"Data file not found: {data_file}", color="warning")

    df = data_sources[data_file].copy()
    if not isinstance(df, pd.DataFrame):
        return dbc.Alert(f"Data source is not a DataFrame: {data_file}", color="warning")

    try:
        # Handle pipe-separated attributes
        if '|' in component['attribute']:
            attributes = component['attribute'].split('|')
        else:
            attributes = [component['attribute']]

        # Get custom layout for metric titles
        custom_layout = safe_literal_eval(component.get('custom_layout', '{}'))
        metric_titles = custom_layout.get('metricTitles', {})

        # Verify attributes exist in dataframe
        missing_attrs = [attr for attr in attributes if attr not in df.columns]
        if missing_attrs:
            return dbc.Alert(f"Missing columns in {data_file}: {', '.join(missing_attrs)}", color="warning")

        # Create landscape data structure
        landscape_data = []
        df_subset = df[attributes].copy()

        # Limit to 26 years and ensure consistent structure
        max_years = min(len(df_subset), 26)
        year_labels = [f"Year {i}" for i in range(max_years)]

        for attr in attributes:
            if attr in df_subset.columns:
                # Get display name from custom titles or generate from attribute name
                display_name = metric_titles.get(attr,
                                                 attr.replace('cf_', '').replace('_', ' ').title())

                row_data = {'Metric': display_name}

                # FIXED: Add year columns with COMMA-SEPARATED CURRENCY FORMATTING
                for i, year_label in enumerate(year_labels):
                    if i < len(df_subset):
                        value = df_subset[attr].iloc[i]
                        formatted_value = format_landscape_currency(value)
                        row_data[year_label] = formatted_value
                    else:
                        row_data[year_label] = "$0"

                landscape_data.append(row_data)

        if not landscape_data:
            return dbc.Alert("No data available for landscape table", color="info")

        # Create columns with consistent structure
        columns = [{"name": "Metric", "id": "Metric", "type": "text"}]
        for year_label in year_labels:
            columns.append({"name": year_label, "id": year_label, "type": "text"})

        # Ensure all rows have all required columns
        for row in landscape_data:
            for col in columns:
                if col['id'] not in row:
                    row[col['id']] = "$0"

        # Header color mapping based on component title and content
        def get_header_color_and_class(component_title, attributes):
            title_lower = component_title.lower()

            # Check for FCAS content
            has_fcas = any('ancillary' in attr or 'fcas' in attr for attr in attributes)

            if has_fcas or 'fcas' in title_lower:
                return "#198754", "bg-success"  # Green for FCAS
            elif 'cashflow' in title_lower or 'cash flow' in title_lower:
                return "#0d6efd", "bg-primary"  # Blue for cashflow
            elif 'payback' in title_lower:
                return "#fd7e14", "bg-warning"  # Orange for payback
            elif 'revenue' in title_lower:
                return "#20c997", "bg-info"  # Teal for revenue
            elif 'cost' in title_lower or 'expense' in title_lower:
                return "#dc3545", "bg-danger"  # Red for costs
            else:
                return "#6c757d", "bg-secondary"  # Gray default

        header_color, header_class = get_header_color_and_class(component['component_title'], attributes)

        return dbc.Card([
            dbc.CardHeader([
                html.H6(component['component_title'], className="mb-0 text-white"),
                html.Small(component.get('tooltip', 'Annual cashflow data - scroll to view all years'),
                           className="text-white opacity-75")
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
                            'maxHeight': '400px',
                            'overflowY': 'auto'
                        },
                        style_cell={
                            'padding': '8px',
                            'textAlign': 'center',
                            'whiteSpace': 'normal',
                            'height': 'auto',
                            'fontFamily': 'Segoe UI',
                            'fontSize': '12px',
                            'minWidth': '80px'
                        },
                        style_cell_conditional=[
                            {
                                'if': {'column_id': 'Metric'},
                                'textAlign': 'left',
                                'fontWeight': 'bold',
                                'backgroundColor': '#f8f9fa',
                                'minWidth': '180px',
                                'width': '180px',
                                'maxWidth': '180px'
                            }
                        ],
                        style_header={
                            'backgroundColor': header_color,
                            'color': 'white',
                            'fontWeight': 'bold',
                            'textAlign': 'center',
                            'fontSize': '11px',
                            'padding': '8px'
                        },
                        style_data_conditional=[
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': '#f8f9fa'
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
        import traceback
        traceback.print_exc()
        return dbc.Alert(f"Error creating landscape table: {str(e)}", color="danger")


# EXAMPLES of the new formatting:
FORMATTING_EXAMPLES = {
    "Input": "Output",
    "1234.56": "$1,235",  # Rounded to no decimals
    "1000000": "$1,000,000",  # No M suffix, full comma-separated
    "500000": "$500,000",  # No K suffix, full comma-separated
    "-25000": "-$25,000",  # Negative values handled
    "0": "$0",  # Zero values
    "123": "$123",  # Small values
    "999999": "$999,999",  # Large values without suffix
}

print("=== FIXED CURRENCY FORMATTING ===")
print("✅ Removed K/M suffixes")
print("✅ Added comma separation for thousands")
print("✅ No decimal places")
print("✅ Proper negative value handling")
print()
print("Examples:")
for input_val, output_val in FORMATTING_EXAMPLES.items():
    print(f"  {input_val} → {output_val}")


# Updated table selection logic for create_section_content_from_csv
UPDATED_TABLE_LOGIC = '''
elif component_type == 'table':
    # Check if this should be a landscape table
    graph_type = component.get('graph_type', '')
    data_source = component.get('data_source', '')

    # Handle NaN values properly
    if pd.isna(graph_type):
        graph_type = ''
    if pd.isna(data_source):
        data_source = ''

    # Use landscape table for:
    # 1. Explicitly marked as landscape
    # 2. Cashflow timeseries data (showing all years)
    if ('landscape' in graph_type.lower() or
    (data_source == 'cashflow_timeseries.parquet' and
     ('cashflow' in component.get('component_title', '').lower() or 
      'payback' in component.get('component_title', '').lower()))):
        comp_element = create_generic_landscape_table(component, data_sources)
    else:
        comp_element = create_pipe_separated_table(component, data_sources)'''

print("=== SOLUTION ===")
print("1. Replace create_enhanced_landscape_table with create_generic_landscape_table")
print("2. Update the table selection logic to use landscape for all cashflow timeseries")
print("3. Fix your JSON syntax error in the custom_layout")
print()
print("=== FIXED CUSTOM_LAYOUT FOR YOUR CASHFLOW TABLE ===")
fixed_layout = '{"metricTitles":{"cf_after_tax_cash_flow":"After Tax Cash Flow","cf_cumulative_payback_with_expenses":"Cumulative Payback With Expenses","cf_cumulative_payback_without_expenses":"Cumulative Payback Without Expenses","cf_cumulative_payback_with_fcas_lrec":"Cumulative Payback inc FCAS & LRECS"}}'
print(fixed_layout)


def create_graph_from_csv(component, data_sources):
    """Enhanced graph creation with waterfall, FCAS support, quarterly data, and value modifiers - FIXED ERROR HANDLING"""
    data_file = component.get('data_source', '')

    if data_file not in data_sources:
        return dbc.Alert(f"Data file not found: {data_file}", color="warning")

    source_data = data_sources[data_file]

    # Handle calculated data for service comparison charts
    if data_file == 'calculated' and component['attribute'] == 'service_revenue_comparison':
        service_data = source_data.get('service_revenue_comparison', [])
        if not service_data:
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

    # Standard graph handling for other data sources
    if not isinstance(source_data, pd.DataFrame):
        return dbc.Alert(f"Data source is not a DataFrame: {data_file}", color="warning")

    df = source_data.copy()
    attributes = component['attribute'].split('|') if '|' in component['attribute'] else [component['attribute']]

    # Verify attributes exist
    missing_attrs = [attr for attr in attributes if attr not in df.columns]
    if missing_attrs:
        return dbc.Alert(f"Missing columns: {', '.join(missing_attrs)}", color="warning")

    # FIXED: Get graph properties and handle metricTitles properly
    visual_props = safe_literal_eval(component.get('visual_properties', '{}'))
    custom_layout = safe_literal_eval(component.get('custom_layout', '{}'))

    # EXTRACT metric titles for processing but remove from layout updates
    metric_titles = custom_layout.pop('metricTitles', {}) if custom_layout else {}

    # Safe graph_type handling
    graph_type = component.get('graph_type', 'line')
    if pd.isna(graph_type):
        graph_type = 'line'
    else:
        graph_type = str(graph_type).strip().lower()

    # PIE CHART HANDLING - ENHANCED VERSION
    if graph_type == 'pie':
        print(f"DEBUG: Creating enhanced pie chart for {len(attributes)} attributes")

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

                    # FIXED: Use custom metric titles for pie chart labels
                    if attr in metric_titles:
                        label = metric_titles[attr]
                    elif custom_layout.get('labels') and i < len(custom_layout['labels']):
                        label = custom_layout['labels'][i]
                    else:
                        # Generate label from attribute name
                        label = attr.replace('cf_ancillary_services_', 'Service ').replace('_revenue', '').replace(
                            '_', ' ').title()

                    pie_labels.append(label)

        print(f"DEBUG: Pie chart data - Values: {pie_values}, Labels: {pie_labels}")

        if not pie_values:
            return dbc.Alert("No non-zero values found for pie chart", color="info")

        # ENHANCED: Modern color palette
        modern_colors = [
            '#667eea',  # Primary blue
            '#10b981',  # Green
            '#f59e0b',  # Amber
            '#ef4444',  # Red
            '#8b5cf6',  # Purple
            '#06b6d4',  # Cyan
            '#84cc16',  # Lime
            '#f97316',  # Orange
            '#ec4899',  # Pink
            '#6b7280'  # Gray
        ]

        # Create enhanced pie chart
        fig = go.Figure()

        # ENHANCED: Add pie trace with modern styling
        fig.add_trace(go.Pie(
            labels=pie_labels,
            values=pie_values,
            hole=custom_layout.get('hole', 0.4),  # Default to donut style
            hovertemplate='<b>%{label}</b><br>' +
                          'Value: <b>%{value:$,.0f}</b><br>' +
                          'Percentage: <b>%{percent}</b><br>' +
                          '<extra></extra>',
            textinfo='label+percent',
            textposition='auto',
            textfont=dict(
                size=12,
                color='white',
                family='Segoe UI'
            ),
            marker=dict(
                colors=modern_colors[:len(pie_values)],
                line=dict(
                    color='white',
                    width=2
                )
            ),
            pull=[0.05 if i == 0 else 0 for i in range(len(pie_values))],  # Highlight first slice
            rotation=90,  # Start from top
            sort=False  # Maintain order
        ))

        # ENHANCED: Modern layout styling
        fig.update_layout(
            title=dict(
                text=custom_layout.get('title', component['component_title']),
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top',
                font=dict(
                    size=18,
                    color='#2d3748',
                    family='Segoe UI',
                    weight='bold'
                )
            ),
            showlegend=custom_layout.get('showlegend', True),
            legend=dict(
                orientation='v',
                yanchor='middle',
                y=0.5,
                xanchor='left',
                x=1.02,
                font=dict(
                    size=11,
                    color='#374151',
                    family='Segoe UI'
                ),
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='rgba(0,0,0,0.1)',
                borderwidth=1
            ),
            height=visual_props.get('height', 500),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Segoe UI", size=12, color='#374151'),
            margin=dict(l=20, r=120, t=60, b=20),
            annotations=[
                dict(
                    text=f'<b>Total</b><br>${sum(pie_values):,.0f}',
                    x=0.5, y=0.5,
                    font_size=14,
                    font_color='#374151',
                    font_family='Segoe UI',
                    showarrow=False
                )
            ] if custom_layout.get('hole', 0.4) > 0 else []  # Only show center text for donuts
        )

        # ENHANCED: Apply additional custom styling
        if custom_layout:
            layout_updates = {k: v for k, v in custom_layout.items()
                              if k not in ['labels', 'hole', 'title', 'showlegend', 'metricTitles']}
            if layout_updates:
                fig.update_layout(**layout_updates)

        return dbc.Card([
            dbc.CardHeader([
                html.H5([
                    html.I(className="fas fa-chart-pie me-2 text-primary"),
                    component['component_title']
                ], className="mb-0"),
                html.Small(component.get('tooltip', 'Revenue breakdown by category'),
                           className="text-muted")
            ], className="bg-light border-0"),
            dbc.CardBody([
                dcc.Graph(
                    figure=fig,
                    style={'height': f'{visual_props.get("height", 500)}px'},
                    config={
                        'displayModeBar': True,
                        'displaylogo': False,
                        'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                        'toImageButtonOptions': {
                            'format': 'png',
                            'filename': f'{component["component_title"]}_pie_chart',
                            'height': 500,
                            'width': 700,
                            'scale': 2
                        }
                    }
                )
            ], className="p-3")
        ], className="shadow-sm border-0")

    # WATERFALL CHART HANDLING
    if graph_type == 'waterfall':
        # Get all attributes and calculations
        calculation = safe_get_calculation(component)
        calculations = calculation.split('|') if calculation and '|' in calculation else [calculation] * len(attributes)

        # Ensure equal lengths
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

                # FIXED: Use custom metric titles for waterfall categories
                if attr in metric_titles:
                    category_name = metric_titles[attr]
                else:
                    category_name = category_mapping.get(attr, attr.replace('cf_', '').replace('_', ' ').title())

                # Determine type based on attribute name and position
                if 'purchases' in attr or 'cost' in attr or 'expense' in attr or 'payment' in attr:
                    waterfall_type = 'negative'
                    value = -abs(value)  # Ensure negative values are negative
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
                title=custom_layout.get('title', component['component_title']),
                title_x=0.5,
                height=visual_props.get('height', 500),
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Segoe UI", size=12),
                yaxis=dict(
                    title="Value ($)",
                    tickformat="$,.0f",
                    gridcolor="#f0f0f0"
                ),
                xaxis=dict(
                    title="Financial Components",
                    tickangle=-45
                )
            )

            # FIXED: Apply custom layout excluding metricTitles
            if custom_layout:
                valid_layout_props = {k: v for k, v in custom_layout.items()
                                      if k not in ['metricTitles']}
                if valid_layout_props:
                    fig.update_layout(**valid_layout_props)

            return dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig, style={'height': f'{visual_props.get("height", 500)}px'})
                ])
            ], className="waterfall-container")
        else:
            return dbc.Alert("No waterfall data available", color="info")

    # Continue with existing chart types
    try:
        # Better index handling based on data type
        primary_attr = attributes[0]

        # Reset index and create proper x-axis
        df = df.reset_index()

        # Determine the appropriate x-axis column
        if 'index' in df.columns:
            x_col = 'index'
        elif len(df.columns) > 0:
            # Use the first column as x-axis if no index
            x_col = df.columns[0]
        else:
            return dbc.Alert("No suitable x-axis column found", color="warning")

        # NEW: Handle quarterly/historical data (like FCAS historical)
        if 'Quarter' in df.columns or data_file.endswith('_historical.csv'):
            # Handle quarterly data
            x_col = 'Quarter'

            # Handle multiple attributes for quarterly data
            if len(attributes) > 1:
                df_subset = df[attributes + ['Quarter']].copy()

                # Apply metric titles
                df_display = df_subset.copy()
                for attr in attributes:
                    if attr in metric_titles:
                        clean_name = metric_titles[attr]
                    else:
                        # Clean up attribute names (remove prefixes/suffixes, format nicely)
                        clean_name = attr.replace('AVG_', '').replace(' ($/MW)', '').replace('_', ' ').title()
                        # Handle specific FCAS naming
                        clean_name = clean_name.replace('Lowerreg', 'Lower Reg').replace('Raisereg', 'Raise Reg')
                        clean_name = clean_name.replace('Lower5min', 'Lower 5min').replace('Raise5min', 'Raise 5min')
                        clean_name = clean_name.replace('Lower60sec', 'Lower 60s').replace('Raise60sec', 'Raise 60s')
                        clean_name = clean_name.replace('Lower6sec', 'Lower 6s').replace('Raise6sec', 'Raise 6s')
                        clean_name = clean_name.replace('Lower1sec', 'Lower 1s').replace('Raise1sec', 'Raise 1s')
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
                                                                                                  'Raise 6s')
                                                        .replace('Lower1sec', 'Lower 1s').replace('Raise1sec',
                                                                                                  'Raise 1s'))
                                      for attr in attributes]

                df_melted = df_display.melt(id_vars=['Quarter'],
                                            value_vars=display_attributes,
                                            var_name='Series',
                                            value_name='Value')

                # Apply value modifiers
                df_melted = apply_value_modifiers(df_melted, attributes, component, metric_titles)

                # Create line chart for historical data
                fig = px.line(df_melted, x='Quarter', y='Value', color='Series',
                              title=component['component_title'],
                              color_discrete_sequence=px.colors.qualitative.Set3)

                # Update layout for quarterly data
                fig.update_layout(
                    xaxis_title="Quarter",
                    yaxis_title="Price ($/MW)",
                    height=visual_props.get('height', 450),
                    xaxis={'tickangle': 45}  # Rotate quarter labels for better readability
                )
            else:
                # Single attribute
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

            # FIXED: Handle multiple attributes for ALL chart types
            if len(attributes) > 1:
                df_subset = df[attributes + ['month_name']].copy()

                # FIXED: Use metric titles if available, otherwise clean up attribute names
                df_display = df_subset.copy()
                for attr in attributes:
                    # Use custom title if available, otherwise generate clean name
                    if attr in metric_titles:
                        clean_name = metric_titles[attr]
                    else:
                        clean_name = attr.replace('monthly_', '').replace('_', ' ').title()
                    df_display = df_display.rename(columns={attr: clean_name})

                # Get the final column names (either custom titles or cleaned names)
                display_attributes = [metric_titles.get(attr, attr.replace('monthly_', '').replace('_', ' ').title())
                                      for attr in attributes]

                df_melted = df_display.melt(id_vars=['month_name'],
                                            value_vars=display_attributes,
                                            var_name='Series',
                                            value_name='Value')

                # *** NEW: Apply value modifiers for negative values ***
                df_melted = apply_value_modifiers(df_melted, attributes, component, metric_titles)

                # FIXED: Create charts based on graph_type for multi-attribute data
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
                    # FIXED: Area chart logic - explicit control via graph_type
                    should_stack = (graph_type == 'stacked_area')

                    fig = go.Figure()
                    colors = px.colors.qualitative.Set3

                    for i, display_attr in enumerate(display_attributes):
                        series_data = df_melted[df_melted['Series'] == display_attr]

                        if should_stack:
                            # Stacked area chart (stacked_area)
                            fill_mode = 'tonexty' if i > 0 else 'tozeroy'
                            opacity = 0.7
                            hovermode = 'closest'
                        else:
                            # Overlapping area chart (area - default)
                            fill_mode = 'tozeroy'  # Each area fills to zero independently
                            opacity = 0.4  # More transparent for better visibility of overlaps
                            hovermode = 'x unified'

                        # Handle color with proper opacity
                        base_color = colors[i % len(colors)]
                        if base_color.startswith('#'):
                            # Convert hex to rgba
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
                            hovertemplate=f'<b>{display_attr}</b><br>' +
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
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )

                    # FIXED: Add zero line for battery flow charts with safe checking
                    if has_battery_attributes(attributes):
                        fig.add_hline(y=0, line_dash="dash", line_color="gray",
                                      annotation_text="Zero Line", annotation_position="bottom right")

                elif graph_type == 'line':
                    # FIXED: Create multi-attribute line chart
                    fig = px.line(df_melted, x='month_name', y='Value', color='Series',
                                  title=component['component_title'],
                                  color_discrete_sequence=px.colors.qualitative.Set3)
                else:
                    # Default to line for multi-attribute
                    fig = px.line(df_melted, x='month_name', y='Value', color='Series',
                                  title=component['component_title'],
                                  color_discrete_sequence=px.colors.qualitative.Set3)
            else:
                # Single attribute charts (existing logic)
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

            # FIXED: Update layout for monthly data (without metricTitles)
            if len(attributes) == 1 or graph_type in ['stacked_bar', 'grouped_bar', 'line']:
                fig.update_layout(
                    xaxis_title="Month",
                    xaxis={'categoryorder': 'array',
                           'categoryarray': month_names},
                    yaxis_title=component.get('display_format', '').replace('{', '').replace('}', '').replace(':', ''),
                    height=visual_props.get('height', 400)
                )

        elif 'hourly' in data_file and len(df) == 8760:
            # Handle hourly data (like batt_SOC_year1)
            df['hour'] = range(1, len(df) + 1)
            x_col = 'hour'

            # FIXED: Handle multiple attributes for hourly data
            if len(attributes) > 1:
                # Create melted DataFrame for multiple series
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

                # *** NEW: Apply value modifiers ***
                df_melted = apply_value_modifiers(df_melted, attributes, component, metric_titles)

                if graph_type in ['area', 'stacked_area']:
                    # FIXED: Area chart logic for hourly data
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
                            hovertemplate=f'<b>{display_attr}</b><br>' +
                                          'Hour: %{x}<br>' +
                                          'Value: %{y:$,.0f}<br>' +
                                          '<extra></extra>'
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
            # Handle annual data
            df['year'] = range(len(df))
            x_col = 'year'

            # FIXED: Handle multiple attributes for annual data
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

                # *** NEW: Apply value modifiers ***
                df_melted = apply_value_modifiers(df_melted, attributes, component, metric_titles)

                if graph_type in ['area', 'stacked_area']:
                    # FIXED: Area chart logic for annual data
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
                            hovertemplate=f'<b>{display_attr}</b><br>' +
                                          'Year: %{x}<br>' +
                                          'Value: %{y:$,.0f}<br>' +
                                          '<extra></extra>'
                        ))

                    fig.update_layout(
                        title=component['component_title'],
                        xaxis_title="Year",
                        yaxis_title="Value",
                        height=visual_props.get('height', 400),
                        hovermode=hovermode
                    )
                elif graph_type == 'line':
                    fig = px.line(df_melted, x='year', y='Value', color='Series',
                                  title=component['component_title'],
                                  color_discrete_sequence=px.colors.qualitative.Set3)
                elif graph_type == 'bar':
                    fig = px.bar(df_melted, x='year', y='Value', color='Series',
                                 title=component['component_title'])
                else:
                    fig = px.line(df_melted, x='year', y='Value', color='Series',
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

            fig.update_layout(
                xaxis_title="Year",
                yaxis_title=primary_attr.replace('_', ' ').title() if len(attributes) == 1 else "Value",
                height=visual_props.get('height', 400)
            )

        else:
            # Generic handling for other data types
            # FIXED: Handle multiple attributes for generic data
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

                # *** NEW: Apply value modifiers ***
                df_melted = apply_value_modifiers(df_melted, attributes, component, metric_titles)

                if graph_type in ['area', 'stacked_area']:
                    # FIXED: Area chart logic for generic data
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
                            hovertemplate=f'<b>{display_attr}</b><br>' +
                                          '%{x}<br>' +
                                          'Value: %{y:$,.0f}<br>' +
                                          '<extra></extra>'
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

        # FIXED: Apply custom layout AFTER removing metricTitles
        if custom_layout:
            # Only apply valid Plotly layout properties
            valid_layout_props = {k: v for k, v in custom_layout.items()
                                  if k not in ['metricTitles']}  # Exclude our custom properties
            if valid_layout_props:
                fig.update_layout(**valid_layout_props)

        height = visual_props.get('height', 400)

        # ENHANCED: Return with proper card wrapper for ALL graph types
        return dbc.Card([
            dbc.CardHeader([
                html.H5([
                    html.I(className="fas fa-chart-line me-2 text-primary"),
                    component['component_title']
                ], className="mb-0"),
                html.Small(component.get('tooltip', 'Data visualization'),
                           className="text-muted")
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
        import traceback
        traceback.print_exc()
        return dbc.Alert(f"Graph creation error: {str(e)}", color="danger")

# ==============================================================================
# OPTIMIZED LAYOUT FUNCTIONS
# ==============================================================================

def create_data_driven_tabs(structure_df):
    """Create tabs from CSV structure"""
    tab_info = structure_df.groupby('tab_name').agg({
        'tab_order': 'min'
    }).reset_index().sort_values('tab_order', na_position='last')

    tabs = []
    tab_icons = {
        'executive summary': '🏠',
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


def create_section_content_from_csv_FIXED(components, data_sources):
    """Fixed section content creation with proper component type handling"""
    content_rows = []
    current_row_components = []
    current_row_width = 0

    width_mapping = {'quarter': 3, 'third': 4, 'half': 6, 'full': 12}

    for _, component in components.iterrows():
        comp_width = width_mapping.get(component.get('width', 'full'), 12)

        # Strip whitespace from component_type
        component_type = str(component['component_type']).strip().lower()

        print(f"DEBUG: Processing component {component.get('component_id', 'unknown')}: "
              f"{component.get('component_title', 'unknown')} - type: '{component_type}'")

        try:
            # Create component based on type
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
                # FIXED: Better table type detection logic
                graph_type = str(component.get('graph_type', '')).strip().lower()
                data_source = str(component.get('data_source', '')).strip()
                component_title = str(component.get('component_title', '')).strip().lower()

                print(
                    f"DEBUG: Table detection - graph_type: '{graph_type}', data_source: '{data_source}', title: '{component_title}'")

                # FIXED: Use landscape table ONLY for:
                # 1. Explicitly marked as landscape in graph_type
                # 2. Tables that are primarily cashflow focused (not mixed data sources)
                # 3. Tables with specific cashflow-focused titles

                use_landscape = False

                # Check for explicit landscape marking
                if 'landscape' in graph_type:
                    use_landscape = True
                    print(f"DEBUG: Using landscape - explicit graph_type")

                # Check for cashflow-focused tables (not mixed source tables)
                elif (data_source == 'cashflow_timeseries.parquet' and
                      ('cashflow' in component_title or 'payback' in component_title)):
                    use_landscape = True
                    print(f"DEBUG: Using landscape - pure cashflow table")

                # FIXED: Exclude mixed-source tables even if they contain cashflow data
                elif ('|' in data_source and
                      'cashflow_timeseries.parquet' in data_source and
                      len(data_source.split('|')) > 3):  # Multiple different sources
                    use_landscape = False
                    print(f"DEBUG: Using pipe-separated - mixed data sources")

                # Default to pipe-separated for everything else
                else:
                    use_landscape = False
                    print(f"DEBUG: Using pipe-separated - default")

                if use_landscape:
                    print(f"DEBUG: Creating landscape table for {component.get('component_title', 'Unknown')}")
                    comp_element = create_generic_landscape_table(component, data_sources)
                else:
                    print(f"DEBUG: Creating pipe-separated table for {component.get('component_title', 'Unknown')}")
                    comp_element = create_pipe_separated_table(component, data_sources)

                current_row_components.append(dbc.Col(comp_element, width=comp_width, className="mb-4"))

            else:
                print(f"WARNING: Unknown component type '{component_type}' for component "
                      f"{component.get('component_id', 'unknown')}")
                continue

        except Exception as e:
            print(f"ERROR: Failed to create component {component.get('component_id', 'unknown')}: {str(e)}")
            import traceback
            traceback.print_exc()
            # Add error component instead of crashing
            error_component = dbc.Alert(
                f"Error loading component: {component.get('component_title', 'Unknown')}",
                color="danger"
            )
            current_row_components.append(dbc.Col(error_component, width=comp_width, className="mb-4"))

        current_row_width += comp_width

        # Start new row if we've reached 12 columns
        if current_row_width >= 12:
            content_rows.append(dbc.Row(current_row_components, className="g-3"))
            current_row_components = []
            current_row_width = 0

    # Add remaining components
    if current_row_components:
        content_rows.append(dbc.Row(current_row_components, className="g-3"))

    return html.Div(content_rows)


def create_standard_table(component, data_sources):
    """Create standard table component"""
    data_file = component.get('data_source', '')

    if data_file not in data_sources:
        return dbc.Alert(f"Data file not found: {data_file}", color="warning")

    df = data_sources[data_file].copy()
    if not isinstance(df, pd.DataFrame):
        return dbc.Alert(f"Data source is not a DataFrame: {data_file}", color="warning")

    # Handle attributes
    if '|' in component['attribute']:
        attributes = component['attribute'].split('|')
        df_display = df[attributes].copy()
    else:
        df_display = df[[component['attribute']]].copy()

    df_display = df_display.reset_index()

    # Format columns
    display_format = component.get('display_format', '')
    if display_format:
        for col in df_display.select_dtypes(include=['float64', 'int64']).columns:
            if col != 'index':
                df_display[col] = df_display[col].apply(lambda x: format_metric_value(x, display_format))

    columns = [{"name": col, "id": col} for col in df_display.columns]
    visual_props = safe_literal_eval(component.get('visual_properties', '{}'))
    page_size = visual_props.get('pageSize', 10)

    return dbc.Card([
        dbc.CardHeader([
            html.H6(component['component_title'], className="mb-0")
        ]),
        dbc.CardBody([
            dash_table.DataTable(
                columns=columns,
                data=df_display.to_dict('records'),
                filter_action="native",
                sort_action="native",
                page_action="native",
                page_size=page_size,
                style_table={'overflowX': 'auto'},
                style_cell={'padding': '10px', 'textAlign': 'left'},
                style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'}
            )
        ])
    ], className="shadow-sm border-0")


def create_data_driven_tab_content(tab_name, structure_df, data_sources):
    """Create tab content with enhanced FCAS support"""
    tab_components = structure_df[structure_df['tab_name'] == tab_name].sort_values('order', na_position='last')

    if tab_components.empty:
        return dbc.Alert(f"No components defined for {tab_name}", color="info")

    # Group by sections
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

        # Check for panel components
        has_panels = any(comp['component_type'] == 'panel' for _, comp in section_components.iterrows())

        if has_panels:
            panel_sections.append(section)
        else:
            regular_sections.append(section)

    # Create layout
    if panel_sections:
        return create_wrapping_layout(tab_name, structure_df, data_sources, panel_sections, regular_sections)
    else:
        return create_standard_tab_layout(tab_name, structure_df, data_sources, sections)


def create_wrapping_layout(tab_name, structure_df, data_sources, panel_sections, regular_sections):
    """Create wrapping layout for tabs with panels"""
    # Create panels
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

    # Create other content
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
            create_section_content_from_csv_FIXED(section_components, data_sources)
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

        # Add section header
        content_sections.append(
            html.H4([
                html.I(className="fas fa-layer-group me-2 text-primary"),
                section
            ], className="section-title mb-4")
        )

        # Create section content
        section_content = create_section_content_from_csv_FIXED(section_components, data_sources)
        content_sections.append(section_content)

    return html.Div(content_sections)


def debug_landscape_table_data(component, data_sources):
    """Debug function to inspect landscape table data structure"""
    data_file = component.get('data_source', '')

    if '|' in data_file:
        source_files = data_file.split('|')
        for source_file in source_files:
            if source_file in data_sources:
                data_file = source_file
                break

    if data_file not in data_sources:
        print(f"DEBUG: Data file {data_file} not found")
        return

    df = data_sources[data_file]
    attributes = component['attribute'].split('|') if '|' in component['attribute'] else [component['attribute']]

    print(f"DEBUG: Data file: {data_file}")
    print(f"DEBUG: DataFrame shape: {df.shape}")
    print(f"DEBUG: DataFrame columns: {list(df.columns)}")
    print(f"DEBUG: Requested attributes: {attributes}")
    print(f"DEBUG: Missing attributes: {[attr for attr in attributes if attr not in df.columns]}")
    print(f"DEBUG: Sample data:")
    for attr in attributes[:3]:  # Show first 3
        if attr in df.columns:
            print(f"  {attr}: {df[attr].head().tolist()}")


print("=== LANDSCAPE TABLE FIXES ===")
print("✅ Fixed data validation and NaN handling")
print("✅ Improved column consistency")
print("✅ Disabled problematic filtering")
print("✅ Added comprehensive error handling")
print("✅ Better component type detection")
print("✅ Added debug function for troubleshooting")


def load_data_with_metadata():
    """Enhanced data loading function"""
    print(f"\n{'=' * 50}\nEnhanced FCAS Results Dashboard Loading\n{'=' * 50}")

    structure_df = pd.DataFrame()
    data_sources = {}

    # Load structure definition
    try:
        if STRUCTURE_PATH.exists():
            structure_df = pd.read_csv(STRUCTURE_PATH)
            print(f"✅ Loaded dashboard structure with {len(structure_df)} components")
        else:
            print(f"⚠️ Structure file not found at {STRUCTURE_PATH}")
    except Exception as e:
        print(f"❌ Error loading structure: {str(e)}")

    # Load data sources
    if not structure_df.empty:
        data_sources = load_data_sources_from_csv(structure_df)
    else:
        print("⚠️ No structure data loaded")

    return structure_df, data_sources


def create_enhanced_app_layout():
    """Create the complete enhanced layout"""
    return html.Div([
        # Enhanced Header
        html.Div([
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.H1("EnTrans Results Dashboard v6", className="text-white mb-2"),
                        html.P("Enhanced FCAS Revenue Analysis & Energy Arbitrage Platform",
                               className="text-white mb-0 opacity-75")
                    ], width=8),
                    dbc.Col([
                        html.Div([
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

        # Footer
        html.Footer([
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.P("EnTrans Energy Solutions", className="mb-1 fw-bold"),
                        html.P("Enhanced FCAS Revenue Analysis Platform v6.0",
                               className="text-muted small mb-0")
                    ], width=6),
                    dbc.Col([
                        html.P(f"Generated: {datetime.now().strftime('%Y-%m-%d at %H:%M')}",
                               className="mb-1 text-end"),
                        html.P("Confidential Analysis",
                               className="text-muted small mb-0 text-end")
                    ], width=6)
                ])
            ])
        ], className="mt-5 pt-4 border-top text-muted")
    ])


# ==============================================================================
# CALLBACKS
# ==============================================================================

@app.callback(
    Output('dynamic-tabs', 'children'),
    [Input('dynamic-tabs', 'id')]
)
def create_tabs(_):
    """Create navigation tabs dynamically"""
    global structure_df
    if structure_df is None or structure_df.empty:
        return html.Div("Loading tabs...", className="mb-4")
    return create_data_driven_tabs(structure_df)


@app.callback(
    Output('tab-content', 'children'),
    [Input('main-tabs', 'active_tab')]
)
def render_tab_content(active_tab):
    """Render tab content with FCAS enhancements"""
    if not active_tab or structure_df is None:
        return dbc.Alert("Loading...", color="info")

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

    return content


# ==============================================================================
# INDEX STRING WITH STYLING
# ==============================================================================

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>EnTrans Results Dashboard v6 - Enhanced FCAS</title>
        {%favicon%}
        {%css%}
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
        <style>
''' + ENHANCED_CSS + '''
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("ENTRANS ENHANCED FCAS RESULTS DASHBOARD v6")
    print("🆕 Enhanced FCAS Revenue Analysis")
    print("🆕 Energy Arbitrage Breakdown")
    print("🆕 Market Integration Metrics")
    print("🆕 Optimized Performance")
    print("=" * 60)

    # Global variables
    structure_df = None
    data_sources = {}

    try:
        # Load data with enhanced FCAS support
        structure_df, data_sources = load_data_with_metadata()

        print(f"✅ Structure components: {len(structure_df)}")
        print(f"✅ Data sources: {len(data_sources)}")

        if len(structure_df) > 0:
            # Analyze FCAS integration
            scalar_data = data_sources.get('scalar_results.csv', {})
            total_fcas = scalar_data.get('Total ancillary services revenue', 0)

            if total_fcas and float(total_fcas) > 0:
                print(f"💰 FCAS Revenue Integration: ${float(total_fcas):,.0f}")

                # Show individual services
                for i in range(1, 5):
                    service_key = f'Ancillary services {i} revenue'
                    if service_key in scalar_data:
                        service_revenue = scalar_data[service_key]
                        print(f"   Service {i}: ${float(service_revenue):,.0f}")
            else:
                print("⚠️ No FCAS revenue found in results")

            # Check for cashflow FCAS data
            cashflow_data = data_sources.get('cashflow_timeseries.parquet')
            if isinstance(cashflow_data, pd.DataFrame):
                fcas_cols = [col for col in cashflow_data.columns if 'cf_ancillary_services_' in col]
                if fcas_cols:
                    print(f"📊 FCAS Cashflow Streams: {len(fcas_cols)} services")
                    for col in fcas_cols[:4]:  # Show first 4
                        total = cashflow_data[col].sum()
                        if total > 0:
                            print(f"   {col}: ${total:,.0f}")

            # Show enhanced metrics
            calculated_data = data_sources.get('calculated', {})
            if calculated_data:
                fcas_annual = calculated_data.get('fcas_revenue_annual', 0)
                participation = calculated_data.get('fcas_participation_rate', 0)

                print(f"📈 Enhanced Metrics:")
                print(f"   Annual FCAS: ${fcas_annual:,.0f}")
                print(f"   Participation Rate: {participation:.1f}%")

        # Set layout
        app.layout = create_enhanced_app_layout()

        print("\n🚀 Enhanced FCAS Dashboard Ready")
        print("✅ Executive Summary: Total Ancillary Revenue")
        print("✅ CashFlow Tab: Annual FCAS Breakdown")
        print("✅ Market Analysis: Detailed Service Revenue")
        print("✅ Optimized Performance & Code")

    except Exception as e:
        print(f"❌ Error in enhanced loading: {e}")
        import traceback

        traceback.print_exc()

        # Fallback layout
        structure_df = pd.DataFrame()
        data_sources = {}
        app.layout = html.Div([
            dbc.Alert("Enhanced dashboard loading error. Check console.", color="danger")
        ])

    print(f"\n🌐 Starting Enhanced FCAS Dashboard on port 8050...")
    print("=" * 60)

    app.run(debug=True, port=8050)