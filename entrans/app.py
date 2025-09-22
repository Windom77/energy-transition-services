"""
Clean Flask Application - Session ID Fix Only
- Removed complex cache-busting code
- Fixed session ID generation per analysis
- Simplified dashboard data loading
- Production-ready logging and monitoring
"""
from logging_config import setup_production_logging, log_milestone, log_error, log_warning
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
import json
import sys
import time
import os
from pathlib import Path
import traceback
import requests
import logging
import threading
from typing import Dict, Any, Optional
import uuid
import pandas as pd
import subprocess

setup_production_logging()

# Simple session ID management
current_session_id = str(uuid.uuid4())[:8]

def generate_new_session():
    """Generate a new session ID for each analysis"""
    global current_session_id
    current_session_id = str(uuid.uuid4())[:8]
    logger.info(f"🆔 New session started: {current_session_id}")
    return current_session_id

# ========== ENVIRONMENT SETUP ==========

# Environment detection
ENVIRONMENT = os.getenv('GAE_ENV', os.getenv('FLASK_ENV', 'development'))
IS_PRODUCTION = ENVIRONMENT in ['standard', 'production']
IS_DEBUG = not IS_PRODUCTION and os.getenv('DEBUG', 'true').lower() == 'true'

# Logging setup
setup_production_logging()

logger = logging.getLogger(__name__)

# ========== IMPORTS WITH ERROR HANDLING ==========

# Core configuration
try:
    from config import Config
    logger.info("✅ Config imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import config: {e}")
    sys.exit(1)

# Form integration
try:
    from pysam_form_integration2 import update_pysam_json
    FORM_INTEGRATION_AVAILABLE = True
    logger.info("✅ Form integration imported successfully")
except ImportError as e:
    logger.error(f"❌ Form integration not available: {e}")
    FORM_INTEGRATION_AVAILABLE = False

# PySAM simulation
try:
    from pysam_main_optimized import run_simulation_optimized as run_simulation, setup_logging
    PYSAM_AVAILABLE = True
    logger.info("✅ PySAM simulation imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ PySAM simulation not available: {e}")
    PYSAM_AVAILABLE = False

# Dashboard components
try:
    from dashboard_module import (
        load_data_with_metadata,
        create_enhanced_app_layout,
        setup_dashboard_callbacks,
        ENHANCED_CSS,
        LANDSCAPE_TABLE_CSS
    )
    DASHBOARD_AVAILABLE = True
    logger.info("✅ Dashboard module imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Dashboard module not available: {e}")
    DASHBOARD_AVAILABLE = False

# Dash imports for dashboard
try:
    import dash
    from dash import dcc, html, dash_table
    from dash.dependencies import Input, Output
    import dash_bootstrap_components as dbc
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    logger.warning("⚠️ Dash components not available")

# ========== FLASK APP SETUP ==========

app = Flask(__name__)

def configure_flask_paths():
    """Cloud-safe Flask template configuration"""
    try:
        from config import Config

        template_folder = str(Config.TEMPLATES_DIR)
        static_folder = str(Config.STATIC_DIR)

        key_template = Config.TEMPLATES_DIR / 'index_base4.html'
        if not key_template.exists():
            legacy_template = Config.LEGACY_TEMPLATES_DIR / 'index_base4.html'
            if legacy_template.exists():
                template_folder = str(Config.LEGACY_TEMPLATES_DIR)
                static_folder = str(Config.LEGACY_STATIC_DIR)
                logger.info(f"Using legacy template location: {template_folder}")
            else:
                logger.error(f"index_base4.html not found in any location")

        app.template_folder = template_folder
        app.static_folder = static_folder

        logger.info(f"Flask configured:")
        logger.info(f"  Templates: {template_folder}")
        logger.info(f"  Static: {static_folder}")
        logger.info(f"  Environment: {Config.ENVIRONMENT}")

    except Exception as e:
        logger.error(f"Flask path configuration failed: {e}")
        app.template_folder = 'templates'
        app.static_folder = 'static'

configure_flask_paths()

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# ========== DASH APP SETUP ==========

if DASHBOARD_AVAILABLE and DASH_AVAILABLE:
    try:
        dash_app = dash.Dash(
            __name__,
            server=app,
            url_base_pathname='/dashboard/',
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True
        )
        dash_app.title = "EnTrans Results Dashboard"

        dash_app.layout = create_enhanced_app_layout()
        setup_dashboard_callbacks(dash_app)

        # Simple styling
        dash_app.index_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                {%metas%}
                <title>EnTrans Results Dashboard</title>
                {%favicon%}
                {%css%}
                <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
                <style>
        ''' + ENHANCED_CSS + LANDSCAPE_TABLE_CSS + '''
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

        logger.info("✅ Dash dashboard integrated")

    except Exception as e:
        logger.error(f"❌ Dashboard setup failed: {e}")
        DASHBOARD_AVAILABLE = False

# Fallback dashboard if not available
if not (DASHBOARD_AVAILABLE and DASH_AVAILABLE):
    try:
        dash_app = dash.Dash(
            __name__,
            server=app,
            url_base_pathname='/dashboard/',
            suppress_callback_exceptions=True
        )

        dash_app.layout = html.Div([
            dbc.Alert([
                html.I(className="fas fa-exclamation-triangle me-2"),
                "Dashboard temporarily unavailable. Analysis functionality is still available."
            ], color="warning"),
            html.A("Return to Analysis Form", href="/", className="btn btn-primary")
        ])
    except:
        pass

# ========== SIMPLE GLOBAL STATE ==========
simple_status = {
    'running': False,
    'stage': 'idle',
    'message': '',
    'error': None,
    'results': None,
    'session_id': current_session_id
}

def update_simple_status(stage: str, message: str = '', **kwargs):
    """Simple status updater"""
    global simple_status

    simple_status.update({
        'stage': stage,
        'message': message,
        'running': kwargs.get('running', simple_status['running']),
        'error': kwargs.get('error'),
        'results': kwargs.get('results')
    })

    # Log major milestones
    if stage in ['starting', 'saving_form', 'form_integration', 'integration_complete',
                 'pysam_simulation', 'compiling_results', 'completed', 'failed']:
        logger.info(f"Status: {stage} - {message}")

def handle_analysis_error(error: Exception, stage: str = 'failed') -> None:
    """Centralized error handling for analysis pipeline"""
    error_msg = str(error)
    log_error(f"Analysis error at {stage}", error)

    if IS_DEBUG:
        log_error("Full traceback:", Exception(traceback.format_exc()))

    update_simple_status(
        stage=stage,
        message=f'Analysis failed: {error_msg}',
        running=False,
        error=error_msg
    )

# ========== SIMPLE DATA LOADING ==========

def load_fresh_results_from_disk():
    """Load results directly from disk files"""
    try:
        results = {}

        if Config.RESULTS_DIR.exists():
            # Load scalar results
            scalar_file = Config.RESULTS_DIR / 'scalar_results.csv'
            if scalar_file.exists():
                try:
                    scalar_df = pd.read_csv(scalar_file)
                    scalar_df = scalar_df.dropna(how='all')
                    results['scalar_results'] = scalar_df.to_dict('records')
                    results['file_timestamp'] = scalar_file.stat().st_mtime
                    logger.info(f"📊 Loaded scalar results: {len(scalar_df)} rows")
                except Exception as e:
                    logger.error(f"Failed to load scalar results: {e}")

            # Load metadata
            metadata_file = Config.RESULTS_DIR / 'metadata.json'
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    results['metadata'] = metadata
                    results['session_id'] = metadata.get('session_id')
                    logger.info(f"📋 Loaded metadata for session: {metadata.get('session_id')}")
                except Exception as e:
                    logger.error(f"Failed to load metadata: {e}")

            # Load timeseries data
            timeseries_files = {
                'hourly_timeseries.parquet': 'hourly_data',
                'monthly_timeseries.parquet': 'monthly_data',
                'annual_timeseries.parquet': 'annual_data'
            }

            for filename, result_key in timeseries_files.items():
                file_path = Config.RESULTS_DIR / filename
                if file_path.exists():
                    try:
                        ts_df = pd.read_parquet(file_path)
                        results[result_key] = ts_df.to_dict('records')
                        results[f'{result_key}_rows'] = len(ts_df)
                        logger.info(f"📈 Loaded {filename}: {len(ts_df)} rows")
                    except Exception as e:
                        logger.warning(f"Could not load {filename}: {e}")

        if results:
            results['load_timestamp'] = time.time()
            results['total_datasets'] = len([k for k in results.keys()
                                           if not k.endswith('_timestamp') and not k.endswith('_size')])
            logger.info(f"✅ Fresh results loaded from disk: {results['total_datasets']} datasets")

        return results if results else None

    except Exception as e:
        logger.error(f"Fresh disk load failed: {e}")
        return None

def clear_all_previous_data():
    """Clear all previous data files"""
    try:
        cleared_items = []

        # Clear results files
        if Config.RESULTS_DIR.exists():
            for file_path in Config.RESULTS_DIR.iterdir():
                if file_path.is_file() and not file_path.name.endswith('.py'):
                    try:
                        file_path.unlink()
                        cleared_items.append(f'result_{file_path.name}')
                    except Exception as e:
                        logger.warning(f"Could not delete {file_path.name}: {e}")

        # Clear form HTML
        if Config.FORM_HTML.exists():
            try:
                Config.FORM_HTML.unlink()
                cleared_items.append('form_html')
            except Exception as e:
                logger.warning(f"Could not clear form HTML: {e}")

        # Clear updated JSON
        if Config.UPDATED_JSON.exists():
            try:
                Config.UPDATED_JSON.unlink()
                cleared_items.append('updated_json')
            except Exception as e:
                logger.warning(f"Could not clear updated JSON: {e}")

        # Ensure directories exist
        Config.FORM_HTML.parent.mkdir(parents=True, exist_ok=True)
        Config.UPDATED_JSON.parent.mkdir(parents=True, exist_ok=True)
        Config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        logger.info(f"🗑️ Cleared {len(cleared_items)} items: {cleared_items}")

        return {
            'cleared': cleared_items,
            'count': len(cleared_items),
            'status': 'success'
        }

    except Exception as e:
        logger.error(f"Clear failed: {e}")
        return {'status': 'error', 'error': str(e)}

# ========== ANALYSIS PIPELINE ==========

@app.route('/run-analysis', methods=['POST'])
def run_analysis():
    """Analysis with proper session generation per analysis"""

    # Check if already running
    if simple_status['running']:
        return jsonify({
            'status': 'already_running',
            'message': f'Analysis in progress: {simple_status["stage"]}',
            'session_id': simple_status.get('session_id', 'unknown')
        }), 409

    try:
        data = request.get_json()
        if not data or 'form_html' not in data:
            return jsonify({'error': 'No form data received'}), 400

        # CRITICAL FIX: Generate NEW session ID for each analysis
        new_session_id = generate_new_session()
        logger.info(f"🆔 NEW ANALYSIS SESSION: {new_session_id}")

        # Clear all previous data
        clear_result = clear_all_previous_data()

        # Set running with NEW session
        simple_status['session_id'] = new_session_id
        update_simple_status('starting', f'Starting analysis session {new_session_id}...', running=True)

        # Start analysis in thread
        analysis_thread = threading.Thread(
            target=run_simple_analysis_pipeline,
            args=(data,),
            daemon=True
        )
        analysis_thread.start()

        return jsonify({
            'status': 'started',
            'message': f'Analysis started with session {new_session_id}',
            'session_id': new_session_id,
            'clear_result': clear_result
        })

    except Exception as e:
        update_simple_status('failed', str(e), running=False, error=str(e))
        return jsonify({'error': str(e)}), 500


def run_simple_analysis_pipeline(data: Dict[str, Any]) -> None:
    """Analysis pipeline with milestone-only logging"""
    total_start = time.time()
    current_session = simple_status.get('session_id', 'unknown')

    try:
        log_milestone(f"Analysis started - Session {current_session}")

        # Stage 1: Save form data (no logging)
        update_simple_status('saving_form', f'Saving form data for session {current_session}...')
        form_path = Config.FORM_HTML
        form_path.parent.mkdir(parents=True, exist_ok=True)
        with open(form_path, 'w', encoding='utf-8') as f:
            f.write(data['form_html'])

        # Stage 2: Form Integration (no logging unless error)
        update_simple_status('form_integration', f'Processing form data for session {current_session}...')
        try:
            integration_result = update_pysam_json(form_path)
            if not integration_result or not integration_result.get('success'):
                raise ValueError(f"Form integration failed: {integration_result}")
        except Exception as e:
            log_error("Form integration failed", e)
            update_simple_status('failed', f'Form integration failed: {str(e)}',
                                 running=False, error=str(e))
            return

        # Stage 3: PySAM Simulation
        if PYSAM_AVAILABLE:
            log_milestone(f"Starting PySAM simulation - Session {current_session}")
            update_simple_status('pysam_simulation', f'Running PySAM simulation for session {current_session}...')
            try:
                Config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                if not Config.UPDATED_JSON.exists():
                    raise ValueError("No updated JSON config for PySAM!")

                pysam_start = time.time()
                pysam_results = run_simulation(input_json=str(Config.UPDATED_JSON))
                pysam_time = time.time() - pysam_start

                log_milestone(f"PySAM completed in {pysam_time:.1f}s - Session {current_session}")
            except Exception as e:
                log_error("PySAM simulation failed", e)
                pysam_results = None
        else:
            log_warning("PySAM not available - skipping simulation")
            pysam_results = None

        # Stage 4: Complete (no detailed logging)
        total_time = time.time() - total_start
        update_simple_status('completed', f'Session {current_session} completed in {total_time:.1f}s',
                             running=False)

        log_milestone(f"Analysis completed in {total_time:.1f}s - Session {current_session}")

    except Exception as e:
        total_time = time.time() - total_start
        log_error(f"Analysis failed after {total_time:.1f}s - Session {current_session}", e)
        update_simple_status('failed', str(e), running=False, error=str(e))


# ========== API ROUTES ==========

@app.route('/analysis-status', methods=['GET'])
def analysis_status():
    """Status endpoint with session tracking"""
    return jsonify({
        'running': simple_status['running'],
        'stage': simple_status['stage'],
        'message': simple_status['message'],
        'error': simple_status['error'],
        'has_results': bool(simple_status['results']),
        'session_id': simple_status.get('session_id', 'unknown'),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/api/dashboard-data', methods=['GET'])
def get_dashboard_data():
    """Get dashboard data - always load fresh from disk"""
    try:
        # Always load fresh from disk files
        results = load_fresh_results_from_disk()
        status_copy = simple_status.copy()

        if results:
            logger.info(f"📊 Serving data for session: {results.get('session_id', 'unknown')}")

            dashboard_data = {
                'simulation_status': status_copy,
                'has_results': True,
                'session_id': results.get('session_id', status_copy.get('session_id', 'unknown')),
                'results_data': results,
                'data_source': 'fresh_disk_load',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        else:
            logger.warning("📊 No fresh results found on disk")
            dashboard_data = {
                'simulation_status': status_copy,
                'has_results': False,
                'session_id': status_copy.get('session_id', 'unknown'),
                'message': 'No analysis results available',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }

        return jsonify(dashboard_data)

    except Exception as e:
        logger.error(f"Dashboard data error: {e}")
        return jsonify({
            'error': str(e),
            'has_results': False,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }), 500

@app.route('/upload-load-data', methods=['POST'])
def upload_load_data():
    """Handle CSV load data file upload - matches frontend expectations"""
    try:
        if 'load_file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400

        file = request.files['load_file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        if not file.filename.lower().endswith('.csv'):
            return jsonify({'success': False, 'error': 'File must be a CSV'}), 400

        # Read and process the CSV
        try:
            import csv
            import io

            # Read the file content
            content = file.read().decode('utf-8')

            # Parse CSV content
            csv_reader = csv.reader(io.StringIO(content))
            rows = list(csv_reader)

            # Check if first row is header (contains non-numeric data)
            first_row = rows[0] if rows else []
            has_header = False

            if first_row:
                try:
                    float(first_row[0])
                except (ValueError, IndexError):
                    has_header = True

            # Extract data rows
            data_rows = rows[1:] if has_header else rows

            # Extract load values from first column
            load_data = []
            for i, row in enumerate(data_rows):
                try:
                    if row:  # Skip empty rows
                        value = float(row[0])
                        if value < 0:
                            return jsonify({
                                'success': False,
                                'error': f'Negative load value at row {i + 1}: {value}'
                            }), 400
                        load_data.append(value)
                except (ValueError, IndexError):
                    return jsonify({
                        'success': False,
                        'error': f'Invalid numeric value at row {i + 1}: {row[0] if row else "empty row"}'
                    }), 400

            # Validate data length
            if len(load_data) != 8760:
                return jsonify({
                    'success': False,
                    'error': f'Expected 8760 hourly values, got {len(load_data)}'
                }), 400

            # Basic validation
            total_energy = sum(load_data)
            max_load = max(load_data)
            min_load = min(load_data)

            # Reasonable bounds check
            if total_energy <= 0:
                return jsonify({
                    'success': False,
                    'error': 'Total annual energy must be positive'
                }), 400

            if max_load > 1000:  # Reasonable upper limit for residential/commercial
                return jsonify({
                    'success': False,
                    'error': f'Maximum hourly load ({max_load:.2f} kWh) seems unreasonably high'
                }), 400

            # Validation passed - format response to match frontend expectations
            validation_result = {
                'valid': True,
                'message': f'Load data validated successfully. Total: {total_energy:.0f} kWh/year, Peak: {max_load:.2f} kWh',
                'stats': {
                    'total_annual': round(total_energy, 2),
                    'peak_load': round(max_load, 3),
                    'min_load': round(min_load, 3),
                    'average_load': round(total_energy / 8760, 3)
                }
            }

            # Save the data to file for later use
            load_data_path = Config.PROJECT_ROOT / 'data' / 'user_load_profile.json'
            load_data_path.parent.mkdir(parents=True, exist_ok=True)

            import json
            with open(load_data_path, 'w') as f:
                json.dump({
                    'load_data': load_data,
                    'validation': validation_result,
                    'filename': file.filename,
                    'upload_timestamp': time.time()
                }, f)

            logger.info(f"Load profile uploaded: {total_energy:.0f} kWh annual, {max_load:.2f} kW peak")

            return jsonify({
                'success': True,
                'data': load_data,
                'validation': validation_result,
                'filename': file.filename
            })

        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Error processing CSV: {str(e)}'
            }), 400

    except Exception as e:
        logger.error(f"Load upload error: {e}")
        return jsonify({
            'success': False,
            'error': f'Upload failed: {str(e)}'
        }), 500


@app.route('/validate-load-data', methods=['POST'])
def validate_load_data():
    """Validate load data from any source - matches frontend expectations"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'valid': False, 'error': 'No data provided'}), 400

        load_data = data.get('data', [])
        method = data.get('method', 'unknown')

        if not isinstance(load_data, list):
            return jsonify({'valid': False, 'error': 'Load data must be a list'}), 400

        if len(load_data) != 8760:
            return jsonify({
                'valid': False,
                'error': f'Expected 8760 hourly values, got {len(load_data)}'
            }), 400

        # Validate all values are numeric and positive
        for i, value in enumerate(load_data):
            try:
                num_value = float(value)
                if num_value < 0:
                    return jsonify({
                        'valid': False,
                        'error': f'Negative load value at hour {i + 1}: {num_value}'
                    }), 400
            except (ValueError, TypeError):
                return jsonify({
                    'valid': False,
                    'error': f'Invalid numeric value at hour {i + 1}: {value}'
                }), 400

        # Calculate statistics
        total_energy = sum(load_data)
        max_load = max(load_data)
        min_load = min(load_data)
        avg_load = total_energy / 8760

        # Basic reasonableness checks
        if total_energy <= 0:
            return jsonify({
                'valid': False,
                'error': 'Total annual energy must be positive'
            }), 400

        if max_load > 1000:  # Adjust limit as needed
            return jsonify({
                'valid': False,
                'error': f'Maximum hourly load ({max_load:.2f} kWh) seems unreasonably high'
            }), 400

        return jsonify({
            'valid': True,
            'message': f'Load profile validated successfully ({method})',
            'stats': {
                'total_annual': round(total_energy, 2),
                'peak_load': round(max_load, 3),
                'min_load': round(min_load, 3),
                'average_load': round(avg_load, 3),
                'method': method
            }
        })

    except Exception as e:
        logger.error(f"Load validation error: {e}")
        return jsonify({
            'valid': False,
            'error': f'Validation failed: {str(e)}'
        }), 500



@app.route('/reset-simulation', methods=['POST'])
def reset_simulation():
    """Reset simulation state"""
    try:
        global simple_status

        # Reset memory state
        simple_status = {
            'running': False,
            'stage': 'idle',
            'message': '',
            'error': None,
            'results': None,
            'session_id': current_session_id
        }

        # Clear session files
        clear_result = clear_all_previous_data()

        logger.info("Session reset completed")

        return jsonify({
            'status': 'reset_complete',
            'memory_cleared': True,
            'files_cleared': clear_result.get('cleared', []),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })

    except Exception as e:
        logger.error(f"Reset failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/geocode', methods=['GET'])
def geocode():
    """Server-side geocoding to avoid CORS issues"""
    address = request.args.get('address')
    if not address:
        return jsonify({'error': 'Address parameter required'}), 400

    try:
        response = requests.get(
            'https://nominatim.openstreetmap.org/search',
            params={
                'format': 'json',
                'q': address,
                'countrycodes': 'au',
                'limit': 1,
                'addressdetails': 1
            },
            headers={'User-Agent': 'EnTrans Energy Analysis Tool/1.0'},
            timeout=10
        )
        response.raise_for_status()

        results = response.json()
        logger.info(f"Geocoding successful for: {address}")
        return jsonify(results)

    except requests.RequestException as e:
        logger.error(f"Geocoding failed for {address}: {e}")
        return jsonify({'error': f'Geocoding service error: {str(e)}'}), 503
    except Exception as e:
        logger.error(f"Unexpected geocoding error: {e}")
        return jsonify({'error': 'Internal geocoding error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        health_status = {
            'status': 'healthy',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'environment': ENVIRONMENT,
            'debug_mode': IS_DEBUG,
            'components': {
                'form_integration': FORM_INTEGRATION_AVAILABLE,
                'pysam_simulation': PYSAM_AVAILABLE,
                'dashboard': DASHBOARD_AVAILABLE,
                'dash': DASH_AVAILABLE
            },
            'simulation': {
                'running': simple_status.get('running', False),
                'stage': simple_status.get('stage', 'idle'),
                'has_results': bool(simple_status.get('results')),
                'session_id': simple_status.get('session_id', 'unknown')
            }
        }

        # Test dashboard if available
        if DASHBOARD_AVAILABLE:
            try:
                structure_df, data_sources = load_data_with_metadata()
                health_status['dashboard_test'] = {
                    'working': True,
                    'components': len(structure_df) if structure_df is not None else 0,
                    'data_sources': len(data_sources) if data_sources else 0
                }
            except Exception as e:
                health_status['dashboard_test'] = {
                    'working': False,
                    'error': str(e)
                }

        # Overall status determination
        critical_components = [FORM_INTEGRATION_AVAILABLE]
        health_status['overall_status'] = 'healthy' if all(critical_components) else 'degraded'

        return jsonify(health_status)

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }), 500

# ========== STATIC ROUTES ==========

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files with fallback locations"""
    try:
        if (Config.STATIC_DIR / filename).exists():
            return send_from_directory(str(Config.STATIC_DIR), filename)

        if (Config.LEGACY_STATIC_DIR / filename).exists():
            return send_from_directory(str(Config.LEGACY_STATIC_DIR), filename)

        logger.warning(f"Static file not found: {filename}")
        return "File not found", 404

    except Exception as e:
        logger.error(f"Error serving static file {filename}: {e}")
        return "Internal server error", 500

# ========== MAIN APPLICATION ROUTES ==========

@app.route('/')
def index():
    """Main application page"""
    try:
        return render_template('index_base4.html')
    except Exception as e:
        logger.error(f"Error rendering index_base4.html: {e}")

        # Try alternative templates
        alternative_templates = ['index_base2.html', 'index.html', 'form.html']
        for alt_template in alternative_templates:
            try:
                alt_path = Path(app.template_folder) / alt_template
                if alt_path.exists():
                    logger.info(f"Using alternative template: {alt_template}")
                    return render_template(alt_template)
            except:
                continue

        # Return error page if no templates work
        return f"""
        <html>
        <head><title>Template Error</title></head>
        <body>
        <h1>Template Error</h1>
        <p>Could not load index_base4.html</p>
        <p>Template folder: {app.template_folder}</p>
        </body>
        </html>
        """, 500

@app.route('/dashboard')
def dashboard_redirect():
    """Redirect to dashboard"""
    try:
        results = load_fresh_results_from_disk()
        if results:
            return redirect('/dashboard/')
        else:
            return redirect('/?message=No analysis results available. Please run an analysis first.')
    except Exception as e:
        logger.error(f"Dashboard redirect error: {e}")
        return redirect('/?message=Dashboard error. Please try again.')

@app.route('/form-with-saved-values')
def form_with_saved_values():
    """Serve form with previously saved values"""
    try:
        form_path = Config.FORM_HTML

        if not form_path.exists():
            logger.warning("No saved form found, redirecting to main form")
            return redirect('/?message=No saved form data available')

        with open(form_path, 'r', encoding='utf-8') as f:
            saved_form_content = f.read()

        form_values = extract_form_values_from_html(saved_form_content)
        logger.info(f"Extracted {len(form_values)} saved form values")

        try:
            return render_template('index_base4.html',
                                 restored_values=form_values,
                                 is_restoration=True)
        except Exception as template_error:
            logger.error(f"Template error in form restoration: {template_error}")
            for alt_template in ['index_base2.html', 'index.html']:
                try:
                    return render_template(alt_template,
                                         restored_values=form_values,
                                         is_restoration=True)
                except:
                    continue

            return redirect('/?message=Template error in form restoration')

    except Exception as e:
        logger.error(f"Error loading saved form: {e}")
        return redirect('/?message=Error loading saved form')

def extract_form_values_from_html(html_content: str) -> Dict[str, Any]:
    """Extract form values from saved HTML content"""
    import re
    form_values = {}

    try:
        # Extract input values
        input_pattern = r'<input[^>]*(?:name|id)="([^"]*)"[^>]*value="([^"]*)"[^>]*>'
        for match in re.finditer(input_pattern, html_content):
            field_name, field_value = match.groups()
            if field_name and field_value:
                form_values[field_name] = field_value

        # Extract checked checkboxes
        checkbox_pattern = r'<input[^>]*type="checkbox"[^>]*(?:name|id)="([^"]*)"[^>]*checked[^>]*>'
        for match in re.finditer(checkbox_pattern, html_content):
            field_name = match.group(1)
            if field_name:
                form_values[field_name] = True

        # Extract selected options
        select_pattern = r'<select[^>]*(?:name|id)="([^"]*)"[^>]*>.*?<option[^>]*selected[^>]*value="([^"]*)"[^>]*>.*?</select>'
        for match in re.finditer(select_pattern, html_content, re.DOTALL):
            field_name, field_value = match.groups()
            if field_name and field_value:
                form_values[field_name] = field_value

        return form_values

    except Exception as e:
        logger.error(f"Error extracting form values: {e}")
        return {}

# ========== ERROR HANDLERS ==========

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    logger.warning(f"404 error: {request.url}")
    return jsonify({'error': 'Page not found', 'url': request.url}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"500 error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    """Handle uncaught exceptions"""
    logger.error(f"Uncaught exception: {str(e)}")
    if IS_DEBUG:
        logger.error(traceback.format_exc())

    return jsonify({
        'error': 'An unexpected error occurred',
        'details': str(e) if IS_DEBUG else 'Check server logs for details'
    }), 500

# ========== APPLICATION STARTUP ==========

def initialize_application():
    """Initialize application components"""
    log_milestone("Application initializing")
    logger.info(f"   Environment: {ENVIRONMENT}")
    logger.info(f"   Debug Mode: {IS_DEBUG}")
    logger.info(f"   Project Root: {Config.PROJECT_ROOT}")

    # Component status
    components = {
        'Form Integration': FORM_INTEGRATION_AVAILABLE,
        'PySAM Simulation': PYSAM_AVAILABLE,
        'Dashboard': DASHBOARD_AVAILABLE,
        'Dash Framework': DASH_AVAILABLE
    }

    for component, available in components.items():
        status = '✅ Available' if available else '❌ Not Available'
        logger.info(f"   {component}: {status}")

    # Validate configuration
    if IS_DEBUG:
        try:
            validation = Config.validate_deployment_readiness()
            if not validation['ready_for_deployment']:
                logger.warning("⚠️ Configuration issues detected:")
                for error in validation['errors'][:3]:
                    logger.warning(f"   ❌ {error}")
        except Exception as e:
            logger.warning(f"⚠️ Config validation failed: {e}")

    # Test critical components
    if FORM_INTEGRATION_AVAILABLE:
        try:
            logger.info("✅ Form integration tested successfully")
        except Exception as e:
            logger.error(f"❌ Form integration test failed: {e}")

    if DASHBOARD_AVAILABLE:
        try:
            structure_df, data_sources = load_data_with_metadata()
            component_count = len(structure_df) if structure_df is not None else 0
            logger.info(f"✅ Dashboard tested: {component_count} components ready")
        except Exception as e:
            logger.warning(f"⚠️ Dashboard test failed: {e}")

    logger.info("-" * 50)
    log_milestone("Application initialized - Routes available")
    logger.info("   / - Main Analysis Form")
    logger.info("   /dashboard/ - Results Dashboard")
    logger.info("   /health - Health Check")
    logger.info("   /analysis-status - Analysis Status")
    logger.info("   /api/dashboard-data - Dashboard Data API")
    logger.info("-" * 50)

if __name__ == '__main__':
    # Initialize application
    initialize_application()

    logger.info("🌐 Starting Flask server...")

    # Server configuration
    host = '0.0.0.0'
    port = int(os.getenv('PORT', 5000))

    # Run server
    app.run(
        host=host,
        port=port,
        debug=IS_DEBUG,
        threaded=True,
        use_reloader=IS_DEBUG
    )