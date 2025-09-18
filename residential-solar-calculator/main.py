from flask import Flask, render_template, redirect, request, session, flash, jsonify
import pandas as pd
import numpy as np
import secrets
import logging
import os
import json
import random
from pathlib import Path
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Import the calculation modules
from modules.profile_builder import ProfileBuilder
from modules.solar_calculator import SolarCalculator
from modules.postcode_lookup import PostcodeLookup
from modules.tariff_calculator import TariffCalculator
from modules.battery_optimizer import BatteryAnalyzer

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Configure template auto-reloading
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create temp directory for profile storage
temp_dir = '/tmp'
os.makedirs(temp_dir, exist_ok=True)

# Initialize calculation modules
profile_builder = ProfileBuilder()
solar_calculator = SolarCalculator()
postcode_lookup = PostcodeLookup()
tariff_calculator = TariffCalculator()
battery_analyzer = BatteryAnalyzer()

def cleanup_session_files(session_id):
    try:
        temp_dir = Path('/tmp')  # Changed from 'temp'
        if temp_dir.exists():
            for pattern in [f'profile_{session_id}.*', f'solar_{session_id}.*']:
                for file_path in temp_dir.glob(pattern):
                    if file_path.is_file():
                        file_path.unlink()
                        logger.info(f"Deleted session file: {file_path}")
    except Exception as e:
        logger.error(f"Error cleaning session files: {e}")


def cleanup_old_temp_files():
    """Clean up temp files older than 24 hours"""
    try:
        temp_dir = Path('temp')
        if not temp_dir.exists():
            return

        current_time = time.time()
        for file_path in temp_dir.glob('*'):
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > 86400:  # 24 hours in seconds
                    file_path.unlink()
                    logger.info(f"Deleted old temp file: {file_path}")
    except Exception as e:
        logger.error(f"Error cleaning temp files: {e}")


# Session cleanup middleware - REPLACE your existing @app.before_request
@app.before_request
def cleanup_old_sessions():
    """Clean up expired sessions and files"""
    if 'session_created' not in session:
        session['session_created'] = datetime.utcnow().isoformat()
    else:
        created = datetime.fromisoformat(session['session_created'])
        if datetime.utcnow() - created > timedelta(hours=4):
            # Clean up this session's files before clearing session
            session_id = session.get('session_id')
            if session_id:
                cleanup_session_files(session_id)
            session.clear()
            return redirect('/')

    # Periodically clean old temp files (10% chance per request)
    if random.random() < 0.1:
        cleanup_old_temp_files()

@app.route('/')
def home():
    """Landing page - redirect to step 1"""
    return redirect('/step1')


@app.route('/step1', methods=['GET', 'POST'])
def step1_profile():
    """Step 1: Build household energy profile"""

    if request.method == 'POST':
        try:
            logger.debug(f"Form data received: {dict(request.form)}")

            # Get and validate postcode
            postcode = request.form.get('postcode', '').strip()
            if not postcode_lookup.validate_postcode(postcode):
                flash('Please enter a valid 4-digit Australian postcode', 'error')
                return render_template('step1.html', form_data=dict(request.form))

            # Get coordinates for postcode
            coordinates = postcode_lookup.get_coordinates(postcode)
            if not coordinates:
                flash('Unable to find coordinates for this postcode', 'error')
                return render_template('step1.html', form_data=dict(request.form))

            # Get data source choice
            data_source = request.form.get('data_source', 'build')
            logger.debug(f"Data source: {data_source}")

            # Clear and initialize session
            session.clear()
            session['session_id'] = secrets.token_hex(8)

            # Replace the upload section in step1 POST handler with this:

            if data_source == 'upload':
                # Handle file upload with enhanced processing
                if 'energy_file' not in request.files or request.files['energy_file'].filename == '':
                    flash('Please upload an energy data file', 'error')
                    return render_template('step1.html', form_data=dict(request.form))

                recent_bill = request.form.get('recent_bill')
                try:
                    recent_bill = float(recent_bill) if recent_bill else None
                    if recent_bill and recent_bill <= 0:
                        flash('Recent bill amount must be greater than 0', 'error')
                        return render_template('step1.html', form_data=dict(request.form))
                except ValueError:
                    flash('Please enter a valid bill amount', 'error')
                    return render_template('step1.html', form_data=dict(request.form))

                try:
                    # Process uploaded file
                    file = request.files['energy_file']

                    # Read file based on extension
                    if file.filename.lower().endswith('.csv'):
                        df = pd.read_csv(file)
                    elif file.filename.lower().endswith(('.xlsx', '.xls')):
                        df = pd.read_excel(file)
                    else:
                        flash('Please upload a CSV or Excel file', 'error')
                        return render_template('step1.html', form_data=dict(request.form))

                    # Process with enhanced profile builder
                    result = profile_builder.process_uploaded_profile(df, recent_bill)

                    if not result['success']:
                        flash(f'Error processing file: {result["error"]}', 'error')
                        return render_template('step1.html', form_data=dict(request.form))

                    # Save processed profile to file
                    profile_file = f"/tmp/profile_{session['session_id']}.csv"
                    profile_df = pd.DataFrame({
                        'interval': range(len(result['profile'])),
                        'energy_use_kwh': result['profile']
                    })
                    profile_df.to_csv(profile_file, index=False)

                    # Store comprehensive session data
                    session['profile_data'] = {
                        'source': 'upload',
                        'annual_usage': result['summary']['annual_usage'],
                        'recent_bill': recent_bill,
                        'postcode': postcode,
                        'coordinates': coordinates,
                        'profile_file': profile_file,
                        'data_points': result['metadata']['original_data_points'],
                        'data_coverage': result['metadata']['data_coverage'],
                        'processing_method': 'full_year' if result['metadata']['data_coverage'][
                            'is_full_year'] else 'extended_partial'
                    }

                    # Provide user feedback about data processing
                    coverage = result['metadata']['data_coverage']
                    if coverage['is_full_year']:
                        flash(f'Successfully loaded full year of data ({coverage["data_days"]} days)', 'success')
                    else:
                        flash(
                            f'Extended {coverage["data_months"]:.1f} months of data to full year using seasonal patterns',
                            'info')

                except Exception as e:
                    logger.error(f"File processing error: {str(e)}")
                    flash(f'Error processing file: {str(e)}', 'error')
                    return render_template('step1.html', form_data=dict(request.form))

            else:
                # Build synthetic profile
                try:
                    occupants = int(request.form.get('occupants', 4))
                    building_type = request.form.get('building_type', 'detached')
                    has_aircon = 'has_aircon' in request.form
                    has_ev = 'has_ev' in request.form
                    has_electric_hw = 'has_electric_hw' in request.form
                    has_pool = 'has_pool' in request.form

                    logger.debug(f"Profile params: {occupants}, {building_type}")

                    # Generate the full profile using ProfileBuilder
                    energy_profile = profile_builder.generate_profile(
                        occupants=occupants,
                        building_type=building_type,
                        has_aircon=has_aircon,
                        has_ev=has_ev,
                        has_electric_hw=has_electric_hw,
                        has_pool=has_pool
                    )

                    # Save profile to file
                    profile_file = f"/tmp/profile_{session['session_id']}.csv"
                    profile_df = pd.DataFrame({
                        'interval': range(len(energy_profile)),
                        'energy_use_kwh': energy_profile
                    })
                    profile_df.to_csv(profile_file, index=False)

                    # Get summary
                    profile_summary = profile_builder.get_profile_summary(energy_profile)

                    # Store parameters and results
                    session['profile_data'] = {
                        'source': 'synthetic',
                        'annual_usage': profile_summary['annual_usage'],
                        'occupants': occupants,
                        'building_type': building_type,
                        'has_aircon': has_aircon,
                        'has_ev': has_ev,
                        'has_electric_hw': has_electric_hw,
                        'has_pool': has_pool,
                        'postcode': postcode,
                        'coordinates': coordinates,
                        'profile_file': profile_file
                    }

                    logger.debug(f"Session data stored: {dict(session)}")

                except Exception as e:
                    logger.error(f"Profile calculation error: {str(e)}")
                    flash(f'Error calculating profile: {str(e)}', 'error')
                    return render_template('step1.html', form_data=dict(request.form))

            flash('Profile created successfully', 'success')
            return redirect('/step2')

        except Exception as e:
            logger.error(f"Step1 error: {str(e)}", exc_info=True)
            flash('An unexpected error occurred', 'error')
            return render_template('step1.html', form_data=dict(request.form))

    # GET request - show form
    return render_template('step1.html', form_data={})


@app.route('/step2', methods=['GET', 'POST'])
def step2_tariffs():
    """Step 2: Configure electricity tariffs"""
    if 'profile_data' not in session:
        flash('Please complete Step 1 first', 'error')
        return redirect('/step1')

    if request.method == 'POST':
        try:
            # Get tariff configuration
            tariff_type = request.form.get('tariff_type', 'flat')
            daily_charge = float(request.form.get('daily_charge', 1.20))

            tariff_config = {
                'type': tariff_type,
                'daily_charge': daily_charge
            }

            if tariff_type == 'flat':
                tariff_config['rate'] = float(request.form.get('flat_rate', 28.5))
            else:  # TOU
                tariff_config.update({
                    'peak_rate': float(request.form.get('peak_rate', 45.0)),
                    'shoulder_rate': float(request.form.get('shoulder_rate', 0)) or float(
                        request.form.get('offpeak_rate', 18.0)),
                    'offpeak_rate': float(request.form.get('offpeak_rate', 18.0))
                })

            # Calculate current bill using TariffCalculator
            annual_usage = session['profile_data']['annual_usage']
            current_bill = tariff_calculator.estimate_bill_from_annual_usage(
                annual_usage, tariff_config, daily_charge
            )

            session['tariff_data'] = {
                'config': tariff_config,
                'current_bill': current_bill
            }

            flash('Tariff configuration saved', 'success')
            return redirect('/step3')

        except Exception as e:
            logger.error(f"Step2 error: {str(e)}")
            flash(f'Error: {str(e)}', 'error')
            return redirect('/step2')

    return render_template('step2.html',
                           profile_data=session['profile_data'],
                           form_data=request.form)


# Updated step3 route in main.py
@app.route('/step3', methods=['GET', 'POST'])
def step3_solar():
    """Step 3: Solar system configuration"""
    if not all(key in session for key in ['profile_data', 'tariff_data']):
        flash('Please complete previous steps first', 'error')
        return redirect('/step1')

    if request.method == 'POST':
        # POST logic remains the same - this is where the main API call happens
        try:
            logger.debug(f"Step3 form data: {dict(request.form)}")

            solar_status = request.form.get('solar_status', 'none')
            system_size = float(request.form.get('system_size', 0))
            installation_cost = float(request.form.get('installation_cost', 0)) if solar_status == 'planning' else 0
            feed_in_tariff = float(request.form.get('feed_in_tariff', 8.0))

            if system_size > 0:
                lat, lon = session['profile_data']['coordinates']
                generation_data = solar_calculator.get_annual_generation(lat, lon, system_size)

                solar_file = f"/tmp/solar_{session['session_id']}.json"
                with open(solar_file, 'w') as f:
                    json.dump(generation_data, f)

                logger.debug(
                    f"Saved solar data to file: {solar_file}, generation: {generation_data.get('annual_kwh', 0):.0f} kWh")

                generation_summary = {
                    'annual_kwh': generation_data.get('annual_kwh', 0),
                    'kwh_per_kw': generation_data.get('kwh_per_kw', 0),
                    'capacity_factor': generation_data.get('capacity_factor', 0)
                }
            else:
                generation_summary = {'annual_kwh': 0, 'kwh_per_kw': 0, 'capacity_factor': 0}
                solar_file = None

            session['solar_data'] = {
                'solar_status': solar_status,
                'system_size': system_size,
                'installation_cost': installation_cost,
                'feed_in_tariff': feed_in_tariff,
                'generation_data': generation_summary,
                'solar_file': solar_file
            }

            logger.debug(f"Session solar_data size: {len(str(session['solar_data']))} characters")

            flash('Solar configuration saved', 'success')
            return redirect('/step4')

        except Exception as e:
            logger.error(f"Step3 POST error: {str(e)}")
            flash(f'Error: {str(e)}', 'error')
            return redirect('/step3')

    # GET request - now uses real NREL data for recommendation
    annual_usage = session['profile_data']['annual_usage']
    lat, lon = session['profile_data']['coordinates']

    # Get recommendation using NREL API (one API call for 1kW test system)
    recommendation = solar_calculator.calculate_system_recommendation(annual_usage, lat, lon)

    logger.debug(f"Step3 recommendation: {recommendation['recommended_size_kw']:.1f}kW, "
                 f"{recommendation['estimated_annual_generation']:.0f} kWh, "
                 f"yield: {recommendation['kwh_per_kw']:.0f} kWh/kW")

    return render_template('step3.html',
                           profile_data=session['profile_data'],
                           tariff_data=session['tariff_data'],
                           recommendation=recommendation,
                           coordinates={'lat': lat, 'lon': lon},
                           form_data={})


@app.route('/step4', methods=['GET', 'POST'])
def step4_battery():
    """Step 4: Battery system and dispatch scenarios"""
    if not all(key in session for key in ['profile_data', 'tariff_data']):
        flash('Please complete previous steps first', 'error')
        return redirect('/step1')

    if request.method == 'POST':
        try:
            # Get battery configuration from form
            include_battery = request.form.get('include_battery') == 'yes'
            battery_size = float(request.form.get('battery_size', 0)) if include_battery else 0
            battery_cost = float(request.form.get('battery_cost', 0)) if include_battery else 0
            battery_life = int(request.form.get('battery_life', 10))

            # Load the energy profile for detailed analysis
            profile_file = session['profile_data']['profile_file']

            if session['profile_data']['source'] == 'upload':
                # Load uploaded profile
                df = pd.read_csv(profile_file)
                load_profile = df['energy_use_kwh'].tolist()
            else:
                # Load generated synthetic profile
                df = pd.read_csv(profile_file)
                load_profile = df['energy_use_kwh'].tolist()

            # Initialize solar_profile at function level to ensure scope
            solar_profile = [0] * len(load_profile)  # Default fallback

            # Generate solar profile if we have solar
            solar_data = session.get('solar_data', {'system_size': 0})

            if solar_data['system_size'] > 0 and solar_data.get('solar_file'):
                # Load full generation data from file
                try:
                    with open(solar_data['solar_file'], 'r') as f:
                        generation_data = json.load(f)
                    solar_profile = solar_calculator.generate_half_hourly_profile(generation_data)
                    logger.debug(f"Loaded solar profile from file: {len(solar_profile)} intervals")
                except Exception as e:
                    logger.error(f"Error loading solar file: {str(e)}")
                    # Fallback to simple solar profile
                    annual_solar = solar_data['generation_data'].get('annual_kwh', 0)
                    solar_profile = solar_calculator.generate_half_hourly_profile({'annual_kwh': annual_solar})

            # Ensure profiles are same length
            min_length = min(len(load_profile), len(solar_profile))
            load_profile = load_profile[:min_length]
            solar_profile = solar_profile[:min_length]

            logger.debug(f"Battery config: include={include_battery}, size={battery_size}")
            logger.debug(f"Profile lengths: load={len(load_profile)}, solar={len(solar_profile)}")
            logger.debug(f"Sample load values: {load_profile[:10]}")
            logger.debug(f"Sample solar values: {solar_profile[:10]}")
            logger.debug(f"Total load: {sum(load_profile):.1f} kWh, Total solar: {sum(solar_profile):.1f} kWh")
            logger.debug(f"Tariff config: {session['tariff_data']['config']}")
            logger.debug(f"Running battery analysis...")

            # Run battery analysis scenarios
            if include_battery and battery_size > 0:
                scenarios = battery_analyzer.analyze_scenarios(
                    load_profile=load_profile,
                    solar_profile=solar_profile,
                    battery_size_kwh=battery_size,
                    tariff_config=session['tariff_data']['config'],
                    feed_in_tariff=solar_data.get('feed_in_tariff', 8.0)
                )

                logger.debug(f"Scenarios generated: {list(scenarios.keys())}")
                for name, scenario in scenarios.items():
                    logger.debug(f"{name}: cost=${scenario.get('annual_cost', 'N/A'):.2f}")

                # Calculate economics
                economics = battery_analyzer.calculate_battery_economics(
                    scenarios, battery_cost, battery_life
                )
            else:
                # Solar only scenario
                scenarios = battery_analyzer.analyze_scenarios(
                    load_profile=load_profile,
                    solar_profile=solar_profile,
                    battery_size_kwh=0,  # No battery
                    tariff_config=session['tariff_data']['config'],
                    feed_in_tariff=solar_data.get('feed_in_tariff', 8.0)
                )
                economics = {}

            # Store battery results
            session['battery_data'] = {
                'include_battery': include_battery,
                'battery_size': battery_size,
                'battery_cost': battery_cost,
                'battery_life': battery_life,
                'scenarios': scenarios,
                'economics': economics
            }

            flash('Battery analysis complete', 'success')
            return redirect('/results')

        except Exception as e:
            logger.error(f"Step4 POST error: {str(e)}")
            flash(f'Error: {str(e)}', 'error')
            return redirect('/step4')

    # GET request - show battery recommendations
    annual_usage = session['profile_data']['annual_usage']

    # Try to get a better battery recommendation if we have profiles
    try:
        profile_file = session['profile_data']['profile_file']

        if session['profile_data']['source'] == 'upload':
            df = pd.read_csv(profile_file)
            load_profile = df['energy_use_kwh'].tolist()
        else:
            df = pd.read_csv(profile_file)
            load_profile = df['energy_use_kwh'].tolist()

        # Generate solar profile for recommendation
        solar_data = session.get('solar_data', {'system_size': 0})
        if solar_data['system_size'] > 0:
            generation_data = solar_data['generation_data']
            solar_profile = solar_calculator.generate_half_hourly_profile(generation_data)
        else:
            solar_profile = [0] * len(load_profile)

        # Get detailed battery recommendation
        recommendation = battery_analyzer.recommend_battery_size(load_profile, solar_profile)

    except Exception as e:
        logger.warning(f"Could not generate detailed battery recommendation: {str(e)}")
        # Fallback to simple recommendation
        recommended_battery = max(0, min(20, annual_usage / 365 * 0.3))  # ~30% of daily usage
        recommendation = {
            'recommended_size_kwh': round(recommended_battery),
            'avg_evening_load': annual_usage / 365 * 0.2,
            'avg_excess_solar': annual_usage / 365 * 0.4,
            'rationale': f"Based on your {annual_usage:.0f} kWh annual usage"
        }

    return render_template('step4.html',
                           profile_data=session['profile_data'],
                           solar_data=session.get('solar_data', {
                               'system_size': 0,
                               'generation_data': {'annual_kwh': 0}
                           }),
                           recommendation=recommendation,
                           form_data={})


@app.route('/results')
def results():
    """Results summary page"""
    if 'profile_data' not in session:
        flash('Please complete all steps first', 'error')
        return redirect('/step1')

    # Compile results with fallback values
    profile_data = session['profile_data']
    tariff_data = session.get('tariff_data', {})
    solar_data = session.get('solar_data', {
        'system_size': 0,
        'generation_data': {'annual_kwh': 0},
        'installation_cost': 0,
        'solar_status': 'none'
    })
    battery_data = session.get('battery_data', {
        'include_battery': False,
        'scenarios': {'solar_only': {'annual_cost': tariff_data.get('current_bill', {}).get('total_cost', 2000),
                                     'name': 'Solar Only'}},
        'economics': {}
    })

    results_data = {
        'profile': profile_data,
        'tariff': tariff_data,
        'solar': solar_data,
        'battery': battery_data
    }

    return render_template('results.html', results=results_data)


# Helper route for AJAX requests (optional)
@app.route('/api/estimate_bill')
def api_estimate_bill():
    """AJAX endpoint for real-time bill estimates"""
    try:
        tariff_type = request.args.get('type', 'flat')
        daily_charge = float(request.args.get('daily_charge', 1.20))
        annual_usage = session.get('profile_data', {}).get('annual_usage', 6100)

        if tariff_type == 'flat':
            rate = float(request.args.get('rate', 28.5))
            tariff_config = {'type': 'flat', 'rate': rate}
        else:
            peak_rate = float(request.args.get('peak_rate', 45.0))
            shoulder_rate = float(request.args.get('shoulder_rate', 25.0))
            offpeak_rate = float(request.args.get('offpeak_rate', 18.0))
            tariff_config = {
                'type': 'tou',
                'peak_rate': peak_rate,
                'shoulder_rate': shoulder_rate,
                'offpeak_rate': offpeak_rate
            }

        bill = tariff_calculator.estimate_bill_from_annual_usage(
            annual_usage, tariff_config, daily_charge
        )

        return jsonify({
            'success': True,
            'total_cost': round(bill['total_cost'], 2),
            'usage_cost': round(bill['usage_cost'], 2),
            'supply_cost': round(bill['supply_cost'], 2)
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)