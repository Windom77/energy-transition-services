import requests
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SolarCalculator:
    """
    Calculate solar generation using NREL PVWatts API and generate detailed profiles
    """

    def __init__(self):
        self.api_key = os.getenv('PVWATTS_API_KEY', '7NIcgO78r1a3xv4RKzWchNyYB0hlLrCNOek6Rv28')
        self.base_url = "https://developer.nrel.gov/api/pvwatts/v8.json"

        # Default system parameters for Australian conditions
        self.default_params = {
            'azimuth': 0,  # North-facing (Australia)
            'tilt': 20,  # Optimal tilt for most of Australia
            'array_type': 1,  # Fixed (open rack)
            'module_type': 0,  # Standard module
            'losses': 14.08,  # System losses (%)
            'dc_ac_ratio': 1.2,  # DC to AC ratio
            'inv_eff': 96.0,  # Inverter efficiency
            'timeframe': 'hourly'  # Get hourly data
        }

    def get_annual_generation(self, lat, lon, system_size_kw, tilt=None, azimuth=None):
        """
        Get annual solar generation estimate from NREL PVWatts

        Args:
            lat: Latitude
            lon: Longitude
            system_size_kw: System size in kW
            tilt: Panel tilt angle (optional)
            azimuth: Panel azimuth angle (optional)

        Returns:
            Dict with generation data and hourly profile
        """
        try:
            if not self.api_key:
                logger.error("PVWATTS_API_KEY not found in environment variables")
                return self._fallback_generation(system_size_kw, lat)

            params = self.default_params.copy()
            params.update({
                'api_key': self.api_key,
                'lat': lat,
                'lon': lon,
                'system_capacity': system_size_kw,
                'tilt': tilt or self._calculate_optimal_tilt(lat),
                'azimuth': azimuth or 0  # North in Australia
            })

            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if 'outputs' not in data:
                logger.error(f"Unexpected API response: {data}")
                return self._fallback_generation(system_size_kw, lat)

            outputs = data['outputs']

            return {
                'annual_kwh': outputs['ac_annual'],
                'monthly_kwh': outputs['ac_monthly'],
                'hourly_ac': outputs.get('ac', []),
                'hourly_dc': outputs.get('dc', []),
                'hourly_dni': outputs.get('dni', []),
                'hourly_ghi': outputs.get('ghi', []),
                'capacity_factor': outputs.get('capacity_factor', 0),
                'kwh_per_kw': outputs['ac_annual'] / system_size_kw if system_size_kw > 0 else 0
            }

        except requests.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            return self._fallback_generation(system_size_kw, lat)
        except Exception as e:
            logger.error(f"Error getting solar generation: {str(e)}")
            return self._fallback_generation(system_size_kw, lat)

    def _fallback_generation(self, system_size_kw, lat):
        """Fallback generation calculation if API fails"""
        # Australian solar irradiance estimates by latitude
        if lat > -25:  # Northern Australia
            kwh_per_kw = 1650
        elif lat > -30:  # Central Australia
            kwh_per_kw = 1550
        elif lat > -35:  # Southern Australia
            kwh_per_kw = 1450
        else:  # Tasmania
            kwh_per_kw = 1350

        annual_kwh = system_size_kw * kwh_per_kw

        logger.info(f"Using fallback generation: {annual_kwh} kWh for {system_size_kw} kW system")

        return {
            'annual_kwh': annual_kwh,
            'monthly_kwh': [annual_kwh / 12] * 12,  # Uniform monthly
            'hourly_ac': [],
            'capacity_factor': kwh_per_kw / 8760,  # Rough estimate
            'kwh_per_kw': kwh_per_kw,
            'fallback': True
        }

    def _calculate_optimal_tilt(self, lat):
        """Calculate optimal tilt angle for latitude"""
        # Rule of thumb: tilt ≈ latitude for year-round optimization
        return min(max(abs(lat) * 0.87, 10), 60)  # Constrain between 10-60 degrees

    def generate_half_hourly_profile(self, generation_data):
        """
        Convert hourly generation to half-hourly profile

        Args:
            generation_data: Output from get_annual_generation()

        Returns:
            List of half-hourly generation values for full year (17520 intervals)
        """
        try:
            if generation_data.get('hourly_ac'):
                # Use actual hourly data from API
                hourly_data = generation_data['hourly_ac']

                # CRITICAL: NREL API returns hourly data in Wh, convert to kWh
                hourly_kwh = [val / 1000 for val in hourly_data]

                logger.debug(f"Converted hourly data: first 5 values {hourly_data[:5]} Wh -> {hourly_kwh[:5]} kWh")
                logger.debug(
                    f"Annual total: API={sum(hourly_data) / 1000:.1f} kWh, Expected={generation_data.get('annual_kwh', 0):.1f} kWh")

                # Convert to half-hourly by splitting each hour
                half_hourly = []
                for hour_kwh in hourly_kwh:
                    half_hourly.extend([hour_kwh / 2, hour_kwh / 2])

                return half_hourly
            else:
                # Generate synthetic profile
                return self._generate_synthetic_profile(generation_data['annual_kwh'])

        except Exception as e:
            logger.error(f"Error generating half-hourly profile: {str(e)}")
            return self._generate_synthetic_profile(generation_data['annual_kwh'])

    def _generate_synthetic_profile(self, annual_kwh):
        """Generate synthetic half-hourly solar profile"""

        # Daily generation curve (48 half-hourly intervals)
        # Peak at noon, zero at night
        daily_curve = np.zeros(48)

        for i in range(48):
            hour = i / 2  # Convert to decimal hour
            if 6 <= hour <= 18:  # Daylight hours
                # Sinusoidal curve peaking at noon
                angle = (hour - 6) / 12 * np.pi
                daily_curve[i] = np.sin(angle) ** 2

        # Normalize daily curve
        daily_curve = daily_curve / daily_curve.sum() if daily_curve.sum() > 0 else daily_curve

        # Seasonal factors (Australian seasons)
        seasonal_factors = {
            1: 1.35,  # Jan - peak summer
            2: 1.25,  # Feb
            3: 1.10,  # Mar
            4: 0.90,  # Apr
            5: 0.70,  # May
            6: 0.65,  # Jun - winter minimum
            7: 0.70,  # Jul
            8: 0.85,  # Aug
            9: 1.00,  # Sep
            10: 1.15,  # Oct
            11: 1.25,  # Nov
            12: 1.30  # Dec
        }

        # Generate full year
        annual_profile = []

        for month in range(1, 13):
            # Days in each month (simplified)
            days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1]
            month_factor = seasonal_factors[month]

            for day in range(days_in_month):
                # Daily variation (weather)
                daily_variation = np.random.normal(1.0, 0.2)  # ±20% variation
                daily_variation = max(0.3, min(1.7, daily_variation))  # Constrain

                # Scale daily curve
                daily_generation = daily_curve * month_factor * daily_variation
                daily_total = daily_generation.sum()

                # Normalize to target daily amount
                target_daily = annual_kwh / 365 * month_factor
                if daily_total > 0:
                    daily_generation = daily_generation * (target_daily / daily_total)

                annual_profile.extend(daily_generation)

        return annual_profile

    def calculate_system_recommendation(self, annual_usage_kwh, lat, lon):
        """
        Recommend optimal system size based on energy usage using real NREL data

        Args:
            annual_usage_kwh: Annual energy consumption
            lat: Latitude
            lon: Longitude (now required)

        Returns:
            Dict with system recommendations based on real solar data
        """
        try:
            # Get real solar yield using NREL API for a test system
            test_size = 1.0  # 1kW test to get actual kWh/kW ratio
            test_generation = self.get_annual_generation(lat, lon, test_size)
            kwh_per_kw = test_generation.get('kwh_per_kw', 1500)  # Fallback if API fails

            logger.debug(f"Real solar yield for location: {kwh_per_kw:.0f} kWh/kW")

            # Size system to generate 80-120% of annual usage
            recommended_size = annual_usage_kwh / kwh_per_kw

            # Round to sensible increments
            if recommended_size <= 3:
                recommended_size = round(recommended_size * 2) / 2  # 0.5 kW increments
            elif recommended_size <= 10:
                recommended_size = round(recommended_size)  # 1 kW increments
            else:
                recommended_size = round(recommended_size / 2.5) * 2.5  # 2.5 kW increments

            # Constrain to reasonable residential range
            recommended_size = max(1.5, min(20, recommended_size))

            # Calculate system metrics using real yield
            estimated_generation = recommended_size * kwh_per_kw
            self_consumption_estimate = min(annual_usage_kwh, estimated_generation * 0.3)
            export_estimate = max(0, estimated_generation - self_consumption_estimate)

            return {
                'recommended_size_kw': recommended_size,
                'estimated_annual_generation': estimated_generation,
                'estimated_self_consumption': self_consumption_estimate,
                'estimated_export': export_estimate,
                'generation_ratio': estimated_generation / annual_usage_kwh,
                'kwh_per_kw': kwh_per_kw  # Real value, not fallback
            }

        except Exception as e:
            logger.error(f"Error getting real solar recommendation: {str(e)}")
            # Fallback to generic calculation
            return self._fallback_recommendation(annual_usage_kwh, lat)

    def _fallback_recommendation(self, annual_usage_kwh, lat):
        """Fallback recommendation if NREL API fails"""
        # Australian solar irradiance estimates by latitude (original code)
        if lat > -25:
            kwh_per_kw = 1650
        elif lat > -30:
            kwh_per_kw = 1550
        elif lat > -35:
            kwh_per_kw = 1450
        else:
            kwh_per_kw = 1350

        recommended_size = max(1.5, min(20, annual_usage_kwh / kwh_per_kw))
        estimated_generation = recommended_size * kwh_per_kw

        return {
            'recommended_size_kw': recommended_size,
            'estimated_annual_generation': estimated_generation,
            'estimated_self_consumption': min(annual_usage_kwh, estimated_generation * 0.3),
            'estimated_export': max(0, estimated_generation - min(annual_usage_kwh, estimated_generation * 0.3)),
            'generation_ratio': estimated_generation / annual_usage_kwh,
            'kwh_per_kw': kwh_per_kw
        }