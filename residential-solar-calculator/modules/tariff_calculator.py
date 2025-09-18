import pandas as pd
import numpy as np
import logging
from datetime import datetime, time

logger = logging.getLogger(__name__)


class TariffCalculator:
    """
    Calculate electricity bills based on different tariff structures
    """

    def __init__(self):
        # Standard TOU periods for most Australian retailers
        self.tou_periods = {
            'peak': {
                'weekday': [(14, 20)],  # 2PM-8PM weekdays
                'weekend': []  # No weekend peak for most retailers
            },
            'shoulder': {
                'weekday': [(7, 14), (20, 22)],  # 7AM-2PM, 8PM-10PM weekdays
                'weekend': [(7, 22)]  # 7AM-10PM weekends
            },
            'offpeak': {
                'weekday': [(22, 24), (0, 7)],  # 10PM-7AM weekdays
                'weekend': [(22, 24), (0, 7)]  # 10PM-7AM weekends
            }
        }

    def calculate_bill(self, energy_profile, tariff_config, daily_charge=1.20):
        """
        Calculate electricity bill for given energy profile and tariff

        Args:
            energy_profile: List of half-hourly energy usage (kWh)
            tariff_config: Dict with tariff structure
            daily_charge: Daily supply charge in dollars

        Returns:
            Dict with bill breakdown
        """
        try:
            if tariff_config['type'] == 'flat':
                return self._calculate_flat_bill(energy_profile, tariff_config, daily_charge)
            elif tariff_config['type'] == 'tou':
                return self._calculate_tou_bill(energy_profile, tariff_config, daily_charge)
            else:
                raise ValueError(f"Unsupported tariff type: {tariff_config['type']}")

        except Exception as e:
            logger.error(f"Error calculating bill: {str(e)}")
            raise

    def _calculate_flat_bill(self, energy_profile, tariff_config, daily_charge):
        """Calculate bill with flat rate tariff"""
        total_usage = sum(energy_profile)
        rate = tariff_config['rate']  # cents/kWh

        usage_cost = (total_usage * rate) / 100  # Convert cents to dollars
        supply_cost = daily_charge * 365  # Annual supply charges
        total_cost = usage_cost + supply_cost

        return {
            'total_cost': total_cost,
            'usage_cost': usage_cost,
            'supply_cost': supply_cost,
            'total_usage': total_usage,
            'average_rate': rate,
            'breakdown': {
                'usage_kwh': total_usage,
                'rate_cents_kwh': rate,
                'usage_cost': usage_cost,
                'daily_charge': daily_charge,
                'supply_cost': supply_cost
            }
        }

    def _calculate_tou_bill(self, energy_profile, tariff_config, daily_charge):
        """Calculate bill with Time of Use tariff"""
        # Initialize counters
        peak_usage = 0
        shoulder_usage = 0
        offpeak_usage = 0

        # Process each half-hour interval
        intervals_per_day = 48
        days = len(energy_profile) // intervals_per_day

        for day in range(days):
            # Determine if weekend (simplified - every 7th day)
            is_weekend = (day % 7) >= 5

            for interval in range(intervals_per_day):
                idx = day * intervals_per_day + interval
                if idx >= len(energy_profile):
                    break

                usage = energy_profile[idx]
                hour = interval // 2  # Convert interval to hour

                # Determine TOU period
                period = self._get_tou_period(hour, is_weekend)

                if period == 'peak':
                    peak_usage += usage
                elif period == 'shoulder':
                    shoulder_usage += usage
                else:
                    offpeak_usage += usage

        # Calculate costs
        peak_cost = (peak_usage * tariff_config['peak_rate']) / 100
        shoulder_cost = (shoulder_usage * tariff_config.get('shoulder_rate', tariff_config['offpeak_rate'])) / 100
        offpeak_cost = (offpeak_usage * tariff_config['offpeak_rate']) / 100

        usage_cost = peak_cost + shoulder_cost + offpeak_cost
        supply_cost = daily_charge * 365
        total_cost = usage_cost + supply_cost

        total_usage = peak_usage + shoulder_usage + offpeak_usage
        weighted_avg_rate = (usage_cost / total_usage * 100) if total_usage > 0 else 0

        return {
            'total_cost': total_cost,
            'usage_cost': usage_cost,
            'supply_cost': supply_cost,
            'total_usage': total_usage,
            'average_rate': weighted_avg_rate,
            'breakdown': {
                'peak_usage': peak_usage,
                'peak_rate': tariff_config['peak_rate'],
                'peak_cost': peak_cost,
                'shoulder_usage': shoulder_usage,
                'shoulder_rate': tariff_config.get('shoulder_rate', tariff_config['offpeak_rate']),
                'shoulder_cost': shoulder_cost,
                'offpeak_usage': offpeak_usage,
                'offpeak_rate': tariff_config['offpeak_rate'],
                'offpeak_cost': offpeak_cost,
                'daily_charge': daily_charge,
                'supply_cost': supply_cost
            }
        }

    def _get_tou_period(self, hour, is_weekend):
        """Determine which TOU period an hour falls into"""
        day_type = 'weekend' if is_weekend else 'weekday'

        # Check each period
        for period, times in self.tou_periods.items():
            for start_hour, end_hour in times[day_type]:
                if start_hour <= hour < end_hour:
                    return period
                # Handle overnight periods (e.g., 22-24 and 0-7)
                elif start_hour > end_hour:  # Crosses midnight
                    if hour >= start_hour or hour < end_hour:
                        return period

        # Default to offpeak if no match found
        return 'offpeak'

    def estimate_bill_from_annual_usage(self, annual_kwh, tariff_config, daily_charge=1.20):
        """
        Quick bill estimate from just annual usage (assumes typical load profile)

        Args:
            annual_kwh: Annual energy usage in kWh
            tariff_config: Tariff structure
            daily_charge: Daily supply charge

        Returns:
            Dict with estimated bill
        """
        try:
            if tariff_config['type'] == 'flat':
                usage_cost = (annual_kwh * tariff_config['rate']) / 100
                supply_cost = daily_charge * 365
                return {
                    'total_cost': usage_cost + supply_cost,
                    'usage_cost': usage_cost,
                    'supply_cost': supply_cost
                }

            else:  # TOU - use typical residential distribution
                # Typical Australian residential TOU distribution
                peak_percentage = 0.25  # 25% in peak
                shoulder_percentage = 0.45  # 45% in shoulder
                offpeak_percentage = 0.30  # 30% in offpeak

                peak_usage = annual_kwh * peak_percentage
                shoulder_usage = annual_kwh * shoulder_percentage
                offpeak_usage = annual_kwh * offpeak_percentage

                peak_cost = (peak_usage * tariff_config['peak_rate']) / 100
                shoulder_cost = (shoulder_usage * tariff_config.get('shoulder_rate',
                                                                    tariff_config['offpeak_rate'])) / 100
                offpeak_cost = (offpeak_usage * tariff_config['offpeak_rate']) / 100

                usage_cost = peak_cost + shoulder_cost + offpeak_cost
                supply_cost = daily_charge * 365

                return {
                    'total_cost': usage_cost + supply_cost,
                    'usage_cost': usage_cost,
                    'supply_cost': supply_cost,
                    'breakdown': {
                        'peak_usage': peak_usage,
                        'peak_cost': peak_cost,
                        'shoulder_usage': shoulder_usage,
                        'shoulder_cost': shoulder_cost,
                        'offpeak_usage': offpeak_usage,
                        'offpeak_cost': offpeak_cost
                    }
                }

        except Exception as e:
            logger.error(f"Error estimating bill: {str(e)}")
            raise