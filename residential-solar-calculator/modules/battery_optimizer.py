import numpy as np
import pandas as pd
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class BatteryAnalyzer:
    """
    Simulate battery storage with three different dispatch strategies
    """

    def __init__(self):
        self.battery_efficiency = 0.90  # Round-trip efficiency
        self.max_charge_rate = 0.5  # Max charge rate as fraction of capacity per hour
        self.max_discharge_rate = 0.5  # Max discharge rate as fraction of capacity per hour

    def analyze_scenarios(self, load_profile, solar_profile, battery_size_kwh,
                          tariff_config, feed_in_tariff):
        """
        Analyze three battery dispatch scenarios

        Args:
            load_profile: List of half-hourly load data (kWh)
            solar_profile: List of half-hourly solar generation (kWh)
            battery_size_kwh: Battery capacity in kWh
            tariff_config: Tariff structure
            feed_in_tariff: Feed-in rate (cents/kWh)

        Returns:
            Dict with results for each scenario
        """
        try:
            scenarios = {}

            # Scenario 1: Solar Self-Consumption Only
            scenarios['solar_only'] = self._simulate_solar_only(
                load_profile, solar_profile, tariff_config, feed_in_tariff
            )

            # Scenario 2: Time of Use Arbitrage
            scenarios['tou_arbitrage'] = self._simulate_tou_arbitrage(
                load_profile, solar_profile, battery_size_kwh, tariff_config, feed_in_tariff
            )

            # Scenario 3: Smart Solar + TOU Hybrid
            scenarios['smart_hybrid'] = self._simulate_smart_hybrid(
                load_profile, solar_profile, battery_size_kwh, tariff_config, feed_in_tariff
            )

            return scenarios

        except Exception as e:
            logger.error(f"Error analyzing battery scenarios: {str(e)}")
            raise

    def _simulate_solar_only(self, load_profile, solar_profile, tariff_config, feed_in_tariff):
        """Scenario 1: Solar self-consumption only (no battery)"""
        try:
            intervals = min(len(load_profile), len(solar_profile))

            grid_import = []
            solar_export = []
            solar_consumption = []

            for i in range(intervals):
                load = load_profile[i]
                solar = solar_profile[i]

                # Direct solar consumption
                self_consumption = min(load, solar)
                solar_consumption.append(self_consumption)

                # Remaining load from grid
                grid_import.append(max(0, load - solar))

                # Excess solar to grid
                solar_export.append(max(0, solar - load))

            # Calculate costs
            costs = self._calculate_costs(grid_import, solar_export,
                                          tariff_config, feed_in_tariff, intervals)

            return {
                'name': 'Solar Only (No Battery)',
                'description': 'Direct solar consumption with grid export of excess',
                'annual_cost': costs['total_cost'],
                'grid_import_kwh': sum(grid_import),
                'solar_export_kwh': sum(solar_export),
                'solar_consumption_kwh': sum(solar_consumption),
                'self_sufficiency': (sum(solar_consumption) / sum(load_profile[:intervals])) * 100,
                'battery_cycles': 0,
                'costs': costs
            }

        except Exception as e:
            logger.error(f"Error in solar-only simulation: {str(e)}")
            raise

    def _simulate_tou_arbitrage(self, load_profile, solar_profile, battery_size,
                                tariff_config, feed_in_tariff):
        """Scenario 2: TOU arbitrage - charge off-peak, discharge peak"""
        try:
            intervals = min(len(load_profile), len(solar_profile))
            battery_soc = battery_size * 0.2  # Start at 20% charge

            grid_import = []
            solar_export = []
            battery_charge = []
            battery_discharge = []

            max_charge_per_interval = battery_size * self.max_charge_rate / 2  # Half-hourly
            max_discharge_per_interval = battery_size * self.max_discharge_rate / 2

            for i in range(intervals):
                load = load_profile[i]
                solar = solar_profile[i]

                # Determine TOU period
                hour = (i % 48) // 2  # Convert interval to hour of day
                is_weekend = ((i // 48) % 7) >= 5
                tou_period = self._get_tou_period(hour, is_weekend)

                # Battery dispatch logic for TOU arbitrage
                charge_amount = 0
                discharge_amount = 0

                if tou_period == 'offpeak':
                    # Charge battery during off-peak (grid only, not solar)
                    available_capacity = battery_size - battery_soc
                    max_charge = min(max_charge_per_interval, available_capacity)
                    charge_amount = max_charge

                elif tou_period == 'peak':
                    # Discharge battery during peak
                    available_energy = battery_soc
                    max_discharge = min(max_discharge_per_interval, available_energy, load)
                    discharge_amount = max_discharge

                # Apply battery actions
                battery_soc += charge_amount * self.battery_efficiency
                battery_soc -= discharge_amount
                battery_soc = max(0, min(battery_size, battery_soc))

                # Calculate flows
                net_load = load - solar - discharge_amount

                if net_load > 0:
                    grid_import.append(net_load + charge_amount)  # Include battery charging
                    solar_export.append(0)
                else:
                    grid_import.append(charge_amount)  # Only battery charging if any
                    solar_export.append(-net_load)  # Excess solar

                battery_charge.append(charge_amount)
                battery_discharge.append(discharge_amount)

            costs = self._calculate_costs(grid_import, solar_export,
                                          tariff_config, feed_in_tariff, intervals)

            total_cycles = sum(battery_discharge) / battery_size if battery_size > 0 else 0

            return {
                'name': 'TOU Arbitrage',
                'description': 'Charge battery off-peak, discharge at peak times',
                'annual_cost': costs['total_cost'],
                'grid_import_kwh': sum(grid_import),
                'solar_export_kwh': sum(solar_export),
                'battery_charge_kwh': sum(battery_charge),
                'battery_discharge_kwh': sum(battery_discharge),
                'battery_cycles': total_cycles,
                'costs': costs
            }

        except Exception as e:
            logger.error(f"Error in TOU arbitrage simulation: {str(e)}")
            raise

    def _simulate_smart_hybrid(self, load_profile, solar_profile, battery_size,
                               tariff_config, feed_in_tariff):
        """Scenario 3: Smart hybrid - solar priority + TOU optimization"""
        try:
            intervals = min(len(load_profile), len(solar_profile))
            battery_soc = battery_size * 0.5  # Start at 50%

            grid_import = []
            solar_export = []
            battery_charge = []
            battery_discharge = []

            max_charge_per_interval = battery_size * self.max_charge_rate / 2
            max_discharge_per_interval = battery_size * self.max_discharge_rate / 2

            for i in range(intervals):
                load = load_profile[i]
                solar = solar_profile[i]

                hour = (i % 48) // 2
                is_weekend = ((i // 48) % 7) >= 5
                tou_period = self._get_tou_period(hour, is_weekend)

                charge_amount = 0
                discharge_amount = 0

                # Smart dispatch logic
                if solar > load:
                    # Excess solar available
                    excess_solar = solar - load

                    # First priority: charge battery with excess solar
                    available_capacity = battery_size - battery_soc
                    battery_charge_from_solar = min(excess_solar, available_capacity,
                                                    max_charge_per_interval)
                    charge_amount = battery_charge_from_solar

                    # Remaining excess goes to grid
                    remaining_excess = excess_solar - battery_charge_from_solar
                    grid_import.append(0)
                    solar_export.append(remaining_excess)

                else:
                    # Load exceeds solar
                    shortfall = load - solar

                    # Decide whether to use battery or grid
                    if tou_period == 'peak':
                        # Peak period - use battery if available
                        available_battery = min(battery_soc, max_discharge_per_interval)
                        discharge_amount = min(shortfall, available_battery)
                        shortfall -= discharge_amount

                    elif tou_period == 'offpeak' and battery_soc < battery_size * 0.8:
                        # Off-peak and battery not full - charge from grid
                        available_capacity = battery_size * 0.8 - battery_soc
                        charge_from_grid = min(available_capacity, max_charge_per_interval)
                        charge_amount = charge_from_grid

                    # Remaining load from grid
                    grid_import.append(shortfall + charge_amount)
                    solar_export.append(0)

                # Update battery state
                battery_soc += charge_amount * self.battery_efficiency
                battery_soc -= discharge_amount
                battery_soc = max(0, min(battery_size, battery_soc))

                battery_charge.append(charge_amount)
                battery_discharge.append(discharge_amount)

            costs = self._calculate_costs(grid_import, solar_export,
                                          tariff_config, feed_in_tariff, intervals)

            total_cycles = sum(battery_discharge) / battery_size if battery_size > 0 else 0
            solar_consumption = sum(min(load_profile[i], solar_profile[i])
                                    for i in range(intervals))

            return {
                'name': 'Smart Hybrid',
                'description': 'Solar priority with intelligent TOU optimization',
                'annual_cost': costs['total_cost'],
                'grid_import_kwh': sum(grid_import),
                'solar_export_kwh': sum(solar_export),
                'solar_consumption_kwh': solar_consumption,
                'battery_charge_kwh': sum(battery_charge),
                'battery_discharge_kwh': sum(battery_discharge),
                'battery_cycles': total_cycles,
                'self_sufficiency': (solar_consumption + sum(battery_discharge)) / sum(load_profile[:intervals]) * 100,
                'costs': costs
            }

        except Exception as e:
            logger.error(f"Error in smart hybrid simulation: {str(e)}")
            raise

    def _get_tou_period(self, hour, is_weekend):
        """Determine TOU period for given hour"""
        if is_weekend:
            if 7 <= hour < 22:
                return 'shoulder'
            else:
                return 'offpeak'
        else:  # Weekday
            if 14 <= hour < 20:  # 2PM-8PM
                return 'peak'
            elif 7 <= hour < 14 or 20 <= hour < 22:  # 7AM-2PM, 8PM-10PM
                return 'shoulder'
            else:
                return 'offpeak'

    def _calculate_costs(self, grid_import, solar_export, tariff_config,
                         feed_in_tariff, intervals):
        """Calculate electricity costs for given import/export profile"""
        try:
            total_import = sum(grid_import)
            total_export = sum(solar_export)

            if tariff_config['type'] == 'flat':
                # Flat rate calculation
                import_cost = total_import * tariff_config['rate'] / 100

            else:
                # TOU calculation
                peak_import = shoulder_import = offpeak_import = 0

                for i in range(intervals):
                    hour = (i % 48) // 2
                    is_weekend = ((i // 48) % 7) >= 5
                    period = self._get_tou_period(hour, is_weekend)

                    if period == 'peak':
                        peak_import += grid_import[i]
                    elif period == 'shoulder':
                        shoulder_import += grid_import[i]
                    else:
                        offpeak_import += grid_import[i]

                import_cost = (
                        peak_import * tariff_config['peak_rate'] / 100 +
                        shoulder_import * tariff_config.get('shoulder_rate', tariff_config['offpeak_rate']) / 100 +
                        offpeak_import * tariff_config['offpeak_rate'] / 100
                )

            # Export income
            export_income = total_export * feed_in_tariff / 100

            # Supply charges (daily charge)
            supply_cost = tariff_config.get('daily_charge', 1.20) * 365

            # Prevent unrealistic negative bills from excessive export income
            # In reality, utilities often cap credits or have different export limits
            max_credit = import_cost + supply_cost * 0.9  # Allow up to 90% credit on supply charges
            capped_export_income = min(export_income, max_credit)

            # Calculate net cost
            total_cost = import_cost + supply_cost - capped_export_income

            # Ensure minimum cost (some utilities have minimum charges)
            total_cost = max(supply_cost * 0.1, total_cost)  # Minimum 10% of supply charges

            # Debug logging to track the calculation
            logger.debug(f"Cost calculation: import={import_cost:.2f}, supply={supply_cost:.2f}, "
                         f"export_income={export_income:.2f}, capped_export={capped_export_income:.2f}, "
                         f"final_cost={total_cost:.2f}")

            return {
                'total_cost': total_cost,
                'import_cost': import_cost,
                'supply_cost': supply_cost,
                'export_income': capped_export_income,  # Return capped value
                'uncapped_export_income': export_income,  # Track what it would have been
                'total_import_kwh': total_import,
                'total_export_kwh': total_export
            }

        except Exception as e:
            logger.error(f"Error calculating costs: {str(e)}")
            raise

    def recommend_battery_size(self, load_profile, solar_profile):
        """
        Recommend optimal battery size based on load and solar patterns

        Args:
            load_profile: Half-hourly load data
            solar_profile: Half-hourly solar generation

        Returns:
            Dict with battery size recommendation and rationale
        """
        try:
            intervals = min(len(load_profile), len(solar_profile))

            # Calculate evening energy needs (6PM-10PM)
            evening_load = []
            excess_solar = []

            for day in range(intervals // 48):
                day_start = day * 48

                # Evening load (6PM-10PM = intervals 36-43)
                evening_intervals = range(36, 44)  # 6PM-10PM
                day_evening_load = sum(load_profile[day_start + i] for i in evening_intervals)
                evening_load.append(day_evening_load)

                # Midday excess solar (10AM-4PM = intervals 20-31)
                midday_intervals = range(20, 32)  # 10AM-4PM
                day_excess = sum(max(0, solar_profile[day_start + i] - load_profile[day_start + i])
                                 for i in midday_intervals)
                excess_solar.append(day_excess)

            # Calculate recommendations
            avg_evening_load = np.mean(evening_load)
            avg_excess_solar = np.mean(excess_solar)
            p90_evening_load = np.percentile(evening_load, 90)

            # Recommend based on evening needs, constrained by available solar
            recommended_size = min(p90_evening_load, avg_excess_solar * 1.2)

            # Round to standard battery sizes
            if recommended_size <= 2:
                recommended_size = 0
            elif recommended_size <= 7:
                recommended_size = round(recommended_size * 2) / 2  # 0.5 kWh increments
            elif recommended_size <= 15:
                recommended_size = round(recommended_size)  # 1 kWh increments
            else:
                recommended_size = round(recommended_size / 2.5) * 2.5  # 2.5 kWh increments

            # Calculate utilization estimate
            if recommended_size > 0:
                utilization = min(100, (avg_evening_load / recommended_size) * 100)
            else:
                utilization = 0

            return {
                'recommended_size_kwh': recommended_size,
                'avg_evening_load': avg_evening_load,
                'avg_excess_solar': avg_excess_solar,
                'estimated_utilization': utilization,
                'rationale': self._get_battery_rationale(recommended_size, avg_evening_load, avg_excess_solar)
            }

        except Exception as e:
            logger.error(f"Error recommending battery size: {str(e)}")
            return {
                'recommended_size_kwh': 10,
                'avg_evening_load': 5,
                'avg_excess_solar': 8,
                'estimated_utilization': 50,
                'rationale': 'Error calculating recommendation - using default'
            }

    def _get_battery_rationale(self, size, evening_load, excess_solar):
        """Generate explanation for battery recommendation"""
        if size == 0:
            if excess_solar < 2:
                return "Insufficient solar generation to justify battery storage"
            elif evening_load < 2:
                return "Low evening energy usage - battery may not be cost-effective"
            else:
                return "Battery not recommended for this usage pattern"

        elif size < evening_load:
            return f"Sized for {size:.1f}kWh to capture most valuable peak-hour savings"

        elif size > excess_solar:
            return f"Sized for {size:.1f}kWh based on evening needs, but may need grid charging"

        else:
            return f"Optimal {size:.1f}kWh size balances evening load coverage with solar availability"

    def calculate_battery_economics(self, scenarios, battery_cost, battery_life_years=10):
        """
        Calculate battery economics across scenarios

        Args:
            scenarios: Results from analyze_scenarios()
            battery_cost: Battery system cost ($)
            battery_life_years: Expected battery life

        Returns:
            Dict with economic analysis
        """
        try:
            baseline_cost = scenarios['solar_only']['annual_cost']

            economics = {}

            for scenario_name, scenario_data in scenarios.items():
                if scenario_name == 'solar_only':
                    continue

                annual_savings = baseline_cost - scenario_data['annual_cost']

                if annual_savings > 0:
                    payback_period = battery_cost / annual_savings
                    lifetime_savings = annual_savings * battery_life_years - battery_cost
                    roi = (lifetime_savings / battery_cost) * 100 if battery_cost > 0 else 0
                else:
                    payback_period = float('inf')
                    lifetime_savings = -battery_cost
                    roi = -100

                economics[scenario_name] = {
                    'annual_savings': annual_savings,
                    'payback_years': payback_period,
                    'lifetime_savings': lifetime_savings,
                    'roi_percent': roi,
                    'cycles_per_year': scenario_data.get('battery_cycles', 0)
                }

            return economics

        except Exception as e:
            logger.error(f"Error calculating battery economics: {str(e)}")
            raise