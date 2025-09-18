import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class ProfileBuilder:
    """
    Builds household energy profiles based on household characteristics
    or processes uploaded user data. Uses baseline data from a reference house.
    """

    def __init__(self):
        self.baseline_annual_kwh = 6100  # Reference 4-person detached house
        self.baseline_occupants = 4

        # Load reference profile from Inputs.csv if available
        self.reference_profile = self._load_reference_profile()

        # Appliance impact factors (additional annual kWh usage)
        self.appliance_factors = {
            'aircon': 2100,  # +2100 kWh/year for air conditioning
            'ev': 2750,  # +2750 kWh/year for electric vehicle (avg 7500km/year)
            'electric_hw': 1800,  # +1800 kWh/year for electric hot water vs gas
            'pool_pump': 900  # +900 kWh/year for pool pump
        }

        # Building type factors
        self.building_factors = {
            'detached': 1.0,
            'semi_detached': 0.85,
            'townhouse': 0.75,
            'apartment': 0.6,
            'unit': 0.7
        }

        # Occupant scaling (logarithmic to account for shared appliances)
        self.occupant_scaling = {
            1: 0.6,
            2: 0.8,
            3: 0.95,
            4: 1.0,  # Baseline
            5: 1.12,
            6: 1.22,
            7: 1.30,
            8: 1.35
        }

    def _load_reference_profile(self):
        """Load the reference energy profile from Inputs.csv"""
        try:
            # Try to load the reference file
            data_path = Path(__file__).parent.parent / 'data' / 'Inputs.csv'

            if data_path.exists():
                df = pd.read_csv(data_path)

                # Handle different column name formats
                datetime_col = None
                energy_col = None

                if 'dt' in df.columns:
                    datetime_col = 'dt'
                elif 'datetime' in df.columns:
                    datetime_col = 'datetime'
                else:
                    logger.error("No datetime column found in Inputs.csv")
                    return self._generate_synthetic_reference()

                if 'kWh' in df.columns:
                    energy_col = 'kWh'
                elif 'energy_use_kwh' in df.columns:
                    energy_col = 'energy_use_kwh'
                else:
                    logger.error("No energy column found in Inputs.csv")
                    return self._generate_synthetic_reference()

                # Parse datetime with Australian format (day first)
                df['datetime'] = pd.to_datetime(df[datetime_col], dayfirst=True, errors='coerce')

                # Handle any parsing errors
                if df['datetime'].isna().any():
                    logger.warning(f"Some dates couldn't be parsed, dropping {df['datetime'].isna().sum()} rows")
                    df = df.dropna(subset=['datetime'])

                df['energy_use_kwh'] = pd.to_numeric(df[energy_col], errors='coerce')
                df = df.dropna(subset=['energy_use_kwh'])
                df = df.set_index('datetime')

                # Extract just the energy profile and normalize
                profile = df['energy_use_kwh'].values

                # Ensure we have a full year of half-hourly data
                if len(profile) >= 17500:  # Accept close to full year
                    return profile[:17520] if len(profile) >= 17520 else np.pad(profile, (0, 17520 - len(profile)),'wrap') # Take first year
                else:
                    logger.warning(f"Reference profile has {len(profile)} intervals, expected 17520. Using synthetic.")
                    return self._generate_synthetic_reference()
            else:
                logger.warning("Inputs.csv not found, using synthetic reference profile")
                return self._generate_synthetic_reference()

        except Exception as e:
            logger.error(f"Error loading reference profile: {str(e)}")
            return self._generate_synthetic_reference()

    def _generate_synthetic_reference(self):
        """Generate a synthetic reference profile based on typical Australian residential patterns"""

        # Daily profile (48 half-hourly intervals) - normalized values
        daily_pattern = [
            # 00:00-06:00 - Night time low usage
            0.25, 0.23, 0.21, 0.20, 0.19, 0.18, 0.17, 0.17, 0.18, 0.19, 0.21, 0.23,
            # 06:00-12:00 - Morning peak
            0.35, 0.55, 0.75, 0.85, 0.65, 0.45, 0.35, 0.30, 0.28, 0.27, 0.26, 0.25,
            # 12:00-18:00 - Midday moderate
            0.30, 0.35, 0.40, 0.38, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.85,
            # 18:00-24:00 - Evening peak
            1.00, 1.15, 1.20, 1.10, 0.95, 0.80, 0.65, 0.50, 0.40, 0.35, 0.30, 0.27
        ]

        # Seasonal factors for each month
        seasonal_factors = [
            1.25, 1.20, 1.10, 0.95, 0.80, 0.75,  # Jan-Jun (summer high, winter low in AU)
            0.70, 0.75, 0.85, 0.95, 1.10, 1.20  # Jul-Dec
        ]

        # Generate full year profile
        annual_profile = []

        np.random.seed(42)  # Consistent random variation

        for month in range(12):
            days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month]
            month_factor = seasonal_factors[month]

            for day in range(days_in_month):
                # Add some random variation (±15%)
                daily_variation = np.random.normal(1.0, 0.15)
                daily_variation = max(0.5, min(1.5, daily_variation))  # Constrain

                for interval in range(48):
                    base_value = daily_pattern[interval] * month_factor * daily_variation
                    annual_profile.append(max(0.05, base_value))  # Minimum usage

        # Normalize to baseline annual usage
        profile_array = np.array(annual_profile)
        profile_array = profile_array * (self.baseline_annual_kwh / profile_array.sum())

        logger.info(
            f"Generated synthetic reference profile: {len(profile_array)} intervals, {profile_array.sum():.1f} kWh annual")

        return profile_array

    def generate_profile(self, occupants=4, building_type='detached',
                         has_aircon=False, has_ev=False, has_electric_hw=False,
                         has_pool=False):
        """
        Generate a household energy profile based on characteristics

        Args:
            occupants: Number of people (1-8)
            building_type: Type of dwelling
            has_aircon: Has air conditioning
            has_ev: Has electric vehicle
            has_electric_hw: Has electric hot water
            has_pool: Has pool pump

        Returns:
            List of annual half-hourly energy usage values (kWh)
        """
        try:
            # Start with baseline profile (4-person detached house = 6100 kWh/year)
            if self.reference_profile is not None:
                base_profile = self.reference_profile.copy()
                # Normalize to baseline annual usage
                base_profile = base_profile * (self.baseline_annual_kwh / base_profile.sum())
            else:
                base_profile = self._generate_synthetic_reference()

            # Calculate total annual usage based on household characteristics

            # 1. Start with baseline for occupants
            occupant_factor = self.occupant_scaling.get(occupants,
                                                        1.0 + 0.05 * (occupants - 4))  # Linear extrapolation beyond 8

            # 2. Apply building type factor
            building_factor = self.building_factors.get(building_type, 1.0)

            # 3. Calculate base annual usage
            base_annual = self.baseline_annual_kwh * occupant_factor * building_factor

            # 4. Add appliance usage
            total_annual = base_annual
            if has_aircon:
                total_annual += self.appliance_factors['aircon']
            if has_ev:
                total_annual += self.appliance_factors['ev']
            if has_electric_hw:
                total_annual += self.appliance_factors['electric_hw']
            if has_pool:
                total_annual += self.appliance_factors['pool_pump']

            # 5. Scale base profile to match total annual usage
            final_profile = base_profile * (total_annual / base_profile.sum())

            # 6. Add appliance-specific load patterns (these modify timing, not total)
            if has_ev:
                final_profile = self._add_ev_charging_pattern(final_profile)

            if has_pool:
                final_profile = self._add_pool_pump_pattern(final_profile)

            if has_aircon:
                final_profile = self._add_aircon_pattern(final_profile)

            logger.info(f"Generated profile: {len(final_profile)} intervals, "
                        f"{sum(final_profile):.1f} kWh annual, "
                        f"{total_annual:.1f} kWh target")

            return final_profile.tolist()

        except Exception as e:
            logger.error(f"Error generating profile: {str(e)}")
            raise

    def _add_ev_charging_pattern(self, profile):
        """Redistribute some energy to EV charging times (typically 10PM-6AM)"""
        # Don't add energy, just redistribute existing EV energy to charging hours
        ev_energy = self.appliance_factors['ev']
        daily_ev = ev_energy / 365

        # Remove proportional energy from all hours, add to charging hours
        energy_to_redistribute = daily_ev * 0.8  # 80% of EV energy at specific times

        for day in range(365):
            day_start = day * 48

            # Charging window: 10PM to 6AM (16 intervals)
            charging_intervals = list(range(44, 48)) + list(range(0, 12))  # 10PM-12AM + 12AM-6AM

            # Remove small amount from all intervals
            for i in range(48):
                idx = day_start + i
                if idx < len(profile):
                    profile[idx] -= energy_to_redistribute / 48

            # Add to charging intervals
            for i in charging_intervals:
                idx = day_start + i
                if idx < len(profile):
                    profile[idx] += energy_to_redistribute / len(charging_intervals)

        return profile

    def _add_pool_pump_pattern(self, profile):
        """Redistribute pool pump energy to daytime hours"""
        pool_energy = self.appliance_factors['pool_pump']

        for day in range(365):
            day_start = day * 48
            month = ((day // 30) % 12) + 1

            # Summer: run 8 hours, Winter: 4 hours, Shoulder: 6 hours
            if month in [11, 12, 1, 2, 3]:  # Summer
                run_hours = 8
            elif month in [5, 6, 7, 8, 9]:  # Winter
                run_hours = 4
            else:
                run_hours = 6

            daily_energy = (pool_energy / 365) * (run_hours / 6)  # Scale by season

            # Run during 8AM-4PM (intervals 16-31)
            run_intervals = list(range(16, 16 + run_hours * 2))

            for i in run_intervals:
                idx = day_start + i
                if idx < len(profile):
                    profile[idx] += daily_energy / len(run_intervals)

        return profile

    def _add_aircon_pattern(self, profile):
        """Redistribute aircon energy to hot periods"""
        ac_energy = self.appliance_factors['aircon']

        for day in range(365):
            day_start = day * 48
            month = ((day // 30) % 12) + 1

            # Seasonal AC usage
            if month in [12, 1, 2]:  # Peak summer
                daily_factor = 2.0
            elif month in [11, 3]:  # Shoulder summer
                daily_factor = 1.2
            elif month in [6, 7, 8]:  # Winter (some heating)
                daily_factor = 0.6
            else:
                daily_factor = 0.3  # Mild months

            daily_energy = (ac_energy / 365) * daily_factor

            # Peak usage 2PM-8PM and overnight 10PM-6AM
            peak_intervals = list(range(28, 40))  # 2PM-8PM
            night_intervals = list(range(44, 48)) + list(range(0, 12))  # 10PM-6AM

            # Distribute 70% to peak, 30% to overnight
            for i in peak_intervals:
                idx = day_start + i
                if idx < len(profile):
                    profile[idx] += (daily_energy * 0.7) / len(peak_intervals)

            for i in night_intervals:
                idx = day_start + i
                if idx < len(profile):
                    profile[idx] += (daily_energy * 0.3) / len(night_intervals)

        return profile

    def get_profile_summary(self, energy_profile):
        """Get summary statistics for an energy profile"""
        if not energy_profile:
            return None

        # Handle both list and dict formats
        if isinstance(energy_profile, list):
            usage_values = energy_profile
        else:
            # Assume it's a list of dicts with 'energy_use_kwh' key
            usage_values = [row['energy_use_kwh'] for row in energy_profile]

        annual_usage = sum(usage_values)

        return {
            'annual_usage': annual_usage,
            'avg_daily_usage': annual_usage / 365,
            'max_interval_usage': max(usage_values),
            'min_interval_usage': min(usage_values),
            'avg_interval_usage': annual_usage / len(usage_values)
        }

    @staticmethod
    def validate_uploaded_data(df):
        """Validate structure and quality of uploaded energy data"""
        issues = []

        # Check for either format of required columns
        datetime_col = None
        energy_col = None

        if 'datetime' in df.columns:
            datetime_col = 'datetime'
        elif 'dt' in df.columns:
            datetime_col = 'dt'
        else:
            issues.append("Missing datetime column (expected 'datetime' or 'dt')")

        if 'energy_use_kwh' in df.columns:
            energy_col = 'energy_use_kwh'
        elif 'kWh' in df.columns:
            energy_col = 'kWh'
        else:
            issues.append("Missing energy column (expected 'energy_use_kwh' or 'kWh')")

        if datetime_col and energy_col:
            # Check data types and convert
            try:
                df['datetime_parsed'] = pd.to_datetime(df[datetime_col], dayfirst=True, errors='coerce')
                invalid_dates = df['datetime_parsed'].isna().sum()
                if invalid_dates > 0:
                    issues.append(f"{invalid_dates} rows have invalid datetime format")
            except Exception as e:
                issues.append(f"Error parsing datetime: {str(e)}")

            try:
                df['energy_parsed'] = pd.to_numeric(df[energy_col], errors='coerce')
                null_count = df['energy_parsed'].isnull().sum()
                if null_count > 0:
                    issues.append(f"{null_count} rows have invalid energy values")
            except Exception as e:
                issues.append(f"Error parsing energy values: {str(e)}")

            # Check data coverage
            valid_rows = len(df) - df['datetime_parsed'].isna().sum() - df['energy_parsed'].isna().sum()
            if valid_rows < 48 * 7:  # Less than a week of data
                issues.append("Insufficient data - need at least one week of half-hourly data")

            # Check for reasonable energy values
            if 'energy_parsed' in df.columns:
                max_val = df['energy_parsed'].max()
                if max_val > 20:  # More than 20kWh in 30 minutes seems excessive
                    issues.append(f"Unusually high energy values detected (max: {max_val:.2f} kWh)")

                annual_estimate = df['energy_parsed'].sum() * (365 * 48 / len(df))
                if annual_estimate > 50000:  # More than 50MWh/year seems excessive for residential
                    issues.append(f"Annual usage estimate seems too high: {annual_estimate:.0f} kWh")
                elif annual_estimate < 1000:  # Less than 1MWh/year seems too low
                    issues.append(f"Annual usage estimate seems too low: {annual_estimate:.0f} kWh")

        return issues

    def process_uploaded_profile(self, df, recent_bill=None):
        """
        Process uploaded energy data, handling part-year data with seasonal adjustment

        Args:
            df: Pandas DataFrame with datetime and energy columns
            recent_bill: Optional recent bill amount for validation

        Returns:
            Dict with processed profile data and metadata
        """
        try:
            # Standardize column names
            df = self._standardize_columns(df)

            # Parse and sort by datetime
            df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True, errors='coerce')
            df = df.dropna(subset=['datetime', 'energy_use_kwh']).sort_values('datetime')

            # Detect data timespan and frequency
            profile_info = self._analyze_profile_coverage(df)

            if profile_info['is_full_year']:
                # Use data as-is for full year
                annual_profile = df['energy_use_kwh'].tolist()
                logger.info(f"Using full year of uploaded data: {len(annual_profile)} intervals")

            elif profile_info['sufficient_data']:
                # Extend part-year data to full year with seasonal adjustment
                annual_profile = self._extend_partial_year(df, profile_info)
                logger.info(
                    f"Extended {profile_info['data_months']} months to full year: {len(annual_profile)} intervals")

            else:
                raise ValueError(f"Insufficient data: only {profile_info['data_days']} days available")

            # Validate energy values are reasonable
            self._validate_energy_values(annual_profile, recent_bill)

            # Generate summary
            summary = self.get_profile_summary(annual_profile)

            return {
                'success': True,
                'profile': annual_profile,
                'summary': summary,
                'metadata': {
                    'source': 'uploaded',
                    'original_data_points': len(df),
                    'data_coverage': profile_info,
                    'validation_status': 'passed'
                }
            }

        except Exception as e:
            logger.error(f"Error processing uploaded profile: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'profile': None
            }

    def _standardize_columns(self, df):
        """Standardize column names to expected format"""
        # Map common column name variations
        column_mapping = {
            'dt': 'datetime',
            'date': 'datetime',
            'timestamp': 'datetime',
            'time': 'datetime',
            'kWh': 'energy_use_kwh',
            'kwh': 'energy_use_kwh',
            'energy': 'energy_use_kwh',
            'usage': 'energy_use_kwh',
            'consumption': 'energy_use_kwh'
        }

        # Rename columns
        df = df.rename(columns=column_mapping)

        # Ensure we have required columns
        if 'datetime' not in df.columns:
            raise ValueError("No datetime column found. Expected 'datetime', 'dt', 'date', or 'timestamp'")
        if 'energy_use_kwh' not in df.columns:
            raise ValueError("No energy column found. Expected 'energy_use_kwh', 'kWh', 'energy', or 'usage'")

        return df[['datetime', 'energy_use_kwh']]

    def _analyze_profile_coverage(self, df):
        """Analyze temporal coverage of uploaded data"""
        start_date = df['datetime'].min()
        end_date = df['datetime'].max()
        data_span = (end_date - start_date).days

        # Detect frequency (30min, hourly, daily)
        time_diffs = df['datetime'].diff().dropna()
        most_common_interval = time_diffs.mode().iloc[0] if len(time_diffs) > 0 else pd.Timedelta(minutes=30)

        if most_common_interval <= pd.Timedelta(minutes=35):
            expected_frequency = 'half_hourly'
            intervals_per_day = 48
        elif most_common_interval <= pd.Timedelta(hours=1.5):
            expected_frequency = 'hourly'
            intervals_per_day = 24
        else:
            expected_frequency = 'daily'
            intervals_per_day = 1

        # Calculate coverage
        expected_intervals = data_span * intervals_per_day
        actual_intervals = len(df)
        coverage_ratio = actual_intervals / expected_intervals if expected_intervals > 0 else 0

        # Determine data quality
        data_days = data_span + 1
        data_months = data_days / 30.44  # Average days per month

        is_full_year = data_days >= 350  # Allow some missing data
        sufficient_data = data_days >= 60 and coverage_ratio >= 0.8  # At least 2 months with good coverage

        return {
            'start_date': start_date,
            'end_date': end_date,
            'data_days': data_days,
            'data_months': data_months,
            'frequency': expected_frequency,
            'intervals_per_day': intervals_per_day,
            'coverage_ratio': coverage_ratio,
            'is_full_year': is_full_year,
            'sufficient_data': sufficient_data,
            'actual_intervals': actual_intervals,
            'expected_intervals': expected_intervals
        }

    def _extend_partial_year(self, df, profile_info):
        """Extend partial year data to full year using seasonal patterns"""

        # Convert to half-hourly if needed
        if profile_info['frequency'] == 'hourly':
            df_extended = self._hourly_to_half_hourly(df)
        elif profile_info['frequency'] == 'daily':
            df_extended = self._daily_to_half_hourly(df)
        else:
            df_extended = df.copy()

        # Create full year template
        full_year_dates = pd.date_range(
            start=pd.Timestamp('2023-01-01'),  # Use consistent year
            end=pd.Timestamp('2023-12-31 23:30:00'),
            freq='30T'  # 30-minute intervals
        )

        # Australian seasonal factors by month
        seasonal_factors = {
            1: 1.25, 2: 1.20, 3: 1.10, 4: 0.95, 5: 0.80, 6: 0.75,  # Summer high to winter low
            7: 0.70, 8: 0.75, 9: 0.85, 10: 0.95, 11: 1.10, 12: 1.20
        }

        # Map uploaded data to consistent year
        df_extended['normalized_date'] = df_extended['datetime'].apply(
            lambda x: x.replace(year=2023)
        )

        # Create daily average patterns from uploaded data
        df_extended['month'] = df_extended['normalized_date'].dt.month
        df_extended['hour'] = df_extended['normalized_date'].dt.hour
        df_extended['minute'] = df_extended['normalized_date'].dt.minute
        df_extended['time_of_day'] = df_extended['hour'] + df_extended['minute'] / 60

        # Calculate average patterns by time of day and month (where data exists)
        daily_patterns = {}
        monthly_averages = {}

        for month in df_extended['month'].unique():
            month_data = df_extended[df_extended['month'] == month]
            monthly_averages[month] = month_data['energy_use_kwh'].mean()

            # Create hourly pattern for this month
            hourly_avg = month_data.groupby('time_of_day')['energy_use_kwh'].mean()
            daily_patterns[month] = hourly_avg

        # Fill missing months using similar months and seasonal adjustment
        base_monthly_avg = np.mean(list(monthly_averages.values()))
        base_pattern = pd.concat(daily_patterns.values()).groupby(level=0).mean()

        # Generate full year profile
        annual_profile = []

        for date in full_year_dates:
            month = date.month
            time_of_day = date.hour + date.minute / 60

            if month in monthly_averages:
                # Use actual data pattern
                if month in daily_patterns and time_of_day in daily_patterns[month]:
                    value = daily_patterns[month][time_of_day]
                else:
                    # Interpolate from available times
                    value = base_pattern.get(time_of_day, base_monthly_avg * seasonal_factors[month])
            else:
                # Estimate using seasonal adjustment
                seasonal_factor = seasonal_factors[month]
                base_value = base_pattern.get(time_of_day, base_monthly_avg)
                value = base_value * seasonal_factor

            # Add some random variation (±10%)
            variation = np.random.normal(1.0, 0.1)
            variation = max(0.5, min(1.5, variation))

            annual_profile.append(max(0.01, value * variation))

        # Normalize to match average usage level from uploaded data
        original_avg = df_extended['energy_use_kwh'].mean()
        profile_avg = np.mean(annual_profile)
        if profile_avg > 0:
            annual_profile = [x * (original_avg / profile_avg) for x in annual_profile]

        return annual_profile

    def _hourly_to_half_hourly(self, df):
        """Convert hourly data to half-hourly by splitting each hour"""
        half_hourly_data = []

        for _, row in df.iterrows():
            # Split each hour into two 30-minute periods
            value = row['energy_use_kwh'] / 2

            half_hourly_data.extend([
                {'datetime': row['datetime'], 'energy_use_kwh': value},
                {'datetime': row['datetime'] + pd.Timedelta(minutes=30), 'energy_use_kwh': value}
            ])

        return pd.DataFrame(half_hourly_data)

    def _daily_to_half_hourly(self, df):
        """Convert daily data to half-hourly using typical daily patterns"""
        # Standard daily pattern (48 half-hourly values)
        daily_pattern = np.array([
            0.25, 0.23, 0.21, 0.20, 0.19, 0.18, 0.17, 0.17, 0.18, 0.19, 0.21, 0.23,  # 00:00-06:00
            0.35, 0.55, 0.75, 0.85, 0.65, 0.45, 0.35, 0.30, 0.28, 0.27, 0.26, 0.25,  # 06:00-12:00
            0.30, 0.35, 0.40, 0.38, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.85,  # 12:00-18:00
            1.00, 1.15, 1.20, 1.10, 0.95, 0.80, 0.65, 0.50, 0.40, 0.35, 0.30, 0.27  # 18:00-24:00
        ])

        # Normalize pattern
        daily_pattern = daily_pattern / daily_pattern.sum()

        half_hourly_data = []

        for _, row in df.iterrows():
            daily_total = row['energy_use_kwh']
            date = row['datetime'].date()

            # Distribute daily total across half-hourly intervals
            for i, pattern_value in enumerate(daily_pattern):
                interval_datetime = pd.Timestamp.combine(date, pd.Timestamp.min.time()) + pd.Timedelta(minutes=30 * i)
                interval_value = daily_total * pattern_value

                half_hourly_data.append({
                    'datetime': interval_datetime,
                    'energy_use_kwh': interval_value
                })

        return pd.DataFrame(half_hourly_data)

    def _validate_energy_values(self, profile, recent_bill=None):
        """Validate that energy values are reasonable"""
        if not profile:
            raise ValueError("Profile is empty")

        max_interval = max(profile)
        min_interval = min(profile)
        annual_total = sum(profile)

        # Check for unrealistic values
        if max_interval > 50:  # More than 50kWh in 30 minutes
            raise ValueError(f"Unrealistically high interval usage: {max_interval:.2f} kWh")

        if min_interval < 0:
            raise ValueError("Negative energy values found")

        if annual_total > 100000:  # More than 100 MWh/year
            raise ValueError(f"Unrealistically high annual usage: {annual_total:.0f} kWh")

        if annual_total < 1000:  # Less than 1 MWh/year
            raise ValueError(f"Unrealistically low annual usage: {annual_total:.0f} kWh")

        # Validate against recent bill if provided
        if recent_bill:
            # Rough validation - bill should be reasonable for usage level
            expected_annual_bill = annual_total * 0.25  # Rough $0.25/kWh estimate
            quarterly_bill = expected_annual_bill / 4

            if recent_bill > quarterly_bill * 3 or recent_bill < quarterly_bill * 0.3:
                logger.warning(
                    f"Bill amount (${recent_bill:.2f}) seems inconsistent with usage ({annual_total:.0f} kWh/year)")

        logger.info(f"Profile validation passed: {annual_total:.0f} kWh/year, max interval: {max_interval:.2f} kWh")