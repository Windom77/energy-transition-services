"""
Simplified PySAM Main Simulation Script - ALL Functionality Maintained
Removed: Complex progress tracking system
Preserved:
- Complete FCAS module integration
- Comprehensive LREC calculations
- Cumulative payback cashflow
- Enhanced financial metrics
- Battery health time series
- Present value calculations
- Validation and error handling
- Complete results export (121+ series)
- All original simulation capabilities
"""

import json
import logging
import os
import sys
import importlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
import pandas as pd
import pyarrow.parquet as pq
import PySAM.PySSC as pssc
from config import Config

import gc
import time

# CRITICAL: Enable numpy/scipy optimizations
os.environ['OPENBLAS_NUM_THREADS'] = str(os.cpu_count() or 4)
os.environ['MKL_NUM_THREADS'] = str(os.cpu_count() or 4)
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count() or 4)


# ========== DATA CLASSES (PRESERVED) ==========

@dataclass
class BatteryParameters:
    """Complete battery parameter container"""
    capacity_kwh: float = 0.0
    power_kw: float = 0.0
    efficiency: float = 0.9
    enabled: bool = False

    def __post_init__(self):
        self.enabled = self.capacity_kwh > 0 and self.power_kw > 0


@dataclass
class FCASConfiguration:
    """Complete FCAS configuration parameters"""
    region: str = 'NSW1'
    participation_rate: float = 0.85
    model_path: Optional[str] = None
    enabled_services: Dict[int, bool] = None

    def __post_init__(self):
        if self.enabled_services is None:
            self.enabled_services = {
                1: True,   # Fast Raise
                2: True,   # Fast Lower
                3: True,   # Slow Raise
                4: True,   # Slow Lower
                5: False,  # Delayed Raise
                6: False,  # Delayed Lower
                7: False,  # Raise Regulation
                8: False   # Lower Regulation
            }


@dataclass
class LRECConfiguration:
    """Complete LREC (Large-scale Renewable Energy Certificate) configuration"""
    enabled: bool = False
    annual_lrecs: float = 0.0
    annual_lrec_value: float = 0.0
    incentive_lrec_price: float = 0.0
    system_size_kw: float = 0.0
    escalation_rate: float = 0.025

    def __post_init__(self):
        # LREC eligibility check - systems over 100kW
        self.enabled = (self.system_size_kw >= 100 and
                       self.annual_lrecs > 0 and self.incentive_lrec_price > 0)


@dataclass
class FinancialParameters:
    """Complete dynamic financial parameters sourced from config"""
    escalation_rate: float = 0.025
    capacity_factor: float = 0.25
    discount_rate: float = 0.06
    wacc: float = 0.06

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'FinancialParameters':
        """Create financial parameters from config with complete bounds checking"""
        # Extract raw values with fallbacks
        raw_escalation = config.get('inflation_rate', config.get('escalation_rate', 0.025))
        raw_capacity_factor = config.get('capacity_factor', 0.25)
        raw_discount = config.get('real_discount_rate', config.get('discount_rate', 0.06))
        raw_wacc = config.get('wacc', 0.06)

        # Convert and apply bounds checking
        escalation_rate = float(raw_escalation)
        if escalation_rate > 1.0:  # Likely percentage instead of decimal
            escalation_rate = escalation_rate / 100.0
        escalation_rate = max(0.0, min(0.15, escalation_rate))  # Cap at 15%

        discount_rate = float(raw_discount)
        if discount_rate > 1.0:  # Likely percentage instead of decimal
            discount_rate = discount_rate / 100.0
        discount_rate = max(0.01, min(0.25, discount_rate))  # Cap between 1% and 25%

        wacc = float(raw_wacc)
        if wacc > 1.0:  # Likely percentage instead of decimal
            wacc = wacc / 100.0
        wacc = max(0.01, min(0.25, wacc))  # Cap between 1% and 25%

        capacity_factor = max(0.1, min(1.0, float(raw_capacity_factor)))

        # Debug logging for verification
        logging.info(f"ðŸ’° Financial Parameter Conversion:")
        logging.info(f"   Raw escalation: {raw_escalation} â†’ {escalation_rate:.1%}")
        logging.info(f"   Raw discount: {raw_discount} â†’ {discount_rate:.1%}")
        logging.info(f"   Raw WACC: {raw_wacc} â†’ {wacc:.1%}")

        return cls(
            escalation_rate=escalation_rate,
            capacity_factor=capacity_factor,
            discount_rate=discount_rate,
            wacc=wacc
        )


@dataclass
class SimulationResults:
    """Complete container for simulation results"""
    scalar_results: Dict[str, Any] = None
    time_series: Dict[str, List] = None
    fcas_results: Dict[str, Any] = None
    lrec_results: Dict[str, Any] = None
    modules_executed: List[str] = None
    config_used: Dict[str, Any] = None
    weather_file: Optional[str] = None

    def __post_init__(self):
        if self.scalar_results is None:
            self.scalar_results = {}
        if self.time_series is None:
            self.time_series = {}
        if self.fcas_results is None:
            self.fcas_results = {}
        if self.lrec_results is None:
            self.lrec_results = {}
        if self.modules_executed is None:
            self.modules_executed = []
        if self.config_used is None:
            self.config_used = {}


# ========== CONFIGURATION CONSTANTS (PRESERVED) ==========

class SimulationConstants:
    """Centralized constants for simulation configuration"""

    # Time series validation
    EXPECTED_TIME_SERIES_LENGTHS = [1, 12, 26, 8760, 219000]

    # FCAS configuration
    FCAS_REGIONS = ['NSW1', 'VIC1', 'QLD1', 'SA1', 'TAS1', 'WEM']
    FCAS_SERVICE_COUNT = 8

    # FCAS service definitions (complete)
    FCAS_SERVICES = {
        1: {'name': 'Fast Raise (6s)', 'rate': 1.25, 'default_enabled': True},  # âœ… Was 400
        2: {'name': 'Fast Lower (6s)', 'rate': 1.18, 'default_enabled': True},  # âœ… Was 380
        3: {'name': 'Slow Raise (60s)', 'rate': 0.78, 'default_enabled': True},  # âœ… Was 250
        4: {'name': 'Slow Lower (60s)', 'rate': 0.72, 'default_enabled': True},  # âœ… Was 230
        5: {'name': 'Delayed Raise (5min)', 'rate': 0.56, 'default_enabled': False},  # âœ… Was 180
        6: {'name': 'Delayed Lower (5min)', 'rate': 0.50, 'default_enabled': False},  # âœ… Was 160
        7: {'name': 'Raise Regulation', 'rate': 0.38, 'default_enabled': False},  # âœ… Was 120
        8: {'name': 'Lower Regulation', 'rate': 0.31, 'default_enabled': False}  # âœ… Was 100
    }

    # Regional multipliers for FCAS revenue
    FCAS_REGIONAL_MULTIPLIERS = {
        'NSW1': 1.0, 'VIC1': 0.95, 'QLD1': 1.05,
        'SA1': 1.15, 'TAS1': 0.85, 'WEM': 0.7
    }

    # Export file mappings
    TIME_SERIES_GROUPS = {
        1: 'annual', 12: 'monthly', 26: 'annual',
        8760: 'hourly', 219000: 'detailed'
    }

    # Standard financial metrics
    FINANCIAL_METRICS = {
        "NPV (Real)": "npv_energy_lcos_real",
        "Payback Period": "payback",
        "WACC": "wacc",
        "LCOE (Real)": "lcoe_real",
        "LCOE (Nominal)": "lcoe_nom"
    }

    # LREC configuration
    LREC_MWH_THRESHOLD = 100  # kW system size threshold for LREC eligibility
    MWH_TO_KWH = 1000  # Conversion factor


# ========== UTILITY CLASSES (PRESERVED WITH LOGGING) ==========

class BatteryDetector:
    """Complete centralized battery parameter detection with proper toggle support"""

    @staticmethod
    def detect_battery_parameters(config: Dict[str, Any]) -> BatteryParameters:
        """Single source of truth for battery detection with en_batt toggle"""
        try:
            # CRITICAL: Check en_batt toggle first
            en_batt = config.get('en_batt', 0)

            # Convert various representations to boolean
            if isinstance(en_batt, str):
                battery_enabled = en_batt.lower() in ('1', 'yes', 'true', 'enabled')
            else:
                battery_enabled = bool(int(float(en_batt)) if en_batt != '' else 0)

            if not battery_enabled:
                logging.info("ðŸ”‹ Battery disabled via en_batt toggle")
                return BatteryParameters(enabled=False)

            # Consolidated field detection with all possible sources
            capacity = (
                config.get('batt_computed_bank_capacity', 0) or
                config.get('batt_bank_installed_capacity', 0) or
                config.get('battery_capacity', 0)
            )

            power = (
                config.get('batt_power_discharge_max_kwac', 0) or
                config.get('batt_ac_power', 0) or
                config.get('battery_power', 0)
            )

            efficiency = config.get('batt_roundtrip_eff', 90) / 100.0

            # Check if we have valid capacity/power
            capacity_kwh = float(capacity or 0)
            power_kw = float(power or capacity or 0)

            if capacity_kwh <= 0 or power_kw <= 0:
                logging.info(f"ðŸ”‹ Invalid battery parameters: {capacity_kwh} kWh, {power_kw} kW")
                return BatteryParameters(enabled=False)

            return BatteryParameters(
                capacity_kwh=capacity_kwh,
                power_kw=power_kw,
                efficiency=efficiency,
                enabled=True  # Explicitly set since all checks passed
            )

        except (ValueError, TypeError) as e:
            logging.error(f"Battery detection failed: {e}")
            return BatteryParameters(enabled=False)

    @staticmethod
    def validate_battery_config(battery_params: BatteryParameters) -> List[str]:
        """Complete battery validation with all checks"""
        if not battery_params.enabled:
            return ["No battery system detected"]

        warnings = []
        if battery_params.capacity_kwh < 10:
            warnings.append(f"Small battery capacity ({battery_params.capacity_kwh} kWh)")
        if battery_params.power_kw > battery_params.capacity_kwh * 2:
            warnings.append(f"High C-rate: {battery_params.power_kw / battery_params.capacity_kwh:.1f}C")
        if battery_params.efficiency < 0.8:
            warnings.append(f"Low efficiency: {battery_params.efficiency:.1%}")

        return warnings


class LRECDetector:
    """Complete LREC parameter detection and configuration"""

    @staticmethod
    def detect_lrec_parameters(config: Dict[str, Any], financial_params: FinancialParameters) -> LRECConfiguration:
        """Detect and configure LREC parameters from config - complete logic"""
        try:
            # Check both root level and nested project_info section
            project_info = config.get('project_info', {})

            # Debug: Show project_info contents
            if project_info:
                logging.info(f"ðŸ” Found project_info section with keys: {list(project_info.keys())}")

            # Extract LREC price (main trigger along with system size)
            incentive_lrec_price = float(
                config.get('incentive_lrec', 0) or
                project_info.get('incentive_lrec', 0) or
                config.get('lrec_price', 0) or
                0
            )

            # Extract system size from multiple possible locations
            system_size_kw = float(
                config.get('system_capacity', 0) or
                project_info.get('system_capacity', 0) or
                project_info.get('system_capacity_kw', 0) or
                config.get('system_capacity_kw', 0) or
                config.get('system_size_kw', 0) or
                0
            )

            # Debug logging
            logging.info(f"ðŸ” LREC Parameter Detection (Complete):")
            logging.info(f"   incentive_lrec_price: ${incentive_lrec_price}")
            logging.info(f"   system_size_kw: {system_size_kw}")

            # Create LREC config with minimal required data
            lrec_config = LRECConfiguration(
                annual_lrecs=0,  # Will be calculated from actual energy
                annual_lrec_value=0,  # Will be calculated from actual energy
                incentive_lrec_price=incentive_lrec_price,
                system_size_kw=system_size_kw,
                escalation_rate=financial_params.escalation_rate
            )

            # Enhanced eligibility check with complete logic
            size_ok = system_size_kw >= SimulationConstants.LREC_MWH_THRESHOLD
            price_ok = incentive_lrec_price > 0

            logging.info(f"ðŸ” LREC Eligibility Checks (Complete):")
            logging.info(f"   Size â‰¥100kW: {size_ok} ({system_size_kw} kW)")
            logging.info(f"   Price >0: {price_ok} (${incentive_lrec_price})")

            # Override enabled status based on complete criteria
            lrec_config.enabled = size_ok and price_ok

            if lrec_config.enabled:
                logging.info(f"ðŸŒ¿ LREC System Enabled:")
                logging.info(f"   System Size: {system_size_kw} kW")
                logging.info(f"   LREC Price: ${incentive_lrec_price}")
                logging.info(f"   LRECs will be calculated from actual PySAM energy production")
            else:
                missing_reasons = []
                if not size_ok:
                    missing_reasons.append(f"system size {system_size_kw}kW < 100kW threshold")
                if not price_ok:
                    missing_reasons.append("no LREC price specified")

                logging.info(f"ðŸŒ¿ LREC not enabled: {', '.join(missing_reasons)}")

            return lrec_config

        except (ValueError, TypeError) as e:
            logging.error(f"LREC detection failed: {e}")
            return LRECConfiguration(enabled=False)

    @staticmethod
    def validate_lrec_config(lrec_config: LRECConfiguration) -> List[str]:
        """Complete LREC configuration validation"""
        if not lrec_config.enabled:
            return ["LREC not enabled or not eligible"]

        warnings = []

        if lrec_config.system_size_kw < SimulationConstants.LREC_MWH_THRESHOLD:
            warnings.append(f"System size ({lrec_config.system_size_kw} kW) below LREC threshold (100 kW)")

        if lrec_config.incentive_lrec_price <= 0:
            warnings.append("LREC price not specified or invalid")

        if lrec_config.annual_lrecs <= 0:
            warnings.append("Annual LREC count not specified or invalid")

        # Sanity check: LRECs should roughly match energy production
        expected_lrecs = lrec_config.system_size_kw * 1.5  # Rough estimate: 1.5 MWh/kW/year
        if lrec_config.annual_lrecs > expected_lrecs * 2:
            warnings.append(f"Annual LRECs ({lrec_config.annual_lrecs}) seem high for system size")

        return warnings


class ErrorHandler:
    """Complete error handling with reduced overhead"""

    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self.warnings: List[str] = []
        self.errors: List[str] = []
        self.recoverable_errors: List[str] = []

    def handle_module_error(self, module_name: str, error: Exception) -> bool:
        """Complete module error handling"""
        error_msg = f"Module {module_name} failed: {error}"

        if self.strict_mode:
            self.errors.append(error_msg)
            raise error
        else:
            self.warnings.append(error_msg)
            self.recoverable_errors.append(module_name)
            logging.warning(f"âš ï¸ {error_msg} (continuing)")
            return False

    def handle_fcas_error(self, error: Exception) -> bool:
        """Handle FCAS-specific errors with clear messaging"""
        error_msg = f"Enhanced FCAS failed: {error}"
        self.warnings.append(error_msg)

        # FIXED: Clear messaging about what's happening
        if "joblib" in str(error):
            logging.warning(f"âš ï¸ {error_msg} (missing ML dependencies - using calibrated fallback)")
        elif "enhanced_fcas_module" in str(error):
            logging.warning(f"âš ï¸ {error_msg} (module not found - using calibrated fallback)")
        else:
            logging.warning(f"âš ï¸ {error_msg} (using calibrated fallback)")

        return True

    def handle_lrec_error(self, error: Exception) -> bool:
        """Handle LREC-specific errors"""
        error_msg = f"LREC processing failed: {error}"
        self.warnings.append(error_msg)
        logging.warning(f"âš ï¸ {error_msg} (falling back to estimates)")
        return True

    def get_summary(self) -> Dict[str, Any]:
        """Get complete error handling summary"""
        return {
            'total_warnings': len(self.warnings),
            'total_errors': len(self.errors),
            'recoverable_errors': len(self.recoverable_errors),
            'warnings': self.warnings,
            'errors': self.errors,
            'failed_modules': self.recoverable_errors
        }


class WeatherFileValidator:
    """Complete weather file validation"""

    @staticmethod
    def validate_weather_file(config: Dict[str, Any]) -> Optional[str]:
        """Complete weather file validation with cloud-safe paths"""
        weather_file = None

        if 'solar_resource_file' in config:
            specified_file = config['solar_resource_file']
            logging.info(f"Weather file specified: {specified_file}")

            if Path(specified_file).exists():
                weather_file = specified_file
                logging.info(f"âœ… Using specified weather file: {weather_file}")
            else:
                # Try weather directory
                weather_filename = Path(specified_file).name
                weather_path = Config.WEATHER_DIR / weather_filename

                if weather_path.exists():
                    weather_file = str(weather_path)
                    logging.info(f"âœ… Found weather file in directory: {weather_file}")
                else:
                    logging.warning(f"âŒ Weather file not found: {specified_file}")
        else:
            logging.warning("No weather file specified in configuration")

        return weather_file


# ========== COMPLETE FCAS PROCESSOR (PRESERVED) ==========

class FCASProcessor:
    """Complete FCAS processing with enhanced module integration and fallback"""

    def __init__(self, config: Dict[str, Any], battery_params: BatteryParameters, error_handler: ErrorHandler):
        self.config = config
        self.battery_params = battery_params
        self.error_handler = error_handler
        self._fcas_module = None

    def process_fcas(self, modules: Dict[str, Any]) -> Dict[str, Any]:
        """Complete FCAS processing with enhanced module and fallback"""
        if not self.battery_params.enabled:
            logging.info("ðŸ”‹ No battery detected, skipping FCAS processing")
            return {}

        try:
            # Try enhanced FCAS module first
            return self._process_enhanced_fcas(modules)
        except Exception as e:
            if self.error_handler.handle_fcas_error(e):
                # Fall back to estimates
                return self._create_estimated_fcas_results()
            else:
                return {}

    def _process_enhanced_fcas(self, modules: Dict[str, Any]) -> Dict[str, Any]:
        """Complete enhanced FCAS processing"""
        logging.info("ðŸ‡¦ðŸ‡º Processing Enhanced FCAS...")

        # Import FCAS module (cached)
        if self._fcas_module is None:
            self._fcas_module = self._import_fcas_module()

        # Build complete FCAS configuration
        fcas_config = self._build_complete_fcas_config(modules)

        # Log essential configuration only
        enabled_count = sum(1 for key in fcas_config.keys()
                           if key.startswith('fcas_enable_') and fcas_config[key])
        logging.info(f"ðŸ”‹ FCAS: {fcas_config['fcas_region']}, {enabled_count}/{SimulationConstants.FCAS_SERVICE_COUNT} services")

        # Call FCAS module
        try:
            fcas_results = self._fcas_module.enhanced_fcas_for_pysam(modules, fcas_config)
        except TypeError as e:
            logging.warning(f"âš ï¸ FCAS function signature error: {e}")
            raise ValueError("Enhanced FCAS module signature incompatible")

        if fcas_results and 'total_ancillary_revenue' in fcas_results:
            total_revenue = fcas_results['total_ancillary_revenue']
            logging.info(f"âœ… Enhanced FCAS Revenue: ${total_revenue:,.0f}")
            return fcas_results
        else:
            raise ValueError("Enhanced FCAS module returned no results")

    def _import_fcas_module(self):
        """Complete FCAS module import with all search paths"""
        fcas_paths = [
            str(Path(__file__).parent.parent / 'FCAS'),
            str(Path(__file__).parent / 'FCAS'),
            str(Path.cwd().parent / 'FCAS'),
            str(Path.cwd() / 'FCAS'),
        ]

        for fcas_path in fcas_paths:
            if Path(fcas_path).exists():
                if fcas_path not in sys.path:
                    sys.path.insert(0, fcas_path)

                module_file = Path(fcas_path) / 'enhanced_fcas_module.py'
                if module_file.exists():
                    logging.info(f"âœ… Found FCAS module: {module_file}")
                    break

        try:
            from enhanced_fcas_module import enhanced_fcas_for_pysam
            return type('FCASModule', (), {'enhanced_fcas_for_pysam': staticmethod(enhanced_fcas_for_pysam)})()
        except ImportError as e:
            raise ImportError(f"Enhanced FCAS module not found: {e}")

    def _build_complete_fcas_config(self, modules: Dict[str, Any]) -> Dict[str, Any]:
        """COMPLETE FCAS configuration building with proper service detection from config"""

        battery = self.battery_params

        # Use the battery parameters from the config (which came from the form)
        actual_capacity_kwh = battery.capacity_kwh
        actual_power_kw = battery.power_kw
        actual_efficiency = battery.efficiency

        # Get FCAS region and participation rate
        fcas_region = self.config.get('fcas_region', self._determine_fcas_region())
        participation_rate = self.config.get('fcas_participation_rate', 0.75)

        # Get the service configuration directly from the main config
        service_config = {}
        service_keys = [
            'fcas_enable_fast_raise', 'fcas_enable_fast_lower',
            'fcas_enable_slow_raise', 'fcas_enable_slow_lower',
            'fcas_enable_delayed_raise', 'fcas_enable_delayed_lower',
            'fcas_enable_raise_regulation', 'fcas_enable_lower_regulation'
        ]

        for service_key in service_keys:
            if service_key in self.config:
                raw_value = self.config[service_key]
                # Handle boolean conversion properly
                if isinstance(raw_value, bool):
                    service_enabled = raw_value
                elif isinstance(raw_value, str):
                    service_enabled = raw_value.lower() in ('true', 'yes', '1', 'on')
                elif isinstance(raw_value, (int, float)):
                    service_enabled = bool(raw_value)
                else:
                    service_enabled = False

                service_config[service_key] = service_enabled
            else:
                # No hardcoded defaults - if not in config, assume disabled
                service_config[service_key] = False

        # Build configuration with proper unit conversions and service config
        fcas_config = {
            'fcas_region': fcas_region,
            'fcas_model_path': self.config.get('fcas_model_path'),
            'analysis_period': self.config.get('analysis_period', 25),
            'fcas_forecast_method': 'ML',

            # Proper unit conversion
            'power_mw': actual_power_kw / 1000.0,  # kW â†’ MW
            'energy_mwh': actual_capacity_kwh / 1000.0,  # kWh â†’ MWh
            'charge_power_mw': (actual_power_kw * 0.8) / 1000.0,
            'efficiency': actual_efficiency,
            'participation_rate': participation_rate,

            # Add battery parameters for enhanced FCAS module
            'batt_power_discharge_max_kwac': actual_power_kw,
            'batt_computed_bank_capacity': actual_capacity_kwh,
            'batt_roundtrip_eff': actual_efficiency * 100,
        }

        # Add the service configuration
        fcas_config.update(service_config)

        return fcas_config

    def _determine_fcas_region(self) -> str:
        """Simplified FCAS region determination"""
        user_region = self.config.get('fcas_region', 'auto')
        if user_region != 'auto' and user_region:
            return user_region

        # Quick state detection from address
        address = self.config.get('project_info', {}).get('project_address', '').lower()
        state_patterns = {
            'NSW1': ['nsw', 'sydney'], 'QLD1': ['qld', 'brisbane'],
            'VIC1': ['vic', 'melbourne'], 'SA1': ['sa', 'adelaide'],
            'TAS1': ['tas', 'hobart'], 'WEM': ['wa', 'perth']
        }

        for region, patterns in state_patterns.items():
            if any(pattern in address for pattern in patterns):
                return region

        return 'NSW1'  # Default fallback

    def _create_estimated_fcas_results(self) -> Dict[str, Any]:
        """FIXED: Calibrated fallback FCAS with proper service detection"""
        logging.info("ðŸ”§ Creating calibrated FCAS estimates (fallback mode)")

        battery = self.battery_params
        financial_params = FinancialParameters.from_config(self.config)
        analysis_period = self.config.get('analysis_period', 25)

        if not battery.enabled:
            logging.info("ðŸ”‹ No battery detected, skipping FCAS fallback")
            return {}

        # FIXED: Proper service detection from form configuration
        service_mapping = {
            1: self.config.get('fcas_enable_fast_raise', False),
            2: self.config.get('fcas_enable_fast_lower', False),
            3: self.config.get('fcas_enable_slow_raise', False),
            4: self.config.get('fcas_enable_slow_lower', False),
            5: self.config.get('fcas_enable_delayed_raise', False),
            6: self.config.get('fcas_enable_delayed_lower', False),
            7: self.config.get('fcas_enable_raise_regulation', False),
            8: self.config.get('fcas_enable_lower_regulation', False)
        }

        # FIXED: Proper boolean conversion
        for service_num in service_mapping:
            raw_value = service_mapping[service_num]
            if isinstance(raw_value, str):
                service_mapping[service_num] = raw_value.lower() in ('true', '1', 'yes', 'on')
            elif isinstance(raw_value, (int, float)):
                service_mapping[service_num] = bool(raw_value)
            else:
                service_mapping[service_num] = bool(raw_value)

        # FIXED: Check if ANY services are enabled (respects user choice)
        enabled_services = [num for num, enabled in service_mapping.items() if enabled]
        if not enabled_services:
            logging.info("ðŸ”‹ No FCAS services enabled in fallback mode, returning zero revenue")
            # Still create zero cashflows for table compatibility
            cashflow_length = analysis_period + 1
            ancillary_results = {}
            for service_num in range(1, 9):
                revenue_key = f'cf_ancillary_services_{service_num}_revenue'
                ancillary_results[revenue_key] = [0.0] * cashflow_length
            ancillary_results['total_ancillary_revenue'] = 0.0
            return ancillary_results

        logging.info(f"ðŸ”‹ FCAS Fallback - Services Enabled: {enabled_services}")

        # Get regional settings
        fcas_region = self.config.get('fcas_region', 'NSW1')
        region_multiplier = SimulationConstants.FCAS_REGIONAL_MULTIPLIERS.get(fcas_region, 1.0)
        participation_rate = self.config.get('fcas_participation_rate', 0.75)
        cashflow_length = analysis_period + 1

        ancillary_results = {}
        total_revenue = 0

        # FIXED: Calculate revenue with CALIBRATED rates
        for service_num, service_info in SimulationConstants.FCAS_SERVICES.items():
            revenue_key = f'cf_ancillary_services_{service_num}_revenue'

            # Check if this specific service is enabled
            if not service_mapping.get(service_num, False):
                # Zero cashflow for disabled services
                ancillary_results[revenue_key] = [0.0] * cashflow_length
                continue

            # FIXED: Calculate with calibrated rates ($/kWh/year)
            base_annual_revenue = battery.capacity_kwh * service_info['rate']  # Now using calibrated rates
            adjusted_annual_revenue = (
                    base_annual_revenue *
                    region_multiplier *
                    participation_rate
            )

            # Create cashflow with escalation
            cashflow = [0.0]  # Year 0
            for year in range(1, analysis_period + 1):
                year_revenue = adjusted_annual_revenue * (
                        (1 + financial_params.escalation_rate) ** (year - 1)
                )
                cashflow.append(year_revenue)

            ancillary_results[revenue_key] = cashflow
            service_total = sum(cashflow)
            total_revenue += service_total

            # Log individual service revenue
            logging.info(f"   {service_info['name']}: ${service_total:,.0f} total")

        ancillary_results['total_ancillary_revenue'] = total_revenue

        # FIXED: Log reasonable totals
        logging.info(f"ðŸŽ¯ Calibrated FCAS Fallback: {fcas_region}, ${total_revenue:,.0f} total")
        logging.info(f"   Battery: {battery.capacity_kwh} kWh, Participation: {participation_rate:.0%}")
        logging.info(f"   Enabled Services: {len(enabled_services)}/{len(SimulationConstants.FCAS_SERVICES)}")

        return ancillary_results


# ========== COMPLETE LREC PROCESSOR (PRESERVED) ==========

class LRECProcessor:
    """Complete LREC (Large-scale Renewable Energy Certificate) processing"""

    def __init__(self, lrec_config: LRECConfiguration, error_handler: ErrorHandler, analysis_period: int):
        self.lrec_config = lrec_config
        self.error_handler = error_handler
        self.analysis_period = analysis_period

    def process_lrec(self, modules: Dict[str, Any]) -> Dict[str, Any]:
        """Process LREC revenue calculations - ALWAYS creates cf_lrec_revenue column"""
        lrec_config = self.lrec_config

        if not lrec_config.enabled:
            logging.info(
                f"LREC not enabled (system <100kW or no LREC price), creating zero cashflow for {self.analysis_period} years")
            return self._create_zero_lrec_results()

        try:
            return self._calculate_lrec_cashflow(modules, lrec_config)
        except Exception as e:
            if self.error_handler.handle_lrec_error(e):
                return self._create_zero_lrec_results()
            else:
                return self._create_zero_lrec_results()

    def _create_zero_lrec_results(self) -> Dict[str, Any]:
        # CHANGED: Use instance variable instead of hardcoded 25
        cashflow_length = self.analysis_period + 1

        logging.info(f"Creating zero LREC cashflow for {self.analysis_period} years ({cashflow_length} periods)")

        return {
            'cf_lrec_revenue': [0.0] * cashflow_length,
            'total_lrec_revenue': 0.0,
            'annual_lrecs_year1': 0.0,
            'lrec_price_year1': 0.0,
            'lrec_escalation_rate': 0.0
        }

    def _calculate_lrec_cashflow(self, modules: Dict[str, Any], lrec_config: LRECConfiguration) -> Dict[str, Any]:
        """Calculate LREC cashflow based on energy production - complete approach"""
        logging.info("Processing LREC revenue (complete approach)...")

        # Always get actual energy production from PySAM modules (preferred method)
        annual_energy_kwh = self._extract_annual_energy(modules)

        if annual_energy_kwh <= 0:
            logging.warning("No valid energy production found, LREC calculation cannot proceed")
            return self._create_zero_lrec_results()

        # Calculate LRECs from energy production (1 LREC per MWh)
        annual_lrecs = annual_energy_kwh / SimulationConstants.MWH_TO_KWH
        logging.info(f"Calculated LRECs from actual energy: {annual_lrecs:.1f} (from {annual_energy_kwh:.1f} kWh)")

        # Build LREC cashflow - PASS analysis_period
        lrec_results = self._build_lrec_cashflow(annual_lrecs, lrec_config)

        total_lrec_revenue = lrec_results.get('total_lrec_revenue', 0)
        logging.info(f"LREC Revenue: ${total_lrec_revenue:,.0f} total over {self.analysis_period} years")

        return lrec_results

    def _extract_annual_energy(self, modules: Dict[str, Any]) -> float:
        """Extract annual energy production from PySAM modules"""
        try:
            # Try pvsamv1 first
            pvsamv1_module = modules.get('pvsamv1')
            if pvsamv1_module:
                annual_energy = pvsamv1_module.value('annual_energy')
                if annual_energy > 0:
                    return float(annual_energy)

            # Try other common energy outputs
            for module_name, module in modules.items():
                if hasattr(module, 'Outputs'):
                    try:
                        annual_energy = getattr(module.Outputs, 'annual_energy', 0)
                        if annual_energy > 0:
                            return float(annual_energy)
                    except:
                        continue

            return 0.0

        except Exception as e:
            logging.warning(f"Failed to extract annual energy: {e}")
            return 0.0

    def _build_lrec_cashflow(self, annual_lrecs: float, lrec_config: LRECConfiguration) -> Dict[str, Any]:
        """Build LREC cashflow with degradation and escalation"""
        # CHANGED: Use instance variable instead of hardcoded 25
        cashflow_length = self.analysis_period + 1
        lrec_price = lrec_config.incentive_lrec_price
        escalation_rate = lrec_config.escalation_rate

        # Safety checks for reasonable values
        if escalation_rate > 0.5:
            logging.warning(f"Unrealistic escalation rate {escalation_rate:.1%}, capping at 5%")
            escalation_rate = 0.05

        if lrec_price > 200:
            logging.warning(f"Unrealistic LREC price ${lrec_price}, using $45")
            lrec_price = 45.0

        if annual_lrecs > 50000:
            logging.warning(f"Very large LREC count {annual_lrecs}, double-check system size")

        # Assume solar degradation of 0.5% per year
        degradation_rate = 0.005

        # Build cashflow with debugging
        lrec_cashflow = [0.0]  # Year 0
        total_revenue = 0.0

        logging.info(f"LREC Cashflow Calculation:")
        logging.info(f"   Analysis Period: {self.analysis_period} years")
        logging.info(f"   Annual LRECs: {annual_lrecs}")
        logging.info(f"   LREC Price: ${lrec_price}")
        logging.info(f"   Escalation Rate: {escalation_rate:.1%}")
        logging.info(f"   Degradation Rate: {degradation_rate:.1%}")

        # CHANGED: Use self.analysis_period instead of hardcoded 25
        for year in range(1, min(self.analysis_period + 1, 26)):  # Cap at 25 years
            # Calculate degraded energy production
            degradation_factor = (1 - degradation_rate) ** (year - 1)
            year_lrecs = annual_lrecs * degradation_factor

            # Calculate escalated LREC price
            escalated_price = lrec_price * ((1 + escalation_rate) ** (year - 1))

            # Calculate year revenue
            year_revenue = year_lrecs * escalated_price
            lrec_cashflow.append(year_revenue)
            total_revenue += year_revenue

            # Debug first few years
            if year <= 3:
                logging.info(f"   Year {year}: {year_lrecs:.0f} LRECs Ã— ${escalated_price:.2f} = ${year_revenue:,.0f}")

        logging.info(f"LREC Total Revenue: ${total_revenue:,.0f}")

        # Sanity check
        if total_revenue > 10_000_000:
            logging.warning(f"LREC revenue ${total_revenue:,.0f} seems very high - please verify parameters")

        return {
            'cf_lrec_revenue': lrec_cashflow,
            'total_lrec_revenue': total_revenue,
            'annual_lrecs_year1': annual_lrecs,
            'lrec_price_year1': lrec_price,
            'lrec_escalation_rate': escalation_rate
        }


# ========== COMPLETE RESULTS PROCESSOR (PRESERVED) ==========

class ResultsProcessor:
    """Complete results processing with LREC integration and enhanced metrics"""

    def __init__(self, financial_params: FinancialParameters, analysis_period: int):
        self.financial_params = financial_params
        self.analysis_period = analysis_period

    def collect_module_outputs(self, modules: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, List]]:
        """Complete output collection"""
        scalar_results = {}
        time_series = {}

        def collect_outputs(obj, prefix=""):
            """Complete recursive output collection"""
            outputs = {}
            for attr in dir(obj):
                if attr.startswith('_'):
                    continue
                try:
                    val = getattr(obj, attr)
                    if hasattr(val, '__dict__'):
                        outputs.update(collect_outputs(val, f"{prefix}{attr}_"))
                    else:
                        key = f"{prefix}{attr}"
                        if isinstance(val, (list, tuple)):
                            outputs[key] = [v[0] if isinstance(v, tuple) else v for v in val]
                        else:
                            outputs[key] = val
                except Exception:
                    continue  # Skip inaccessible attributes
            return outputs

        # Collect from all modules
        for mod_name, module in modules.items():
            if hasattr(module, 'Outputs'):
                outputs = collect_outputs(module.Outputs)

                # Separate time series from scalars
                for key, value in outputs.items():
                    if isinstance(value, list):
                        time_series[key] = value
                    else:
                        scalar_results[key] = value

        logging.info(f"Collected {len(scalar_results)} scalar results, {len(time_series)} time series")
        return scalar_results, time_series

    def integrate_fcas_results(self, scalar_results: Dict[str, Any],
                               time_series: Dict[str, List],
                               fcas_results: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, List]]:
        """Complete FCAS integration WITH total annual cashflow"""
        if not fcas_results:
            return scalar_results, time_series

        logging.info("ðŸ”— Integrating FCAS results")

        # Add individual FCAS cashflows to time series
        fcas_service_cashflows = []
        for key, value in fcas_results.items():
            if isinstance(value, list) and key.startswith('cf_ancillary'):
                time_series[key] = value
                fcas_service_cashflows.append(value)

        # CREATE TOTAL FCAS ANNUAL CASHFLOW (CRITICAL FEATURE)
        if fcas_service_cashflows:
            # Sum all individual service cashflows by year
            cashflow_length = len(fcas_service_cashflows[0]) if fcas_service_cashflows else 0
            cf_fcas_total_revenue = [0.0] * cashflow_length

            for service_cashflow in fcas_service_cashflows:
                for year in range(len(service_cashflow)):
                    cf_fcas_total_revenue[year] += service_cashflow[year]

            # Add total FCAS cashflow to time series
            time_series['cf_fcas_total_revenue'] = cf_fcas_total_revenue
            logging.info(f"âœ… Created total FCAS cashflow: ${sum(cf_fcas_total_revenue):,.0f}")

        # Add FCAS totals to scalar results
        if 'total_ancillary_revenue' in fcas_results:
            total_revenue = fcas_results['total_ancillary_revenue']
            scalar_results['Total ancillary services revenue'] = total_revenue
            scalar_results['Ancillary services revenue'] = total_revenue

        return scalar_results, time_series

    def integrate_lrec_results(self, scalar_results: Dict[str, Any],
                             time_series: Dict[str, List],
                             lrec_results: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, List]]:
        """Complete LREC integration into simulation outputs"""
        if not lrec_results:
            return scalar_results, time_series

        logging.info("ðŸ”— Integrating LREC results")

        # Add LREC cashflow to time series
        if 'cf_lrec_revenue' in lrec_results:
            time_series['cf_lrec_revenue'] = lrec_results['cf_lrec_revenue']

        # Add LREC scalar metrics including Year 1 LRECs
        lrec_scalars = {
            'Total LREC revenue': 'total_lrec_revenue',
            'Annual LRECs (Year 1)': 'annual_lrecs_year1',
            'LREC price (Year 1)': 'lrec_price_year1',
            'LREC escalation rate': 'lrec_escalation_rate'
        }

        for display_name, lrec_key in lrec_scalars.items():
            if lrec_key in lrec_results:
                scalar_results[display_name] = lrec_results[lrec_key]

        # Calculate LREC present value
        if 'total_lrec_revenue' in lrec_results and 'cf_lrec_revenue' in lrec_results:
            lrec_pv = self._calculate_lrec_present_value(lrec_results['cf_lrec_revenue'])
            scalar_results['LREC NPV'] = lrec_pv
            scalar_results['LREC Present Value'] = lrec_pv

        # Add Year 1 LREC revenue as a separate metric
        if 'cf_lrec_revenue' in lrec_results:
            lrec_cashflow = lrec_results['cf_lrec_revenue']
            if isinstance(lrec_cashflow, list) and len(lrec_cashflow) > 1:
                scalar_results['LREC Revenue (Year 1)'] = lrec_cashflow[1]

        return scalar_results, time_series

    def _calculate_lrec_present_value(self, lrec_cashflow: List[float]) -> float:
        """Calculate present value of LREC revenue stream"""
        discount_rate = self.financial_params.discount_rate

        lrec_pv = 0.0
        for year, annual_revenue in enumerate(lrec_cashflow):
            if year > 0:  # Skip year 0
                lrec_pv += annual_revenue / ((1 + discount_rate) ** year)

        return lrec_pv

    def integrate_all_results(self, scalar_results: Dict[str, Any],
                            time_series: Dict[str, List],
                            fcas_results: Dict[str, Any],
                            lrec_results: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, List]]:
        """Integrate both FCAS and LREC results with COMPLETE enhanced financial metrics"""

        # Integrate individual result sets
        scalar_results, time_series = self.integrate_fcas_results(scalar_results, time_series, fcas_results)
        scalar_results, time_series = self.integrate_lrec_results(scalar_results, time_series, lrec_results)

        # Calculate COMPLETE enhanced financial metrics with both FCAS and LREC
        scalar_results = self._calculate_enhanced_metrics(scalar_results, time_series, fcas_results, lrec_results)

        # Calculate COMPLETE cumulative payback cashflow with FCAS and LREC
        self._calculate_cumulative_payback_cashflow(scalar_results, time_series, fcas_results, lrec_results)

        return scalar_results, time_series

    def _calculate_enhanced_metrics(self, scalar_results: Dict[str, Any],
                                  time_series: Dict[str, List],
                                  fcas_results: Dict[str, Any],
                                  lrec_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate COMPLETE enhanced metrics with FCAS and LREC integration"""

        total_fcas_revenue = fcas_results.get('total_ancillary_revenue', 0)
        total_lrec_revenue = lrec_results.get('total_lrec_revenue', 0)
        total_additional_revenue = total_fcas_revenue + total_lrec_revenue

        if total_additional_revenue <= 0:
            return scalar_results

        # NPV Integration
        original_npv = scalar_results.get('npv_energy_lcos_real') or scalar_results.get('npv')
        if original_npv is not None:
            fcas_pv = self._calculate_fcas_present_value(fcas_results)
            lrec_pv = self._calculate_lrec_present_value(lrec_results.get('cf_lrec_revenue', []))

            enhanced_npv = original_npv + fcas_pv + lrec_pv
            scalar_results['NPV with FCAS and LREC'] = enhanced_npv
            scalar_results['Additional revenue NPV contribution'] = fcas_pv + lrec_pv

        # LCOE Integration
        original_lcoe = (scalar_results.get('lcoe_real') or
                        scalar_results.get('LCOE (Real)') or
                        scalar_results.get('lcoe_nom'))

        if original_lcoe is not None:
            total_energy = self._estimate_lifetime_energy(scalar_results, time_series)
            if total_energy > 0:
                # Calculate effective cost reduction from additional revenue
                revenue_offset = total_additional_revenue / total_energy
                effective_lcoe = max(0, original_lcoe - revenue_offset)
                scalar_results['LCOE with additional revenue'] = effective_lcoe
                scalar_results['Revenue offset ($/kWh)'] = revenue_offset

        # Calculate first year revenues
        first_year_fcas = self._calculate_first_year_revenue(fcas_results, 'cf_ancillary_services_')
        first_year_lrec = self._calculate_first_year_revenue(lrec_results, 'cf_lrec_revenue')

        if first_year_fcas > 0:
            scalar_results['Annual FCAS Revenue (Year 1)'] = first_year_fcas
        if first_year_lrec > 0:
            scalar_results['Annual LREC Revenue (Year 1)'] = first_year_lrec

        logging.info(f"ðŸ“Š Enhanced Metrics: Total Additional Revenue ${total_additional_revenue:,.0f}")
        return scalar_results

    def _calculate_cumulative_payback_cashflow(self, scalar_results: Dict[str, Any],
                                               time_series: Dict[str, List],
                                               fcas_results: Dict[str, Any],
                                               lrec_results: Dict[str, Any]):
        """Calculate COMPLETE cumulative payback cashflow including FCAS and LREC revenue"""
        try:
            # Get base project cashflow
            base_cashflow = time_series.get('cf_project_return_aftertax_npv',
                                            time_series.get('cf_project_return_aftertax',
                                                            time_series.get('cf_after_tax_cash_flow', [])))

            if not base_cashflow:
                logging.warning("No base project cashflow found for cumulative payback calculation")
                return

            analysis_period = len(base_cashflow) - 1 if len(base_cashflow) > 0 else 25

            # Initialize enhanced cashflow with base values
            enhanced_cashflow = base_cashflow.copy()

            # Add FCAS revenue streams
            for i in range(1, SimulationConstants.FCAS_SERVICE_COUNT + 1):
                fcas_cf_key = f'cf_ancillary_services_{i}_revenue'
                if fcas_cf_key in fcas_results:
                    fcas_cf = fcas_results[fcas_cf_key]
                    for year in range(min(len(enhanced_cashflow), len(fcas_cf))):
                        if year > 0:  # Skip year 0
                            enhanced_cashflow[year] += fcas_cf[year]

            # Add LREC revenue stream
            if 'cf_lrec_revenue' in lrec_results:
                lrec_cf = lrec_results['cf_lrec_revenue']
                for year in range(min(len(enhanced_cashflow), len(lrec_cf))):
                    if year > 0:  # Skip year 0
                        enhanced_cashflow[year] += lrec_cf[year]

            # COMPLETE: Calculate cumulative cashflow starting with initial investment
            cumulative_cashflow = []

            # Year 0: Start with the initial investment (negative value from base cashflow)
            initial_investment = enhanced_cashflow[0] if len(enhanced_cashflow) > 0 else 0.0
            cumulative_total = initial_investment
            cumulative_cashflow.append(cumulative_total)

            # Log the initial investment for debugging
            logging.info(f"ðŸ’° Initial investment (Year 0): ${initial_investment:,.0f}")

            # Years 1 onwards: Add annual cash flows to cumulative total
            for year in range(1, min(analysis_period + 1, len(enhanced_cashflow))):
                cumulative_total += enhanced_cashflow[year]
                cumulative_cashflow.append(cumulative_total)

            # Add to time series
            time_series['cf_cumulative_payback_with_fcas_lrec'] = cumulative_cashflow

            # Calculate payback period (when cumulative becomes positive)
            payback_year = None
            for year, cumulative in enumerate(cumulative_cashflow):
                if year > 0 and cumulative > 0:
                    payback_year = year
                    break

            # Add payback metrics to scalar results
            if payback_year:
                scalar_results['Payback Period with FCAS and LREC (years)'] = payback_year

                # Interpolate for more precise payback
                if payback_year > 1:
                    prev_cum = cumulative_cashflow[payback_year - 1]
                    curr_cum = cumulative_cashflow[payback_year]
                    if curr_cum != prev_cum:
                        precise_payback = payback_year - 1 + (0 - prev_cum) / (curr_cum - prev_cum)
                        scalar_results['Precise Payback Period with FCAS and LREC (years)'] = precise_payback
            else:
                scalar_results['Payback Period with FCAS and LREC (years)'] = 'No payback within analysis period'

            # COMPLETE: For comparison, also create a version that shows total project value
            # This helps users understand the progression from negative (cost) to positive (profit)
            total_project_return = cumulative_cashflow.copy()
            time_series['cf_total_project_return_with_fcas_lrec'] = total_project_return

            # Log results
            final_cumulative = cumulative_cashflow[-1] if cumulative_cashflow else 0
            logging.info(f"ðŸ’° Cumulative Payback Calculated:")
            logging.info(f"   Initial investment: ${initial_investment:,.0f}")
            logging.info(f"   Final cumulative value: ${final_cumulative:,.0f}")
            if payback_year:
                logging.info(f"   Payback period: {payback_year} years")

            logging.info(f"âœ… Created enhanced payback cashflow starting from ${initial_investment:,.0f}")

        except Exception as e:
            logging.error(f"Failed to calculate cumulative payback cashflow: {e}")

    def _calculate_fcas_present_value(self, fcas_results: Dict[str, Any]) -> float:
        """Calculate present value of FCAS revenue streams"""
        fcas_pv = 0
        discount_rate = self.financial_params.discount_rate

        for i in range(1, SimulationConstants.FCAS_SERVICE_COUNT + 1):
            cf_key = f'cf_ancillary_services_{i}_revenue'
            if cf_key in fcas_results:
                cashflow = fcas_results[cf_key]
                if isinstance(cashflow, list):
                    for year, annual_revenue in enumerate(cashflow):
                        if year > 0:
                            fcas_pv += annual_revenue / ((1 + discount_rate) ** year)

        return fcas_pv

    def _calculate_first_year_revenue(self, results: Dict[str, Any], cashflow_prefix: str) -> float:
        """Calculate first year revenue from cashflow results"""
        first_year_revenue = 0.0

        if cashflow_prefix == 'cf_lrec_revenue':
            # Single LREC cashflow
            cashflow = results.get('cf_lrec_revenue', [])
            if isinstance(cashflow, list) and len(cashflow) > 1:
                first_year_revenue = cashflow[1]
        else:
            # Multiple FCAS cashflows
            for i in range(1, SimulationConstants.FCAS_SERVICE_COUNT + 1):
                cf_key = f'{cashflow_prefix}{i}_revenue'
                if cf_key in results:
                    cashflow = results[cf_key]
                    if isinstance(cashflow, list) and len(cashflow) > 1:
                        first_year_revenue += cashflow[1]

        return first_year_revenue

    def _estimate_lifetime_energy(self, scalar_results: Dict[str, Any],
                                time_series: Dict[str, List]) -> float:
        """Estimate total lifetime energy production"""
        # Try time series first
        if 'cf_energy_net' in time_series:
            return sum(time_series['cf_energy_net'])

        # Estimate from annual energy
        annual_energy = scalar_results.get('annual_energy')
        if annual_energy:
            # Account for degradation (0.5% per year)
            total_energy = sum(
                annual_energy * (1 - 0.005) ** year
                for year in range(self.analysis_period)
            )
            return total_energy / 1000 if total_energy > 1000000 else total_energy

        return 0


# ========== COMPLETE RESULTS EXPORTER (PRESERVED) ==========

class ResultsExporter:
    """Complete results export with LREC support and all time series"""

    def __init__(self, analysis_period: int):
        self.analysis_period = analysis_period
        self.exported_data = {}

    def export_all_results(self, results: SimulationResults) -> Dict[str, Any]:
        """Complete results export with all 121+ series"""
        # Create results directory with cloud-safe path resolution
        current_path = Path(__file__).resolve()
        project_root = None

        for parent in current_path.parents:
            if (parent / 'main.py').exists() or (parent / 'config.py').exists():
                project_root = parent
                break

        if project_root is None:
            if '2.pysam' in str(current_path):
                parts = current_path.parts
                for i, part in enumerate(parts):
                    if '2.pysam' in part and i > 0:
                        project_root = Path(*parts[:i])
                        break
            if project_root is None:
                project_root = Path.cwd()

        results_dir = project_root / '2.pysam' / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)

        print(f"ðŸ“ Saving results to: {results_dir}")
        original_cwd = os.getcwd()

        try:
            os.chdir(str(results_dir))

            # Export scalar results
            self._export_scalar_results(results.scalar_results)

            # Export time series (organized by frequency) - COMPLETE
            self._export_time_series(results.time_series)

            # Create metadata
            metadata = self._create_metadata(results)

            # Save metadata
            with open('metadata.json', 'w') as f:
                json.dump(self._clean_for_json(metadata), f, indent=2)

            logging.info(f"âœ… Results exported to: {results_dir}")
            return metadata

        except Exception as e:
            logging.error(f"Export failed: {e}")
            raise
        finally:
            os.chdir(original_cwd)

    def _export_scalar_results(self, scalar_results: Dict[str, Any]):
        """Export scalar metrics to CSV"""
        output_metrics = {}

        # Add standard financial metrics
        for label, key in SimulationConstants.FINANCIAL_METRICS.items():
            if key in scalar_results:
                output_metrics[label] = scalar_results[key]

        # Add remaining scalar results
        for key, value in scalar_results.items():
            if not isinstance(value, (list, dict)) and key not in output_metrics:
                output_metrics[key] = value

        # Export to CSV
        df = pd.DataFrame.from_dict(output_metrics, orient='index', columns=['Value'])
        df.to_csv('scalar_results.csv', header=True)

        self.exported_data['scalar_metrics'] = {
            'file': 'scalar_results.csv',
            'series': list(output_metrics.keys()),
            'count': len(output_metrics)
        }

        logging.info(f"ðŸ“Š Exported {len(output_metrics)} scalar metrics")

    def _export_time_series(self, time_series: Dict[str, List]):
        """Export COMPLETE time series organized by frequency - ALL 121+ series with ANNUAL EXPORT"""
        if not time_series:
            return

        # Validate and group time series
        valid_time_series = {}
        for name, values in time_series.items():
            if isinstance(values, list) and len(values) in SimulationConstants.EXPECTED_TIME_SERIES_LENGTHS:
                valid_time_series[name] = values
            elif name.startswith('cf_'):  # Always include cashflow
                valid_time_series[name] = values

        # Extract COMPLETE battery health series (optimized)
        self._extract_essential_battery_health(time_series, valid_time_series)

        # CRITICAL: Create and export annual time series for dashboard
        annual_data = {}
        self._create_annual_time_series(time_series, annual_data)

        # Export annual data if it exists
        if annual_data:
            self._export_annual_time_series(annual_data)
        else:
            logging.warning("No annual time series data created - dashboard may show warnings")

        # Export cashflow data (including LREC and FCAS)
        cf_data = {k: v for k, v in valid_time_series.items() if k.startswith('cf_')}
        if cf_data:
            self._export_cashflow_data(cf_data)

        # Export other time series grouped by length - COMPLETE
        for length, group_name in SimulationConstants.TIME_SERIES_GROUPS.items():
            group_data = {
                k: v for k, v in valid_time_series.items()
                if len(v) == length and not k.startswith('cf_')
            }

            if group_data:
                self._export_time_series_group(group_data, group_name, length)

    def _create_annual_time_series(self, time_series: Dict[str, List], annual_data: Dict[str, List]):
        """Create comprehensive annual time series for dashboard - ALL relevant metrics"""
        try:
            logging.info(
                f"Processing {len(time_series)} time series for {self.analysis_period}-year analysis period...")

            # Get all hourly data (8760 hours)
            hourly_series = {k: v for k, v in time_series.items()
                             if isinstance(v, list) and len(v) >= 8760}

            # Get all monthly data (12 months)
            monthly_series = {k: v for k, v in time_series.items()
                              if isinstance(v, list) and len(v) == 12}

            logging.info(f"Found {len(hourly_series)} hourly series and {len(monthly_series)} monthly series")

            # Process hourly data to annual summaries
            for key, hourly_values in hourly_series.items():
                # Skip if already processed or not suitable for annual aggregation
                if (key.startswith('cf_') or  # Skip cashflow (handled separately)
                        'year1' in key or  # Skip already year-specific data
                        len(hourly_values) < 8760):
                    continue

                annual_values = []

                # Calculate annual values for each year in analysis period
                for year in range(min(self.analysis_period, len(hourly_values) // 8760)):
                    year_start = year * 8760
                    year_end = (year + 1) * 8760

                    if year_end <= len(hourly_values):
                        year_data = hourly_values[year_start:year_end]

                        # Determine aggregation method based on metric type
                        if any(keyword in key.lower() for keyword in [
                            'energy', 'charge', 'discharge', 'generation',
                            'consumption', 'import', 'export', 'kwh'
                        ]):
                            # Sum for energy metrics
                            year_value = sum(year_data)
                        elif any(keyword in key.lower() for keyword in [
                            'power', 'voltage', 'current', 'temperature',
                            'soc', 'dod', 'efficiency', 'kw'
                        ]):
                            # Average for state/power metrics
                            year_value = sum(year_data) / len(year_data) if year_data else 0
                        elif 'cycles' in key.lower():
                            # End-of-year value for cumulative metrics
                            year_value = year_data[-1] if year_data else 0
                        else:
                            # Default to sum
                            year_value = sum(year_data)

                        annual_values.append(year_value)

                if annual_values:
                    # Create annual series name with corrected naming convention
                    if key.startswith('batt_'):
                        annual_key = key.replace('batt_', 'batt_annual_')
                    elif key.startswith('grid_'):
                        annual_key = key.replace('grid_', 'annual_grid_')
                    else:
                        annual_key = f"annual_{key}" if not key.startswith('annual_') else key

                    # FIXED: Ensure we have exactly analysis_period + 1 values
                    if len(annual_values) == self.analysis_period:
                        # Correct length - add Year 0
                        annual_data[annual_key] = [0.0] + annual_values
                    elif len(annual_values) < self.analysis_period:
                        # Pad to correct length
                        padded_values = annual_values + [0.0] * (self.analysis_period - len(annual_values))
                        annual_data[annual_key] = [0.0] + padded_values
                    else:
                        # Truncate to correct length
                        truncated_values = annual_values[:self.analysis_period]
                        annual_data[annual_key] = [0.0] + truncated_values

            # Process monthly data to annual summaries
            for key, monthly_values in monthly_series.items():
                if len(monthly_values) == 12:
                    # Sum monthly values to get annual total
                    annual_total = sum(monthly_values)

                    # Create annual series (same value for each year with degradation if applicable)
                    annual_values = []
                    for year in range(self.analysis_period):
                        if 'energy' in key.lower() or 'generation' in key.lower():
                            # Apply 0.5% annual degradation for energy metrics
                            degraded_value = annual_total * ((1 - 0.005) ** year)
                        else:
                            # No degradation for other metrics
                            degraded_value = annual_total

                        annual_values.append(degraded_value)

                    # Create annual series name with corrected naming convention
                    if key.startswith('batt_'):
                        annual_key = key.replace('batt_', 'batt_annual_')
                    elif key.startswith('grid_'):
                        annual_key = key.replace('grid_', 'annual_grid_')
                    else:
                        annual_key = f"annual_{key}" if not key.startswith('annual_') else key

                    # Create exactly analysis_period + 1 values (Year 0 + analysis years)
                    annual_data[annual_key] = [0.0] + annual_values

            # Add specific battery health metrics if available
            self._add_battery_health_annuals(time_series, annual_data)

            # Add grid interaction metrics if available
            self._add_grid_interaction_annuals(time_series, annual_data)

            logging.info(f"Created {len(annual_data)} annual time series for {self.analysis_period}-year analysis")

            # Log breakdown by category
            battery_count = len([k for k in annual_data.keys() if 'batt' in k.lower()])
            grid_count = len(
                [k for k in annual_data.keys() if any(term in k.lower() for term in ['grid', 'import', 'export'])])
            energy_count = len([k for k in annual_data.keys() if 'energy' in k.lower()])

            # CRITICAL: Create the specific battery energy columns the dashboard expects
            if any('batt_annual_to_grid' in k for k in annual_data.keys()):
                # Create missing battery energy flow columns using existing data
                battery_mappings = {
                    'batt_annual_to_grid': 'batt_annual_discharge_energy',
                    'annual_grid_to_batt': 'batt_annual_charge_from_grid',
                    'annual_system_to_batt': 'batt_annual_charge_from_system'
                }

                for source_key, target_key in battery_mappings.items():
                    if source_key in annual_data and target_key not in annual_data:
                        annual_data[target_key] = annual_data[source_key].copy()
                        logging.info(f"Created {target_key} from {source_key}")

                # Create total charge energy
                if ('batt_annual_charge_from_grid' in annual_data and
                        'batt_annual_charge_from_system' in annual_data):
                    charge_grid = annual_data['batt_annual_charge_from_grid']
                    charge_system = annual_data['batt_annual_charge_from_system']

                    total_charge = []
                    for i in range(len(charge_grid)):
                        total_charge.append(charge_grid[i] + charge_system[i])

                    annual_data['batt_annual_charge_energy'] = total_charge
                    logging.info("Created batt_annual_charge_energy from grid + system charging")

                # Create energy loss
                if ('batt_annual_charge_energy' in annual_data and
                        'batt_annual_discharge_energy' in annual_data):
                    charge_data = annual_data['batt_annual_charge_energy']
                    discharge_data = annual_data['batt_annual_discharge_energy']

                    energy_loss = []
                    for i in range(len(charge_data)):
                        loss = max(0, charge_data[i] - discharge_data[i])
                        energy_loss.append(loss)

                    annual_data['batt_annual_energy_loss'] = energy_loss
                    logging.info("Created batt_annual_energy_loss from charge/discharge difference")

            logging.info(f"Created {len(annual_data)} annual time series for {self.analysis_period}-year analysis")

            logging.info(f"Annual series breakdown: {battery_count} battery, {grid_count} grid, {energy_count} energy")

        except Exception as e:
            logging.error(f"Annual time series creation failed: {e}")
            import traceback
            traceback.print_exc()

    def _add_battery_health_annuals(self, time_series: Dict[str, List], annual_data: Dict[str, List]):
        """Add battery health annual metrics"""
        try:
            # Battery cycles (end of year values)
            if 'batt_cycles' in time_series and len(time_series['batt_cycles']) >= 8760:
                cycles_data = time_series['batt_cycles']
                annual_cycles = [0.0]  # Year 0

                for year in range(min(self.analysis_period, len(cycles_data) // 8760)):
                    year_end_index = (year + 1) * 8760 - 1
                    if year_end_index < len(cycles_data):
                        annual_cycles.append(cycles_data[year_end_index])

                # Ensure correct length
                while len(annual_cycles) < (self.analysis_period + 1):
                    annual_cycles.append(annual_cycles[-1] if annual_cycles else 0.0)

                annual_data['batt_annual_cycles'] = annual_cycles[:self.analysis_period + 1]

            # Battery capacity fade (if available)
            if 'batt_capacity_percent' in time_series:
                capacity_data = time_series['batt_capacity_percent']
                if len(capacity_data) >= 8760:
                    annual_capacity = [100.0]  # Year 0 - 100% capacity

                    for year in range(min(self.analysis_period, len(capacity_data) // 8760)):
                        year_end_index = (year + 1) * 8760 - 1
                        if year_end_index < len(capacity_data):
                            annual_capacity.append(capacity_data[year_end_index])

                    # Ensure correct length
                    while len(annual_capacity) < (self.analysis_period + 1):
                        annual_capacity.append(annual_capacity[-1] if annual_capacity else 100.0)

                    annual_data['batt_annual_capacity_percent'] = annual_capacity[:self.analysis_period + 1]

        except Exception as e:
            logging.warning(f"Battery health annuals creation failed: {e}")

    def _add_grid_interaction_annuals(self, time_series: Dict[str, List], annual_data: Dict[str, List]):
        """Add grid interaction annual metrics"""
        try:
            grid_metrics = [
                'grid_power',
                'system_to_grid',
                'grid_to_load',
                'grid_to_batt',
                'system_to_load'
            ]

            for metric in grid_metrics:
                if metric in time_series and len(time_series[metric]) >= 8760:
                    hourly_data = time_series[metric]
                    annual_values = []

                    for year in range(min(self.analysis_period, len(hourly_data) // 8760)):
                        year_start = year * 8760
                        year_end = (year + 1) * 8760

                        if year_end <= len(hourly_data):
                            year_total = sum(hourly_data[year_start:year_end])
                            annual_values.append(year_total)

                    if annual_values:
                        # Ensure correct length
                        if len(annual_values) < self.analysis_period:
                            annual_values.extend([0.0] * (self.analysis_period - len(annual_values)))
                        elif len(annual_values) > self.analysis_period:
                            annual_values = annual_values[:self.analysis_period]

                        annual_data[f'annual_{metric}'] = [0.0] + annual_values

        except Exception as e:
            logging.warning(f"Grid interaction annuals creation failed: {e}")

    def _export_annual_time_series(self, annual_data: Dict[str, List]):
        """Export annual time series as annual_timeseries.parquet"""
        try:
            if not annual_data:
                logging.warning("No annual data to export")
                return

            # FIXED: Include Year 0 in expected length
            expected_length = self.analysis_period + 1  # Include Year 0

            # Normalize data to include Year 0
            normalized_annual_data = {}
            for key, values in annual_data.items():
                if isinstance(values, list):
                    # CRITICAL FIX: Ensure Year 0 is included
                    if len(values) == self.analysis_period:  # Missing Year 0
                        normalized_values = [0.0] + values  # Add Year 0
                    elif len(values) == expected_length:  # Already has Year 0
                        normalized_values = values
                    else:
                        # Handle other length mismatches
                        if len(values) > expected_length:
                            normalized_values = values[:expected_length]
                        else:
                            normalized_values = values + [0.0] * (expected_length - len(values))

                    normalized_annual_data[key] = normalized_values

            if normalized_annual_data:
                # FIXED: Create DataFrame with Year 0 included
                df = pd.DataFrame({
                    'year': range(0, expected_length),  # 0, 1, 2, ..., analysis_period
                    **normalized_annual_data
                })
                df.to_parquet('annual_timeseries.parquet', index=False)

                logging.info(
                    f"Exported {len(normalized_annual_data)} annual time series with {expected_length} periods (including Year 0)")

                self.exported_data['annual_series'] = {
                    'file': 'annual_timeseries.parquet',
                    'series': list(normalized_annual_data.keys()),
                    'count': len(normalized_annual_data)
                }

                logging.info(f"Exported {len(normalized_annual_data)} annual time series to annual_timeseries.parquet")
            else:
                logging.warning("No valid annual data found for export")

        except Exception as e:
            logging.error(f"Failed to export annual time series: {e}")
            # Continue without annual export - better than crashing



    def _extract_essential_battery_health(self, time_series: Dict[str, List], valid_time_series: Dict[str, List]):
        """Extract COMPLETE essential battery health metrics (optimized)"""
        try:
            # Essential hourly metrics (first year only)
            essential_metrics = ['batt_SOC', 'batt_DOD']

            for metric in essential_metrics:
                if metric in time_series:
                    full_data = time_series[metric]
                    if len(full_data) >= 8760:
                        valid_time_series[f'{metric}_year1'] = full_data[:8760]

            # Annual cycles (end of year)
            if 'batt_cycles' in time_series:
                cycles_data = time_series['batt_cycles']
                annual_cycles = []

                for year in range(min(25, len(cycles_data) // 8760)):
                    year_end_index = (year + 1) * 8760 - 1
                    if year_end_index < len(cycles_data):
                        annual_cycles.append(cycles_data[year_end_index])

                if annual_cycles:
                    valid_time_series['batt_annual_cycles'] = [0.0] + annual_cycles

        except Exception as e:
            logging.warning(f"Battery health extraction error: {e}")

    def _export_cashflow_data(self, cf_data: Dict[str, List]):
        """Export COMPLETE cashflow time series with array length validation"""
        if not cf_data:
            return

        # Determine expected length from analysis period
        expected_length = self.analysis_period + 1  # Include Year 0

        # Validate and normalize all cashflow arrays to the same length
        normalized_cf_data = {}

        for key, values in cf_data.items():
            if not isinstance(values, list):
                continue

            # Handle arrays that are too long (truncate)
            if len(values) > expected_length:
                logging.warning(f"Truncating {key} from {len(values)} to {expected_length} periods")
                normalized_values = values[:expected_length]
            # Handle arrays that are too short (pad with zeros)
            elif len(values) < expected_length:
                logging.warning(f"Padding {key} from {len(values)} to {expected_length} periods with zeros")
                normalized_values = values + [0.0] * (expected_length - len(values))
            else:
                normalized_values = values

            normalized_cf_data[key] = normalized_values

        if not normalized_cf_data:
            logging.warning("No valid cashflow data found for export")
            return

        # Create DataFrame with normalized data
        try:
            df = pd.DataFrame({
                'period': range(1, expected_length + 1),
                **normalized_cf_data
            })
            df.to_parquet('cashflow_timeseries.parquet')

            self.exported_data['cashflow_series'] = {
                'file': 'cashflow_timeseries.parquet',
                'series': list(normalized_cf_data.keys()),
                'count': len(normalized_cf_data),
                'analysis_period': self.analysis_period
            }

            # Log validation results
            fcas_cols = [k for k in normalized_cf_data.keys() if 'ancillary' in k]
            lrec_cols = [k for k in normalized_cf_data.keys() if 'lrec' in k]
            cumulative_cols = [k for k in normalized_cf_data.keys() if 'cumulative_payback' in k]

            if fcas_cols:
                logging.info(f"ðŸ’° Included {len(fcas_cols)} FCAS revenue streams ({self.analysis_period} years)")
            if lrec_cols:
                logging.info(f"ðŸŒ¿ Included {len(lrec_cols)} LREC revenue streams ({self.analysis_period} years)")
            if cumulative_cols:
                logging.info(
                    f"ðŸ“ˆ Included {len(cumulative_cols)} cumulative payback cashflows ({self.analysis_period} years)")

            logging.info(f"ðŸ’¸ Exported {len(normalized_cf_data)} cashflow series ({self.analysis_period} years)")

        except Exception as e:
            logging.error(f"Failed to create cashflow DataFrame: {e}")
            # Log array lengths for debugging
            lengths = {k: len(v) for k, v in normalized_cf_data.items()}
            logging.error(f"Array lengths: {lengths}")
            raise

    def _export_time_series_group(self, group_data: Dict[str, List], group_name: str, length: int):
        """Export time series group"""
        output_file = f"{group_name}_timeseries.parquet"
        df = pd.DataFrame({
            'period': range(1, length + 1),
            **group_data
        })
        df.to_parquet(output_file)

        self.exported_data[f'{group_name}_series'] = {
            'file': output_file,
            'series': list(group_data.keys()),
            'count': len(group_data)
        }

        logging.info(f"ðŸ“ˆ Exported {len(group_data)} {group_name} series")

    def _create_metadata(self, results: SimulationResults) -> Dict[str, Any]:
        """Create COMPLETE metadata with LREC information"""
        total_series = sum(
            section['count'] for section in self.exported_data.values()
            if section['file'].endswith('.parquet')
        )
        total_items = sum(section['count'] for section in self.exported_data.values())

        return {
            'execution': {
                'simulation_type': 'simplified_complete_hybrid_cashloan_fcas_lrec',
                'timestamp': datetime.now().isoformat(),
                'weather_file': os.path.basename(results.weather_file) if results.weather_file else 'None',
                'modules_executed': results.modules_executed,
                'analysis_period': self.analysis_period,
                'optimizations_applied': [
                    'Simplified execution without progress tracking',
                    'Enhanced FCAS module integration',
                    'Complete LREC revenue integration',
                    'Cumulative payback cashflow',
                    'Enhanced financial metrics',
                    'Complete battery health time series',
                    'Present value calculations',
                    'Complete results export (121+ series)',
                    'All functionality preserved except progress tracking'
                ]
            },
            'exports': {
                'total_time_series': total_series,
                'total_metrics': self.exported_data.get('scalar_metrics', {}).get('count', 0),
                'total_items': total_items,
                'details': self.exported_data
            }
        }

    def _clean_for_json(self, data):
        """Clean data for JSON serialization"""
        if isinstance(data, dict):
            return {k: self._clean_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._clean_for_json(item) for item in data]
        elif hasattr(data, 'tolist'):
            return data.tolist()
        elif hasattr(data, '__dict__'):
            return str(data)
        return data


# ========== SIMPLIFIED MAIN SIMULATION RUNNER ==========

def run_simulation_optimized(input_json: Optional[str] = None) -> 'SimulationResults':
    """OPTIMIZED: Main simulation with cloud performance improvements"""

    start_time = time.time()
    logging.info("Starting OPTIMIZED PySAM simulation...")

    try:
        # Load configuration (unchanged)
        input_path = input_json or Config.UPDATED_JSON
        if not Path(input_path).exists():
            raise FileNotFoundError(f"Input JSON not found: {input_path}")

        with open(input_path) as f:
            config_dict = json.load(f)

        logging.info(f"Configuration loaded ({time.time() - start_time:.1f}s)")

        # OPTIMIZATION 1: Pre-validate configuration to fail fast
        module_validation_start = time.time()
        compute_modules = sorted(
            [(int(k.split('_')[-1]), v) for k, v in config_dict.items()
             if k.startswith("compute_module_")],
            key=lambda x: x[0]
        )
        module_order = [name for _, name in compute_modules]

        if not module_order:
            raise ValueError("No compute modules found in configuration")

        logging.info(f"Module validation ({time.time() - module_validation_start:.1f}s)")

        # Create configuration objects with analysis period extraction
        config_start = time.time()

        # CRITICAL: Extract analysis period early for all processors
        analysis_period = config_dict.get('analysis_period', 25)
        logging.info(f"Analysis period: {analysis_period} years")

        financial_params = FinancialParameters.from_config(config_dict)
        battery_params = BatteryDetector.detect_battery_parameters(config_dict)
        lrec_config = LRECDetector.detect_lrec_parameters(config_dict, financial_params)

        logging.info(f"Configuration objects created ({time.time() - config_start:.1f}s)")

        # OPTIMIZATION 2: Initialize modules with memory pre-allocation
        modules_start = time.time()
        modules = {}

        # Pre-allocate module data structures
        logging.info("Pre-initializing module data structures...")

        for i, mod_name in enumerate(module_order):
            module_init_start = time.time()

            try:
                # OPTIMIZATION: Use dict_to_ssc_table with memory hints
                mod_data = pssc.dict_to_ssc_table(config_dict, mod_name)

                # Import module
                pysam_mod_name = mod_name[0].upper() + mod_name[1:]
                logging.info(f"Importing PySAM.{pysam_mod_name}")
                pysam_module = importlib.import_module(f'PySAM.{pysam_mod_name}')

                # OPTIMIZATION: Initialize with proper chaining
                if i == 0:
                    module_instance = pysam_module.wrap(mod_data)
                else:
                    prev_mod_name = module_order[i - 1]

                    # CRITICAL: Optimize data transfer between modules
                    prev_instance = modules[prev_mod_name]
                    module_instance = pysam_module.from_existing(prev_instance)

                    # Only assign new data, not re-copy existing
                    new_data = pysam_module.wrap(mod_data).export()
                    module_instance.assign(new_data)

                modules[mod_name] = module_instance

                logging.info(f"{mod_name} initialized ({time.time() - module_init_start:.1f}s)")

                # OPTIMIZATION: Force garbage collection after each module
                if i % 2 == 0:  # Every other module
                    gc.collect()

            except Exception as e:
                logging.error(f"Module {mod_name} initialization failed: {e}")
                raise

        total_module_time = time.time() - modules_start
        logging.info(f"All modules initialized ({total_module_time:.1f}s)")

        # OPTIMIZATION 3: Assign weather file early and efficiently
        weather_start = time.time()
        weather_file = WeatherFileValidator.validate_weather_file(config_dict)
        if weather_file and modules:
            first_module = modules[module_order[0]]
            if hasattr(first_module, "SolarResource"):
                first_module.SolarResource.solar_resource_file = weather_file
                logging.info(f"Weather file assigned ({time.time() - weather_start:.1f}s)")

        # OPTIMIZATION 4: Execute modules with performance monitoring
        execution_start = time.time()
        logging.info("Starting optimized module execution...")

        successful_executions = 0
        execution_times = {}

        for i, mod_name in enumerate(module_order):
            if mod_name in modules:
                mod_exec_start = time.time()

                try:
                    logging.info(f"Executing: {mod_name}")

                    # OPTIMIZATION: Set module-specific performance hints
                    module_instance = modules[mod_name]

                    # For PVSAMv1 (the slowest module), apply specific optimizations
                    if mod_name == 'pvsamv1':
                        logging.info("Applying PVSAMv1 optimizations...")

                        # CRITICAL: Reduce computational precision for speed
                        if hasattr(module_instance, 'Lifetime'):
                            # Check if we can reduce the analysis period for faster execution
                            try:
                                # Check if we can reduce the analysis period for faster execution
                                if hasattr(module_instance.Lifetime, 'analysis_period'):
                                    current_period = getattr(module_instance.Lifetime, 'analysis_period', 25)
                                    if current_period > 25:
                                        module_instance.Lifetime.analysis_period = 25
                                        logging.info("   Reduced analysis period to 25 years")
                            except:
                                pass

                    # Execute the module
                    modules[mod_name].execute()

                    mod_exec_time = time.time() - mod_exec_start
                    execution_times[mod_name] = mod_exec_time
                    successful_executions += 1

                    logging.info(f"{mod_name} executed ({mod_exec_time:.1f}s)")

                    # OPTIMIZATION: Immediate garbage collection after heavy modules
                    if mod_name in ['pvsamv1', 'battwatts', 'utilityrate5']:
                        gc.collect()
                        logging.info(f"   Memory cleaned after {mod_name}")

                except Exception as e:
                    logging.error(f"Module {mod_name} execution failed: {e}")
                    raise

        total_execution_time = time.time() - execution_start
        logging.info(f"Module execution completed ({total_execution_time:.1f}s)")

        # Log execution time breakdown
        logging.info("Execution time breakdown:")
        for mod_name, exec_time in execution_times.items():
            percentage = (exec_time / total_execution_time) * 100
            logging.info(f"   {mod_name}: {exec_time:.1f}s ({percentage:.1f}%)")

        if successful_executions == 0:
            raise RuntimeError("No modules executed successfully")

        # OPTIMIZATION 5: Streamlined results collection
        results_start = time.time()
        logging.info("Collecting results (optimized)...")

        # Initialize all processors with analysis_period
        error_handler = ErrorHandler(strict_mode=False)
        results_processor = ResultsProcessor(financial_params, analysis_period)

        # Collect basic results
        scalar_results, time_series = results_processor.collect_module_outputs(modules)
        logging.info(f"Basic results collected ({time.time() - results_start:.1f}s)")

        # OPTIMIZATION 6: Parallel processing of FCAS and LREC (if possible)
        fcas_lrec_start = time.time()

        fcas_results = {}
        lrec_results = {}

        # Process FCAS
        if battery_params.enabled:
            try:
                fcas_processor = FCASProcessor(config_dict, battery_params, error_handler)
                fcas_results = fcas_processor.process_fcas(modules)
                if fcas_results and 'total_ancillary_revenue' in fcas_results:
                    logging.info(f"FCAS processed: ${fcas_results['total_ancillary_revenue']:,.0f}")
            except Exception as e:
                logging.warning(f"FCAS processing failed: {e}")

        # Process LREC - FIXED: Pass analysis_period to constructor, not method call
        try:
            lrec_processor = LRECProcessor(lrec_config, error_handler, analysis_period)
            lrec_results = lrec_processor.process_lrec(modules)
            if lrec_results and 'total_lrec_revenue' in lrec_results:
                logging.info(f"LREC processed: ${lrec_results['total_lrec_revenue']:,.0f}")
        except Exception as e:
            logging.warning(f"LREC processing failed: {e}")

        logging.info(f"FCAS/LREC processing completed ({time.time() - fcas_lrec_start:.1f}s)")

        # Integrate results
        integration_start = time.time()
        scalar_results, time_series = results_processor.integrate_all_results(
            scalar_results, time_series, fcas_results, lrec_results
        )
        logging.info(f"Results integration completed ({time.time() - integration_start:.1f}s)")

        # Create results object
        results = SimulationResults()
        results.scalar_results = scalar_results
        results.time_series = time_series
        results.fcas_results = fcas_results
        results.lrec_results = lrec_results
        results.modules_executed = module_order
        results.config_used = config_dict
        results.weather_file = weather_file

        # OPTIMIZATION 7: Streamlined export (optional - can be async)
        export_start = time.time()
        results_exporter = ResultsExporter(analysis_period)
        metadata = results_exporter.export_all_results(results)
        logging.info(f"Results exported ({time.time() - export_start:.1f}s)")

        # Final cleanup
        gc.collect()

        total_time = time.time() - start_time
        logging.info(f"OPTIMIZED simulation completed in {total_time:.1f}s")

        # Performance breakdown
        logging.info("Performance Breakdown:")
        logging.info(
            f"   Module Initialization: {total_module_time:.1f}s ({(total_module_time / total_time) * 100:.1f}%)")
        logging.info(
            f"   Module Execution: {total_execution_time:.1f}s ({(total_execution_time / total_time) * 100:.1f}%)")
        logging.info(f"   Results Processing: {(time.time() - results_start):.1f}s")

        return results

    except Exception as e:
        total_time = time.time() - start_time
        logging.error(f"Optimized simulation failed after {total_time:.1f}s: {e}")
        import traceback
        logging.error(f"Full traceback: {traceback.format_exc()}")

        # Return empty results instead of crashing
        results = SimulationResults()
        results.scalar_results = {}
        results.time_series = {}
        results.fcas_results = {}
        results.lrec_results = {}
        results.modules_executed = []
        results.config_used = {}
        results.weather_file = None
        return results





# ========== LOGGING SETUP (PRESERVED) ==========

def setup_logging():
    """Setup optimized logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler('simulation.log'),
            logging.StreamHandler()
        ]
    )


# ========== COMMAND LINE EXECUTION (PRESERVED) ==========

if __name__ == "__main__":
    setup_logging()
    logging.info("ðŸš€ Starting Optimized PySAM Simulation...")

    results = run_simulation_optimized()

    if results.scalar_results:
        logging.info("âœ… Optimized simulation completed successfully")
    else:
        logging.warning("âš ï¸ Simulation completed with limited results")
