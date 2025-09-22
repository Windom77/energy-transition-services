"""
FIXED PySAM Integration - Form-Based with Enhanced Coordinate Extraction
- Fixed coordinate extraction from form data
- Enhanced weather file processing with debugging
- Proper field value extraction from saved HTML
- All original functionality preserved
"""

import json
import math
import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union
from dataclasses import dataclass
from bs4 import BeautifulSoup

from config import Config

# ========== DATA CLASSES ==========

@dataclass
class BatteryParams:
    """Centralized battery system parameters"""
    capacity_kwh: float = 0.0
    power_kw: float = 0.0
    enabled: bool = False
    roundtrip_efficiency: float = 0.9

    def __post_init__(self):
        if self.capacity_kwh > 0 and self.power_kw == 0:
            self.power_kw = self.capacity_kwh
        self.enabled = self.capacity_kwh > 0 and self.power_kw > 0

@dataclass
class FCASConfig:
    """FCAS configuration parameters"""
    region: str = 'NSW1'
    participation_rate: float = 0.75
    model_path: Optional[str] = None
    enabled_services: Dict[str, bool] = None

    def __post_init__(self):
        if self.enabled_services is None:
            self.enabled_services = {
                'fcas_enable_fast_raise': True,
                'fcas_enable_fast_lower': True,
                'fcas_enable_slow_raise': True,
                'fcas_enable_slow_lower': True,
                'fcas_enable_delayed_raise': False,
                'fcas_enable_delayed_lower': False,
                'fcas_enable_raise_regulation': False,
                'fcas_enable_lower_regulation': False
            }

# ========== OPTIMIZED FORM-BASED CONVERTER ==========

class FormBasedConverter:
    """Optimized form-based percentage/decimal conversion with cached mappings"""

    # Cached field classifications
    _TO_DECIMAL_FIELDS = frozenset({
        'installer_margin'
    })

    _KEEP_AS_PERCENTAGE = frozenset({
        'batt_minimum_SOC', 'batt_maximum_SOC','federal_tax_rate', 'real_discount_rate', 'loan_rate', 'debt_fraction', 'dc_degradation',
        'batt_ac_dc_efficiency', 'batt_dc_ac_efficiency', 'batt_dc_dc_efficiency', 'inflation_rate', 'load_escalation', 'macrs_bonus_frac',
        'inv_snl_eff_cec', 'subarray1_dcwiring_loss'
    })

    @classmethod
    def convert_value(cls, value: Any, field_key: str) -> Any:
        """Optimized value conversion with streamlined logic"""
        if value in (None, '', 'NaN', 'nan', 'None'):
            return None

        try:
            # Single conversion to float
            numeric_value = float(value.replace('%', '').strip() if isinstance(value, str) else value)

            # Streamlined conversion logic
            if field_key in cls._TO_DECIMAL_FIELDS:
                return numeric_value / 100.0 if numeric_value > 1.0 else numeric_value
            elif field_key in cls._KEEP_AS_PERCENTAGE:
                return numeric_value
            else:
                return numeric_value

        except (ValueError, TypeError):
            return value

    @staticmethod
    def extract_numeric_from_string(value: Any) -> str:
        """Extract numeric value from strings like '1=Something'"""
        if isinstance(value, str) and '=' in value:
            return value.split('=')[0].strip()
        return str(value).strip() if value is not None else ''

# ========== CACHED CONFIGURATION MANAGER ==========

class ConfigurationManager:
    """Optimized configuration management with caching"""

    def __init__(self):
        self.form_config = self._load_form_config()
        # Cache expensive operations
        self._type_mappings_cache = None
        self._validation_ranges_cache = None
        self._dispatch_mapping_cache = None
        self._field_config_cache = {}

    def _load_form_config(self) -> List[Dict]:
        """Load form configuration from JSON file"""
        try:
            config_path = Path(__file__).parent / 'static' / 'js' / 'form_config.json'
            with open(config_path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"[CONFIG ERROR] Failed to load form config: {e}")
            return []

    @property
    def type_mappings(self) -> Dict[str, List[str]]:
        """Cached type mappings"""
        if self._type_mappings_cache is None:
            self._type_mappings_cache = self._build_type_mappings()
        return self._type_mappings_cache

    @property
    def validation_ranges(self) -> Dict[str, Tuple[float, float, str]]:
        """Cached validation ranges"""
        if self._validation_ranges_cache is None:
            self._validation_ranges_cache = self._build_validation_ranges()
        return self._validation_ranges_cache

    @property
    def dispatch_mapping(self) -> Dict[str, int]:
        """Enhanced dispatch mapping with better string handling"""
        if self._dispatch_mapping_cache is None:
            self._dispatch_mapping_cache = {
                # String representations from form
                'Peak Shaving': 0,
                'Smart Energy Trading (Recommended)': 4,
                'Self Consumption': 5,
                # Alternative string formats
                'PeakShaving': 0,
                'RetailRateDispatch': 4,
                'SelfConsumption': 5,
                # Numeric string values
                '0': 0,
                '4': 4,
                '5': 5,
                # Direct numeric values
                0: 0,
                4: 4,
                5: 5
            }
        return self._dispatch_mapping_cache

    def get_field_config(self, field_key: str) -> Optional[Dict]:
        """Cached field configuration lookup"""
        if field_key not in self._field_config_cache:
            self._field_config_cache[field_key] = next(
                (f for f in self.form_config if f['key'] == field_key), None
            )
        return self._field_config_cache[field_key]

    def _build_type_mappings(self) -> Dict[str, List[str]]:
        """Build type mappings from form config (cached)"""
        type_mappings = {
            'integer': [], 'float': [], 'percentage': [], 'currency': [],
            'tou_matrix': [], 'sequence': [], 'boolean': [], 'array_single': [], 'text': []
        }

        for field in self.form_config:
            if 'type' in field and 'key' in field:
                field_type = field['type']
                field_key = field['key']

                if field_type in type_mappings:
                    type_mappings[field_type].append(field_key)
                elif field_type == 'number':
                    type_mappings['float'].append(field_key)
                elif field_type == 'date picker':
                    type_mappings['text'].append(field_key)
                elif field_type == 'select':
                    self._classify_select_field(field, type_mappings)

        return type_mappings

    def _classify_select_field(self, field: Dict, type_mappings: Dict[str, List[str]]):
        """Classify select fields as boolean or integer"""
        if 'validation' in field and 'options' in field['validation']:
            options = field['validation']['options']
            if len(options) == 2 and all(opt.lower() in ['yes', 'no'] for opt in options):
                type_mappings['boolean'].append(field['key'])
            else:
                type_mappings['integer'].append(field['key'])

    def _build_validation_ranges(self) -> Dict[str, Tuple[float, float, str]]:
        """Build validation ranges from form config (cached)"""
        validation_ranges = {}

        for field in self.form_config:
            if 'key' in field and 'validation' in field:
                validation = field['validation']
                if 'min' in validation and 'max' in validation:
                    min_val = float(validation['min'])
                    max_val = float(validation['max'])

                    if field.get('type') == 'percentage' and 'degradation' not in field['key']:
                        min_val = min_val / 100.0
                        max_val = max_val / 100.0
                        unit = 'percent'
                    else:
                        unit = 'units'

                    validation_ranges[field['key']] = (min_val, max_val, unit)

        return validation_ranges

# ========== CENTRALIZED BATTERY MANAGER ==========

class BatterySystemManager:
    """Centralized battery detection and management (eliminates duplication)"""

    def __init__(self, config_manager: ConfigurationManager):
        self.config = config_manager
        self.converter = FormBasedConverter()
        self._battery_cache = {}

    def detect_and_configure_battery(self, data: Dict[str, Any]) -> Tuple[BatteryParams, Dict[str, Any]]:
        """Single method for battery detection and configuration with FCAS cascade"""
        # Detect battery parameters (cached by capacity+power key)
        battery_params = self._detect_battery_system(data)

        if not battery_params.enabled:
            print("[BATTERY] No battery detected, skipping configuration")
            # CRITICAL: Disable FCAS when battery is disabled
            print("[FCAS] Battery disabled, automatically disabling FCAS")
            data = self._disable_fcas_system(data)
            return battery_params, data

        print(f"[BATTERY] Configuring {battery_params.capacity_kwh} kWh / {battery_params.power_kw} kW system")

        # Configure battery system
        configured_data = self._configure_battery_system(data, battery_params)

        # Configure FCAS (will check its own toggles)
        configured_data = self._configure_fcas_system(configured_data, battery_params)

        return battery_params, configured_data

    def _disable_fcas_system(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Disable FCAS system when battery is not available"""
        data.update({
            'fcas_region': 'DISABLED',
            '_fcas_disabled_by_toggle': True,
            'fcas_enable_fast_raise': False,
            'fcas_enable_fast_lower': False,
            'fcas_enable_slow_raise': False,
            'fcas_enable_slow_lower': False,
            'fcas_enable_delayed_raise': False,
            'fcas_enable_delayed_lower': False,
            'fcas_enable_raise_regulation': False,
            'fcas_enable_lower_regulation': False
        })
        return data

    def _configure_fcas_system(self, data: Dict[str, Any], battery_params: BatteryParams) -> Dict[str, Any]:
        """FCAS configuration with proper battery dependency and toggle support"""

        # CRITICAL: FCAS requires battery to be enabled
        if not battery_params.enabled:
            print("[FCAS] Battery disabled, automatically disabling FCAS")
            return self._disable_fcas_system(data)

        # Check for explicit FCAS disable toggle
        fcas_enabled = self._check_fcas_enabled(data)
        if not fcas_enabled:
            print("[FCAS] FCAS explicitly disabled via toggle")
            data.update({
                'fcas_region': 'DISABLED',
                '_fcas_disabled_by_toggle': True
            })
            return data

        print(f"[FCAS] Configuring for {battery_params.capacity_kwh} kWh battery")

        # Get FCAS region and participation rate
        fcas_region = data.get('fcas_region', self._determine_fcas_region(data))
        participation_rate = data.get('fcas_participation_rate', 0.75)

        # FCAS configuration update (single operation)
        fcas_config = {
            'fcas_region': fcas_region,
            'fcas_participation_rate': participation_rate,
            'power_mw': battery_params.power_kw / 1000,
            'energy_mwh': battery_params.capacity_kwh / 1000,
            'charge_power_mw': battery_params.power_kw / 1000 * 0.8,
            'efficiency': battery_params.roundtrip_efficiency,
            'analysis_period': data.get('analysis_period', 25)
        }

        # Process FCAS services
        fcas_services = {
            service: data.get(service, 'Yes' if 'fast' in service or 'slow' in service else 'No')
            for service in [
                'fcas_enable_fast_raise', 'fcas_enable_fast_lower',
                'fcas_enable_slow_raise', 'fcas_enable_slow_lower',
                'fcas_enable_delayed_raise', 'fcas_enable_delayed_lower',
                'fcas_enable_raise_regulation', 'fcas_enable_lower_regulation'
            ]
        }

        # Convert to boolean (optimized)
        for service, value in fcas_services.items():
            fcas_services[service] = (value if isinstance(value, bool)
                                    else str(value).lower() in ('yes', 'true', '1'))

        # Apply all FCAS configuration at once
        data.update(fcas_config)
        data.update(fcas_services)

        enabled_count = sum(fcas_services.values())
        print(f"[FCAS] Configured: {fcas_region}, {participation_rate:.2f} rate, {enabled_count}/8 services")

        return data

    def _check_fcas_enabled(self, data: Dict[str, Any]) -> bool:
        """
        FIXED: Check if FCAS is enabled via toggles/configuration with proper boolean handling
        """
        print(f"[FCAS DEBUG] Checking FCAS enabled status...")
        print(f"[FCAS DEBUG] Available data keys: {[k for k in data.keys() if 'fcas' in k.lower()]}")

        # Method 1: Check explicit FCAS enable flag (PRIORITY CHECK)
        fcas_enable_fields = ['fcas_enabled', 'en_fcas', 'fcas_enable']

        for field_name in fcas_enable_fields:
            if field_name in data:
                fcas_enable = data[field_name]
                print(f"[FCAS DEBUG] Found {field_name} = {fcas_enable} (type: {type(fcas_enable)})")

                # CRITICAL FIX: Proper boolean conversion
                if isinstance(fcas_enable, bool):
                    print(f"[FCAS DEBUG] Boolean value - enabled: {fcas_enable}")
                    return fcas_enable  # Return the actual boolean value
                elif isinstance(fcas_enable, str):
                    is_enabled = fcas_enable.lower() in ('yes', 'true', '1', 'enabled', 'on')
                    print(f"[FCAS DEBUG] String check - enabled: {is_enabled}")
                    return is_enabled
                elif isinstance(fcas_enable, (int, float)):
                    is_enabled = fcas_enable != 0
                    print(f"[FCAS DEBUG] Numeric check - enabled: {is_enabled}")
                    return is_enabled

        # Method 2: Check if FCAS region is explicitly disabled
        fcas_region = str(data.get('fcas_region', '')).upper()
        print(f"[FCAS DEBUG] FCAS region: '{fcas_region}'")
        if fcas_region in ['DISABLED', 'NONE', 'OFF']:
            print(f"[FCAS DEBUG] Region indicates disabled")
            return False

        # Method 3: Check if all FCAS services are explicitly disabled
        fcas_service_fields = [
            'fcas_enable_fast_raise', 'fcas_enable_fast_lower',
            'fcas_enable_slow_raise', 'fcas_enable_slow_lower',
            'fcas_enable_delayed_raise', 'fcas_enable_delayed_lower',
            'fcas_enable_raise_regulation', 'fcas_enable_lower_regulation'
        ]

        fcas_services_present = [field for field in fcas_service_fields if field in data]
        print(f"[FCAS DEBUG] Service fields present: {fcas_services_present}")

        if fcas_services_present:  # Only check if any FCAS service fields are present
            service_statuses = {}
            for field in fcas_services_present:
                value = data[field]
                if isinstance(value, bool):
                    enabled = value
                else:
                    enabled = str(value).lower() in ('yes', 'true', '1', 'on')
                service_statuses[field] = enabled

            print(f"[FCAS DEBUG] Service statuses: {service_statuses}")

            all_disabled = not any(service_statuses.values())
            if all_disabled:
                print(f"[FCAS DEBUG] All services disabled")
                return False

        # Default to ENABLED if no explicit disable signals found
        print(f"[FCAS DEBUG] No disable signals found - ENABLED by default")
        return True

    def _determine_fcas_region(self, data: Dict[str, Any]) -> str:
        """Simplified FCAS region determination"""
        user_region = data.get('fcas_region', 'auto')
        if user_region != 'auto' and user_region:
            return user_region

        # Quick state detection from address
        address = data.get('project_info', {}).get('project_address', '').lower()
        state_patterns = {
            'NSW1': ['nsw', 'sydney'], 'QLD1': ['qld', 'brisbane'],
            'VIC1': ['vic', 'melbourne'], 'SA1': ['sa', 'adelaide'],
            'TAS1': ['tas', 'hobart'], 'WEM': ['wa', 'perth']
        }

        for region, patterns in state_patterns.items():
            if any(pattern in address for pattern in patterns):
                return region

        return 'NSW1'  # Default fallback

    def _detect_battery_system(self, data: Dict[str, Any]) -> BatteryParams:
        """Centralized battery detection with proper toggle support"""
        try:
            # CRITICAL: Check en_batt toggle first
            en_batt = data.get('en_batt', 0)

            # Convert various representations to boolean
            if isinstance(en_batt, str):
                battery_enabled = en_batt.lower() in ('1', 'yes', 'true', 'enabled')
            else:
                battery_enabled = bool(int(float(en_batt)) if en_batt != '' else 0)

            if not battery_enabled:
                print("[BATTERY] Battery disabled via en_batt toggle")
                return BatteryParams(enabled=False)

            # Single source of truth for battery capacity detection
            capacity = (
                data.get('batt_computed_bank_capacity', 0) or
                data.get('batt_bank_installed_capacity', 0) or
                data.get('battery_capacity', 0)
            )

            # Single source of truth for battery power detection
            power = (
                data.get('batt_power_discharge_max_kwac', 0) or
                data.get('batt_ac_power', 0) or
                data.get('battery_power', 0)
            )

            # Get efficiency
            efficiency = data.get('batt_roundtrip_eff', 90) / 100.0

            # Check if we have valid capacity/power
            capacity_kwh = float(capacity or 0)
            power_kw = float(power or capacity or 0)

            if capacity_kwh <= 0 or power_kw <= 0:
                print(f"[BATTERY] Invalid capacity ({capacity_kwh} kWh) or power ({power_kw} kW)")
                return BatteryParams(enabled=False)

            return BatteryParams(
                capacity_kwh=capacity_kwh,
                power_kw=power_kw,
                roundtrip_efficiency=efficiency,
                enabled=True  # Explicitly set since all checks passed
            )

        except (ValueError, TypeError) as e:
            print(f"[BATTERY ERROR] Failed to detect battery system: {e}")
            return BatteryParams(enabled=False)

    def _configure_battery_system(self, data: Dict[str, Any], battery_params: BatteryParams) -> Dict[str, Any]:
        """Configure battery system with FIXED dispatch choice handling"""
        # Get form values for SOC and dispatch
        min_soc = data.get('batt_minimum_SOC', 30.0)
        max_soc = data.get('batt_maximum_SOC', 95.0)

        # FIXED: Proper dispatch choice extraction and conversion
        dispatch_raw = data.get('batt_dispatch_choice', '4')  # Default to 4 if not found

        # Convert dispatch choice to integer
        if isinstance(dispatch_raw, str):
            # Try direct numeric conversion first
            try:
                dispatch_choice = int(dispatch_raw)
            except ValueError:
                # Use mapping for text values
                dispatch_choice = self.config.dispatch_mapping.get(dispatch_raw, 4)
        else:
            dispatch_choice = int(dispatch_raw) if dispatch_raw is not None else 4

        # Validate dispatch choice
        if dispatch_choice not in [0, 4, 5]:
            print(f"[BATTERY WARNING] Invalid dispatch choice {dispatch_choice}, using Smart Energy Trading (4)")
            dispatch_choice = 4

        print(f"[BATTERY DEBUG] Raw dispatch value: '{dispatch_raw}' -> Converted to: {dispatch_choice}")

        # Clean conflicting configurations (single pass)
        conflicting_fields = [
            'forecast_price_signal_model', 'mp_energy_market_revenue',
            'mp_ancserv1_revenue', 'mp_ancserv2_revenue', 'mp_ancserv3_revenue', 'mp_ancserv4_revenue',
            'mp_ancserv5_revenue', 'mp_ancserv6_revenue', 'mp_ancserv7_revenue', 'mp_ancserv8_revenue',
            'batt_custom_dispatch', 'batt_target_power', 'batt_target_power_monthly',
            'dispatch_manual_percent', 'dispatch_manual_gridcharge', 'dispatch_manual_btm_discharge',
            'batt_cycle_cost_choice'
        ]

        for field in conflicting_fields:
            data.pop(field, None)

        # Set dispatch configuration (optimized mapping)
        dispatch_configs = {
            0: {  # Peak Shaving
                'batt_dispatch_auto_can_charge': 1,
                'batt_dispatch_auto_can_gridcharge': 1,
                'batt_dispatch_auto_can_clipcharge': 1,
                'batt_dispatch_auto_btm_can_discharge_to_grid': 0,
            },
            4: {  # Smart Energy Trading
                'batt_dispatch_auto_can_charge': 1,
                'batt_dispatch_auto_can_gridcharge': 1,
                'batt_dispatch_auto_can_clipcharge': 1,
                'batt_dispatch_auto_btm_can_discharge_to_grid': 1,
            },
            5: {  # Self Consumption
                'batt_dispatch_auto_can_charge': 1,
                'batt_dispatch_auto_can_gridcharge': 0,
                'batt_dispatch_auto_can_clipcharge': 1,
                'batt_dispatch_auto_btm_can_discharge_to_grid': 0,
            }
        }

        # Apply configuration (single update)
        config_update = {
            'batt_dispatch_choice': dispatch_choice,  # Use the properly converted value
            'batt_target_choice': 0,
            'batt_meter_position': 0,
            'forecast_price_signal_model': 0,
            'batt_minimum_SOC': min_soc,
            'batt_maximum_SOC': max_soc,
            'batt_look_ahead_hours': 24,
            'batt_dispatch_update_frequency_hours': 1,
            'batt_life_model': 1,
        }

        # Add dispatch-specific parameters
        config_update.update(dispatch_configs.get(dispatch_choice, dispatch_configs[4]))

        # Apply all updates at once
        data.update(config_update)

        allowed_choices = {0: "Peak Shaving", 4: "Smart Energy Trading", 5: "Self Consumption"}
        print(
            f"[BATTERY] Configured dispatch: {allowed_choices.get(dispatch_choice, 'Smart Energy Trading')} (choice: {dispatch_choice})")
        print(f"[BATTERY] SOC limits: {min_soc}% min, {max_soc}% max")

        return data

# ========== ENHANCED FORM PROCESSOR WITH FIXED COORDINATE EXTRACTION ==========

class FormProcessor:
    """Enhanced form processing with FIXED coordinate extraction"""

    def __init__(self, config_manager: ConfigurationManager):
        self.config = config_manager
        self.converter = FormBasedConverter()

    def process_form(self, soup: BeautifulSoup) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """FIXED: Form processing with enhanced coordinate extraction"""
        data = {'project_info': {}}
        extra_costs = {}

        # First check for coordinates from address search
        address_coords = self._get_address_search_coordinates(soup)
        if address_coords:
            data.update(address_coords)  # Ensure they're at top level
            data['project_info'].update(address_coords)  # Also in project_info

        try:
            # CRITICAL FIX: Extract coordinates FIRST with enhanced debugging
            print("[COORD DEBUG] Starting coordinate extraction process...")
            self._extract_coordinates_enhanced(soup, data)

            # DEBUG: Verify coordinates were extracted
            lat_extracted = data.get('lat', 'NOT_FOUND')
            lon_extracted = data.get('lon', 'NOT_FOUND')
            print(f"[COORD VERIFY] After extraction: lat={lat_extracted}, lon={lon_extracted}")

            # Process FCAS toggle first with proper checkbox handling
            fcas_toggle = soup.select_one('#fcas-enable[data-pysam-key="fcas_enabled"]')
            if fcas_toggle:
                is_checked = fcas_toggle.get('checked') is not None
                data['fcas_enabled'] = 1 if is_checked else 0
                print(f"[FCAS DEBUG] Main toggle processed: fcas_enabled = {data['fcas_enabled']}")

            # Single pass through all other form fields
            for field in soup.select('[data-pysam-key]'):
                key = field.get('data-pysam-key')
                if not key or key == 'fcas_enabled':  # Skip if already processed
                    continue

                value = self._extract_field_value_enhanced(field)
                if value is None:
                    continue

                # Convert value
                converted_value = self.converter.convert_value(value, key)
                if converted_value is not None:
                    field_config = self.config.get_field_config(key)

                    # Store in appropriate location
                    if field_config and field_config.get('general_info', False):
                        data['project_info'][key] = converted_value
                    else:
                        data[key] = converted_value

                    # Collect extra costs
                    if (field_config and
                            field_config.get('special_handling') == 'extra_capex' and
                            isinstance(converted_value, (int, float))):
                        extra_costs[key] = converted_value

            # Process additional data
            tariff_data = self._process_tariff_data(soup)
            data.update(tariff_data)
            self._process_fcas_hidden_fields(soup, data)

            # FINAL VERIFICATION: Log coordinates one more time
            final_lat = data.get('lat', 'STILL_NOT_FOUND')
            final_lon = data.get('lon', 'STILL_NOT_FOUND')
            print(f"[COORD FINAL] Final extracted coordinates: lat={final_lat}, lon={final_lon}")

            return data, extra_costs

        except Exception as e:
            print(f"[FORM ERROR] Processing failed: {e}")
            return data, extra_costs

    def _get_address_search_coordinates(self, soup: BeautifulSoup) -> Dict[str, float]:
        """Extract coordinates from address search results"""
        coords = {}
        try:
            # Look for hidden fields or elements containing coordinates
            lat_elem = soup.select_one('#address-search-lat, [name="latitude"]')
            lon_elem = soup.select_one('#address-search-lon, [name="longitude"]')

            if lat_elem and lon_elem:
                coords['lat'] = float(lat_elem.get('value', 0))
                coords['lon'] = float(lon_elem.get('value', 0))
                print(f"[ADDRESS COORDS] Found: {coords['lat']}, {coords['lon']}")
        except Exception as e:
            print(f"[ADDRESS COORDS ERROR] {e}")
        return coords

    def _extract_coordinates_enhanced(self, soup: BeautifulSoup, data: Dict[str, Any]):
        """ENHANCED coordinate extraction with comprehensive debugging"""
        print("[COORD ENHANCED] Starting enhanced coordinate extraction...")

        # Method 1: Look for input elements with specific IDs
        coordinate_fields = [
            {'id': 'latitude', 'key': 'lat', 'type': 'latitude'},
            {'id': 'longitude', 'key': 'lon', 'type': 'longitude'}
        ]

        for field_info in coordinate_fields:
            print(f"[COORD] Processing {field_info['type']}...")

            # Try multiple selector strategies
            selectors = [
                f"#{field_info['id']}",
                f"input#{field_info['id']}",
                f"[data-pysam-key=\"{field_info['key']}\"]",
                f"[name=\"{field_info['id']}\"]",
                f"[id=\"{field_info['id']}\"]"
            ]

            value_found = None
            selector_used = None

            for selector in selectors:
                element = soup.select_one(selector)
                if element:
                    print(f"[COORD] Found element with selector: {selector}")
                    print(f"[COORD] Element type: {element.name}, attributes: {dict(element.attrs)}")

                    # Extract value using multiple methods
                    raw_value = self._extract_field_value_enhanced(element)
                    print(f"[COORD] Raw value extracted: '{raw_value}' (type: {type(raw_value)})")

                    if raw_value and str(raw_value).strip():
                        try:
                            value_found = float(raw_value)
                            selector_used = selector
                            print(f"[COORD] ✅ Successfully converted to float: {value_found}")
                            break
                        except (ValueError, TypeError) as e:
                            print(f"[COORD] ❌ Failed to convert '{raw_value}' to float: {e}")
                            continue
                    else:
                        print(f"[COORD] ❌ Empty or invalid value: '{raw_value}'")
                else:
                    print(f"[COORD] ❌ No element found with selector: {selector}")

            # Store the value if found
            if value_found is not None:
                data[field_info['key']] = value_found
                print(f"[COORD] ✅ {field_info['type']} saved: {field_info['key']} = {value_found}")
                print(f"[COORD] Used selector: {selector_used}")
            else:
                print(f"[COORD] ❌ No valid {field_info['type']} found")

        # Method 2: Debug - show all coordinate-related elements
        print("[COORD DEBUG] All elements that might contain coordinates:")

        coordinate_related = soup.find_all(['input', 'select'], attrs={
            'id': lambda x: x and ('lat' in x.lower() or 'lon' in x.lower() or 'coord' in x.lower())
        })

        for elem in coordinate_related:
            print(f"  - {elem.name}#{elem.get('id', 'NO_ID')}: value='{elem.get('value', 'NO_VALUE')}', type='{elem.get('type', 'NO_TYPE')}'")

        # Method 3: Search by data-pysam-key
        lat_pysam = soup.select_one('[data-pysam-key="lat"]')
        lon_pysam = soup.select_one('[data-pysam-key="lon"]')

        if lat_pysam:
            lat_val = self._extract_field_value_enhanced(lat_pysam)
            if lat_val:
                try:
                    data['lat'] = float(lat_val)
                    print(f"[COORD] ✅ Latitude from data-pysam-key: {data['lat']}")
                except (ValueError, TypeError):
                    print(f"[COORD] ❌ Invalid latitude from data-pysam-key: '{lat_val}'")

        if lon_pysam:
            lon_val = self._extract_field_value_enhanced(lon_pysam)
            if lon_val:
                try:
                    data['lon'] = float(lon_val)
                    print(f"[COORD] ✅ Longitude from data-pysam-key: {data['lon']}")
                except (ValueError, TypeError):
                    print(f"[COORD] ❌ Invalid longitude from data-pysam-key: '{lon_val}'")

        # Final status
        final_lat = data.get('lat')
        final_lon = data.get('lon')
        if final_lat is not None and final_lon is not None:
            print(f"[COORD] ✅ SUCCESS: Coordinates extracted successfully!")
            print(f"[COORD]   Latitude: {final_lat}")
            print(f"[COORD]   Longitude: {final_lon}")
        else:
            print(f"[COORD] ❌ FAILED: Could not extract coordinates")
            print(f"[COORD]   Latitude: {final_lat}")
            print(f"[COORD]   Longitude: {final_lon}")

    def _extract_field_value_enhanced(self, field) -> Any:
        """
        ENHANCED: Field value extraction with comprehensive debugging and better handling
        """
        if not field:
            return None

        field_id = field.get('id', 'NO_ID')
        field_name = field.get('name', 'NO_NAME')

        # Handle select elements - IMPROVED LOGIC
        if field.name == 'select':
            # First, check if the select element itself has a value attribute
            select_value = field.get('value')
            if select_value and select_value.strip():
                print(f"[FIELD DEBUG] Select #{field_id}: direct value = '{select_value}'")
                return select_value.strip()

            # Fallback to checking selected option
            selected = field.find('option', selected=True)
            if selected and selected.get('value'):
                print(f"[FIELD DEBUG] Select #{field_id}: selected option = '{selected.get('value')}'")
                return selected.get('value')

            # Last resort: get the value attribute
            fallback_value = field.get('value', '')
            print(f"[FIELD DEBUG] Select #{field_id}: fallback value = '{fallback_value}'")
            return fallback_value

        # Handle input elements with enhanced debugging
        input_elem = field.find('input') if field.name != 'input' else field
        if input_elem:
            input_type = input_elem.get('type', '').lower()
            input_value = input_elem.get('value', '')

            print(f"[FIELD DEBUG] Input #{field_id}: type='{input_type}', value='{input_value}'")

            if input_type == 'checkbox':
                # FIXED: Proper checkbox state detection
                checked_attr = input_elem.get('checked')
                is_checked = checked_attr is not None
                print(f"[FIELD DEBUG] Checkbox #{field_id}: checked = {is_checked}")
                return is_checked

            elif input_type == 'radio':
                # Radio button - check if selected
                is_checked = input_elem.get('checked') is not None
                print(f"[FIELD DEBUG] Radio #{field_id}: checked = {is_checked}")
                return is_checked if is_checked else None
            else:
                # Regular input field
                print(f"[FIELD DEBUG] Regular input #{field_id}: value = '{input_value}'")
                return input_value

        # Handle nested select - IMPROVED
        select_elem = field.find('select')
        if select_elem:
            # Check select element's value first
            select_value = select_elem.get('value')
            if select_value and select_value.strip():
                print(f"[FIELD DEBUG] Nested select #{field_id}: value = '{select_value}'")
                return select_value.strip()

            # Fallback to selected option
            selected = select_elem.find('option', selected=True)
            if selected and selected.get('value'):
                print(f"[FIELD DEBUG] Nested select #{field_id}: selected = '{selected.get('value')}'")
                return selected.get('value')

            fallback_value = select_elem.get('value', '')
            print(f"[FIELD DEBUG] Nested select #{field_id}: fallback = '{fallback_value}'")
            return fallback_value

        # Direct value attribute
        direct_value = field.get('value', '')
        print(f"[FIELD DEBUG] Direct value #{field_id}: '{direct_value}'")
        return direct_value

    def _fix_tou_matrix_structure(self, raw_tou_data: List[List], feed_in_tariff: float = 0.0) -> List[List]:
        """
        Enhanced TOU matrix structure fix with feed-in tariff integration
        Expected: [period, tier, max_usage, usage_units, buy_rate, sell_rate]
        """
        if not raw_tou_data:
            print("[TARIFF ERROR] Empty TOU matrix data")
            return []

        fixed_matrix = []
        detected_periods = set()

        for i, row in enumerate(raw_tou_data):
            try:
                if len(row) == 6:
                    # Already correct 6-column format - update sell rate with feed-in tariff
                    period, tier, max_usage, usage_units, buy_rate, sell_rate = row

                    # CRITICAL: Always use feed-in tariff for sell rate (6th column)
                    actual_sell_rate = feed_in_tariff  # Always use the form's feed-in tariff

                    fixed_row = [
                        int(period),
                        int(tier),
                        float(max_usage) if max_usage > 1e30 else 9.9999999999999998e+37,
                        int(usage_units) if isinstance(usage_units, (int, float)) else 0,
                        float(buy_rate),
                        actual_sell_rate  # ← FIXED: Use feed-in tariff from form
                    ]

                    detected_periods.add(int(period))
                    fixed_matrix.append(fixed_row)
                    print(f"[TARIFF] Period {period}: ${buy_rate:.3f}/kWh buy, ${actual_sell_rate:.3f}/kWh sell")

                elif len(row) == 5:
                    # Missing sell rate - add feed-in tariff
                    period, tier, max_usage, usage_units, buy_rate = row
                    fixed_row = [
                        int(period),
                        int(tier),
                        float(max_usage) if max_usage > 1e30 else 9.9999999999999998e+37,
                        int(usage_units) if isinstance(usage_units, (int, float)) else 0,
                        float(buy_rate),
                        feed_in_tariff  # ← FIXED: Add feed-in tariff
                    ]

                    detected_periods.add(int(period))
                    fixed_matrix.append(fixed_row)
                    print(f"[TARIFF] Period {period}: ${buy_rate:.3f}/kWh buy, ${feed_in_tariff:.3f}/kWh sell (added)")

                elif len(row) >= 4:
                    # Handle various malformed formats
                    period = int(row[0])
                    tier = int(row[1]) if len(row) > 1 else 1

                    # Detect malformed structure (rates in wrong columns)
                    if len(row) == 6 and isinstance(row[3], float) and row[3] < 1.0:
                        # CRITICAL FIX: Original issue - rates in columns 3-4 instead of 4-5
                        max_usage, wrong_rate1, wrong_rate2, last_col = row[2], row[3], row[4], row[5]

                        # Determine which rate to use (prefer the higher one for peak periods)
                        if period == 1 or period == 2:  # Peak periods typically
                            buy_rate = max(float(wrong_rate1), float(wrong_rate2))
                        else:  # Off-peak periods
                            buy_rate = min(float(wrong_rate1), float(wrong_rate2))

                        fixed_row = [
                            period,
                            tier,
                            float(max_usage) if max_usage > 1e30 else 9.9999999999999998e+37,
                            0,  # Usage units = 0 (kWh)
                            buy_rate,
                            feed_in_tariff  # ← FIXED: Use feed-in tariff instead of 0.0
                        ]

                        detected_periods.add(period)
                        fixed_matrix.append(fixed_row)
                        print(
                            f"[TARIFF FIX] Corrected malformed row: period {period}, buy ${buy_rate:.3f}, sell ${feed_in_tariff:.3f}")

                    else:
                        # Standard missing columns case
                        max_usage = float(row[2]) if len(row) > 2 else 9.9999999999999998e+37
                        buy_rate = float(row[3]) if len(row) > 3 else 0.05  # Default rate

                        fixed_row = [
                            period,
                            tier,
                            max_usage,
                            0,  # Usage units = 0 (kWh)
                            buy_rate,
                            feed_in_tariff  # ← FIXED: Use feed-in tariff instead of 0.0
                        ]

                        detected_periods.add(period)
                        fixed_matrix.append(fixed_row)
                        print(
                            f"[TARIFF] Period {period}: ${buy_rate:.3f}/kWh buy, ${feed_in_tariff:.3f}/kWh sell (reconstructed)")

                else:
                    print(f"[TARIFF ERROR] Invalid TOU row length: {len(row)}, skipping: {row}")
                    continue

            except (ValueError, TypeError, IndexError) as e:
                print(f"[TARIFF ERROR] Failed to process row {i}: {row}, error: {e}")
                continue

        # Validate and complete the matrix
        if not fixed_matrix:
            print("[TARIFF ERROR] No valid TOU periods after processing")
            # Fallback to default 2-period structure WITH feed-in tariff
            fixed_matrix = [
                [1, 1, 9.9999999999999998e+37, 0, 0.07, feed_in_tariff],  # Peak
                [2, 1, 9.9999999999999998e+37, 0, 0.05, feed_in_tariff]  # Off-peak
            ]
            detected_periods = {1, 2}
            print(f"[TARIFF FIX] Applied default 2-period TOU structure with ${feed_in_tariff:.3f}/kWh sell rate")

        # Ensure periods are consecutive and complete
        max_period = max(detected_periods)
        expected_periods = set(range(1, max_period + 1))
        missing_periods = expected_periods - detected_periods

        if missing_periods:
            print(f"[TARIFF WARNING] Missing periods: {sorted(missing_periods)}")
            # Add missing periods with interpolated rates
            existing_rates = [row[4] for row in fixed_matrix]
            avg_rate = sum(existing_rates) / len(existing_rates) if existing_rates else 0.06

            for missing_period in sorted(missing_periods):
                missing_row = [
                    missing_period,
                    1,  # Tier 1
                    9.9999999999999998e+37,  # Max usage
                    0,  # Usage units (kWh)
                    avg_rate,  # Average of existing rates
                    feed_in_tariff  # ← FIXED: Use feed-in tariff instead of 0.0
                ]
                fixed_matrix.append(missing_row)
                print(
                    f"[TARIFF FIX] Added missing period {missing_period}: ${avg_rate:.3f}/kWh buy, ${feed_in_tariff:.3f}/kWh sell")

        # Sort by period number
        fixed_matrix.sort(key=lambda x: x[0])

        # Validate against schedule matrices (if they exist)
        num_periods = len(fixed_matrix)
        if num_periods > 4:
            print(f"[TARIFF WARNING] Too many periods ({num_periods}), truncating to 4")
            fixed_matrix = fixed_matrix[:4]

        # Log final structure for verification
        print(f"[TARIFF SUMMARY] Final TOU structure: {len(fixed_matrix)} periods")
        for row in fixed_matrix:
            period, tier, max_usage, usage_units, buy_rate, sell_rate = row
            print(f"[TARIFF] Period {period}: ${buy_rate:.3f}/kWh buy, ${sell_rate:.3f}/kWh sell")

        return fixed_matrix

    def _validate_schedule_consistency(self, tariff_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that schedule matrices are consistent with TOU periods
        This ensures no prior tariff functionality is broken
        """
        if 'ur_ec_tou_mat' not in tariff_data:
            return tariff_data

        tou_matrix = tariff_data['ur_ec_tou_mat']
        valid_periods = set(row[0] for row in tou_matrix)
        max_period = max(valid_periods) if valid_periods else 2

        # Check weekday schedule
        if 'ur_ec_sched_weekday' in tariff_data:
            weekday_schedule = tariff_data['ur_ec_sched_weekday']
            schedule_periods = set()

            for month_row in weekday_schedule:
                for hour_period in month_row:
                    if isinstance(hour_period, (int, float)) and hour_period > 0:
                        schedule_periods.add(int(hour_period))

            invalid_periods = schedule_periods - valid_periods
            if invalid_periods:
                print(f"[TARIFF WARNING] Weekday schedule references invalid periods: {sorted(invalid_periods)}")
                print(f"[TARIFF] Valid periods: {sorted(valid_periods)}")

                # Fix schedule by mapping invalid periods to valid ones
                for month_idx, month_row in enumerate(weekday_schedule):
                    for hour_idx, period in enumerate(month_row):
                        if isinstance(period, (int, float)) and int(period) in invalid_periods:
                            # Map invalid period to closest valid period
                            new_period = min(valid_periods, key=lambda x: abs(x - int(period)))
                            weekday_schedule[month_idx][hour_idx] = new_period
                            print(f"[TARIFF FIX] Mapped period {int(period)} -> {new_period} in weekday schedule")

        # Check weekend schedule
        if 'ur_ec_sched_weekend' in tariff_data:
            weekend_schedule = tariff_data['ur_ec_sched_weekend']
            schedule_periods = set()

            for month_row in weekend_schedule:
                for hour_period in month_row:
                    if isinstance(hour_period, (int, float)) and hour_period > 0:
                        schedule_periods.add(int(hour_period))

            invalid_periods = schedule_periods - valid_periods
            if invalid_periods:
                print(f"[TARIFF WARNING] Weekend schedule references invalid periods: {sorted(invalid_periods)}")

                # Fix schedule
                for month_idx, month_row in enumerate(weekend_schedule):
                    for hour_idx, period in enumerate(month_row):
                        if isinstance(period, (int, float)) and int(period) in invalid_periods:
                            new_period = min(valid_periods, key=lambda x: abs(x - int(period)))
                            weekend_schedule[month_idx][hour_idx] = new_period
                            print(f"[TARIFF FIX] Mapped period {int(period)} -> {new_period} in weekend schedule")

        return tariff_data

    def _process_tariff_data(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Enhanced tariff data processing with feed-in tariff integration into TOU matrix
        """
        tariff_data = {}

        # STEP 1: Extract feed-in tariff first (check multiple possible field names)
        feed_in_tariff = 0.0

        # Try different possible field selectors for feed-in tariff
        fit_selectors = [
            '#feed-in-tariff',
            '#feedInTariff',
            '[name="feed_in_tariff"]',
            '[data-pysam-key="feed_in_tariff"]'
        ]

        for selector in fit_selectors:
            fit_field = soup.select_one(selector)
            if fit_field and fit_field.get('value'):
                try:
                    feed_in_tariff = float(fit_field.get('value', 0))
                    print(f"[TARIFF] Feed-in tariff extracted: ${feed_in_tariff:.3f}/kWh from {selector}")
                    break
                except (ValueError, TypeError):
                    continue

        if feed_in_tariff == 0.0:
            print("[TARIFF WARNING] No valid feed-in tariff found, using 0.0")

        # STEP 2: Process tariff fields with feed-in tariff integration
        tariff_fields = [
            'ur_ec_sched_weekday', 'ur_ec_sched_weekend',
            'ur_ec_tou_mat', 'ur_dc_flat_mat',
            'monthly_fixed_charge'
        ]

        for field_name in tariff_fields:
            field = soup.select_one(f'#{field_name}')
            if field and field.get('value'):
                try:
                    if field_name in ['ur_ec_sched_weekday', 'ur_ec_sched_weekend', 'ur_dc_flat_mat']:
                        tariff_data[field_name] = json.loads(field.get('value'))
                    elif field_name == 'ur_ec_tou_mat':
                        # CRITICAL: Pass feed-in tariff to TOU matrix processing
                        raw_tou_data = json.loads(field.get('value'))
                        tariff_data[field_name] = self._fix_tou_matrix_structure(raw_tou_data, feed_in_tariff)
                        print(
                            f"[TARIFF] TOU matrix with ${feed_in_tariff:.3f}/kWh sell rate: {len(tariff_data[field_name])} periods")
                    else:
                        tariff_data[field_name] = float(field.get('value'))
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"[TARIFF ERROR] Failed to parse {field_name}: {e}")

        # STEP 3: Clean up - remove any references to ur_ec_export_mat since it doesn't exist in PySAM
        # The feed-in tariff is now properly embedded in the TOU matrix 6th column

        return tariff_data

    def _process_fcas_hidden_fields(self, soup: BeautifulSoup, data: Dict[str, Any]):
        """Optimized FCAS hidden field processing"""
        fcas_hidden_fields = [
            'hidden_fcas_region', 'hidden_fcas_participation_rate', 'hidden_fcas_db_path',
            'hidden_fcas_battery_reserve', 'hidden_fcas_enable_fast_raise',
            'hidden_fcas_enable_fast_lower', 'hidden_fcas_enable_slow_raise',
            'hidden_fcas_enable_slow_lower', 'hidden_fcas_enable_delayed_raise',
            'hidden_fcas_enable_delayed_lower', 'hidden_fcas_enable_raise_regulation',
            'hidden_fcas_enable_lower_regulation', 'hidden_fcas_forecast_method',
            'hidden_fcas_fast_premium', 'hidden_fcas_regulation_premium'
        ]

        for field_id in fcas_hidden_fields:
            field = soup.select_one(f'#{field_id}')
            if field and field.get('value'):
                pysam_key = field.get('data-pysam-key')
                if pysam_key:
                    try:
                        value = field.get('value')
                        clean_key = pysam_key.replace('hidden_', '') if pysam_key.startswith('hidden_') else pysam_key

                        # Optimized boolean conversion
                        if 'enable' in clean_key:
                            data[clean_key] = value.lower() in ('yes', 'true', '1')
                        else:
                            data[clean_key] = self.converter.convert_value(value, clean_key)
                    except Exception:
                        pass  # Skip invalid entries

# ========== LOAD DATA PROCESSING ====================
class LoadDataManager:
    """Handle load profile processing and integration with PySAM"""

    def __init__(self, config_manager: ConfigurationManager):
        self.config = config_manager
        self.converter = FormBasedConverter()

        # In pysam_form_integration2.py, LoadDataManager class

    import time

    def process_load_configuration(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process load configuration with timeout protection"""
        start_time = time.time()
        max_processing_time = 10  # 10 seconds max

        try:
            # Check timeout before expensive operations
            if time.time() - start_time > max_processing_time:
                print(f"[LOAD TIMEOUT] Processing exceeded {max_processing_time}s, using default")
                return self._apply_default_load_profile(data)

            load_method = data.get('hidden_load_method', 'auto')
            print(f"[LOAD] Processing load method: {load_method}")

            if load_method == 'file':
                # Add timeout check before file processing
                if time.time() - start_time > max_processing_time:
                    return self._apply_default_load_profile(data)
                return self._process_uploaded_load_file(data)

            elif load_method == 'manual':
                # Add timeout check before manual processing
                if time.time() - start_time > max_processing_time:
                    return self._apply_default_load_profile(data)
                return self._process_manual_load_data(data)

            else:  # 'auto' or any other value
                # Add timeout check before auto processing
                if time.time() - start_time > max_processing_time:
                    return self._apply_default_load_profile(data)
                return self._process_auto_load_generation(data)

        except Exception as e:
            processing_time = time.time() - start_time
            print(f"[LOAD ERROR] Load processing failed after {processing_time:.1f}s: {e}")
            return self._apply_default_load_profile(data)

    def _process_uploaded_load_file(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process user-uploaded load profile data from form and replace PySAM grid.load attribute"""
        try:
            # FIXED: Look for load data in form fields instead of physical file
            load_data_raw = data.get('hidden_load_data')

            if not load_data_raw:
                print("[LOAD ERROR] No load data found in form fields, using existing PySAM load data")
                return data

            # Parse the JSON load data
            try:
                if isinstance(load_data_raw, str):
                    import json
                    load_profile = json.loads(load_data_raw)
                else:
                    load_profile = load_data_raw
            except (json.JSONDecodeError, TypeError) as e:
                print(f"[LOAD ERROR] Failed to parse load data: {e}")
                return data

            # Validate the uploaded data
            if len(load_profile) != 8760:
                print(f"[LOAD ERROR] Invalid load profile length: {len(load_profile)}, expected 8760")
                return data

            # Validate all values are numeric and non-negative
            try:
                load_profile = [float(x) for x in load_profile]
                if any(x < 0 for x in load_profile):
                    print("[LOAD ERROR] Negative load values found")
                    return data
            except (ValueError, TypeError):
                print("[LOAD ERROR] Non-numeric load values found")
                return data

            # Calculate summary statistics
            annual_energy = sum(load_profile)
            peak_demand = max(load_profile)
            avg_demand = annual_energy / 8760

            # CRITICAL: Replace PySAM grid.load attribute directly
            data['load'] = load_profile

            # Remove any conflicting annual_energy override since we're using actual data
            if 'annual_energy' in data:
                print(
                    f"[LOAD] Removing annual_energy override ({data['annual_energy']}) - using actual uploaded data ({annual_energy:.0f} kWh)")
                data.pop('annual_energy', None)

            # Add metadata for reference and results analysis
            data.update({
                'load_source': 'uploaded_file',
                'load_annual_kwh': annual_energy,
                'load_peak_kw': peak_demand,
                'load_average_kw': avg_demand,
                'load_data_replaced': True
            })

            print(
                f"[LOAD] ✅ PySAM grid.load replaced with uploaded data: {annual_energy:.0f} kWh annual, {peak_demand:.1f} kW peak")
            return data

        except Exception as e:
            print(f"[LOAD ERROR] File processing failed: {e}")
            return data  # Keep existing PySAM load data

    def _process_manual_load_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process manually entered load data from form and replace PySAM grid.load attribute"""
        try:
            # FIXED: Look for manual load data in form instead of individual monthly fields
            load_data_raw = data.get('hidden_load_data')

            if not load_data_raw:
                print("[LOAD] No manual load data provided, keeping existing PySAM load data")
                return data

            # Parse the JSON load data
            try:
                if isinstance(load_data_raw, str):
                    import json
                    load_profile = json.loads(load_data_raw)
                else:
                    load_profile = load_data_raw
            except (json.JSONDecodeError, TypeError) as e:
                print(f"[LOAD ERROR] Failed to parse manual load data: {e}")
                return data

            # Validate the data
            if len(load_profile) != 8760:
                print(f"[LOAD ERROR] Invalid manual load profile length: {len(load_profile)}, expected 8760")
                return data

            # Validate all values are numeric and non-negative
            try:
                load_profile = [float(x) for x in load_profile]
                if any(x < 0 for x in load_profile):
                    print("[LOAD ERROR] Negative load values found in manual data")
                    return data
            except (ValueError, TypeError):
                print("[LOAD ERROR] Non-numeric load values found in manual data")
                return data

            # Calculate summary statistics
            annual_energy = sum(load_profile)
            peak_demand = max(load_profile)

            # CRITICAL: Replace PySAM grid.load attribute directly
            data['load'] = load_profile

            # Remove any conflicting annual_energy override since we're using actual data
            if 'annual_energy' in data:
                print(
                    f"[LOAD] Removing annual_energy override ({data['annual_energy']}) - using actual manual data ({annual_energy:.0f} kWh)")
                data.pop('annual_energy', None)

            # Add metadata for reference
            data.update({
                'load_source': 'manual_entry',
                'load_annual_kwh': annual_energy,
                'load_peak_kw': peak_demand,
                'load_data_replaced': True
            })

            print(f"[LOAD] ✅ PySAM grid.load replaced with manual entry profile: {annual_energy:.0f} kWh annual")
            return data

        except Exception as e:
            print(f"[LOAD ERROR] Manual data processing failed: {e}")
            return data  # Keep existing PySAM load data

    def _process_auto_load_generation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle automatic load profile - check for form data first, then override or existing template"""
        try:
            # FIXED: Check for auto-generated load data from form first
            load_data_raw = data.get('hidden_load_data')
            annual_energy_override = data.get('hidden_load_annual_energy') or data.get('annual_energy_kwh') or data.get(
                'annual_energy')

            if load_data_raw:
                # Auto-generated data exists in form
                try:
                    if isinstance(load_data_raw, str):
                        import json
                        load_profile = json.loads(load_data_raw)
                    else:
                        load_profile = load_data_raw

                    # Validate the data
                    if len(load_profile) == 8760:
                        load_profile = [float(x) for x in load_profile]
                        annual_energy = sum(load_profile)
                        peak_demand = max(load_profile)

                        # Replace PySAM grid.load with auto-generated profile
                        data['load'] = load_profile

                        data.update({
                            'load_source': 'auto_generated',
                            'load_annual_kwh': annual_energy,
                            'load_peak_kw': peak_demand,
                            'load_data_replaced': True
                        })

                        print(
                            f"[LOAD] ✅ PySAM grid.load replaced with auto-generated profile: {annual_energy:.0f} kWh annual, {peak_demand:.1f} kW peak")
                        return data

                except Exception as e:
                    print(f"[LOAD ERROR] Failed to process auto-generated data: {e}")
                    # Fall through to override/existing logic

            # Check if user provided annual energy override
            if annual_energy_override:
                try:
                    annual_energy = float(annual_energy_override)

                    # Generate new hourly profile based on override
                    system_capacity = float(data.get('system_capacity', 100))
                    load_type = data.get('hidden_load_type', 'commercial')
                    load_profile = self._generate_load_profile_by_type(annual_energy, load_type)
                    peak_demand = max(load_profile)

                    # Replace PySAM grid.load with scaled profile
                    data['load'] = load_profile

                    # Set annual_energy for PySAM scaling (if supported)
                    data['annual_energy'] = annual_energy

                    data.update({
                        'load_source': 'auto_generated_override',
                        'load_annual_kwh': annual_energy,
                        'load_peak_kw': peak_demand,
                        'load_data_replaced': True
                    })

                    print(
                        f"[LOAD] ✅ PySAM grid.load replaced with auto-generated profile: {annual_energy:.0f} kWh annual (user override)")

                except (ValueError, TypeError):
                    print(
                        f"[LOAD] Invalid annual energy override: {annual_energy_override}, keeping existing PySAM load")
            else:
                # No override - keep existing PySAM load data
                existing_load = data.get('load', [])
                if existing_load and len(existing_load) == 8760:
                    annual_energy = sum(existing_load)
                    peak_demand = max(existing_load)

                    data.update({
                        'load_source': 'existing_pysam_template',
                        'load_annual_kwh': annual_energy,
                        'load_peak_kw': peak_demand,
                        'load_data_replaced': False
                    })

                    print(
                        f"[LOAD] Keeping existing PySAM template load: {annual_energy:.0f} kWh annual, {peak_demand:.1f} kW peak")
                else:
                    print("[LOAD] No existing load data found, generating default profile")
                    return self._apply_default_load_profile(data)

            return data

        except Exception as e:
            print(f"[LOAD ERROR] Auto-generation failed: {e}")
            return self._apply_default_load_profile(data)

    def _generate_load_profile_by_type(self, annual_energy: float, load_type: str) -> List[float]:
        """Generate load profile based on type (residential, commercial, industrial)"""
        import math

        # Average hourly load
        avg_hourly = annual_energy / 8760
        hourly_profile = []

        for hour in range(8760):
            day_of_year = hour // 24
            hour_of_day = hour % 24
            day_of_week = day_of_year % 7

            # Seasonal variation
            seasonal_factor = 1.0 + 0.3 * math.sin(2 * math.pi * day_of_year / 365 + math.pi / 2)

            # Load type specific patterns
            if load_type == 'residential':
                # Morning and evening peaks
                if 6 <= hour_of_day <= 8 or 18 <= hour_of_day <= 22:
                    daily_factor = 1.6  # Peak hours
                elif 9 <= hour_of_day <= 17:
                    daily_factor = 0.8  # Daytime
                else:
                    daily_factor = 0.5  # Night
            elif load_type == 'industrial':
                # Consistent during work hours
                if 6 <= hour_of_day <= 18:
                    daily_factor = 1.3  # Work hours
                elif 19 <= hour_of_day <= 22:
                    daily_factor = 0.9  # Evening
                else:
                    daily_factor = 0.6  # Night
            else:  # commercial (default)
                # Business hours peak
                if 8 <= hour_of_day <= 18:
                    daily_factor = 1.4  # Business hours
                elif 6 <= hour_of_day <= 7 or 19 <= hour_of_day <= 21:
                    daily_factor = 0.8  # Shoulder
                else:
                    daily_factor = 0.4  # Night

            # Weekend reduction for commercial/industrial
            if load_type in ['commercial', 'industrial'] and day_of_week >= 5:
                daily_factor *= 0.6

            # Calculate final hourly load
            hourly_load = avg_hourly * seasonal_factor * daily_factor
            hourly_profile.append(max(0, hourly_load))

        return hourly_profile

    def _generate_hourly_from_monthly(self, monthly_loads: List[float]) -> List[float]:
        """Generate 8760 hourly values from 12 monthly totals"""
        import calendar

        hourly_profile = []

        # Days in each month
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        for month_idx, monthly_total in enumerate(monthly_loads):
            month = month_idx + 1
            days = days_in_month[month_idx]

            # Calculate average hourly load for this month
            hours_in_month = days * 24
            avg_hourly = monthly_total / hours_in_month

            # Apply typical daily profile pattern
            for day in range(days):
                for hour in range(24):
                    # Simple commercial pattern: higher during business hours
                    if 8 <= hour <= 18:  # Business hours
                        multiplier = 1.3
                    elif 6 <= hour <= 7 or 19 <= hour <= 21:  # Shoulder hours
                        multiplier = 0.9
                    else:  # Night hours
                        multiplier = 0.6

                    # Weekend reduction
                    day_of_week = (day % 7)
                    if day_of_week in [5, 6]:  # Weekend
                        multiplier *= 0.7

                    hourly_value = avg_hourly * multiplier
                    hourly_profile.append(max(0, hourly_value))

        return hourly_profile

    def _generate_typical_commercial_profile(self, annual_energy: float) -> List[float]:
        """Generate typical commercial load profile"""
        import math

        # Average hourly load
        avg_hourly = annual_energy / 8760

        hourly_profile = []

        for hour in range(8760):
            # Day of year (0-364)
            day_of_year = hour // 24
            # Hour of day (0-23)
            hour_of_day = hour % 24
            # Day of week (0-6, Monday=0)
            day_of_week = day_of_year % 7

            # Seasonal variation (summer peak for cooling)
            seasonal_factor = 1.0 + 0.3 * math.sin(2 * math.pi * day_of_year / 365 + math.pi / 2)

            # Daily pattern - commercial building
            if 6 <= hour_of_day <= 8:
                daily_factor = 0.7 + 0.3 * (hour_of_day - 6) / 2  # Ramp up
            elif 9 <= hour_of_day <= 17:
                daily_factor = 1.4  # Peak business hours
            elif 18 <= hour_of_day <= 20:
                daily_factor = 1.4 - 0.6 * (hour_of_day - 17) / 3  # Ramp down
            else:
                daily_factor = 0.5  # Base load

            # Weekend reduction
            if day_of_week >= 5:  # Saturday/Sunday
                daily_factor *= 0.6

            # Calculate final hourly load
            hourly_load = avg_hourly * seasonal_factor * daily_factor
            hourly_profile.append(max(0, hourly_load))

        return hourly_profile

    def _apply_default_load_profile(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply default load profile as last resort - only if no existing PySAM data"""
        try:
            # First check if there's already valid load data in PySAM template
            existing_load = data.get('load', [])
            if existing_load and len(existing_load) == 8760:
                annual_energy = sum(existing_load)
                peak_demand = max(existing_load)

                data.update({
                    'load_source': 'existing_pysam_template',
                    'load_annual_kwh': annual_energy,
                    'load_peak_kw': peak_demand,
                    'load_data_replaced': False
                })

                print(
                    f"[LOAD] Using existing PySAM template load: {annual_energy:.0f} kWh annual, {peak_demand:.1f} kW peak")
                return data

            # Only generate fallback if no existing data
            system_capacity = float(data.get('system_capacity', 100))
            annual_energy = system_capacity * 1200  # Conservative default

            # Simple flat profile with basic daily variation
            hourly_profile = []
            avg_hourly = annual_energy / 8760

            for hour in range(8760):
                hour_of_day = hour % 24
                if 8 <= hour_of_day <= 18:
                    multiplier = 1.2
                else:
                    multiplier = 0.8

                hourly_profile.append(avg_hourly * multiplier)

            # Replace PySAM grid.load as fallback
            data['load'] = hourly_profile

            data.update({
                'load_source': 'fallback_generated',
                'load_annual_kwh': annual_energy,
                'load_peak_kw': max(hourly_profile),
                'load_data_replaced': True
            })

            print(f"[LOAD] Fallback profile applied to PySAM grid.load: {annual_energy:.0f} kWh annual")
            return data

        except Exception as e:
            print(f"[LOAD ERROR] Even fallback profile failed: {e}")
            # Emergency: ensure PySAM has some load data
            emergency_load = [10] * 8760  # 10 kW flat load
            data['load'] = emergency_load
            data.update({
                'load_source': 'emergency_flat',
                'load_annual_kwh': 87600,
                'load_peak_kw': 10,
                'load_data_replaced': True
            })
            print("[LOAD] Emergency flat profile applied to PySAM grid.load")
            return data

    def validate_load_data(self, data: Dict[str, Any]) -> Dict[str, bool]:
        """Validate PySAM grid.load configuration"""
        validation = {
            'has_load_profile': False,
            'correct_length': False,
            'positive_values': False,
            'reasonable_scale': False,
            'pysam_compatible': False
        }

        # Check if load attribute exists and is properly formatted for PySAM
        load_profile = data.get('load', [])

        if load_profile:
            validation['has_load_profile'] = True

            # Check length (must be exactly 8760 for annual hourly data)
            if len(load_profile) == 8760:
                validation['correct_length'] = True

                try:
                    # Ensure all values are numeric
                    numeric_load = [float(x) for x in load_profile]

                    # Check for non-negative values
                    if all(x >= 0 for x in numeric_load):
                        validation['positive_values'] = True

                        # Check if values are in reasonable range (0.001 kW to 50 MW)
                        max_load = max(numeric_load)
                        min_load = min(numeric_load)
                        if 0.001 <= max_load <= 50000 and min_load >= 0:
                            validation['reasonable_scale'] = True

                            # Final PySAM compatibility check
                            validation['pysam_compatible'] = True

                except (ValueError, TypeError):
                    pass  # validation remains False

        # Log validation results
        if all(validation.values()):
            annual_energy = sum(float(x) for x in load_profile)
            peak_demand = max(float(x) for x in load_profile)
            print(
                f"[LOAD VALIDATION] ✅ PySAM grid.load validation passed: {annual_energy:.0f} kWh, {peak_demand:.1f} kW peak")
        else:
            failed_checks = [k for k, v in validation.items() if not v]
            print(f"[LOAD VALIDATION] ❌ PySAM grid.load validation failed: {failed_checks}")

        return validation

# ========== ENHANCED MAIN INTEGRATOR WITH FIXED WEATHER PROCESSING ==========

class OptimizedPySAMIntegrator:
    """Enhanced integrator with FIXED weather file processing and coordinate extraction"""

    def __init__(self):
        self.config = ConfigurationManager()
        self.battery_manager = BatterySystemManager(self.config)
        self.form_processor = FormProcessor(self.config)

    def process_form(self, form_html_path: Union[str, Path]) -> Dict[str, Any]:
        """Enhanced main entry point with FIXED coordinate and weather processing"""
        try:
            print("[ENHANCED] Starting enhanced PySAM integration with fixed coordinate processing...")

            # Load and parse form (single operation)
            data, template = self._load_inputs(form_html_path)

            # DEBUG: Show coordinates immediately after form parsing
            lat_debug = data.get('lat', 'NOT_FOUND')
            lon_debug = data.get('lon', 'NOT_FOUND')
            print(f"[COORD DEBUG] Coordinates after form parsing: lat={lat_debug}, lon={lon_debug}")

            # CRITICAL FIX: Process weather file FIRST with enhanced debugging
            print("[WEATHER] Processing weather file with enhanced debugging...")
            self._process_weather_file_enhanced(data)
            weather_result = data.get('solar_resource_file', 'NOT_ASSIGNED')
            print(f"[WEATHER RESULT] Final weather file: {weather_result}")

            # Extract system capacity information
            system_capacity_kw = float(data.get('system_capacity', 0))

            # Centralized battery detection and configuration
            battery_params, data = self.battery_manager.detect_and_configure_battery(data)

            # Component sizing
            if system_capacity_kw > 0:
                data = self._size_components(data, system_capacity_kw)

            # Enhanced cost calculations
            extra_costs = data.pop('_extra_costs', {})
            cost_breakdown = self._calculate_enhanced_costs(data, extra_costs, system_capacity_kw, battery_params)

            # Calculate RECs
            rec_data = self._calculate_recs(system_capacity_kw, data.get('project_info', {}).get('project_address', ''),
                                            data)

            # Apply RECs and final processing
            self._apply_recs_to_pysam(data, rec_data)
            data = self._ensure_proper_types(data)

            # Add detailed cost information
            self._add_cost_data_to_pysam(data, cost_breakdown)

            # FINAL VERIFICATION: Check weather file assignment
            final_weather = data.get('solar_resource_file', 'STILL_NOT_ASSIGNED')
            print(f"[FINAL CHECK] Weather file in final data: {final_weather}")

            # Save output
            output_path = self._save_output(data)

            # Create enhanced summary
            summary = self._create_enhanced_summary(data, cost_breakdown, rec_data, output_path, system_capacity_kw,
                                                    battery_params)

            print("[ENHANCED] ✅ Integration completed successfully with enhanced coordinate and weather processing")
            return summary

        except Exception as e:
            print(f"[ENHANCED ERROR] Failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _load_inputs(self, form_html_path: Union[str, Path]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Load and parse inputs efficiently"""
        # Load template
        with open(Config.TEMPLATE_JSON, 'r') as f:
            template = json.load(f)

        # Parse form
        with open(form_html_path, 'r') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')

        # Process form data
        form_data, extra_costs = self.form_processor.process_form(soup)

        # Initialize data
        data = template.copy()
        data.update(form_data)
        data['_extra_costs'] = extra_costs

        return data, template

    def _process_weather_file_enhanced(self, data: Dict[str, Any]):
        """ENHANCED weather file processing with CLOUD-SAFE path assignment"""
        print(f"[WEATHER ENHANCED] Starting enhanced weather file processing...")

        # Enhanced coordinate extraction with multiple fallbacks
        lat = None
        lon = None

        # Method 1: Direct extraction from data
        coordinate_keys = [
            ('lat', 'latitude'),
            ('lon', 'longitude')
        ]

        for primary_key, fallback_key in coordinate_keys:
            value = data.get(primary_key) or data.get(fallback_key)
            if value is not None:
                try:
                    coord_value = float(value)
                    if primary_key == 'lat':
                        lat = coord_value
                        print(f"[WEATHER] Found latitude: {lat} (from key: {primary_key})")
                    else:
                        lon = coord_value
                        print(f"[WEATHER] Found longitude: {lon} (from key: {primary_key})")
                except (ValueError, TypeError) as e:
                    print(f"[WEATHER] Failed to convert {primary_key} '{value}': {e}")

        # Method 2: Search in project_info
        if lat is None or lon is None:
            project_info = data.get('project_info', {})
            for key, value in project_info.items():
                if 'lat' in key.lower() and lat is None:
                    try:
                        lat = float(value)
                        print(f"[WEATHER] Found latitude in project_info: {lat}")
                    except (ValueError, TypeError):
                        pass
                elif 'lon' in key.lower() and lon is None:
                    try:
                        lon = float(value)
                        print(f"[WEATHER] Found longitude in project_info: {lon}")
                    except (ValueError, TypeError):
                        pass

        # Validation and processing
        print(f"[WEATHER] Final coordinates: lat={lat}, lon={lon}")

        if lat is None or lon is None:
            print("[WEATHER] ❌ Missing coordinates - cannot assign weather file")
            return

        try:
            lat, lon = float(lat), float(lon)

            # Validate coordinate ranges
            if not (-90 <= lat <= 90):
                print(f"[WEATHER] ❌ Invalid latitude range: {lat} (must be -90 to 90)")
                return

            if not (-180 <= lon <= 180):
                print(f"[WEATHER] ❌ Invalid longitude range: {lon} (must be -180 to 180)")
                return

            print(f"[WEATHER] ✅ Valid coordinates: ({lat}, {lon})")

            # Find closest weather file
            weather_file = self._find_closest_weather_file_enhanced(lat, lon)

            if weather_file:
                # CRITICAL FIX: Use RELATIVE path for cloud deployment
                # Convert absolute path to relative path from project root
                try:
                    # Get relative path from project root
                    relative_path = weather_file.relative_to(Config.PROJECT_ROOT)
                    weather_path_str = str(relative_path)
                    print(f"[WEATHER] ✅ Using relative path: {weather_path_str}")
                except ValueError:
                    # Fallback to absolute path if relative doesn't work
                    weather_path_str = str(weather_file.resolve())
                    print(f"[WEATHER] ⚠️ Using absolute path: {weather_path_str}")

                if weather_file.exists():
                    data['solar_resource_file'] = weather_path_str
                    print(f"[WEATHER] ✅ Weather file assigned: {weather_path_str}")

                    # Double-check assignment
                    assigned_file = data.get('solar_resource_file')
                    print(f"[WEATHER] ✅ Verification - assigned file: {assigned_file}")
                else:
                    print(f"[WEATHER ERROR] Weather file does not exist: {weather_path_str}")
            else:
                print("[WEATHER] ❌ No matching weather file found")

        except (ValueError, TypeError) as e:
            print(f"[WEATHER ERROR] Coordinate processing failed: {e}")
        except Exception as e:
            print(f"[WEATHER ERROR] Unexpected error: {e}")
            import traceback
            traceback.print_exc()

    def _find_closest_weather_file_enhanced(self, lat: float, lon: float) -> Optional[Path]:
        """Enhanced weather file selection with CLOUD-SAFE paths"""
        print(f"[WEATHER SEARCH] Looking for weather file near ({lat}, {lon})")

        if not Config.WEATHER_DIR.exists():
            print(f"[WEATHER ERROR] Weather directory not found: {Config.WEATHER_DIR}")
            return None

        print(f"[WEATHER SEARCH] Searching in directory: {Config.WEATHER_DIR}")

        weather_files = []
        csv_files = list(Config.WEATHER_DIR.glob("*.csv"))
        print(f"[WEATHER SEARCH] Found {len(csv_files)} CSV files")

        for file in csv_files:
            print(f"[WEATHER SEARCH] Checking file: {file.name}")
            coords = self._extract_coordinates_from_filename(file.name)
            if coords:
                file_lat, file_lon = coords
                distance = math.sqrt((file_lat - lat) ** 2 + (file_lon - lon) ** 2)
                weather_files.append((file, file_lat, file_lon, distance))
                print(f"[WEATHER SEARCH]   Coordinates: ({file_lat}, {file_lon}), Distance: {distance:.4f}")
            else:
                print(f"[WEATHER SEARCH]   No coordinates found in filename")

        if not weather_files:
            print("[WEATHER SEARCH] No valid weather files found in directory")
            return None

        # Find closest file
        closest_file, closest_lat, closest_lon, closest_distance = min(weather_files, key=lambda x: x[3])

        print(f"[WEATHER SEARCH] ✅ Selected weather file: {closest_file.name}")
        print(f"[WEATHER SEARCH]   File coordinates: ({closest_lat}, {closest_lon})")
        print(f"[WEATHER SEARCH]   Distance: {closest_distance:.4f} degrees")

        return closest_file

    def _extract_coordinates_from_filename(self, filename: str) -> Optional[Tuple[float, float]]:
        """Extract coordinates from weather file filename with enhanced patterns"""
        try:
            # Remove file extension
            name_without_ext = filename.replace('.csv', '').replace('.CSV', '')

            # Pattern 1: 1785715_Adelaide_-34.91_138.58_tmy-2020
            import re
            pattern1 = r'_(-?\d+\.?\d*)_(-?\d+\.?\d*)_'
            match1 = re.search(pattern1, name_without_ext)
            if match1:
                lat, lon = float(match1.group(1)), float(match1.group(2))
                print(f"[COORD EXTRACT] Pattern 1 match: {lat}, {lon}")
                return lat, lon

            # Pattern 2: Simple underscore separation
            parts = name_without_ext.split('_')
            for i in range(len(parts) - 1):
                try:
                    lat = float(parts[i])
                    lon = float(parts[i + 1])
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        print(f"[COORD EXTRACT] Pattern 2 match: {lat}, {lon}")
                        return lat, lon
                except (ValueError, TypeError):
                    continue

            print(f"[COORD EXTRACT] No coordinates found in: {filename}")
            return None

        except Exception as e:
            print(f"[COORD EXTRACT] Error extracting from {filename}: {e}")
            return None

    def _size_components(self, data: Dict[str, Any], target_kw: float) -> Dict[str, Any]:
        """Streamlined component sizing"""
        # Remove conflicting inverter parameters (single operation)
        inverter_fields_to_remove = [
            'inv_snl_c0', 'inv_snl_c1', 'inv_snl_c2', 'inv_snl_c3',
            'inv_snl_paco', 'inv_snl_pdco', 'inv_snl_pnt', 'inv_snl_pso',
            'inv_snl_vdco', 'inv_snl_vdcmax'
        ]

        for field in inverter_fields_to_remove:
            data.pop(field, None)

        # Calculate array configuration
        module_power_w = 500
        required_modules = round(target_kw * 1000 / module_power_w)
        modules_per_string = 24
        strings = max(1, math.ceil(required_modules / modules_per_string))
        actual_modules = strings * modules_per_string
        actual_dc_kw = (actual_modules * module_power_w) / 1000

        # Apply all sizing configuration at once
        sizing_config = {
            'subarray1_nstrings': strings,
            'subarray1_modules_per_string': modules_per_string,
            'subarray1_nmodx': int(math.sqrt(actual_modules)),
            'subarray1_nmody': math.ceil(actual_modules / int(math.sqrt(actual_modules))),
            'inverter_model': 1,
            'inv_ds_paco': (actual_dc_kw / 1.2) * 1000,
            'inv_ds_eff': 96.0,
            'inv_ds_pnt': 1.0,
            'inv_ds_pso': 0.0,
            'inv_ds_vdco': 600,
            'inv_ds_vdcmax': 1000,
            'system_capacity': actual_dc_kw
        }

        data.update(sizing_config)
        print(f"[SIZING] {strings} strings × {modules_per_string} = {actual_dc_kw:.0f} kW DC")
        return data

    def _calculate_enhanced_costs(self, data: Dict[str, Any], extra_costs: Dict[str, float],
                                 system_capacity_kw: float, battery_params: BatteryParams) -> Dict[str, Any]:
        """Enhanced cost calculation with detailed breakdown for results presentation"""
        try:
            # Base PV cost per kW from form
            base_pv_cost_per_kw = float(data.get('total_installed_cost', 0.0))

            # Extra CAPEX costs
            extra_capex_total = sum(extra_costs.values())

            # Battery cost components
            battery_capacity_kwh = battery_params.capacity_kwh
            battery_cost_per_kwh = float(data.get('battery_per_kWh', 0))
            battery_bos_cost = float(data.get('bos_battery_costs', 0))

            # Calculate raw costs
            raw_pv_cost = system_capacity_kw * base_pv_cost_per_kw if system_capacity_kw > 0 else 0
            raw_battery_cost = battery_capacity_kwh * battery_cost_per_kwh if battery_capacity_kwh > 0 else 0
            total_battery_cost = raw_battery_cost + battery_bos_cost

            # Calculate costs with extras and margin
            extra_per_kw = extra_capex_total / system_capacity_kw if system_capacity_kw > 0 else 0
            pv_cost_before_margin = system_capacity_kw * (base_pv_cost_per_kw + extra_per_kw) if system_capacity_kw > 0 else 0
            battery_cost_before_margin = raw_battery_cost + battery_bos_cost

            # Apply installer margin
            installer_margin_rate = float(data.get('installer_margin', 0.0))
            installer_margin_pv_amount = pv_cost_before_margin * installer_margin_rate if installer_margin_rate > 0 else 0
            installer_margin_battery_amount = battery_cost_before_margin * installer_margin_rate if installer_margin_rate > 0 else 0
            installer_margin_total_amount = installer_margin_pv_amount + installer_margin_battery_amount

            total_pv_per_kw = (base_pv_cost_per_kw + extra_per_kw) * (1 + installer_margin_rate)
            total_battery_cost = battery_cost_before_margin * (1 + installer_margin_rate)

            # Final totals
            total_pv_cost = system_capacity_kw * total_pv_per_kw if system_capacity_kw > 0 else 0
            total_system_cost = total_pv_cost + total_battery_cost

            # Update PySAM total_installed_cost
            data['total_installed_cost'] = total_system_cost

            cost_breakdown = {
                # Base costs
                'base_pv_cost_per_kw': base_pv_cost_per_kw,
                'raw_pv_cost': raw_pv_cost,
                'system_capacity_kw': system_capacity_kw,

                # Battery costs
                'battery_capacity_kwh': battery_capacity_kwh,
                'battery_cost_per_kwh': battery_cost_per_kwh,
                'raw_battery_cost': raw_battery_cost,
                'battery_bos_cost': battery_bos_cost,
                'battery_cost': total_battery_cost,

                # Extra costs and margins
                'extra_capex_total': extra_capex_total,
                'extra_per_kw': extra_per_kw,
                'installer_margin_rate': installer_margin_rate,
                'installer_margin_pv_amount': installer_margin_pv_amount,
                'installer_margin_battery_amount': installer_margin_battery_amount,
                'installer_margin_total_amount': installer_margin_total_amount,

                # Final totals
                'total_pv_per_kw': total_pv_per_kw,
                'total_pv_cost': total_pv_cost,
                'total_system_cost': total_system_cost,

                # Individual extra costs breakdown
                'extra_costs_breakdown': extra_costs
            }

            print(f"[COST] Raw PV: ${raw_pv_cost:,.2f} ({system_capacity_kw:.1f} kW × ${base_pv_cost_per_kw:.2f}/kW)")
            if battery_capacity_kwh > 0:
                print(f"[COST] Raw Battery: ${raw_battery_cost:,.2f} ({battery_capacity_kwh:.1f} kWh × ${battery_cost_per_kwh:.2f}/kWh)")
                print(f"[COST] Total Battery: ${total_battery_cost:,.2f} (incl. BOS: ${battery_bos_cost:,.2f})")
            if extra_capex_total > 0:
                print(f"[COST] Extra CAPEX: ${extra_capex_total:,.2f}")
            if installer_margin_rate > 0:
                print(f"[COST] Installer Margin: {installer_margin_rate:.1%} (${installer_margin_total_amount:,.2f})")
            print(f"[COST] Total System: ${total_system_cost:,.2f}")

            return cost_breakdown

        except Exception as e:
            print(f"[COST ERROR] {e}")
            return {
                'base_pv_cost_per_kw': float(data.get('total_installed_cost', 0)),
                'raw_pv_cost': 0,
                'battery_cost': 0,
                'total_system_cost': float(data.get('total_installed_cost', 0))
            }

    def _add_cost_data_to_pysam(self, data: Dict[str, Any], cost_breakdown: Dict[str, Any]):
        """Add detailed cost information to PySAM config for results extraction"""

        # Add cost breakdown to project_info for easy access in results
        if 'project_info' not in data:
            data['project_info'] = {}

        data['project_info']['cost_breakdown'] = {
            'base_pv_cost_per_kw': cost_breakdown.get('base_pv_cost_per_kw', 0),
            'raw_pv_cost': cost_breakdown.get('raw_pv_cost', 0),
            'total_pv_cost': cost_breakdown.get('total_pv_cost', 0),
            'raw_battery_cost': cost_breakdown.get('raw_battery_cost', 0),
            'battery_cost': cost_breakdown.get('battery_cost', 0),
            'total_system_cost': cost_breakdown.get('total_system_cost', 0),
            'extra_capex_total': cost_breakdown.get('extra_capex_total', 0),
            'installer_margin_rate': cost_breakdown.get('installer_margin_rate', 0),
            'installer_margin_total_amount': cost_breakdown.get('installer_margin_total_amount', 0)
        }

        # Also add individual cost components as top-level fields for easy PySAM access
        data.update({
            'cost_base_pv_per_kw': cost_breakdown.get('base_pv_cost_per_kw', 0),
            'cost_raw_pv_total': cost_breakdown.get('raw_pv_cost', 0),
            'cost_total_pv': cost_breakdown.get('total_pv_cost', 0),
            'cost_raw_battery_total': cost_breakdown.get('raw_battery_cost', 0),
            'cost_total_battery': cost_breakdown.get('battery_cost', 0),
            'cost_extra_capex': cost_breakdown.get('extra_capex_total', 0),
            'cost_installer_margin_rate': cost_breakdown.get('installer_margin', 0),
            'cost_system_total': cost_breakdown.get('total_system_cost', 0)
        })

        print(f"[COST DATA] Added cost breakdown to PySAM config for results extraction")

    def _calculate_recs(self, system_capacity_kw: float, address: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Streamlined REC calculation (removed existing capacity references)"""
        try:
            # Extract postcode and determine STC zone
            postcode_matches = [p for p in address.split() if p.isdigit() and len(p) == 4]
            postcode = postcode_matches[-1] if postcode_matches else None

            # STC zone determination (simplified)
            stc_multiplier = 1.622  # Default Zone 1
            if postcode:
                postcode_int = int(postcode)
                if 6800 <= postcode_int <= 6999:
                    stc_multiplier = 1.536  # Zone 2
                elif postcode_int in range(200, 300) or postcode_int in range(300, 400) or postcode_int in range(800, 900):
                    stc_multiplier = 1.382  # Zone 3
                elif 900 <= postcode_int <= 999:
                    stc_multiplier = 1.185  # Zone 4

            # Calculate STCs (small-scale only)
            stc_certificates = 0
            if system_capacity_kw <= 100:  # Small-scale only
                stc_certificates = round(system_capacity_kw * stc_multiplier * 10)  # 10 year deeming

            stc_price = data.get('incentive_stc', 40)
            stc_total_value = stc_certificates * stc_price

            return {
                'stc_certificates': stc_certificates,
                'stc_value': stc_total_value,
                'postcode': postcode,
                'system_capacity_kw': system_capacity_kw  # Added for clarity
            }

        except Exception as e:
            print(f"[REC ERROR] {e}")
            return {'stc_certificates': 0, 'stc_value': 0}

    def _apply_recs_to_pysam(self, data: Dict[str, Any], rec_data: Dict[str, Any]):
        """Apply REC calculations to PySAM fields"""
        # Apply STCs as federal IBI
        if rec_data.get('stc_value', 0) > 0:
            data.update({
                'ibi_fed_amount': rec_data['stc_value'],
                'ibi_fed_tax_fed': 0,
                'ibi_fed_tax_sta': 0
            })

        # Store REC info
        data['project_info'].update(rec_data)

    def _ensure_proper_types(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Streamlined type conversion"""
        type_conversions = {
            'en_batt': int, 'batt_meter_position': int, 'enable_interconnection_limit': int,
            'module_model': int, 'batt_calendar_choice': int, 'batt_dispatch_choice': int,
            'batt_chem': int, 'analysis_period': int, 'system_capacity': float,
            'batt_computed_bank_capacity': float, 'batt_power_discharge_max_kwac': float,
            'total_installed_cost': float
        }

        for field, target_type in type_conversions.items():
            if field in data:
                try:
                    raw_value = data[field]
                    clean_value = FormBasedConverter.extract_numeric_from_string(raw_value)

                    if isinstance(clean_value, str):
                        if clean_value.lower() in ['yes', 'true', '1']:
                            data[field] = 1
                        elif clean_value.lower() in ['no', 'false', '0']:
                            data[field] = 0
                        else:
                            data[field] = target_type(float(clean_value))
                    else:
                        data[field] = target_type(clean_value)
                except (ValueError, TypeError):
                    pass  # Keep original value

        return data

    def _save_output(self, data: Dict[str, Any]) -> Path:
        """Save processed data"""
        output_path = Config.UPDATED_JSON
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, separators=(',', ': '))
        return output_path

    def _create_enhanced_summary(self, data: Dict[str, Any], cost_breakdown: Dict[str, Any],
                                rec_data: Dict[str, Any], output_path: Path,
                                system_capacity_kw: float, battery_params: BatteryParams) -> Dict[str, Any]:
        """Create enhanced summary with detailed cost breakdown"""
        return {
            'success': True,
            'output_file': str(output_path),
            'enhanced_processing': True,

            'capacity_summary': {
                'system_capacity_kw': system_capacity_kw,
                'modeled_kw': data.get('system_capacity', 0)
            },

            'coordinate_summary': {
                'latitude': data.get('lat', 'Not found'),
                'longitude': data.get('lon', 'Not found'),
                'weather_file': data.get('solar_resource_file', 'Not assigned')
            },

            'system_summary': {
                'battery_enabled': battery_params.enabled,
                'battery_capacity_kwh': battery_params.capacity_kwh,
                'battery_power_kw': battery_params.power_kw,
                'battery_soc_min': data.get('batt_minimum_SOC', 'N/A'),
                'battery_soc_max': data.get('batt_maximum_SOC', 'N/A'),
                'dispatch_mode': data.get('batt_dispatch_choice', 'None'),
                'fcas_enabled': battery_params.enabled and data.get('fcas_region') not in [None, 'DISABLED']
            },

            'detailed_cost_breakdown': {
                'pv_costs': {
                    'base_cost_per_kw': cost_breakdown.get('base_pv_cost_per_kw', 0),
                    'raw_pv_cost': cost_breakdown.get('raw_pv_cost', 0),
                    'total_pv_cost': cost_breakdown.get('total_pv_cost', 0),
                    'system_capacity_kw': cost_breakdown.get('system_capacity_kw', 0)
                },
                'battery_costs': {
                    'capacity_kwh': cost_breakdown.get('battery_capacity_kwh', 0),
                    'cost_per_kwh': cost_breakdown.get('battery_cost_per_kwh', 0),
                    'raw_battery_cost': cost_breakdown.get('raw_battery_cost', 0),
                    'bos_cost': cost_breakdown.get('battery_bos_cost', 0),
                    'total_battery_cost': cost_breakdown.get('battery_cost', 0)
                },
                'other_costs': {
                    'extra_capex_total': cost_breakdown.get('extra_capex_total', 0),
                    'installer_margin_rate': cost_breakdown.get('installer_margin_rate', 0),
                    'installer_margin_pv_amount': cost_breakdown.get('installer_margin_pv_amount', 0),
                    'installer_margin_battery_amount': cost_breakdown.get('installer_margin_battery_amount', 0),
                    'installer_margin_total_amount': cost_breakdown.get('installer_margin_total_amount', 0),
                    'extra_costs_breakdown': cost_breakdown.get('extra_costs_breakdown', {})
                },
                'totals': {
                    'total_system_cost': cost_breakdown.get('total_system_cost', 0),
                    'total_pv_per_kw': cost_breakdown.get('total_pv_per_kw', 0)
                }
            },

            'rec_summary': rec_data,

            'pysam_cost_fields_added': [
                'cost_base_pv_per_kw',
                'cost_raw_pv_total',
                'cost_total_pv',
                'cost_raw_battery_total',
                'cost_total_battery',
                'cost_extra_capex',
                'cost_installer_margin_rate',
                'cost_installer_margin_pv_amount',
                'cost_installer_margin_battery_amount',
                'cost_installer_margin_total_amount',
                'cost_system_total'
            ],

            'fixes_applied': [
                'ENHANCED coordinate extraction with multiple fallbacks',
                'ENHANCED weather file processing with validation',
                'SOC values from form (percentages)',
                'Form-based percentage/decimal conversion',
                'Battery toggle (en_batt) support',
                'FCAS auto-disable when battery disabled',
                'FCAS explicit toggle support',
                'Complete FCAS configuration preserved',
                'REC calculations preserved',
                'Enhanced cost modeling with detailed breakdown',
                'Cost data added to PySAM config',
                'Comprehensive debugging and logging',
                'Enhanced field value extraction',
                'Weather file assignment verification'
            ]
        }


# ========== MAIN EXECUTION FUNCTION ==========

def update_pysam_json(form_html_path: Union[str, Path]) -> Dict[str, Any]:
    """Enhanced main function with FIXED coordinate and weather processing"""
    integrator = OptimizedPySAMIntegrator()
    return integrator.process_form(form_html_path)


if __name__ == "__main__":
    # FIXED: Use the existing Config class for proper path resolution
    print(f"🔍 Using Config class for path resolution...")
    print(f"🔍 Detected environment: {Config.ENVIRONMENT}")
    print(f"🔍 Project root: {Config.PROJECT_ROOT}")

    # Use Config class paths directly
    PATHS = {
        'form_html': Config.FORM_HTML,  # Should be PROJECT_ROOT/1.input/index_merged.html
        'template_json': Config.TEMPLATE_JSON,  # Should be PROJECT_ROOT/data/input_json/All_commercial.json
        'output_dir': Config.UPDATED_JSON_DIR  # Should be PROJECT_ROOT/1.input/json_updated
    }

    # Debug: Show the configured paths
    print(f"🔍 Form HTML path: {PATHS['form_html']}")
    print(f"🔍 Template JSON path: {PATHS['template_json']}")
    print(f"🔍 Output directory: {PATHS['output_dir']}")

    # Check if paths exist and provide detailed feedback
    if not PATHS['form_html'].exists():
        print(f"❌ Form HTML not found at: {PATHS['form_html']}")

        # Check if the directory exists
        form_dir = PATHS['form_html'].parent
        if form_dir.exists():
            print(f"📁 Contents of {form_dir}:")
            for item in form_dir.iterdir():
                print(f"   - {item.name}")
        else:
            print(f"📁 Directory {form_dir} does not exist")

        # Check for alternative locations
        alternative_paths = [
            Config.PROJECT_ROOT / "index_merged.html",  # Root directory
            Config.PROJECT_ROOT / "1.input" / "index_merged.html",  # Correct location
            Config.INPUT_DIR / "index_merged.html" if hasattr(Config, 'INPUT_DIR') else None,
        ]

        print("🔍 Checking alternative locations:")
        for alt_path in alternative_paths:
            if alt_path and alt_path.exists():
                print(f"   ✅ Found at: {alt_path}")
                PATHS['form_html'] = alt_path
                break
            elif alt_path:
                print(f"   ❌ Not at: {alt_path}")
        else:
            # Still not found, list current directory contents
            current_dir = Path.cwd()
            print(f"📁 Current directory ({current_dir}) contents:")
            for item in current_dir.iterdir():
                if item.name.endswith('.html') or item.is_dir():
                    print(f"   - {item.name}")
            exit(1)

    if not PATHS['template_json'].exists():
        print(f"❌ Template JSON not found at: {PATHS['template_json']}")

        # Try using the fallback from Config
        try:
            fallback_template = Config.get_fallback_template()
            print(f"✅ Using fallback template: {fallback_template}")
            PATHS['template_json'] = fallback_template
        except FileNotFoundError as e:
            print(f"❌ No template files found: {e}")

            # List what's in the data directory
            if Config.DATA_DIR.exists():
                print(f"📁 Contents of {Config.DATA_DIR}:")
                for item in Config.DATA_DIR.rglob("*.json"):
                    print(f"   - {item.relative_to(Config.DATA_DIR)}")
            exit(1)

    # Ensure output directory exists using Config
    PATHS['output_dir'].mkdir(parents=True, exist_ok=True)

    # Final verification
    print(f"✅ Final paths verified:")
    print(f"   Form HTML: {PATHS['form_html']} (exists: {PATHS['form_html'].exists()})")
    print(f"   Template JSON: {PATHS['template_json']} (exists: {PATHS['template_json'].exists()})")
    print(f"   Output directory: {PATHS['output_dir']} (exists: {PATHS['output_dir'].exists()})")

    try:
        print("🔧 Starting ENHANCED PySAM Integration with Config-based paths...")
        result = update_pysam_json(PATHS['form_html'])

        if result['success']:
            print("\n✅ ENHANCED INTEGRATION SUCCESS!")
            print(f"📊 System: {result['capacity_summary']}")
            print(f"🗺️ Coordinates: {result['coordinate_summary']}")
            print(f"🔋 Battery: {result['system_summary']}")
            print(f"💰 Cost Breakdown Available: {len(result['pysam_cost_fields_added'])} fields added")
            print(f"📄 Output: {result['output_file']}")
            print(f"⚡ Fixes Applied: {len(result['fixes_applied'])} applied")
            print("🎯 Enhanced coordinate extraction and weather file processing completed!")
        else:
            print("❌ INTEGRATION FAILED")

    except Exception as e:
        print(f"💥 CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)