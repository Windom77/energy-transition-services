"""
Clean Deployment-Ready Configuration Script
- Google Cloud App Engine compatible paths
- No hardcoded local machine paths
- Robust fallbacks for cloud environment
- Environment-aware path resolution
- Maintains all original functionality
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict, Any


class Config:
    """
    Environment-aware configuration management for Google Cloud deployment
    Automatically detects local vs cloud environment and sets appropriate paths
    """

    # ========== ENVIRONMENT DETECTION ==========

    @classmethod
    def _detect_environment(cls) -> str:
        """Detect if running locally or in Google Cloud"""
        # Google App Engine sets GAE_ENV
        if os.getenv('GAE_ENV'):
            return 'gcloud'

        # Check for Google Cloud Run
        if os.getenv('K_SERVICE'):
            return 'gcloud'

        # Check for common cloud indicators
        if os.getenv('GOOGLE_CLOUD_PROJECT'):
            return 'gcloud'

        # Check if we're in a container with /app directory
        if os.path.exists('/app') and os.getcwd().startswith('/app'):
            return 'container'

        return 'local'

    @classmethod
    def _get_project_root(cls) -> Path:
        """
        Intelligently determine project root based on environment
        Works in both local development and cloud deployment
        """
        environment = cls._detect_environment()

        if environment in ['gcloud', 'container']:
            # In cloud/container: start from /app or current working directory
            if os.path.exists('/app'):
                return Path('/app')
            else:
                return Path.cwd()
        else:
            # Local development: find project root by looking for marker files
            current_path = Path(__file__).resolve()

            # Look for project markers in parent directories
            for parent in [current_path.parent] + list(current_path.parents):
                if any((parent / marker).exists() for marker in [
                    'main.py', 'requirements.txt', 'app.yaml', '.git',
                    '1.input', '2.pysam', '3.output'
                ]):
                    return parent

            # Fallback to script's parent directory
            return current_path.parent

    # ========== PATH CONFIGURATION ==========

    # Environment detection - Fixed: Call classmethods properly
    @classmethod
    def _get_environment(cls) -> str:
        """Get current environment"""
        return cls._detect_environment()

    @classmethod
    def _get_root(cls) -> Path:
        """Get project root"""
        return cls._get_project_root()

    # Initialize class variables - Fixed: Use properties instead of direct classmethod calls
    _ENVIRONMENT = None
    _PROJECT_ROOT = None

    @classmethod
    def _initialize_paths(cls):
        """Initialize environment and paths on first access"""
        if cls._ENVIRONMENT is None:
            cls._ENVIRONMENT = cls._detect_environment()
        if cls._PROJECT_ROOT is None:
            cls._PROJECT_ROOT = cls._get_project_root()

    @property
    @classmethod
    def ENVIRONMENT(cls) -> str:
        """Get environment with lazy initialization"""
        cls._initialize_paths()
        return cls._ENVIRONMENT

    @property
    @classmethod
    def PROJECT_ROOT(cls) -> Path:
        """Get project root with lazy initialization"""
        cls._initialize_paths()
        return cls._PROJECT_ROOT

    # ========== PATH CONFIGURATION ==========

    # Class variables will be set after class definition
    ENVIRONMENT = None
    PROJECT_ROOT = None

    # Core directories (will be set after initialization)
    INPUT_DIR = None
    PYSAM_DIR = None
    OUTPUT_DIR = None
    DATA_DIR = None

    # Data subdirectories
    WEATHER_DIR = None
    INPUT_JSON_DIR = None
    PROFILES_DIR = None

    # Template files with fallbacks
    TEMPLATE_JSON = None
    TEMPLATE_JSON_FALLBACK = None

    # Input files (with fallbacks)
    FORM_HTML = None
    FORM_HTML_FALLBACK = None

    # Output files
    UPDATED_JSON_DIR = None
    UPDATED_JSON = None

    # Results directory
    RESULTS_DIR = None

    # Static assets (with fallbacks for migration)
    STATIC_DIR = None
    TEMPLATES_DIR = None

    # Legacy paths for backwards compatibility
    LEGACY_STATIC_DIR = None
    LEGACY_TEMPLATES_DIR = None

    # Form configuration
    FORM_CONFIG_JSON = None

    # Additional required paths
    BASE_JSON = None

    @classmethod
    def _initialize_all_paths(cls):
        """Initialize all paths after class creation"""
        # Set environment and project root
        cls.ENVIRONMENT = cls._detect_environment()
        cls.PROJECT_ROOT = cls._get_project_root()

        # Core directories
        cls.INPUT_DIR = cls.PROJECT_ROOT / '1.input'
        cls.PYSAM_DIR = cls.PROJECT_ROOT / '2.pysam'
        cls.OUTPUT_DIR = cls.PROJECT_ROOT / '3.output'
        cls.DATA_DIR = cls.PROJECT_ROOT / 'data'

        # Data subdirectories
        cls.WEATHER_DIR = cls.DATA_DIR / 'weather'
        cls.INPUT_JSON_DIR = cls.DATA_DIR / 'input_json'
        cls.PROFILES_DIR = cls.DATA_DIR / 'profiles'

        # Template files with fallbacks
        cls.TEMPLATE_JSON = cls.INPUT_JSON_DIR / 'All_commercial.json'
        cls.TEMPLATE_JSON_FALLBACK = cls.INPUT_JSON_DIR / 'pvsamv1.json'

        # Input files (with fallbacks)
        cls.FORM_HTML = cls.INPUT_DIR / 'index_merged.html'
        cls.FORM_HTML_FALLBACK = cls.INPUT_DIR / 'templates' / 'index_base2.html'

        # Output files
        cls.UPDATED_JSON_DIR = cls.INPUT_DIR / 'json_updated'
        cls.UPDATED_JSON = cls.UPDATED_JSON_DIR / 'All_commercial_updated.json'

        # Results directory
        cls.RESULTS_DIR = cls.PYSAM_DIR / 'results'

        # UPDATED: Check for templates in new structure (root/templates) first
        if (cls.PROJECT_ROOT / 'templates').exists():
            cls.TEMPLATES_DIR = cls.PROJECT_ROOT / 'templates'  # NEW LOCATION
            cls.STATIC_DIR = cls.PROJECT_ROOT / 'static'        # NEW LOCATION
            cls.LEGACY_TEMPLATES_DIR = cls.INPUT_DIR / 'templates'  # OLD LOCATION
            cls.LEGACY_STATIC_DIR = cls.INPUT_DIR / 'static'        # OLD LOCATION
        else:
            # Fallback to old structure
            cls.TEMPLATES_DIR = cls.INPUT_DIR / 'templates'     # OLD LOCATION
            cls.STATIC_DIR = cls.INPUT_DIR / 'static'           # OLD LOCATION
            cls.LEGACY_TEMPLATES_DIR = cls.PROJECT_ROOT / 'templates'  # FALLBACK
            cls.LEGACY_STATIC_DIR = cls.PROJECT_ROOT / 'static'        # FALLBACK

        # Form configuration - check both locations
        if (cls.STATIC_DIR / 'js' / 'form_config.json').exists():
            cls.FORM_CONFIG_JSON = cls.STATIC_DIR / 'js' / 'form_config.json'
        elif (cls.LEGACY_STATIC_DIR / 'js' / 'form_config.json').exists():
            cls.FORM_CONFIG_JSON = cls.LEGACY_STATIC_DIR / 'js' / 'form_config.json'
        else:
            cls.FORM_CONFIG_JSON = cls.STATIC_DIR / 'js' / 'form_config.json'  # Default

        # Additional required paths
        cls.BASE_JSON = cls.INPUT_JSON_DIR / 'All_commercial.json'

    # ========== ENVIRONMENT-SPECIFIC SETTINGS ==========

    @classmethod
    def get_debug_mode(cls) -> bool:
        """Get debug mode based on environment"""
        if cls.ENVIRONMENT == 'gcloud':
            return False
        return os.getenv('DEBUG', 'true').lower() == 'true'

    @classmethod
    def get_weather_file_by_coordinates(cls, lat: float, lon: float) -> Optional[Path]:
        """
        Find closest weather file by coordinates
        Cloud-safe implementation with error handling
        """
        try:
            if not cls.WEATHER_DIR.exists():
                print(f"⚠️ Weather directory not found: {cls.WEATHER_DIR}")
                return None

            weather_files = list(cls.WEATHER_DIR.glob("*.csv"))
            if not weather_files:
                print(f"⚠️ No weather files found in: {cls.WEATHER_DIR}")
                return None

            # Find closest file by coordinates
            closest_file = None
            min_distance = float('inf')

            for weather_file in weather_files:
                file_coords = cls._extract_coordinates_from_filename(weather_file.name)
                if file_coords:
                    distance = ((file_coords[0] - lat) ** 2 + (file_coords[1] - lon) ** 2) ** 0.5
                    if distance < min_distance:
                        min_distance = distance
                        closest_file = weather_file

            if closest_file:
                print(f"🌦️ Selected weather file: {closest_file.name} (distance: {min_distance:.2f})")
                return closest_file
            else:
                print("⚠️ No weather files with valid coordinates found")
                return None

        except Exception as e:
            print(f"❌ Error finding weather file: {e}")
            return None

    @classmethod
    def _extract_coordinates_from_filename(cls, filename: str) -> Optional[Tuple[float, float]]:
        """Extract coordinates from weather filename"""
        try:
            # Remove .csv extension and split by underscore
            parts = filename.replace('.csv', '').split('_')

            # Look for numeric parts that could be coordinates
            coords = []
            for part in parts:
                try:
                    # Handle negative numbers and decimals
                    coord = float(part)
                    coords.append(coord)
                except ValueError:
                    continue

            # Return last two coordinates (usually lat, lon)
            if len(coords) >= 2:
                return (coords[-2], coords[-1])
            return None

        except Exception:
            return None

    @classmethod
    def ensure_directories(cls) -> None:
        """Ensure all required directories exist"""
        directories = [
            cls.INPUT_DIR,
            cls.PYSAM_DIR,
            cls.OUTPUT_DIR,
            cls.DATA_DIR,
            cls.WEATHER_DIR,
            cls.INPUT_JSON_DIR,
            cls.PROFILES_DIR,
            cls.UPDATED_JSON_DIR,
            cls.RESULTS_DIR,
            cls.STATIC_DIR,
            cls.TEMPLATES_DIR
        ]

        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"⚠️ Could not create directory {directory}: {e}")

        if cls.get_debug_mode():
            print(f"✅ Directory structure verified from {cls.PROJECT_ROOT}")

    @classmethod
    def validate_deployment_readiness(cls) -> dict:
        """
        Validate that the configuration is ready for deployment
        Returns comprehensive status for troubleshooting
        """
        validation_results = {
            'environment': cls.ENVIRONMENT,
            'project_root': str(cls.PROJECT_ROOT),
            'debug_mode': cls.get_debug_mode(),
            'critical_paths': {},
            'weather_files': {},
            'template_files': {},
            'ready_for_deployment': True,
            'warnings': [],
            'errors': []
        }

        # Check critical paths
        critical_paths = {
            'INPUT_DIR': cls.INPUT_DIR,
            'PYSAM_DIR': cls.PYSAM_DIR,
            'DATA_DIR': cls.DATA_DIR,
            'WEATHER_DIR': cls.WEATHER_DIR,
            'TEMPLATE_JSON': cls.TEMPLATE_JSON,
            'STATIC_DIR': cls.STATIC_DIR,
            'TEMPLATES_DIR': cls.TEMPLATES_DIR
        }

        for name, path in critical_paths.items():
            exists = path.exists()
            validation_results['critical_paths'][name] = {
                'path': str(path),
                'exists': exists,
                'absolute_path': str(path.resolve()) if exists else None
            }

            if not exists:
                validation_results['errors'].append(f"Missing critical path: {name} ({path})")
                validation_results['ready_for_deployment'] = False

        # Check weather files
        if cls.WEATHER_DIR.exists():
            weather_files = list(cls.WEATHER_DIR.glob("*.csv"))
            validation_results['weather_files'] = {
                'count': len(weather_files),
                'files': [f.name for f in weather_files[:5]],  # First 5 files
                'coordinates_valid': []
            }

            for weather_file in weather_files[:3]:  # Check first 3 files
                coords = cls._extract_coordinates_from_filename(weather_file.name)
                validation_results['weather_files']['coordinates_valid'].append({
                    'file': weather_file.name,
                    'coordinates': coords
                })

            if len(weather_files) == 0:
                validation_results['errors'].append("No weather files found")
                validation_results['ready_for_deployment'] = False
        else:
            validation_results['errors'].append("Weather directory not found")
            validation_results['ready_for_deployment'] = False

        # Check template files
        template_files = {
            'TEMPLATE_JSON': cls.TEMPLATE_JSON,
            'TEMPLATE_JSON_FALLBACK': cls.TEMPLATE_JSON_FALLBACK,
            'FORM_CONFIG_JSON': cls.FORM_CONFIG_JSON
        }

        for name, path in template_files.items():
            exists = path.exists()
            validation_results['template_files'][name] = {
                'path': str(path),
                'exists': exists
            }

            if not exists:
                validation_results['warnings'].append(f"Template file not found: {name} ({path})")

        # Environment-specific checks
        if cls.ENVIRONMENT == 'gcloud':
            # Cloud-specific validations
            if not Path('/app').exists():
                validation_results['warnings'].append("Not running in expected /app directory")

        # Summary
        validation_results['summary'] = {
            'total_errors': len(validation_results['errors']),
            'total_warnings': len(validation_results['warnings']),
            'deployment_ready': validation_results['ready_for_deployment']
        }

        return validation_results

    @classmethod
    def get_fallback_template(cls) -> Path:
        """Get fallback template file with robust checking"""
        if cls.TEMPLATE_JSON.exists():
            return cls.TEMPLATE_JSON
        elif cls.TEMPLATE_JSON_FALLBACK.exists():
            print(f"⚠️ Using fallback template: {cls.TEMPLATE_JSON_FALLBACK}")
            return cls.TEMPLATE_JSON_FALLBACK
        else:
            raise FileNotFoundError("No template JSON files found")

    @classmethod
    def get_template_path(cls) -> Path:
        """Get template path with comprehensive fallback checking"""
        template_locations = [
            cls.TEMPLATES_DIR / 'index_base4.html' if cls.TEMPLATES_DIR else None,
            cls.LEGACY_TEMPLATES_DIR / 'index_base4.html' if cls.LEGACY_TEMPLATES_DIR else None,
            cls.INPUT_DIR / 'templates' / 'index_base4.html' if cls.INPUT_DIR else None,
            cls.INPUT_DIR / 'index_base4.html' if cls.INPUT_DIR else None,
            cls.PROJECT_ROOT / 'templates' / 'index_base4.html',
            Path('templates/index_base4.html'),
            Path('index_base4.html')
        ]

        for location in template_locations:
            if location and location.exists():
                return location.parent  # Return the directory containing the template

        # If no template found, return the primary template directory
        return cls.TEMPLATES_DIR if cls.TEMPLATES_DIR else Path('templates')

    @classmethod
    def debug_template_locations(cls) -> Dict[str, Any]:
        """Debug template locations for troubleshooting"""
        debug_info = {
            'primary_templates_dir': str(cls.TEMPLATES_DIR) if cls.TEMPLATES_DIR else 'None',
            'legacy_templates_dir': str(cls.LEGACY_TEMPLATES_DIR) if cls.LEGACY_TEMPLATES_DIR else 'None',
            'recommended_template_dir': str(cls.get_template_path()),
            'template_exists_in_locations': {}
        }

        template_locations = [
            ('PRIMARY', cls.TEMPLATES_DIR / 'index_base4.html' if cls.TEMPLATES_DIR else None),
            ('LEGACY', cls.LEGACY_TEMPLATES_DIR / 'index_base4.html' if cls.LEGACY_TEMPLATES_DIR else None),
            ('INPUT/templates', cls.INPUT_DIR / 'templates' / 'index_base4.html' if cls.INPUT_DIR else None),
            ('INPUT', cls.INPUT_DIR / 'index_base4.html' if cls.INPUT_DIR else None),
            ('ROOT/templates', cls.PROJECT_ROOT / 'templates' / 'index_base4.html'),
            ('RELATIVE templates', Path('templates/index_base4.html')),
            ('CURRENT DIR', Path('index_base4.html'))
        ]

        for name, location in template_locations:
            if location:
                debug_info['template_exists_in_locations'][name] = {
                    'path': str(location),
                    'exists': location.exists(),
                    'parent_exists': location.parent.exists() if location.parent != location else False
                }

        return debug_info

    @classmethod
    def get_form_config_path(cls) -> Path:
        """Get form config with fallback checking"""
        if cls.FORM_CONFIG_JSON.exists():
            return cls.FORM_CONFIG_JSON
        else:
            # Try legacy location
            legacy_path = cls.LEGACY_STATIC_DIR / 'js' / 'form_config.json'
            if legacy_path.exists():
                print(f"⚠️ Using legacy form config: {legacy_path}")
                return legacy_path
            else:
                raise FileNotFoundError("Form configuration file not found")


# ========== AUTO-INITIALIZATION ==========

# Initialize all paths after class definition
Config._initialize_all_paths()

# Ensure directories exist on import
Config.ensure_directories()

# Auto-validate in debug mode
if Config.get_debug_mode():
    validation = Config.validate_deployment_readiness()
    if not validation['ready_for_deployment']:
        print("⚠️ Configuration validation warnings:")
        for error in validation['errors'][:3]:  # Show first 3 errors
            print(f"   ❌ {error}")
        for warning in validation['warnings'][:2]:  # Show first 2 warnings
            print(f"   ⚠️ {warning}")


# ========== CLOUD STORAGE SETTINGS ==========

# Cloud storage configuration (for future use)
CLOUD_STORAGE_ENABLED = os.getenv('CLOUD_STORAGE_ENABLED', 'false').lower() == 'true'
STORAGE_BUCKET_DATA = os.getenv('STORAGE_BUCKET_DATA', 'entrans-data')
STORAGE_BUCKET_UPLOADS = os.getenv('STORAGE_BUCKET_UPLOADS', 'entrans-uploads')

# ========== LOGGING CONFIGURATION ==========

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO' if Config.ENVIRONMENT == 'gcloud' else 'DEBUG')
LOG_FILE = Config.PROJECT_ROOT / 'app.log'
SIMULATION_LOG = Config.PYSAM_DIR / 'simulation.log'