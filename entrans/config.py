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

    # Class variables - will be initialized after class definition
    ENVIRONMENT = None
    PROJECT_ROOT = None
    INPUT_DIR = None
    PYSAM_DIR = None
    OUTPUT_DIR = None
    DATA_DIR = None
    WEATHER_DIR = None
    INPUT_JSON_DIR = None
    PROFILES_DIR = None
    TEMPLATE_JSON = None
    TEMPLATE_JSON_FALLBACK = None
    FORM_HTML = None
    FORM_HTML_FALLBACK = None
    UPDATED_JSON_DIR = None
    UPDATED_JSON = None
    RESULTS_DIR = None
    STATIC_DIR = None
    TEMPLATES_DIR = None
    LEGACY_STATIC_DIR = None
    LEGACY_TEMPLATES_DIR = None
    FORM_CONFIG_JSON = None
    BASE_JSON = None

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

        # Check for templates in new structure (root/templates) first
        if (cls.PROJECT_ROOT / 'templates').exists():
            cls.TEMPLATES_DIR = cls.PROJECT_ROOT / 'templates'
            cls.STATIC_DIR = cls.PROJECT_ROOT / 'static'
            cls.LEGACY_TEMPLATES_DIR = cls.INPUT_DIR / 'templates'
            cls.LEGACY_STATIC_DIR = cls.INPUT_DIR / 'static'
        else:
            # Fallback to old structure
            cls.TEMPLATES_DIR = cls.INPUT_DIR / 'templates'
            cls.STATIC_DIR = cls.INPUT_DIR / 'static'
            cls.LEGACY_TEMPLATES_DIR = cls.PROJECT_ROOT / 'templates'
            cls.LEGACY_STATIC_DIR = cls.PROJECT_ROOT / 'static'

        # Form configuration - check both locations
        if (cls.STATIC_DIR / 'js' / 'form_config.json').exists():
            cls.FORM_CONFIG_JSON = cls.STATIC_DIR / 'js' / 'form_config.json'
        elif (cls.LEGACY_STATIC_DIR / 'js' / 'form_config.json').exists():
            cls.FORM_CONFIG_JSON = cls.LEGACY_STATIC_DIR / 'js' / 'form_config.json'
        else:
            cls.FORM_CONFIG_JSON = cls.STATIC_DIR / 'js' / 'form_config.json'

        # Additional required paths
        cls.BASE_JSON = cls.INPUT_JSON_DIR / 'All_commercial.json'

    @classmethod
    def get_debug_mode(cls) -> bool:
        """Get debug mode based on environment"""
        if cls.ENVIRONMENT == 'gcloud':
            return False
        return os.getenv('DEBUG', 'true').lower() == 'true'

    @classmethod
    def get_weather_file_by_coordinates(cls, lat: float, lon: float) -> Optional[Path]:
        """Find closest weather file by coordinates"""
        try:
            if not cls.WEATHER_DIR.exists():
                print(f"Weather directory not found: {cls.WEATHER_DIR}")
                return None

            weather_files = list(cls.WEATHER_DIR.glob("*.csv"))
            if not weather_files:
                print(f"No weather files found in: {cls.WEATHER_DIR}")
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
                print(f"Selected weather file: {closest_file.name} (distance: {min_distance:.2f})")
                return closest_file
            else:
                print("No weather files with valid coordinates found")
                return None

        except Exception as e:
            print(f"Error finding weather file: {e}")
            return None

    @classmethod
    def _extract_coordinates_from_filename(cls, filename: str) -> Optional[Tuple[float, float]]:
        """Extract coordinates from weather filename"""
        try:
            parts = filename.replace('.csv', '').split('_')
            coords = []
            for part in parts:
                try:
                    coord = float(part)
                    coords.append(coord)
                except ValueError:
                    continue

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
                print(f"Could not create directory {directory}: {e}")

        if cls.get_debug_mode():
            print(f"Directory structure verified from {cls.PROJECT_ROOT}")

    @classmethod
    def validate_deployment_readiness(cls) -> dict:
        """Validate that the configuration is ready for deployment"""
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

        return validation_results

    @classmethod
    def get_fallback_template(cls) -> Path:
        """Get fallback template file with robust checking"""
        if cls.TEMPLATE_JSON.exists():
            return cls.TEMPLATE_JSON
        elif cls.TEMPLATE_JSON_FALLBACK.exists():
            print(f"Using fallback template: {cls.TEMPLATE_JSON_FALLBACK}")
            return cls.TEMPLATE_JSON_FALLBACK
        else:
            raise FileNotFoundError("No template JSON files found")


# Initialize all paths after class definition
Config._initialize_all_paths()

# Ensure directories exist on import
Config.ensure_directories()

# Auto-validate in debug mode
if Config.get_debug_mode():
    validation = Config.validate_deployment_readiness()
    if not validation['ready_for_deployment']:
        print("Configuration validation warnings:")
        for error in validation['errors'][:3]:
            print(f"   {error}")


# Cloud storage configuration (for future use)
CLOUD_STORAGE_ENABLED = os.getenv('CLOUD_STORAGE_ENABLED', 'false').lower() == 'true'
STORAGE_BUCKET_DATA = os.getenv('STORAGE_BUCKET_DATA', 'entrans-data')
STORAGE_BUCKET_UPLOADS = os.getenv('STORAGE_BUCKET_UPLOADS', 'entrans-uploads')

# Logging configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO' if Config.ENVIRONMENT == 'gcloud' else 'DEBUG')
LOG_FILE = Config.PROJECT_ROOT / 'app.log'
SIMULATION_LOG = Config.PYSAM_DIR / 'simulation.log'