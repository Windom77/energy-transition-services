"""
Fixed Configuration for EnTrans Subdirectory Structure
"""
import os
from pathlib import Path
from typing import Optional, Tuple

class Config:
    """Environment-aware configuration for EnTrans in subdirectory"""

    # Environment detection
    ENVIRONMENT = os.getenv('GAE_ENV', 'local')

    # Project root - go up one level from entrans/config.py to energy-transition-services/
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # EnTrans root - the entrans/ subdirectory
    ENTRANS_ROOT = Path(__file__).resolve().parent

    # Core directories (relative to ENTRANS_ROOT)
    INPUT_DIR = ENTRANS_ROOT / '1.input'
    PYSAM_DIR = ENTRANS_ROOT / '2.pysam'
    OUTPUT_DIR = ENTRANS_ROOT / '3.output'
    DATA_DIR = ENTRANS_ROOT / 'data'

    # Data subdirectories
    WEATHER_DIR = DATA_DIR / 'weather'
    INPUT_JSON_DIR = DATA_DIR / 'input_json'

    # Template files
    TEMPLATE_JSON = INPUT_JSON_DIR / 'All_commercial.json'

    # Form files
    FORM_HTML = INPUT_DIR / 'index_merged.html'

    # Output files
    UPDATED_JSON_DIR = INPUT_DIR / 'json_updated'
    UPDATED_JSON = UPDATED_JSON_DIR / 'All_commercial_updated.json'

    # Results
    RESULTS_DIR = PYSAM_DIR / 'results'

    # Static assets
    STATIC_DIR = INPUT_DIR / 'static'
    TEMPLATES_DIR = INPUT_DIR / 'templates'

    # Form config
    FORM_CONFIG_JSON = STATIC_DIR / 'js' / 'form_config.json'

    @classmethod
    def ensure_directories(cls) -> None:
        """Ensure required directories exist"""
        directories = [
            cls.INPUT_DIR, cls.PYSAM_DIR, cls.OUTPUT_DIR, cls.DATA_DIR,
            cls.WEATHER_DIR, cls.INPUT_JSON_DIR, cls.UPDATED_JSON_DIR,
            cls.RESULTS_DIR
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_debug_mode(cls) -> bool:
        """Get debug mode"""
        return cls.ENVIRONMENT != 'standard' and os.getenv('DEBUG', 'true').lower() == 'true'

# Initialize directories
Config.ensure_directories()