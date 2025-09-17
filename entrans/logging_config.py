# logging_config.py - Clean production logging configuration

import logging
import os


def setup_production_logging():
    """Configure logging for production with milestones only"""

    # Get environment
    environment = os.getenv('FLASK_ENV', 'development')
    is_production = environment == 'production'

    if is_production:
        # PRODUCTION: Only milestones, warnings, and errors
        log_level = logging.WARNING
        log_format = '[%(levelname)s] %(message)s'
    else:
        # DEVELOPMENT: More detailed logging
        log_level = logging.INFO
        log_format = '[%(levelname)s] %(asctime)s - %(message)s'

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt='%H:%M:%S',
        force=True  # Override any existing configuration
    )

    # Suppress verbose third-party libraries
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    logging.getLogger('urllib3').setLevel(logging.ERROR)
    logging.getLogger('requests').setLevel(logging.ERROR)
    logging.getLogger('google').setLevel(logging.ERROR)
    logging.getLogger('dash').setLevel(logging.ERROR)

    # Create milestone logger for important events
    milestone_logger = logging.getLogger('milestones')
    milestone_logger.setLevel(logging.INFO)

    return milestone_logger


def create_milestone_logger():
    """Create a logger specifically for milestone events"""
    logger = logging.getLogger('milestones')
    logger.setLevel(logging.INFO)

    # Custom handler for milestones (always logs regardless of root level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('🎯 [MILESTONE] %(message)s'))
        logger.addHandler(handler)
        logger.propagate = False  # Don't send to root logger

    return logger


def log_milestone(message: str):
    """Log important milestones that should always be visible"""
    milestone_logger = logging.getLogger('milestones')
    if not milestone_logger.handlers:
        create_milestone_logger()
    milestone_logger = logging.getLogger('milestones')
    milestone_logger.info(message)


def log_error(message: str, error: Exception = None):
    """Log errors with full context"""
    if error:
        logging.error(f"❌ {message}: {str(error)}")
    else:
        logging.error(f"❌ {message}")


def log_warning(message: str):
    """Log warnings"""
    logging.warning(f"⚠️ {message}")