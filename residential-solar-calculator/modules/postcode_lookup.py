import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PostcodeLookup:
    """
    Lookup latitude and longitude coordinates for Australian postcodes
    """

    def __init__(self):
        self.postcode_data = self._load_postcode_data()

        # Fallback coordinates for major cities
        self.city_fallbacks = {
            '1000': (-35.2809, 149.1300),  # Canberra
            '2000': (-33.8688, 151.2093),  # Sydney
            '3000': (-37.8136, 144.9631),  # Melbourne
            '4000': (-27.4698, 153.0251),  # Brisbane
            '5000': (-34.9285, 138.6007),  # Adelaide
            '6000': (-31.9505, 115.8605),  # Perth
            '7000': (-42.8821, 147.3272),  # Hobart
            '8000': (-12.4634, 130.8456),  # Darwin
        }

    def _load_postcode_data(self):
        """Load postcode data from CSV file or create basic lookup"""
        try:
            # Try to load from data directory
            data_path = Path(__file__).parent.parent / 'data' / 'postcodes_longlat.csv'

            if data_path.exists():
                df = pd.read_csv(data_path)
                # Ensure required columns exist (note: your CSV has 'long' not 'longitude')
                if all(col in df.columns for col in ['postcode', 'lat', 'long']):
                    # Convert to dict with postcode as key
                    postcode_dict = {}
                    for _, row in df.iterrows():
                        postcode_dict[str(row['postcode']).zfill(4)] = {
                            'latitude': row['lat'],
                            'longitude': row['long'],
                            'locality': row.get('locality', ''),
                            'state': row.get('state', '')
                        }
                    return postcode_dict
                else:
                    logger.warning("Postcode CSV missing required columns")
                    return {}
            else:
                logger.warning("postcodes_longlat.csv not found, using fallbacks only")
                return {}

        except Exception as e:
            logger.error(f"Error loading postcode data: {str(e)}")
            return {}

    def get_coordinates(self, postcode):
        """
        Get latitude and longitude for a given Australian postcode

        Args:
            postcode (str): 4-digit Australian postcode

        Returns:
            tuple: (latitude, longitude) or None if not found
        """
        try:
            postcode = str(postcode).zfill(4)  # Ensure 4 digits with leading zeros

            # First try exact lookup
            if postcode in self.postcode_data:
                data = self.postcode_data[postcode]
                return (data['latitude'], data['longitude'])

            # Try city fallbacks
            if postcode in self.city_fallbacks:
                return self.city_fallbacks[postcode]

            # Try state-based fallback using first digit
            state_fallbacks = {
                '1': (-35.2809, 149.1300),  # ACT
                '2': (-33.8688, 151.2093),  # NSW
                '3': (-37.8136, 144.9631),  # VIC
                '4': (-27.4698, 153.0251),  # QLD
                '5': (-34.9285, 138.6007),  # SA
                '6': (-31.9505, 115.8605),  # WA
                '7': (-42.8821, 147.3272),  # TAS
                '8': (-12.4634, 130.8456),  # NT
                '9': (-12.4634, 130.8456),  # Remote areas
            }

            first_digit = postcode[0]
            if first_digit in state_fallbacks:
                logger.info(f"Using state fallback for postcode {postcode}")
                return state_fallbacks[first_digit]

            # Ultimate fallback - Sydney
            logger.warning(f"No coordinates found for postcode {postcode}, using Sydney")
            return (-33.8688, 151.2093)

        except Exception as e:
            logger.error(f"Error looking up postcode {postcode}: {str(e)}")
            return (-33.8688, 151.2093)  # Sydney fallback

    def validate_postcode(self, postcode):
        """
        Validate if a postcode is a valid Australian postcode format

        Args:
            postcode (str): Postcode to validate

        Returns:
            bool: True if valid format, False otherwise
        """
        try:
            # Remove whitespace and ensure string
            postcode = str(postcode).strip()

            # Must be 4 digits
            if not postcode.isdigit() or len(postcode) != 4:
                return False

            # Must start with valid state digit (1-9)
            if not postcode[0] in '123456789':
                return False

            return True

        except:
            return False

    def get_state_name(self, postcode):
        """Get state name from postcode"""
        try:
            postcode = str(postcode).zfill(4)
            state_map = {
                '1': 'ACT',
                '2': 'NSW',
                '3': 'VIC',
                '4': 'QLD',
                '5': 'SA',
                '6': 'WA',
                '7': 'TAS',
                '8': 'NT',
                '9': 'Remote'
            }
            return state_map.get(postcode[0], 'Unknown')
        except:
            return 'Unknown'