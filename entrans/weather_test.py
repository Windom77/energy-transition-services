#!/usr/bin/env python3
"""
Weather File Test Script
Run this in your cloud environment to debug weather file issues
"""

import sys
from pathlib import Path
import math

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_weather_functionality():
    """Comprehensive weather functionality test"""
    print("🌦️ WEATHER FUNCTIONALITY TEST")
    print("=" * 60)

    # Test 1: Import config
    try:
        from config import Config
        print("✅ Config imported successfully")

        # Run config validation
        validation = Config.validate_paths()
        print(f"✅ Config validation completed")

        # Run weather debug
        Config.debug_weather_setup()

    except Exception as e:
        print(f"❌ Config import/validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 2: Test coordinate extraction
    print("\n🧪 Testing coordinate extraction:")
    test_filenames = [
        "1785715_Adelaide_-34.91_138.58_tmy-2020.csv",
        "1955370_Melbourne_-37.83_144.98_tmy-2020.csv",
        "2060764_Sydney_-33.87_151.22_tmy-2020.csv",
        "2077383_Brisbane_-27.47_153.02_tmy-2020.csv"
    ]

    for filename in test_filenames:
        coords = extract_coordinates_test(filename)
        print(f"   {filename} -> {coords}")

    # Test 3: Test closest file logic
    print("\n🎯 Testing closest file selection:")
    test_coordinates = [
        ("Sydney", -33.87, 151.22),
        ("Melbourne", -37.83, 144.98),
        ("Brisbane", -27.47, 153.02),
        ("Adelaide", -34.91, 138.58),
        ("Perth", -31.95, 115.86)  # Not in dataset
    ]

    for city, lat, lon in test_coordinates:
        try:
            closest = find_closest_weather_file_test(lat, lon)
            if closest:
                distance = calculate_distance(lat, lon, closest[1], closest[2])
                print(f"   {city} ({lat}, {lon}) -> {closest[0]} (distance: {distance:.2f}°)")
            else:
                print(f"   {city} ({lat}, {lon}) -> No file found")
        except Exception as e:
            print(f"   {city} ({lat}, {lon}) -> Error: {e}")

    # Test 4: Test integration module import
    print("\n🔧 Testing integration module:")
    try:
        from pysam_form_integration2 import update_pysam_json
        print("✅ Integration module imported successfully")

        # Test with sample data
        test_data = {
            'lat': -33.87,
            'lon': 151.22,
            'solar_resource_file': '/old/path/to/weather.csv'
        }

        # This would normally update the JSON, but we'll just test the path
        print(f"✅ Test data prepared: {test_data}")

    except Exception as e:
        print(f"❌ Integration module test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("🏁 Weather functionality test completed")


def extract_coordinates_test(filename: str):
    """Test coordinate extraction"""
    try:
        parts = filename.replace('.csv', '').split('_')
        coords = []
        for part in parts:
            clean_part = part.replace('-', '').replace('.', '')
            if clean_part.isdigit() or (part.startswith('-') and clean_part.isdigit()):
                coords.append(float(part))

        return (coords[-2], coords[-1]) if len(coords) >= 2 else None
    except:
        return None


def find_closest_weather_file_test(lat: float, lon: float):
    """Test closest file finding"""
    try:
        from config import Config

        if not Config.WEATHER_DIR.exists():
            return None

        weather_files = []
        for file in Config.WEATHER_DIR.glob("*.csv"):
            coords = extract_coordinates_test(file.name)
            if coords:
                weather_files.append((file.name, coords[0], coords[1]))

        if not weather_files:
            return None

        closest = min(weather_files,
                      key=lambda x: math.sqrt((x[1] - lat) ** 2 + (x[2] - lon) ** 2))

        return closest
    except Exception as e:
        print(f"Error in closest file test: {e}")
        return None


def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate simple distance between coordinates"""
    return math.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)


if __name__ == "__main__":
    test_weather_functionality()