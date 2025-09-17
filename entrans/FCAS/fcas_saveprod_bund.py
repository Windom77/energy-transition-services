#!/usr/bin/env python3
"""
Save Production Bundle from Existing Models
Run this to create a production bundle from your saved models
"""

import pickle
import joblib
import numpy as np
from datetime import datetime
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def create_production_bundle():
    """Create production bundle from existing saved models"""

    # Load the saved models (update with your actual filename)
    model_file = 'fcas_trained_models_20250617_1546.pkl'  # Your saved models

    logging.info(f"📂 Loading models from {model_file}...")

    try:
        with open(model_file, 'rb') as f:
            model_data = pickle.load(f)
    except FileNotFoundError:
        logging.error(f"❌ Model file {model_file} not found!")
        logging.info("💡 Check your directory for the correct filename")
        return None

    # Extract components
    trained_models = model_data['models']
    model_performance = model_data['performance']
    battery_config = model_data['battery_config']
    region = model_data['region']

    logging.info(f"✅ Loaded {len(trained_models)} trained models")

    # Historical stats (from your module)
    historical_stats = {
        'raise6sec': {'mean': 8.50, 'std': 12.30, 'min': 0.0, 'max': 35.0, 'p95': 28.5},
        'lower6sec': {'mean': 7.20, 'std': 10.80, 'min': 0.0, 'max': 35.0, 'p95': 25.2},
        'raise60sec': {'mean': 5.80, 'std': 8.90, 'min': 0.0, 'max': 35.0, 'p95': 22.1},
        'lower60sec': {'mean': 4.90, 'std': 7.60, 'min': 0.0, 'max': 35.0, 'p95': 18.9},
        'raise5min': {'mean': 3.20, 'std': 5.40, 'min': 0.0, 'max': 35.0, 'p95': 14.2},
        'lower5min': {'mean': 2.80, 'std': 4.80, 'min': 0.0, 'max': 35.0, 'p95': 12.1},
        'raisereg': {'mean': 12.40, 'std': 18.60, 'min': 0.0, 'max': 35.0, 'p95': 32.8},
        'lowerreg': {'mean': 11.60, 'std': 17.20, 'min': 0.0, 'max': 35.0, 'p95': 31.2}
    }

    # FCAS services definition
    fcas_services = {
        'raise6sec': {'name': 'Fast Raise (6s)', 'response_time': 6, 'duration': 60, 'market_cap': 35.0},
        'lower6sec': {'name': 'Fast Lower (6s)', 'response_time': 6, 'duration': 60, 'market_cap': 35.0},
        'raise60sec': {'name': 'Slow Raise (60s)', 'response_time': 60, 'duration': 300, 'market_cap': 35.0},
        'lower60sec': {'name': 'Slow Lower (60s)', 'response_time': 60, 'duration': 300, 'market_cap': 35.0},
        'raise5min': {'name': 'Delayed Raise (5min)', 'response_time': 300, 'duration': 600, 'market_cap': 35.0},
        'lower5min': {'name': 'Delayed Lower (5min)', 'response_time': 300, 'duration': 600, 'market_cap': 35.0},
        'raisereg': {'name': 'Raise Regulation', 'response_time': 1, 'duration': 'continuous', 'market_cap': 35.0},
        'lowerreg': {'name': 'Lower Regulation', 'response_time': 1, 'duration': 'continuous', 'market_cap': 35.0}
    }

    # Historical prices for statistical forecasting
    historical_prices = {service: stats['mean'] for service, stats in historical_stats.items()}

    # Seasonal and daily factors
    seasonal_factors = {
        'winter': 1.2, 'summer': 1.15, 'autumn': 0.9, 'spring': 0.95
    }

    daily_factors = {
        'peak': 1.3, 'shoulder': 1.0, 'off_peak': 0.7
    }

    # Create production bundle
    logging.info("📦 Creating production bundle...")

    bundle = {
        # Core models and performance
        'models': trained_models,
        'performance': model_performance,

        # Static configuration data (no database needed)
        'historical_stats': historical_stats,
        'fcas_services': fcas_services,
        'historical_prices': historical_prices,
        'seasonal_factors': seasonal_factors,
        'daily_factors': daily_factors,

        # Metadata for tracking
        'version': f"v{datetime.now().strftime('%Y%m%d_%H%M')}",
        'training_date': datetime.now().isoformat(),
        'battery_config': battery_config,
        'region': region,

        # Bundle info
        'bundle_type': 'production',
        'model_count': len(trained_models),
        'avg_mae_pct': np.mean([p['mae_pct'] for p in model_performance.values()]) if model_performance else 0
    }

    # Save bundle
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    bundle_path = f'fcas_production_bundle_{timestamp}.pkl'

    joblib.dump(bundle, bundle_path, compress=3)

    # Get file size
    file_size_mb = Path(bundle_path).stat().st_size / (1024 * 1024)

    # Success summary
    print("\n" + "=" * 60)
    print("🎉 PRODUCTION BUNDLE CREATED SUCCESSFULLY!")
    print("=" * 60)
    print(f"📦 Production bundle: {bundle_path}")
    print(f"💾 File size: {file_size_mb:.1f}MB")
    print(f"🤖 ML models: {len(trained_models)}")
    print(f"📊 Average model accuracy: {bundle['avg_mae_pct']:.1f}% MAE")
    print(f"🎯 Region: {region}")
    print(f"⚡ Battery config: {battery_config['power_mw']}MW / {battery_config['energy_mwh']}MWh")

    print("\n📋 Model Performance:")
    for service, perf in model_performance.items():
        print(f"   {service}: MAE=${perf['mae']:.2f} ({perf['mae_pct']:.1f}%)")

    print("\n🚀 Ready for web deployment!")
    print("   - Lightweight bundle for production")
    print("   - No database dependencies")
    print("   - Fast inference capability")
    print("   - Statistical fallback included")
    print("=" * 60)

    return bundle_path


if __name__ == "__main__":
    try:
        bundle_path = create_production_bundle()
        if bundle_path:
            print(f"\n✅ Success! Your production bundle is ready for deployment.")

    except Exception as e:
        print(f"\n❌ Error creating production bundle: {e}")
        import traceback

        traceback.print_exc()