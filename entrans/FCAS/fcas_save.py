#!/usr/bin/env python3
"""
Save FCAS Models Script
Run this to save your trained models from the test session
"""

from enhanced_fcas_module import EnhancedFCASEstimator
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def save_trained_models():
    """Save the trained models from your test session"""

    # Recreate the estimator with the same config from your test
    test_config = {
        'power_mw': 2.0,
        'energy_mwh': 0.1,
        'charge_power_mw': 1.6,
        'efficiency': 0.90,
        'participation_rate': 0.85
    }

    # Initialize estimator
    estimator = EnhancedFCASEstimator(test_config, 'NSW1',
                                      db_path='/media/dswhome/Elements/FCAS_data/nem_data_final.db')

    # Load the same historical data your test used
    logging.info("🔄 Loading historical data...")
    historical_data = estimator.load_historical_data('2023-01-01', '2023-03-31')

    # Train models (this will recreate your trained models)
    logging.info("🤖 Training ML models (recreating from test)...")
    performance = estimator.train_ml_models(historical_data)

    # Create timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')

    # Save full models
    model_file = f'fcas_trained_models_{timestamp}.pkl'
    logging.info(f"💾 Saving models to {model_file}...")
    estimator.save_models(model_file)

    # Save production bundle
    bundle_file = f'fcas_production_bundle_{timestamp}.pkl'
    logging.info(f"📦 Creating production bundle {bundle_file}...")
    estimator.save_production_bundle(bundle_file)

    # Generate test forecast to verify
    logging.info("🔮 Testing saved models with forecast...")
    test_forecast = estimator.generate_ml_forecast(datetime(2024, 1, 1), 5)
    total_revenue = test_forecast['total_revenue'].sum()

    # Summary
    print("\n" + "=" * 60)
    print("🎉 FCAS MODELS SUCCESSFULLY SAVED!")
    print("=" * 60)
    print(f"✅ Models saved to: {model_file}")
    print(f"📦 Production bundle: {bundle_file}")
    print(f"🤖 ML models trained: {len(estimator.trained_models)}")
    print(f"🎯 Test revenue forecast: ${total_revenue:,.0f} over 5 years")
    print(f"📊 Average annual: ${total_revenue / 5:,.0f}")
    print("\n📋 Model Performance:")
    for service, perf in performance.items():
        print(f"   {service}: MAE=${perf['mae']:.2f} ({perf['mae_pct']:.1f}%)")

    print("\n🚀 Ready for production deployment!")
    print("=" * 60)

    return model_file, bundle_file, total_revenue


if __name__ == "__main__":
    try:
        model_file, bundle_file, revenue = save_trained_models()
        print(f"\n✅ Success! Your FCAS models are saved and ready to use.")

    except Exception as e:
        print(f"\n❌ Error saving models: {e}")
        print("💡 Make sure you're in the correct directory with enhanced_fcas_module.py")
        print("💡 And your database path is correct")