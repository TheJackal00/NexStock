"""
Generate ML forecasts and save to the forecast table
"""
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from ml_demand_forecast import DemandForecaster

def generate_forecasts(days=90):
    """Generate forecasts for all SKUs and save to forecast table"""
    print(f"\n{'='*60}")
    print(f"Generating ML forecasts for {days} days")
    print(f"{'='*60}\n")
    
    # Load forecaster
    forecaster = DemandForecaster()
    if not forecaster.load_models('ml_models.pkl'):
        print("❌ Models not trained. Training now...")
        # Get all SKUs
        conn = sqlite3.connect('inventory.db')
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT SKU FROM TRANSACTIONS ORDER BY SKU")
        skus = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        # Train models
        for sku in skus:
            print(f"Training model for SKU {sku}...")
            forecaster.train_model(sku, forecast_horizon=days)
        
        # Save models
        forecaster.save_models('ml_models.pkl')
        print("✓ Models trained and saved\n")
    else:
        print("✓ Models loaded successfully\n")
    
    # Get all SKUs
    conn = sqlite3.connect('inventory.db')
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT SKU FROM TRANSACTIONS ORDER BY SKU")
    skus = [row[0] for row in cursor.fetchall()]
    
    # Clear old forecasts
    cursor.execute("DELETE FROM forecast")
    conn.commit()
    print("✓ Cleared old forecasts\n")
    
    forecast_inserts = []
    total_predictions = 0
    
    for sku in skus:
        print(f"Generating forecast for SKU {sku}...")
        predictions = forecaster.predict_demand(sku, days=days)
        
        if predictions:
            demand_values = [p['predicted_demand'] for p in predictions]
            print(f"  ├─ Total demand: {sum(demand_values):.0f} units")
            print(f"  ├─ Avg daily: {np.mean(demand_values):.1f} units")
            print(f"  ├─ Range: {min(demand_values):.1f} - {max(demand_values):.1f} units")
            print(f"  └─ Std dev: {np.std(demand_values):.1f} units\n")
            
            # Prepare data for forecast table
            for i, pred in enumerate(predictions):
                forecast_inserts.append((
                    sku,
                    i + 1,  # Day number
                    pred['predicted_demand'],
                    pred['confidence'],
                    pred['date'],
                    days,
                    'LightGBM_v1'
                ))
            
            total_predictions += len(predictions)
    
    # Save to forecast table
    if forecast_inserts:
        cursor.executemany('''
            INSERT INTO forecast (SKU, DAY, PREDICTED_DEMAND, CONFIDENCE, DATE, FORECAST_HORIZON, MODEL_VERSION)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', forecast_inserts)
        conn.commit()
        print(f"{'='*60}")
        print(f"✓ Saved {total_predictions} predictions to forecast table")
        print(f"  ({len(skus)} SKUs × {days} days = {len(skus)*days} expected)")
        print(f"{'='*60}\n")
    
    conn.close()
    return total_predictions

if __name__ == "__main__":
    total = generate_forecasts(days=90)
    print(f"\n✓ Forecast generation complete! {total} predictions saved.\n")
