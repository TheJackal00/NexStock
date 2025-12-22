import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import lightgbm as lgb
import pickle
import json

class DemandForecaster:
    """Machine Learning demand forecaster using LightGBM"""
    
    def __init__(self, db_path='inventory.db'):
        self.db_path = db_path
        self.models = {}  # Store one model per SKU
        self.feature_columns = []
        self.scalers = {}
        self.historical_variance = {}  # Store variance for realistic noise
        
    def load_transaction_data(self, start_date='2024-12-01'):
        """Load and aggregate transaction data by day"""
        conn = sqlite3.connect(self.db_path)
        
        query = f"""
        SELECT 
            DATE,
            SKU,
            SUM(CASE WHEN VOLUME < 0 THEN ABS(VOLUME) ELSE 0 END) as demand,
            SUM(CASE WHEN VOLUME > 0 THEN VOLUME ELSE 0 END) as supply
        FROM TRANSACTIONS
        WHERE DATE >= '{start_date}'
        GROUP BY DATE, SKU
        ORDER BY DATE, SKU
        """
        
        df = pd.read_sql_query(query, conn)
        df['DATE'] = pd.to_datetime(df['DATE'])
        conn.close()
        
        return df
    
    def calculate_historical_variance(self, sku):
        """Calculate historical demand variance for a SKU"""
        conn = sqlite3.connect(self.db_path)
        
        query = f"""
        SELECT 
            DATE,
            SUM(CASE WHEN VOLUME < 0 THEN ABS(VOLUME) ELSE 0 END) as demand
        FROM TRANSACTIONS
        WHERE SKU = '{sku}'
        GROUP BY DATE
        ORDER BY DATE
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(df) > 1:
            # Calculate standard deviation of daily demand
            variance = df['demand'].std()
            return variance if not pd.isna(variance) else 0
        return 0
    
    def load_economic_data(self):
        """Load economic indicators"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            query = """
            SELECT DATE, INDICATOR, VALUE
            FROM ECONOMIC_DATA
            ORDER BY DATE, INDICATOR
            """
            df = pd.read_sql_query(query, conn)
            df['DATE'] = pd.to_datetime(df['DATE'])
            
            # Pivot to wide format
            df_pivot = df.pivot(index='DATE', columns='INDICATOR', values='VALUE')
            df_pivot = df_pivot.reset_index()
            
            print(f"✓ Loaded {len(df_pivot)} economic data points")
            return df_pivot
            
        except Exception as e:
            print(f"⚠️  No economic data available: {e}")
            return None
        finally:
            conn.close()
    
    def engineer_features(self, df, sku):
        """Create time-based and lag features"""
        df = df[df['SKU'] == sku].copy()
        df = df.sort_values('DATE')
        
        # Time features
        df['day_of_week'] = df['DATE'].dt.dayofweek
        df['day_of_month'] = df['DATE'].dt.day
        df['month'] = df['DATE'].dt.month
        df['quarter'] = df['DATE'].dt.quarter
        df['week_of_year'] = df['DATE'].dt.isocalendar().week
        
        # Lag features (1, 7, 14, 30 days)
        for lag in [1, 7, 14, 30]:
            df[f'demand_lag_{lag}'] = df['demand'].shift(lag)
        
        # Rolling averages
        for window in [7, 14, 30]:
            df[f'demand_rolling_mean_{window}'] = df['demand'].rolling(window=window, min_periods=1).mean()
            df[f'demand_rolling_std_{window}'] = df['demand'].rolling(window=window, min_periods=1).std()
        
        # Trend: difference from 7-day average
        df['demand_trend'] = df['demand'] - df['demand_rolling_mean_7']
        
        return df
    
    def merge_economic_data(self, df, econ_df):
        """Merge economic indicators with transaction data"""
        if econ_df is None:
            return df
        
        # Economic data is typically monthly, forward-fill for daily data
        df['year_month'] = df['DATE'].dt.to_period('M')
        econ_df['year_month'] = econ_df['DATE'].dt.to_period('M')
        
        # Merge on year-month
        df_merged = df.merge(
            econ_df.drop('DATE', axis=1), 
            on='year_month', 
            how='left'
        )
        
        # Forward fill economic indicators
        econ_cols = [col for col in econ_df.columns if col not in ['DATE', 'year_month']]
        df_merged[econ_cols] = df_merged[econ_cols].fillna(method='ffill')
        
        df_merged = df_merged.drop('year_month', axis=1)
        
        return df_merged
    
    def train_model(self, sku, forecast_horizon=180):
        """Train LightGBM model for specific SKU"""
        print(f"\n{'='*60}")
        print(f"Training model for SKU: {sku}")
        print(f"{'='*60}")
        
        # Load data
        trans_df = self.load_transaction_data()
        econ_df = self.load_economic_data()
        
        # Engineer features
        df = self.engineer_features(trans_df, sku)
        df = self.merge_economic_data(df, econ_df)
        
        # Drop rows with NaN in lag features (first 30 days)
        df = df.dropna()
        
        if len(df) < 60:
            print(f"⚠️  Insufficient data for {sku}: {len(df)} days")
            return None
        
        # Prepare features and target
        feature_cols = [col for col in df.columns if col not in ['DATE', 'SKU', 'demand', 'supply']]
        X = df[feature_cols]
        y = df['demand']
        
        self.feature_columns = feature_cols
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        best_model = None
        best_score = float('inf')
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train LightGBM
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            params = {
                'objective': 'regression',
                'metric': 'mae',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'verbose': -1
            }
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=200,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
            )
            
            # Evaluate
            y_pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            
            if mae < best_score:
                best_score = mae
                best_model = model
            
            print(f"  Fold {fold+1} - MAE: {mae:.2f}")
        
        # Final evaluation on entire dataset
        y_pred_all = best_model.predict(X)
        mae_final = mean_absolute_error(y, y_pred_all)
        rmse_final = np.sqrt(mean_squared_error(y, y_pred_all))
        mape_final = mean_absolute_percentage_error(y, y_pred_all) * 100
        
        print(f"\n✓ Final Metrics:")
        print(f"  MAE:  {mae_final:.2f}")
        print(f"  RMSE: {rmse_final:.2f}")
        print(f"  MAPE: {mape_final:.2f}%")
        
        # Store model
        self.models[sku] = best_model
        
        return {
            'sku': sku,
            'mae': mae_final,
            'rmse': rmse_final,
            'mape': mape_final,
            'training_samples': len(df)
        }
    
    def predict_demand(self, sku, start_date=None, days=180):
        """Generate demand forecast for SKU"""
        if sku not in self.models:
            print(f"⚠️  No trained model for SKU {sku}")
            return None
        
        model = self.models[sku]
        
        # Load latest data
        trans_df = self.load_transaction_data()
        econ_df = self.load_economic_data()
        
        # Engineer features
        df = self.engineer_features(trans_df, sku)
        df = self.merge_economic_data(df, econ_df)
        df = df.dropna()
        
        # Get last date
        last_date = df['DATE'].max()
        
        if start_date is None:
            start_date = last_date + timedelta(days=1)
        else:
            start_date = pd.to_datetime(start_date)
        
        predictions = []
        current_date = start_date
        
        # Use last known values for rolling prediction
        last_row = df.iloc[-1].copy()
        
        # Get historical variance for this SKU to add realistic noise
        if sku not in self.historical_variance:
            self.historical_variance[sku] = self.calculate_historical_variance(sku)
        
        hist_std = self.historical_variance[sku]
        
        for day in range(days):
            # Update date features
            pred_row = last_row.copy()
            pred_row['day_of_week'] = current_date.dayofweek
            pred_row['day_of_month'] = current_date.day
            pred_row['month'] = current_date.month
            pred_row['quarter'] = (current_date.month - 1) // 3 + 1
            pred_row['week_of_year'] = current_date.isocalendar()[1]
            
            # Prepare features
            X_pred = pred_row[self.feature_columns].values.reshape(1, -1)
            
            # Predict with realistic variance
            base_pred = model.predict(X_pred)[0]
            
            # Add noise based on historical variance (reduced for longer horizons)
            noise_factor = 1.0 if day < 30 else 0.8 if day < 90 else 0.6
            noise = np.random.normal(0, hist_std * noise_factor * 0.5)  # 50% of historical std
            
            demand_pred = max(0, base_pred + noise)  # No negative demand
            
            predictions.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'sku': sku,
                'predicted_demand': round(demand_pred, 2),
                'confidence': 'medium' if day < 60 else 'low'
            })
            
            # Update lag features for next prediction (rolling forecast)
            last_row['demand_lag_1'] = demand_pred
            last_row['demand'] = demand_pred
            
            current_date += timedelta(days=1)
        
        return predictions
    
    def save_models(self, filepath='ml_models.pkl'):
        """Save trained models to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'feature_columns': self.feature_columns,
                'timestamp': datetime.now().isoformat()
            }, f)
        print(f"\n✓ Models saved to {filepath}")
    
    def load_models(self, filepath='ml_models.pkl'):
        """Load trained models from disk"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.models = data['models']
                self.feature_columns = data['feature_columns']
            print(f"✓ Models loaded from {filepath}")
            return True
        except FileNotFoundError:
            print(f"⚠️  Model file not found: {filepath}")
            return False

def train_all_skus():
    """Train models for all SKUs in database"""
    print("\n" + "="*70)
    print("ML DEMAND FORECASTING - TRAINING PIPELINE")
    print("="*70)
    
    forecaster = DemandForecaster()
    
    # Get all SKUs
    conn = sqlite3.connect('inventory.db')
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT SKU FROM TRANSACTIONS ORDER BY SKU")
    skus = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    print(f"\nFound {len(skus)} SKUs: {skus}")
    
    results = []
    for sku in skus:
        result = forecaster.train_model(sku)
        if result:
            results.append(result)
    
    # Save models
    forecaster.save_models()
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    for result in results:
        print(f"SKU {result['sku']:5} | MAE: {result['mae']:6.2f} | RMSE: {result['rmse']:6.2f} | MAPE: {result['mape']:5.2f}%")
    
    return forecaster

if __name__ == "__main__":
    forecaster = train_all_skus()
    
    # Example prediction
    print("\n" + "="*70)
    print("EXAMPLE FORECAST - SKU 001 (Next 30 days)")
    print("="*70)
    predictions = forecaster.predict_demand('001', days=30)
    
    if predictions:
        for i, pred in enumerate(predictions[:10], 1):  # Show first 10 days
            print(f"Day {i:2} ({pred['date']}): {pred['predicted_demand']:6.2f} units")
        print(f"... (showing first 10 of {len(predictions)} days)")
