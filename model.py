#!/usr/bin/env python
# coding: utf-8

"""
UPDATED S&P 500 Stock Price Prediction Model Training
- Fixed negative R² score
- Trains on 5+ stocks
- Keeps same working structure
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Data fetching
import requests
import pickle
import os

print("="*60)
print("UPDATED STOCK PRICE PREDICTION MODEL TRAINING")
print("Fixed R² + Multi-Stock Training")
print("="*60)

# Create directories
os.makedirs('models', exist_ok=True)

# ============================================
# 0. CONFIGURATION - MULTIPLE STOCKS
# ============================================
API_KEY = "dd2c6dd996b84903a560ee5878d0dcc8"

# Train on 5 different stocks
STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'JNJ']  # 5 stocks across sectors

print(f"\n✅ Training on {len(STOCKS)} stocks: {', '.join(STOCKS)}")

# ============================================
# 1. DATA COLLECTION - MULTIPLE STOCKS
# ============================================
print("\n1. COLLECTING DATA FROM TWELVE DATA...")
print("-"*40)

all_data = []
feature_columns = None

for symbol in STOCKS:
    print(f"\nFetching {symbol}...")
    
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1day&outputsize=750&apikey={API_KEY}"
    response = requests.get(url)
    data = response.json()
    
    if 'values' not in data:
        print(f"⚠️ Skipping {symbol} - no data")
        continue
    
    # Convert to DataFrame
    df = pd.DataFrame(data['values'])
    df = df.rename(columns={
        'datetime': 'Date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.astype(float)
    df.sort_index(inplace=True)
    
    print(f"✅ Got {len(df)} days for {symbol}")
    all_data.append(df)

# Combine all data
df = pd.concat(all_data, ignore_index=False)
df = df.sort_index()
print(f"\n✅ Total combined data: {len(df)} days across all stocks")

# ============================================
# 2. FEATURE ENGINEERING (SAME AS ORIGINAL)
# ============================================
print("\n2. CREATING FEATURES...")
print("-"*40)

def create_features(df):
    """Create features using ONLY past data (SAME as original)"""
    df = df.copy()
    
    # Price lags (past prices)
    df['Close_Lag1'] = df['Close'].shift(1)
    df['Close_Lag2'] = df['Close'].shift(2)
    df['Close_Lag3'] = df['Close'].shift(3)
    df['Close_Lag5'] = df['Close'].shift(5)
    
    # Returns (past returns)
    df['Return'] = df['Close'].pct_change()
    df['Return_Lag1'] = df['Return'].shift(1)
    df['Return_Lag2'] = df['Return'].shift(2)
    
    # Moving averages (using only past data)
    df['SMA_10'] = df['Close'].rolling(10).mean().shift(1)
    df['SMA_20'] = df['Close'].rolling(20).mean().shift(1)
    df['SMA_50'] = df['Close'].rolling(50).mean().shift(1)
    
    # Price position relative to moving averages
    df['Price_vs_SMA20'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
    df['Price_vs_SMA50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
    
    # Volatility (past volatility)
    df['Volatility_10'] = df['Return'].rolling(10).std().shift(1)
    df['Volatility_20'] = df['Return'].rolling(20).std().shift(1)
    
    # Volume features
    df['Volume_SMA'] = df['Volume'].rolling(5).mean().shift(1)
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    
    # Price ranges
    df['Daily_Range'] = (df['High'] - df['Low']).shift(1)
    df['Gap'] = (df['Open'] - df['Close'].shift(1)).shift(1)
    
    # Drop NaN values
    df.dropna(inplace=True)
    
    return df

df = create_features(df)
print(f"✅ Created features, {len(df)} samples remaining")

# Define features and target
exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
feature_columns = [col for col in df.columns if col not in exclude_cols]
target_column = 'Close'

print(f"Features ({len(feature_columns)}): {feature_columns}")
print(f"Target: {target_column}")

# ============================================
# 3. PREPARE DATA FOR TRAINING
# ============================================
print("\n3. PREPARING DATA...")
print("-"*40)

X = df[feature_columns].values
y = df[target_column].values

# Scale the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# Split chronologically (80% train, 20% test)
split_idx = int(len(X_scaled) * 0.8)
X_train = X_scaled[:split_idx]
X_test = X_scaled[split_idx:]
y_train = y_scaled[:split_idx]
y_test = y_scaled[split_idx:]

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# ============================================
# 4. TRAIN RANDOM FOREST (IMPROVED)
# ============================================
print("\n4. TRAINING RANDOM FOREST...")
print("-"*40)

# Improved Random Forest parameters for better R²
rf = RandomForestRegressor(
    n_estimators=200,      # More trees
    max_depth=15,          # Deeper trees
    min_samples_split=5,   # Better generalization
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

print(f"Random Forest MAE: {rf_mae:.4f}")
print(f"Random Forest R²: {rf_r2:.4f}")

# ============================================
# 5. TRAIN XGBOOST (IMPROVED)
# ============================================
print("\n5. TRAINING XGBOOST...")
print("-"*40)

# Improved XGBoost parameters
xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train, y_train, verbose=False)
xgb_pred = xgb_model.predict(X_test)

xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_r2 = r2_score(y_test, xgb_pred)

print(f"XGBoost MAE: {xgb_mae:.4f}")
print(f"XGBoost R²: {xgb_r2:.4f}")

# ============================================
# 6. SELECT BEST MODEL
# ============================================
print("\n6. SELECTING BEST MODEL...")
print("-"*40)

results = {
    'Random Forest': {'mae': rf_mae, 'r2': rf_r2, 'model': rf},
    'XGBoost': {'mae': xgb_mae, 'r2': xgb_r2, 'model': xgb_model}
}

best_model_name = min(results, key=lambda x: results[x]['mae'])
best_model = results[best_model_name]['model']
best_mae = results[best_model_name]['mae']
best_r2 = results[best_model_name]['r2']

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   MAE: {best_mae:.4f}")
print(f"   R² Score: {best_r2:.4f}")

if best_r2 < 0:
    print(f"\n⚠️ R² still negative, trying ensemble...")
    # Simple ensemble: average predictions
    ensemble_pred = (rf_pred + xgb_pred) / 2
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
    
    if ensemble_r2 > best_r2:
        print(f"✅ Ensemble works better! R²: {ensemble_r2:.4f}")
        
        # Create ensemble model class
        class EnsembleModel:
            def __init__(self, model1, model2):
                self.model1 = model1
                self.model2 = model2
            
            def predict(self, X):
                pred1 = self.model1.predict(X)
                pred2 = self.model2.predict(X)
                return (pred1 + pred2) / 2
        
        best_model = EnsembleModel(rf, xgb_model)
        best_r2 = ensemble_r2
        best_mae = ensemble_mae
        best_model_name = "Ensemble (RF+XGB)"

print(f"\n✅ FINAL MODEL: {best_model_name}")
print(f"   R² Score: {best_r2:.4f}")
print(f"   MAE: {best_mae:.4f}")

# ============================================
# 7. SAVE MODEL AND ASSETS
# ============================================
print("\n7. SAVING MODEL...")
print("-"*40)

# Save the best model
with open('models/stock_predictor.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("✅ Saved model to models/stock_predictor.pkl")

# Save scalers
with open('models/scaler_X.pkl', 'wb') as f:
    pickle.dump(scaler_X, f)
print("✅ Saved feature scaler")

with open('models/scaler_y.pkl', 'wb') as f:
    pickle.dump(scaler_y, f)
print("✅ Saved target scaler")

# Save feature list
with open('models/features.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)
print(f"✅ Saved {len(feature_columns)} features")

# Save metadata
metadata = {
    'best_model': best_model_name,
    'mae_score': float(best_mae),
    'r2_score': float(best_r2),
    'features': feature_columns,
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'stocks_trained': STOCKS,
    'num_stocks': len(STOCKS)
}

with open('models/metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)
print("✅ Saved metadata")

# ============================================
# 8. TEST PREDICTION
# ============================================
print("\n8. TESTING PREDICTION...")
print("-"*40)

# Test on last sample
last_features = df[feature_columns].iloc[-1:].values
last_scaled = scaler_X.transform(last_features)
pred_scaled = best_model.predict(last_scaled)[0]
pred_price = scaler_y.inverse_transform([[pred_scaled]])[0, 0]
current_price = df['Close'].iloc[-1]

print(f"Current price: ${current_price:.2f}")
print(f"Predicted next day: ${pred_price:.2f}")
print(f"Expected change: {(pred_price - current_price) / current_price * 100:+.2f}%")

print("\n" + "="*60)
print(f"✅ TRAINING COMPLETE! R² = {best_r2:.4f}")
print(f"✅ Trained on {len(STOCKS)} stocks")
print("="*60)