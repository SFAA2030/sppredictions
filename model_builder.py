"""
S&P 500 Stock Price Prediction Model Training
Run this first to create and save your model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
import xgboost as xgb

# Deep Learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Stock data
import yfinance as yf
import pickle
import os

print("="*60)
print("TRAINING S&P 500 PRICE PREDICTION MODEL")
print("="*60)

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# ============================================
# 1. DATA COLLECTION
# ============================================
print("\n📥 1. COLLECTING DATA...")
print("-"*40)

# Use top S&P 500 stocks for training
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'BRK-B', 'LLY', 'V', 'JPM']
print(f"Training on {len(tickers)} stocks: {', '.join(tickers)}")

# Download data
all_data = {}
for ticker in tickers:
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="5y")
        if len(data) > 0:
            all_data[ticker] = data
            print(f"  ✅ {ticker}: {len(data)} days")
    except:
        print(f"  ❌ {ticker}: Failed")

# ============================================
# 2. DATA PREPROCESSING
# ============================================
print("\n🧹 2. PREPROCESSING DATA...")
print("-"*40)

def preprocess_data(df):
    """Clean and add technical indicators"""
    df = df.copy()
    
    # Remove duplicates and sort
    df = df[~df.index.duplicated(keep='first')]
    df.sort_index(inplace=True)
    
    # Handle missing values
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    # Calculate returns
    df['Returns'] = df['Close'].pct_change()
    
    # Technical indicators
    # Moving averages
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Price channels
    df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
    df['Price_Change'] = (df['Close'] - df['Open']) / df['Open'] * 100
    
    # Volume indicators
    df['Volume_SMA'] = df['Volume'].rolling(window=5).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    
    # Volatility
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    
    # Momentum
    df['Momentum_5'] = df['Close'].pct_change(5)
    df['Momentum_10'] = df['Close'].pct_change(10)
    
    # Drop NaN values
    df.dropna(inplace=True)
    
    return df

# Process all stocks
processed_data = {}
for ticker, df in all_data.items():
    processed_data[ticker] = preprocess_data(df)
    print(f"  ✅ {ticker}: {len(processed_data[ticker])} features")

# Combine data from all stocks for training
combined_data = pd.concat(processed_data.values(), axis=0)
print(f"\nTotal training samples: {len(combined_data)}")

# ============================================
# 3. FEATURE SELECTION
# ============================================
print("\n🎯 3. SELECTING FEATURES...")
print("-"*40)

# Define features (exclude price columns)
feature_columns = [col for col in combined_data.columns if col not in 
                  ['Open', 'High', 'Low', 'Close', 'Volume']]
target_column = 'Close'

print(f"Features ({len(feature_columns)}): {feature_columns[:10]}...")

# ============================================
# 4. TRAIN/TEST SPLIT
# ============================================
print("\n✂️ 4. SPLITTING DATA...")
print("-"*40)

X = combined_data[feature_columns].values
y = combined_data[target_column].values

# Scale the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# Split (keeping temporal order)
split_idx = int(len(X_scaled) * 0.8)
X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# ============================================
# 5. TRAIN MULTIPLE MODELS
# ============================================
print("\n🤖 5. TRAINING MODELS...")
print("-"*40)

results = {}

# 1. Linear Regression
print("\n📈 Linear Regression...")
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
results['Linear Regression'] = {
    'model': lr,
    'r2': r2_score(y_test, lr_pred),
    'mse': mean_squared_error(y_test, lr_pred)
}
print(f"   R² Score: {results['Linear Regression']['r2']:.4f}")

# 2. Random Forest
print("\n🌲 Random Forest...")
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
results['Random Forest'] = {
    'model': rf,
    'r2': r2_score(y_test, rf_pred),
    'mse': mean_squared_error(y_test, rf_pred)
}
print(f"   R² Score: {results['Random Forest']['r2']:.4f}")

# 3. XGBoost
print("\n⚡ XGBoost...")
xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
results['XGBoost'] = {
    'model': xgb_model,
    'r2': r2_score(y_test, xgb_pred),
    'mse': mean_squared_error(y_test, xgb_pred)
}
print(f"   R² Score: {results['XGBoost']['r2']:.4f}")

# 4. LSTM
print("\n🧠 LSTM Neural Network...")
# Reshape for LSTM (samples, timesteps, features)
timesteps = 10
X_lstm = []
y_lstm = []
for i in range(len(X_scaled) - timesteps):
    X_lstm.append(X_scaled[i:i+timesteps])
    y_lstm.append(y_scaled[i+timesteps])

X_lstm = np.array(X_lstm)
y_lstm = np.array(y_lstm)

# Split
split = int(len(X_lstm) * 0.8)
X_train_lstm, X_test_lstm = X_lstm[:split], X_lstm[split:]
y_train_lstm, y_test_lstm = y_lstm[:split], y_lstm[split:]

# Build LSTM
lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(timesteps, X.shape[1])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Train
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lstm_model.fit(
    X_train_lstm, y_train_lstm,
    validation_data=(X_test_lstm, y_test_lstm),
    epochs=30,
    batch_size=32,
    callbacks=[early_stop],
    verbose=0
)

# Evaluate
lstm_pred = lstm_model.predict(X_test_lstm, verbose=0).flatten()
results['LSTM'] = {
    'model': lstm_model,
    'r2': r2_score(y_test_lstm, lstm_pred),
    'mse': mean_squared_error(y_test_lstm, lstm_pred)
}
print(f"   R² Score: {results['LSTM']['r2']:.4f}")

# ============================================
# 6. SELECT BEST MODEL
# ============================================
print("\n🏆 6. SELECTING BEST MODEL...")
print("-"*40)

# Find model with highest R²
best_model_name = max(results, key=lambda x: results[x]['r2'])
best_model = results[best_model_name]['model']
best_r2 = results[best_model_name]['r2']

print(f"\n✅ BEST MODEL: {best_model_name}")
print(f"   R² Score: {best_r2:.4f}")
print(f"   MSE: {results[best_model_name]['mse']:.4f}")

# Print all results
print("\n📊 All Models Performance:")
for name, metrics in results.items():
    print(f"   {name:20} R²: {metrics['r2']:.4f} | MSE: {metrics['mse']:.4f}")

# ============================================
# 7. SAVE MODEL AND ASSETS
# ============================================
print("\n💾 7. SAVING MODEL...")
print("-"*40)

# Save the best model
if best_model_name == 'LSTM':
    best_model.save('models/stock_predictor.h5')
    model_type = 'lstm'
else:
    with open('models/stock_predictor.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    model_type = 'sklearn'

# Save scalers
with open('models/scaler_X.pkl', 'wb') as f:
    pickle.dump(scaler_X, f)
with open('models/scaler_y.pkl', 'wb') as f:
    pickle.dump(scaler_y, f)

# Save feature list
with open('models/features.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)

# Save metadata
metadata = {
    'best_model': best_model_name,
    'r2_score': best_r2,
    'features': feature_columns,
    'model_type': model_type,
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'num_samples': len(combined_data),
    'all_results': {name: {'r2': m['r2'], 'mse': m['mse']} for name, m in results.items()}
}

with open('models/metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print(f"\n✅ Model saved as: models/stock_predictor.{'h5' if model_type=='lstm' else 'pkl'}")
print("✅ Scalers saved: models/scaler_X.pkl, models/scaler_y.pkl")
print("✅ Features saved: models/features.pkl")
print("✅ Metadata saved: models/metadata.pkl")

print("\n" + "="*60)
print("TRAINING COMPLETE! 🎉")
print("="*60)
print("\nNext step: Run 'streamlit run app.py' to start the prediction app")