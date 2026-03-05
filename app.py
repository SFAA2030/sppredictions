"""
S&P 500 Stock Price Predictor Web App
Uses trained model from train_model.py
Fetches live + historical data with multiple sources and local caching
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import os
import json
from pathlib import Path
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from pandas_datareader import data as pdr
import warnings
warnings.filterwarnings('ignore')

# Try importing optional data sources
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(
    page_title="S&P 500 Predictor",
    page_icon="📈",
    layout="wide"
)

# Create data directory if it doesn't exist
DATA_DIR = Path("stock_data")
DATA_DIR.mkdir(exist_ok=True)

# -------------------------------
# LOAD MODEL AND METADATA
# -------------------------------
@st.cache_resource
def load_model():
    try:
        with open('models/metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        with open('models/features.pkl', 'rb') as f:
            features = pickle.load(f)
        with open('models/scaler_X.pkl', 'rb') as f:
            scaler_X = pickle.load(f)
        with open('models/scaler_y.pkl', 'rb') as f:
            scaler_y = pickle.load(f)

        if metadata['model_type'] == 'lstm':
            from tensorflow.keras.models import load_model
            model = load_model('models/stock_predictor.h5')
        else:
            with open('models/stock_predictor.pkl', 'rb') as f:
                model = pickle.load(f)

        return model, scaler_X, scaler_y, features, metadata
    except FileNotFoundError:
        st.error("❌ Model files not found! Please run model_creator.py first to train the model.")
        st.stop()

model, scaler_X, scaler_y, features, metadata = load_model()

# -------------------------------
# DATA CACHING FUNCTIONS
# -------------------------------
def save_stock_data(symbol, df):
    """Save stock data to local file"""
    if df.empty:
        return False
    
    filename = DATA_DIR / f"{symbol}_{datetime.now().strftime('%Y%m%d')}.parquet"
    metadata_file = DATA_DIR / f"{symbol}_metadata.json"
    
    # Save data
    df.to_parquet(filename)
    
    # Update metadata
    metadata = {
        'symbol': symbol,
        'last_update': datetime.now().isoformat(),
        'filename': str(filename),
        'rows': len(df),
        'start_date': df.index.min().isoformat() if not df.empty else None,
        'end_date': df.index.max().isoformat() if not df.empty else None
    }
    
    # Keep track of latest file
    latest_file = DATA_DIR / f"{symbol}_latest.parquet"
    df.to_parquet(latest_file)
    
    # Save metadata
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f)
    
    return True

def load_latest_cached_data(symbol):
    """Load the most recent cached data for a symbol"""
    latest_file = DATA_DIR / f"{symbol}_latest.parquet"
    metadata_file = DATA_DIR / f"{symbol}_metadata.json"
    
    if latest_file.exists():
        try:
            df = pd.read_parquet(latest_file)
            
            # Load metadata if available
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                last_update = datetime.fromisoformat(metadata['last_update'])
                days_old = (datetime.now() - last_update).days
                
                st.info(f"📂 Loaded cached data from {last_update.strftime('%Y-%m-%d')} ({days_old} days old)")
            else:
                st.info("📂 Loaded cached data")
            
            return df
        except Exception as e:
            st.warning(f"Could not load cached data: {e}")
    
    return pd.DataFrame()

def list_cached_symbols():
    """List all symbols with cached data"""
    symbols = set()
    for file in DATA_DIR.glob("*_latest.parquet"):
        symbol = file.name.replace("_latest.parquet", "")
        symbols.add(symbol)
    return sorted(symbols)

# -------------------------------
# DATA FETCHING FUNCTIONS
# -------------------------------
def fetch_from_yfinance(symbol, start, end):
    """Fetch data from Yahoo Finance"""
    if not YFINANCE_AVAILABLE:
        return pd.DataFrame()
    
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end)
        
        if not df.empty:
            # Standardize column names
            df.columns = [col.capitalize() for col in df.columns]
            if 'Volume' in df.columns:
                df['Volume'] = df['Volume'].astype(float)
            
            st.success(f"✅ Fetched from Yahoo Finance: {len(df)} days")
            return df
    except Exception as e:
        st.warning(f"Yahoo Finance failed: {str(e)[:50]}...")
    
    return pd.DataFrame()

def fetch_from_stooq(symbol, start, end):
    """Fetch data from Stooq"""
    try:
        df = pdr.DataReader(symbol, 'stooq', start, end)
        if not df.empty:
            df.sort_index(inplace=True)
            st.success(f"✅ Fetched from Stooq: {len(df)} days")
            return df
    except Exception as e:
        st.warning(f"Stooq failed: {str(e)[:50]}...")
    
    return pd.DataFrame()

def fetch_from_alphavantage(symbol, start, end, api_key=None):
    """Fetch data from Alpha Vantage (requires API key)"""
    if not REQUESTS_AVAILABLE or not api_key:
        return pd.DataFrame()
    
    try:
        # Alpha Vantage free endpoint (adjust as needed)
        url = f"https://www.alphavantage.co/query"
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'apikey': api_key,
            'outputsize': 'full'
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if 'Time Series (Daily)' in data:
            time_series = data['Time Series (Daily)']
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Rename columns
            df = df.rename(columns={
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. volume': 'Volume'
            })
            
            # Convert to float
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = pd.to_numeric(df[col])
            
            # Filter by date range
            df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
            
            if not df.empty:
                st.success(f"✅ Fetched from Alpha Vantage: {len(df)} days")
                return df
    except Exception as e:
        st.warning(f"Alpha Vantage failed: {str(e)[:50]}...")
    
    return pd.DataFrame()

def fetch_sample_data(symbol):
    """Generate sample data for testing when all sources fail"""
    st.warning("⚠️ Using sample data for demonstration")
    
    end = datetime.now()
    start = end - timedelta(days=5*365)
    dates = pd.date_range(start=start, end=end, freq='D')
    
    np.random.seed(hash(symbol) % 2**32)
    
    # Generate realistic stock data
    base_price = 100 + np.random.randn() * 50
    returns = np.random.randn(len(dates)) * 0.02
    price = base_price * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame(index=dates)
    df['Close'] = price
    df['Open'] = price * (1 + np.random.randn(len(dates)) * 0.005)
    df['High'] = price * (1 + abs(np.random.randn(len(dates)) * 0.01))
    df['Low'] = price * (1 - abs(np.random.randn(len(dates)) * 0.01))
    df['Volume'] = np.random.randint(1000000, 10000000, len(dates))
    
    return df

@st.cache_data(ttl=3600)
def fetch_stock_data(symbol, use_cache=True, force_refresh=False):
    """
    Fetch historical data from multiple sources with local caching
    """
    end = datetime.now()
    start = end - timedelta(days=5*365)
    
    # Try to load from cache first
    if use_cache and not force_refresh:
        cached_df = load_latest_cached_data(symbol)
        if not cached_df.empty:
            # Check if cache is recent (less than 1 day old)
            metadata_file = DATA_DIR / f"{symbol}_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                last_update = datetime.fromisoformat(metadata['last_update'])
                if (datetime.now() - last_update).days < 1:
                    return cached_df
    
    # Try multiple data sources in order
    sources = []
    
    # 1. Yahoo Finance (most reliable)
    df = fetch_from_yfinance(symbol, start, end)
    if not df.empty:
        sources.append(("Yahoo Finance", len(df)))
        save_stock_data(symbol, df)
        return df
    
    # 2. Stooq
    df = fetch_from_stooq(symbol, start, end)
    if not df.empty:
        sources.append(("Stooq", len(df)))
        save_stock_data(symbol, df)
        return df
    
    # 3. Alpha Vantage (if API key available)
    alpha_vantage_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
    if alpha_vantage_key:
        df = fetch_from_alphavantage(symbol, start, end, alpha_vantage_key)
        if not df.empty:
            sources.append(("Alpha Vantage", len(df)))
            save_stock_data(symbol, df)
            return df
    
    # If all live sources fail, try cached data again
    if use_cache:
        cached_df = load_latest_cached_data(symbol)
        if not cached_df.empty:
            st.warning("⚠️ Using cached data (live sources unavailable)")
            return cached_df
    
    # Last resort: generate sample data
    df = fetch_sample_data(symbol)
    if not df.empty:
        st.warning("⚠️ Using generated sample data")
        return df
    
    st.error(f"❌ Failed to fetch data for {symbol} from any source")
    return pd.DataFrame()

# -------------------------------
# FEATURE ENGINEERING 
# -------------------------------
def prepare_features(df):
    """
    Prepare features EXACTLY as done in model_creator.py
    Using ONLY past data (shifted) to avoid look-ahead bias
    """
    df = df.copy()
    
    if len(df) < 60:
        return df
    
    # Create features using ONLY past data
    df['Close_Lag1'] = df['Close'].shift(1)
    df['Close_Lag2'] = df['Close'].shift(2)
    df['Close_Lag3'] = df['Close'].shift(3)
    df['Close_Lag5'] = df['Close'].shift(5)
    
    df['Returns_Lag1'] = df['Close'].pct_change().shift(1)
    df['Returns_Lag2'] = df['Close'].pct_change().shift(2)
    
    df['SMA_10'] = df['Close'].rolling(window=10, min_periods=10).mean().shift(1)
    df['SMA_20'] = df['Close'].rolling(window=20, min_periods=20).mean().shift(1)
    df['SMA_50'] = df['Close'].rolling(window=50, min_periods=50).mean().shift(1)
    
    df['High_Low_Range'] = (df['High'] - df['Low']).shift(1)
    df['Open_Close_Range'] = (df['Close'] - df['Open']).shift(1)
    
    df['Volume_SMA'] = df['Volume'].rolling(window=5, min_periods=5).mean().shift(1)
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    
    returns = df['Close'].pct_change()
    df['Volatility_10'] = returns.rolling(window=10).std().shift(1)
    df['Volatility_20'] = returns.rolling(window=20).std().shift(1)
    
    df.dropna(inplace=True)
    return df

# -------------------------------
# PREDICTION FUNCTIONS 
# -------------------------------
def predict_next_day(current_data, available_features, historical_volatility):
    """
    Predict the next day's closing price with realistic bounds
    """
    # Get features
    latest_features = current_data[available_features].iloc[-1:].values
    
    if np.any(np.isnan(latest_features)) or np.any(np.isinf(latest_features)):
        return None
    
    # Scale and predict
    latest_scaled = scaler_X.transform(latest_features)
    
    try:
        if metadata['model_type'] == 'lstm':
            latest_scaled = latest_scaled.reshape(1, 1, -1)
            pred_scaled = model.predict(latest_scaled, verbose=0)[0, 0]
        else:
            pred_scaled = model.predict(latest_scaled)[0]
        
        pred_price = scaler_y.inverse_transform([[pred_scaled]])[0, 0]
        
        # Apply realistic bounds based on historical volatility
        last_price = current_data['Close'].iloc[-1]
        
        # Use 3 standard deviations of historical daily returns as bound
        max_daily_change = last_price * (historical_volatility * 3 / 100)  # Convert to price
        
        # Also cap at 10% as absolute maximum for any single day
        max_allowed_change = min(max_daily_change, last_price * 0.10)
        
        if abs(pred_price - last_price) > max_allowed_change:
            # Adjust prediction toward the bound
            if pred_price > last_price:
                pred_price = last_price + max_allowed_change
            else:
                pred_price = last_price - max_allowed_change
        
        return pred_price
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def predict_future(data, days, available_features):
    """
    Predict future prices with realistic constraints
    """
    predictions = []
    confidence_intervals = []
    current_data = data.copy()
    
    # Calculate historical volatility from recent data
    recent_returns = data['Close'].pct_change().dropna().tail(60)
    historical_volatility = recent_returns.std() * 100  # Convert to percentage
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(days):
        status_text.text(f"Predicting day {i+1}/{days}...")
        
        next_price = predict_next_day(current_data, available_features, historical_volatility)
        
        if next_price is None:
            break
        
        predictions.append(next_price)
        
        # Calculate confidence interval
        vol = historical_volatility / 100  # Convert back to decimal
        ci = 1.96 * vol * next_price * np.sqrt((i+1)/252)
        confidence_intervals.append(ci)
        
        # Create next day's data
        last_date = current_data.index[-1]
        next_date = last_date + pd.Timedelta(days=1)
        
        # Create new row with reasonable values
        new_row = pd.DataFrame(index=[next_date])
        new_row['Open'] = current_data['Close'].iloc[-1]
        new_row['High'] = max(current_data['Close'].iloc[-1], next_price) * 1.002
        new_row['Low'] = min(current_data['Close'].iloc[-1], next_price) * 0.998
        new_row['Close'] = next_price
        
        # Estimate volume based on recent average
        avg_volume = data['Volume'].tail(20).mean()
        volume_std = data['Volume'].tail(20).std()
        new_row['Volume'] = max(1000, np.random.normal(avg_volume, volume_std * 0.1))
        
        current_data = pd.concat([current_data, new_row])
        current_data = prepare_features(current_data)
        
        progress_bar.progress((i + 1) / days)
    
    progress_bar.empty()
    status_text.empty()
    
    return predictions, confidence_intervals

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("🔍 Stock Predictor")

# Symbol input with autocomplete from cache
cached_symbols = list_cached_symbols()
if cached_symbols:
    symbol = st.sidebar.selectbox(
        "Select Stock Symbol",
        options=[""] + cached_symbols + ["Enter custom..."],
        format_func=lambda x: "Choose a symbol..." if x == "" else x
    )
    if symbol == "Enter custom...":
        symbol = st.sidebar.text_input("Enter custom symbol", "AAPL").upper()
    elif symbol == "":
        symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL)", "AAPL").upper()
else:
    symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL)", "AAPL").upper()

days = st.sidebar.slider("Prediction Horizon (Days)", 1, 30, 7)

# Data source options
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Data Options")

use_cache = st.sidebar.checkbox("Use cached data", value=True)
force_refresh = st.sidebar.checkbox("Force refresh from live sources", value=False)

if st.sidebar.button("🔄 Clear Cache for This Symbol"):
    cache_files = list(DATA_DIR.glob(f"{symbol}_*"))
    for f in cache_files:
        f.unlink()
    st.sidebar.success(f"Cleared cache for {symbol}")
    st.cache_data.clear()

# Model info
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Model Info")
st.sidebar.info(f"**Best Model:** {metadata['best_model']}")
if 'mae_score' in metadata:
    st.sidebar.info(f"**MAE:** {metadata['mae_score']:.4f}")
st.sidebar.info(f"**R² Score:** {metadata['r2_score']:.3f}")
st.sidebar.info(f"**Training Date:** {metadata['training_date'][:10]}")

# -------------------------------
# MAIN APP
# -------------------------------
if symbol:
    st.header(f"📈 {symbol} Stock Prediction")

    # Fetch data
    with st.spinner(f"Fetching historical data for {symbol}..."):
        hist = fetch_stock_data(symbol, use_cache=use_cache, force_refresh=force_refresh)
        
        if hist.empty:
            st.stop()
        
        # Show data source info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Data Points", f"{len(hist)} days")
        with col2:
            st.metric("Date Range", f"{hist.index[0].strftime('%Y-%m-%d')} to {hist.index[-1].strftime('%Y-%m-%d')}")
        with col3:
            st.metric("Latest Price", f"${hist['Close'].iloc[-1]:.2f}")
    
    # Prepare features
    with st.spinner("Preparing features..."):
        featured_data = prepare_features(hist)
        available_features = [f for f in features if f in featured_data.columns]
        
        if len(available_features) < 5:
            st.error("Too many features missing.")
            st.stop()
    
    # Make predictions
    with st.spinner(f"Predicting next {days} days..."):
        try:
            future_prices, confidence_intervals = predict_future(featured_data, days, available_features)
            
            if not future_prices:
                st.error("❌ Failed to generate predictions")
                st.stop()
            
            last_date = featured_data.index[-1]
            future_dates = [last_date + timedelta(days=x+1) for x in range(len(future_prices))]
            
            st.success("✅ Predictions completed!")
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            st.stop()

    # -------------------------------
    # CHARTS
    # -------------------------------
    st.subheader("📊 Price History & Prediction")
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3]
    )

    # Historical (last 180 days for better context)
    historical_days = min(180, len(featured_data))
    hist_prices = featured_data['Close'][-historical_days:]
    hist_dates = featured_data.index[-historical_days:]
    
    fig.add_trace(
        go.Scatter(x=hist_dates, y=hist_prices, mode='lines', name='Historical', 
                  line=dict(color='#1f77b4', width=2)),
        row=1, col=1
    )

    # Prediction
    fig.add_trace(
        go.Scatter(x=future_dates, y=future_prices, mode='lines+markers', name='Prediction',
                  line=dict(color='#ff7f0e', width=2), marker=dict(size=6)),
        row=1, col=1
    )
    
    # Confidence interval
    upper = [p + ci for p, ci in zip(future_prices, confidence_intervals)]
    lower = [p - ci for p, ci in zip(future_prices, confidence_intervals)]
    
    fig.add_trace(
        go.Scatter(x=future_dates + future_dates[::-1], y=upper + lower[::-1],
                  fill='toself', fillcolor='rgba(255, 127, 14, 0.2)',
                  line=dict(color='rgba(255,255,255,0)'), name='95% Confidence'),
        row=1, col=1
    )

    # Volume
    fig.add_trace(
        go.Bar(x=hist_dates, y=featured_data['Volume'][-historical_days:],
               name='Volume', marker_color='rgba(31, 119, 180, 0.3)'),
        row=2, col=1
    )

    fig.update_layout(height=600, hovermode='x unified', showlegend=True)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------
    # PREDICTION SUMMARY
    # -------------------------------
    st.subheader("📊 Prediction Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = featured_data['Close'].iloc[-1]
    predicted_price = future_prices[-1]
    price_change = predicted_price - current_price
    percent_change = (price_change / current_price) * 100
    
    with col1:
        st.metric("Current Price", f"${current_price:.2f}")
    
    with col2:
        st.metric(f"{days}-Day Prediction", f"${predicted_price:.2f}", 
                  f"{price_change:+.2f} ({percent_change:+.1f}%)")
    
    with col3:
        if confidence_intervals:
            st.metric("Prediction Range", 
                      f"${predicted_price - confidence_intervals[-1]:.2f} - ${predicted_price + confidence_intervals[-1]:.2f}")
    
    with col4:
        st.metric("Confidence", f"{95}%")

    # -------------------------------
    # DETAILED PREDICTIONS
    # -------------------------------
    st.subheader("📋 Daily Predictions")
    
    pred_df = pd.DataFrame({
        'Date': [d.strftime('%Y-%m-%d') for d in future_dates],
        'Price': [f"${p:.2f}" for p in future_prices],
        'Daily Change': [f"${future_prices[i] - (future_prices[i-1] if i>0 else current_price):+.2f}" 
                        for i in range(len(future_prices))],
        'Total Change': [f"{(p-current_price)/current_price*100:+.1f}%" for p in future_prices]
    })
    
    st.dataframe(pred_df, use_container_width=True, hide_index=True)
    
    # Download buttons
    col1, col2 = st.columns(2)
    with col1:
        csv = pred_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions (CSV)",
            data=csv,
            file_name=f"{symbol}_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Option to download raw data
        if st.button("💾 Save Current Data to Cache"):
            if save_stock_data(symbol, hist):
                st.success(f"Data saved to cache!")
            else:
                st.error("Failed to save data")

    # -------------------------------
    # HISTORICAL DATA VIEWER
    # -------------------------------
    with st.expander("📜 View Historical Data"):
        st.dataframe(hist.tail(50), use_container_width=True)

else:
    st.info("👈 Enter a stock symbol in the sidebar to start")
    
    # Show cached symbols if available
    if cached_symbols:
        st.subheader("📁 Available Cached Symbols")
        st.write(f"Previously viewed symbols: {', '.join(cached_symbols)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 10px;'>
        <b>Disclaimer:</b> Created for Academic Purpose Only. Data may be cached or from multiple sources.
    </div>
    """,
    unsafe_allow_html=True
)