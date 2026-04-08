"""
S&P 500 Stock Price Predictor - UPDATED for Sklearn Model
No LSTM - Uses Random Forest/XGBoost from fixed training
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import requests
from pathlib import Path
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="S&P 500 Predictor", page_icon="📈", layout="wide")

# Twelve Data API Key
TWELVE_DATA_API_KEY = "dd2c6dd996b84903a560ee5878d0dcc8"

# Realistic prediction limits (now just for safety, model should be realistic)
MAX_DAILY_CHANGE_PCT = 0.03  # Max 3% per day (tighter)
MAX_TOTAL_CHANGE_PCT = 0.10  # Max 10% total over prediction period

# -------------------------------
# LOAD MODEL (SKLEARN ONLY)
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
        
        # Load the sklearn model (no LSTM)
        with open('models/stock_predictor.pkl', 'rb') as f:
            model = pickle.load(f)
        
        st.success(f"✅ Loaded {metadata['best_model']} model (trained on {metadata['symbol']})")
        
        return model, scaler_X, scaler_y, features, metadata
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("Please run fixed_model_creator.py first to train the model")
        st.stop()

model, scaler_X, scaler_y, features, metadata = load_model()

# -------------------------------
# SAFETY CAP (just in case model goes crazy)
# -------------------------------
def safe_cap_prediction(current_price, predicted_price):
    """
    Safety cap - only applies if model produces unrealistic predictions
    """
    change_pct = (predicted_price - current_price) / current_price
    
    # Only cap if change is > 5% (model should be better than this)
    if abs(change_pct) > 0.05:
        if change_pct > 0:
            capped_price = current_price * 1.05
            st.caption(f"⚠️ Safety cap applied: {change_pct*100:.1f}% → 5.0%")
        else:
            capped_price = current_price * 0.95
            st.caption(f"⚠️ Safety cap applied: {change_pct*100:.1f}% → -5.0%")
        return capped_price
    
    return predicted_price

# -------------------------------
# FETCH REAL DATA FROM TWELVE DATA
# -------------------------------
@st.cache_data(ttl=3600)
def fetch_real_data(symbol, days=500):
    """Fetch REAL stock data from Twelve Data API"""
    
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1day&outputsize={days}&apikey={TWELVE_DATA_API_KEY}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if 'values' not in data:
            st.error(f"Error fetching {symbol}: {data.get('message', 'Unknown error')}")
            return pd.DataFrame()
        
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
        
        # Convert types
        df['Date'] = pd.to_datetime(df['Date'])
        df['Open'] = df['Open'].astype(float)
        df['High'] = df['High'].astype(float)
        df['Low'] = df['Low'].astype(float)
        df['Close'] = df['Close'].astype(float)
        df['Volume'] = df['Volume'].astype(float)
        
        # Set index and sort
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        
        return df
        
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        return pd.DataFrame()

# -------------------------------
# FEATURE ENGINEERING (MUST MATCH TRAINING)
# -------------------------------
def prepare_features(df):
    """Prepare features EXACTLY as in training"""
    df = df.copy()
    
    if len(df) < 100:
        return df
    
    # Price lags
    df['Close_Lag1'] = df['Close'].shift(1)
    df['Close_Lag2'] = df['Close'].shift(2)
    df['Close_Lag3'] = df['Close'].shift(3)
    df['Close_Lag5'] = df['Close'].shift(5)
    df['Close_Lag10'] = df['Close'].shift(10)
    
    # Returns
    df['Return'] = df['Close'].pct_change()
    df['Return_Lag1'] = df['Return'].shift(1)
    df['Return_Lag2'] = df['Return'].shift(2)
    df['Return_Lag3'] = df['Return'].shift(3)
    df['Return_Lag5'] = df['Return'].shift(5)
    
    # Moving averages
    df['SMA_5'] = df['Close'].rolling(5).mean().shift(1)
    df['SMA_10'] = df['Close'].rolling(10).mean().shift(1)
    df['SMA_20'] = df['Close'].rolling(20).mean().shift(1)
    df['SMA_50'] = df['Close'].rolling(50).mean().shift(1)
    
    # Price position relative to moving averages
    df['Price_vs_SMA20'] = ((df['Close'] - df['SMA_20']) / df['SMA_20']).shift(1)
    df['Price_vs_SMA50'] = ((df['Close'] - df['SMA_50']) / df['SMA_50']).shift(1)
    
    # Volatility
    df['Volatility_10'] = df['Return'].rolling(10).std().shift(1)
    df['Volatility_20'] = df['Return'].rolling(20).std().shift(1)
    
    # Volume features
    df['Volume_SMA'] = df['Volume'].rolling(5).mean().shift(1)
    df['Volume_Ratio'] = (df['Volume'] / df['Volume_SMA']).shift(1)
    
    # Price ranges
    df['Daily_Range'] = ((df['High'] - df['Low']) / df['Close']).shift(1)
    df['Gap'] = ((df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)).shift(1)
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = (100 - (100 / (1 + rs))).shift(1)
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = (exp1 - exp2).shift(1)
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean().shift(1)
    
    # Drop NaN
    df.dropna(inplace=True)
    
    return df

# -------------------------------
# PREDICTION FUNCTION
# -------------------------------
def predict_future(data, days, available_features):
    """Predict future prices using sklearn model"""
    predictions = []
    confidence_intervals = []
    current_data = data.copy()
    
    # Calculate historical volatility
    if 'Return' in current_data.columns:
        recent_returns = current_data['Return'].dropna().tail(60)
        historical_volatility = recent_returns.std() * 100 if len(recent_returns) > 0 else 1.5
    else:
        historical_volatility = 1.5
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(days):
        status_text.text(f"Predicting day {i+1}/{days}...")
        
        # Get latest features
        latest_features = current_data[available_features].iloc[-1:].values
        
        if np.any(np.isnan(latest_features)):
            break
        
        # Scale and predict
        latest_scaled = scaler_X.transform(latest_features)
        pred_scaled = model.predict(latest_scaled)[0]
        raw_pred_price = scaler_y.inverse_transform([[pred_scaled]])[0, 0]
        
        # Apply safety cap (only if unrealistic)
        current_price = current_data['Close'].iloc[-1]
        capped_price = safe_cap_prediction(current_price, raw_pred_price)
        
        predictions.append(capped_price)
        
        # Confidence interval based on volatility
        daily_vol = historical_volatility / 100
        ci = 1.96 * daily_vol * capped_price * np.sqrt((i+1)/252)
        confidence_intervals.append(ci)
        
        # Create next day's data
        last_date = current_data.index[-1]
        next_date = last_date + timedelta(days=1)
        
        new_row = pd.DataFrame(index=[next_date])
        new_row['Open'] = current_price
        new_row['High'] = max(current_price, capped_price)
        new_row['Low'] = min(current_price, capped_price)
        new_row['Close'] = capped_price
        new_row['Volume'] = data['Volume'].tail(20).mean() if len(data['Volume']) >= 20 else 1000000
        
        current_data = pd.concat([current_data, new_row])
        current_data = prepare_features(current_data)
        
        progress_bar.progress((i + 1) / days)
    
    progress_bar.empty()
    status_text.empty()
    
    return predictions, confidence_intervals

# -------------------------------
# MAIN APP
# -------------------------------
st.title("📈 S&P 500 Stock Price Predictor")
st.markdown(f"**Using REAL data from Twelve Data API | Model: {metadata['best_model']}**")

# Sidebar
st.sidebar.header("Stock Selection")
symbol = st.sidebar.text_input("Stock Symbol", value="AAPL").upper()
days = st.sidebar.slider("Prediction Days", 1, 30, 7)

# Model info
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Model Info")
st.sidebar.info(f"**Model:** {metadata['best_model']}")
st.sidebar.info(f"**Trained on:** {metadata.get('symbol', 'AAPL')}")
st.sidebar.info(f"**R² Score:** {metadata['r2_score']:.4f}")
st.sidebar.info(f"**MAE:** {metadata['mae_score']:.4f}")
st.sidebar.info(f"**Training Date:** {metadata['training_date'][:10]}")

# Fetch data
if symbol:
    with st.spinner(f"Fetching real data for {symbol}..."):
        hist = fetch_real_data(symbol, days=500)
        
        if hist.empty:
            st.error(f"Could not fetch data for {symbol}. Please check the symbol.")
            st.stop()
        
        # Calculate volatility safely
        returns = hist['Close'].pct_change().dropna()
        volatility = returns.std() * 100 if len(returns) > 0 else 0
        
        # Show current price
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"${hist['Close'].iloc[-1]:.2f}")
        with col2:
            st.metric("Data Points", f"{len(hist)} days")
        with col3:
            st.metric("Date Range", f"{hist.index[0].strftime('%Y-%m-%d')}")
        with col4:
            st.metric("Volatility", f"{volatility:.2f}%")
    
    # Prepare features
    with st.spinner("Preparing features..."):
        featured_data = prepare_features(hist)
        
        # Get only the features the model expects
        available_features = [f for f in features if f in featured_data.columns]
        
        if len(available_features) < len(features) * 0.8:
            st.warning(f"Only {len(available_features)} of {len(features)} features available")
            missing = set(features) - set(available_features)
            st.write(f"Missing features: {list(missing)[:5]}")
        
        st.success(f"✅ Features prepared: {len(available_features)} indicators")
    
    # Predict
    with st.spinner(f"Predicting next {days} days..."):
        future_prices, confidence_intervals = predict_future(featured_data, days, available_features)
        
        if not future_prices:
            st.error("Prediction failed")
            st.stop()
        
        future_dates = [featured_data.index[-1] + timedelta(days=x+1) for x in range(len(future_prices))]
    
    # Plot
    st.subheader("📊 Price Chart")
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
    
    # Historical (last 180 days)
    hist_days = min(180, len(featured_data))
    fig.add_trace(go.Scatter(
        x=featured_data.index[-hist_days:],
        y=featured_data['Close'][-hist_days:],
        mode='lines',
        name='Historical (Real Data)',
        line=dict(color='#1f77b4', width=2)
    ), row=1, col=1)
    
    # Prediction
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_prices,
        mode='lines+markers',
        name='Prediction',
        line=dict(color='#ff7f0e', width=2),
        marker=dict(size=8, color='red')
    ), row=1, col=1)
    
    # Confidence interval
    upper = [p + ci for p, ci in zip(future_prices, confidence_intervals)]
    lower = [p - ci for p, ci in zip(future_prices, confidence_intervals)]
    
    fig.add_trace(go.Scatter(
        x=future_dates + future_dates[::-1],
        y=upper + lower[::-1],
        fill='toself',
        fillcolor='rgba(255,127,14,0.2)',
        line=dict(color='rgba(255,127,14,0)'),
        name='95% Confidence'
    ), row=1, col=1)
    
    # Volume
    fig.add_trace(go.Bar(
        x=featured_data.index[-hist_days:],
        y=featured_data['Volume'][-hist_days:],
        name='Volume',
        marker_color='rgba(31,119,180,0.3)'
    ), row=2, col=1)
    
    fig.update_layout(height=600, showlegend=True, hovermode='x unified')
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary
    st.subheader("📊 Prediction Summary")
    
    current_price = featured_data['Close'].iloc[-1]
    predicted_price = future_prices[-1]
    change = predicted_price - current_price
    change_pct = (change / current_price) * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Price", f"${current_price:.2f}")
    with col2:
        st.metric(f"{days}-Day Prediction", f"${predicted_price:.2f}", 
                  f"{change:+.2f} ({change_pct:+.2f}%)")
    with col3:
        st.metric("Confidence Range", 
                  f"${predicted_price - confidence_intervals[-1]:.2f} - ${predicted_price + confidence_intervals[-1]:.2f}")
    
    # Daily predictions table
    st.subheader("📋 Daily Predictions")
    
    pred_df = pd.DataFrame({
        'Date': [d.strftime('%Y-%m-%d') for d in future_dates],
        'Predicted Price': [f"${p:.2f}" for p in future_prices],
        'Daily Change': [f"${future_prices[i] - (future_prices[i-1] if i>0 else current_price):+.2f}" 
                        for i in range(len(future_prices))],
        'Daily Change %': [f"{(future_prices[i] - (future_prices[i-1] if i>0 else current_price)) / (future_prices[i-1] if i>0 else current_price) * 100:+.2f}%" 
                          for i in range(len(future_prices))],
        'Total Return': [f"{(p-current_price)/current_price*100:+.2f}%" for p in future_prices]
    })
    
    st.dataframe(pred_df, use_container_width=True, hide_index=True)
    
    # Download button
    csv = pred_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Predictions (CSV)",
        data=csv,
        file_name=f"{symbol}_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 10px;'>
        <b>Data Source:</b> Twelve Data API (Real-time) | 
        <b>Model:</b> Random Forest/XGBoost |
        <b>Disclaimer:</b> For Academic Purposes Only
    </div>
    """,
    unsafe_allow_html=True
)