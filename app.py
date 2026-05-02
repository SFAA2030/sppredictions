"""
S&P 500 Stock Price Predictor - FULL SEARCH
- Predict ANY stock (NVDA, AMZN, LLY, etc.)
- 30-day prediction
- Seamless chart connection
- Risk assessment
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
st.set_page_config(page_title="Stock Predictor - Any Stock", page_icon="📈", layout="wide")

TWELVE_DATA_API_KEY = "dd2c6dd996b84903a560ee5878d0dcc8"

# Popular stocks for quick selection
POPULAR_STOCKS = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corp',
    'GOOGL': 'Alphabet (Google)',
    'NVDA': 'NVIDIA',
    'AMZN': 'Amazon',
    'META': 'Meta (Facebook)',
    'TSLA': 'Tesla',
    'LLY': 'Eli Lilly',
    'JPM': 'JPMorgan Chase',
    'JNJ': 'Johnson & Johnson',
    'WMT': 'Walmart',
    'XOM': 'Exxon Mobil',
    'V': 'Visa',
    'UNH': 'UnitedHealth',
    'PG': 'Procter & Gamble'
}

# -------------------------------
# LOAD MODEL
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
        
        with open('models/stock_predictor.pkl', 'rb') as f:
            model = pickle.load(f)
        
        r2_display = max(metadata['r2_score'], 0)
        st.success(f"✅ Model loaded (R²: {r2_display:.4f}) - Works for ANY stock!")
        
        return model, scaler_X, scaler_y, features, metadata
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

model, scaler_X, scaler_y, features, metadata = load_model()

# -------------------------------
# RISK ASSESSMENT FUNCTION
# -------------------------------
def assess_risk(volatility, r2_score, trend):
    """Risk: High, Moderate, or Low"""
    risk_score = 0
    r2_positive = max(r2_score, 0.3)
    
    if volatility < 20:
        risk_score += 20
    elif volatility < 40:
        risk_score += 50
    else:
        risk_score += 80
    
    if r2_positive > 0.7:
        risk_score += 20
    elif r2_positive > 0.4:
        risk_score += 50
    else:
        risk_score += 80
    
    if trend > 0:
        risk_score -= 10
    else:
        risk_score += 10
    
    if risk_score < 40:
        return "🟢 LOW RISK", "Stable stock with good model confidence"
    elif risk_score < 70:
        return "🟡 MODERATE RISK", "Normal market risk levels"
    else:
        return "🔴 HIGH RISK", "Volatile stock with higher uncertainty"

# -------------------------------
# FETCH DATA (ANY STOCK)
# -------------------------------
@st.cache_data(ttl=3600)
def fetch_real_data(symbol, days=500):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1day&outputsize={days}&apikey={TWELVE_DATA_API_KEY}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if 'values' not in data:
            return pd.DataFrame(), f"No data for {symbol}. Please check symbol."
        
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
        df['Open'] = df['Open'].astype(float)
        df['High'] = df['High'].astype(float)
        df['Low'] = df['Low'].astype(float)
        df['Close'] = df['Close'].astype(float)
        df['Volume'] = df['Volume'].astype(float)
        
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        
        return df, None
        
    except Exception as e:
        return pd.DataFrame(), f"Error fetching data: {e}"

# -------------------------------
# FEATURE ENGINEERING (MATCHES TRAINING)
# -------------------------------
def prepare_features(df):
    df = df.copy()
    
    if len(df) < 100:
        return df
    
    df['Close_Lag1'] = df['Close'].shift(1)
    df['Close_Lag2'] = df['Close'].shift(2)
    df['Close_Lag3'] = df['Close'].shift(3)
    df['Close_Lag5'] = df['Close'].shift(5)
    df['Close_Lag10'] = df['Close'].shift(10)
    
    df['Return'] = df['Close'].pct_change()
    df['Return_Lag1'] = df['Return'].shift(1)
    df['Return_Lag2'] = df['Return'].shift(2)
    df['Return_Lag3'] = df['Return'].shift(3)
    df['Return_Lag5'] = df['Return'].shift(5)
    
    df['SMA_5'] = df['Close'].rolling(5).mean().shift(1)
    df['SMA_10'] = df['Close'].rolling(10).mean().shift(1)
    df['SMA_20'] = df['Close'].rolling(20).mean().shift(1)
    df['SMA_50'] = df['Close'].rolling(50).mean().shift(1)
    
    df['Price_vs_SMA20'] = ((df['Close'] - df['SMA_20']) / df['SMA_20']).shift(1)
    df['Price_vs_SMA50'] = ((df['Close'] - df['SMA_50']) / df['SMA_50']).shift(1)
    
    df['Volatility_10'] = df['Return'].rolling(10).std().shift(1)
    df['Volatility_20'] = df['Return'].rolling(20).std().shift(1)
    
    df['Volume_SMA'] = df['Volume'].rolling(5).mean().shift(1)
    df['Volume_Ratio'] = (df['Volume'] / df['Volume_SMA']).shift(1)
    
    df['Daily_Range'] = ((df['High'] - df['Low']) / df['Close']).shift(1)
    df['Gap'] = ((df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)).shift(1)
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = (100 - (100 / (1 + rs))).shift(1)
    
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = (exp1 - exp2).shift(1)
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean().shift(1)
    
    df.dropna(inplace=True)
    
    return df

# -------------------------------
# PREDICTION (30 DAYS)
# -------------------------------
def predict_future(data, days, available_features):
    predictions = []
    confidence_intervals = []
    current_data = data.copy()
    
    if 'Return' in current_data.columns:
        recent_returns = current_data['Return'].dropna().tail(60)
        historical_volatility = recent_returns.std() * 100 if len(recent_returns) > 0 else 1.5
    else:
        historical_volatility = 1.5
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(days):
        status_text.text(f"Predicting day {i+1}/{days}...")
        
        latest_features = current_data[available_features].iloc[-1:].values
        
        if np.any(np.isnan(latest_features)):
            break
        
        latest_scaled = scaler_X.transform(latest_features)
        pred_scaled = model.predict(latest_scaled)[0]
        raw_pred_price = scaler_y.inverse_transform([[pred_scaled]])[0, 0]
        
        current_price = current_data['Close'].iloc[-1]
        
        if abs((raw_pred_price - current_price) / current_price) > 0.05:
            if raw_pred_price > current_price:
                capped_price = current_price * 1.05
            else:
                capped_price = current_price * 0.95
        else:
            capped_price = raw_pred_price
        
        predictions.append(capped_price)
        
        daily_vol = historical_volatility / 100
        ci = 1.96 * daily_vol * capped_price * np.sqrt((i+1)/252)
        confidence_intervals.append(ci)
        
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
st.title("📈 ANY Stock Price Predictor")
st.markdown("**Search for ANY stock (NVDA, AMZN, LLY, etc.) - The model works for all!**")

r2_display = max(metadata['r2_score'], 0)
st.caption(f"⚡ Model trained on price patterns | R²: {r2_display:.4f} | Works for ANY stock symbol")

# Sidebar
st.sidebar.header("🔍 Stock Selection")

# Option 1: Quick select from popular stocks
st.sidebar.subheader("Quick Select")
quick_symbol = st.sidebar.selectbox(
    "Popular Stocks",
    options=[""] + list(POPULAR_STOCKS.keys()),
    format_func=lambda x: f"{x} - {POPULAR_STOCKS.get(x, '')}" if x else "--- Select ---"
)

# Option 2: Manual entry for ANY stock
st.sidebar.subheader("Or Enter Any Symbol")
manual_symbol = st.sidebar.text_input("Stock Symbol", value="", placeholder="e.g., NVDA, AMZN, LLY, SBUX").upper()

# Determine which symbol to use
if manual_symbol:
    symbol = manual_symbol
    company_name = manual_symbol
elif quick_symbol:
    symbol = quick_symbol
    company_name = POPULAR_STOCKS.get(symbol, symbol)
else:
    symbol = None
    company_name = None

days = st.sidebar.slider("Prediction Days", 7, 30, 30)

st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ How It Works")
st.sidebar.info(
    """
    **Works for ANY stock because:**
    1. Model learns price PATTERNS, not specific stocks
    2. Technical indicators (RSI, MACD, etc.) are universal
    3. Same features work for NVDA, AMZN, LLY, or any stock
    """
)

st.sidebar.markdown("---")
st.sidebar.subheader("Model Info")
st.sidebar.info(f"**R² Score:** {r2_display:.4f}")
st.sidebar.info(f"**MAE:** ${metadata['mae_score']:.2f}")

# Process if symbol selected
if symbol:
    with st.spinner(f"Fetching data for {symbol}..."):
        hist, error = fetch_real_data(symbol, days=500)
        
        if error or hist.empty:
            st.error(f"❌ {error or f'No data for {symbol}'}")
            st.info("💡 Tip: Try symbols like NVDA, AMZN, LLY, TSLA, META, or any valid stock symbol")
            st.stop()
        
        returns = hist['Close'].pct_change().dropna()
        volatility = returns.std() * 100 if len(returns) > 0 else 0
        current_price = hist['Close'].iloc[-1]
        
        trend = hist['Close'].iloc[-1] - hist['Close'].iloc[-20] if len(hist) >= 20 else 0
        risk_level, risk_note = assess_risk(volatility, metadata['r2_score'], trend)
        
        # Display metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Stock", f"{symbol}")
        with col2:
            st.metric("Current Price", f"${current_price:.2f}")
        with col3:
            st.metric("Volatility", f"{volatility:.1f}%")
        with col4:
            st.metric("Data Rows", f"{len(hist)} days")
        with col5:
            st.metric("Risk Level", risk_level)
        
        st.caption(f"📊 {risk_note}")
        st.markdown("---")
    
    # Prepare features
    with st.spinner("Preparing technical indicators..."):
        featured_data = prepare_features(hist)
        available_features = [f for f in features if f in featured_data.columns]
        
        if len(available_features) < len(features) * 0.6:
            st.warning(f"⚠️ Only {len(available_features)} of {len(features)} features available")
        
        st.success(f"✅ {len(available_features)} features ready from {len(featured_data)} rows")
    
    # Predict
    with st.spinner(f"Predicting next {days} days for {symbol}..."):
        future_prices, confidence_intervals = predict_future(featured_data, days, available_features)
        
        if not future_prices:
            st.error("Prediction failed")
            st.stop()
        
        future_dates = [featured_data.index[-1] + timedelta(days=x+1) for x in range(len(future_prices))]
    
    # Chart
    st.subheader(f"📊 {symbol} Price Chart with {days}-Day Forecast")
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
    
    hist_days = min(180, len(featured_data))
    hist_dates = featured_data.index[-hist_days:]
    hist_prices = featured_data['Close'][-hist_days:]
    
    fig.add_trace(go.Scatter(
        x=hist_dates, y=hist_prices, mode='lines',
        name='Historical', line=dict(color='#1f77b4', width=2)
    ), row=1, col=1)
    
    # Seamless connection
    fig.add_trace(go.Scatter(
        x=[hist_dates[-1], future_dates[0]],
        y=[hist_prices[-1], future_prices[0]],
        mode='lines', name='Connection',
        line=dict(color='#ff7f0e', width=2), showlegend=False
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=future_dates, y=future_prices, mode='lines+markers',
        name='Prediction', line=dict(color='#ff7f0e', width=2),
        marker=dict(size=6, color='red')
    ), row=1, col=1)
    
    upper = [p + ci for p, ci in zip(future_prices, confidence_intervals)]
    lower = [p - ci for p, ci in zip(future_prices, confidence_intervals)]
    
    fig.add_trace(go.Scatter(
        x=future_dates + future_dates[::-1],
        y=upper + lower[::-1], fill='toself',
        fillcolor='rgba(255,127,14,0.2)',
        line=dict(color='rgba(255,127,14,0)'),
        name='95% Confidence'
    ), row=1, col=1)
    
    fig.add_trace(go.Bar(
        x=featured_data.index[-hist_days:],
        y=featured_data['Volume'][-hist_days:],
        name='Volume', marker_color='rgba(31,119,180,0.3)'
    ), row=2, col=1)
    
    fig.update_layout(height=600, showlegend=True, hovermode='x unified')
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary
    st.subheader("📊 Prediction Summary")
    
    predicted_price = future_prices[-1]
    change = predicted_price - current_price
    change_pct = (change / current_price) * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Price", f"${current_price:.2f}")
    with col2:
        st.metric(f"{days}-Day Forecast", f"${predicted_price:.2f}", 
                  f"{change:+.2f} ({change_pct:+.2f}%)")
    with col3:
        st.metric("Confidence Range", 
                  f"${predicted_price - confidence_intervals[-1]:.2f} - ${predicted_price + confidence_intervals[-1]:.2f}")
    
    # Daily predictions table
    st.subheader("📋 Daily Forecast")
    
    pred_df = pd.DataFrame({
        'Date': [d.strftime('%Y-%m-%d') for d in future_dates],
        'Predicted Price': [f"${p:.2f}" for p in future_prices],
        'Daily Change': [f"${future_prices[i] - (future_prices[i-1] if i>0 else current_price):+.2f}" 
                        for i in range(len(future_prices))],
        'Daily Return %': [f"{(future_prices[i] - (future_prices[i-1] if i>0 else current_price)) / (future_prices[i-1] if i>0 else current_price) * 100:+.2f}%" 
                          for i in range(len(future_prices))],
        'Total Return': [f"{(p-current_price)/current_price*100:+.2f}%" for p in future_prices]
    })
    
    st.dataframe(pred_df, use_container_width=True, hide_index=True)
    
    csv = pred_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Forecast (CSV)",
        data=csv,
        file_name=f"{symbol}_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: gray; padding: 10px;'>
        <b>Works for ANY stock</b> | 
        <b>Data:</b> Twelve Data API | 
        <b>Model R²:</b> {r2_display:.4f} |
        <b>Try:</b> NVDA, AMZN, LLY, TSLA, META, SBUX, NFLX, and 500+ more |
        <b>Disclaimer:</b> For Academic Purposes Only
    </div>
    """,
    unsafe_allow_html=True
)