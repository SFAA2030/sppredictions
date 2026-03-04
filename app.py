"""
S&P 500 Stock Price Predictor Web App
Uses trained model from train_model.py
Fetches live + historical data via pandas_datareader
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import os
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from pandas_datareader import data as pdr
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(
    page_title="S&P 500 Predictor",
    page_icon="📈",
    layout="wide"
)

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
        st.error(" Model files not found! Please run model_creator.py first to train the model.")
        st.stop()

model, scaler_X, scaler_y, features, metadata = load_model()

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("🔍 Stock Predictor")
symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL)", "AAPL").upper()
days = st.sidebar.slider("Prediction Horizon (Days)", 1, 30, 7)



st.sidebar.markdown("---")
st.sidebar.subheader("Model Info")
st.sidebar.info(f"**Best Model:** {metadata['best_model']}")
if 'mae_score' in metadata:
    st.sidebar.info(f"**MAE:** {metadata['mae_score']:.4f}")
st.sidebar.info(f"**R² Score:** {metadata['r2_score']:.3f}")
st.sidebar.info(f"**Training Date:** {metadata['training_date'][:10]}")

# -------------------------------
# FETCH DATA
# -------------------------------
@st.cache_data(ttl=3600)
def fetch_stock_data(symbol):
    """Fetch historical data using pandas_datareader"""
    end = datetime.now()
    start = end - timedelta(days=5*365)
    
    try:
        df = pdr.DataReader(symbol, 'stooq', start, end)
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        st.error(f"Failed to fetch data for {symbol}: {e}")
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
# MAIN APP
# -------------------------------
if symbol:
    st.header(f"📈 {symbol} Stock Prediction")

    # Fetch data
    with st.spinner(f"Fetching historical data for {symbol}..."):
        hist = fetch_stock_data(symbol)
        if hist.empty:
            st.stop()
        
        st.success(f"Data fetched: {len(hist)} days")
    
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
                st.error(" Failed to generate predictions")
                st.stop()
            
            last_date = featured_data.index[-1]
            future_dates = [last_date + timedelta(days=x+1) for x in range(len(future_prices))]
            
            st.success(" Predictions completed!")
        except Exception as e:
            st.error(f" Prediction failed: {str(e)}")
            st.stop()

    # -------------------------------
    # CHART
    # -------------------------------
    st.subheader("Price History & Prediction")
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3]
    )

    # Historical (last 90 days)
    historical_days = min(90, len(featured_data))
    hist_prices = featured_data['Close'][-historical_days:]
    hist_dates = featured_data.index[-historical_days:]
    
    fig.add_trace(
        go.Scatter(x=hist_dates, y=hist_prices, mode='lines', name='Historical', line=dict(color='#1f77b4', width=2)),
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
    
    # Download
    csv = pred_df.to_csv(index=False)
    st.download_button(
        label="Download Predictions",
        data=csv,
        file_name=f"{symbol}_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

else:
    st.info("Enter a stock symbol in the sidebar to start")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 10px;'>
        <b>Disclaimer:</b> Created For Academic Purpose!!!.
    </div>
    """,
    unsafe_allow_html=True
)