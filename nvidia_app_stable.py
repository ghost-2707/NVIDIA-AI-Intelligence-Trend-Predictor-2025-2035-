import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

# --- 1. Page Configuration & Custom UI Styling ---
st.set_page_config(page_title="NVIDIA AI Intelligence", layout="wide", initial_sidebar_state="expanded")

# Injecting Custom CSS for a professional dark theme
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { 
        background-color: #1e2130; 
        padding: 15px; 
        border-radius: 10px; 
        border-left: 5px solid #76b900; 
    }
    div.stButton > button:first-child { 
        background-color: #76b900; 
        color: white; 
        border-radius: 20px; 
        width: 100%;
        border: none;
        height: 3em;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Data Engine ---
@st.cache_data
def load_data():
    df = pd.read_csv('nvidia_stock_2015_to_2024.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date')

@st.cache_resource
def get_forecast(_df):
    temp_df = _df.copy()
    temp_df['days'] = (temp_df['date'] - temp_df['date'].min()).dt.days
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(temp_df[['days']].values, temp_df['close'].values)
    
    last_day = temp_df['days'].max()
    future_days = np.array([i for i in range(last_day + 1, last_day + 4000)]).reshape(-1, 1)
    preds = model.predict(future_days)
    
    last_date = temp_df['date'].max()
    dates = [(last_date + timedelta(days=i)) for i in range(1, 4000)]
    return pd.DataFrame({'date': dates, 'price': preds})

# --- 3. App Sidebar ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/2/21/Nvidia_logo.svg", width=150)
    st.title("AI Settings")
    train_btn = st.button("🚀 Train & Run Forecast")
    st.info("Analysis based on 10 years of NVIDIA historical price action.")

# --- 4. Main Dashboard Logic ---
df = load_data()

if train_btn or 'forecast' in st.session_state:
    if train_btn:
        with st.spinner("AI is analyzing 10-year market cycles..."):
            st.session_state['forecast'] = get_forecast(df)
    
    forecast = st.session_state['forecast']
    
    st.title("NVIDIA Stock Intelligence Dashboard")
    selected_year = st.select_slider("Select Forecast Year", options=sorted(forecast['date'].dt.year.unique()))
    
    year_data = forecast[forecast['date'].dt.year == selected_year]

    # Metrics Row
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Average", f"${year_data['price'].mean():.2f}")
    col2.metric("Predicted Peak", f"${year_data['price'].max():.2f}")
    col3.metric("Predicted Low", f"${year_data['price'].min():.2f}")

    # Interactive Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=year_data['date'], y=year_data['price'], mode='lines', 
                             line=dict(color='#76b900', width=4), fill='tozeroy',
                             fillcolor='rgba(118, 185, 0, 0.1)', name="AI Prediction"))
    fig.update_layout(template="plotly_dark", hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # Full View
    with st.expander("📊 Compare Historical vs. AI Future Trend (2015 - 2035)"):
        full_fig = go.Figure()
        full_fig.add_trace(go.Scatter(x=df['date'], y=df['close'], name="Historical", line=dict(color='#4b4b4b')))
        full_fig.add_trace(go.Scatter(x=forecast['date'], y=forecast['price'], name="Forecast", line=dict(color='#76b900', dash='dash')))
        full_fig.update_layout(template="plotly_dark")
        st.plotly_chart(full_fig, use_container_width=True)
else:
    st.warning("👈 Click the button in the sidebar to generate the AI forecast.")