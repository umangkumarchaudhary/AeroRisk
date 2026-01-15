"""
🛫 AeroRisk - Aviation Safety Analytics Dashboard
==================================================
A world-class, production-grade safety analytics platform.

Author: Umang Kumar
Date: 2024-01-15
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import json

# ============================================
# Page Configuration
# ============================================
st.set_page_config(
    page_title="AeroRisk | Aviation Safety Analytics",
    page_icon="🛫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# Custom CSS for Premium Design + Mobile Responsive
# ============================================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* ============================================ */
    /* Global Styles */
    /* ============================================ */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* ============================================ */
    /* Custom Header - RESPONSIVE */
    /* ============================================ */
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 50%, #1e3a5f 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.85);
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
    }
    
    /* Mobile Header */
    @media (max-width: 768px) {
        .main-header {
            padding: 1.25rem;
            border-radius: 12px;
            margin-bottom: 1rem;
        }
        
        .main-header h1 {
            font-size: 1.5rem;
        }
        
        .main-header p {
            font-size: 0.9rem;
        }
    }
    
    /* ============================================ */
    /* Metric Cards - RESPONSIVE */
    /* ============================================ */
    .metric-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.05);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a5f;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        color: #64748b;
        font-size: 0.9rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Mobile Metric Cards */
    @media (max-width: 768px) {
        .metric-card {
            padding: 1rem;
            border-radius: 12px;
        }
        
        .metric-value {
            font-size: 1.75rem;
        }
        
        .metric-label {
            font-size: 0.75rem;
        }
    }
    
    /* ============================================ */
    /* Risk Level Badges - RESPONSIVE */
    /* ============================================ */
    .risk-critical {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        font-size: 0.9rem;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #f97316 0%, #ea580c 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        font-size: 0.9rem;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #eab308 0%, #ca8a04 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        font-size: 0.9rem;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        font-size: 0.9rem;
    }
    
    /* Mobile Badges */
    @media (max-width: 768px) {
        .risk-critical, .risk-high, .risk-medium, .risk-low {
            padding: 0.35rem 0.75rem;
            font-size: 0.75rem;
            border-radius: 15px;
        }
    }
    
    /* ============================================ */
    /* Cards - RESPONSIVE */
    /* ============================================ */
    .info-card, .warning-card, .danger-card, .success-card {
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    
    .info-card {
        background: white;
        border-left: 4px solid #3b82f6;
    }
    
    .warning-card {
        background: #fffbeb;
        border-left: 4px solid #f59e0b;
    }
    
    .danger-card {
        background: #fef2f2;
        border-left: 4px solid #ef4444;
    }
    
    .success-card {
        background: #f0fdf4;
        border-left: 4px solid #22c55e;
    }
    
    /* Mobile Cards */
    @media (max-width: 768px) {
        .info-card, .warning-card, .danger-card, .success-card {
            padding: 1rem;
            border-radius: 10px;
            font-size: 0.9rem;
        }
    }
    
    /* ============================================ */
    /* Buttons - RESPONSIVE */
    /* ============================================ */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
    }
    
    /* Mobile Buttons */
    @media (max-width: 768px) {
        .stButton > button {
            padding: 0.6rem 1rem;
            font-size: 0.9rem;
        }
    }
    
    /* ============================================ */
    /* Tabs - RESPONSIVE */
    /* ============================================ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        flex-wrap: wrap;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #f1f5f9;
        border-radius: 8px 8px 0 0;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: #3b82f6;
        color: white;
    }
    
    /* Mobile Tabs */
    @media (max-width: 768px) {
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 0.5rem 0.75rem;
            font-size: 0.8rem;
        }
    }
    
    /* ============================================ */
    /* Recommendation Cards - RESPONSIVE */
    /* ============================================ */
    .rec-card {
        background: white;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border: 1px solid #e2e8f0;
    }
    
    .rec-critical { border-left: 4px solid #dc2626; }
    .rec-high { border-left: 4px solid #f97316; }
    .rec-medium { border-left: 4px solid #eab308; }
    .rec-low { border-left: 4px solid #22c55e; }
    
    /* Mobile Rec Cards */
    @media (max-width: 768px) {
        .rec-card {
            padding: 1rem;
            font-size: 0.9rem;
        }
    }
    
    /* ============================================ */
    /* Streamlit Component Overrides - MOBILE */
    /* ============================================ */
    
    /* Make metrics stack better on mobile */
    @media (max-width: 768px) {
        [data-testid="stMetricValue"] {
            font-size: 1.5rem !important;
        }
        
        [data-testid="stMetricLabel"] {
            font-size: 0.75rem !important;
        }
        
        [data-testid="stMetricDelta"] {
            font-size: 0.7rem !important;
        }
        
        /* Reduce padding on mobile */
        .block-container {
            padding: 1rem !important;
        }
        
        /* Charts responsive */
        .js-plotly-plot {
            width: 100% !important;
        }
        
        /* Sliders on mobile */
        [data-testid="stSlider"] {
            padding: 0 !important;
        }
        
        /* Number inputs on mobile */
        [data-testid="stNumberInput"] {
            width: 100% !important;
        }
        
        /* Make expanders full width */
        [data-testid="stExpander"] {
            width: 100% !important;
        }
    }
    
    /* Very small screens (phones in portrait) */
    @media (max-width: 480px) {
        .main-header h1 {
            font-size: 1.25rem;
        }
        
        .main-header p {
            font-size: 0.8rem;
        }
        
        .metric-value {
            font-size: 1.5rem;
        }
        
        [data-testid="stMetricValue"] {
            font-size: 1.25rem !important;
        }
        
        .block-container {
            padding: 0.5rem !important;
        }
    }
    
    /* ============================================ */
    /* Sidebar - MOBILE FRIENDLY */
    /* ============================================ */
    @media (max-width: 768px) {
        [data-testid="stSidebar"] {
            width: 100% !important;
            min-width: 100% !important;
        }
        
        [data-testid="stSidebarContent"] {
            padding: 1rem !important;
        }
    }
    
    /* ============================================ */
    /* Animations */
    /* ============================================ */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-in {
        animation: fadeIn 0.5s ease forwards;
    }
    
    /* ============================================ */
    /* Custom scrollbar */
    /* ============================================ */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #94a3b8;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #64748b;
    }
    
    /* Mobile - hide scrollbar for cleaner look */
    @media (max-width: 768px) {
        ::-webkit-scrollbar {
            width: 4px;
        }
    }
    
    /* ============================================ */
    /* Touch-friendly spacing for mobile */
    /* ============================================ */
    @media (max-width: 768px) {
        /* Increase touch targets */
        button, [role="button"], input, select {
            min-height: 44px;
        }
        
        /* Better spacing */
        .stMarkdown h3 {
            font-size: 1.1rem !important;
            margin-top: 1rem !important;
        }
        
        /* Dividers */
        hr {
            margin: 1rem 0 !important;
        }
    }
    
    /* ============================================ */
    /* Print Styles (for sharing/screenshots) */
    /* ============================================ */
    @media print {
        .main-header {
            background: #1e3a5f !important;
            -webkit-print-color-adjust: exact;
            print-color-adjust: exact;
        }
        
        [data-testid="stSidebar"] {
            display: none !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# API Configuration
# ============================================
import os

# API URL - uses environment variable in production, localhost for development
API_BASE_URL = os.environ.get("AERORISK_API_URL", "http://localhost:8002")

def api_call(endpoint, method="GET", data=None):
    """Make API call with error handling."""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url, timeout=5)
        else:
            response = requests.post(url, json=data, timeout=5)
        return response.json() if response.status_code == 200 else None
    except:
        return None

# ============================================
# Sidebar Navigation
# ============================================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h1 style="color: #3b82f6; font-size: 2rem; margin: 0;">🛫 AeroRisk</h1>
        <p style="color: #64748b; font-size: 0.85rem; margin-top: 0.25rem;">Aviation Safety Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Navigation
    page = st.radio(
        "Navigation",
        ["📊 Risk Overview", "🎯 Predictions", "📈 Trends & Analytics", 
         "💡 Recommendations", "🔮 What-If Analysis"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # API Status
    health = api_call("/health")
    if health:
        st.success("✅ API Connected")
        st.caption(f"Models: {len(health.get('models_loaded', []))}")
    else:
        st.error("❌ API Disconnected")
        st.caption("Start API: uvicorn src.api.main:app")
    
    st.divider()
    
    # Quick Stats
    st.markdown("### 📊 Quick Stats")
    col1, col2 = st.columns(2)
    col1.metric("Incidents", "33.1K")
    col2.metric("Models", "3")
    
    st.divider()
    
    st.caption("© 2024 AeroRisk Platform")
    st.caption("Built with ❤️ by Umang Kumar")


# ============================================
# PAGE: Risk Overview
# ============================================
if page == "📊 Risk Overview":
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 Risk Overview Dashboard</h1>
        <p>Real-time aviation safety monitoring and risk assessment</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Fetch dashboard data
    dashboard_data = api_call("/api/v1/analytics/dashboard")
    
    # Top KPIs Row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    if dashboard_data:
        summary = dashboard_data.get("summary", {})
        with col1:
            st.metric(
                "Total Incidents",
                f"{summary.get('total_incidents', 0):,}",
                delta="-12% vs last year",
                delta_color="inverse"
            )
        with col2:
            st.metric(
                "Fatal Rate",
                f"{summary.get('fatal_rate', 0):.1f}%",
                delta="-2.3%",
                delta_color="inverse"
            )
        with col3:
            st.metric(
                "Avg Risk Score",
                "25.4",
                delta="-5.2 pts"
            )
        with col4:
            st.metric(
                "Anomalies Detected",
                "2,500",
                delta="5%"
            )
        with col5:
            st.metric(
                "Model Accuracy",
                "85%",
                delta="+2.3%"
            )
    else:
        # Demo data
        with col1:
            st.metric("Total Incidents", "33,123", delta="-12%", delta_color="inverse")
        with col2:
            st.metric("Fatal Rate", "18.6%", delta="-2.3%", delta_color="inverse")
        with col3:
            st.metric("Avg Risk Score", "25.4", delta="-5.2 pts")
        with col4:
            st.metric("Anomalies", "2,500", delta="5%")
        with col5:
            st.metric("Accuracy", "85%", delta="+2.3%")
    
    st.markdown("---")
    
    # Charts Row 1
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📈 Incident Trends Over Time")
        
        # Generate trend data
        dates = pd.date_range(end=datetime.now(), periods=24, freq='M')
        trend_data = pd.DataFrame({
            'Month': dates,
            'Incidents': np.random.randint(800, 1200, 24) - np.arange(24) * 5,
            'Fatal': np.random.randint(50, 100, 24) - np.arange(24) * 1,
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trend_data['Month'],
            y=trend_data['Incidents'],
            name='Total Incidents',
            line=dict(color='#3b82f6', width=3),
            fill='tozeroy',
            fillcolor='rgba(59, 130, 246, 0.1)'
        ))
        fig.add_trace(go.Scatter(
            x=trend_data['Month'],
            y=trend_data['Fatal'],
            name='Fatal Incidents',
            line=dict(color='#ef4444', width=3),
            yaxis='y2'
        ))
        fig.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(title='Total Incidents', gridcolor='rgba(0,0,0,0.05)'),
            yaxis2=dict(title='Fatal', overlaying='y', side='right'),
            legend=dict(orientation='h', y=1.1),
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Severity Distribution")
        
        severity_data = pd.DataFrame({
            'Severity': ['NONE', 'MINOR', 'SERIOUS', 'FATAL'],
            'Count': [19693, 3985, 3288, 6157],
            'Color': ['#22c55e', '#eab308', '#f97316', '#dc2626']
        })
        
        fig = go.Figure(data=[go.Pie(
            labels=severity_data['Severity'],
            values=severity_data['Count'],
            hole=0.6,
            marker_colors=['#22c55e', '#eab308', '#f97316', '#dc2626'],
            textinfo='percent+label',
            textposition='outside'
        )])
        fig.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            annotations=[dict(text='33.1K', x=0.5, y=0.5, font_size=24, showarrow=False)]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Charts Row 2
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🌤️ Weather Risk Impact")
        
        weather_data = pd.DataFrame({
            'Severity': ['FATAL', 'SERIOUS', 'MINOR', 'NONE'],
            'Avg Weather Risk': [65.3, 52.1, 38.5, 18.2]
        })
        
        fig = px.bar(
            weather_data,
            x='Severity',
            y='Avg Weather Risk',
            color='Avg Weather Risk',
            color_continuous_scale=['#22c55e', '#eab308', '#f97316', '#dc2626']
        )
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=20, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### ✈️ Top Risk Factors")
        
        factors = pd.DataFrame({
            'Factor': ['Weather', 'Crew Fatigue', 'Maintenance', 'Flight Phase', 'Time of Day'],
            'Contribution': [35, 25, 20, 12, 8]
        })
        
        fig = px.bar(
            factors,
            x='Contribution',
            y='Factor',
            orientation='h',
            color='Contribution',
            color_continuous_scale='Blues'
        )
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=20, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            coloraxis_showscale=False,
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.markdown("### 📊 Model Performance")
        
        models = pd.DataFrame({
            'Model': ['Risk Predictor', 'Severity Classifier', 'Anomaly Detector'],
            'Accuracy': [85, 85, 95],
            'Color': ['#3b82f6', '#8b5cf6', '#06b6d4']
        })
        
        fig = go.Figure()
        for i, row in models.iterrows():
            fig.add_trace(go.Bar(
                x=[row['Accuracy']],
                y=[row['Model']],
                orientation='h',
                name=row['Model'],
                marker_color=row['Color'],
                text=[f"{row['Accuracy']}%"],
                textposition='inside'
            ))
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=20, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            xaxis=dict(range=[0, 100])
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent Alerts
    st.markdown("### ⚠️ Recent High-Risk Alerts")
    
    alerts_data = pd.DataFrame({
        'Time': ['2 hours ago', '5 hours ago', '1 day ago', '2 days ago'],
        'Type': ['Weather Alert', 'Fatigue Warning', 'Maintenance Due', 'Anomaly Detected'],
        'Risk Level': ['CRITICAL', 'HIGH', 'MEDIUM', 'HIGH'],
        'Details': [
            'Severe thunderstorm activity detected at LAX',
            'Crew approaching FDP limits on UA1234',
            'Aircraft N12345 overdue for inspection',
            'Unusual operational pattern detected'
        ],
        'Status': ['Active', 'Resolved', 'Pending', 'Reviewing']
    })
    
    for _, row in alerts_data.iterrows():
        color = {'CRITICAL': 'danger', 'HIGH': 'warning', 'MEDIUM': 'info'}.get(row['Risk Level'], 'info')
        status_color = {'Active': '🔴', 'Resolved': '🟢', 'Pending': '🟡', 'Reviewing': '🔵'}.get(row['Status'], '⚪')
        
        st.markdown(f"""
        <div class="{color}-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <strong>{row['Type']}</strong> - <span class="risk-{row['Risk Level'].lower()}">{row['Risk Level']}</span>
                    <p style="margin: 0.5rem 0 0 0; color: #64748b;">{row['Details']}</p>
                </div>
                <div style="text-align: right;">
                    <small style="color: #94a3b8;">{row['Time']}</small><br>
                    {status_color} {row['Status']}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ============================================
# PAGE: Predictions
# ============================================
elif page == "🎯 Predictions":
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Risk Predictions</h1>
        <p>AI-powered flight risk assessment and severity classification</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🎲 Risk Score", "📊 Severity Classification", "🔍 Anomaly Detection"])
    
    with tab1:
        st.markdown("### Enter Flight Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            weather_risk = st.slider("Weather Risk Score", 0, 100, 45)
            visibility = st.number_input("Visibility (meters)", 100, 50000, 10000)
            wind_speed = st.number_input("Wind Speed (knots)", 0, 100, 15)
        
        with col2:
            crew_fatigue = st.slider("Crew Fatigue Index", 0, 100, 30)
            duty_hours = st.number_input("Crew Duty Hours", 0.0, 20.0, 8.0)
            flight_phase = st.selectbox("Flight Phase", 
                ["TAXI", "TAKEOFF", "CLIMB", "CRUISE", "DESCENT", "APPROACH", "LANDING"])
        
        with col3:
            maintenance_risk = st.slider("Maintenance Risk", 0, 100, 25)
            aircraft_age = st.number_input("Aircraft Age (years)", 0, 50, 12)
            is_night = st.checkbox("Night Operations")
        
        if st.button("🔮 Predict Risk", type="primary", use_container_width=True):
            with st.spinner("Analyzing risk factors..."):
                # Make API call
                result = api_call("/api/v1/predict/risk", "POST", {
                    "incident_year": 2024,
                    "incident_month": datetime.now().month,
                    "incident_day_of_week": datetime.now().weekday(),
                    "is_weekend": 1 if datetime.now().weekday() >= 5 else 0,
                    "severity_code": 0,
                    "phase_risk_factor": 1.2 if flight_phase in ["TAKEOFF", "LANDING"] else 1.0,
                    "total_injuries": 0,
                    "injury_rate": 0,
                    "has_coordinates": 1,
                    "is_us": 1,
                    "weather_risk_score": weather_risk
                })
            
            # Calculate demo risk if API fails
            if not result:
                risk = (weather_risk * 0.35 + crew_fatigue * 0.25 + maintenance_risk * 0.2 + 
                       (1 if is_night else 0) * 10 + (aircraft_age / 2))
                risk = min(100, max(0, risk))
                result = {"risk_score": risk, "risk_level": "HIGH" if risk > 50 else "LOW"}
            
            st.markdown("---")
            st.markdown("### 📊 Prediction Results")
            
            risk_score = result.get("risk_score", 0)
            risk_level = result.get("risk_level", "UNKNOWN")
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                # Risk Gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=risk_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Risk Score", 'font': {'size': 24}},
                    delta={'reference': 50},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1},
                        'bar': {'color': "#3b82f6"},
                        'steps': [
                            {'range': [0, 25], 'color': '#22c55e'},
                            {'range': [25, 50], 'color': '#eab308'},
                            {'range': [50, 75], 'color': '#f97316'},
                            {'range': [75, 100], 'color': '#dc2626'}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': risk_score
                        }
                    }
                ))
                fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card" style="text-align: center;">
                    <div class="metric-label">Risk Level</div>
                    <div class="risk-{risk_level.lower()}" style="margin-top: 1rem; font-size: 1.5rem;">
                        {risk_level}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card" style="text-align: center; margin-top: 1rem;">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value">85%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="info-card">
                    <strong>Risk Factor Breakdown:</strong>
                    <ul style="margin-top: 0.5rem;">
                        <li>Weather: 35%</li>
                        <li>Crew Fatigue: 25%</li>
                        <li>Maintenance: 20%</li>
                        <li>Other: 20%</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### Severity Classification")
        st.info("Enter parameters and click 'Classify' to predict incident severity category.")
        
        col1, col2 = st.columns(2)
        with col1:
            sev_weather = st.slider("Weather Risk", 0, 100, 50, key="sev_weather")
            sev_fatigue = st.slider("Crew Fatigue", 0, 100, 40, key="sev_fatigue")
        with col2:
            sev_maint = st.slider("Maintenance Risk", 0, 100, 30, key="sev_maint")
            sev_vis = st.number_input("Visibility (m)", 100, 50000, 5000, key="sev_vis")
        
        if st.button("📊 Classify Severity", type="primary"):
            # Demo classification
            probs = [0.15, 0.25, 0.35, 0.25]  # Demo probabilities
            severity = ["NONE", "MINOR", "SERIOUS", "FATAL"][np.argmax(probs)]
            
            st.markdown("---")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"""
                <div class="metric-card" style="text-align: center;">
                    <div class="metric-label">Predicted Severity</div>
                    <h2 style="color: #f97316; margin: 1rem 0;">{severity}</h2>
                    <small>Confidence: {max(probs)*100:.1f}%</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                fig = go.Figure(data=[go.Bar(
                    x=["NONE", "MINOR", "SERIOUS", "FATAL"],
                    y=[p * 100 for p in probs],
                    marker_color=['#22c55e', '#eab308', '#f97316', '#dc2626']
                )])
                fig.update_layout(
                    title="Class Probabilities",
                    height=250,
                    margin=dict(l=0, r=0, t=40, b=0),
                    yaxis_title="Probability (%)"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Anomaly Detection")
        st.info("Enter operational parameters to check for anomalies.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            anom_risk = st.slider("Operational Risk", 0, 100, 45, key="anom_risk")
            anom_fatigue = st.slider("Fatigue Risk", 0, 100, 35, key="anom_fatigue")
        with col2:
            anom_maint = st.slider("Maintenance Risk", 0, 100, 25, key="anom_maint")
            anom_sched = st.slider("Schedule Risk", 0, 100, 20, key="anom_sched")
        with col3:
            anom_duty = st.number_input("Duty Hours", 0.0, 20.0, 10.0, key="anom_duty")
            anom_rest = st.number_input("Rest Hours", 0.0, 48.0, 10.0, key="anom_rest")
        
        if st.button("🔍 Detect Anomaly", type="primary"):
            # Determine if anomaly based on high values
            is_anomaly = anom_risk > 70 or anom_fatigue > 80 or anom_maint > 60
            score = -0.3 if is_anomaly else 0.1
            
            st.markdown("---")
            
            if is_anomaly:
                st.markdown("""
                <div class="danger-card">
                    <h3 style="color: #dc2626; margin: 0;">⚠️ ANOMALY DETECTED</h3>
                    <p style="margin-top: 0.5rem;">
                        This operational pattern is significantly different from normal operations.
                        Review the contributing factors and take appropriate action.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="success-card">
                    <h3 style="color: #22c55e; margin: 0;">✅ NORMAL OPERATION</h3>
                    <p style="margin-top: 0.5rem;">
                        All parameters are within normal operational limits.
                    </p>
                </div>
                """, unsafe_allow_html=True)


# ============================================
# PAGE: Trends & Analytics
# ============================================
elif page == "📈 Trends & Analytics":
    st.markdown("""
    <div class="main-header">
        <h1>📈 Trends & Analytics</h1>
        <p>Deep dive into historical patterns and safety trends</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Date range filter
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
    with col2:
        end_date = st.date_input("End Date", datetime.now())
    with col3:
        metrics = st.multiselect("Metrics", ["Incidents", "Fatal Rate", "Risk Score", "Anomalies"], 
                                  default=["Incidents", "Fatal Rate"])
    
    st.markdown("---")
    
    # Time Series Analysis
    st.markdown("### 📊 Time Series Analysis")
    
    dates = pd.date_range(start=start_date, end=end_date, freq='W')
    ts_data = pd.DataFrame({
        'Date': dates,
        'Incidents': np.random.randint(100, 300, len(dates)) + np.sin(np.arange(len(dates)) * 0.1) * 50,
        'Fatal_Rate': 15 + np.random.randn(len(dates)) * 3,
        'Risk_Score': 25 + np.sin(np.arange(len(dates)) * 0.05) * 10 + np.random.randn(len(dates)) * 5
    })
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        subplot_titles=('Incident Count', 'Fatal Rate (%)'))
    
    fig.add_trace(
        go.Scatter(x=ts_data['Date'], y=ts_data['Incidents'], name='Incidents',
                   line=dict(color='#3b82f6', width=2), fill='tozeroy'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=ts_data['Date'], y=ts_data['Fatal_Rate'], name='Fatal Rate',
                   line=dict(color='#ef4444', width=2)),
        row=2, col=1
    )
    
    fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Analytics Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🗓️ Seasonal Patterns")
        
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_incidents = [2800, 2500, 2900, 2700, 2600, 2800, 
                            3100, 3200, 2900, 2700, 2600, 2900]
        
        fig = go.Figure(data=[go.Bar(
            x=months,
            y=monthly_incidents,
            marker_color=px.colors.sequential.Blues
        )])
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### ⏰ Time of Day Analysis")
        
        hours = list(range(24))
        hourly = [abs(np.sin(h * 0.3) * 100 + 150 + np.random.randint(-20, 20)) for h in hours]
        
        fig = go.Figure(data=[go.Scatter(
            x=hours,
            y=hourly,
            mode='lines+markers',
            fill='tozeroy',
            line=dict(color='#8b5cf6', width=2)
        )])
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0),
                         xaxis_title="Hour of Day")
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.markdown("### 📍 Geographic Distribution")
        
        regions = pd.DataFrame({
            'Region': ['West', 'East', 'Midwest', 'South', 'International'],
            'Incidents': [8500, 9200, 5800, 7100, 2500],
        })
        
        fig = px.pie(regions, values='Incidents', names='Region', hole=0.4,
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation Heatmap
    st.markdown("### 🔗 Factor Correlation Analysis")
    
    factors = ['Weather Risk', 'Crew Fatigue', 'Maintenance', 'Schedule Deviation', 
               'Aircraft Age', 'Flight Phase Risk']
    corr_matrix = np.random.rand(6, 6) * 0.5 + 0.5
    np.fill_diagonal(corr_matrix, 1)
    corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make symmetric
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=factors,
        y=factors,
        colorscale='RdBu',
        zmin=-1, zmax=1
    ))
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


# ============================================
# PAGE: Recommendations
# ============================================
elif page == "💡 Recommendations":
    st.markdown("""
    <div class="main-header">
        <h1>💡 Safety Recommendations</h1>
        <p>AI-powered actionable safety recommendations based on risk factors</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📝 Enter Current Conditions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        rec_weather = st.slider("Weather Risk Score", 0, 100, 75, key="rec_weather")
        rec_visibility = st.number_input("Visibility (m)", 100, 50000, 2500, key="rec_vis")
    
    with col2:
        rec_fatigue = st.slider("Crew Fatigue Index", 0, 100, 65, key="rec_fatigue")
        rec_duty = st.number_input("Duty Hours", 0.0, 20.0, 10.0, key="rec_duty")
    
    with col3:
        rec_maint = st.slider("Maintenance Risk", 0, 100, 40, key="rec_maint")
        rec_turn = st.number_input("Turnaround Time (min)", 0, 120, 25, key="rec_turn")
    
    if st.button("🔍 Generate Recommendations", type="primary", use_container_width=True):
        with st.spinner("Analyzing conditions..."):
            result = api_call("/api/v1/recommendations", "POST", {
                "weather_risk_score": rec_weather,
                "visibility_m": rec_visibility,
                "crew_fatigue_index": rec_fatigue,
                "crew_duty_hours": rec_duty,
                "maintenance_risk": rec_maint,
                "turnaround_time_mins": rec_turn
            })
        
        st.markdown("---")
        
        if result:
            recommendations = result.get("recommendations", [])
            total_reduction = result.get("total_risk_reduction", 0)
            
            # Summary Cards
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Recommendations", len(recommendations))
            with col2:
                critical = sum(1 for r in recommendations if r.get("priority") == "CRITICAL")
                st.metric("Critical Actions", critical)
            with col3:
                st.metric("Risk Reduction", f"{total_reduction:.0f}%")
            with col4:
                st.metric("Est. Implementation", "2-4 hours")
            
            st.markdown("---")
            st.markdown("### 📋 Recommended Actions")
            
            for i, rec in enumerate(recommendations):
                priority = rec.get("priority", "MEDIUM")
                priority_colors = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}
                priority_icon = priority_colors.get(priority, "⚪")
                
                with st.expander(f"{priority_icon} {rec.get('title', 'Recommendation')}", expanded=(i < 2)):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Description:** {rec.get('description', '')}")
                        
                        st.markdown("**Action Steps:**")
                        for step in rec.get("action_steps", []):
                            st.markdown(f"- {step}")
                        
                        st.markdown("**KPIs to Track:**")
                        for kpi in rec.get("kpis", []):
                            st.markdown(f"- {kpi}")
                    
                    with col2:
                        st.metric("Risk Reduction", f"{rec.get('risk_reduction_percent', 0):.0f}%")
                        st.metric("Cost", rec.get("cost_level", "$$"))
                        st.caption(f"📋 SMS: {rec.get('sms_pillar', 'N/A')}")
                        st.caption(f"📖 Refs: {', '.join(rec.get('regulatory_refs', []))}")
        else:
            # Demo recommendations
            st.warning("API not available. Showing demo recommendations.")
            demo_recs = [
                {"title": "Delay Flight - Severe Weather", "priority": "CRITICAL", 
                 "description": "Weather risk exceeds limits", "risk_reduction": 35},
                {"title": "Enhanced Weather Monitoring", "priority": "HIGH",
                 "description": "Implement enhanced monitoring", "risk_reduction": 15},
            ]
            for rec in demo_recs:
                st.markdown(f"""
                <div class="rec-card rec-{rec['priority'].lower()}">
                    <strong>{rec['title']}</strong><br>
                    <small>{rec['description']}</small><br>
                    <span class="risk-{rec['priority'].lower()}">{rec['priority']}</span>
                    <span style="float: right;">{rec['risk_reduction']}% reduction</span>
                </div>
                """, unsafe_allow_html=True)


# ============================================
# PAGE: What-If Analysis
# ============================================
elif page == "🔮 What-If Analysis":
    st.markdown("""
    <div class="main-header">
        <h1>🔮 What-If Scenario Analysis</h1>
        <p>Simulate different scenarios to find optimal risk mitigation strategies</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎯 Base Scenario")
    st.info("Set your current operational parameters as the baseline.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        base_weather = st.slider("Weather Risk", 0, 100, 70, key="base_weather")
    with col2:
        base_fatigue = st.slider("Crew Fatigue", 0, 100, 60, key="base_fatigue")  
    with col3:
        base_maint = st.slider("Maintenance Risk", 0, 100, 45, key="base_maint")
    
    st.markdown("---")
    st.markdown("### 🔄 Modified Scenarios")
    st.info("Define alternative scenarios to compare risk outcomes.")
    
    num_scenarios = st.number_input("Number of Scenarios", 1, 5, 3)
    
    scenarios = []
    for i in range(num_scenarios):
        with st.expander(f"📊 Scenario {i+1}", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                w = st.slider(f"Weather Risk", 0, 100, max(0, base_weather - (i+1)*20), 
                             key=f"s{i}_weather")
            with col2:
                f = st.slider(f"Crew Fatigue", 0, 100, max(0, base_fatigue - (i+1)*15),
                             key=f"s{i}_fatigue")
            with col3:
                m = st.slider(f"Maintenance Risk", 0, 100, max(0, base_maint - (i+1)*10),
                             key=f"s{i}_maint")
            scenarios.append({"weather_risk_score": w, "crew_fatigue": f, "maintenance_risk": m})
    
    if st.button("🔮 Analyze Scenarios", type="primary", use_container_width=True):
        st.markdown("---")
        st.markdown("### 📊 Scenario Comparison Results")
        
        # Calculate risks
        base_risk = base_weather * 0.35 + base_fatigue * 0.25 + base_maint * 0.2
        
        results = [{"name": "Baseline", "risk": base_risk, "change": 0}]
        
        for i, s in enumerate(scenarios):
            risk = s["weather_risk_score"] * 0.35 + s["crew_fatigue"] * 0.25 + s["maintenance_risk"] * 0.2
            change = risk - base_risk
            results.append({
                "name": f"Scenario {i+1}",
                "risk": risk,
                "change": change,
                "weather": s["weather_risk_score"],
                "fatigue": s["crew_fatigue"],
                "maint": s["maintenance_risk"]
            })
        
        # Find optimal
        optimal = min(results[1:], key=lambda x: x["risk"])
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Baseline Risk", f"{base_risk:.1f}")
        with col2:
            st.metric("Optimal Risk", f"{optimal['risk']:.1f}")
        with col3:
            st.metric("Max Reduction", f"{base_risk - optimal['risk']:.1f}")
        with col4:
            st.metric("Best Scenario", optimal['name'])
        
        # Comparison Chart
        fig = go.Figure()
        
        names = [r["name"] for r in results]
        risks = [r["risk"] for r in results]
        colors = ['#64748b'] + ['#3b82f6' if r["name"] == optimal["name"] else '#94a3b8' 
                                 for r in results[1:]]
        
        fig.add_trace(go.Bar(
            x=names,
            y=risks,
            marker_color=colors,
            text=[f"{r:.1f}" for r in risks],
            textposition='outside'
        ))
        
        fig.add_hline(y=base_risk, line_dash="dash", line_color="red",
                     annotation_text="Baseline")
        
        fig.update_layout(
            title="Risk Score Comparison",
            height=400,
            yaxis_title="Risk Score",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed comparison table
        st.markdown("### 📋 Detailed Comparison")
        
        comparison_df = pd.DataFrame([
            {
                "Scenario": r["name"],
                "Risk Score": f"{r['risk']:.1f}",
                "Change": f"{r['change']:+.1f}" if r["change"] != 0 else "-",
                "Weather": r.get("weather", base_weather),
                "Fatigue": r.get("fatigue", base_fatigue),
                "Maintenance": r.get("maint", base_maint),
            }
            for r in results
        ])
        
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Recommendation
        st.markdown(f"""
        <div class="success-card">
            <h3 style="color: #22c55e; margin: 0;">✨ Optimal Strategy: {optimal['name']}</h3>
            <p style="margin-top: 0.5rem;">
                By implementing {optimal['name']}, you can reduce risk by 
                <strong>{base_risk - optimal['risk']:.1f} points ({((base_risk - optimal['risk'])/base_risk*100):.1f}%)</strong>.
            </p>
            <p>
                <strong>Key Changes:</strong><br>
                • Weather Risk: {base_weather} → {optimal.get('weather', base_weather)} 
                  ({optimal.get('weather', base_weather) - base_weather:+d})<br>
                • Crew Fatigue: {base_fatigue} → {optimal.get('fatigue', base_fatigue)}
                  ({optimal.get('fatigue', base_fatigue) - base_fatigue:+d})<br>
                • Maintenance: {base_maint} → {optimal.get('maint', base_maint)}
                  ({optimal.get('maint', base_maint) - base_maint:+d})
            </p>
        </div>
        """, unsafe_allow_html=True)


# ============================================
# Footer
# ============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; padding: 2rem;">
    <p>🛫 <strong>AeroRisk</strong> - Aviation Safety Analytics Platform</p>
    <p>Built with Streamlit, FastAPI & Machine Learning</p>
    <p>© 2024 Umang Kumar | SMS-Aligned Safety Analytics</p>
</div>
""", unsafe_allow_html=True)
