import streamlit as st
from datetime import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="MediPredict: Medicine Inventory & Disease Forecasting",
    layout="wide",
)

# --- CUSTOM STYLES ---
st.markdown("""
    <style>
    .main-title {
        font-size: 3em;
        font-weight: 700;
        color: #1a5276;
        margin-bottom: 0.5em;
    }
    .sub-title {
        font-size: 1.5em;
        font-weight: 500;
        color: #2874a6;
        margin-top: 0.5em;
    }
    .description {
        font-size: 1.1em;
        line-height: 1.6em;
        color: #1c2833;
        padding: 0.5em 0;
    }
    .footer {
        margin-top: 3em;
        font-size: 0.9em;
        color: gray;
    }
    /* --- DARK BLUE BUTTON STYLING --- */
    a.st-emotion-cache-1v0mbdj, a.st-emotion-cache-1v0mbdj:visited {
        background-color: #003366 !important;
        color: white !important;
        border-radius: 8px;
        padding: 0.5em 1em;
        font-weight: bold;
        text-align: center;
        display: inline-block;
        margin-bottom: 0.5em;
    }
    a.st-emotion-cache-1v0mbdj:hover {
        background-color: #0055aa !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- MAIN CONTENT ---
st.markdown('<div class="main-title">MEDIPREDICT: MEDICINE INVENTORY MANAGEMENT & DISEASE FORECASTING SYSTEM</div>', unsafe_allow_html=True)

st.markdown('<div class="description">MediPredict is a data-driven intelligent system designed to enhance hospital and public health operations. By combining medicine inventory management with disease outbreak forecasting, this platform empowers healthcare professionals to make informed decisions, minimize resource wastage, and optimize emergency preparedness.</div>', unsafe_allow_html=True)

st.markdown('<div class="sub-title">CORE FEATURES:</div>', unsafe_allow_html=True)

st.markdown("""
- **MEDICINE INVENTORY MANAGEMENT**: Track stock levels, monitor expiry dates, and optimize procurement schedules.
- **DISEASE OUTBREAK FORECASTING**: Predict potential disease surges using statistical and machine learning models trained on real-time and historical health data.
- **RESOURCE OPTIMIZATION**: Provide actionable insights to ensure availability of critical supplies during health crises.
- **DASHBOARD AND VISUAL REPORTS**: Visualize trends, forecasts, and inventory movements through interactive charts and summaries.
""")

# --- NAVIGATION LINKS ---
st.markdown('<div class="sub-title">EXPLORE MODULES:</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.page_link("pages/1_INVENTORY_MANAGEMENT.py", label="INVENTORY MANAGEMENT")
    st.page_link("pages/3_DISEASE_FORECAST.py", label="DISEASE FORECASTING")
with col2:
    st.page_link("pages/2_INVENTORY_TREND_ANALYSIS.py", label="INVENTORY TREND ANALYSIS")
    st.page_link("pages/4_GENERIC_MEDICINE_FINDER.py", label="GENERIC MEDICINE FINDER")
