import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from collections import Counter
import PyPDF2 # For PDF reading
import io
from datetime import datetime, timedelta, date
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import base64
import warnings # For suppressing warnings during ARIMA grid search

# Time Series / Forecasting
try:
    from prophet import Prophet
    # from prophet.plot import plot_plotly, plot_components_plotly # Not directly used but good to have if extending
except ImportError:
    st.error("Prophet library not found. Please install it by running: pip install prophet")
    st.stop()

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller # For ADF test in manual ARIMA
    from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning # For suppressing specific statsmodels warnings
except ImportError:
    st.error("statsmodels library not found. Please install: pip install statsmodels")
    st.stop()

try:
    from sklearn.linear_model import LinearRegression
    # from sklearn.model_selection import train_test_split # Not used in this version
    # from sklearn.metrics import mean_squared_error # Not used in this version
except ImportError:
    st.error("scikit-learn library not found. Please install it: pip install scikit-learn")
    st.stop()

# AI Model related imports have been removed.

# --- Page Config and CSS ---
st.set_page_config(
    page_title="Disease Forecast Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS ( 그대로 유지 ) - Keep your existing CSS here
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    body { font-family: 'Inter', sans-serif; }
    .main { padding: 2rem 1rem; }
    .main-header { font-size: 2.5rem; color: #1E3A8A; text-align: center; margin-bottom: 1rem; font-weight: 700; text-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .subtitle { font-size: 1.1rem; color: #64748B; text-align: center; margin-bottom: 2rem; font-weight: 400; }
    .upload-section, .settings-section, .forecasting-parameters-section { background: linear-gradient(135deg, #F8FAFC 0%, #E2E8F0 100%); padding: 2rem; border-radius: 16px; border: 1px solid #CBD5E1; margin-bottom: 2rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
    .settings-section { background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%); border: 1px solid #FDE68A; }
    .upload-header { font-size: 1.4rem; color: #1E3A8A; margin-bottom: 1rem; font-weight: 600; }
    .metric-card { background: linear-gradient(135deg, #DBEAFE 0%, #BFDBFE 100%); padding: 1.5rem; border-radius: 12px; border: 1px solid #93C5FD; margin-bottom: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.05); text-align: center;}
    .metric-value { font-size: 2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.5rem; }
    .metric-label { font-size: 0.9rem; color: #64748B; font-weight: 500; }
    .insight-card { background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%); padding: 1rem; border-radius: 8px; border-left: 4px solid #0284C7; margin-bottom: 0.5rem; font-weight: 500; color: #0F172A; }
    .section-header { font-size: 1.5rem; color: #1E3A8A; margin: 2rem 0 1rem 0; font-weight: 600; border-bottom: 2px solid #E2E8F0; padding-bottom: 0.5rem; }
    .info-box { background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%); border: 1px solid #BAE6FD; border-radius: 8px; padding: 1rem; margin: 1rem 0; color: #0F172A; }
    .stSelectbox > div > div > div, .stNumberInput > div > div > input { background-color: #F8FAFC !important; border: 1px solid #CBD5E1 !important; border-radius: 8px !important; }
    .stButton > button { background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%) !important; color: white !important; border: none !important; border-radius: 8px !important; padding: 0.5rem 1rem !important; font-weight: 500 !important; box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important; }
    .stButton > button:hover { background: linear-gradient(135deg, #2563EB 0%, #1E40AF 100%) !important; box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important; }
    .disease-category { background: linear-gradient(135deg, #EDE9FE 0%, #DDD6FE 100%); padding: 0.5rem 1rem; border-radius: 6px; margin: 0.25rem; display: inline-block; font-size: 0.85rem; color: #5B21B6; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# --- Constants and Mappings ---
DISEASE_MEDICINE_MAP = {
    'hypertension': ['amlodipine', 'lisinopril', 'metoprolol', 'losartan', 'hydrochlorothiazide'],
    'diabetes': ['metformin', 'insulin', 'glipizide', 'glyburide', 'sitagliptin'],
    'asthma': ['albuterol', 'budesonide', 'fluticasone', 'montelukast', 'salmeterol'],
    'pain': ['ibuprofen', 'acetaminophen', 'naproxen', 'tramadol', 'codeine'],
    'depression': ['sertraline', 'fluoxetine', 'citalopram', 'escitalopram', 'venlafaxine'],
    'anxiety': ['alprazolam', 'lorazepam', 'clonazepam', 'diazepam', 'buspirone'],
    'bacterial_infection': ['amoxicillin', 'azithromycin', 'doxycycline', 'ciprofloxacin', 'cephalexin'],
    'gerd': ['omeprazole', 'lansoprazole', 'pantoprazole', 'esomeprazole', 'ranitidine'],
    'allergies': ['cetirizine', 'loratadine', 'fexofenadine', 'diphenhydramine', 'fluticasone'],
    'copd': ['tiotropium', 'formoterol', 'budesonide', 'ipratropium'],
    'heart_disease': ['aspirin', 'clopidogrel', 'simvastatin', 'atorvastatin', 'warfarin'],
    'arthritis': ['diclofenac', 'celecoxib', 'meloxicam', 'prednisone'],
    'migraine': ['sumatriptan', 'rizatriptan', 'topiramate', 'propranolol'],
    'insomnia': ['zolpidem', 'eszopiclone', 'trazodone', 'melatonin'],
    'uti': ['trimethoprim', 'nitrofurantoin', 'sulfamethoxazole'],
    'hypothyroidism': ['levothyroxine', 'liothyronine', 'synthroid'],
    'epilepsy': ['levetiracetam', 'carbamazepine', 'valproic acid', 'lamotrigine'],
    'osteoporosis': ['alendronate', 'risedronate', 'zoledronic acid', 'calcium', 'vitamin d'],
    'ulcer': ['sucralfate', 'misoprostol', 'clarithromycin']
    # Add more mappings as needed
}
# AI-related constants have been removed.

# --- AI Model Functions have been removed ---

# --- Data Processing Functions ---
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text: text += page_text
            except Exception: pass # Ignore pages that fail extraction
        return text if text.strip() else None
    except Exception as e:
        st.error(f"Error reading PDF file structure: {str(e)}")
        return None

def pdf_to_dataframe(pdf_text):
    if not pdf_text: return None
    lines = pdf_text.split('\n'); data = []; headers = None
    potential_headers_info = []

    for i, line in enumerate(lines):
        line = line.strip()
        if not line: continue
        parts = re.split(r'\s{2,}|\t|,', line)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) < 2: continue

        is_likely_header = sum(bool(re.search(r'[a-zA-Z]', part)) for part in parts) / len(parts) > 0.6 and \
                           sum(part.replace('.', '', 1).isdigit() for part in parts) < len(parts) * 0.4
        if is_likely_header and i < len(lines) * 0.3: # Header likely in first 30%
            potential_headers_info.append({'index': i, 'headers': parts, 'num_cols': len(parts)})
    
    if potential_headers_info:
        # Simplistic: pick the one with most columns or first one
        best_header = sorted(potential_headers_info, key=lambda x: x['num_cols'], reverse=True)[0]
        headers = best_header['headers']
        header_line_idx = best_header['index']
        for k in range(header_line_idx + 1, len(lines)):
            parts_k = re.split(r'\s{2,}|\t|,', lines[k].strip())
            parts_k = [p.strip() for p in parts_k if p.strip()]
            if parts_k: data.append(parts_k)
    else: # Fallback: treat first non-empty multi-part line as header
        for line in lines:
            parts = re.split(r'\s{2,}|\t|,', line.strip())
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) > 1:
                if headers is None: headers = parts
                else: data.append(parts)
    
    if headers and data:
        max_cols = len(headers)
        cleaned_data = []
        for row in data:
            final_row = row[:max_cols] + [''] * (max_cols - len(row)) if len(row) < max_cols else row[:max_cols]
            cleaned_data.append(final_row)
        if cleaned_data: return pd.DataFrame(cleaned_data, columns=headers)
    elif data and not headers: # Data but no headers found
        return pd.DataFrame(data, columns=[f"Column_{j+1}" for j in range(len(data[0]))])
    return None

def identify_medicine_column(df):
    possible_names = ['medicine', 'drug', 'medication', 'product', 'name', 'product_name', 'item_name', 'generic_name']
    candidate_cols = []
    for col in df.columns:
        col_str = str(col).lower().replace("_", "").replace(" ", "")
        score = 0
        for p_name in possible_names:
            if p_name in col_str: score += 2
        is_object_dtype = df[col].dtype == 'object'
        has_strings = is_object_dtype and df[col].dropna().apply(lambda x: isinstance(x, str) and bool(re.search(r'[a-zA-Z]', x))).any()
        if has_strings:
            num_unique = df[col].nunique()
            if num_unique > 1 and num_unique < len(df): score +=1 # Good diversity
            if df[col].dropna().astype(str).str.len().mean() > 3 : score +=1 # Reasonable length
        if score > 0: candidate_cols.append({'col': col, 'score': score})
    if candidate_cols: return sorted(candidate_cols, key=lambda x: x['score'], reverse=True)[0]['col']
    for col in df.columns: # Fallback to first object column
        if df[col].dtype == 'object': return col
    return df.columns[0] if len(df.columns) > 0 else None

def clean_medicine_name(medicine):
    if pd.isna(medicine): return ""
    med = str(medicine).lower().strip()
    med = re.sub(r'\b\d+(\.\d+)?\s*(mg(?:/ml)?|mcg(?:/ml)?|g(?:/ml)?|ml|iu|meq|mmol|%)\b', '', med, flags=re.IGNORECASE)
    med = re.sub(r'\b(tab(?:let)?s?|cap(?:sule)?s?|inj(?:ection)?s?|sol(?:ution)?|oral|cream|oint(?:ment)?|susp(?:ension)?|syrup|powder|patch|aerosol|spray|gel|lotion|drops)\b', '', med, flags=re.IGNORECASE)
    med = re.sub(r'\(.*?\)|\[.*?\]', '', med)
    med = re.sub(r'\b(hcl|sodium|potassium|calcium|phosphate|acetate|maleate|succinate|tartrate|besylate|mesylate)\b', '', med, flags=re.IGNORECASE)
    med = re.sub(r'[^a-z0-9\s-]', '', med)
    med = re.sub(r'\s{2,}', ' ', med).strip().strip('- ')
    return med

def predict_disease_from_dictionary(medicine_name):
    clean_med = clean_medicine_name(medicine_name)
    if not clean_med: return 'unknown_empty_clean_med', 0
    best_match_len = 0; best_disease = 'unknown_dictionary_miss'; best_conf = 0
    for disease, med_keywords_list in DISEASE_MEDICINE_MAP.items():
        for med_keyword in med_keywords_list:
            if re.search(r'\b' + re.escape(med_keyword) + r'\b', clean_med):
                if len(med_keyword) > best_match_len: # Prioritize longer specific matches
                    best_match_len = len(med_keyword)
                    best_disease = disease
                    best_conf = 0.8 # Base confidence for dictionary match
    return best_disease, best_conf

def analyze_disease_patterns(df, medicine_col):
    disease_counter = Counter()
    medicine_disease_map = {}
    if medicine_col not in df.columns:
        return disease_counter, medicine_disease_map

    unique_medicines_raw = df[medicine_col].dropna().astype(str).unique()
    unique_medicine_predictions = {}

    progress_bar = None
    if len(unique_medicines_raw) > 0:
        progress_bar = st.progress(0, text=f"Analyzing {len(unique_medicines_raw)} unique medicines...")

    for i, med_name_raw in enumerate(unique_medicines_raw):
        # Always use the dictionary-based prediction method
        disease, confidence = predict_disease_from_dictionary(med_name_raw)
        unique_medicine_predictions[med_name_raw] = (disease, confidence)
        if progress_bar:
            progress_bar.progress((i + 1) / len(unique_medicines_raw), text=f"Analyzing medicine {i+1}/{len(unique_medicines_raw)}")
    
    if progress_bar:
        progress_bar.empty()

    for original_med_name_series in df[medicine_col]:
        original_med_name = str(original_med_name_series)
        if pd.notna(original_med_name_series) and original_med_name.strip() != "":
            disease, confidence = unique_medicine_predictions.get(original_med_name, ('unknown_mapping_error', 0))
            disease_counter[disease] += 1
            medicine_disease_map[original_med_name] = (disease, confidence)
        else:
            disease_counter['unknown_empty_medicine_field'] += 1
            medicine_disease_map[original_med_name if pd.notna(original_med_name_series) else "NaN_Entry"] = ('unknown_empty_medicine_field', 0)
    return disease_counter, medicine_disease_map

# --- General Multi-Disease Forecast & Charts ---
def generate_general_forecast(disease_counter, days=30, history_days=90):
    forecast_data = pd.DataFrame({'date': pd.to_datetime([datetime.now() + timedelta(days=i) for i in range(days)])})
    known_disease_counts = {k: v for k, v in disease_counter.items() if not k.startswith('unknown') and v > 0}
    if not known_disease_counts: return forecast_data
    
    for disease, current_count in known_disease_counts.items():
        hist_dates = pd.to_datetime([datetime.now() - timedelta(days=i) for i in range(history_days, 0, -1)])
        sim_y = np.clip(np.linspace(current_count * 0.3, current_count * 0.9, history_days) + np.random.normal(0, current_count * 0.15, history_days), 0, None)
        hist_df = pd.DataFrame({'ds': hist_dates, 'y': sim_y})
        hist_df = pd.concat([hist_df, pd.DataFrame({'ds': [pd.to_datetime(datetime.now())], 'y': [current_count]})]).sort_values(by='ds')
        try:
            m = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False, uncertainty_samples=0) # No CIs for speed
            if len(hist_df) > 1 : m.fit(hist_df)
            else: raise ValueError("Not enough data for Prophet fit")
            future = m.make_future_dataframe(periods=days)
            fcst = m.predict(future)
            forecast_data[disease] = np.clip(fcst['yhat'].iloc[-days:].values, 0, None).astype(int)
        except Exception: forecast_data[disease] = current_count # Fallback: flat line
    return forecast_data

def create_disease_distribution_chart(disease_counter):
    valid_diseases = {k:v for k,v in disease_counter.items() if not k.startswith('unknown') and v > 0}
    if not valid_diseases: return None
    return px.pie(values=list(valid_diseases.values()), names=[d.replace('_',' ').title() for d in valid_diseases.keys()], title="Overall Disease Distribution", hole=0.3)

def create_general_forecast_chart(forecast_df):
    if forecast_df.empty or len(forecast_df.columns) <=1: return None
    plot_cols = [col for col in forecast_df.columns if col != 'date' and not col.startswith('unknown')]
    if not plot_cols: return None
    
    # Select top N diseases by average forecast value if too many
    if len(plot_cols) > 7:
        avg_fcst = forecast_df[plot_cols].mean().sort_values(ascending=False)
        plot_cols = avg_fcst.head(7).index.tolist()

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly # Get a color sequence
    for i, col in enumerate(plot_cols):
        fig.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df[col], name=col.replace('_',' ').title(), line=dict(color=colors[i % len(colors)])))
    fig.update_layout(title="General Disease Forecast (Top Trends)", hovermode="x unified")
    return fig

def create_medicine_frequency_chart(df, medicine_col, top_n=15):
    if medicine_col not in df.columns or df[medicine_col].isnull().all(): return None
    try:
        cleaned_medicines = df[medicine_col].dropna().astype(str).apply(clean_medicine_name)
        counts = cleaned_medicines[cleaned_medicines != ""].value_counts().nlargest(top_n)
    except Exception: counts = df[medicine_col].dropna().astype(str).value_counts().nlargest(top_n) # Fallback
    if counts.empty: return None
    return px.bar(x=counts.values, y=counts.index, orientation='h', title=f"Top {top_n} Medicines (Cleaned Names)", labels={'x':'Frequency', 'y':'Medicine'}, color=counts.values, color_continuous_scale=px.colors.sequential.Blues_r)

def generate_insights(disease_counter, df, medicine_col):
    insights = []; known_counts = {k:v for k,v in disease_counter.items() if not k.startswith('unknown') and v > 0}
    if known_counts: insights.append(f"**Most Prominent Disease:** {Counter(known_counts).most_common(1)[0][0].replace('_',' ').title()} ({Counter(known_counts).most_common(1)[0][1]} mentions)")
    else: insights.append("No prominent diseases identified from known categories.")
    total_med_mentions = len(df[medicine_col].dropna()) if medicine_col in df.columns else 0
    insights.append(f"**Total Medicine Mentions Analyzed:** {total_med_mentions:,}")
    insights.append(f"**Unique Disease Categories Identified:** {len(known_counts)}")
    total_preds = sum(disease_counter.values()); unknown_total = sum(v for k,v in disease_counter.items() if k.startswith('unknown'))
    if total_preds > 0: insights.append(f"**Mentions Not Mapped to Known Diseases:** {(unknown_total/total_preds*100):.1f}% ({unknown_total:,} mentions)")
    return insights

# --- Single Disease Forecasting Section ---
def prepare_historical_data_for_disease(df_full, date_col_name, target_disease_name, medicine_col, analyzed_disease_map, synthetic_history_days=180):
    history_df = pd.DataFrame()
    actual_data_used = False

    if date_col_name and date_col_name in df_full.columns:
        try:
            df_temp = df_full.copy()
            df_temp['date_parsed'] = pd.to_datetime(df_temp[date_col_name], errors='coerce')
            df_temp.dropna(subset=['date_parsed'], inplace=True)
            df_temp['predicted_disease'] = df_temp[medicine_col].astype(str).map(lambda x: analyzed_disease_map.get(x, ('unknown', 0))[0])
            disease_specific_cases = df_temp[df_temp['predicted_disease'] == target_disease_name]

            if not disease_specific_cases.empty:
                daily_counts = disease_specific_cases.groupby(pd.Grouper(key='date_parsed', freq='D')).size().fillna(0)
                history_df = daily_counts.reset_index(); history_df.columns = ['ds', 'y']
                if len(history_df) >= 20: actual_data_used = True # Min threshold for some models
                else: st.caption(f"Less than 20 days of actual data for '{target_disease_name}'. Using synthetic history.")
            else: st.caption(f"No occurrences of '{target_disease_name}' with valid dates. Using synthetic history.")
        except Exception as e: st.caption(f"Error processing date column '{date_col_name}': {e}. Using synthetic history.")
    
    if not actual_data_used: # Generate synthetic history
        st.caption(f"Generating synthetic historical data for '{target_disease_name}' ({synthetic_history_days} days).")
        current_count = df_full[df_full[medicine_col].astype(str).apply(lambda x: analyzed_disease_map.get(x, ('unknown',0))[0] == target_disease_name)].shape[0]
        if current_count == 0: current_count = np.random.randint(1, 3) # Small base

        dates_hist = pd.to_datetime([date.today() - timedelta(days=i) for i in range(synthetic_history_days, 0, -1)])
        y_trend = np.linspace(current_count * 0.1, current_count * 0.7, synthetic_history_days)
        y_noise = np.random.normal(0, current_count * 0.1 + 0.5, synthetic_history_days) # Add small base noise
        y_weekly = np.sin(np.arange(synthetic_history_days) * (2 * np.pi / 7)) * (current_count * 0.05 + 0.2) # Small weekly seasonality
        y_values = np.clip(y_trend + y_noise + y_weekly, 0, None)

        history_df = pd.DataFrame({'ds': dates_hist, 'y': y_values})
        history_df = pd.concat([history_df, pd.DataFrame({'ds': [pd.to_datetime(date.today())], 'y': [current_count]})], ignore_index=True)
    
    history_df = history_df.sort_values(by='ds').reset_index(drop=True)
    return history_df

@st.cache_data(show_spinner="Running Prophet forecast...", ttl=3600)
def run_prophet_forecast(_historical_data_df, periods):
    if _historical_data_df.empty or len(_historical_data_df) < 2: return None, "Not enough data for Prophet."
    model = Prophet(yearly_seasonality='auto', weekly_seasonality='auto', daily_seasonality=False, uncertainty_samples=200) # Fewer samples for speed
    try: model.fit(_historical_data_df)
    except Exception as e: return None, f"Prophet fitting error: {e}"
    future = model.make_future_dataframe(periods=periods)
    forecast_df = model.predict(future)
    return forecast_df, None

@st.cache_data(show_spinner="Running ARIMA forecast...", ttl=3600)
def run_arima_forecast(_historical_data_df, periods):
    if _historical_data_df.empty or len(_historical_data_df) < 20: return None, "Not enough data for ARIMA (min 20 points)."
    series = _historical_data_df['y'].astype(float)
    d = 0; temp_series = series.copy()
    if temp_series.nunique() == 1: pass # Handle constant series later or let grid search find (0,0,0)
    else:
        for i in range(3):
            try:
                adf_result = adfuller(temp_series, regression='c', autolag='AIC') # Use constant, let AIC pick lag
                if adf_result[1] <= 0.05: d = i; break
                if len(temp_series.diff().dropna()) < 5 : break # Not enough data after diff
                temp_series = temp_series.diff().dropna()
            except Exception: d = i; break # Error in ADF, use current d
        else: d = 2
    
    best_aic = np.inf; best_order = None; best_model_fit = None
    p_range = range(0, 3); q_range = range(0, 3) # Smaller ranges for speed

    warnings.simplefilter('ignore', ConvergenceWarning)
    warnings.simplefilter('ignore', ValueWarning)
    warnings.simplefilter('ignore', UserWarning)

    for p_val in p_range:
        for q_val in q_range:
            if d > 0 and p_val == 0 and q_val == 0: continue
            current_order = (p_val, d, q_val)
            if len(series) < (p_val + d + 5): continue
            try:
                model = ARIMA(series, order=current_order, enforce_stationarity=False, enforce_invertibility=False)
                model_fit = model.fit()
                if model_fit.aic < best_aic:
                    best_aic = model_fit.aic; best_order = current_order; best_model_fit = model_fit
            except Exception: continue
    
    warnings.resetwarnings() # Reset warnings to default

    if best_model_fit is None: return None, "ARIMA fitting failed for all tried orders."
    st.caption(f"Best ARIMA order: {best_order} (AIC: {best_aic:.2f})")
    try:
        forecast_results = best_model_fit.get_forecast(steps=periods)
        forecast_values = np.asarray(forecast_results.predicted_mean)
        conf_int = np.asarray(forecast_results.conf_int())
        last_date = _historical_data_df['ds'].iloc[-1]
        forecast_dates = pd.to_datetime([last_date + timedelta(days=i) for i in range(1, periods + 1)])
        return pd.DataFrame({
            'ds': forecast_dates, 'yhat': np.clip(forecast_values, 0, None),
            'yhat_lower': np.clip(conf_int[:, 0], 0, None), 'yhat_upper': np.clip(conf_int[:, 1], 0, None)
        }), None
    except Exception as e: return None, f"ARIMA forecasting error with {best_order}: {e}"

@st.cache_data(show_spinner="Running Linear Regression forecast...", ttl=3600)
def run_linear_regression_forecast(_historical_data_df, periods):
    if _historical_data_df.empty or len(_historical_data_df) < 5: return None, "Not enough data for Linear Regression (min 5 points)."
    df = _historical_data_df.copy(); df['time_index'] = np.arange(len(df))
    X = df[['time_index']]; y = df['y']
    model = LinearRegression()
    try: model.fit(X, y)
    except Exception as e: return None, f"Linear Regression fitting error: {e}"
    last_time_idx = df['time_index'].iloc[-1]
    future_time_indices = np.arange(last_time_idx + 1, last_time_idx + 1 + periods).reshape(-1, 1)
    future_preds = np.clip(model.predict(future_time_indices), 0, None)
    last_date = df['ds'].iloc[-1]
    forecast_dates = pd.to_datetime([last_date + timedelta(days=i) for i in range(1, periods + 1)])
    return pd.DataFrame({'ds': forecast_dates, 'yhat': future_preds}), None

def plot_single_disease_forecast(historical_df, forecast_df_model, disease_name, model_name):
    fig = go.Figure()
    if not historical_df.empty:
        fig.add_trace(go.Scatter(x=historical_df['ds'], y=historical_df['y'], mode='lines+markers', name='Historical', line=dict(color='grey', width=1.5), marker=dict(size=4)))
    if forecast_df_model is not None and not forecast_df_model.empty:
        fig.add_trace(go.Scatter(x=forecast_df_model['ds'], y=forecast_df_model['yhat'], mode='lines', name=f'{model_name} Forecast', line=dict(color='dodgerblue', width=2)))
        if 'yhat_lower' in forecast_df_model.columns and 'yhat_upper' in forecast_df_model.columns:
            # Plot yhat_lower first, then yhat_upper filling to yhat_lower
            fig.add_trace(go.Scatter(x=forecast_df_model['ds'], y=forecast_df_model['yhat_lower'], mode='lines', line=dict(width=0), showlegend=False, name='Lower CI'))
            fig.add_trace(go.Scatter(x=forecast_df_model['ds'], y=forecast_df_model['yhat_upper'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0,176,246,0.2)', showlegend=False, name='Upper CI'))
    fig.update_layout(title=f"{disease_name.replace('_',' ').title()} - {model_name} Forecast", xaxis_title="Date", yaxis_title="Case Count", hovermode="x unified", legend_title_text='Legend')
    return fig

# --- Main Application ---
def main():
    st.markdown('<h1 class="main-header">Disease Forecast Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Advanced Healthcare Analytics & Disease Prediction System</p>', unsafe_allow_html=True)

    # Initialize session state variables
    session_defaults = {
        'df_processed': None,
        'disease_counter': None,
        'medicine_disease_map': None,
        'selected_medicine_col': None,
        'last_uploaded_filename': None,
        'date_col_name': None,
        'single_disease_hist_data': None,
        'single_disease_forecast_df': None,
        'last_single_disease_params': None # To track if params changed
    }
    for key, value in session_defaults.items():
        if key not in st.session_state: st.session_state[key] = value
    
    # --- Data Upload and Initial Setup ---
    with st.container(): # Use container for better grouping
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown('<h3 class="upload-header">1. Data Upload & Initial Analysis Setup</h3>', unsafe_allow_html=True)
        
        col1_upload, col2_upload_info = st.columns([2,1])
        with col1_upload:
            uploaded_file = st.file_uploader("Upload medicine data (CSV, Excel, PDF)", type=['csv', 'xlsx', 'xls', 'pdf'], key="fileuploader_main")
        with col2_upload_info:
            st.markdown("<div class='info-box' style='height:100%; font-size:0.9em;'>Supported: CSV, Excel, PDF (tabular data). <br>For PDF, ensure text is selectable.</div>", unsafe_allow_html=True)

        if uploaded_file is not None:
            current_filename = uploaded_file.name
            if current_filename != st.session_state.last_uploaded_filename:
                st.session_state.last_uploaded_filename = current_filename
                st.session_state.df_processed = None # Reset df for new file
                st.session_state.disease_counter = None; st.session_state.medicine_disease_map = None # Reset analysis products
                
                with st.spinner(f"Processing '{current_filename}'..."):
                    df_loaded = None
                    file_extension = current_filename.split('.')[-1].lower()
                    try:
                        if file_extension == "pdf":
                            pdf_text = extract_text_from_pdf(uploaded_file)
                            if pdf_text: df_loaded = pdf_to_dataframe(pdf_text)
                            else: st.warning("PDF text extraction yielded no content.")
                        elif file_extension in ["xlsx", "xls"]: df_loaded = pd.read_excel(uploaded_file)
                        else: df_loaded = pd.read_csv(uploaded_file, low_memory=False) # low_memory=False for mixed types

                        if df_loaded is not None and not df_loaded.empty:
                            st.session_state.df_processed = df_loaded.copy() # Store a copy
                            st.success(f"File '{current_filename}' loaded: {df_loaded.shape[0]} rows, {df_loaded.shape[1]} columns.")
                        elif df_loaded is None: st.error("Failed to parse file into a table.")
                        else: st.error("Uploaded file is empty after parsing.")
                    except Exception as e: st.error(f"Error processing file: {e}")
        st.markdown('</div>', unsafe_allow_html=True) # End upload-section

    # Proceed only if data is loaded
    if st.session_state.df_processed is not None and not st.session_state.df_processed.empty:
        df = st.session_state.df_processed

        # --- Analysis Configuration ---
        with st.container():
            st.markdown('<div class="settings-section">', unsafe_allow_html=True)
            st.markdown('<h3 class="upload-header">2. Analysis Configuration</h3>', unsafe_allow_html=True)
            
            # Auto-select or let user choose medicine column
            if st.session_state.selected_medicine_col is None or st.session_state.selected_medicine_col not in df.columns:
                st.session_state.selected_medicine_col = identify_medicine_column(df)

            selected_med_col_ui = st.selectbox(
                "Select Medicine Column", df.columns, 
                index=list(df.columns).index(st.session_state.selected_medicine_col) if st.session_state.selected_medicine_col in df.columns else 0,
                key="med_col_selector_main", help="Column containing medicine names."
            )
            if selected_med_col_ui != st.session_state.selected_medicine_col: # If user changes it
                st.session_state.selected_medicine_col = selected_med_col_ui
                st.session_state.disease_counter = None; st.session_state.medicine_disease_map = None # Force re-analysis

            # AI Toggle and model loading have been removed.

            # Optional Date Column for specific forecasting
            date_col_options = ["(Optional) Select Date Column"] + [col for col in df.columns if 'date' in str(col).lower() or df[col].dtype in ['datetime64[ns]', 'object']] # Basic filter
            st.session_state.date_col_name = st.selectbox("Date Column for Detailed Forecasts", date_col_options, index=0, key="date_col_main", help="Used for historical trends in single disease forecasting.")
            if st.session_state.date_col_name == "(Optional) Select Date Column": st.session_state.date_col_name = None
            
            st.markdown('</div>', unsafe_allow_html=True) # End settings-section

        # Run initial analysis if needed
        if st.session_state.selected_medicine_col and (st.session_state.disease_counter is None or st.session_state.medicine_disease_map is None):
            with st.spinner("Performing disease pattern analysis..."):
                st.session_state.disease_counter, st.session_state.medicine_disease_map = analyze_disease_patterns(
                    df, st.session_state.selected_medicine_col
                )
        
        # --- Display Overall Analysis Results ---
        if st.session_state.disease_counter:
            st.markdown('<h2 class="section-header">Overall Disease Insights & Trends</h2>', unsafe_allow_html=True)
            
            # Key Metrics
            # ... (your metric card display logic) ...

            insights_list = generate_insights(st.session_state.disease_counter, df, st.session_state.selected_medicine_col)
            st.markdown("#### Key Insights:")
            for item in insights_list: st.markdown(f"<div class='insight-card'>{item}</div>", unsafe_allow_html=True)

            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                fig_pie = create_disease_distribution_chart(st.session_state.disease_counter)
                if fig_pie: st.plotly_chart(fig_pie, use_container_width=True)
                else: st.info("No data for disease distribution chart.")
            with col_chart2:
                fig_med_freq = create_medicine_frequency_chart(df, st.session_state.selected_medicine_col)
                if fig_med_freq: st.plotly_chart(fig_med_freq, use_container_width=True)
                else: st.info("No data for medicine frequency chart.")
            
            with st.expander("Show General Disease Forecast (All Identified Diseases)", expanded=False):
                with st.spinner("Generating general forecast for all diseases..."):
                    general_fcst_df = generate_general_forecast(st.session_state.disease_counter)
                fig_gen_fcst = create_general_forecast_chart(general_fcst_df)
                if fig_gen_fcst: st.plotly_chart(fig_gen_fcst, use_container_width=True)
                else: st.info("Not enough data for general forecast visualization.")
            
            # --- Detailed Single Disease Outbreak Forecasting Section ---
            st.markdown("---")
            st.markdown('<h2 class="section-header" style="margin-top: 2rem;">Detailed Disease Outbreak Forecasting</h2>', unsafe_allow_html=True)
            
            known_diseases_for_select = sorted([d.replace("_", " ").title() for d, count in st.session_state.disease_counter.items() if not d.startswith('unknown') and count > 0])
            if not known_diseases_for_select:
                st.info("No specific diseases identified in the data to forecast individually.")
            else:
                st.markdown('<div class="forecasting-parameters-section">', unsafe_allow_html=True)
                st.markdown('<h4 class="upload-header">Forecasting Parameters</h4>', unsafe_allow_html=True)
                
                col_param1, col_param2, col_param3 = st.columns(3)
                with col_param1:
                    selected_disease_display = st.selectbox("Select Disease", options=known_diseases_for_select, key="single_disease_select_main")
                with col_param2:
                    forecast_periods = st.number_input("Forecast Period (days)", min_value=7, max_value=180, value=30, step=7, key="forecast_days_single_main")
                with col_param3:
                    model_type = st.selectbox("Model Type", options=["Prophet", "ARIMA", "Linear Regression"], key="model_type_single_main")
                st.markdown('</div>', unsafe_allow_html=True) # End forecasting-parameters-section

                selected_disease_key_current = selected_disease_display.lower().replace(" ", "_")
                
                # Store current forecast parameters
                current_forecast_params = (selected_disease_key_current, forecast_periods, model_type)

                # Button to generate forecast for single disease
                if st.button(f"Generate Forecast for {selected_disease_display}", key="generate_single_fcst_btn_main", type="primary"):
                    st.session_state.last_single_disease_params = current_forecast_params # Store params used for this run
                    with st.spinner(f"Forecasting {selected_disease_display} using {model_type}..."):
                        st.session_state.single_disease_hist_data = prepare_historical_data_for_disease(
                            df, st.session_state.date_col_name, selected_disease_key_current,
                            st.session_state.selected_medicine_col, st.session_state.medicine_disease_map,
                            synthetic_history_days=max(90, forecast_periods * 3) # More history for longer forecasts
                        )
                        
                        forecast_df_single_run = None; error_msg_run = None
                        if st.session_state.single_disease_hist_data.empty or len(st.session_state.single_disease_hist_data) < 2:
                             error_msg_run = f"Not enough historical data for '{selected_disease_display}'."
                        else:
                            if model_type == "Prophet": forecast_df_single_run, error_msg_run = run_prophet_forecast(st.session_state.single_disease_hist_data, forecast_periods)
                            elif model_type == "ARIMA": forecast_df_single_run, error_msg_run = run_arima_forecast(st.session_state.single_disease_hist_data, forecast_periods)
                            elif model_type == "Linear Regression": forecast_df_single_run, error_msg_run = run_linear_regression_forecast(st.session_state.single_disease_hist_data, forecast_periods)
                        
                        st.session_state.single_disease_forecast_df = forecast_df_single_run # Store result
                        st.session_state.single_disease_error_msg = error_msg_run # Store error

                # Display forecast visualization if available and params match last run
                if st.session_state.last_single_disease_params == current_forecast_params:
                    if 'single_disease_error_msg' in st.session_state and st.session_state.single_disease_error_msg:
                        st.error(f"Forecast Error: {st.session_state.single_disease_error_msg}")
                    elif st.session_state.single_disease_forecast_df is not None:
                        st.markdown(f"#### Forecast Visualization: {selected_disease_display} ({model_type})")
                        fig_single = plot_single_disease_forecast(st.session_state.single_disease_hist_data, st.session_state.single_disease_forecast_df, selected_disease_key_current, model_type)
                        st.plotly_chart(fig_single, use_container_width=True)
                        with st.expander("View Forecasted Data Table"):
                            display_fcst = st.session_state.single_disease_forecast_df[['ds', 'yhat']].copy()
                            display_fcst['ds'] = display_fcst['ds'].dt.strftime('%Y-%m-%d')
                            display_fcst.rename(columns={'ds': 'Date', 'yhat': 'Predicted Cases'}, inplace=True)
                            st.dataframe(display_fcst.head(min(15, forecast_periods)))
                    # else: No forecast generated yet for these parameters / after button click

                # Disease Overview always visible for the currently selected disease in dropdown
                st.markdown(f"--- \n#### Disease Overview: {selected_disease_display}")
                count_in_ds = st.session_state.disease_counter.get(selected_disease_key_current, 0)
                st.markdown(f"- **Mentions in current dataset:** {count_in_ds}")
                assoc_meds = DISEASE_MEDICINE_MAP.get(selected_disease_key_current, [])
                if assoc_meds: st.markdown(f"- **Commonly associated medicines (dictionary):** {', '.join(m.title() for m in assoc_meds[:5])}{'...' if len(assoc_meds) > 5 else ''}")
                else: st.markdown("- No specific medicines listed in the internal dictionary.")
                st.markdown(f"- *Further details about '{selected_disease_display}' could be integrated here.*")
                # Add Word Cloud and Data Download options if desired (from original code)
                if st.checkbox("Show Medicine Word Cloud for Overall Data", value=False, key="wc_main_toggle"):
                     if st.session_state.selected_medicine_col:
                        try:
                            med_text = ' '.join(df[st.session_state.selected_medicine_col].dropna().astype(str).apply(clean_medicine_name))
                            if med_text:
                                wc = WordCloud(width=800, height=300, background_color='white', colormap='Blues_r').generate(med_text)
                                fig_wc, ax = plt.subplots(figsize=(10,4)); ax.imshow(wc, interpolation='bilinear'); ax.axis('off'); st.pyplot(fig_wc)
                            else: st.info("Not enough text for word cloud.")
                        except Exception as e: st.error(f"Word cloud error: {e}")
                
                if st.checkbox("Show Raw Data Preview", value=False, key="raw_data_main_toggle"):
                    st.dataframe(df.head(50))
                    csv_export = df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Full Raw Data as CSV", csv_export, f"raw_data_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")


    else: # No file uploaded or df is empty
        st.markdown('<div class="info-box">**Welcome! Please upload a data file to begin analysis.**</div>', unsafe_allow_html=True)
        st.markdown("##### Example Data Structure:")
        sample_df = pd.DataFrame({
            'Patient_ID': [101, 102, 103, 104, 105],
            'Prescription_Date': pd.to_datetime(['2024-01-15', '2024-01-16', '2024-01-16', '2024-01-17', '2024-01-18']),
            'Medicine_Name': ['Metformin 500mg Tab', 'Lisinopril 10mg', 'Albuterol Inhaler 90mcg', 'Sertraline HCl 50mg', 'Ibuprofen 200mg'],
            'Quantity': [60, 30, 1, 30, 100]
        })
        st.dataframe(sample_df)
        st.markdown("##### Features:")
        st.markdown("- Automated disease pattern identification from medicine data using a built-in dictionary.")
        st.markdown("- General forecast trends for all identified diseases.")
        st.markdown("- Detailed, configurable outbreak forecasting for specific diseases using Prophet, ARIMA, or Linear Regression.")
        st.markdown("- Interactive visualizations and downloadable insights.")

if __name__ == "__main__":
    main()