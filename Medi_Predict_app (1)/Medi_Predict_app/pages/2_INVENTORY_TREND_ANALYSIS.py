import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
from plotly.subplots import make_subplots
import re
import fitz  # PyMuPDF
from pathlib import Path

# --- Page Configuration ---
st.set_page_config(
    page_title="Inventory Trend Analysis",
    page_icon="📊",
    layout="wide"
)

# --- Custom CSS for Professional Styling ---
st.markdown("""
<style>
    .main {
        background-color: #FEFEFE;
    }
    .stApp > header {
        background-color: transparent;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .main-header {
        background: linear-gradient(135deg, #2E5C8A 0%, #4A90C2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        color: white;
        margin-bottom: 0.5rem;
        font-size: 2.5rem;
        font-weight: 600;
    }
    .main-header p {
        color: #E8F4F8;
        font-size: 1.1rem;
        margin: 0;
    }
    .filter-section {
        background-color: #F8FAFB;
        border: 1px solid #E1E8ED;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 2rem;
    }
    .filter-header {
        color: #2E5C8A;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 2px solid #4A90C2;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #F5F7FA 0%, #C3CFE2 100%);
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #4A90C2;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .section-header {
        color: #2E5C8A;
        font-size: 1.4rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-left: 0.5rem;
        border-left: 4px solid #4A90C2;
    }
    .insight-item {
        background-color: #F0F4F7;
        border-left: 3px solid #4A90C2;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    .chart-container {
        background-color: white;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        border: 1px solid #E8F0F5;
        margin-bottom: 1rem;
    }
    .dataframe {
        border: 1px solid #E1E8ED !important;
        border-radius: 8px !important;
    }
    .dataframe th {
        background-color: #2E5C8A !important;
        color: white !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Data Processing Functions ---

def detect_columns(df):
    """Automatically detect relevant columns from a DataFrame."""
    columns = df.columns.str.lower().str.strip()
    original_columns = df.columns.tolist()
    
    col_map = {
        'date': ['date', 'time', 'period'],
        'medicine': ['medicine', 'drug', 'product', 'item', 'name'],
        'quantity': ['quantity', 'stock', 'inventory', 'units', 'qty']
    }
    
    detected = {}
    for key, keywords in col_map.items():
        detected[key] = None
        for col in columns:
            if any(keyword in col for keyword in keywords):
                detected[key] = original_columns[columns.tolist().index(col)]
                break
    return detected

def process_pdf_file(uploaded_file):
    """Extracts inventory data from a PDF file."""
    filename = Path(uploaded_file.name).stem
    date_match = re.search(r'(\d{4}[-_]?\d{2}[-_]?\d{2})|(\d{2}[-_]?\d{2}[-_]?\d{4})', filename)
    report_date = None
    if date_match:
        try:
            report_date = pd.to_datetime(date_match.group(0).replace('_', '-')).date()
        except ValueError:
            report_date = None

    if not report_date:
        report_date = datetime.now().date()
        st.info(f"Could not parse date from filename '{uploaded_file.name}'. Using today's date ({report_date}) for this file. For best results, name files with dates (e.g., 'inventory_2023-11-30.pdf').")

    records = []
    try:
        file_bytes = uploaded_file.getvalue()
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            parts = line.rsplit(maxsplit=1)
            if len(parts) == 2 and parts[1].replace('.', '', 1).isdigit():
                medicine_name = parts[0].strip()
                quantity = float(parts[1])
                if len(medicine_name) > 3 and not medicine_name.lower().startswith(('page', 'total', 'date', 'report')):
                    records.append({'Date': report_date, 'Medicine': medicine_name, 'Quantity': quantity})
    
    except Exception as e:
        st.error(f"Error processing PDF file '{uploaded_file.name}': {e}")
        return pd.DataFrame()

    if not records:
        st.warning(f"Could not automatically extract data from '{uploaded_file.name}'. The parser expects a simple text format where each line contains a medicine name followed by a quantity.")
        return pd.DataFrame()

    return pd.DataFrame(records)

@st.cache_data
def load_inventory_data(uploaded_files):
    """Load and process data from uploaded files (CSV, Excel, PDF)."""
    if not uploaded_files:
        return None, "No files uploaded. Please upload inventory data to begin."
    
    all_data = []
    for uploaded_file in uploaded_files:
        try:
            file_extension = Path(uploaded_file.name).suffix.lower()
            
            if file_extension == '.csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(uploaded_file)
            elif file_extension == '.pdf':
                df = process_pdf_file(uploaded_file)
                if df.empty:
                    continue
            else:
                st.warning(f"Unsupported file format: '{file_extension}' in {uploaded_file.name}. Skipping.")
                continue
            
            df = df.dropna(how='all')
            df.columns = df.columns.str.strip()
            all_data.append(df)
            
        except Exception as e:
            st.error(f"Error reading file '{uploaded_file.name}': {str(e)}")
            continue
    
    if not all_data:
        return None, "No valid data could be loaded from the uploaded files."
    
    combined_df = pd.concat(all_data, ignore_index=True)
    detected_cols = detect_columns(combined_df)
    
    missing_cols = [key for key, val in detected_cols.items() if val is None]
    if missing_cols:
        return combined_df, f"Could not automatically detect columns for: {', '.join(missing_cols)}. Please ensure your files have columns for Date, Medicine/Product, and Quantity/Stock."
    
    df_std = combined_df[[detected_cols['date'], detected_cols['medicine'], detected_cols['quantity']]].copy()
    df_std.columns = ['Date', 'Medicine', 'Quantity']
    
    try:
        df_std['Date'] = pd.to_datetime(df_std['Date'], errors='coerce')
        df_std.dropna(subset=['Date'], inplace=True)
        df_std['Quantity'] = pd.to_numeric(df_std['Quantity'], errors='coerce')
        df_std.dropna(subset=['Quantity'], inplace=True)
        df_std['Medicine'] = df_std['Medicine'].astype(str).str.strip().str.title()
    except Exception as e:
        return df_std, f"Error during data cleaning: {e}"
        
    df_std['Month'] = df_std['Date'].dt.to_period('M').astype(str)
    df_std['Year'] = df_std['Date'].dt.year
    df_std['MonthName'] = df_std['Date'].dt.strftime('%B %Y')
    
    return df_std.sort_values(by='Date'), None

def initialize_file_upload():
    """Manages the file upload widget and session state."""
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []

    if not st.session_state.uploaded_files:
        st.markdown("""
        <div style="background-color: #FFF3CD; border-left: 5px solid #FFEAA7; border-radius: 8px; padding: 1.5rem; margin-bottom: 2rem;">
            <h4 style="color: #856404; margin-bottom: 1rem;">No Inventory Data Found</h4>
            <p style="color: #856404;">Please upload your inventory data files (CSV, Excel, or PDF) to proceed.</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Upload Inventory Data Files",
            type=['csv', 'xlsx', 'xls', 'pdf'],
            accept_multiple_files=True,
            help="Upload one or more files containing your inventory data. Supported formats: CSV, Excel, PDF."
        )
        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files
            st.rerun()
        return False
    return True

def generate_insights(df, selected_medicines):
    """Generate automated insights based on the data."""
    if df.empty:
        return ["No data available for the selected filters."]
    
    insights = []
    monthly_summary = df.groupby(['Month', 'Medicine'])['Quantity'].sum().reset_index()
    
    if monthly_summary['Month'].nunique() > 1:
        months = sorted(monthly_summary['Month'].unique())
        latest_month_total = monthly_summary[monthly_summary['Month'] == months[-1]]['Quantity'].sum()
        prev_month_total = monthly_summary[monthly_summary['Month'] == months[-2]]['Quantity'].sum()
        
        if prev_month_total > 0:
            change_pct = ((latest_month_total - prev_month_total) / prev_month_total) * 100
            if change_pct > 5:
                insights.append(f"📈 **Overall Trend:** Total inventory for selected items increased by **{change_pct:.1f}%** in the last month.")
            elif change_pct < -5:
                insights.append(f"📉 **Overall Trend:** Total inventory for selected items decreased by **{abs(change_pct):.1f}%** in the last month.")
    
    if len(selected_medicines) > 1 and monthly_summary['Month'].nunique() > 1:
        changes = {}
        for med in selected_medicines:
            med_data = monthly_summary[monthly_summary['Medicine'] == med]
            if len(med_data) > 1:
                latest_qty = med_data.iloc[-1]['Quantity']
                prev_qty = med_data.iloc[-2]['Quantity']
                if prev_qty > 0:
                    changes[med] = ((latest_qty - prev_qty) / prev_qty) * 100
        if changes:
            max_increase_med = max(changes, key=changes.get)
            if changes[max_increase_med] > 10:
                insights.append(f"⬆️ **Top Mover:** **{max_increase_med}** saw the largest stock increase ({changes[max_increase_med]:.0f}%).")
            min_decrease_med = min(changes, key=changes.get)
            if changes[min_decrease_med] < -10:
                insights.append(f"⬇️ **Stock Alert:** **{min_decrease_med}** stock decreased the most ({changes[min_decrease_med]:.0f}%). Consider restocking.")

    for med in selected_medicines:
        med_data = df[df['Medicine'] == med]
        if not med_data.empty:
            avg_stock = med_data['Quantity'].mean()
            latest_stock = med_data.sort_values('Date').iloc[-1]['Quantity']
            if latest_stock < avg_stock * 0.5:
                insights.append(f"⚠️ **Low Stock:** **{med}** is currently at **{latest_stock:,.0f} units**, which is significantly below its average of {avg_stock:,.0f} units.")
    
    return insights if insights else ["Analysis complete. No significant patterns detected for the current selection."]


# --- Main App Logic ---
# Refined custom palette with only shades of blue and cream/beige
color_palette = ['#2E5C8A', '#4A90C2', '#87CEEB', '#D2B48C', '#F5F5DC', '#B0E0E6']

# Header Section
st.markdown("""
<div class="main-header">
    <h1>Inventory Trend Analysis</h1>
    <p>Upload and analyze medicine inventory data to uncover trends and insights.</p>
</div>
""", unsafe_allow_html=True)

try:
    if not initialize_file_upload():
        st.stop() 

    df, error_message = load_inventory_data(st.session_state.uploaded_files)

    if df is None:
        st.error(error_message)
        st.stop()
    
    if error_message:
        st.warning(error_message)
        st.info("Attempting to proceed with the data that was successfully processed...")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.success(f"Successfully loaded {len(df)} records from {len(st.session_state.uploaded_files)} file(s).")
    with col2:
        if st.button("Clear Data & Upload New", help="Remove all loaded data and start over"):
            keys_to_clear = ['uploaded_files', 'selected_medicines']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    with st.expander("Preview Loaded Data and Summary"):
        st.dataframe(df.head(10), use_container_width=True)
        st.metric("Total Records", f"{len(df):,}")
        st.metric("Unique Medicines", f"{df['Medicine'].nunique():,}")
        st.metric("Date Range", f"{df['Date'].min().strftime('%d-%b-%Y')} to {df['Date'].max().strftime('%d-%b-%Y')}")
        st.metric("Total Quantity in Stock", f"{df['Quantity'].sum():,.0f}")

    if df.empty or df['Medicine'].nunique() == 0:
        st.error("No valid data or medicines found after processing. Please check your files.")
        st.stop()
        
    st.markdown('<div class="filter-header">Analysis Filters</div>', unsafe_allow_html=True)
    filter_col1, filter_col2 = st.columns(2)
    
    with filter_col1:
        all_medicines = sorted(df['Medicine'].unique())
        if 'selected_medicines' not in st.session_state:
            st.session_state.selected_medicines = all_medicines[:min(3, len(all_medicines))]
            
        selected_medicines = st.multiselect(
            "Select Medicines to Analyze:",
            all_medicines,
            default=st.session_state.selected_medicines,
            help="Choose one or more medicines for the analysis."
        )
        st.session_state.selected_medicines = selected_medicines
        
    with filter_col2:
        min_date, max_date = df['Date'].min().date(), df['Date'].max().date()
        date_range = st.date_input(
            "Select Date Range:",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            help="Select the time period for the analysis."
        )

    if not selected_medicines:
        st.warning("Please select at least one medicine to analyze.")
        st.stop()
        
    if len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        df_filtered = df[
            (df['Medicine'].isin(selected_medicines)) &
            (df['Date'] >= start_date) &
            (df['Date'] <= end_date)
        ]
    else:
        df_filtered = df[df['Medicine'].isin(selected_medicines)]

    if df_filtered.empty:
        st.warning("No data available for the selected medicines and date range. Please adjust your filters.")
        st.stop()

    st.markdown("---")

    st.markdown('<div class="section-header">Dashboard</div>', unsafe_allow_html=True)
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    kpi_col1.metric("Total Quantity (Selected)", f"{df_filtered['Quantity'].sum():,.0f}")
    kpi_col2.metric("Avg. Monthly Quantity", f"{df_filtered.groupby('Month')['Quantity'].sum().mean():,.0f}")
    kpi_col3.metric("Months Analyzed", df_filtered['Month'].nunique())

    st.markdown('<div class="section-header">Automated Insights</div>', unsafe_allow_html=True)
    insights = generate_insights(df_filtered, selected_medicines)
    for insight in insights:
        st.markdown(f'<div class="insight-item">{insight}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Inventory Visualizations</div>', unsafe_allow_html=True)
    
    st.markdown("#### Inventory Levels Over Time")
    fig_trend = px.line(
        df_filtered, x='Date', y='Quantity', color='Medicine',
        title="Daily Inventory Levels", markers=True,
        labels={'Quantity': 'Quantity (Units)', 'Date': 'Date'},
        color_discrete_sequence=color_palette
    )
    fig_trend.update_layout(legend_title_text='Medicine', plot_bgcolor='white', paper_bgcolor='white')
    st.plotly_chart(fig_trend, use_container_width=True)

    viz_col1, viz_col2 = st.columns(2)
    with viz_col1:
        st.markdown("#### Monthly Aggregated Stock")
        monthly_data = df_filtered.groupby(['MonthName', 'Medicine'])['Quantity'].sum().reset_index()
        fig_bar = px.bar(
            monthly_data, x='MonthName', y='Quantity', color='Medicine',
            title="Total Monthly Stock by Medicine",
            labels={'MonthName': 'Month'},
            color_discrete_sequence=color_palette
        )
        fig_bar.update_layout(xaxis_tickangle=-45, plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig_bar, use_container_width=True)

    with viz_col2:
        st.markdown("#### Current Stock Distribution")
        
        latest_stock_per_medicine = df_filtered.loc[df_filtered.groupby('Medicine')['Date'].idxmax()]

        if not latest_stock_per_medicine.empty:
            fig_pie = px.pie(
                latest_stock_per_medicine, 
                values='Quantity', 
                names='Medicine',
                title="Latest Stock Share per Medicine",
                color_discrete_sequence=color_palette
            )
            fig_pie.update_traces(
                textposition='inside', 
                textinfo='percent+label', 
                textfont_size=12, 
                textfont_color='white',
                hovertemplate="<b>%{label}</b><br>Latest Quantity: %{value}<br>Share: %{percent}<extra></extra>"
            )
            fig_pie.update_layout(showlegend=False, paper_bgcolor='white')
            st.plotly_chart(fig_pie, use_container_width=True)
            st.caption("Shows the stock distribution based on the most recent record for each selected medicine.")
        else:
            st.info("No stock data available to display in pie chart.")

    st.markdown('<div class="section-header">Detailed Data Table</div>', unsafe_allow_html=True)
    st.dataframe(df_filtered, use_container_width=True)
    
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
    st.exception(e)
    st.info("Please try clearing the data and uploading your files again.")

st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #6B7280; padding: 1rem;">'
    'Inventory Trend Analysis Dashboard'
    '</div>', 
    unsafe_allow_html=True
)