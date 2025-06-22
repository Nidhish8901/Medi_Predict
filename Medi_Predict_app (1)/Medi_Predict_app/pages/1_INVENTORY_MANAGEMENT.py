import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import PyPDF2
import openpyxl

# --- NEW LIBRARIES FROM THE SNIPPET ---
import fitz  # PyMuPDF
import re

from typing import Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Medicine Inventory Management",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional styling (kept from original)
# Added a style for the primary (dangerous) button
st.markdown("""
<style>
    /* Main theme colors: whites, blues, beige */
    .main {
        background-color: #fafafa;
    }
    .stApp {
        background-color: #fafafa;
    }
    .main-header {
        background: linear-gradient(135deg, #2c5aa0 0%, #1e3a5f 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        color: white !important;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 600;
    }
    .section-card {
        background: white;
        border: 1px solid #e0e6ed;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(44, 90, 160, 0.1);
        height: 100%;
    }
    .section-title {
        color: #2c5aa0;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e0e6ed;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e0e6ed;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2c5aa0;
    }
    .metric-label {
        color: #666;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    .stButton > button {
        background-color: #2c5aa0;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #1e3a5f;
        box-shadow: 0 4px 12px rgba(44, 90, 160, 0.3);
    }
    /* Style for the dangerous/primary buttons */
    .stButton > button[kind="primary"] {
        background-color: #d9534f;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #c9302c;
    }
    .alert-success { background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724; padding: 1rem; border-radius: 6px; margin: 1rem 0; }
    .alert-warning { background-color: #fff3cd; border: 1px solid #ffeaa7; color: #856404; padding: 1rem; border-radius: 6px; margin: 1rem 0; }
    .alert-error { background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; padding: 1rem; border-radius: 6px; margin: 1rem 0; }
    .dataframe { border: 1px solid #e0e6ed !important; border-radius: 6px; }
    .stFileUploader > div { background-color: #f8f9fa; border: 2px dashed #c3cfe2; border-radius: 8px; padding: 2rem; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'inventory_df' not in st.session_state:
    st.session_state.inventory_df = pd.DataFrame(columns=[
        'medicine_name', 'quantity', 'unit_price', 'expiration_date',
        'manufacturer', 'location', 'stock_quantity', 'minimum_stock_level', 'formulation'
    ])

if 'transaction_history' not in st.session_state:
    st.session_state.transaction_history = pd.DataFrame(columns=[
        'timestamp', 'medicine_name', 'transaction_type', 'quantity_change', 'new_stock_level'
    ])


# --- DATA LOADING AND VALIDATION FUNCTIONS ---

def load_csv_data(file):
    try:
        df = pd.read_csv(file)
        return validate_inventory_data(df)
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        return None

def load_excel_data(file):
    try:
        df = pd.read_excel(file)
        return validate_inventory_data(df)
    except Exception as e:
        st.error(f"Error loading Excel: {str(e)}")
        return None

def validate_inventory_data(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]

    name_columns = ['medicine_name', 'product_name', 'name', 'medicine', 'product']
    medicine_col = None

    for col in name_columns:
        if col in df.columns:
            medicine_col = col
            break

    if medicine_col is None:
        st.error("Medicine name column not found. Please ensure your data contains a column like 'medicine_name', 'product_name', 'name', etc.")
        st.write("Current columns found:", df.columns.tolist())
        return None

    if medicine_col != 'medicine_name':
        df = df.rename(columns={medicine_col: 'medicine_name'})

    df = df.dropna(subset=['medicine_name'])

    column_mapping = {
        'qty': 'quantity',
        'stock_qty': 'stock_quantity',
        'stock': 'stock_quantity',
        'price': 'unit_price',
        'cost': 'unit_price',
        'expiry': 'expiration_date',
        'exp_date': 'expiration_date',
        'mfg': 'manufacturer',
        'brand': 'manufacturer',
        'loc': 'location',
        'warehouse': 'location',
        'min_stock': 'minimum_stock_level'
    }
    df = df.rename(columns=column_mapping)

    required_columns = {
        'quantity': 0,
        'unit_price': 0.0,
        'expiration_date': None,
        'manufacturer': 'Unknown',
        'location': 'Not Specified',
        'stock_quantity': 0,
        'minimum_stock_level': 10,
        'formulation': 'N/A'
    }

    for col, default_val in required_columns.items():
        if col not in df.columns:
            df[col] = default_val

    if 'stock_quantity' in df.columns and df['stock_quantity'].eq(0).all():
        if 'quantity' in df.columns:
            df['stock_quantity'] = df['quantity']

    try:
        for col in ['quantity', 'stock_quantity', 'unit_price', 'minimum_stock_level']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        if 'expiration_date' in df.columns:
            df['expiration_date'] = pd.to_datetime(df['expiration_date'], errors='coerce')
    except Exception as e:
        st.warning(f"Could not convert all data types correctly: {str(e)}")

    return df

# --- HELPER FUNCTIONS ---

def add_transaction(medicine_name, transaction_type, quantity_change, new_stock_level):
    new_transaction = pd.DataFrame({
        'timestamp': [datetime.now()], 'medicine_name': [medicine_name],
        'transaction_type': [transaction_type], 'quantity_change': [quantity_change],
        'new_stock_level': [new_stock_level]
    })
    st.session_state.transaction_history = pd.concat([st.session_state.transaction_history, new_transaction], ignore_index=True)

def get_low_stock_items():
    if st.session_state.inventory_df.empty: return pd.DataFrame()
    return st.session_state.inventory_df[st.session_state.inventory_df['stock_quantity'] <= st.session_state.inventory_df['minimum_stock_level']]

def get_expiring_items(days=30):
    if st.session_state.inventory_df.empty or 'expiration_date' not in st.session_state.inventory_df.columns: return pd.DataFrame()
    cutoff_date = datetime.now() + timedelta(days=days)
    df = st.session_state.inventory_df.copy().dropna(subset=['expiration_date'])
    return df[df['expiration_date'] <= cutoff_date]


def main():
    # Header
    st.markdown('<div class="main-header"><h1>Medicine Inventory Management</h1></div>', unsafe_allow_html=True)

    # --- Current Inventory & Data Import ---
  
    st.markdown('<div class="section-title">Current Inventory & Data Import</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Current Inventory Overview")
        if not st.session_state.inventory_df.empty:
            search_term = st.text_input("Search medicines in current inventory:", "", placeholder="Type to search by name...")

            filtered_df = st.session_state.inventory_df.copy()
            if search_term:
                filtered_df = filtered_df[filtered_df['medicine_name'].str.contains(search_term, case=False, na=False)]

            st.dataframe(filtered_df, use_container_width=True, height=350)
        else:
            st.info("Inventory is empty. Upload a file or add items manually to get started.")

    with col2:
        st.subheader("Import New Inventory File")
        st.info("Importing a new file will replace the entire current inventory.", icon="ℹ️")

        tab_pdf, tab_csv = st.tabs(["PDF Upload", "CSV Upload"])

        with tab_pdf:
            pdf_file = st.file_uploader("Upload PDF File", type=["pdf"], key="pdf_uploader")

            def extract_medicines_from_pdf(file):
                file.seek(0)
                pdf_bytes = file.read()
                if not pdf_bytes:
                    raise ValueError("The uploaded PDF file is empty.")
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                text = "\n".join(page.get_text() for page in doc)
                pattern = re.compile(r"(?P<name>[A-Z0-9 \-\(\)]+)\s+(?P<formulation>[A-Za-z0-9 \+\.,%]+)\s+(?P<stock>\d{2,6})")
                records = []
                for match in pattern.finditer(text):
                    name = match.group("name").strip()
                    formulation = match.group("formulation").strip()
                    stock = int(match.group("stock").strip())
                    records.append((name, formulation, stock))
                return records

            if pdf_file:
                if st.button("Extract & Import from PDF", use_container_width=True):
                    try:
                        with st.spinner("Extracting data from PDF..."):
                            extracted_records = extract_medicines_from_pdf(pdf_file)
                        if not extracted_records:
                            st.error("Could not extract any medicines from the PDF. The text format may not match the expected pattern.")
                        else:
                            st.success(f"Extracted {len(extracted_records)} medicines from PDF.")
                            df_from_pdf = pd.DataFrame(extracted_records, columns=['name', 'formulation', 'stock'])
                            validated_df = validate_inventory_data(df_from_pdf)
                            if validated_df is not None:
                                st.session_state.inventory_df = validated_df
                                st.session_state.transaction_history = pd.DataFrame(columns=st.session_state.transaction_history.columns)
                                st.success("PDF data successfully imported and set as new inventory!")
                                st.rerun()
                            else:
                                st.error("Failed to validate the extracted PDF data.")
                    except Exception as e:
                        st.error(f"An error occurred while processing the PDF: {e}")

        with tab_csv:
            csv_file = st.file_uploader("Upload CSV File", type=["csv"], key="csv_uploader")
            if csv_file:
                if st.button("Import from CSV", use_container_width=True):
                    with st.spinner("Processing CSV file..."):
                        new_data = load_csv_data(csv_file)
                    if new_data is not None and not new_data.empty:
                        st.session_state.inventory_df = new_data.copy()
                        st.session_state.transaction_history = pd.DataFrame(columns=st.session_state.transaction_history.columns)
                        st.success(f"Successfully imported {len(new_data)} items from CSV!")
                        st.rerun()
                    else:
                        st.error("Could not process the CSV file. Please check its format.")
        
        # --- NEW: Clear Inventory Option ---
        st.markdown("---")
        st.subheader(" Danger Zone")
        with st.expander("Clear Entire Inventory"):
            st.warning("This action is irreversible and will delete all inventory and transaction data.")
            if st.button("Yes, Clear All Data Now", use_container_width=True, type="primary"):
                st.session_state.inventory_df = pd.DataFrame(columns=[
                    'medicine_name', 'quantity', 'unit_price', 'expiration_date', 
                    'manufacturer', 'location', 'stock_quantity', 'minimum_stock_level', 'formulation'
                ])
                st.session_state.transaction_history = pd.DataFrame(columns=[
                    'timestamp', 'medicine_name', 'transaction_type', 'quantity_change', 'new_stock_level'
                ])
                st.success("All inventory and transaction data has been cleared.")
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # --- DASHBOARD METRICS ---
    if not st.session_state.inventory_df.empty:
        st.markdown("<br>", unsafe_allow_html=True)
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{len(st.session_state.inventory_df)}</div><div class="metric-label">Total Items</div></div>', unsafe_allow_html=True)
        with metric_cols[1]:
            total_value = (st.session_state.inventory_df['stock_quantity'] * st.session_state.inventory_df['unit_price']).sum()
            # MODIFIED: Changed currency to Rupees
            st.markdown(f'<div class="metric-card"><div class="metric-value">₹{total_value:,.0f}</div><div class="metric-label">Total Value</div></div>', unsafe_allow_html=True)
        with metric_cols[2]:
            low_stock = len(get_low_stock_items())
            st.markdown(f'<div class="metric-card"><div class="metric-value">{low_stock}</div><div class="metric-label">Low Stock Items</div></div>', unsafe_allow_html=True)
        with metric_cols[3]:
            expiring = len(get_expiring_items())
            st.markdown(f'<div class="metric-card"><div class="metric-value">{expiring}</div><div class="metric-label">Expiring Soon (30d)</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # --- ACTION SECTIONS (MODIFIED LAYOUT) ---
    col1, col2, col3 = st.columns(3)

    with col1: # Add New Medicine
        
        st.markdown('<div class="section-title">Add New Medicine</div>', unsafe_allow_html=True)
        with st.form("add_medicine_form"):
            medicine_name = st.text_input("Medicine Name", placeholder="e.g., Paracetamol 500mg")
            initial_stock = st.number_input("Initial Stock", min_value=0, value=100)
            minimum_stock = st.number_input("Minimum Stock Level", min_value=0, value=20)
            # MODIFIED: Changed currency to Rupees
            unit_price = st.number_input("Unit Price (₹)", min_value=0.0, value=1.50, step=0.01, format="%.2f")
            manufacturer = st.text_input("Manufacturer", placeholder="e.g., Pharma Inc.")
            location = st.text_input("Location", placeholder="e.g., Shelf A-3")
            expiration_date = st.date_input("Expiration Date", value=None)
            submitted = st.form_submit_button("Add Medicine", use_container_width=True)
            if submitted:
                if not medicine_name.strip():
                    st.error("Medicine name is required!")
                elif not st.session_state.inventory_df[st.session_state.inventory_df['medicine_name'].str.lower() == medicine_name.lower()].empty:
                    st.error("This medicine already exists in inventory!")
                else:
                    new_medicine = pd.DataFrame({
                        'medicine_name': [medicine_name], 'quantity': [initial_stock], 'unit_price': [unit_price],
                        'expiration_date': [pd.to_datetime(expiration_date) if expiration_date else None],
                        'manufacturer': [manufacturer or 'Unknown'], 'location': [location or 'Not Specified'],
                        'stock_quantity': [initial_stock], 'minimum_stock_level': [minimum_stock]
                    })
                    st.session_state.inventory_df = pd.concat([st.session_state.inventory_df, new_medicine], ignore_index=True)
                    add_transaction(medicine_name, "Initial Stock", initial_stock, initial_stock)
                    st.success(f"Successfully added {medicine_name}!")
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col2: # Update Stock

        st.markdown('<div class="section-title">Update Stock</div>', unsafe_allow_html=True)
        if not st.session_state.inventory_df.empty:
            medicine_list = sorted(st.session_state.inventory_df['medicine_name'].tolist())
            with st.form("update_stock_form"):
                selected_medicine = st.selectbox("Select Medicine", medicine_list, index=None, placeholder="Choose an item...")
                if selected_medicine:
                    current_stock = st.session_state.inventory_df[st.session_state.inventory_df['medicine_name'] == selected_medicine]['stock_quantity'].iloc[0]
                    st.info(f"Current stock for {selected_medicine}: {current_stock}")
                    update_type = st.radio("Update Type", ["Add Stock", "Remove Stock", "Set New Level"], horizontal=True)
                    if update_type == "Set New Level":
                        new_stock = st.number_input("New Stock Level", min_value=0, value=int(current_stock))
                    else:
                        change_amount = st.number_input("Quantity", min_value=1, value=10)
                submitted = st.form_submit_button("Update Stock", use_container_width=True)
                if submitted and selected_medicine:
                    idx = st.session_state.inventory_df[st.session_state.inventory_df['medicine_name'] == selected_medicine].index[0]
                    if update_type == "Add Stock":
                        new_stock_level, qty_change, trans_type = current_stock + change_amount, change_amount, "Stock Added"
                    elif update_type == "Remove Stock":
                        new_stock_level = max(0, current_stock - change_amount)
                        qty_change, trans_type = -(current_stock - new_stock_level), "Stock Removed"
                    else:
                        new_stock_level = new_stock
                        qty_change, trans_type = new_stock - current_stock, "Stock Adjusted"
                    st.session_state.inventory_df.loc[idx, 'stock_quantity'] = new_stock_level
                    add_transaction(selected_medicine, trans_type, qty_change, new_stock_level)
                    st.success(f"Stock for {selected_medicine} updated! New level: {new_stock_level}")
                    st.rerun()
        else:
            st.warning("No medicines in inventory.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3: # --- NEW: Delete Medicine Option ---
     
        st.markdown('<div class="section-title">Delete Medicine</div>', unsafe_allow_html=True)
        if not st.session_state.inventory_df.empty:
            medicine_list = sorted(st.session_state.inventory_df['medicine_name'].tolist())
            with st.form("delete_medicine_form"):
                medicine_to_delete = st.selectbox(
                    "Select Medicine to Delete", medicine_list, index=None, placeholder="Choose an item to delete..."
                )
                confirmed = st.checkbox("I confirm I want to permanently delete this item.")
                submitted = st.form_submit_button("Delete Medicine", use_container_width=True, type="primary")
                if submitted:
                    if not medicine_to_delete:
                        st.warning("Please select a medicine to delete.")
                    elif not confirmed:
                        st.warning("You must check the confirmation box to delete the item.")
                    else:
                        item_stock = st.session_state.inventory_df.loc[st.session_state.inventory_df['medicine_name'] == medicine_to_delete, 'stock_quantity'].iloc[0]
                        st.session_state.inventory_df = st.session_state.inventory_df[
                            st.session_state.inventory_df['medicine_name'] != medicine_to_delete
                        ].reset_index(drop=True)
                        add_transaction(medicine_to_delete, "Item Deleted", -item_stock, 0)
                        st.success(f"Successfully deleted '{medicine_to_delete}' from the inventory.")
                        st.rerun()
        else:
            st.warning("No medicines to delete.")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- REPORTING SECTION (MOVED) ---
    report_type = None
    generate_report = False
    
   
    st.markdown('<div class="section-title">Generate Report</div>', unsafe_allow_html=True)
    if not st.session_state.inventory_df.empty:
        report_type = st.selectbox("Select Report Type", ["Stock Status", "Low Stock Report", "Expiry Report", "Stock Value Report", "Transaction History"])
        generate_report = st.button("Generate Report", use_container_width=True)
    else:
        st.warning("No inventory data for reporting.")
    st.markdown('</div>', unsafe_allow_html=True)

    if generate_report:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="section-title">{report_type}</div>', unsafe_allow_html=True)
        if report_type == "Stock Status":
            df_sorted = st.session_state.inventory_df.sort_values('stock_quantity', ascending=False)
            fig = px.bar(df_sorted, x='medicine_name', y='stock_quantity', title="Current Stock Levels", labels={'stock_quantity': 'Stock Quantity', 'medicine_name': 'Medicine'}, color='stock_quantity', color_continuous_scale='Blues', height=500)
            st.plotly_chart(fig, use_container_width=True)
        elif report_type == "Low Stock Report":
            low_stock_items = get_low_stock_items()
            if not low_stock_items.empty:
                st.warning(f"{len(low_stock_items)} items are below minimum stock level.")
                st.dataframe(low_stock_items[['medicine_name', 'stock_quantity', 'minimum_stock_level']], use_container_width=True)
            else:
                st.success("All items are above minimum stock levels!")
        elif report_type == "Expiry Report":
            days_ahead = st.slider("Show items expiring within (days):", 1, 365, 30)
            expiring_items = get_expiring_items(days_ahead)
            if not expiring_items.empty:
                display_expiry = expiring_items[['medicine_name', 'stock_quantity', 'expiration_date']].copy()
                display_expiry['days_to_expiry'] = (display_expiry['expiration_date'] - pd.Timestamp.now()).dt.days
                st.dataframe(display_expiry.sort_values('days_to_expiry'), use_container_width=True)
            else:
                st.success(f"No items expiring within {days_ahead} days!")
        elif report_type == "Stock Value Report":
            df_with_value = st.session_state.inventory_df.copy()
            df_with_value['total_value'] = df_with_value['stock_quantity'] * df_with_value['unit_price']
            st.dataframe(df_with_value[['medicine_name', 'stock_quantity', 'unit_price', 'total_value']].sort_values('total_value', ascending=False), use_container_width=True)
        elif report_type == "Transaction History":
            st.dataframe(st.session_state.transaction_history.sort_values('timestamp', ascending=False), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- ALERTS & MONITORING ---
    
    st.markdown('<div class="section-title">Alerts & Monitoring</div>', unsafe_allow_html=True)
    if not st.session_state.inventory_df.empty:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Low Stock Alerts")
            low_stock_items = get_low_stock_items()
            if not low_stock_items.empty:
                st.markdown(f'<div class="alert-error"><strong>Warning:</strong> {len(low_stock_items)} items are below minimum!</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert-success"><strong>Good:</strong> All items are above minimum levels!</div>', unsafe_allow_html=True)
        with col2:
            st.subheader("Expiry Alerts")
            expiring_7, expiring_30 = get_expiring_items(7), get_expiring_items(30)
            if not expiring_7.empty:
                st.markdown(f'<div class="alert-error"><strong>Critical:</strong> {len(expiring_7)} items expire within 7 days!</div>', unsafe_allow_html=True)
            elif not expiring_30.empty:
                st.markdown(f'<div class="alert-warning"><strong>Warning:</strong> {len(expiring_30)} items expire within 30 days.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert-success"><strong>Good:</strong> No items expiring soon.</div>', unsafe_allow_html=True)
        with col3:
            st.subheader("Procurement Recommendations")
            low_stock = get_low_stock_items()
            if not low_stock.empty:
                for _, item in low_stock.iterrows():
                    recommended_order = max(item['minimum_stock_level'] * 2 - item['stock_quantity'], item['minimum_stock_level'])
                    st.write(f"• **{item['medicine_name']}**: Order ~{int(recommended_order)} units")
            else:
                st.markdown('<div class="alert-success"><strong>Good:</strong> No immediate procurement needed!</div>', unsafe_allow_html=True)
    else:
        st.info("No inventory data for monitoring.")
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()