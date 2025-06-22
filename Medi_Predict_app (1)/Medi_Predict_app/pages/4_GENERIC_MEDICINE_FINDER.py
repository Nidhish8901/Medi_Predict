import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime
import plotly.express as px

# --- Page Configuration ---
# Using a base64 encoded SVG for a professional and clean icon
CAPSULE_ICON_SVG = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZGhtPSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiMwYTIzNDIiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48cGF0aCBkPSJNMTEuODYgMy4yMmE4LjA0IDguMDQgMCAwIDAtOC4wNCA4LjA0IDguMDQgOC4wNCAwIDAgMCA4LjA0IDguMDRoMi4yOGE4LjA0IDguMDQgMCAwIDAgOC4wNC04LjA0IDguMDQgOC4wNCAwIDAgMC04LjA0LTguMDRaTTE1LjE0IDEyLjloLTYuMjgiLz48L3N2Zz4="

st.set_page_config(
    page_title="Generic Medicine Finder",
    page_icon=CAPSULE_ICON_SVG,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS for Professional Styling (Strictly Whites & Blues) ---
st.markdown("""
<style>
    /* Main app background and font */
    .stApp {
        background-color: #F5F8FA; /* Cool, Light Blue-White */
        font-family: 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif;
    }

    /* Main Header */
    .main-header {
        font-size: 2.75rem;
        color: #0A2342; /* Dark Navy Blue */
        text-align: center;
        font-weight: 600;
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    
    /* Sub-header description under the main title */
    .sub-header-description {
        font-size: 1.1rem;
        color: #5A789A; /* Slate Blue-Gray */
        text-align: center;
        margin-bottom: 2.5rem;
    }
    
    /* Custom styling for Streamlit's subheaders (h2) */
    h2 {
        color: #0A2342; /* Dark Navy Blue */
        border-bottom: 2px solid #DDE5ED; /* Light Blue-Gray */
        padding-bottom: 10px;
    }
    
    /* Custom styling for Streamlit's subheaders (h3) */
    h3 {
        color: #3B5998; /* Softer Corporate Blue */
        font-weight: 600;
    }

    /* Custom box for displaying neutral AI-generated info */
    .info-box {
        background-color: #EBF4FF; /* Very Light Blue */
        padding: 1.25rem;
        border-radius: 8px;
        border-left: 5px solid #6495ED; /* Cornflower Blue */
        margin: 1rem 0;
        font-size: 0.95rem;
        color: #1E3A5F;
    }

    /* Custom box for warnings/precautions, still using blue */
    .warning-box {
        background-color: #E4EDF7; /* Slightly darker light blue */
        padding: 1.25rem;
        border-radius: 8px;
        border-left: 5px solid #4169E1; /* Royal Blue */
        margin: 1rem 0;
        color: #153265;
    }
    
    /* Custom style for brand name alternatives */
    .brand-alternative {
        background-color: #F0F4F8; /* Lightest Blue-Gray */
        padding: 0.5rem 1rem;
        margin: 0.25rem 0;
        border-radius: 5px;
        border-left: 3px solid #B0C4DE; /* Light Steel Blue */
        color: #283D5B;
    }

</style>
""", unsafe_allow_html=True)


# --- Session State Initialization ---
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'favorites' not in st.session_state:
    st.session_state.favorites = []
if 'search_query' not in st.session_state:
    st.session_state.search_query = ""


# --- Model Loading and Caching ---
@st.cache_resource
def load_biogpt_model():
    """Load and cache the BioGPT model and tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
        model = AutoModelForCausalLM.from_pretrained("microsoft/biogpt")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def generate_medical_info(prompt, tokenizer, model, max_new_tokens_count=200):
    """
    Generate text using the BioGPT model with a robust method for extracting the response.
    """
    try:
        # Encode the prompt.
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            # Generate output using max_new_tokens to control response length
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens_count, 
                do_sample=True, 
                temperature=0.6, 
                top_p=0.95
            )
        
        # Decode the entire output sequence
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Robustly find the start of the answer. 
        # The answer is whatever comes after the original prompt.
        prompt_length = len(prompt)
        answer = full_text[prompt_length:].strip()
        
        # If the answer is empty, it means the model only repeated the prompt.
        if not answer:
            return "The AI model could not generate a specific answer for this query. Please try rephrasing."
            
        return answer
        
    except Exception as e:
        st.error(f"An error occurred during AI generation: {e}")
        return "Could not generate AI insights due to an error."


# --- Data Loading ---
@st.cache_data
def load_medicine_database():
    """Load and cache a sample medicine database."""
    return {
        "Paracetamol": {"generic_name": "Acetaminophen", "brand_names": ["Tylenol", "Panadol", "Calpol"], "category": "Analgesic/Antipyretic", "dosage_forms": ["Tablet", "Syrup", "IV"], "common_dosages": ["500mg", "650mg", "1000mg"], "price_range": "$2 - $8"},
        "Ibuprofen": {"generic_name": "Ibuprofen", "brand_names": ["Advil", "Motrin", "Brufen"], "category": "NSAID", "dosage_forms": ["Tablet", "Capsule", "Syrup"], "common_dosages": ["200mg", "400mg", "600mg"], "price_range": "$3 - $12"},
        "Aspirin": {"generic_name": "Acetylsalicylic Acid", "brand_names": ["Bayer", "Disprin", "Ecosprin"], "category": "Antiplatelet/NSAID", "dosage_forms": ["Tablet", "Dispersible Tablet"], "common_dosages": ["75mg", "150mg", "325mg"], "price_range": "$1 - $5"},
        "Metformin": {"generic_name": "Metformin HCl", "brand_names": ["Glucophage", "Glycomet", "Diabex"], "category": "Antidiabetic", "dosage_forms": ["Tablet", "Extended Release"], "common_dosages": ["500mg", "850mg", "1000mg"], "price_range": "$5 - $15"},
        "Omeprazole": {"generic_name": "Omeprazole", "brand_names": ["Prilosec", "Losec", "Omez"], "category": "Proton Pump Inhibitor", "dosage_forms": ["Capsule", "Tablet", "IV"], "common_dosages": ["20mg", "40mg"], "price_range": "$8 - $25"},
    }


# --- UI Page Functions ---

def medicine_search_page():
    st.header("Medicine Search & Information")
    
    # Search input
    medicine_name = st.text_input(
        "Enter a medicine name to search:",
        placeholder="e.g., Paracetamol, Tylenol, Ibuprofen...",
        key="search_input"
    )
    
    st.subheader("Quick Search")
    quick_medicines = ["Paracetamol", "Ibuprofen", "Aspirin", "Metformin", "Omeprazole"]
    cols = st.columns(len(quick_medicines))
    for idx, med in enumerate(quick_medicines):
        if cols[idx].button(med, key=f"quick_{med}", use_container_width=True):
            medicine_name = med
            # Directly trigger the search logic
            st.session_state.search_query = med
            st.rerun()

    # Use session state to persist search across reruns from button clicks
    if medicine_name:
        st.session_state.search_query = medicine_name

    if st.session_state.search_query:
        query = st.session_state.search_query
        
        # Add to search history
        if not any(d['medicine'] == query for d in st.session_state.search_history):
            st.session_state.search_history.append({
                "medicine": query,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

        medicines_db = load_medicine_database()
        medicine_info = next((v for k, v in medicines_db.items() if query.lower() in k.lower() or query.lower() in [b.lower() for b in v["brand_names"]]), None)

        if medicine_info:
            display_medicine_info(query, medicine_info)
        else:
            st.warning(f"Medicine '{query}' not found in our database. Using AI to generate information.")
            if st.session_state.model_loaded:
                generate_ai_medicine_info(query)
            else:
                st.error("Please load the AI model from the top of the page to search for unlisted medicines.")
    
    # Clear the query after displaying results to allow for a new search
    st.session_state.search_query = ""


def display_medicine_info(medicine_name, medicine_info):
    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(f"Information for {medicine_info['generic_name']}")
        with col2:
            if st.button("Add to Favorites", use_container_width=True):
                if not any(fav['name'] == medicine_name for fav in st.session_state.favorites):
                    st.session_state.favorites.append({"name": medicine_name, "info": medicine_info, "added_date": datetime.now().strftime("%Y-%m-%d")})
                    st.success(f"{medicine_name} added to favorites!")
                else:
                    st.info(f"{medicine_name} is already in your favorites.")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Basic Information**")
            st.write(f"**Generic Name:** {medicine_info['generic_name']}")
            st.write(f"**Category:** {medicine_info['category']}")
            st.write(f"**Available Forms:** {', '.join(medicine_info['dosage_forms'])}")
            st.write(f"**Common Dosages:** {', '.join(medicine_info['common_dosages'])}")
            st.write(f"**Estimated Price Range:** {medicine_info['price_range']}")
        with col2:
            st.markdown("**Common Brand Names**")
            for brand in medicine_info['brand_names']:
                st.markdown(f'<div class="brand-alternative">{brand}</div>', unsafe_allow_html=True)
    
    if st.session_state.model_loaded:
        st.subheader("AI-Generated Medical Insights")
        t1, t2, t3, t4 = st.tabs(["Summary", "Side Effects", "Uses", "Precautions"])
        with t1, st.spinner("Generating summary..."):
            prompt = f"A medical summary of {medicine_info['generic_name']} is: "
            summary = generate_medical_info(prompt, st.session_state.tokenizer, st.session_state.model)
            st.markdown(f'<div class="info-box">{summary}</div>', unsafe_allow_html=True)
        with t2, st.spinner("Generating side effects..."):
            prompt = f"The common and serious side effects of {medicine_info['generic_name']} are: "
            side_effects = generate_medical_info(prompt, st.session_state.tokenizer, st.session_state.model)
            st.markdown(f'<div class="warning-box">{side_effects}</div>', unsafe_allow_html=True)
        with t3, st.spinner("Generating uses..."):
            prompt = f"The medical uses and indications for {medicine_info['generic_name']} include: "
            uses = generate_medical_info(prompt, st.session_state.tokenizer, st.session_state.model)
            st.markdown(f'<div class="info-box">{uses}</div>', unsafe_allow_html=True)
        with t4, st.spinner("Generating precautions..."):
            prompt = f"Important precautions and contraindications for {medicine_info['generic_name']} are: "
            precautions = generate_medical_info(prompt, st.session_state.tokenizer, st.session_state.model)
            st.markdown(f'<div class="warning-box">{precautions}</div>', unsafe_allow_html=True)


def generate_ai_medicine_info(medicine_name):
    with st.container(border=True):
        st.subheader(f"AI-Generated Information for {medicine_name}")
        t1, t2, t3 = st.tabs(["General Info", "Side Effects", "Precautions"])
        with t1, st.spinner("Generating general information..."):
            prompt = f"Comprehensive medical information about {medicine_name}, including its uses, mechanism of action, and category, is as follows: "
            info = generate_medical_info(prompt, st.session_state.tokenizer, st.session_state.model, 300)
            st.markdown(f'<div class="info-box">{info}</div>', unsafe_allow_html=True)
        with t2, st.spinner("Generating side effects..."):
            prompt = f"The common and serious side effects of {medicine_name} are: "
            side_effects = generate_medical_info(prompt, st.session_state.tokenizer, st.session_state.model)
            st.markdown(f'<div class="warning-box">{side_effects}</div>', unsafe_allow_html=True)
        with t3, st.spinner("Generating precautions..."):
            prompt = f"Important precautions and contraindications for {medicine_name} are: "
            precautions = generate_medical_info(prompt, st.session_state.tokenizer, st.session_state.model)
            st.markdown(f'<div class="warning-box">{precautions}</div>', unsafe_allow_html=True)


def drug_interactions_page():
    st.header("Drug Interaction Checker")
    st.info("Enter two or more medicines to check for potential interactions. This feature is powered by AI.")
    
    with st.container(border=True):
        col1, col2 = st.columns(2)
        drug1 = col1.text_input("First Medicine", placeholder="e.g., Warfarin")
        drug2 = col2.text_input("Second Medicine", placeholder="e.g., Aspirin")
        additional_drugs = st.text_area("Additional Medicines (comma-separated)", placeholder="e.g., Metformin, Lisinopril")

        if st.button("Check Interactions", type="primary"):
            if drug1 and drug2:
                drugs_list = [drug1, drug2] + [d.strip() for d in additional_drugs.split(',') if d.strip()]
                
                st.subheader("Medicines Being Checked:")
                st.write(", ".join(drugs_list))
                
                if st.session_state.model_loaded:
                    with st.spinner("Analyzing potential interactions..."):
                        prompt = f"An analysis of potential drug interactions between {', '.join(drugs_list)}, including severity levels and clinical significance, is as follows: "
                        interactions = generate_medical_info(prompt, st.session_state.tokenizer, st.session_state.model, 400)
                        st.subheader("Interaction Analysis Result:")
                        st.markdown(f'<div class="warning-box">{interactions}</div>', unsafe_allow_html=True)
                        st.caption("Disclaimer: This is AI-generated information. Always consult a healthcare professional for clinical decisions.")
                else:
                    st.error("Please load the AI model from the top of the page to check interactions.")
            else:
                st.warning("Please enter at least two medicines to check for interactions.")


def price_comparison_page():
    st.header("Medicine Price Comparison")
    st.info("This is a sample comparison. Prices are illustrative and vary by location and pharmacy.")

    price_data = {
        "Medicine": ["Paracetamol 500mg", "Ibuprofen 400mg", "Aspirin 75mg", "Metformin 500mg", "Omeprazole 20mg"],
        "Brand Price ($)": [12, 25, 8, 45, 60],
        "Generic Price ($)": [3, 8, 2, 15, 25],
    }
    df = pd.DataFrame(price_data)
    df["Potential Savings ($)"] = df["Brand Price ($)"] - df["Generic Price ($)"]

    with st.container(border=True):
        st.subheader("Brand vs. Generic Price Chart")
        fig = px.bar(
            df, x="Medicine", y=["Brand Price ($)", "Generic Price ($)"],
            title="Brand vs. Generic Medicine Prices",
            barmode="group",
            color_discrete_map={"Brand Price ($)": "#0A2342", "Generic Price ($)": "#4169E1"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with st.container(border=True):
        st.subheader("Potential Savings with Generic Medicines")
        fig2 = px.bar(
            df, x="Medicine", y="Potential Savings ($)",
            title="Potential Savings per Medicine",
            color="Potential Savings ($)",
            color_continuous_scale=px.colors.sequential.Blues
        )
        st.plotly_chart(fig2, use_container_width=True)


def search_history_page():
    st.header("Search History")
    if st.session_state.search_history:
        with st.container(border=True):
            st.subheader("Your Recent Searches")
            if st.button("Clear History"):
                st.session_state.search_history = []
                st.success("Search history cleared!")
                st.rerun()

            for search in reversed(st.session_state.search_history[-20:]):
                col1, col2 = st.columns([4, 1])
                col1.text(f"Medicine: {search['medicine']} (Searched on: {search['timestamp']})")
        
        st.subheader("Search Analytics")
        with st.container(border=True):
            medicines = [search['medicine'] for search in st.session_state.search_history]
            medicine_counts = pd.Series(medicines).value_counts().nlargest(10)
            fig = px.bar(
                medicine_counts, y=medicine_counts.index, x=medicine_counts.values,
                orientation='h', title="Top 10 Searched Medicines",
                labels={'y': 'Medicine', 'x': 'Search Count'},
                color=medicine_counts.values, color_continuous_scale=px.colors.sequential.Blues_r
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No search history found. Start searching to build your history.")


def favorites_page():
    st.header("Favorite Medicines")
    if st.session_state.favorites:
        st.subheader("Your Saved Medicines")
        for i, favorite in enumerate(st.session_state.favorites):
            with st.expander(f"{favorite['name']} (Added: {favorite['added_date']})"):
                info = favorite['info']
                st.write(f"**Generic Name:** {info.get('generic_name', 'N/A')}")
                st.write(f"**Category:** {info.get('category', 'N/A')}")
                st.write(f"**Price Range:** {info.get('price_range', 'N/A')}")
                st.write(f"**Brand Names:** {', '.join(info.get('brand_names', []))}")
                if st.button("Remove", key=f"remove_fav_{i}", type="secondary"):
                    st.session_state.favorites.pop(i)
                    st.success("Removed from favorites!")
                    st.rerun()
        
        # Export favorites
        if st.download_button(
            label="Export Favorites to CSV",
            data=pd.DataFrame([
                {"Medicine": f["name"], "Generic_Name": f["info"].get("generic_name"), "Date_Added": f["added_date"]}
                for f in st.session_state.favorites
            ]).to_csv(index=False).encode('utf-8'),
            file_name="favorite_medicines.csv",
            mime="text/csv",
            type="primary"
        ):
            st.success("Export successful!")
    else:
        st.info("You haven't added any favorite medicines yet.")


# --- Main Application Logic ---
def main():
    st.markdown('<h1 class="main-header">Generic Medicine Finder</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header-description">Your intelligent guide to understanding medications, interactions, and costs.</p>', unsafe_allow_html=True)
    
    # --- AI Model Loading Section ---
    with st.container(border=True):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("AI Model Status")
            if not st.session_state.model_loaded:
                st.warning("The AI model is not loaded. Advanced features like interaction checking and searching for unlisted drugs are disabled.")
            else:
                st.success("The BioGPT AI model is loaded and ready.")
        with col2:
            if not st.session_state.model_loaded:
                if st.button("Load AI Model", type="primary", use_container_width=True):
                    with st.spinner("Loading BioGPT model... This may take a moment."):
                        tokenizer, model = load_biogpt_model()
                        if tokenizer and model:
                            st.session_state.tokenizer = tokenizer
                            st.session_state.model = model
                            st.session_state.model_loaded = True
                            st.success("Model loaded successfully!")
                            st.rerun()
    
    st.divider()

    # --- Main Navigation Tabs ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Medicine Search", 
        "Drug Interactions", 
        "Price Comparison", 
        "Search History", 
        "Favorites"
    ])

    with tab1:
        medicine_search_page()
    with tab2:
        drug_interactions_page()
    with tab3:
        price_comparison_page()
    with tab4:
        search_history_page()
    with tab5:
        favorites_page()

if __name__ == "__main__":
    main()