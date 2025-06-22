"""
📍 Pharmacy Locator – GPS, PIN, Area, or City lookup
----------------------------------------------------
• Chain / Local filter
• Download results (CSV or PDF)
"""

import math, re, io, sys
import streamlit as st
import pandas as pd
from streamlit_geolocation import streamlit_geolocation

# ───────────────────────────────────────────────────────────── Helpers
def haversine(lat1, lon1, lat2, lon2):
    R = 6_371.0  # km
    φ1, φ2 = map(math.radians, [lat1, lat2])
    dφ, dλ = map(math.radians, [lat2 - lat1, lon2 - lon1])
    a = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

@st.cache_data(show_spinner=False)
def load_db(path: str = "Pharmacies.csv") -> pd.DataFrame:
    """Load CSV and normalise column names → name | address | pin | lat | lon."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(
        columns={
            "pharmacy name": "name",
            "address": "address",
            "pincode":  "pin",
            "latitude":  "lat",
            "longitude": "lon",
        }
    )
    df["pin"] = df["pin"].astype(str).str.zfill(6)

    # ── tag chain vs local ────────────────────────────────────────────
    if "is_chain" not in df.columns:
        chain_keywords = ("apollo", "medplus", "pharmeasy", "1mg", "netmeds",
                          "wellness", "anjaney", "dvs", "guardian")
        df["is_chain"] = df["name"].str.lower().apply(
            lambda x: any(kw in x for kw in chain_keywords)
        )
    else:
        # normalise to boolean
        df["is_chain"] = df["is_chain"].astype(bool, errors="ignore")

    return df

def guess_city(address: str) -> str:
    m = re.search(r",\s*([^,]+)\s+\d{6}$", address)
    if m:
        return m.group(1).strip()
    parts = [p.strip() for p in address.split(",") if p.strip()]
    return parts[-1] if parts else ""

@st.cache_data(show_spinner=False)
def unique_cities(df: pd.DataFrame) -> list[str]:
    cities = df["address"].apply(guess_city).str.title().unique()
    return sorted(filter(None, cities))

@st.cache_data(show_spinner=False)
def build_pin_centers(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    return (
        df.groupby("pin")[["lat", "lon"]]
          .mean()
          .apply(tuple, axis=1)
          .to_dict()
    )

def dataframe_to_pdf_bytes(df: pd.DataFrame) -> bytes | None:
    """Return a PDF (as bytes) of the dataframe, or None if ReportLab unavailable."""
    try:
        from reportlab.lib.pagesizes import letter, landscape
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
    except ImportError:
        return None

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter), leftMargin=30,
                            rightMargin=30, topMargin=30, bottomMargin=30)
    data = [df.columns.tolist()] + df.astype(str).values.tolist()
    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))
    doc.build([table])
    buffer.seek(0)
    return buffer.read()

# ───────────────────────────────────────────────────────────────  UI
st.set_page_config("Pharmacy Locator", "📍", layout="wide")
st.title("📍 Pharmacy Locator")

df           = load_db()
all_cities   = unique_cities(df)
pin_centers  = build_pin_centers(df)

# ------------------------------- Sidebar – search + filters
with st.sidebar:
    st.header("Choose your location")

    pin  = st.text_input("Enter 6-digit PIN code").strip()
    area = st.text_input("…or type an area / locality").strip()
    city = st.text_input("…or start typing a city").strip()

    # non-clickable suggestions
    if 1 <= len(city) < 50:
        hints = [
            c for c in all_cities
            if c.lower().startswith(city.lower()) and c.lower() != city.lower()
        ][:5]
        if hints:
            st.markdown("**Did you mean:**")
            for h in hints:
                st.markdown(f"- {h}")

    # chain / local filter  ---------------------------
    type_choice = st.radio(
        "Pharmacy type",
        ("All", "Chain only", "Local only"),
        horizontal=True,
    )

    st.markdown("**— or —**")
    loc = streamlit_geolocation()
    if loc and loc.get("latitude"):
        user_lat, user_lon = loc["latitude"], loc["longitude"]
        st.success(f"GPS acquired: {user_lat:.4f}, {user_lon:.4f}")
    else:
        user_lat = user_lon = None

    radius_km = st.slider("Search radius (km)", 1, 20, 5)

# -------------------------------------------------- 1) City filter
if city:
    rows = df[df["address"].str.contains(city, case=False, na=False)]
    if rows.empty:
        st.error(f"No pharmacies found in “{city}”.")
        st.stop()

    st.success(f"{len(rows)} pharmacies found in “{city}”.")
    # apply chain/local filter
    if type_choice == "Chain only":
        rows = rows[rows["is_chain"]]
    elif type_choice == "Local only":
        rows = rows[~rows["is_chain"]]

    if rows.empty:
        st.warning("No pharmacies match that type filter in this city.")
    else:
        st.dataframe(rows.drop(columns=["lat", "lon"]),
                     use_container_width=True)
        st.map(rows.rename(columns={"lat": "latitude", "lon": "longitude"}))

        # download block
        fmt = st.selectbox("Download results as", ("CSV", "PDF"))
        if fmt == "CSV":
            csv_bytes = rows.to_csv(index=False).encode()
            st.download_button("🔽 Download CSV",
                               csv_bytes,
                               file_name="pharmacies.csv",
                               mime="text/csv")
        else:
            pdf_bytes = dataframe_to_pdf_bytes(rows)
            if pdf_bytes is None:
                st.info("ReportLab not available – install it with "
                        "`pip install reportlab` for PDF export.")
            else:
                st.download_button("🔽 Download PDF",
                                   pdf_bytes,
                                   file_name="pharmacies.pdf",
                                   mime="application/pdf")
    st.stop()

# -------------------------------------------------- 2) GPS → PIN → Area chain
if user_lat is None and user_lon is None:
    if pin and pin in pin_centers:
        user_lat, user_lon = pin_centers[pin]
        st.success(f"Using centroid of PIN {pin}: {user_lat:.4f}, {user_lon:.4f}")
    elif area:
        area_rows = df[df["address"].str.contains(area, case=False, na=False)]
        if not area_rows.empty:
            user_lat, user_lon = area_rows[["lat", "lon"]].mean()
            st.success(f"Using centroid of “{area.title()}”: "
                       f"{user_lat:.4f}, {user_lon:.4f}")
        else:
            st.warning("That area/locality isn’t in the database.")
    else:
        st.info("Enter a PIN, area, city or enable GPS to begin.")

# -------------------------------------------------- 3) Radius search around point
if user_lat is not None and user_lon is not None:
    df["distance_km"] = df.apply(
        lambda r: haversine(user_lat, user_lon, r["lat"], r["lon"]), axis=1
    )
    rows = df[df["distance_km"] <= radius_km]

    # apply chain/local filter
    if type_choice == "Chain only":
        rows = rows[rows["is_chain"]]
    elif type_choice == "Local only":
        rows = rows[~rows["is_chain"]]

    rows = rows.sort_values("distance_km").reset_index(drop=True)

    st.write(f"### {len(rows)} pharmacies within {radius_km} km")
    if rows.empty:
        st.warning("No pharmacies match that type filter in this radius.")
    else:
        st.dataframe(
            rows.drop(columns=["lat", "lon"]).round({"distance_km": 2}),
            use_container_width=True,
        )
        st.map(rows.rename(columns={"lat": "latitude", "lon": "longitude"}))

        # download block
        fmt = st.selectbox("Download results as", ("CSV", "PDF"), key="dl2")
        if fmt == "CSV":
            csv_bytes = rows.to_csv(index=False).encode()
            st.download_button("🔽 Download CSV",
                               csv_bytes,
                               file_name="pharmacies.csv",
                               mime="text/csv",
                               key="csv2")
        else:
            pdf_bytes = dataframe_to_pdf_bytes(rows)
            if pdf_bytes is None:
                st.info("ReportLab not available – install it with "
                        "`pip install reportlab` for PDF export.")
            else:
                st.download_button("🔽 Download PDF",
                                   pdf_bytes,
                                   file_name="pharmacies.pdf",
                                   mime="application/pdf",
                                   key="pdf2")
