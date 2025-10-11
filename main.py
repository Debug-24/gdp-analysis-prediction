
"""
GDP Predictor Tool — UI Shell (Streamlit)
Description: Layout-only shell with sidebar inputs (country dropdown, timeline slider),
metric cards, and tabs. Only included DEMO data.
"""
import streamlit as st
from datetime import date
import pandas as pd
import numpy as np
from pathlib import Path
import altair as alt




# ---------- Page config ----------
st.set_page_config(
    page_title="GDP Predictor Tool",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- File locations ----------

MULTI_PATH = Path ("gdp_dataset_for_ml.csv")
USONLY_PATH = Path ("us_only_data.csv")
USPRED_PATH = Path("us_gdp_predictors.csv")

# ---------- Load Data ----------

# You can replace - included for demo/layout
@st.cache_data(show_spinner=False)
def load_dataset(which: str) -> pd.DataFrame:
    if which == "multi":
        df = pd.read_csv(MULTI_PATH)
    else:
        df = pd.read_csv(USONLY_PATH)

    # Parse quarter column
    q = df["quarter"].astype(str)
    try:
        dt = pd.to_datetime(q, errors="raise")
    except Exception:
        period = pd.PeriodIndex(q, freq="Q")
        dt = period.to_timestamp(how="end")

    # Add year column
    df = df.assign(year=dt.dt.year)

    # Clean up types
    df["country"] = df["country"].astype(str)
    df["gross_domestic_product"] = pd.to_numeric(df["gross_domestic_product"], errors="coerce")

    # Drop bad rows
    df = df.dropna(subset=["country", "year", "gross_domestic_product"])
    return df

# ---------- Helper functions  ----------
def get_country_options():

    return [
        "United States","Canada", "Spain", "South Korea","Italy","Turkey","India","Chile","Australia",
        "Colombia","Hungary","France","Sweden","United Kingdom","Poland", "Germany", "Mexico", "Israel"
    ]

# Demo to keep charts working
def make_demo_df(start_year: int, end_year: int, seed: int = 7) -> pd.DataFrame:
    years = list(range(start_year, end_year + 1))
    rng = np.random.default_rng(seed)
    base = np.linspace(1000, 2000, len(years)) + rng.normal(0, 50, len(years)).cumsum()
    return pd.DataFrame({"Year": years, "GDP (billions, demo)": np.round(base, 2)})

# ---------- Sidebar : Data & Controls ----------
# Dataset switcher -  Optional (?)
with st.sidebar:
    st.header("Data Source")
    dataset = st.radio(
        "Choose dataset",
        [
            "Multi-country GDP (gdp_dataset_for_ml.csv)",
            "US-only GDP (us_only_data.csv)",
            "US GDP + Predictors (us_gdp_predictors.csv)"
        ],
        index=0
    )

    st.header("Filters")
    if dataset.startswith("Multi-country"):
        country = st.selectbox("Country", [
            "United States","Canada","Spain","South Korea","Italy","Turkey","India","Chile","Australia",
            "Colombia","Hungary","France","Sweden","United Kingdom","Poland","Germany","Mexico","Israel"
        ], index=0)
    else:
        country = "United States"
        st.text_input("Country", value=country, disabled=True)

    year_min, year_max_default = 1990, date.today().year
    years = st.slider("Timeline (Years)", min_value=year_min, max_value=year_max_default, value=(2010, year_max_default))

# Optional - can be removed
    st.header("Transformations")
    norm_type = st.selectbox("Normalize", ["None", "Per capita (needs population)", "Real GDP (deflate)"], index=0)
    calc_growth = st.checkbox("Show YoY growth %", value=False)

    st.header("Model")
    model_choice = st.selectbox(
        "Model",
        ["— Select —","Forecasting Model 1","Forecasting Model 2","Forecasting Model 3"],
        index=0
    )


    if dataset.startswith("US GDP + Predictors"):
        use_predictors = st.checkbox("Use predictor variables (X)", value=True)
        selected_predictors = st.multiselect(
            "Select predictors",
            ["unemployment_rate", "cpi_or_inflation", "interest_rate"],
            default=["unemployment_rate", "cpi_or_inflation"]
        )

    run = st.button("Run Prediction")

# ---------- Header ----------
st.title("GDP Predictor Tool")


# ---------- KPI cards ----------
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Selected Country", country)
with c2:
    st.metric("Years", f"{years[0]}–{years[1]}")
with c3:
    st.metric("Model", model_choice if model_choice != "— Select —" else "TBD")

st.divider()

# ---------- Tabs ----------
tab_overview, tab_model, tab_data = st.tabs(["Overview", "Model Output", "Data"])

# Overview Tab
with tab_overview:
    st.subheader("Trend Preview")
    demo = make_demo_df(*years)

    # Casting Year to string so Streamlit doesn’t add commas
    demo["Year"] = demo["Year"].astype(str)

    chart = alt.Chart(demo).mark_line(color="#174734", strokeWidth=3).encode(
        x=alt.X("Year", title="Year"),
        y=alt.Y("GDP (billions, demo)", title="GDP (Billions)")
    )

    st.altair_chart(chart, use_container_width=True)
    st.warning("This is demo data for layout only. Replace with real GDP series from your datasource.")

# Model Output Tab
with tab_model:
    st.subheader("Predictions")
    if run and model_choice != "— Select —":
        st.success(f"Ran {model_choice} for {country} on {years[0]}–{years[1]} (demo).")
        #TODO: replace with real prediction results/plots
        pred_years = list(range(years[1] - 4, years[1] + 1))
        pred_df = pd.DataFrame({
            "Year": [str(y) for y in pred_years],  # convert to strings
            "Predicted GDP (demo)": np.linspace(1800, 2100, 5)
        })
        bar_chart = alt.Chart(pred_df).mark_bar(color="#e87503").encode(
            x=alt.X("Year", title="Year"),
            y=alt.Y("Predicted GDP (demo)", title="Predicted GDP (Billions)")
        )

        st.altair_chart(bar_chart, use_container_width=True)

    else:
        st.warning("Click **Run Prediction** after selecting a model to see results.")

# Data Tab
with tab_data:
    st.subheader("Data Table (Demo)")
    st.dataframe(demo, use_container_width=True)
    st.info("Replace this table with your datasource ")

# ---------- Footer ----------
st.divider()
st.markdown(
    '''
    <small>
    <b>Notes / TODO:</b><br>
    • Add caching with <code>@st.cache_data</code> for data retrieval<br>
    </small>
    ''',
    unsafe_allow_html=True
)
