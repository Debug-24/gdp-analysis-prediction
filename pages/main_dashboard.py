
"""
GDP - Dashboard
"""
import streamlit as st
from datetime import date
import pandas as pd
import numpy as np
from pathlib import Path
from data_utils import filter_data
import plotly.graph_objects as go
import altair as alt
import glob
import re
import sys

# ---------- Page config ----------
st.set_page_config(
    page_title="GDP Predictor Tool",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Custom CSS Styling ----
st.markdown("""
<style>
/* Metric styling */
div[data-testid="stMetricValue"] {
    font-size: 24px !important;
    font-weight: 700;
    color: #174734;
}
div[data-testid="stMetricLabel"] {
    font-size: 12px !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #6b7280;
    font-weight: 500;
}

/* Button styling */
.stButton>button {
    background: linear-gradient(90deg, #174734 0%, #2d5a47 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1.5rem;
    font-weight: 600;
    font-size: 14px;
    transition: all 0.3s ease;
    box-shadow: 0 2px 4px rgba(23, 71, 52, 0.2);
    width: 100%;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #2d5a47 0%, #174734 100%);
    box-shadow: 0 4px 8px rgba(23, 71, 52, 0.3);
    transform: translateY(-1px);
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #e8f4f0 0%, #d0e7df 100%);
    color: #2d1810;
}

section[data-testid="stSidebar"] .element-container {
    padding: 0.5rem 0;
}

section[data-testid="stSidebar"] h3 {
    color: #174734;
    font-weight: 700;
    margin-top: 1.5rem;
    margin-bottom: 0.5rem;
    font-size: 1.1rem;
    border-bottom: 2px solid #8b6f47;
    padding-bottom: 0.5rem;
}

/* Form controls in sidebar with semi-transparent white background for readability */
section[data-testid="stSidebar"] .stSelectbox,
section[data-testid="stSidebar"] .stSlider,
section[data-testid="stSidebar"] .stRadio,
section[data-testid="stSidebar"] .stInfo,
section[data-testid="stSidebar"] .stSuccess,
section[data-testid="stSidebar"] .stWarning {
    background-color: rgba(255, 255, 255, 0.7);
    border-radius: 6px;
    padding: 0.5rem;
    margin-bottom: 0.5rem;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: #f9fafb;
    padding: 8px;
    border-radius: 8px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 6px;
    padding: 10px 20px;
    font-weight: 600;
    transition: all 0.3s ease;
}

.stTabs [aria-selected="true"] {
    background-color: #174734;
    color: white;
}

.stTabs [aria-selected="false"] {
    background-color: white;
    color: #374151;
}

.stTabs [aria-selected="false"]:hover {
    background-color: #e5e7eb;
}

/* Card/Container styling */
div[data-testid="stHorizontalBlock"] {
    gap: 1rem;
}

/* Info/Success/Warning boxes */
.stInfo {
    background-color: #eff6ff;
    border-left: 4px solid #3b82f6;
}

.stSuccess {
    background-color: #f0fdf4;
    border-left: 4px solid #22c55e;
}

.stWarning {
    background-color: #fffbeb;
    border-left: 4px solid #f59e0b;
}

.stError {
    background-color: #fef2f2;
    border-left: 4px solid #ef4444;
}

/* Selectbox and input styling */
.stSelectbox label, .stSlider label {
    font-weight: 600;
    color: #374151;
    font-size: 14px;
}

/* Radio button styling */
.stRadio [role="radiogroup"] {
    gap: 1rem;
    padding: 0.5rem;
    background-color: white;
    border-radius: 6px;
}

/* Divider styling */
hr {
    border: none;
    border-top: 2px solid #e5e7eb;
    margin: 1.5rem 0;
}


/* Title styling */
h1 {
    color: #174734;
    font-weight: 800;
    margin-bottom: 1rem;
}

h2 {
    color: #1f2937;
    font-weight: 700;
    margin-top: 1.5rem;
    margin-bottom: 1rem;
}

h3 {
    color: #374151;
    font-weight: 600;
    margin-top: 1rem;
    margin-bottom: 0.75rem;
}

/* Chart container */
.plotly {
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    padding: 1rem;
    background: white;
}

/* Dataframe styling */
.dataframe {
    border-radius: 8px;
    overflow: hidden;
}

/* Expander styling */
.streamlit-expanderHeader {
    font-weight: 600;
    color: #174734;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: #f1f1f1;
}

::-webkit-scrollbar-thumb {
    background: #174734;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #2d5a47;
}
</style>
""", unsafe_allow_html=True)

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# Import forecast function from your ML pipeline
from src.models.predict import generate_forecast, TRADITIONAL, ENHANCED

DATA_DIR = ROOT / "data" / "data-processed"

CLEAN_PATH = DATA_DIR / "clean_quarterly.csv"

SUMMARY_PATH = DATA_DIR / "summary_metrics.csv"


@st.cache_data(show_spinner=False)
def discover_assets(data_dir: Path):
    
    importance_files = sorted(glob.glob(str(data_dir / "importance_*.csv")))
    preds_files = sorted(glob.glob(str(data_dir / "preds_*.csv")))

    countries = set()
    models = set()

    for f in importance_files + preds_files:
        #Example filename: importance_United States_Enhanced.csv
        name = Path(f).stem
        name = name.replace("importance_", "").replace("preds_", "")
        parts = name.split("_")
        if len(parts) >= 2:
            country = "_".join(parts[:-1]).replace("_", " ")
            model = parts[-1]
            countries.add(country)
            models.add(model)

    #US comes first
    def sort_key(x):
        return (x != "United States", x)

    return sorted(countries, key=sort_key), sorted(models)

@st.cache_data(show_spinner=False)
def load_summary(path: Path) -> pd.DataFrame:
    """
    Read the summary file and normalize column names.
    If the file isn't there, return an empty dataframe.
    """
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    return df

def _try_paths(prefix: str, country: str, model: str):
    return [
        DATA_DIR / f"{prefix}_{country}_{model}.csv",
        DATA_DIR / f"{prefix}_{country.replace(' ', '_')}_{model}.csv",
    ]

@st.cache_data(show_spinner=False)
def load_full_data() -> pd.DataFrame:
    """Load the modeling-ready quarterly data the ML pipeline uses."""
    if not CLEAN_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(CLEAN_PATH, parse_dates=["quarter"])

def pick_best_config(summary_df: pd.DataFrame, country: str):
    """
    Return the best row for this country from summary_metrics.csv.
    Keys used: feature_set, winner_model. Ranking: highest R2, else lowest RMSE, else lowest MAE.
    """
    if summary_df.empty:
        return None
    df = summary_df.copy()
    df.columns = [c.lower() for c in df.columns]
    if "country" in df.columns:
        df = df[df["country"] == country]
        if df.empty:
            return None
    if "r2" in df.columns:
        df = df.sort_values("r2", ascending=False)
    elif "rmse" in df.columns:
        df = df.sort_values("rmse", ascending=True)
    elif "mae" in df.columns:
        df = df.sort_values("mae", ascending=True)
    return df.iloc[0].to_dict()

def load_country_model_df(country: str, feature_set_label: str):
    """
    Returns (country_df, features_used) with columns:
    quarter (datetime), GDP (float), and available features for the selected feature set.
    """
    df = load_full_data()
    if df.empty:
        return pd.DataFrame(), []

    cdf = df[df["country"] == country].copy()
    if cdf.empty:
        return pd.DataFrame(), []

    # Ensure datetime for quarter
    if not np.issubdtype(cdf["quarter"].dtype, np.datetime64):
        cdf["quarter"] = pd.to_datetime(cdf["quarter"], errors="coerce")

    # Choose features from pipeline lists
    base_feats = ENHANCED if feature_set_label == "Enhanced" else TRADITIONAL
    feats = [f for f in base_feats if f in cdf.columns]

    keep_cols = ["quarter", "GDP"] + feats
    cdf = cdf[keep_cols].dropna().sort_values("quarter")
    return cdf, feats

# ---- --Standardize Predictions DF (Helper) -------
def _pick_col(df, exact_names, fuzzy_keywords):
    lower_map = {c.lower(): c for c in df.columns}
    for name in exact_names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in fuzzy_keywords):
            return c
    return None

def standardize_preds_df(df):
    """
    Returns df with columns standardized to:
       t, y_true, y_pred
    """
    if df is None or df.empty:
        st.error("Predictions DataFrame is empty.")
        return pd.DataFrame(columns=["t","y_true","y_pred"])

    t_col = _pick_col(df,
                      ["quarter", "date", "period", "time", "year", "t", "ds"],
                      ["quarter", "date", "period", "time", "year"])
    y_true_col = _pick_col(df,
                           ["y_true", "y", "actual", "gdp", "target", "truth", "actuals"],
                           ["true", "actual", "target", "gdp"])
    y_pred_col = _pick_col(df,
                           ["y_pred", "yhat", "prediction", "pred", "forecast", "predicted", "gdp_pred"],
                           ["pred", "forecast", "yhat"])

    missing = []
    if t_col is None: missing.append("time axis (e.g. 'quarter'/'date')")
    if y_true_col is None: missing.append("actuals (e.g. 'y_true'/'actual')")
    if y_pred_col is None: missing.append("predictions (e.g. 'y_pred'/'yhat')")
    if missing:
        st.error(f"Missing: {', '.join(missing)}. Columns found: {list(df.columns)}")
        return pd.DataFrame(columns=["t","y_true","y_pred"])

    out = df.rename(columns={t_col: "t", y_true_col: "y_true", y_pred_col: "y_pred"}).copy()
    return out[["t", "y_true", "y_pred"]]


@st.cache_data(show_spinner=False)
def load_importance(country: str, model: str):
    for path in _try_paths("importance", country, model):
        if path.exists():
            df = pd.read_csv(path)
            df.columns = [c.lower() for c in df.columns]
            feature_col = "feature" if "feature" in df.columns else df.columns[0]
            imp_col = "importance" if "importance" in df.columns else df.columns[1]
            return df[[feature_col, imp_col]].rename(columns={feature_col: "feature", imp_col: "importance"})
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_preds(country: str, model: str):
    for path in _try_paths("preds", country, model):
        if path.exists():
            df = pd.read_csv(path)
            df.columns = [c.lower() for c in df.columns]

            # Make sure we have y_true and y_pred
            if "y_true" not in df.columns:
                for alt in ("actual", "actuals", "target", "ground_truth"):
                    if alt in df.columns:
                        df = df.rename(columns={alt: "y_true"})
                        break

            if "y_pred" not in df.columns:
                for alt in ("prediction", "pred", "forecast", "yhat"):
                    if alt in df.columns:
                        df = df.rename(columns={alt: "y_pred"})
                        break

            
            time_col = None
            for col in ("quarter", "date", "period", "time", "year_quarter", "index"):
                if col in df.columns:
                    time_col = col
                    break

            if time_col is None:
                df["index"] = range(len(df))
                time_col = "index"

            return df, time_col

    return pd.DataFrame(), None


# --------- Load custom CSS --------
def load_css(path: str = "styles.css"):
    p = Path(path)
    if p.exists():
        st.markdown(f"<style>{p.read_text()}</style>", unsafe_allow_html=True)

load_css()


# ---------- File locations ----------

# Updated paths to use ROOT variable (file is now in pages/ directory)
DATA_RAW_DIR = ROOT / "data" / "data-raw"
DATASET_PATH = DATA_RAW_DIR / "gdp_dataset_for_ml.csv"


# ---------- Load Data ----------

@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    """Load and preprocess the GDP dataset"""
    df = pd.read_csv(DATASET_PATH)

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

# ---------- Sidebar : Data & Controls ----------
with st.sidebar:
    st.header("Data Source")
    st.info("Multi-country GDP Dataset (gdp_dataset_for_ml.csv)")

    st.header("Filters")
    country = st.selectbox("Country", [
        "Australia", "Austria", "Belgium", "Bulgaria", "Canada", "Chile", "Colombia",
        "Costa Rica", "Croatia", "Czechia", "Denmark", "Estonia", "Finland", "France",
        "Germany", "Greece", "Hungary", "Iceland", "Ireland", "Israel", "Italy", "Japan",
        "Korea", "Latvia", "Lithuania", "Luxembourg", "Netherlands", "Norway", "Poland",
        "Portugal", "Slovak Republic", "Slovenia", "Spain", "Sweden", "Türkiye",
        "United Kingdom", "United States"
    ], index=0)

    year_min, year_max_default = 1990, date.today().year
    years = st.slider("Timeline (Years)",
                      min_value=year_min,
                      max_value=year_max_default,
                      value=(2010, year_max_default))

    st.header("Forecast")
    horizon = st.slider("Forecast quarters", 1, 20, 8)   

    feature_set_choice = st.selectbox(
        "Features Set to display",
        ["Auto (best)", "Traditional", "Enhanced"],
        index=0
    )

    summary_df = load_summary(SUMMARY_PATH)
    best = pick_best_config(summary_df, country)

    if best:
        feature_set_label = best.get("feature_set", "Enhanced")              # "Enhanced" or "Traditional"
        winner_model = best.get("winner_model", "RandomForest")              # "Linear" | "Ridge" | "RandomForest"
        st.success(f"Best Set by RSME: {feature_set_label}")
    else:
        st.warning("summary_metrics.csv didn't have a row for this country. Using default.")
        feature_set_label = "Enhanced"  # Default fallback
        winner_model = "RandomForest"  # Default fallback

    model_label = f"{feature_set_label} · {winner_model}"

    run = st.button("Run Prediction")


# ---------- Header ----------
st.title("GDP Predictor Tool")


# ---------- KPI cards ----------
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Selected Country", country)
with c2:
    st.metric("Years", f"{years[0]}–{years[1]}")

st.divider()

# ---------- Tabs ----------
tab_prediction, tab_features, tab_data = st.tabs(["Prediction", "Features Ranked", "Data"])

# Prediction Tab
with tab_prediction:
    st.subheader("Prediction")
    
    if not run:
        st.info("**Please select a country from the sidebar and click 'Run Prediction' to see the GDP prediction and forecast results.**")
    else:
        st.success(f"Generating prediction chart for {country} on {years[0]}–{years[1]}.")

        # --- Load summary metrics to find the winning feature set/model for this country ---
        summary_path = SUMMARY_PATH
        winner_model = None
        chosen_feature_set = None
        chosen_rmse = None
        try:
            summary_df = pd.read_csv(summary_path)
            country_rows = summary_df[summary_df["country"] == country]
            if not country_rows.empty:
                # If user chooses Auto, choose the best feature set by RMSE
                if feature_set_choice == "Auto (best)":
                    best_row = country_rows.loc[country_rows["RMSE"].idxmin()]
                    chosen_feature_set = best_row["feature_set"]
                    winner_model = best_row["winner_model"]
                    chosen_rmse = best_row["RMSE"]
                else:
                    # Use user's selection if available, otherwise attempt to load preds file for that feature set
                    fs_rows = country_rows[country_rows["feature_set"] == feature_set_choice]
                    if not fs_rows.empty:
                        row = fs_rows.iloc[0]
                        chosen_feature_set = row["feature_set"]
                        winner_model = row["winner_model"]
                        chosen_rmse = row["RMSE"]
                    else:
                        chosen_feature_set = feature_set_choice
            else:
                st.info(f"No summary metrics found for {country} in {summary_path}.")
        except FileNotFoundError:
            st.info(f"Summary metrics file not found at {summary_path}. KPI card will be empty.")

        # Load predictions CSV for the chosen feature set 
        preds_path = None
        preds_df = None
        if chosen_feature_set is not None:
            preds_path = DATA_DIR / f"preds_{country}_{chosen_feature_set}.csv"
            if preds_path.exists():
                preds_df = pd.read_csv(preds_path, parse_dates=["quarter"]) 
            else:
                for alt_fs in ["Traditional", "Enhanced"]:
                    alt_path = DATA_DIR / f"preds_{country}_{alt_fs}.csv"
                    if alt_path.exists():
                        preds_df = pd.read_csv(alt_path, parse_dates=["quarter"])
                        preds_path = alt_path
                        break

        if preds_df is None:
            st.warning(f"No prediction file found for {country}. Expected at `preds_{country}_<FeatureSet>.csv` in `{DATA_DIR}`.")
        else:
            # Ensure columns present
            if not set(["quarter", "GDP_actual", "GDP_pred"]).issubset(preds_df.columns):
                st.error(f"Predictions file {preds_path} missing required columns. Found: {list(preds_df.columns)}")
            else:
                # Filter by selected years
                start_dt = pd.to_datetime(f"{years[0]}-01-01")
                end_dt = pd.to_datetime(f"{years[1]}-12-31")
                mask = (preds_df["quarter"] >= start_dt) & (preds_df["quarter"] <= end_dt)
                plot_df = preds_df.loc[mask].copy()

                # Plotly figure with two lines
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=plot_df["quarter"],
                    y=plot_df["GDP_actual"],
                    mode="lines+markers",
                    name="GDP Actual",
                    line=dict(color="#174734")
                ))
                fig.add_trace(go.Scatter(
                    x=plot_df["quarter"],
                    y=plot_df["GDP_pred"],
                    mode="lines+markers",
                    name="GDP Predicted",
                    line=dict(color="#e87503", dash="dash")
                ))

                fig.update_layout(
                    title=f"{country}: Actual vs Predicted GDP ({chosen_feature_set if chosen_feature_set else 'unknown feature set'})",
                    xaxis_title="Date",
                    yaxis_title="GDP (Billions)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )

                st.plotly_chart(fig, use_container_width=True)

                # KPI / summary display
                st.divider()
                st.markdown("### Model Performance Comparison")

                # Reload summary using the helper for consistent column names (rmse, r2)
                metrics_df = load_summary(SUMMARY_PATH)

                if not metrics_df.empty:
                    # Filter for the current country
                    country_metrics = metrics_df[metrics_df["country"] == country].copy()

                    if not country_metrics.empty:
                        # Sort by RMSE so the best model appears first
                        if "rmse" in country_metrics.columns:
                            country_metrics = country_metrics.sort_values("rmse", ascending=True)

                        # Loop through each row (Traditional, Enhanced) and display them
                        for i, row in country_metrics.iterrows():
                            # Create 4 columns: Feature Set | Model | RMSE | R2
                            c1, c2, c3, c4 = st.columns(4)

                            c1.metric("Feature Set", row.get("feature_set", "N/A"))
                            c2.metric("Model", row.get("winner_model", "N/A"))
                            
                            # Format RMSE
                            val_rmse = row.get("rmse")
                            c3.metric("RMSE", f"{val_rmse:.2f}" if pd.notnull(val_rmse) else "N/A")

                            # Format R2
                            val_r2 = row.get("r2")
                            c4.metric("R²", f"{val_r2:.4f}" if pd.notnull(val_r2) else "N/A")
                    else:
                        st.info(f"No detailed metrics found for {country}.")
                else:
                    st.warning("Summary metrics file is empty or missing.")

        #------------Forecast Chart-----------
        st.divider()
        st.markdown("### Forecast (Using Best Config)")

        if run and chosen_feature_set and winner_model:
            try:
                # Build the modeling dataframe for the selected country + feature set
                country_df, used_feats = load_country_model_df(country, chosen_feature_set)
                if country_df.empty or "GDP" not in country_df.columns:
                    st.warning("Modeling data not available or GDP column missing. Check data/data-processed/clean_quarterly.csv.")
                else:
                    df_fore = generate_forecast(
                        model_name=winner_model,
                        country_df=country_df,
                        feature_set_label=chosen_feature_set,
                        future_quarters=horizon
                    )

                    # Merge historical + forecast for one visual
                    hist = country_df[["quarter", "GDP"]].rename(columns={"GDP": "value"})
                    hist["type"] = "Historical"

                    fore = df_fore.rename(columns={"GDP_pred": "value"})
                    fore["type"] = "Forecast"

                    viz = pd.concat([hist, fore], ignore_index=True)

                    line = (
                        alt.Chart(viz)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("quarter:T", title="Quarter"),
                            y=alt.Y("value:Q", title="GDP"),
                            color="type:N",
                            tooltip=["quarter:T", "type:N", "value:Q"]
                        )
                        .properties(height=380)
                    )
                    st.altair_chart(line, use_container_width=True)

                    with st.expander("Details", expanded=False):
                        st.write(f"Winner model: **{winner_model}** · Feature set: **{chosen_feature_set}**")
                        st.write("Features used:", used_feats)
                        st.dataframe(df_fore, use_container_width=True)

            except Exception as e:
                st.error(f"Forecast failed: {e}")
        elif run:
            st.info("Please run a prediction first to generate forecast data.")
        else:
            st.caption("Set a horizon and click **Run Prediction** to produce future quarters.")
                
    

# Features Ranked Tab
with tab_features:
    st.subheader("Feature Importance Analysis")
    
    # Control for Traditional vs Enhanced features
    features_type = st.radio(
        "Feature Set Type",
        ["Traditional", "Enhanced"],
        horizontal=True
    )
    
    # Load and process feature importance data for selected country
    importance_path = DATA_DIR / f"importance_{country}_{features_type}.csv"
    try:
        importance_df = pd.read_csv(importance_path)
        
        # Ensure we have feature and importance columns
        if not set(["feature", "importance"]).issubset(importance_df.columns):
            st.error(f"Invalid columns in importance file. Expected 'feature' and 'importance', got: {list(importance_df.columns)}")
        else:
            # Sort importance by descending order
            importance_df = importance_df.sort_values("importance", ascending=True)
            
            # Create horizontal bar chart with color coding
            fig = go.Figure()
            colors = ['#d73027' if x < 0 else '#1a9850' for x in importance_df['importance']]
            
            fig.add_trace(go.Bar(
                x=importance_df["importance"],
                y=importance_df["feature"],
                orientation="h",
                marker_color=colors,
                hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
            ))
            
            fig.update_layout(
                title=f"Feature Importance Ranking for {country} ({features_type} Features)",
                xaxis_title="Importance Score",
                yaxis_title="Feature",
                height=max(400, len(importance_df) * 30),  
                margin=dict(l=200),  
                showlegend=False,
                # color-coding
                annotations=[
                    dict(
                        text="Green = Positive Impact | Red = Negative Impact",
                        xref="paper", yref="paper",
                        x=0, y=1.05,
                        showarrow=False,
                        font=dict(size=10),
                        xanchor="left"
                    )
                ]
            )

            st.plotly_chart(fig, use_container_width=True)
            
            # Features data table
            st.subheader("Feature Importance Scores")
            st.dataframe(
                importance_df,
                column_config={
                    "feature": "Feature Name",
                    "importance": st.column_config.NumberColumn(
                        "Importance Score",
                        format="%.4f",
                    )
                },
                hide_index=True
            )
    except FileNotFoundError:
        st.warning(f"No feature importance data found for {country} with {features_type} features.")
    

# --Data Tab
with tab_data:
    st.subheader("Data Table")

    view = st.radio("Choose table", ["Predictions (Actuals + Preds)", "Summary Metrics"], horizontal=True)

    if view == "Predictions (Actuals + Preds)":
        preds_raw, _ = load_preds(country, feature_set_label)
        preds_std = standardize_preds_df(preds_raw)
        if preds_std.empty:
            st.info("No predictions/actuals found for this selection.")
        else:
            st.dataframe(preds_std, use_container_width=True)
    else:
        summary_df = load_summary(SUMMARY_PATH)
        if summary_df.empty:
            st.info("summary_metrics.csv not found or empty.")
        else:
            st.dataframe(summary_df, use_container_width=True)
    

# ---------- Footer ----------
st.divider()
st.markdown(
    '''
    <small>
    <b>2025 GDP Predictor </b><br>
    </small>
    ''',
    unsafe_allow_html=True
)

