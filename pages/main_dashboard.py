
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

# ---- Custom CSS to resize metric text ----
st.markdown("""
<style>
div[data-testid="stMetricValue"] {
    font-size: 18px !important;
    font-weight: 600;
}
div[data-testid="stMetricLabel"] {
    font-size: 12px !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #888;
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
    else:
        st.caption(f"Tip: Add a `{path}` file for custom styling.")

load_css()


# ---------- File locations ----------

# Updated paths to use ROOT variable (file is now in pages/ directory)
DATA_RAW_DIR = ROOT / "data" / "data-raw"
DATASET_PATH = DATA_DIR / "gdp_dataset_for_ml.csv"


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
        st.success(f"Best Model: {feature_set_label} · {winner_model}")
    else:
        st.warning("summary_metrics.csv didn't have a row for this country. Choose manually.")
        feature_set_label = st.selectbox("Feature Set", ["Enhanced", "Traditional"], index=0)
        winner_model = st.selectbox("Model", ["RandomForest", "Ridge", "Linear"], index=0)

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
with c3:
    st.metric("Model", model_label)

st.divider()

# ---------- Tabs ----------
tab_prediction, tab_features, tab_data = st.tabs(["Prediction", "Features Ranked", "Data"])

# Prediction Tab
with tab_prediction:
    st.subheader("Prediction")
    
    if not run:
        st.info("**Please select a country from the sidebar and click 'Run Prediction' to see the GDP prediction results and visualization.**")
    else:
        st.success(f"Generating prediction chart for {country} on {years[0]}–{years[1]}.")

        # --- Load summary metrics to find the winning feature set/model for this country ---
        summary_path = Path("data") / "data-processed" / "summary_metrics.csv"
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
            preds_path = Path("data") / "data-processed" / f"preds_{country}_{chosen_feature_set}.csv"
            if preds_path.exists():
                preds_df = pd.read_csv(preds_path, parse_dates=["quarter"]) 
            else:
                for alt_fs in ["Traditional", "Enhanced"]:
                    alt_path = Path("data") / "data-processed" / f"preds_{country}_{alt_fs}.csv"
                    if alt_path.exists():
                        preds_df = pd.read_csv(alt_path, parse_dates=["quarter"])
                        preds_path = alt_path
                        break

        if preds_df is None:
            st.warning(f"No prediction file found for {country}. Expected at preds_{country}_<FeatureSet>.csv in data/data-processed.")
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
                k1, k2, k3 = st.columns(3)
                with k1:
                    if winner_model:
                        st.metric("Winning model", f"{winner_model}")
                    else:
                        st.metric("Winning model", "N/A")
                with k2:
                    st.metric("Feature set", f"{chosen_feature_set}" if chosen_feature_set else "N/A")
                with k3:
                    st.metric("RMSE", f"{chosen_rmse:.2f}" if chosen_rmse is not None else "N/A")

        #------------Forecast Chart-----------
        st.divider()
        st.markdown("### Forecast (Using Best Config)")

        if run:
            try:
                # Build the modeling dataframe for the selected country + feature set
                country_df, used_feats = load_country_model_df(country, feature_set_label)
                if country_df.empty or "GDP" not in country_df.columns:
                    st.warning("Modeling data not available or GDP column missing. Check data/data-processed/clean_quarterly.csv.")
                else:
                    df_fore = generate_forecast(
                        model_name=winner_model,
                        country_df=country_df,
                        feature_set_label=feature_set_label,
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
                        st.write(f"Winner model: **{winner_model}** · Feature set: **{feature_set_label}**")
                        st.write("Features used:", used_feats)
                        st.dataframe(df_fore, use_container_width=True)

            except Exception as e:
                st.error(f"Forecast failed: {e}")
        else:
            st.caption("Set a horizon and click **Run** to produce future quarters.")
                
    

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
    importance_path = Path("data") / "data-processed" / f"importance_{country}_{features_type}.csv"
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

