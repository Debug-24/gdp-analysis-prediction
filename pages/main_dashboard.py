
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
MULTI_PATH = DATA_RAW_DIR / "gdp_dataset_for_ml.csv"
USONLY_PATH = DATA_RAW_DIR / "us_only_data.csv"
USPRED_PATH = DATA_RAW_DIR / "us_gdp_predictors.csv"

# ---------- Load Data ----------

@st.cache_data(show_spinner=False)
def load_dataset(which: str) -> pd.DataFrame:
    if which == "multi":
        if not MULTI_PATH.exists():
            st.warning(f"File not found: {MULTI_PATH}")
            return pd.DataFrame()
        df = pd.read_csv(MULTI_PATH)
    else:
        if not USONLY_PATH.exists():
            st.warning(f"File not found: {USONLY_PATH}")
            return pd.DataFrame()
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

# ---------- Sidebar : Data & Controls ----------
with st.sidebar:
    st.header("Data Source")
    dataset = st.radio(
        "Choose dataset",
        [
            "Multi-country GDP",
            "US-only GDP ",
            "US GDP + Predictors "
        ],
        index=0
    )

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

    st.header("Model (From Pipeline Summary)")
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

    st.header("Forecast")
    horizon = st.slider("Forecast quarters", 1, 20, 8)   
    run = st.button("Run")


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
tab_overview, tab_model, tab_data = st.tabs(["Overview", "Model Output", "Data"])

# --Overview Tab
with tab_overview:
    st.subheader("Trend Preview (Actuals)")

    preds_raw, _ = load_preds(country, feature_set_label)
    preds_std = standardize_preds_df(preds_raw)

    if preds_std.empty:
        st.info("No predictions/actuals file found yet for this selection. Run a forecast or check data-processed.")
    else:
        df_plot = preds_std.copy()
        try:
            t_parsed = pd.to_datetime(df_plot["t"], errors="coerce")
            mask = (t_parsed.dt.year >= years[0]) & (t_parsed.dt.year <= years[1])
            if mask.notna().any():
                df_plot = df_plot[mask.fillna(False)]
        except Exception:
            pass

        # Plot actuals only (change to ["y_true","y_pred"] to plot both)
        long_df = df_plot.melt(
            id_vars=["t"], value_vars=["y_true"],
            var_name="series", value_name="value"
        )

        chart = (
            alt.Chart(long_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("t:O", title="Time"),
                y=alt.Y("value:Q", title="GDP (Actual)"),
                tooltip=["t", "value"]
            )
            .properties(height=380)
        )
        st.altair_chart(chart, use_container_width=True)
    

# --Model Output Tab
with tab_model:
    st.subheader(f"Model Output — {country} · {model_label}")

    # -------- Summary Metrics Table --------
    summary_df = load_summary(SUMMARY_PATH)
    with st.expander("Summary Metrics", expanded=True):
        if summary_df.empty:
            st.info("summary_metrics.csv not found.")
        else:
            if "country" in summary_df.columns:
                show_df = summary_df[summary_df["country"] == country]
            else:
                show_df = summary_df
            st.dataframe(show_df, use_container_width=True)

    st.divider()

    # -------- Feature Importance Chart --------
    st.markdown("### Feature Importance (Permutation)")
    imp_df = load_importance(country, feature_set_label)
    if imp_df.empty:
        st.info(f"No importance file found for {country} - {feature_set_label}")
    else:
        top_k = st.slider("Top features to display", 5, 30, 10)
        plot_df = imp_df.sort_values("importance", ascending=False).head(top_k)

        friendly_names = {
            "grossfixedcapitalformation": "Investment",
            "fixedcapitalformation": "Investment",
            "gfcf": "Investment",
            "investment": "Investment",

            "tradebalance": "Trade Balance",
            "consumerpriceindex": "CPI",
            "cpi": "CPI",

            "unemploymentrate": "Unemployment",
            "unemployment": "Unemployment",

            "consumption": "Consumption",
            "governmentexpenditure": "Government Spending",
            "governmentspending": "Government Spending",
            "government_spending": "Government Spending",
            "exports": "Exports",
            "imports": "Imports",
        }

        # Normalize a key for lookup, then map to friendly label
        plot_df["__key"] = (
            plot_df["feature"]
            .astype(str)
            .str.replace(r"[\s_]+", "", regex=True)
            .str.lower()
        )
        plot_df["feature_ui"] = plot_df["__key"].map(friendly_names).fillna(plot_df["feature"])

        #For labels not to get cut
        row_height = 40  # maybe 36–44 for tighter
        chart_height = max(row_height * len(plot_df), 200)
        chart = (
            alt.Chart(plot_df)
            .mark_bar()
            .encode(
                x=alt.X("importance:Q", title="Importance"),
                y=alt.Y("feature_ui:N", sort='-x', title="Feature", axis=alt.Axis(labelLimit=220)),
                tooltip=["feature_ui:N", "importance:Q"]
        )
    .properties(height=chart_height)
)
        
        st.altair_chart(chart, use_container_width=True)

    st.divider()
    

    # -------- Predictions vs Actual Chart --------
    st.markdown("### Predictions vs Actual")

    # UNPACK: load_preds returns (df, time_col)
    preds_raw, _ = load_preds(country, feature_set_label)

    # Standardize to columns t, y_true, y_pred
    preds_df_std = standardize_preds_df(preds_raw)

    if preds_df_std.empty:
        st.info(f"No prediction file found or missing columns for {country} - {feature_set_label}")
    else:
        long_df = preds_df_std.melt(
            id_vars=["t"], value_vars=["y_true", "y_pred"],
            var_name="series", value_name="value"
        )
        line = (
            alt.Chart(long_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("t:O", title="Time"),
                y=alt.Y("value:Q", title="GDP Value"),
                color="series:N",
                tooltip=["t", "series", "value"]
            )
            .properties(height=380)
        )
        st.altair_chart(line, use_container_width=True)
    
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

