
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
from data_utils import filter_data
import plotly.graph_objects as go

# ---------- Page config ----------
st.set_page_config(
    page_title="GDP Predictor Tool",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- File locations ----------
DATA_DIR = Path("data")
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

# ---------- Sidebar : Filters & Controls ----------
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
    years = st.slider("Timeline (Years)", min_value=year_min, max_value=year_max_default, value=(2010, year_max_default))

    feature_set_choice = st.selectbox(
        "Features Set to display",
        ["Auto (best)", "Traditional", "Enhanced"],
        index=0
    )

    run = st.button("Run Prediction")

# Page Header
st.title("GDP Predictor Tool")

# ---------- KPI cards ----------
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Selected Country", country)
with c2:
    st.metric("Years", f"{years[0]}–{years[1]}")


st.divider()

# Tabs 
tab_prediction, tab_features = st.tabs(["Prediction", "Features Ranked"])

# Prediction Tab
with tab_prediction:
    st.subheader("Prediction")
    if run:
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
    else:
        st.warning("Click **Run Prediction** after selecting a model to see results.")
