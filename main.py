"""
Landing Page — GDP Predictor Tool

"""
from pathlib import Path
<<<<<<< HEAD
from data_utils import filter_data
import plotly.graph_objects as go
=======
import base64
import mimetypes
import streamlit as st
>>>>>>> main

st.set_page_config(page_title="GDP Predictor Tool", page_icon="🌍", layout="wide", initial_sidebar_state="collapsed")

<<<<<<< HEAD
# ---------- File locations ----------
DATA_DIR = Path("data")
DATASET_PATH = DATA_DIR / "gdp_dataset_for_ml.csv"

# ---------- Load Data ----------
@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    """Load and preprocess the GDP dataset"""
    df = pd.read_csv(DATASET_PATH)
=======
st.markdown("""
<style>
/* Hide the sidebar + its collapse handle */
section[data-testid="stSidebar"] { display: none !important; }
button[kind="header"] { display: none !important; }  /* mobile sidebar toggle */

/* Expand the main area to full width */
div[data-testid="stAppViewContainer"] { margin-left: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ---------- Full Globe as BG ----------

BG_FILE = Path("assets/globe.jpg")  # <-- change to your file: .jpg/.png/.svg
>>>>>>> main

if BG_FILE.exists():
    mime, _ = mimetypes.guess_type(BG_FILE.name)
    if mime is None:
        # fallback based on extension
        ext = BG_FILE.suffix.lower()
        mime = "image/svg+xml" if ext == ".svg" else "image/png"

    b64 = base64.b64encode(BG_FILE.read_bytes()).decode("utf-8")

<<<<<<< HEAD
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

=======
# ---------- Full-page background ----------
    st.markdown(
        f"""
            <style>
            html, body, .stApp, [data-testid="stAppViewContainer"] {{
                background: transparent !important;
            }}
            [data-testid="stHeader"], [data-testid="stToolbar"] {{
                background: rgba(255,255,255,0) !important;
            }}
            [data-testid="stSidebar"] {{
                background: rgba(255,255,255,0.85) !important;
                backdrop-filter: blur(2px);
            }}
            #bg-wrap {{
              position: fixed;
              inset: 0;
              z-index: -1;
              background: url("data:{mime};base64,{b64}") no-repeat center center fixed;
              background-size: cover;
            }}
            </style>
            <div id="bg-wrap"></div>
            """,
        unsafe_allow_html=True
    )

# ---------- Add dark overlay (Overall on top of the globe) ----------
    st.markdown(
        """
        <style>
        #overlay {
            position: fixed;
            inset: 0;
            background: rgba(0, 0, 0, 0.35); /* adjust darkness */
            z-index: -1;
        }
        </style>
        <div id="overlay"></div>
        """,
        unsafe_allow_html=True
    )

else:
    st.info("Background image not found. Put it at assets/globe.jpg (or update BG_FILE).")


# ---------------- CSS Styles  ----------------
st.markdown("""
<style>
/* Center the hero content and add breathing space */
.hero {
  max-width: 800px;
  margin: 0 auto;
  padding-top: 2rem;         /* small, even spacing at top */
  padding-bottom: 1rem;
  text-align: center;        
  display: flex;
  flex-direction: column;
  justify-content: center;   
  align-items: center;
}
.hero h1 {
  font-size: clamp(2rem, 6vw, 3.25rem);
  line-height: 1.1;
  margin-bottom: 0.25rem;
  font-weight: 800;
}
.hero p.lead {
  font-size: clamp(1rem, 2.6vw, 1.25rem);
  color: #374151;
  margin-bottom: 1.25rem;
}
.hero img {
  width: 100%;
  height: auto;
  border-radius: 16px;
  box-shadow: 0 10px 30px rgba(0,0,0,.08);
}
>>>>>>> main


<<<<<<< HEAD
# Tabs 
tab_prediction, tab_features = st.tabs(["Prediction", "Features Ranked"])

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

=======

/* CTA buttons */
.cta { margin: 1.25rem 0 0 }
.cta .row { display:flex; gap: 12px; justify-content: center; flex-wrap: wrap; }
.cta a, .cta button {
  border-radius: 9999px !important;
  padding: .6rem 1.1rem !important;
  font-weight: 600 !important;
}
/* Theme harmony with green primary */
:root {
  --primary: #174734;
  --tint: #fef4eb;
}
a[kind="primary"], .stButton>button[kind="primary"] { background: var(--primary); color: #fff; }
a[kind="secondary"], .stButton>button[kind="secondary"] { background: var(--tint); color: #111827; border: 1px solid rgba(0,0,0,.06); }
</style>
""", unsafe_allow_html=True)

# ---------- Background Globe ----------
globe_path = Path("assets/globe.jpg")

if globe_path.exists():
    # Set the background via inline CSS
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background: url("{globe_path.as_posix()}") no-repeat center center fixed;
            background-size: cover;
        }}
        [data-testid="stHeader"], [data-testid="stToolbar"] {{
            background: rgba(255, 255, 255, 0); /* make Streamlit header transparent */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.info("Add a globe image to assets/ to display the background.")

# ---------------- Hero/Hero Box (around GDP P.T ----------------
st.markdown(
    """
    <div class="hero-box">
      <h1>GDP Predictor Tool</h1>
      <p>Forecast smarter. Plan better. See what’s next.</p>
    </div>

    <style>
        .hero-box {
            max-width: 800px;
            margin: 0 auto;
            text-align: center;
            background: rgba(208, 224, 204, 0.55);
            border-radius: 16px;
            padding: 1.1rem 1.6rem;           
            box-shadow: 0 0 25px rgba(23, 71, 52, 0.25);
            backdrop-filter: blur(6px);
            transition: all 0.3s ease-in-out;
        }
        .hero-box:hover {
            box-shadow: 0 0 35px rgba(23, 71, 52, 0.35);
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
            transform: scale(1.02);
        }
        .hero-box h1 {
            margin-bottom: 0rem;               
            line-height: 1.1;
            color: #174734;
            font-weight: 800;
        }
        .hero-box p {
            margin-top: 0;                        
            color: #374151;
            font-size: 1.1rem;
            line-height: 1.35;
        }
    </style>
    """,
    unsafe_allow_html=True
)
#------CTA Button + Effect-----

left, middle, right = st.columns([1.1, 0.4, 1])
st.markdown("""
<style>
            
.stButton>button {
    background: #d97706 !important;               /* darker orange */
    color:#fff !important;
    padding:12px 22px;
    border-radius:9999px;
    font-weight:600;
    margin-top:40px;
    text-decoration:none;
    display:inline-block;
    box-shadow:0 0 15px rgba(217,119,6,.55);  /* glow */
    backdrop-filter: blur(6px);
    transition: all .25s ease;
}
.stButton>button:hover {
    background:#f59e0b;              
    box-shadow:0 0 26px rgba(245,158,11,.85);
    transform: scale(1.04);
}
</style>
""", unsafe_allow_html=True)

if middle.button("Launch Dashboard"):
    st.switch_page("pages/main_dashboard.py")


# ---------------- Features ----------------
st.markdown("""
<h3>Why this tool?</h3>
<div class="grid">
  <div class="card">
    <h4>Interactive Trends</h4>
    <p>Slice by country and time. Visualize GDP with smooth, responsive charts.</p>
  </div>

  <div class="card">
    <h4>ML Predictions</h4>
    <p>Experiment with regression models and evaluate metrics (R², MAE, RMSE).</p>
  </div>

  <div class="card">
    <h4>Clean Data Flow</h4>
    <p>Simple CSV loading, caching, and consistent filters across tabs.</p>
  </div>

  <div class="card">
    <h4>Transparent Methods</h4>
    <p>Learn the assumptions and data sources on the Methodology page.</p>
  </div>
</div>
""", unsafe_allow_html=True)


# ---------------- Feature Card Styling ----------------
st.markdown(
    """
    <style>
    
        h3 {
            color: rgba(255,255,255,0.95);
            text-shadow: 0 1px 3px rgba(0,0,0,0.1);
            font-weight: 700;
            margin-bottom: 1.5rem;
        }
      /* --- 2×2 layout on desktop, 1 column on mobile --- */
      .grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));  
        gap: 1.2rem;                                    /* spacing between cards */
        max-width: 960px;
        margin: 1.5rem auto 3rem;           /* center the grid */
        align-items: stretch;                     /* equal heights per row */
      }

      .card {
        background: rgba(255, 243, 230, 0.75);
        border-radius: 20px;
        padding: 1.25rem 1.1rem;
        border: 1px solid rgba(217, 119, 6, 0.10); /* faint warm border */
        box-shadow:
            0 4px 12px rgba(217, 119, 6, 0.08),     /* soft orange glow */
            0 2px 6px rgba(0, 0, 0, 0.05);          /* subtle depth */
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
        display: flex;
        flex-direction: column;
        gap: 0.35rem;
        height: 100%;
        transition: all 0.3s ease;
        position: relative;
        z-index: 2;
        color: #1a1a1a;                           
}

      .card:hover {
        transform: translateY(-6px);
        background: rgba(255, 255, 255, 0.85);   /* lighter, cleaner */
        box-shadow:
            0 10px 25px rgba(217, 119, 6, 0.20),   /* stronger orange glow */
            0 4px 12px rgba(0, 0, 0, 0.05);
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
        cursor: pointer;
     }

      .card h4 {
        margin: 0;          
        line-height: 1.2;
        color: #92400e;
        font-weight: 700;
        font-size: 1.05rem;
      }

      .card p {
        margin: 0;          
        line-height: 1.35;
        color: #4b2e05;
        font-size: 1rem;
      }

      /* Stack to 1 column on smaller screens */
      @media (max-width: 860px) {
        .grid { grid-template-columns: 1fr; }
      }
    </style>
    """,
    unsafe_allow_html=True
)


#---------Feature Cards + Effect ----------
st.markdown(
    """
    <style>
    /* --- Features (scoped) --- */
    .features-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
      gap: 1rem;
      margin: 2.5rem auto 4rem;
      max-width: 1000px;
    }


    .feature-card:hover {
      transform: translateY(-8px);
      box-shadow: 0 12px 25px rgba(23, 71, 52, 0.25);
      background: rgba(255, 255, 255, 0.9);
      cursor: pointer;
    }

    .feature-card h4 {
      /* ↓ this controls the space between the header and paragraph */
      margin: 0 0 0.06rem;
      line-height: 1.2;
      color: #174734;
      font-weight: 700;
      font-size: 1.05rem;
    }

    .feature-card p {
      margin: 0;              
      line-height: 1.35;
      color: #374151;
      font-size: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- Footer ----------------
st.divider()
st.caption("Group 44, 2025")
>>>>>>> main
