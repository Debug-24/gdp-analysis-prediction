"""
Landing Page — GDP Predictor Tool

"""
from pathlib import Path
import base64
import mimetypes
import streamlit as st

st.set_page_config(page_title="GDP Predictor Tool", page_icon="🌍", layout="wide", initial_sidebar_state="collapsed")

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

if BG_FILE.exists():
    mime, _ = mimetypes.guess_type(BG_FILE.name)
    if mime is None:
        # fallback based on extension
        ext = BG_FILE.suffix.lower()
        mime = "image/svg+xml" if ext == ".svg" else "image/png"

    b64 = base64.b64encode(BG_FILE.read_bytes()).decode("utf-8")

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
    st.switch_page("pages/main.py")


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
