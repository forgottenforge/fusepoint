"""
Fuse Web App - upload a CSV, get instant analysis of every column.

Run via the installed CLI:
    fuse-ui

Or directly:
    python -m streamlit run fusepoint/web.py
"""

import io
import os
import time
import base64
import streamlit as st
import pandas as pd
import numpy as np

_ASSETS = os.path.join(os.path.dirname(__file__), "assets")

# Pre-encode logos for inline HTML
with open(os.path.join(_ASSETS, "ff.png"), "rb") as _f:
    _FF_LOGO_B64 = base64.b64encode(_f.read()).decode()
with open(os.path.join(_ASSETS, "fuse_logo.png"), "rb") as _f:
    _FUSE_LOGO_B64 = base64.b64encode(_f.read()).decode()

st.set_page_config(
    page_title="Fuse — Find Your Breaking Point",
    page_icon="📊",
    layout="wide",
)

# Custom CSS — spacing, styled uploader + demo box
st.markdown("""
<style>
    .block-container { padding-top: 2.5rem; }
    [data-testid="stFileUploader"] {
        border: 2px dashed #3b82f6;
        border-radius: 10px;
        padding: 0.3rem 0.6rem;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #60a5fa;
        background: rgba(59, 130, 246, 0.04);
    }
    /* Blue dashed border — only the INNERMOST block that holds the marker */
    [data-testid="stVerticalBlock"]:has(.demo-box-marker):not(:has([data-testid="stVerticalBlock"] .demo-box-marker)) {
        border: 2px dashed #3b82f6;
        border-radius: 10px;
        padding: 0.1rem 1rem 0.5rem 1rem;
    }
    .demo-box-marker { display: none; }
    /* Compact metrics in results list */
    [data-testid="stMetricValue"] { font-size: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Header — logos + tagline + paper reference
# ---------------------------------------------------------------------------
_hdr_left, _hdr_right = st.columns([4, 1])
with _hdr_left:
    st.image(os.path.join(_ASSETS, "fuse_logo.png"), width=180)
with _hdr_right:
    st.markdown(
        f'<a href="https://forgottenforge.xyz" target="_blank">'
        f'<img src="data:image/png;base64,{_FF_LOGO_B64}" '
        f'width="100" style="border-radius:8px; float:right; margin-top:4px;" />'
        f'</a>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<p style="font-size:1.4rem; font-weight:700; margin-bottom:0.15rem; letter-spacing:-0.01em;">'
    'Find the exact value where your system breaks.'
    '</p>'
    '<p style="font-size:0.95rem; color:#9ca3af; margin-top:0;">'
    'Nonparametric tipping-point detection with bootstrap CI &amp; permutation tests. '
    'Based on: <a href="https://doi.org/10.1116/5.0312410" '
    'target="_blank" style="color:#60a5fa; text-decoration:none;">'
    'M. C. Wurm, <em>AVS Quantum Sci.</em> <strong>8</strong>, 013804 (2026)</a>. No ML — just statistics.'
    '</p>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# HERO — interactive demo card, instant visual impact
# ---------------------------------------------------------------------------
from fusepoint import analyze as _fuse_analyze
from fusepoint.card import render_card as _fuse_card
import matplotlib.pyplot as plt

# Check whether data is already loaded (hides the hero when analyzing)
_has_data = "demo_df" in st.session_state or (
    "_uploader" in st.session_state and st.session_state["_uploader"] is not None
)


@st.cache_data(show_spinner=False)
def _hero_demo(current_x):
    """Cached hero analysis — fast on repeated slider values."""
    rq = np.linspace(100, 10000, 100)
    lat = 10 + 50 / (1 - np.clip(rq / 7500, 0, 0.98))
    lat += np.random.default_rng(7).normal(0, 5, 100)
    return _fuse_analyze(rq, lat, current_x=current_x,
                         x_name="Concurrent Requests",
                         y_name="Response Time (ms)",
                         label="Server Load Test",
                         n_boot=200, n_perm=500)


@st.cache_data(show_spinner=False)
def _showcase_analyses():
    """Cached showcase mini-card analyses — computed once, reused always."""
    rng = np.random.default_rng(42)

    B = np.linspace(0.5, 4.0, 80)
    ent = 0.2 + 3.0 / (1 + np.exp(-8 * (B - 2.3))) + rng.normal(0, 0.1, 80)
    r1 = _fuse_analyze(B, ent, current_x=1.9,
                       x_name="Magnetic Field (T)", y_name="Entanglement Entropy",
                       label="Quantum Phase Transition", n_boot=200, n_perm=500)

    dose = np.linspace(0, 100, 80)
    surv = np.clip(100 - 90 / (1 + np.exp(-0.2 * (dose - 42))) + rng.normal(0, 2, 80), 0, 100)
    r2 = _fuse_analyze(dose, surv, current_x=30,
                       x_name="Dose (mg/L)", y_name="Cell Survival (%)",
                       label="Cytotoxicity Assay", n_boot=200, n_perm=500)

    tf = np.linspace(400, 1600, 100)
    ero = 0.5 + 20.0 / (1 + np.exp(-0.03 * (tf - 1180))) + rng.normal(0, 0.3, 100)
    r3 = _fuse_analyze(tf, ero, current_x=1100,
                       x_name="Plasma Temp (\u00b0C)", y_name="Erosion (\u00b5m/pulse)",
                       label="Fusion Reactor Erosion", n_boot=200, n_perm=500)

    return r1, r2, r3


if not _has_data:
    _hero_left, _hero_right = st.columns([2, 3], gap="large")

    with _hero_left:
        st.markdown(
            '<p style="font-size:1.05rem; color:#e0e0e0; line-height:1.6; margin-top:0.5rem;">'
            'Drag the slider. Watch the score change as you approach the tipping point.'
            '</p>',
            unsafe_allow_html=True,
        )

        # Interactive operating point slider
        _hero_x = st.slider(
            "Your operating point (req/s)",
            min_value=500, max_value=9500, value=5000, step=100,
            key="_hero_slider",
        )

        # Dynamic feedback based on slider position
        if _hero_x < 5000:
            st.success(f"**{_hero_x} req/s** — Safe. Plenty of room before the cliff.")
        elif _hero_x < 6500:
            st.info(f"**{_hero_x} req/s** — Getting closer. Plan capacity now.")
        elif _hero_x < 7200:
            st.warning(f"**{_hero_x} req/s** — Danger zone. Add servers or throttle.")
        else:
            st.error(f"**{_hero_x} req/s** — Past the tipping point. System is degrading.")

        st.markdown(
            '<p style="font-size:0.85rem; color:#6b7280; margin-top:0.5rem;">'
            'This is real analysis running live — the same engine that analyzes '
            'quantum phase transitions, reactor erosion, and drug toxicity curves.'
            '</p>',
            unsafe_allow_html=True,
        )

    with _hero_right:
        _demo_r = _hero_demo(_hero_x)
        _demo_fig = _fuse_card(_demo_r)
        st.pyplot(_demo_fig, use_container_width=True)
        plt.close(_demo_fig)

    st.divider()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Import parsers from fusepoint (single source of truth)
from fusepoint.parsers import parse_json as _parse_json, detect_x_column as _detect_x_column


@st.cache_data(show_spinner=False)
def _analyze_single(df_csv, x_col, y_col):
    """Analyze a single column pair. Cached per column."""
    from fusepoint import analyze
    df = pd.read_csv(io.StringIO(df_csv))
    try:
        r = analyze(df, x=x_col, y=y_col,
                    n_boot=1000, n_perm=2000)
        return {
            "y_column": y_col,
            "score": r.score,
            "grade": r.grade,
            "tipping_point": r.critical_x,
            "kappa": r.kappa,
            "p_value": r.p_value,
            "significant": r.p_value < 0.05,
        }
    except Exception as e:
        return {
            "y_column": y_col,
            "score": None,
            "grade": "ERROR",
            "tipping_point": None,
            "kappa": None,
            "p_value": None,
            "significant": False,
            "error": str(e),
        }


def _scan_all_columns_with_progress(df_csv, x_col, y_cols):
    """Scan all columns with a visible progress bar."""
    results = []
    bar = st.progress(0, text="Starting analysis...")
    for i, y_col in enumerate(y_cols):
        bar.progress((i + 1) / len(y_cols),
                     text=f"Analyzing {y_col}  ({i+1}/{len(y_cols)})")
        results.append(_analyze_single(df_csv, x_col, y_col))
    bar.empty()
    # Sort: significant first, then by score descending
    results.sort(key=lambda r: (
        -(r["significant"] if r["score"] is not None else False),
        -(r["score"] or 0)
    ))
    return results


@st.cache_data(show_spinner=False)
def _full_analysis(df_csv, x_col, y_col, current_x):
    """Full analysis with high-quality bootstrap + permutation."""
    from fusepoint import analyze

    df = pd.read_csv(io.StringIO(df_csv))
    return analyze(df, x=x_col, y=y_col,
                   current_x=current_x,
                   n_boot=1000, n_perm=2000)


# ---------------------------------------------------------------------------
# File upload — prominent CTA
# ---------------------------------------------------------------------------
if not _has_data:
    st.markdown(
        '<p style="font-size:1.15rem; font-weight:600; color:#e0e0e0; margin-bottom:0.2rem;">'
        'Try it with your data'
        '</p>',
        unsafe_allow_html=True,
    )
uploaded = st.file_uploader("Upload CSV, TSV, JSON", type=["csv", "tsv", "txt", "json"],
                            key="_uploader", label_visibility="collapsed")

# Persist the parsed upload across reruns. On Streamlit Cloud / HF Spaces the
# file_uploader widget can transiently report None after st.rerun() following a
# button click. Caching the parsed DataFrame in session_state shields the rest
# of the flow from that race.
if uploaded is not None:
    _stash = st.session_state.get("_uploaded_df_stash") or {}
    if _stash.get("name") != uploaded.name or _stash.get("size") != uploaded.size:
        try:
            if uploaded.name.endswith(".json"):
                import json as _json
                _df_up = _parse_json(_json.load(uploaded))
            else:
                _sep = "\t" if uploaded.name.endswith(".tsv") else ","
                _df_up = pd.read_csv(uploaded, sep=_sep)
            st.session_state["_uploaded_df_stash"] = {
                "name": uploaded.name,
                "size": uploaded.size,
                "df": _df_up,
            }
        except Exception as _exc:
            st.error(f"Could not parse file: {_exc}")
            st.stop()

# Effective "do we have an uploaded dataset" — survives transient widget None.
_has_uploaded_data = (
    "_uploaded_df_stash" in st.session_state and "demo_df" not in st.session_state
)

if uploaded is None and not _has_uploaded_data:
    # Demo selector — in a blue-bordered box
    if "demo_df" not in st.session_state:
        st.markdown(
            '<p style="text-align:center; color:#6b7280; font-size:0.9rem; '
            'margin:0.8rem 0 0.8rem;">'
            'or try a demo'
            '</p>',
            unsafe_allow_html=True,
        )

    _DEMO_NAMES = [
        "Server Load Test",
        "ML Training Stability",
        "Quantum Phase Transition",
        "Fusion Reactor Erosion",
        "Chemistry — Reaction Kinetics",
        "Biology — Cytotoxicity",
    ]
    with st.container():
        st.markdown('<div class="demo-box-marker"></div>', unsafe_allow_html=True)
        _demo_left, _demo_right = st.columns([3, 1], gap="small")
        with _demo_left:
            demo_choice = st.selectbox(
                "Choose a demo",
                _DEMO_NAMES,
                key="demo_choice",
                label_visibility="collapsed",
            )
        with _demo_right:
            _load_demo = st.button("Load demo", use_container_width=True)
    if _load_demo:
        rng = np.random.default_rng(42)
        if demo_choice == "Server Load Test":
            rq = np.linspace(100, 10000, 100)
            st.session_state["demo_df"] = pd.DataFrame({
                "concurrent_requests": rq,
                "response_time_ms": 10 + 50 / (1 - np.clip(rq / 7500, 0, 0.98)) + rng.normal(0, 5, 100),
                "cpu_percent": 5 + 90 / (1 + np.exp(-0.002 * (rq - 6000))) + rng.normal(0, 2, 100),
                "memory_mb": 20 + 0.005 * rq + rng.normal(0, 3, 100),
                "error_rate": np.clip(0.1 + 50 / (1 + np.exp(-0.003 * (rq - 8000))) + rng.normal(0, 0.5, 100), 0, None),
            })
        elif demo_choice == "ML Training Stability":
            lr = np.linspace(0.001, 0.1, 100)
            st.session_state["demo_df"] = pd.DataFrame({
                "learning_rate": lr,
                "gradient_norm": 1.0 + 50 / (1 + np.exp(-200 * (lr - 0.052))) + rng.normal(0, 0.3, 100),
                "training_loss": 0.1 + 0.8 / (1 + np.exp(-180 * (lr - 0.055))) + rng.normal(0, 0.015, 100),
                "validation_accuracy": np.clip(0.95 - 0.6 / (1 + np.exp(-180 * (lr - 0.054))) + rng.normal(0, 0.01, 100), 0, 1),
            })
        elif demo_choice == "Quantum Phase Transition":
            B = np.linspace(0.5, 4.0, 80)
            st.session_state["demo_df"] = pd.DataFrame({
                "magnetic_field_T": B,
                "entanglement_entropy": 0.2 + 3.0 / (1 + np.exp(-8 * (B - 2.3))) + rng.normal(0, 0.1, 80),
                "magnetization": np.clip(1.0 - 0.9 / (1 + np.exp(-10 * (B - 2.3))) + rng.normal(0, 0.05, 80), 0, 1),
                "susceptibility_chi": 0.5 + 8.0 / (1 + np.exp(-6 * (B - 2.35))) + rng.normal(0, 0.2, 80),
            })
        elif demo_choice == "Fusion Reactor Erosion":
            temp_f = np.linspace(400, 1600, 100)
            st.session_state["demo_df"] = pd.DataFrame({
                "plasma_temperature_C": temp_f,
                "erosion_rate_um_per_pulse": 0.5 + 20.0 / (1 + np.exp(-0.03 * (temp_f - 1180))) + rng.normal(0, 0.3, 100),
                "surface_roughness_nm": 5.0 + 40.0 / (1 + np.exp(-0.025 * (temp_f - 1200))) + rng.normal(0, 1.0, 100),
                "deuterium_retention": 0.1 + 2.0 / (1 + np.exp(-0.02 * (temp_f - 1100))) + rng.normal(0, 0.05, 100),
            })
        elif demo_choice == "Chemistry — Reaction Kinetics":
            temp_c = np.linspace(20, 200, 100)
            st.session_state["demo_df"] = pd.DataFrame({
                "temperature_C": temp_c,
                "reaction_rate_mol_s": 0.01 + 5.0 / (1 + np.exp(-0.15 * (temp_c - 150))) + rng.normal(0, 0.06, 100),
                "yield_percent": np.clip(95 - 60 / (1 + np.exp(-0.12 * (temp_c - 155))) + rng.normal(0, 1.5, 100), 0, 100),
                "viscosity_mPas": 100 - 80 / (1 + np.exp(-0.1 * (temp_c - 80))) + rng.normal(0, 2, 100),
            })
        elif demo_choice == "Biology — Cytotoxicity":
            dose = np.linspace(0, 100, 80)
            st.session_state["demo_df"] = pd.DataFrame({
                "dose_mg_L": dose,
                "cell_survival_pct": np.clip(100 - 90 / (1 + np.exp(-0.2 * (dose - 42))) + rng.normal(0, 2, 80), 0, 100),
                "biomarker_level": 1.0 + 8.0 / (1 + np.exp(-0.15 * (dose - 45))) + rng.normal(0, 0.2, 80),
                "inflammation_index": 0.5 + 4.0 / (1 + np.exp(-0.2 * (dose - 35))) + rng.normal(0, 0.15, 80),
            })
        # No explicit st.rerun(): clicking "Load demo" already triggered a
        # rerun. demo_df is now in session_state and the script falls through
        # to the analysis section below.

    if "demo_df" in st.session_state:
        df = st.session_state["demo_df"]
    else:
        # -------------------------------------------------------------------
        # Landing page — value props + showcase + details
        # -------------------------------------------------------------------

        # --- Value proposition strip (Bauhaus: geometry + type hierarchy) ---
        st.markdown('<div style="height:0.6rem;"></div>', unsafe_allow_html=True)
        _vp1, _vp2, _vp3 = st.columns(3)
        with _vp1:
            st.markdown(
                '<div style="border-left:4px solid #3b82f6; padding:1.1rem 1.2rem; '
                'min-height:120px; background:rgba(59,130,246,0.04); '
                'border-radius:0 8px 8px 0;">'
                '<div style="font-size:0.65rem; text-transform:uppercase; '
                'letter-spacing:0.14em; color:#3b82f6; font-weight:700; '
                'margin-bottom:0.4rem;">Validated</div>'
                '<div style="font-size:1.05rem; font-weight:600; color:#e0e0e0; '
                'margin-bottom:0.3rem;">6 domains tested</div>'
                '<div style="color:#9ca3af; font-size:0.82rem; line-height:1.55;">'
                'Physics, Fusion, Chemistry,<br/>Biology, ML, Infrastructure</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        with _vp2:
            st.markdown(
                '<div style="border-left:4px solid #10b981; padding:1.1rem 1.2rem; '
                'min-height:120px; background:rgba(16,185,129,0.04); '
                'border-radius:0 8px 8px 0;">'
                '<div style="font-size:0.65rem; text-transform:uppercase; '
                'letter-spacing:0.14em; color:#10b981; font-weight:700; '
                'margin-bottom:0.4rem;">Published</div>'
                '<div style="font-size:1.05rem; font-weight:600; color:#e0e0e0; '
                'margin-bottom:0.3rem;">Peer-reviewed method</div>'
                '<a href="https://doi.org/10.1116/5.0312410" target="_blank" '
                'style="color:#9ca3af; font-size:0.82rem; text-decoration:none; '
                'line-height:1.55; display:block;">'
                'AVS Quantum Sci. <strong>8</strong>,<br/>013804 (2026)</a>'
                '</div>',
                unsafe_allow_html=True,
            )
        with _vp3:
            st.markdown(
                '<div style="border-left:4px solid #f59e0b; padding:1.1rem 1.2rem; '
                'min-height:120px; background:rgba(245,158,11,0.04); '
                'border-radius:0 8px 8px 0;">'
                '<div style="font-size:0.65rem; text-transform:uppercase; '
                'letter-spacing:0.14em; color:#f59e0b; font-weight:700; '
                'margin-bottom:0.4rem;">Transparent</div>'
                '<div style="font-size:1.05rem; font-weight:600; color:#e0e0e0; '
                'margin-bottom:0.3rem;">No black box</div>'
                '<div style="color:#9ca3af; font-size:0.82rem; line-height:1.55;">'
                'Bootstrap CI + permutation test.<br/>Every number has a p-value.</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        # --- Showcase mini-cards ---
        st.markdown('<div style="height:1.2rem;"></div>', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:1.05rem; font-weight:600; color:#9ca3af; '
            'text-align:center; letter-spacing:0.03em; margin-bottom:0.2rem;">'
            'Works across domains'
            '</p>',
            unsafe_allow_html=True,
        )
        _r1, _r2, _r3 = _showcase_analyses()
        _show1, _show2, _show3 = st.columns(3, gap="medium")

        with _show1:
            _f1 = _fuse_card(_r1)
            st.pyplot(_f1, use_container_width=True)
            plt.close(_f1)
            st.caption("Physics \u2014 phase transition at 2.27 T")

        with _show2:
            _f2 = _fuse_card(_r2)
            st.pyplot(_f2, use_container_width=True)
            plt.close(_f2)
            st.caption("Biology \u2014 toxicity cliff at 42 mg/L")

        with _show3:
            _f3 = _fuse_card(_r3)
            st.pyplot(_f3, use_container_width=True)
            plt.close(_f3)
            st.caption("Fusion \u2014 erosion spike at 1176\u00b0C")

        # --- How it works — single expander with grades + domains ---
        with st.expander("What the results mean \u2014 grades, tested domains & recommendations"):
            st.markdown("""
#### Grades & what to do

| Grade | Meaning | Action |
|---|---|---|
| **STABLE** (85+) | Far from the edge | You're fine. Set alerts at 90% of the tipping point. |
| **MODERATE** (60-84) | Room left, but limited | **Hold where you are.** Don't push harder without a plan. |
| **WARNING** (35-59) | Close to the edge | **Back off.** Lower temp, cut dose, add capacity. |
| **CRITICAL** (<35) | Past the breaking point | **Act now.** Reduce immediately. |

---

#### All 6 tested domains

| Domain | Tipping Point | Score | What to do |
|---|---|---|---|
| **Physics** \u2014 B-field | **2.27 T** (CI: 2.18\u20132.45) | 73 | Keep below 2.0 T. Hunting the transition? Sweep 2.1\u20132.5 T. |
| **Fusion** \u2014 Plasma temp | **1176\u00b0C** (CI: 1152\u20131212) | 73 | Hard limit at 1060\u00b0C. **Increase coolant** or **cut plasma power**. |
| **Chemistry** \u2014 Reaction temp | **151\u00b0C** (CI: 142\u2013155) | 76 | Reactor limit at 136\u00b0C. Past it? **Shutdown**, cut the feed. |
| **Biology** \u2014 Drug dose | **42 mg/L** (CI: 38\u201348) | 56 | Safe max: 38 mg/L. Need more effect? Try combination therapy. |
| **ML** \u2014 Learning rate | **0.053** (CI: 0.048\u20130.056) | 73 | Set lr to 0.04. Past 0.06? **Stop**, lower to 0.03, restart. |
| **Infra** \u2014 Requests | **7200 req/s** (CI: 7000\u20137300) | 85 | Autoscale at 5700. Past TP? **Shed load** \u2014 rate-limit, serve cached. |

**Rule of thumb:** Keep your operating point at **80\u201390% of the tipping point**. That's your real limit.

---

**Works with** any CSV/TSV/JSON, 2+ numeric columns, 8+ rows.
**Won't work** on pure noise, linear trends with no cliff, or fewer than 8 points.
""")

        # --- Footer with logos + copyright ---
        st.divider()
        st.markdown(
            f'<div style="display:flex; align-items:center; justify-content:space-between; '
            f'padding:0.5rem 0;">'
            f'<img src="data:image/png;base64,{_FUSE_LOGO_B64}" height="32" '
            f'style="opacity:0.7;" />'
            f'<span style="color:#6b7280; font-size:0.8rem;">'
            f'\u00a9 2026 Forgotten Forge \u2022 AGPL-3.0 \u2022 '
            f'<a href="https://forgottenforge.xyz" target="_blank" '
            f'style="color:#60a5fa; text-decoration:none;">forgottenforge.xyz</a>'
            f'</span>'
            f'<a href="https://forgottenforge.xyz" target="_blank">'
            f'<img src="data:image/png;base64,{_FF_LOGO_B64}" height="36" '
            f'style="border-radius:6px; opacity:0.7;" />'
            f'</a>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.stop()
else:
    # Read the parsed DataFrame from the stash filled when the upload arrived.
    # This keeps the rest of the script untouched whether the widget currently
    # returns a file object or has transiently dropped it across a rerun.
    df = st.session_state["_uploaded_df_stash"]["df"]

# ---------------------------------------------------------------------------
# Data overview
# ---------------------------------------------------------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) < 2:
    st.error(f"Need at least 2 numeric columns. Found: {numeric_cols}")
    st.stop()

# ---------------------------------------------------------------------------
# Data validation
# ---------------------------------------------------------------------------
if len(df) > 50_000:
    st.warning(f"Large dataset ({len(df):,} rows). Analysis may take a while. Consider sampling.")
if len(df) < 8:
    st.error(f"Only {len(df)} rows — FUSE needs at least 8 data points for reliable results.")
    st.stop()

with st.expander(f"Data preview ({len(df)} rows, {len(numeric_cols)} numeric columns)", expanded=False):
    st.dataframe(df.head(30), use_container_width=True)

# Auto-detect x column, let user override
detected_x = _detect_x_column(df, numeric_cols)
x_col = st.selectbox(
    "X-axis (independent variable)",
    numeric_cols,
    index=numeric_cols.index(detected_x),
    help="Auto-detected. Change if needed."
)

# ---------------------------------------------------------------------------
# Scan all columns
# ---------------------------------------------------------------------------
y_cols = [c for c in numeric_cols if c != x_col]
if not y_cols:
    st.warning("Need at least one other numeric column besides the x-axis.")
    st.stop()

st.divider()

df_csv = df.to_csv(index=False)
t0 = time.time()
scan_results = _scan_all_columns_with_progress(df_csv, x_col, y_cols)
scan_time = time.time() - t0

# Separate OK vs error results
ok_results = [r for r in scan_results if r["score"] is not None]
err_results = [r for r in scan_results if r["score"] is None]

# ---------------------------------------------------------------------------
# Two-column layout: results list (left) + card (right)
# ---------------------------------------------------------------------------
grade_emoji = {"STABLE": "🟢", "MODERATE": "🟡", "WARNING": "🟠", "CRITICAL": "🔴", "ERROR": "⚫"}

col_list, col_card = st.columns([1, 2], gap="large")

with col_list:
    st.subheader(f"{len(ok_results)} of {len(y_cols)} columns scanned")
    st.caption(f"Scanned in {scan_time:.1f}s")

    # Column search for large datasets
    if len(ok_results) > 8:
        search = st.text_input("Filter columns", key="col_search",
                               placeholder="Type to filter...",
                               label_visibility="collapsed")
    else:
        search = ""

    shown = 0
    for i, res in enumerate(scan_results):
        if res["score"] is None:
            continue
        if search and search.lower() not in res["y_column"].lower():
            continue

        emoji = grade_emoji.get(res["grade"], "")
        sig = " ✱" if res["significant"] else ""
        tp_str = f"{res['tipping_point']:.4g}" if res['tipping_point'] is not None else "—"
        is_selected = st.session_state.get("selected_y") == res["y_column"]

        with st.container():
            r_score, r_info, r_btn = st.columns([0.8, 3, 1.2])
            with r_score:
                st.markdown(f"### {emoji} {res['score']}")
            with r_info:
                st.markdown(f"**{res['y_column']}**")
                tp_line = f"TP: **{tp_str}**" if res["significant"] else "No TP"
                st.caption(f"{res['grade']}{sig} | k={res['kappa']:.1f} | p={res['p_value']:.3f} | {tp_line}")
            with r_btn:
                if st.button("View", key=f"view_{i}", use_container_width=True,
                             type="primary" if is_selected else "secondary"):
                    st.session_state["selected_y"] = res["y_column"]
                    # No explicit st.rerun(): clicking a Streamlit button
                    # already triggers a rerun. The explicit call caused a
                    # double-rerun on Streamlit 1.46 that wiped session_state
                    # on HF Spaces (page went blank after View click).
            st.divider()
        shown += 1

    if search and shown == 0:
        st.caption(f"No columns matching '{search}'")

    # Show errors if any
    if err_results:
        with st.expander(f"{len(err_results)} columns failed"):
            for er in err_results:
                st.caption(f"**{er['y_column']}** — {er.get('error', 'Unknown error')}")

with col_card:
    # Clear stale selection if column doesn't exist in current data
    if "selected_y" in st.session_state and st.session_state["selected_y"] not in y_cols:
        del st.session_state["selected_y"]

    if "selected_y" in st.session_state:
        y_sel = st.session_state["selected_y"]

        st.subheader(f"Stability Card: {y_sel}")

        current_x = st.number_input(
            "Current operating point (optional)", value=None, format="%g",
            key="current_x_full",
            help="Enter your current x-value to calculate safety margin. "
                 "Example: if x is 'concurrent_requests' and you're running at 5000, enter 5000. "
                 "FUSE will show how far you are from the tipping point."
        )

        with st.spinner("Running full analysis..."):
            result = _full_analysis(df_csv, x_col, y_sel, current_x)

        # Score + diagnosis
        sc_col, diag_col = st.columns([1, 2.5])
        with sc_col:
            st.metric("Score", f"{result.score} / 100", delta=result.grade)
        with diag_col:
            st.markdown(f"**{result.diagnosis}**")

        # Card
        from fusepoint.card import render_card
        import matplotlib.pyplot as plt

        fig = render_card(result)
        st.pyplot(fig, use_container_width=True)

        # Download buttons
        dl_single, dl_all = st.columns(2)
        with dl_single:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=200,
                        facecolor=fig.get_facecolor(), edgecolor="none")
            buf.seek(0)
            plt.close(fig)
            st.download_button(
                "Download this Card",
                data=buf,
                file_name=f"stability_{y_sel}.png",
                mime="image/png",
                use_container_width=True,
            )
        with dl_all:
            if st.button("Download ALL Cards (ZIP)", use_container_width=True):
                import zipfile
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    prog = st.progress(0, text="Generating cards...")
                    for idx, res in enumerate(ok_results):
                        prog.progress((idx + 1) / len(ok_results),
                                      text=f"Rendering {res['y_column']}...")
                        try:
                            r = _full_analysis(df_csv, x_col, res["y_column"], None)
                            f = render_card(r)
                            card_buf = io.BytesIO()
                            f.savefig(card_buf, format="png", dpi=200,
                                      facecolor=f.get_facecolor(), edgecolor="none")
                            plt.close(f)
                            card_buf.seek(0)
                            safe_name = res["y_column"].replace("/", "_").replace("\\", "_")
                            zf.writestr(f"stability_{safe_name}.png", card_buf.getvalue())
                        except Exception:
                            pass
                    prog.empty()
                zip_buf.seek(0)
                st.download_button(
                    "Save ZIP",
                    data=zip_buf,
                    file_name="fuse_cards.zip",
                    mime="application/zip",
                    use_container_width=True,
                )

        with st.expander("Full metrics"):
            st.code(result.summary())
    else:
        st.markdown(
            "<div style='display:flex; align-items:center; justify-content:center; "
            "height:400px; color:#6b7280; font-size:1.1rem;'>"
            "Select a column on the left to view its Stability Card"
            "</div>",
            unsafe_allow_html=True,
        )

# --- Reference block (always available, also on results pages) ---
st.divider()
with st.expander("What the results mean \u2014 grades, tested domains & recommendations"):
    st.markdown("""
#### Grades & what to do

| Grade | Meaning | Action |
|---|---|---|
| **STABLE** (85+) | Far from the edge | You're fine. Set alerts at 90% of the tipping point. |
| **MODERATE** (60-84) | Room left, but limited | **Hold where you are.** Don't push harder without a plan. |
| **WARNING** (35-59) | Close to the edge | **Back off.** Lower temp, cut dose, add capacity. |
| **CRITICAL** (<35) | Past the breaking point | **Act now.** Reduce immediately. |

---

#### All 6 tested domains

| Domain | Tipping Point | Score | What to do |
|---|---|---|---|
| **Physics** \u2014 B-field | **2.27 T** (CI: 2.18\u20132.45) | 73 | Keep below 2.0 T. Hunting the transition? Sweep 2.1\u20132.5 T. |
| **Fusion** \u2014 Plasma temp | **1176\u00b0C** (CI: 1152\u20131212) | 73 | Hard limit at 1060\u00b0C. **Increase coolant** or **cut plasma power**. |
| **Chemistry** \u2014 Reaction temp | **151\u00b0C** (CI: 142\u2013155) | 76 | Reactor limit at 136\u00b0C. Past it? **Shutdown**, cut the feed. |
| **Biology** \u2014 Drug dose | **42 mg/L** (CI: 38\u201348) | 56 | Safe max: 38 mg/L. Need more effect? Try combination therapy. |
| **ML** \u2014 Learning rate | **0.053** (CI: 0.048\u20130.056) | 73 | Set lr to 0.04. Past 0.06? **Stop**, lower to 0.03, restart. |
| **Infra** \u2014 Requests | **7200 req/s** (CI: 7000\u20137300) | 85 | Autoscale at 5700. Past TP? **Shed load** \u2014 rate-limit, serve cached. |

**Rule of thumb:** Keep your operating point at **80\u201390% of the tipping point**. That's your real limit.

---

**Works with** any CSV/TSV/JSON, 2+ numeric columns, 8+ rows.
**Won't work** on pure noise, linear trends with no cliff, or fewer than 8 points.
""")

# --- Footer ---
st.markdown(
    f'<div style="display:flex; align-items:center; justify-content:space-between; '
    f'padding:0.3rem 0;">'
    f'<img src="data:image/png;base64,{_FUSE_LOGO_B64}" height="28" '
    f'style="opacity:0.6;" />'
    f'<span style="color:#6b7280; font-size:0.78rem;">'
    f'\u00a9 2026 Forgotten Forge \u2022 AGPL-3.0 \u2022 '
    f'<a href="https://forgottenforge.xyz" target="_blank" '
    f'style="color:#60a5fa; text-decoration:none;">forgottenforge.xyz</a>'
    f'</span>'
    f'<a href="https://forgottenforge.xyz" target="_blank">'
    f'<img src="data:image/png;base64,{_FF_LOGO_B64}" height="32" '
    f'style="border-radius:6px; opacity:0.6;" />'
    f'</a>'
    f'</div>',
    unsafe_allow_html=True,
)
