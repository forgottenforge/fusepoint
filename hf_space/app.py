"""
FUSE for Hugging Face Spaces - full elaborate UI with the HF-safe card pattern.

Brings back the slider hero, the demo selector, and the open showcase that
the user liked, while keeping the View-click card area in the minimal
"no st.spinner / no nested st.columns / no st.number_input(value=None)"
shape that has been proven to render on Streamlit 1.46 inside HF Spaces.

Library calls (analyze, render_card, parsers) come from the fusepoint
PyPI package; this file is the UI shell.
"""
import base64
import io
import os
import sys
import traceback

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import fusepoint
from fusepoint import analyze
from fusepoint.card import render_card
from fusepoint.parsers import parse_json, detect_x_column

# Locate bundled brand assets (shipped with the fusepoint wheel)
_ASSETS_DIR = os.path.join(os.path.dirname(fusepoint.__file__), "assets")


def _b64(name):
    p = os.path.join(_ASSETS_DIR, name)
    if not os.path.exists(p):
        return ""
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode()


_FF_LOGO = _b64("ff.png")
_FUSE_LOGO = _b64("fuse_logo.png")

print(f"[fuse-hf] script run | fusepoint {fusepoint.__version__} | "
      f"streamlit {st.__version__}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Page config + minimal CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FUSE - Find Your Breaking Point",
    page_icon=":bar_chart:",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* Always show the page scrollbar so its appear/disappear cannot
       reshuffle viewport width and pull right-aligned content sideways.
       This is the single biggest source of right-side wobble in Chrome
       inside the HF Spaces iframe. */
    html {
        overflow-y: scroll !important;
        scrollbar-gutter: stable;
    }
    .stApp {
        min-height: 100vh;
    }
    /* Push first content below the HF Spaces iframe top bar so the
       FUSE logo crown is not clipped. */
    .block-container {padding-top: 2.6rem; padding-bottom: 1.5rem;}
    /* Chrome inside the HF iframe reflows aggressively when matplotlib
       images load; pinning images to their natural aspect ratio and
       containing their layout stops the cascade-wobble. Every FUSE card
       PNG is rendered from figsize=(14, 8), so 14/8 is the universal
       aspect ratio for st.image content. */
    [data-testid="stImage"] {
        contain: layout style paint;
        aspect-ratio: 14 / 8;
        overflow: hidden;
        /* Force GPU layer so Chrome does not invalidate the composited
           pixels on every neighbouring layout change. */
        transform: translateZ(0);
        backface-visibility: hidden;
        will-change: transform;
    }
    [data-testid="stImage"] img {
        display: block;
        width: 100%;
        height: 100%;
        object-fit: contain;
    }
    /* Same GPU-layer trick for the header logos, which also wobble in
       Chrome but not Firefox. */
    .header-logo img {
        transform: translateZ(0);
        backface-visibility: hidden;
    }
    /* Kill ALL transitions / animations globally (not just inside .stApp)
       since Streamlit injects elements outside that scope as well. */
    *, *::before, *::after {
        animation-duration: 0s !important;
        animation-delay: 0s !important;
        transition-duration: 0s !important;
        transition-delay: 0s !important;
        animation-iteration-count: 1 !important;
    }
    .vp-card {
        border-left: 4px solid var(--vp-color, #3b82f6);
        padding: 1rem 1.2rem;
        background: rgba(59,130,246,0.04);
        border-radius: 0 8px 8px 0;
        min-height: 120px;
        height: 100%;
    }
    .vp-tag {
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: var(--vp-color, #3b82f6);
        font-weight: 700;
        margin-bottom: 0.4rem;
    }
    .vp-headline {
        font-size: 1.05rem;
        font-weight: 600;
        color: #e5e7eb;
        margin-bottom: 0.3rem;
    }
    .vp-body {
        color: #9ca3af;
        font-size: 0.82rem;
        line-height: 1.55;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Header - both logos in a single flex container so they share the same
# vertical baseline. No st.columns -> no float -> no offset drift.
# ---------------------------------------------------------------------------
st.markdown(
    f'<div class="header-logo" style="display:flex; align-items:center; '
    f'justify-content:space-between; gap:1rem; '
    f'padding:0.2rem 0 0.6rem; min-height:104px;">'
    f'<img src="data:image/png;base64,{_FUSE_LOGO}" width="180" height="100" '
    f'style="display:block; object-fit:contain;" />'
    f'<a href="https://forgottenforge.xyz" target="_blank" '
    f'style="display:flex; align-items:center;">'
    f'<img src="data:image/png;base64,{_FF_LOGO}" width="100" height="100" '
    f'style="display:block; border-radius:8px; object-fit:contain;" />'
    f'</a>'
    f'</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<p style="font-size:1.4rem; font-weight:700; margin-bottom:0.15rem; '
    'letter-spacing:-0.01em;">Find the exact value where your system breaks.</p>'
    '<p style="font-size:0.95rem; color:#9ca3af; margin-top:0;">'
    'Nonparametric tipping-point detection with bootstrap CI &amp; permutation tests. '
    'Based on: <a href="https://doi.org/10.1116/5.0312410" '
    'target="_blank" style="color:#60a5fa; text-decoration:none;">'
    'M. C. Wurm, <em>AVS Quantum Sci.</em> <strong>8</strong>, 013804 (2026)</a>. '
    'No ML &mdash; just statistics.'
    '</p>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Shared state: do we already have data uploaded or a demo loaded?
# ---------------------------------------------------------------------------
_has_data = (
    "_uploaded_df_stash" in st.session_state
    or "demo_df" in st.session_state
)


# ---------------------------------------------------------------------------
# Hero: interactive slider against a synthetic Server Load Test dataset.
# Only shown on landing (no data yet).
# ---------------------------------------------------------------------------
def _render_to_png(result):
    """Render a result card to PNG bytes. dpi=80 keeps the file size low
    so Chrome decodes it in a single paint frame (no progressive load /
    no layout-shift cascade)."""
    fig = render_card(result)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80,
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _png_to_html_img(png_bytes, max_width_px=None):
    """Inline a PNG as an HTML <img> with EXPLICIT pixel dimensions so
    Chrome reserves the exact final box before decoding. This bypasses
    Streamlit's aspect-ratio container which was still leaving room for
    a sub-pixel reflow on image decode."""
    b64 = base64.b64encode(png_bytes).decode()
    style = (
        "display:block; width:100%; height:auto; "
        "transform:translateZ(0); backface-visibility:hidden;"
    )
    if max_width_px is not None:
        style += f" max-width:{max_width_px}px;"
    # Aspect-ratio attributes are honoured by all modern browsers and let
    # Chrome reserve the final height before the bytes arrive.
    return (
        f'<img src="data:image/png;base64,{b64}" '
        f'width="1120" height="640" '
        f'style="{style}" />'
    )


@st.cache_data(show_spinner=False)
def _hero_png(current_x):
    """Cache the hero PNG by slider value. After the first visit at a given
    slider position, subsequent renders are an instant cache hit -- no
    matplotlib reflow, no iframe wobble."""
    rq = np.linspace(100, 10000, 100)
    lat = 10 + 50 / (1 - np.clip(rq / 7500, 0, 0.98))
    lat += np.random.default_rng(7).normal(0, 5, 100)
    r = analyze(rq, lat,
                current_x=current_x,
                x_name="Concurrent Requests",
                y_name="Response Time (ms)",
                label="Server Load Test",
                n_boot=200, n_perm=500)
    return _render_to_png(r)


if not _has_data:
    hero_left, hero_right = st.columns([2, 3], gap="large")
    with hero_left:
        st.markdown(
            '<p style="font-size:1.05rem; color:#e0e0e0; line-height:1.6; '
            'margin-top:0.5rem;">'
            'Drag the slider. Watch the score change as you approach the '
            'tipping point.'
            '</p>',
            unsafe_allow_html=True,
        )
        hero_x = st.slider(
            "Your operating point (req/s)",
            min_value=500, max_value=9500, value=5000, step=100,
            key="_hero_slider",
        )
        if hero_x < 5000:
            st.success(f"**{hero_x} req/s** - Safe. Plenty of room before the cliff.")
        elif hero_x < 6500:
            st.info(f"**{hero_x} req/s** - Getting closer. Plan capacity now.")
        elif hero_x < 7200:
            st.warning(f"**{hero_x} req/s** - Danger zone. Add servers or throttle.")
        else:
            st.error(f"**{hero_x} req/s** - Past the tipping point. System is degrading.")
        st.markdown(
            '<p style="font-size:0.85rem; color:#6b7280; margin-top:0.5rem;">'
            'This is real analysis running live - the same engine that '
            'analyzes quantum phase transitions, reactor erosion, and drug '
            'toxicity curves.'
            '</p>',
            unsafe_allow_html=True,
        )
    with hero_right:
        hero_png = _hero_png(hero_x)
        st.markdown(_png_to_html_img(hero_png), unsafe_allow_html=True)

    st.divider()


# ---------------------------------------------------------------------------
# Value props strip (only on landing)
# ---------------------------------------------------------------------------
def _vp_card(tag, headline, body, color):
    st.markdown(
        f'<div class="vp-card" style="--vp-color:{color};">'
        f'<div class="vp-tag">{tag}</div>'
        f'<div class="vp-headline">{headline}</div>'
        f'<div class="vp-body">{body}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


if not _has_data:
    vp1, vp2, vp3 = st.columns(3)
    with vp1:
        _vp_card("Validated", "6 domains tested",
                 "Physics, fusion, chemistry, biology, ML, infrastructure. "
                 "Each verified to find a known tipping point.",
                 "#3b82f6")
    with vp2:
        _vp_card("Honest", "Statistical, not heuristic",
                 "Permutation tests, bootstrap CIs, no curve-fitting. "
                 "Tells you when there's no signal.",
                 "#10b981")
    with vp3:
        _vp_card("Open", "Paper + code + DOI",
                 "Backed by a Zenodo-archived foundation paper. "
                 "Library is dual-licensed (AGPL or commercial).",
                 "#f59e0b")
    st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Showcase: 3 example analyses (visible, not hidden in an expander)
# Rendered at the library's native (14, 8) figsize so the gauge label
# does not overlap with the score on small layouts.
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _showcase_pngs():
    """Cache the showcase mini-cards as pre-rendered PNG bytes. After the
    first cache fill they reload instantly with no matplotlib reflow --
    eliminates the Chrome-specific layout wobble on landing."""
    rng = np.random.default_rng(42)

    B = np.linspace(0.5, 4.0, 80)
    ent = 0.2 + 3.0 / (1 + np.exp(-8 * (B - 2.3))) + rng.normal(0, 0.1, 80)
    r1 = analyze(B, ent, current_x=1.9,
                 x_name="Magnetic Field (T)",
                 y_name="Entanglement Entropy",
                 label="Quantum Phase Transition",
                 n_boot=200, n_perm=500)

    dose = np.linspace(0, 100, 80)
    surv = np.clip(100 - 90 / (1 + np.exp(-0.2 * (dose - 42)))
                   + rng.normal(0, 2, 80), 0, 100)
    r2 = analyze(dose, surv, current_x=30,
                 x_name="Dose (mg/L)", y_name="Cell Survival (%)",
                 label="Cytotoxicity Assay",
                 n_boot=200, n_perm=500)

    tf = np.linspace(400, 1600, 100)
    ero = (0.5 + 20.0 / (1 + np.exp(-0.03 * (tf - 1180)))
           + rng.normal(0, 0.3, 100))
    r3 = analyze(tf, ero, current_x=1100,
                 x_name="Plasma Temp (deg C)",
                 y_name="Erosion (um/pulse)",
                 label="Fusion Reactor Erosion",
                 n_boot=200, n_perm=500)
    return _render_to_png(r1), _render_to_png(r2), _render_to_png(r3)


if not _has_data:
    st.markdown(
        "<div style='margin-top:0.6rem; margin-bottom:0.4rem; "
        "font-size:0.9rem; color:#9ca3af; font-weight:600; "
        "letter-spacing:0.05em; text-transform:uppercase;'>"
        "See it in action - 3 example analyses"
        "</div>",
        unsafe_allow_html=True,
    )
    p1, p2, p3 = _showcase_pngs()
    sc1, sc2, sc3 = st.columns(3)
    for col, png, caption in (
        (sc1, p1, "Physics - quantum phase transition at 2.27 T"),
        (sc2, p2, "Biology - toxicity cliff at 42 mg/L"),
        (sc3, p3, "Fusion - erosion spike at 1176 deg C"),
    ):
        with col:
            st.markdown(_png_to_html_img(png), unsafe_allow_html=True)
            st.caption(caption)

    st.divider()


# ---------------------------------------------------------------------------
# Upload + Demo selector
# ---------------------------------------------------------------------------
uploaded = st.file_uploader(
    "Upload CSV / TSV / JSON",
    type=["csv", "tsv", "txt", "json"],
    key="_uploader",
    label_visibility="collapsed" if _has_data else "visible",
)

if uploaded is None and "demo_df" not in st.session_state:
    st.markdown(
        '<p style="text-align:center; color:#6b7280; font-size:0.9rem; '
        'margin:0.8rem 0;">or try a demo</p>',
        unsafe_allow_html=True,
    )
    DEMOS = [
        "Server Load Test",
        "ML Training Stability",
        "Quantum Phase Transition",
        "Fusion Reactor Erosion",
        "Chemistry - Reaction Kinetics",
        "Biology - Cytotoxicity",
    ]
    dleft, dright = st.columns([3, 1])
    with dleft:
        demo_choice = st.selectbox(
            "Choose a demo",
            DEMOS,
            label_visibility="collapsed",
            key="demo_choice",
        )
    with dright:
        load_demo = st.button("Load demo", use_container_width=True)
    if load_demo:
        rng = np.random.default_rng(42)
        if demo_choice == "Server Load Test":
            rq = np.linspace(100, 10000, 100)
            df_demo = pd.DataFrame({
                "concurrent_requests": rq,
                "response_time_ms": (10 + 50 / (1 - np.clip(rq/7500, 0, 0.98))
                                     + rng.normal(0, 5, 100)),
                "cpu_percent": (5 + 90 / (1 + np.exp(-0.002 * (rq - 6000)))
                                + rng.normal(0, 2, 100)),
                "memory_mb": 20 + 0.005 * rq + rng.normal(0, 3, 100),
                "error_rate": np.clip(
                    0.1 + 50 / (1 + np.exp(-0.003 * (rq - 8000)))
                    + rng.normal(0, 0.5, 100), 0, None),
            })
        elif demo_choice == "ML Training Stability":
            lr = np.linspace(0.001, 0.1, 100)
            df_demo = pd.DataFrame({
                "learning_rate": lr,
                "gradient_norm": (1.0 + 50 / (1 + np.exp(-200 * (lr - 0.052)))
                                  + rng.normal(0, 0.3, 100)),
                "training_loss": (0.1 + 0.8 / (1 + np.exp(-180 * (lr - 0.055)))
                                  + rng.normal(0, 0.015, 100)),
                "validation_accuracy": np.clip(
                    0.95 - 0.6 / (1 + np.exp(-180 * (lr - 0.054)))
                    + rng.normal(0, 0.01, 100), 0, 1),
            })
        elif demo_choice == "Quantum Phase Transition":
            B = np.linspace(0.5, 4.0, 80)
            df_demo = pd.DataFrame({
                "magnetic_field_T": B,
                "entanglement_entropy": (0.2 + 3.0 / (1 + np.exp(-8 * (B - 2.3)))
                                         + rng.normal(0, 0.1, 80)),
                "magnetization": np.clip(
                    1.0 - 0.9 / (1 + np.exp(-10 * (B - 2.3)))
                    + rng.normal(0, 0.05, 80), 0, 1),
                "susceptibility_chi": (0.5 + 8.0 / (1 + np.exp(-6 * (B - 2.35)))
                                       + rng.normal(0, 0.2, 80)),
            })
        elif demo_choice == "Fusion Reactor Erosion":
            tf = np.linspace(400, 1600, 100)
            df_demo = pd.DataFrame({
                "plasma_temperature_C": tf,
                "erosion_rate_um_per_pulse": (0.5 + 20.0 / (1 + np.exp(-0.03 * (tf - 1180)))
                                              + rng.normal(0, 0.3, 100)),
                "surface_roughness_nm": (5.0 + 40.0 / (1 + np.exp(-0.025 * (tf - 1200)))
                                         + rng.normal(0, 1.0, 100)),
                "deuterium_retention": (0.1 + 2.0 / (1 + np.exp(-0.02 * (tf - 1100)))
                                        + rng.normal(0, 0.05, 100)),
            })
        elif demo_choice == "Chemistry - Reaction Kinetics":
            tc = np.linspace(20, 200, 100)
            df_demo = pd.DataFrame({
                "temperature_C": tc,
                "reaction_rate_mol_s": (0.01 + 5.0 / (1 + np.exp(-0.15 * (tc - 150)))
                                        + rng.normal(0, 0.06, 100)),
                "yield_percent": np.clip(
                    95 - 60 / (1 + np.exp(-0.12 * (tc - 155)))
                    + rng.normal(0, 1.5, 100), 0, 100),
                "viscosity_mPas": (100 - 80 / (1 + np.exp(-0.1 * (tc - 80)))
                                   + rng.normal(0, 2, 100)),
            })
        else:  # Biology - Cytotoxicity
            dose = np.linspace(0, 100, 80)
            df_demo = pd.DataFrame({
                "dose_mg_L": dose,
                "cell_survival_pct": np.clip(
                    100 - 90 / (1 + np.exp(-0.2 * (dose - 42)))
                    + rng.normal(0, 2, 80), 0, 100),
                "biomarker_level": (1.0 + 8.0 / (1 + np.exp(-0.15 * (dose - 45)))
                                    + rng.normal(0, 0.2, 80)),
                "inflammation_index": (0.5 + 4.0 / (1 + np.exp(-0.2 * (dose - 35)))
                                       + rng.normal(0, 0.15, 80)),
            })
        st.session_state["demo_df"] = df_demo
        print(f"[fuse-hf] demo loaded: {demo_choice}",
              file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Parse upload (cached in session_state stash)
# ---------------------------------------------------------------------------
if uploaded is not None:
    stash = st.session_state.get("_uploaded_df_stash") or {}
    if stash.get("name") != uploaded.name or stash.get("size") != uploaded.size:
        try:
            if uploaded.name.endswith(".json"):
                import json
                df_new = parse_json(json.load(uploaded))
            else:
                sep = "\t" if uploaded.name.endswith(".tsv") else ","
                df_new = pd.read_csv(uploaded, sep=sep)
        except Exception:
            st.error(f"Could not parse file:\n\n```\n{traceback.format_exc()}\n```")
            st.stop()
        st.session_state["_uploaded_df_stash"] = {
            "name": uploaded.name,
            "size": uploaded.size,
            "df": df_new,
        }
        # New upload overrides demo
        st.session_state.pop("demo_df", None)
        print(f"[fuse-hf] upload parsed: {uploaded.name} shape={df_new.shape}",
              file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# If still nothing loaded, stop here (landing already rendered above)
# ---------------------------------------------------------------------------
if "demo_df" not in st.session_state and "_uploaded_df_stash" not in st.session_state:
    st.stop()


# ---------------------------------------------------------------------------
# Pick the active dataframe (upload wins over demo)
# ---------------------------------------------------------------------------
if "_uploaded_df_stash" in st.session_state:
    df = st.session_state["_uploaded_df_stash"]["df"]
else:
    df = st.session_state["demo_df"]


# ---------------------------------------------------------------------------
# Column validation + preview
# ---------------------------------------------------------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) < 2:
    st.error(f"Need at least 2 numeric columns. Found: {numeric_cols}")
    st.stop()
if len(df) < 8:
    st.error(f"Need at least 8 rows; got {len(df)}.")
    st.stop()

with st.expander(
    f"Data preview ({len(df)} rows, {len(numeric_cols)} numeric columns)",
    expanded=False,
):
    st.dataframe(df.head(30), use_container_width=True)


# ---------------------------------------------------------------------------
# X-axis pick
# ---------------------------------------------------------------------------
detected_x = detect_x_column(df, numeric_cols)
x_col = st.selectbox(
    "X-axis (independent variable)",
    numeric_cols,
    index=numeric_cols.index(detected_x),
    help="Auto-detected; change if needed.",
)
y_cols = [c for c in numeric_cols if c != x_col]


# ---------------------------------------------------------------------------
# Scan all y-columns
# ---------------------------------------------------------------------------
def _scan_one(df, x_col, y_col):
    try:
        r = analyze(df, x=x_col, y=y_col, n_boot=500, n_perm=1000)
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


bar = st.progress(0, text="Scanning columns...")
scan = []
for i, y in enumerate(y_cols):
    bar.progress((i + 1) / len(y_cols), text=f"Analyzing {y} ({i+1}/{len(y_cols)})")
    scan.append(_scan_one(df, x_col, y))
bar.empty()

scan.sort(key=lambda r: (
    -(r["significant"] if r["score"] is not None else False),
    -(r["score"] or 0)
))
print(f"[fuse-hf] scan complete: {len(scan)} cols",
      file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Two-column results layout
# ---------------------------------------------------------------------------
selected_y = st.session_state.get("selected_y")
if selected_y and selected_y not in y_cols:
    selected_y = None
    st.session_state.pop("selected_y", None)

GRADE_EMOJI = {
    "STABLE":   "\U0001F7E2",  # green circle
    "MODERATE": "\U0001F7E1",  # yellow circle
    "WARNING":  "\U0001F7E0",  # orange circle
    "CRITICAL": "\U0001F534",  # red circle
    "ERROR":    "⚫",      # black circle
}

col_list, col_card = st.columns([1, 2], gap="large")

with col_list:
    n_ok = sum(1 for r in scan if r["score"] is not None)
    st.subheader(f"{n_ok} of {len(y_cols)} columns scanned")

    for i, res in enumerate(scan):
        if res["score"] is None:
            continue
        is_sel = (selected_y == res["y_column"])
        with st.container():
            r_score, r_info, r_btn = st.columns([0.8, 3, 1.2])
            with r_score:
                emoji = GRADE_EMOJI.get(res["grade"], "")
                st.markdown(f"### {emoji} {res['score']}")
            with r_info:
                st.markdown(f"**{res['y_column']}**")
                sig = " *" if res["significant"] else ""
                tp = (f"TP: **{res['tipping_point']:.4g}**"
                      if res["tipping_point"] is not None else "No TP")
                st.caption(
                    f"{res['grade']}{sig} | k={res['kappa']:.1f} | "
                    f"p={res['p_value']:.3f} | {tp}"
                )
            with r_btn:
                if st.button(
                    "View",
                    key=f"view_{i}",
                    use_container_width=True,
                    type="primary" if is_sel else "secondary",
                ):
                    st.session_state["selected_y"] = res["y_column"]
                    selected_y = res["y_column"]
                    print(f"[fuse-hf] selected {res['y_column']!r}",
                          file=sys.stderr, flush=True)
        st.divider()

    errs = [r for r in scan if r["score"] is None]
    if errs:
        with st.expander(f"{len(errs)} columns failed"):
            for er in errs:
                st.caption(f"**{er['y_column']}** - {er.get('error', 'Unknown')}")


with col_card:
    print(f"[fuse-hf] col_card render, selected_y={selected_y!r}",
          file=sys.stderr, flush=True)

    if selected_y is None:
        st.caption("Click 'View' on a column to render its Stability Card.")
    else:
        st.subheader(f"Stability Card: {selected_y}")

        try:
            result = analyze(df, x=x_col, y=selected_y,
                             n_boot=1000, n_perm=2000)
        except Exception:
            st.error(f"analyze() failed:\n\n```\n{traceback.format_exc()}\n```")
            st.stop()

        st.metric("Score", f"{result.score} / 100", delta=result.grade)
        st.markdown(f"**{result.diagnosis}**")

        try:
            fig = render_card(result)
            st.pyplot(fig, use_container_width=True)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=200,
                        facecolor=fig.get_facecolor(), edgecolor="none")
            buf.seek(0)
            plt.close(fig)
            st.download_button(
                "Download this Card",
                data=buf,
                file_name=f"stability_{selected_y}.png",
                mime="image/png",
                use_container_width=True,
            )
        except Exception:
            st.error(f"render_card failed:\n\n```\n{traceback.format_exc()}\n```")
            st.stop()

        with st.expander("Full metrics"):
            st.code(result.summary())


# ---------------------------------------------------------------------------
# Reference block - always available, also on results pages.
# Restored 2026-06-16 (was missing from the HF Space version compared to
# the local fuse-ui). expanded=True so the explanations are visible
# without a click.
# ---------------------------------------------------------------------------
st.divider()
with st.expander(
    "What the results mean — grades, tested domains & recommendations",
    expanded=True,
):
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
| **Physics** — B-field | **2.27 T** (CI: 2.18–2.45) | 73 | Keep below 2.0 T. Hunting the transition? Sweep 2.1–2.5 T. |
| **Fusion** — Plasma temp | **1176°C** (CI: 1152–1212) | 73 | Hard limit at 1060°C. **Increase coolant** or **cut plasma power**. |
| **Chemistry** — Reaction temp | **151°C** (CI: 142–155) | 76 | Reactor limit at 136°C. Past it? **Shutdown**, cut the feed. |
| **Biology** — Drug dose | **42 mg/L** (CI: 38–48) | 56 | Safe max: 38 mg/L. Need more effect? Try combination therapy. |
| **ML** — Learning rate | **0.053** (CI: 0.048–0.056) | 73 | Set lr to 0.04. Past 0.06? **Stop**, lower to 0.03, restart. |
| **Infra** — Requests | **7200 req/s** (CI: 7000–7300) | 85 | Autoscale at 5700. Past TP? **Shed load** — rate-limit, serve cached. |

**Rule of thumb:** Keep your operating point at **80–90% of the tipping point**. That's your real limit.

---

**Works with** any CSV/TSV/JSON, 2+ numeric columns, 8+ rows.
**Won't work** on pure noise, linear trends with no cliff, or fewer than 8 points.
""")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.markdown(
    f'<div style="display:flex; align-items:center; '
    f'justify-content:space-between; padding:0.4rem 0;">'
    f'<img src="data:image/png;base64,{_FUSE_LOGO}" height="36" '
    f'style="opacity:0.75;" />'
    f'<span style="color:#9ca3af; font-size:0.82rem; text-align:center;">'
    f'(c) 2026 Forgotten Forge &nbsp;&bull;&nbsp; AGPL-3.0 &nbsp;&bull;&nbsp; '
    f'<a href="https://forgottenforge.xyz" target="_blank" '
    f'style="color:#60a5fa; text-decoration:none;">forgottenforge.xyz</a>'
    f'<br/>'
    f'<span style="font-size:0.75rem; color:#6b7280;">'
    f'fusepoint {fusepoint.__version__} &nbsp;|&nbsp; '
    f'<a href="https://doi.org/10.5281/zenodo.20548818" target="_blank" '
    f'style="color:#60a5fa; text-decoration:none;">'
    f'paper: doi:10.5281/zenodo.20548818</a> &nbsp;|&nbsp; '
    f'<a href="https://github.com/forgottenforge/fusepoint" target="_blank" '
    f'style="color:#60a5fa; text-decoration:none;">source</a>'
    f'</span>'
    f'</span>'
    f'<a href="https://forgottenforge.xyz" target="_blank">'
    f'<img src="data:image/png;base64,{_FF_LOGO}" height="44" '
    f'style="border-radius:6px; opacity:0.75;" />'
    f'</a>'
    f'</div>',
    unsafe_allow_html=True,
)
