"""
Fuse Report — the publication-quality visualization.

Redesigned for clarity and shareability:
  - Light background (works in Slack, papers, presentations)
  - Clean gauge without segmentation artifacts
  - Subtle zone indicators (not competing with data)
  - Actionable recommendation text in the user's language
  - Column names on axes, not "Parameter" / "Observable"
  - Smart rounding (2-3 significant figures, not 5 decimals)
  - Fuse / Forgotten Forge branding
"""

from __future__ import annotations
import math
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image


# ---------------------------------------------------------------------------
# Color palette — Light theme
# ---------------------------------------------------------------------------
C_BG       = "#ffffff"
C_CARD     = "#f8f9fb"
C_TEXT     = "#1a1a2e"
C_DIM      = "#6b7280"
C_LIGHT    = "#d1d5db"
C_GREEN    = "#059669"
C_YELLOW   = "#d97706"
C_ORANGE   = "#ea580c"
C_RED      = "#dc2626"
C_BLUE     = "#2563eb"
C_PURPLE   = "#7c3aed"
C_GRID     = "#e5e7eb"
C_SAFE_BG  = "#ecfdf5"
C_WARN_BG  = "#fffbeb"
C_CRIT_BG  = "#fef2f2"

# Brand colors (from Forgotten Forge / Fuse logos)
C_BRAND_DARK  = "#1a1a1e"
C_BRAND_LIGHT = "#f0f0f0"

GRADE_COLORS = {
    "STABLE":   C_GREEN,
    "MODERATE": C_YELLOW,
    "WARNING":  C_ORANGE,
    "CRITICAL": C_RED,
}

ZONE_CMAP = LinearSegmentedColormap.from_list(
    "stability", [C_GREEN, C_YELLOW, C_ORANGE, C_RED], N=256
)


def _score_color(score):
    if score >= 85:
        return C_GREEN
    if score >= 60:
        return C_YELLOW
    if score >= 35:
        return C_ORANGE
    return C_RED


def _clean_label(label):
    """Clean up nested JSON path names into readable labels.

    'results.Collatz_3np1_cycle.gamma_values.gamma_M' → 'Gamma M'
    'results.3np1_single_step.gamma_values.M' → 'M'
    'temperature_c' → 'Temperature C'
    """
    # Take only the last segment of dotted paths
    if "." in label:
        label = label.rsplit(".", 1)[-1]
    # Replace underscores/hyphens with spaces, then title-case
    label = label.replace("_", " ").replace("-", " ").strip()
    if label == label.lower() or label == label.upper():
        label = label.title()
    return label


def _truncate_label(label, max_len=28):
    """Clean and shorten long column names for axis labels."""
    label = _clean_label(label)
    if len(label) <= max_len:
        return label
    # Try to truncate at a word boundary
    trunc = label[:max_len - 1]
    last_space = trunc.rfind(" ")
    if last_space > max_len // 2:
        trunc = trunc[:last_space]
    return trunc + "\u2026"


# ---------------------------------------------------------------------------
# Logo / branding helpers
# ---------------------------------------------------------------------------
_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")


def _load_logo(name, height_px=28):
    """Load a logo from assets, remove dark bg, crop to content."""
    path = os.path.join(_ASSETS_DIR, name)
    if not os.path.exists(path):
        return None
    try:
        img = Image.open(path).convert("RGBA")
        arr = np.array(img)

        # Make dark background transparent (pixels darker than ~25%)
        brightness = arr[:, :, :3].max(axis=2)
        arr[brightness < 64, 3] = 0

        # Crop to non-transparent bounding box
        alpha = arr[:, :, 3]
        rows = np.any(alpha > 0, axis=1)
        cols = np.any(alpha > 0, axis=0)
        if not rows.any():
            return None
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        arr = arr[r0:r1+1, c0:c1+1]

        # Scale to target height
        cropped = Image.fromarray(arr)
        aspect = cropped.width / cropped.height
        new_w = int(height_px * aspect)
        cropped = cropped.resize((new_w, height_px), Image.LANCZOS)
        return np.array(cropped)
    except Exception:
        return None


def _draw_brand_footer(fig):
    """Draw the Fuse / Forgotten Forge brand bar at the bottom."""
    footer = fig.add_axes([0, 0, 1, 0.055])
    footer.set_xlim(0, 1)
    footer.set_ylim(0, 1)
    footer.set_facecolor(C_BRAND_DARK)
    footer.set_zorder(10)
    footer.set_xticks([])
    footer.set_yticks([])
    for spine in footer.spines.values():
        spine.set_visible(False)

    # FUSE — left
    footer.text(0.03, 0.5, "FUSE", ha="left", va="center",
                fontsize=18, fontweight="bold", color="#ffffff",
                fontfamily="sans-serif")

    # Centered: brand + license
    footer.text(0.5, 0.5,
                "\u00a9 Forgotten Forge  \u2022  AGPL-3.0  \u2022  forgottenforge.xyz",
                ha="center", va="center", fontsize=7.5,
                color="#aaaaaa", fontfamily="monospace")

    # FORGOTTEN FORGE — right
    footer.text(0.97, 0.5, "FORGOTTEN FORGE",
                ha="right", va="center",
                fontsize=11, fontweight="bold", color="#ffffff",
                fontfamily="sans-serif")


# ---------------------------------------------------------------------------
# Smart rounding — human-friendly numbers
# ---------------------------------------------------------------------------

def _smart_round(value, sig_figs=2):
    """Round to N significant figures, producing human-friendly numbers.

    _smart_round(6480, 2) → 6500
    _smart_round(0.03097, 2) → 0.031
    _smart_round(7200, 2) → 7200
    _smart_round(0.04459, 2) → 0.045
    """
    if value == 0:
        return 0
    d = math.ceil(math.log10(abs(value)))
    power = sig_figs - d
    factor = 10 ** power
    return round(value * factor) / factor


def _fmt(value, sig_figs=2):
    """Format a number with smart rounding, dropping unnecessary decimals."""
    rounded = _smart_round(value, sig_figs)
    # Use 'g' format to avoid trailing zeros but keep precision
    if abs(rounded) >= 1:
        return f"{rounded:g}"
    else:
        # For small numbers, show enough decimals
        decimals = max(0, sig_figs - math.floor(math.log10(abs(rounded))) - 1) if rounded != 0 else 1
        return f"{rounded:.{decimals}f}"


# ---------------------------------------------------------------------------
# Actionable recommendation generator — in the user's language
# ---------------------------------------------------------------------------

def _generate_recommendation(result):
    """Generate a human-readable, actionable recommendation using column names."""
    score = result.score
    critical_x = result.critical_x
    current_x = result.current_x
    margin = result.safety_margin
    p = result.p_value
    kappa = result.kappa
    xn = _clean_label(result.x_name)
    yn = _clean_label(result.y_name)

    cx_str = _fmt(critical_x)
    cur_str = _fmt(current_x) if current_x is not None else None

    if p >= 0.05:
        if kappa > 3.0 and p < 0.20:
            return (f"A possible pattern was found near {xn} = {cx_str}, but the data "
                    f"is too noisy to be certain (p = {p:.2f}). "
                    f"Try adding more data points or reducing measurement noise.")
        return (f"No reliable tipping point found in this data. "
                f"The variations in {yn} are consistent with random noise. "
                f"Collect more data or verify the relationship between {xn} and {yn}.")

    # Significant tipping point
    alert_val = _fmt(critical_x * 0.9)
    margin_pct = f"{margin * 100:.0f}%"

    if score >= 85:
        if cur_str:
            return (f"System is stable. {yn} tips at {xn} = {cx_str}. "
                    f"Currently at {cur_str} — {margin_pct} safety margin. "
                    f"Recommendation: set alert at {alert_val}.")
        return (f"Clear tipping point at {xn} = {cx_str} with high confidence. "
                f"Safety margin: {margin_pct}. "
                f"Recommendation: stay below {alert_val}.")
    elif score >= 60:
        keep_below = _fmt(critical_x * 0.8)
        if cur_str:
            return (f"{yn} changes sharply at {xn} = {cx_str}. "
                    f"Currently at {cur_str} — {margin_pct} from the edge. "
                    f"Recommendation: keep {xn} below {keep_below}.")
        return (f"Tipping point at {xn} = {cx_str} with moderate confidence. "
                f"Safety margin: {margin_pct}. Consider adding headroom.")
    else:
        reduce_to = _fmt(critical_x * 0.7)
        if cur_str:
            return (f"Warning: operating close to tipping point at {xn} = {cx_str}. "
                    f"Current {xn} ({cur_str}) is only {margin_pct} away. "
                    f"Immediate action recommended: reduce to {reduce_to} or below.")
        return (f"Tipping point at {xn} = {cx_str}. Low safety margin ({margin_pct}). "
                f"Reduce {xn} or increase system resilience.")


# ---------------------------------------------------------------------------
# Clean gauge (smooth arc, no segmentation)
# ---------------------------------------------------------------------------

def _draw_gauge(ax, score, grade):
    """Draw a semi-circular gauge with colored needle and score below."""
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.95, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")

    n_pts = 500
    r_outer, r_inner = 1.0, 0.72
    score_frac = np.clip(score / 100, 0, 1)
    needle_color = _score_color(score)

    # Background arc — full range, subtle
    theta_bg = np.linspace(np.pi, 0, n_pts)
    for i in range(len(theta_bg) - 1):
        t0, t1 = theta_bg[i], theta_bg[i + 1]
        frac = 1.0 - t0 / np.pi
        color = ZONE_CMAP(frac)
        xs = [r_inner * np.cos(t0), r_inner * np.cos(t1),
              r_outer * np.cos(t1), r_outer * np.cos(t0)]
        ys = [r_inner * np.sin(t0), r_inner * np.sin(t1),
              r_outer * np.sin(t1), r_outer * np.sin(t0)]
        ax.fill(xs, ys, color=color, alpha=0.12, linewidth=0)

    # Filled arc up to score
    theta_fill = np.linspace(np.pi, np.pi * (1 - score_frac), n_pts)
    for i in range(len(theta_fill) - 1):
        t0, t1 = theta_fill[i], theta_fill[i + 1]
        frac = 1.0 - t0 / np.pi
        color = ZONE_CMAP(frac)
        xs = [r_inner * np.cos(t0), r_inner * np.cos(t1),
              r_outer * np.cos(t1), r_outer * np.cos(t0)]
        ys = [r_inner * np.sin(t0), r_inner * np.sin(t1),
              r_outer * np.sin(t1), r_outer * np.sin(t0)]
        ax.fill(xs, ys, color=color, alpha=0.9, linewidth=0)

    # Tick marks (0, 20, 40, 60, 80, 100)
    for tick_val in [0, 20, 40, 60, 80, 100]:
        t = np.pi * (1 - tick_val / 100)
        x0 = r_outer * np.cos(t)
        y0 = r_outer * np.sin(t)
        x1 = (r_outer + 0.08) * np.cos(t)
        y1 = (r_outer + 0.08) * np.sin(t)
        ax.plot([x0, x1], [y0, y1], color=C_LIGHT, linewidth=1.2, zorder=5)
        lx = (r_outer + 0.19) * np.cos(t)
        ly = (r_outer + 0.19) * np.sin(t)
        ax.text(lx, ly, str(tick_val), ha="center", va="center",
                fontsize=7, color=C_DIM)

    # Needle — colored to match the score zone
    needle_theta = np.pi * (1 - score_frac)
    needle_len = r_outer + 0.04
    nx = needle_len * np.cos(needle_theta)
    ny = needle_len * np.sin(needle_theta)
    # Triangular needle
    perp = needle_theta + np.pi / 2
    base_w = 0.05
    tri_x = [nx,
             base_w * np.cos(perp), -base_w * np.cos(perp)]
    tri_y = [ny,
             base_w * np.sin(perp), -base_w * np.sin(perp)]
    ax.fill(tri_x, tri_y, color=needle_color, linewidth=0, zorder=8,
            alpha=0.9)
    # Center hub — same color
    ax.plot(0, 0, "o", color=needle_color, markersize=7,
            markeredgecolor="white", markeredgewidth=1.0, zorder=9)

    # Score number — centered below the arc
    ax.text(0, -0.22, str(score), ha="center", va="top",
            fontsize=42, fontweight="bold", color=needle_color)

    # Grade label — clearly separated below score
    ax.text(0, -0.85, grade, ha="center", va="top",
            fontsize=11, fontweight="bold", fontfamily="sans-serif",
            color=GRADE_COLORS.get(grade, C_TEXT))


# ---------------------------------------------------------------------------
# Stability map — clean, subtle zones, with user's column names
# ---------------------------------------------------------------------------

def _draw_stability_map(ax, result, compact=False):
    """Draw the observable with subtle zone indication."""
    x, y, chi = result.x, result.y, result.chi
    critical_x = result.critical_x
    ci_low, ci_high = result.ci

    # Subtle background: green left of critical, red right (only if significant)
    if result.p_value < 0.05:
        ax.axvspan(x[0], critical_x, alpha=0.04, color=C_GREEN, linewidth=0)
        ax.axvspan(critical_x, x[-1], alpha=0.04, color=C_RED, linewidth=0)

        # CI band
        ax.axvspan(ci_low, ci_high, alpha=0.10, color=C_RED,
                   label="95% CI", zorder=1)
        # Critical line
        ax.axvline(critical_x, color=C_RED, linewidth=1.8, linestyle="--",
                   alpha=0.7, label="Tipping point", zorder=4)

    # Observable curve
    ax.plot(x, y, color=C_BLUE, linewidth=2.2, label=_truncate_label(result.y_name, 22), zorder=3)

    # Current operating point
    if result.current_x is not None:
        y_interp = np.interp(result.current_x, x, y)
        ax.plot(result.current_x, y_interp, "o", color=C_GREEN, markersize=9,
                markeredgecolor="white", markeredgewidth=2, zorder=5,
                label="Current")

    ax.set_xlabel(_truncate_label(result.x_name), fontsize=10, color=C_DIM)
    if not compact:
        ax.set_ylabel(_truncate_label(result.y_name), fontsize=10, color=C_DIM)
    ax.set_facecolor(C_CARD)
    ax.tick_params(colors=C_DIM, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(C_GRID)
    ax.grid(True, alpha=0.3, color=C_GRID, linewidth=0.5)
    ax.legend(fontsize=8, loc="best", framealpha=0.8,
              facecolor=C_BG, edgecolor=C_GRID, labelcolor=C_TEXT)


# ---------------------------------------------------------------------------
# Sensitivity curve (was "Susceptibility")
# ---------------------------------------------------------------------------

def _draw_sensitivity(ax, result):
    """Draw the sensitivity (rate of change) curve."""
    x, chi = result.x, result.chi
    critical_x = result.critical_x

    ax.fill_between(x, 0, chi, alpha=0.15, color=C_PURPLE)
    ax.plot(x, chi, color=C_PURPLE, linewidth=1.8)

    if result.p_value < 0.05:
        ax.axvline(critical_x, color=C_RED, linewidth=1.2,
                   linestyle="--", alpha=0.6)
        peak_y = chi[np.argmin(np.abs(x - critical_x))]
        ax.annotate(f"k = {result.kappa:.1f}",
                    xy=(critical_x, peak_y),
                    xytext=(critical_x + np.ptp(x) * 0.1, peak_y * 0.85),
                    fontsize=9, color=C_TEXT,
                    arrowprops=dict(arrowstyle="->", color=C_DIM, lw=0.8))

    ax.set_xlabel(_truncate_label(result.x_name), fontsize=10, color=C_DIM)
    ax.set_ylabel("Sensitivity", fontsize=10, color=C_DIM)

    # Subtitle explaining what this shows
    ax.text(0.98, 0.96, f"How fast {_truncate_label(result.y_name)} changes",
            transform=ax.transAxes, fontsize=7.5, color=C_LIGHT,
            ha="right", va="top", style="italic")

    ax.set_facecolor(C_CARD)
    ax.tick_params(colors=C_DIM, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(C_GRID)
    ax.grid(True, alpha=0.3, color=C_GRID, linewidth=0.5)


# ---------------------------------------------------------------------------
# Recommendation panel (replaces raw metrics)
# ---------------------------------------------------------------------------

def _draw_recommendation(ax, result, verbose=False):
    """Draw actionable recommendation text, with optional full metrics."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(C_CARD)

    # Recommendation text — the main event
    rec = _generate_recommendation(result)

    ax.text(0.05, 0.95, "Recommendation", fontsize=11, fontweight="bold",
            color=C_TEXT, va="top")

    ax.text(0.05, 0.82, rec, fontsize=9, color=C_TEXT, va="top",
            wrap=True, linespacing=1.4)

    xn = _clean_label(result.x_name)
    if verbose:
        # Full metrics table for experts
        sig_mark = "sig." if result.p_value < 0.05 else "n.s."
        metrics = [
            ("Tipping Point", f"{xn} = {_fmt(result.critical_x, 3)}"),
            ("95% CI", f"{_fmt(result.ci[0], 3)} - {_fmt(result.ci[1], 3)}"),
            ("Sharpness (k)", f"{result.kappa:.2f}"),
            ("p-value", f"{result.p_value:.4f}  {sig_mark}"),
            ("Safety Margin", f"{result.safety_margin * 100:.0f}%"),
        ]
        y_pos = 0.38
        ax.text(0.05, y_pos + 0.05, "Details", fontsize=9, fontweight="bold",
                color=C_DIM, va="top")
        for label, value in metrics:
            ax.text(0.05, y_pos, label, fontsize=8, color=C_DIM, va="top")
            ax.text(0.95, y_pos, value, fontsize=8, color=C_TEXT, va="top",
                    ha="right", fontfamily="monospace")
            y_pos -= 0.07
    else:
        # Compact one-liner for casual viewers
        sig_mark = "sig." if result.p_value < 0.05 else "n.s."
        details = (
            f"{xn}={_fmt(result.critical_x, 3)}  "
            f"CI:{_fmt(result.ci[0], 3)}-{_fmt(result.ci[1], 3)}  "
            f"k={result.kappa:.1f}  "
            f"p={result.p_value:.3f} {sig_mark}  "
            f"margin:{result.safety_margin * 100:.0f}%"
        )
        ax.text(0.05, 0.06, details, fontsize=7.5, color=C_DIM, va="bottom",
                fontfamily="monospace")


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_card(result, figsize=(14, 8), verbose=False):
    """Render a Stability Card and return the matplotlib Figure.

    Parameters
    ----------
    verbose : bool
        If True, show full metrics table alongside the recommendation.
        Default False shows clean recommendation + tiny detail line.
    """
    fig = plt.figure(figsize=figsize, facecolor=C_BG)
    fig.subplots_adjust(left=0.06, right=0.94, top=0.88, bottom=0.12)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30,
                           width_ratios=[1.1, 1.5, 1.3])

    # Title — clean, modern
    fig.text(0.5, 0.96, "Fuse Report",
             ha="center", va="top", fontsize=20, fontweight="bold",
             color=C_TEXT)
    fig.text(0.5, 0.925, result.label,
             ha="center", va="top", fontsize=11, color=C_DIM)

    # Gauge (top-left)
    ax_gauge = fig.add_subplot(gs[0, 0])
    _draw_gauge(ax_gauge, result.score, result.grade)

    # Stability Map (top-center + top-right)
    ax_map = fig.add_subplot(gs[0, 1:])
    _draw_stability_map(ax_map, result)

    # Sensitivity (bottom-left + bottom-center)
    ax_chi = fig.add_subplot(gs[1, :2])
    _draw_sensitivity(ax_chi, result)

    # Recommendation (bottom-right)
    ax_rec = fig.add_subplot(gs[1, 2])
    _draw_recommendation(ax_rec, result, verbose=verbose)

    # Brand footer
    _draw_brand_footer(fig)

    return fig


# ---------------------------------------------------------------------------
# Comparison card (before / after)
# ---------------------------------------------------------------------------

def render_comparison_card(comp, figsize=(15, 9)):
    """Render a before/after comparison card."""
    fig = plt.figure(figsize=figsize, facecolor=C_BG)

    gs = gridspec.GridSpec(3, 2, figure=fig,
                           height_ratios=[1.2, 1.4, 0.5],
                           hspace=0.30, wspace=0.25)
    fig.subplots_adjust(left=0.06, right=0.94, top=0.88, bottom=0.12)

    # Title
    delta = comp.delta_score
    sign = "+" if delta >= 0 else ""
    delta_color = C_GREEN if delta > 0 else (C_RED if delta < 0 else C_DIM)
    fig.text(0.5, 0.97, "Fuse Comparison",
             ha="center", va="top", fontsize=20, fontweight="bold",
             color=C_TEXT)
    fig.text(0.5, 0.935,
             f"{comp.before.score}  \u2192  {comp.after.score}   ({sign}{delta})",
             ha="center", va="top", fontsize=24, fontweight="bold",
             color=delta_color)

    # Before gauge
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title(comp.before.label, fontsize=12, color=C_DIM, pad=6)
    _draw_gauge(ax1, comp.before.score, comp.before.grade)

    # After gauge
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title(comp.after.label, fontsize=12, color=C_DIM, pad=6)
    _draw_gauge(ax2, comp.after.score, comp.after.grade)

    # Before stability map
    ax3 = fig.add_subplot(gs[1, 0])
    _draw_stability_map(ax3, comp.before, compact=True)

    # After stability map
    ax4 = fig.add_subplot(gs[1, 1])
    _draw_stability_map(ax4, comp.after, compact=True)

    # Recommendation text (bottom row, spanning both columns)
    ax_rec = fig.add_subplot(gs[2, :])
    ax_rec.axis("off")
    ax_rec.set_facecolor(C_BG)

    rec_after = _generate_recommendation(comp.after)

    # Compact comparison summary
    if delta > 0:
        summary = (f"Improvement: +{delta} points. "
                   f"After: {rec_after}")
    elif delta < 0:
        summary = (f"Regression: {delta} points. "
                   f"After: {rec_after}")
    else:
        summary = f"No change. {rec_after}"

    ax_rec.text(0.5, 0.7, summary, fontsize=9.5, color=C_TEXT,
                ha="center", va="top", wrap=True, linespacing=1.4,
                transform=ax_rec.transAxes)

    # Brand footer
    _draw_brand_footer(fig)

    return fig
