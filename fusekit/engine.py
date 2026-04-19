"""
Score engine — the mathematical heart of Fuse.

Computes:
- Susceptibility chi(x) = |dO/dx|
- Kappa (peak sharpness): chi_max / mean(chi)
- Bootstrap confidence interval on critical point
- Permutation p-value (signal vs noise)
- Safety margin (distance to tipping point)
- Stability Score 0-100 (universal, self-calibrating)
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter


# ---------------------------------------------------------------------------
# Derivative / Susceptibility
# ---------------------------------------------------------------------------

def _deduplicate(x, y):
    """Average y-values at duplicate x positions."""
    ux, inv = np.unique(x, return_inverse=True)
    if len(ux) == len(x):
        return x, y
    uy = np.zeros(len(ux))
    counts = np.zeros(len(ux))
    for j in range(len(inv)):
        uy[inv[j]] += y[j]
        counts[inv[j]] += 1
    uy /= counts
    return ux, uy


def _adaptive_sigma(n, base_sigma=0.6):
    """Scale kernel sigma with data size so smoothing stays meaningful.

    For ~100 points, sigma=0.6 is fine (smooths ~1-2 neighbors).
    For 10k points, we need sigma~6 to smooth the same *fraction* of data.
    """
    return base_sigma * max(1.0, n / 100.0) ** 0.5


def compute_susceptibility(x, y, method="auto", kernel_sigma=None):
    """Compute susceptibility chi = |dO/dx|.

    Parameters
    ----------
    x, y : array-like, same length
        Parameter values and corresponding observable.
    method : str
        "gaussian", "savgol", or "auto" (picks based on data size & noise).
    kernel_sigma : float or None
        Smoothing width for Gaussian method. None = adaptive.

    Returns
    -------
    chi : ndarray  — susceptibility at each x
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Deduplicate x values to avoid NaN in gradient
    x, y = _deduplicate(x, y)

    if kernel_sigma is None:
        kernel_sigma = _adaptive_sigma(len(x))

    if method == "auto":
        method = _pick_method(x, y)

    if method == "gaussian":
        smoothed = gaussian_filter1d(y, sigma=kernel_sigma)
        chi = np.abs(np.gradient(smoothed, x))
    elif method == "savgol":
        win = min(max(11, len(y) // 10 | 1), len(y))
        if win % 2 == 0:
            win -= 1
        win = max(win, 5)
        poly = min(3, win - 2)
        dx = np.mean(np.diff(x)) if len(x) > 1 else 1.0
        chi = np.abs(savgol_filter(y, win, poly, deriv=1, delta=dx))
    else:
        raise ValueError(f"Unknown method: {method}")

    # Replace any NaN/inf with 0
    chi = np.nan_to_num(chi, nan=0.0, posinf=0.0, neginf=0.0)

    return chi


def _pick_method(x, y):
    n = len(y)
    if n < 20:
        return "gaussian"
    # Estimate SNR
    win = min(11, n)
    if win % 2 == 0:
        win -= 1
    win = max(win, 5)
    poly = min(3, win - 2)
    try:
        smoothed = savgol_filter(y, win, poly)
        residuals = y - smoothed
        snr = np.std(smoothed) / (np.std(residuals) + 1e-12)
        return "savgol" if snr >= 5 else "gaussian"
    except Exception:
        return "gaussian"


# ---------------------------------------------------------------------------
# Kappa (peak sharpness)
# ---------------------------------------------------------------------------

def compute_kappa(chi):
    """kappa = chi_max / mean(chi).  Higher = sharper peak."""
    baseline = np.mean(chi)
    if baseline < 1e-12:
        return 0.0
    return float(np.max(chi) / baseline)


# ---------------------------------------------------------------------------
# Critical point location
# ---------------------------------------------------------------------------

def find_critical_point(x, chi):
    """Return x-value where susceptibility peaks."""
    idx = int(np.argmax(chi))
    return float(x[idx]), idx


# ---------------------------------------------------------------------------
# Bootstrap confidence interval on sigma_c
# ---------------------------------------------------------------------------

def bootstrap_critical_point(x, y, n_boot=1000, kernel_sigma=None, rng=None):
    """Bootstrap CI for the critical point location.

    Resamples (x, y) pairs with replacement, recomputes chi, finds peak.
    Returns (sigma_c, ci_low, ci_high, boot_values).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(x)
    boot_peaks = np.empty(n_boot)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bx = x[idx]
        by = y[idx]
        order = np.argsort(bx)
        bx, by = bx[order], by[order]
        # Remove duplicates by averaging
        ux, inv = np.unique(bx, return_inverse=True)
        if len(ux) < 4:
            boot_peaks[i] = np.nan
            continue
        uy = np.zeros(len(ux))
        counts = np.zeros(len(ux))
        for j in range(len(inv)):
            uy[inv[j]] += by[j]
            counts[inv[j]] += 1
        uy /= counts
        chi_b = compute_susceptibility(ux, uy, method="gaussian", kernel_sigma=kernel_sigma)
        boot_peaks[i] = ux[np.argmax(chi_b)]

    valid = boot_peaks[~np.isnan(boot_peaks)]
    if len(valid) == 0:
        sigma_c = float(x[np.argmax(compute_susceptibility(x, y))])
        return sigma_c, sigma_c, sigma_c, np.array([sigma_c])

    ci_low = float(np.percentile(valid, 2.5))
    ci_high = float(np.percentile(valid, 97.5))
    sigma_c = float(np.median(valid))
    return sigma_c, ci_low, ci_high, valid


# ---------------------------------------------------------------------------
# Permutation test  (H0: no characteristic scale)
# ---------------------------------------------------------------------------

def permutation_test(x, y, n_perm=2000, kernel_sigma=None, rng=None):
    """Test whether the observed kappa is significantly above noise.

    Returns dict with p_value, observed_kappa, null_mean, null_std.
    """
    if rng is None:
        rng = np.random.default_rng(123)

    chi_obs = compute_susceptibility(x, y, method="gaussian", kernel_sigma=kernel_sigma)
    kappa_obs = compute_kappa(chi_obs)

    null_kappas = np.empty(n_perm)
    for i in range(n_perm):
        y_perm = rng.permutation(y)
        chi_p = compute_susceptibility(x, y_perm, method="gaussian", kernel_sigma=kernel_sigma)
        null_kappas[i] = compute_kappa(chi_p)

    p_value = float((np.sum(null_kappas >= kappa_obs) + 1) / (n_perm + 1))

    return {
        "p_value": p_value,
        "observed_kappa": kappa_obs,
        "null_mean": float(np.mean(null_kappas)),
        "null_std": float(np.std(null_kappas)),
        "null_95th": float(np.percentile(null_kappas, 95)),
        "significant": p_value < 0.05,
        "percentile_rank": float(np.mean(null_kappas < kappa_obs) * 100),
    }


# ---------------------------------------------------------------------------
# Safety margin
# ---------------------------------------------------------------------------

def compute_safety_margin(x, y, critical_x, current_x=None):
    """How far is the operating point from the tipping point?

    If current_x is None, uses the first x value.
    Returns margin as fraction of parameter range [0, 1].
    """
    if current_x is None:
        current_x = x[0]

    param_range = float(np.ptp(x))
    if param_range < 1e-12:
        return 0.0

    distance = abs(current_x - critical_x)
    margin = distance / param_range
    return float(np.clip(margin, 0.0, 1.0))


# ---------------------------------------------------------------------------
# STABILITY SCORE  (0 – 100, universal, self-calibrating)
# ---------------------------------------------------------------------------

def compute_stability_score(kappa, p_value, ci_width, param_range,
                            safety_margin, current_x=None, critical_x=None):
    """Combine metrics into a single 0-100 Stability Score.

    The score answers: "How stable is your system, and how confident are we?"

    Components (each 0-1, then weighted):
      1. Detection  (40%) — is the tipping point real?
         Based on permutation p-value → percentile rank.
         p < 0.001 → 1.0,  p > 0.20 → 0.0

      2. Clarity    (20%) — how sharp is the peak?
         Based on kappa (peak/mean ratio).
         Saturating curve: 1 - exp(-kappa / 5)

      3. Precision  (15%) — how precisely located?
         Based on CI width relative to parameter range.
         1 - (ci_width / param_range), clipped to [0, 1]

      4. Safety     (25%) — how far from the tipping point?
         Based on distance-to-critical / parameter range.
         Only if detection is significant; else contributes 0.

    Score = 100 * (w_d * detection + w_c * clarity + w_p * precision + w_s * safety)

    Self-calibrating: detection is measured against YOUR data's null distribution
    (permutation test), not absolute thresholds.
    """

    # 1. Detection (p-value → score via logistic transform)
    if p_value <= 0.001:
        detection = 1.0
    elif p_value >= 0.20:
        detection = 0.0
    else:
        # Smooth logistic mapping: steep drop around p=0.05
        detection = 1.0 / (1.0 + np.exp(15 * (p_value - 0.05)))

    # 2. Clarity (kappa → saturating curve)
    clarity = 1.0 - np.exp(-kappa / 5.0)

    # 3. Precision (CI width → relative narrowness)
    if param_range > 1e-12:
        precision = 1.0 - np.clip(ci_width / param_range, 0.0, 1.0)
    else:
        precision = 0.0

    # 4. Safety (margin, only meaningful if detection is real)
    if detection > 0.3 and safety_margin is not None:
        safety = np.clip(safety_margin * 2.0, 0.0, 1.0)  # 50% margin → max
    else:
        safety = 0.5  # neutral when no clear tipping point

    # Weights
    w_d, w_c, w_p, w_s = 0.40, 0.20, 0.15, 0.25

    raw = w_d * detection + w_c * clarity + w_p * precision + w_s * safety
    score = int(round(100 * np.clip(raw, 0.0, 1.0)))

    return {
        "score": score,
        "components": {
            "detection": round(detection, 3),
            "clarity": round(clarity, 3),
            "precision": round(precision, 3),
            "safety": round(safety, 3),
        },
        "weights": {"detection": w_d, "clarity": w_c, "precision": w_p, "safety": w_s},
    }


# ---------------------------------------------------------------------------
# Diagnosis text
# ---------------------------------------------------------------------------

def generate_diagnosis(score, kappa, p_value, critical_x, ci_low, ci_high,
                       safety_margin, current_x=None):
    """One-sentence human-readable diagnosis."""

    # Grade
    if score >= 85:
        grade = "STABLE"
        icon = "green"
    elif score >= 60:
        grade = "MODERATE"
        icon = "yellow"
    elif score >= 35:
        grade = "WARNING"
        icon = "orange"
    else:
        grade = "CRITICAL"
        icon = "red"

    # Tipping point text
    if p_value < 0.05:
        tp_text = f"Tipping point detected at x = {critical_x:.4g} (95% CI: {ci_low:.4g} - {ci_high:.4g})."
    elif kappa > 3.0 and p_value < 0.20:
        tp_text = (f"Possible tipping point at x = {critical_x:.4g} but not statistically "
                   f"significant (p = {p_value:.3f}). Consider adding more data points or reducing noise.")
    else:
        tp_text = f"No tipping point detected (p = {p_value:.3f})."

    # Safety text
    if safety_margin is not None and p_value < 0.05:
        pct = safety_margin * 100
        if current_x is not None:
            safety_text = f"Current operating point (x = {current_x:.4g}) is {pct:.0f}% of parameter range from the tipping point."
        else:
            safety_text = f"Safety margin: {pct:.0f}% of parameter range."
    else:
        safety_text = ""

    # Sharpness text
    if kappa > 20:
        sharp = "extremely sharp"
    elif kappa > 10:
        sharp = "very clear"
    elif kappa > 5:
        sharp = "clear"
    elif kappa > 3:
        sharp = "moderate"
    else:
        sharp = "weak"

    if p_value < 0.05:
        sharp_text = f"Peak sharpness: {sharp} (κ = {kappa:.1f})."
    else:
        sharp_text = ""

    parts = [tp_text, sharp_text, safety_text]
    detail = " ".join(p for p in parts if p)

    return {
        "grade": grade,
        "icon": icon,
        "detail": detail,
        "score": score,
    }
