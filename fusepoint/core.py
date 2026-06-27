"""
Core API — the three functions most users need.

    from fusepoint import analyze, scan, compare
    card = analyze(x, y)
    card = analyze(df, x="load", y="latency")
    results = scan("data.csv")           # auto-detect x, analyze all y columns
    delta = compare(x_before, y_before, x_after, y_after)
"""

from __future__ import annotations
import numpy as np
from fusepoint.engine import (
    compute_susceptibility,
    compute_kappa,
    find_critical_point,
    bootstrap_critical_point,
    permutation_test,
    compute_safety_margin,
    compute_stability_score,
    generate_diagnosis,
)
from fusepoint.result import StabilityResult
from fusepoint.parsers import parse_json, detect_x_column


def _col_to_label(col_name):
    """Turn a column name like 'response_time_ms' into 'Response Time Ms'."""
    return col_name.replace("_", " ").replace("-", " ").title()


def analyze(data, y=None, *, x=None, current_x=None, label=None,
            x_name=None, y_name=None,
            n_boot=1000, n_perm=2000, kernel_sigma=0.6,
            method="auto", deep=False):
    """Analyze your data for tipping points. Returns a StabilityResult.

    Parameters
    ----------
    data : array-like or DataFrame
        Either the x-values (parameter array), or a pandas DataFrame.
    y : array-like or str or None
        Either the y-values (observable array), or a column name (str)
        when data is a DataFrame.
    x : str, optional
        Column name for x-values when data is a DataFrame.
    current_x : float, optional
        Your current operating point. If given, the safety margin is
        computed relative to this point.
    label : str, optional
        Name for this analysis (shown on the Stability Card).
    x_name : str, optional
        Human-readable name for the x-axis (e.g., "Learning Rate").
        Auto-detected from DataFrame column names.
    y_name : str, optional
        Human-readable name for the y-axis (e.g., "Loss").
        Auto-detected from DataFrame column names.
    n_boot : int
        Number of bootstrap resamples (default 1000).
    n_perm : int
        Number of permutations for significance test (default 2000).
    kernel_sigma : float
        Smoothing width (default 0.6).
    method : str
        Derivative method: "auto", "gaussian", or "savgol".
    deep : bool
        If True, run the sigma_c_v4 theorem-anchored layer in addition
        to the FUSE statistical pipeline. The result gains .regime,
        .citations, .gamma_O, and .paper_doi fields, each traceable to a
        proof in the foundation paper (doi:10.5281/zenodo.20548818).

    Returns
    -------
    StabilityResult
        .score (0-100), .grade, .diagnosis, .show(), .save(), etc.

    Examples
    --------
    >>> # Array mode
    >>> card = analyze(x_array, y_array, current_x=0.2, label="My System")
    >>>
    >>> # DataFrame mode — column names become axis labels automatically
    >>> card = analyze(df, x="learning_rate", y="loss", current_x=0.01)
    >>> card.save("report.png")
    >>>
    >>> # Theorem-anchored mode
    >>> card = analyze(df, x="time", y="signal", deep=True)
    >>> print(card.regime)        # "I_geom" / "II_geom" / "III_geom"
    >>> print(card.citations)     # ["def:sigmac", "thm:trichotomy-geometric", ...]
    """
    # --- String input: JSON or file path ---
    if isinstance(data, str):
        import json as _json
        try:
            parsed = _json.loads(data)
        except (_json.JSONDecodeError, ValueError):
            # Not JSON — try as file path
            import os
            if os.path.isfile(data):
                ext = os.path.splitext(data)[1].lower()
                import pandas as pd
                if ext == ".json":
                    with open(data) as f:
                        raw = _json.load(f)
                    data = parse_json(raw)
                elif ext in (".csv", ".tsv", ".txt"):
                    sep = "\t" if ext == ".tsv" else ","
                    data = pd.read_csv(data, sep=sep)
                elif ext in (".xls", ".xlsx"):
                    data = pd.read_excel(data)
                elif ext == ".parquet":
                    data = pd.read_parquet(data)
                else:
                    raise ValueError(f"Unsupported file type: {ext}")
            else:
                raise ValueError(
                    "String input must be valid JSON or a path to a "
                    "CSV/TSV/JSON/Excel/Parquet file"
                )
        else:
            data = parse_json(parsed)

    # --- dict/list input: use smart parser ---
    if isinstance(data, (dict, list)):
        data = parse_json(data)

    # --- DataFrame detection ---
    _is_df = hasattr(data, 'columns') and hasattr(data, '__getitem__')

    if _is_df:
        df = data
        if isinstance(y, str) and isinstance(x, str):
            # analyze(df, x="col_a", y="col_b")  or  analyze(df, "col_b", x="col_a")
            x_col, y_col = x, y
        elif isinstance(y, str) and x is None:
            # analyze(df, "col_b") — x defaults to first column that isn't y
            y_col = y
            x_col = [c for c in df.columns if c != y_col][0]
        elif y is None and x is None:
            # analyze(df) — use first two columns
            cols = list(df.columns)
            if len(cols) < 2:
                raise ValueError("DataFrame must have at least 2 columns")
            x_col, y_col = cols[0], cols[1]
        else:
            raise ValueError(
                "For DataFrame input, use: analyze(df, x='col', y='col') "
                "or analyze(df) to use the first two columns"
            )

        x_arr = np.asarray(df[x_col], dtype=np.float64).ravel()
        y_arr = np.asarray(df[y_col], dtype=np.float64).ravel()
        if x_name is None:
            x_name = _col_to_label(x_col)
        if y_name is None:
            y_name = _col_to_label(y_col)
    else:
        # Array mode: analyze(x_array, y_array)
        x_arr = np.asarray(data, dtype=np.float64).ravel()
        if y is None:
            raise ValueError("y is required when x is an array")
        y_arr = np.asarray(y, dtype=np.float64).ravel()

    if len(x_arr) != len(y_arr):
        raise ValueError(f"x and y must have the same length, got {len(x_arr)} and {len(y_arr)}")
    if len(x_arr) < 5:
        raise ValueError(f"Need at least 5 data points, got {len(x_arr)}")

    # Sort by x
    order = np.argsort(x_arr)
    x_arr, y_arr = x_arr[order], y_arr[order]

    # Deduplicate: average y-values at duplicate x positions
    # (must happen here so all arrays stay the same length)
    from fusepoint.engine import _deduplicate
    x_arr, y_arr = _deduplicate(x_arr, y_arr)

    if len(x_arr) < 5:
        raise ValueError(f"Need at least 5 unique x-values, got {len(x_arr)}")

    # 1. Susceptibility
    chi = compute_susceptibility(x_arr, y_arr, method=method, kernel_sigma=kernel_sigma)

    # 2. Kappa
    kappa = compute_kappa(chi)

    # 3. Critical point location
    critical_x, peak_idx = find_critical_point(x_arr, chi)

    # 4. Bootstrap CI
    sigma_c, ci_low, ci_high, _ = bootstrap_critical_point(
        x_arr, y_arr, n_boot=n_boot, kernel_sigma=kernel_sigma
    )

    # 5. Permutation test
    perm = permutation_test(x_arr, y_arr, n_perm=n_perm, kernel_sigma=kernel_sigma)
    p_value = perm["p_value"]

    # 6. Safety margin
    safety = compute_safety_margin(x_arr, y_arr, critical_x, current_x=current_x)

    # 7. Stability Score
    ci_width = ci_high - ci_low
    param_range = float(np.ptp(x_arr))
    score_info = compute_stability_score(
        kappa=kappa,
        p_value=p_value,
        ci_width=ci_width,
        param_range=param_range,
        safety_margin=safety,
        current_x=current_x,
        critical_x=critical_x,
    )

    # 8. Diagnosis
    diag = generate_diagnosis(
        score=score_info["score"],
        kappa=kappa,
        p_value=p_value,
        critical_x=critical_x,
        ci_low=ci_low,
        ci_high=ci_high,
        safety_margin=safety,
        current_x=current_x,
    )

    # 9. Theorem-anchored layer (opt-in via deep=True)
    v4_regime = v4_regime_detail = v4_citations = v4_gamma_O = None
    v4_framework = v4_paper_doi = None
    if deep:
        import sigma_c_v4 as _sv4
        # Share FUSE's chi as the single source of truth so v4 classifies
        # the same curve FUSE scored. Declare A1 explicitly: FUSE's
        # parameter-sweep preprocessing (dedup + smoothing) is scale-
        # equivariant, so A1 holds at the observable layer.
        _v4_kwargs = {"chi": chi}
        try:
            v4_result = _sv4.analyze(
                x_arr, y_arr,
                preprocessing_scale_equivariant=True,
                **_v4_kwargs,
            )
        except TypeError:
            # Older sigma_c_v4 (<4.1) without the A1 kwarg.
            v4_result = _sv4.analyze(x_arr, y_arr, **_v4_kwargs)
        v4_regime = v4_result.regime.geometric
        v4_regime_detail = v4_result.regime.as_dict()
        v4_citations = list(v4_result.citations)
        v4_gamma_O = v4_result.gamma_O
        v4_framework = (v4_result.framework.value
                        if v4_result.framework is not None else None)
        v4_paper_doi = _sv4.__paper_doi__

        # Augment FUSE's diagnosis with the regime verdict.
        regime_label = {
            "I_geom": "single mode (regime I, paper Thm 8.5)",
            "II_geom": "multi-mode (regime II, paper Thm 8.5)",
            "III_geom": "no characteristic scale (regime III, paper Thm 8.5)",
        }.get(v4_regime, v4_regime)
        diag["detail"] = f"{diag['detail']} Regime: {regime_label}."

        # v4 >= 5.0 surfaces an explicit operational-floor flag when the
        # spectral discriminator hits the eta_O threshold. Pass it through
        # as a visible caveat rather than burying it in regime_detail.
        if v4_regime_detail.get("operational_floor_triggered"):
            diag["detail"] += (
                " Operational-floor triggered: peak height is at the"
                " threshold; treat sigma_c as a soft reading.")

        # v4 >= 5.0 reports an orthogonal spectral-type axis when the
        # input is an autocorrelation; parameter-sweep inputs leave it
        # None, so this only fires on time-series-style data passed via
        # the same API.
        spec = v4_regime_detail.get("spectral")
        if spec:
            diag["detail"] += f" Spectral type: {spec}."

        # Branch-(ii) standing caveat: a single visible mode in the
        # chi-profile of a single probe does NOT exclude a slower mode
        # that this probe suppressed below the visibility threshold.
        # See addendum Remark on Branch-(ii) (regime I with one mode).
        if v4_regime == "I_geom":
            diag["detail"] += (
                " Note: single-probe regime I — a second probe with"
                " different modal coupling could surface a suppressed"
                " slower mode (two-probe verification recommended).")

    return StabilityResult(
        score=score_info["score"],
        grade=diag["grade"],
        icon=diag["icon"],
        diagnosis=diag["detail"],
        critical_x=critical_x,
        ci_low=ci_low,
        ci_high=ci_high,
        kappa=kappa,
        p_value=p_value,
        safety_margin=safety,
        components=score_info["components"],
        x=x_arr,
        y=y_arr,
        chi=chi,
        current_x=current_x,
        label=label,
        x_name=x_name,
        y_name=y_name,
        regime=v4_regime,
        regime_detail=v4_regime_detail,
        citations=v4_citations,
        gamma_O=v4_gamma_O,
        paper_doi=v4_paper_doi,
        framework=v4_framework,
    )


def compare(data1, y1=None, data2=None, y2=None, *,
            x1=None, x2=None,
            label_before="Before", label_after="After",
            current_x=None, **kwargs):
    """Analyze two datasets and produce a side-by-side comparison.

    Supports both array mode and DataFrame mode:
        compare(x1, y1, x2, y2, ...)
        compare(df1, x="col", y="col_b", data2=df2, ...)

    Returns a ComparisonResult with .show() and .save().
    """
    # For array mode, data2 and y2 are positional
    r1 = analyze(data1, y1, x=x1, current_x=current_x, label=label_before, **kwargs)
    r2 = analyze(data2, y2, x=x2, current_x=current_x, label=label_after, **kwargs)
    return ComparisonResult(before=r1, after=r2)


class ComparisonResult:
    """Side-by-side comparison of two StabilityResults."""

    def __init__(self, before: StabilityResult, after: StabilityResult):
        self.before = before
        self.after = after
        self.delta_score = after.score - before.score

    def __repr__(self):
        sign = "+" if self.delta_score >= 0 else ""
        return (
            f"ComparisonResult(before={self.before.score}, "
            f"after={self.after.score}, delta={sign}{self.delta_score})"
        )

    def summary(self) -> str:
        sign = "+" if self.delta_score >= 0 else ""
        return (
            f"  BEFORE: {self.before.score}/100 [{self.before.grade}]\n"
            f"  AFTER:  {self.after.score}/100 [{self.after.grade}]\n"
            f"  DELTA:  {sign}{self.delta_score} points\n"
        )

    def show(self, **kwargs):
        from fusepoint.card import render_comparison_card
        fig = render_comparison_card(self, **kwargs)
        import matplotlib.pyplot as plt
        plt.show()
        return fig

    def save(self, path: str = "comparison_card.png", dpi: int = 200, **kwargs):
        from fusepoint.card import render_comparison_card
        fig = render_comparison_card(self, **kwargs)
        fig.savefig(path, dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor(), edgecolor="none")
        import matplotlib.pyplot as plt
        plt.close(fig)
        return path


# ---------------------------------------------------------------------------
# scan() — auto-detect x, analyze every numeric column
# ---------------------------------------------------------------------------

def scan(data, *, x=None, current_x=None, n_boot=1000, n_perm=2000,
         kernel_sigma=0.6, method="auto", top_n=None, deep=False):
    """Scan all numeric columns in a dataset for tipping points.

    Automatically detects the x (independent variable) column and runs
    ``analyze()`` on every remaining numeric column.

    Parameters
    ----------
    data : DataFrame, dict, list, str
        Input data.  Accepts anything ``analyze()`` accepts: DataFrame,
        column-oriented dict, JSON string, or a file path to CSV/JSON/
        Excel/Parquet.
    x : str, optional
        Column name for the x-axis.  If *None*, the most likely x column
        is detected automatically (monotonic, evenly spaced, keyword match).
    current_x : float, optional
        Current operating point (passed to each ``analyze()`` call).
    n_boot : int
        Bootstrap resamples per column (default 1000).
    n_perm : int
        Permutation iterations per column (default 2000).
    kernel_sigma : float
        Smoothing width (default 0.6).
    method : str
        Derivative method: "auto", "gaussian", or "savgol".
    top_n : int, optional
        If set, only return the top *n* results by score (descending).

    Returns
    -------
    list[StabilityResult]
        One result per numeric column, sorted by score descending.

    Examples
    --------
    >>> results = scan("data.csv")
    >>> for r in results:
    ...     print(f"{r.y_name}: {r.score} ({r.grade})")

    >>> results = scan(df, x="time", current_x=120, top_n=5)
    >>> results[0].show()
    """
    import pandas as pd

    # Convert to DataFrame
    if isinstance(data, str):
        import json as _json
        try:
            parsed = _json.loads(data)
            data = parse_json(parsed)
        except (_json.JSONDecodeError, ValueError):
            import os
            if os.path.isfile(data):
                ext = os.path.splitext(data)[1].lower()
                if ext == ".json":
                    with open(data) as f:
                        raw = _json.load(f)
                    data = parse_json(raw)
                elif ext in (".csv", ".tsv", ".txt"):
                    sep = "\t" if ext == ".tsv" else ","
                    data = pd.read_csv(data, sep=sep)
                elif ext in (".xls", ".xlsx"):
                    data = pd.read_excel(data)
                elif ext == ".parquet":
                    data = pd.read_parquet(data)
                else:
                    raise ValueError(f"Unsupported file type: {ext}")
            else:
                raise ValueError(f"Not valid JSON and not a file: {data}")

    if isinstance(data, (dict, list)):
        data = parse_json(data)

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "scan() requires tabular data (DataFrame, dict, JSON, or file path)"
        )

    df = data
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        raise ValueError(
            f"Need at least 2 numeric columns for scan(), got {len(numeric_cols)}"
        )

    # Detect x column
    if x is not None:
        x_col = x
    else:
        x_col = detect_x_column(df, numeric_cols)

    y_cols = [c for c in numeric_cols if c != x_col]
    if not y_cols:
        raise ValueError("No y columns remaining after x detection")

    results = []
    for y_col in y_cols:
        try:
            r = analyze(
                df, x=x_col, y=y_col,
                current_x=current_x,
                label=_col_to_label(y_col),
                n_boot=n_boot,
                n_perm=n_perm,
                kernel_sigma=kernel_sigma,
                method=method,
                deep=deep,
            )
            results.append(r)
        except (ValueError, RuntimeError):
            # Skip columns that fail (too few unique values, etc.)
            continue

    results.sort(key=lambda r: r.score)

    if top_n is not None:
        results = results[:top_n]

    return results
