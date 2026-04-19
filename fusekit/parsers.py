"""
Data parsing utilities — comprehensive JSON/dict/file handling + x-column detection.

These are the same parsers that power the Fuse web app, now available
in the package so ``scan()`` and ``analyze()`` handle any input format.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Smart JSON parser — handles Plotly, Elasticsearch, GraphQL, Pandas formats
# ---------------------------------------------------------------------------

def parse_json(raw):
    """Parse a JSON-decoded object into a DataFrame.

    Handles many common formats automatically:

    1. Records:         ``[{"col1": val, "col2": val}, ...]``
    2. Column-oriented: ``{"col1": [...], "col2": [...]}``
    3. Pandas split:    ``{"columns": [...], "data": [[...], ...]}``
    4. Pandas table:    ``{"schema": {"fields": [...]}, "data": [...]}``
    5. Pandas index:    ``{"row_0": {"col": val}, "row_1": {...}}``
    6. Wrapped data:    ``{"data": [...]}`` or ``{"results": [...]}``
    7. Plotly traces:   ``{"data": [{"x": [...], "y": [...]}]}``
    8. Nested dicts with numeric arrays or records

    Parameters
    ----------
    raw : list or dict
        Decoded JSON object.

    Returns
    -------
    pd.DataFrame
    """
    # --- Case 1: top-level list ---
    if isinstance(raw, list):
        if raw and isinstance(raw[0], dict):
            return pd.DataFrame(raw)
        if raw and isinstance(raw[0], list):
            return pd.DataFrame(raw)
        return pd.DataFrame(raw)

    if not isinstance(raw, dict):
        return pd.DataFrame(raw)

    keys = set(raw.keys())

    # --- Case 3: Pandas split format ---
    if "columns" in keys and "data" in keys:
        cols = raw["columns"]
        data = raw["data"]
        if isinstance(cols, list) and isinstance(data, list):
            return pd.DataFrame(data, columns=cols, index=raw.get("index"))

    # --- Case 4: Pandas table format ---
    if "schema" in keys and "data" in keys:
        schema = raw["schema"]
        if isinstance(schema, dict) and "fields" in schema:
            col_names = [f["name"] for f in schema["fields"]
                         if f.get("name") != "index"]
            if col_names:
                return pd.DataFrame(raw["data"])[col_names]
            return pd.DataFrame(raw["data"])

    # --- Case 7: Plotly traces (check BEFORE wrapper detection) ---
    if "data" in keys and isinstance(raw["data"], list) and raw["data"]:
        first = raw["data"][0]
        if isinstance(first, dict) and ("x" in first or "y" in first):
            has_arrays = any(isinstance(first.get(k), list) for k in ("x", "y"))
            if has_arrays:
                df_parts = {}
                for i, trace in enumerate(raw["data"]):
                    name = trace.get("name", f"trace_{i}")
                    if "x" in trace and isinstance(trace["x"], list):
                        if "x" not in df_parts:
                            df_parts["x"] = trace["x"]
                    if "y" in trace and isinstance(trace["y"], list):
                        ykey = name if len(raw["data"]) > 1 else "y"
                        df_parts[ykey] = trace["y"]
                if df_parts:
                    min_len = min(len(v) for v in df_parts.values())
                    return pd.DataFrame({k: v[:min_len]
                                         for k, v in df_parts.items()})

    # --- Case 6: Wrapped data (API responses) ---
    _wrapper_keys = {"data", "results", "items", "records", "rows",
                     "values", "hits", "entries", "response", "series"}
    for wk in _wrapper_keys:
        if wk not in keys:
            continue
        inner = raw[wk]
        if isinstance(inner, list) and inner:
            if isinstance(inner[0], dict):
                sample_vals = list(inner[0].values())
                is_flat = all(not isinstance(v, (list, dict)) for v in sample_vals)
                if is_flat:
                    if "_source" in inner[0]:
                        return pd.DataFrame([h["_source"] for h in inner])
                    return pd.DataFrame(inner)
            if isinstance(inner[0], list) and "columns" in keys:
                return pd.DataFrame(inner, columns=raw["columns"])
        if isinstance(inner, dict):
            for sub_key, sub_val in inner.items():
                if isinstance(sub_val, list) and sub_val:
                    if isinstance(sub_val[0], dict):
                        if "_source" in sub_val[0]:
                            return pd.DataFrame([h["_source"] for h in sub_val])
                        return pd.DataFrame(sub_val)

    # Nested Elasticsearch: {"hits": {"hits": [{_source: ...}]}}
    if "hits" in keys and isinstance(raw["hits"], dict):
        hits_inner = raw["hits"]
        if "hits" in hits_inner and isinstance(hits_inner["hits"], list):
            docs = hits_inner["hits"]
            if docs and isinstance(docs[0], dict) and "_source" in docs[0]:
                return pd.DataFrame([h["_source"] for h in docs])

    # --- Case 2: Column-oriented ---
    vals = list(raw.values())
    if vals and all(isinstance(v, list) for v in vals):
        lengths = [len(v) for v in vals]
        if len(set(lengths)) == 1:
            return pd.DataFrame(raw)

    # --- Case 5: Pandas index format (dict of dicts) ---
    if vals and all(isinstance(v, dict) for v in vals):
        inner_keys = set(vals[0].keys())
        if all(set(v.keys()) == inner_keys for v in vals):
            return pd.DataFrame.from_dict(raw, orient="index")

    # --- Case 8: deeply nested — extract all numeric arrays ---
    tables = {}
    _extract_arrays(raw, tables, prefix="")

    if not tables:
        return pd.DataFrame(raw)

    by_length = {}
    for key, arr in tables.items():
        n = len(arr)
        if n not in by_length:
            by_length[n] = {}
        by_length[n][key] = arr

    best_n = max(by_length, key=lambda n: len(by_length[n]))
    return pd.DataFrame(by_length[best_n])


def _extract_arrays(obj, out, prefix):
    """Recursively extract all numeric arrays from a nested dict."""
    if isinstance(obj, list) and len(obj) > 0:
        if all(isinstance(v, (int, float)) for v in obj):
            out[prefix] = obj
        elif all(isinstance(v, dict) for v in obj):
            for key in obj[0]:
                values = [row.get(key) for row in obj]
                if all(isinstance(v, (int, float)) for v in values if v is not None):
                    clean = [v for v in values if v is not None]
                    if len(clean) == len(obj):
                        col_name = f"{prefix}.{key}" if prefix else key
                        out[col_name] = clean
    elif isinstance(obj, dict):
        for key, val in obj.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            _extract_arrays(val, out, new_prefix)


# ---------------------------------------------------------------------------
# Auto-detect x column
# ---------------------------------------------------------------------------

_X_KEYWORDS = frozenset({
    "time", "step", "index", "depth", "age", "x", "date",
    "epoch", "iteration", "round", "trial", "sample", "id",
    "frequency", "freq", "wavelength", "position", "distance",
})


def detect_x_column(df, numeric_cols=None):
    """Auto-detect the most likely x (independent variable) column.

    Heuristics (cumulative score):
      +3  monotonically increasing
      +2  monotonically decreasing
      +2  evenly spaced values
      +2  name matches common x-axis keywords
      +1  high uniqueness ratio (> 80%)

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    numeric_cols : list[str], optional
        Columns to consider. Defaults to all numeric columns.

    Returns
    -------
    str
        Best candidate column name.
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    best = None
    best_score = -1

    for col in numeric_cols:
        vals = df[col].dropna().values
        if len(vals) < 5:
            continue

        score = 0

        diffs = np.diff(vals)
        if np.all(diffs >= 0):
            score += 3
        elif np.all(diffs <= 0):
            score += 2

        if len(vals) > 2:
            diffs_pos = diffs[diffs != 0]
            if len(diffs_pos) > 0:
                cv = np.std(diffs_pos) / (np.mean(np.abs(diffs_pos)) + 1e-12)
                if cv < 0.01:
                    score += 2

        col_lower = col.lower().replace("_", "").replace("-", "").replace(" ", "")
        for kw in _X_KEYWORDS:
            if kw in col_lower:
                score += 2
                break

        uniqueness = len(np.unique(vals)) / len(vals)
        if uniqueness > 0.8:
            score += 1

        if score > best_score:
            best_score = score
            best = col

    return best or numeric_cols[0]
