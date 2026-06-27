# FUSE

[![PyPI](https://img.shields.io/pypi/v/fusepoint.svg)](https://pypi.org/project/fusepoint/)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.9-blue.svg)](https://python.org)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Commercial License](https://img.shields.io/badge/License-Commercial-orange.svg)](mailto:nfo@forgottenforge.xyz)
[![Paper DOI](https://img.shields.io/badge/paper-10.5281%2Fzenodo.20548818-blue)](https://doi.org/10.5281/zenodo.20548818)
[![Live demo](https://img.shields.io/badge/demo-huggingface-yellow)](https://huggingface.co/spaces/ForgottenForge/fuse)

**Try it without installing:** [huggingface.co/spaces/ForgottenForge/fuse](https://huggingface.co/spaces/ForgottenForge/fuse)

**Know your breaking point.**

You have a parameter you turn and a result you measure. FUSE tells you where
the system breaks, how confident it is, and how far you are from the edge —
in one function call, with a publication-ready report.

```python
from fusepoint import analyze

card = analyze(x, y, current_x=0.01)
print(card.score)   # 87
card.save("fuse_report.png")
```

## What FUSE answers

- **Where is the tipping point?** Bootstrap confidence interval included.
- **Is it real or noise?** Permutation test against your own data, not an
  arbitrary threshold.
- **How sharp is it?** k = peak / mean (cliff vs gentle slope).
- **How safe is the current operating point?** Distance to the edge as a
  fraction of the parameter range.
- **One number to act on.** A 0-100 Stability Score combining the above.

All of this in a single `StabilityResult` you can `print`, `.show()`, or
`.save("report.png")`.

## Install

```bash
pip install fusepoint              # core library
pip install "fusepoint[ui]"        # + Streamlit web UI (fuse-ui)
```

Core dependencies: numpy, scipy, pandas, matplotlib,
[sigma-c-framework](https://pypi.org/project/sigma-c-framework/) (theorem-anchored layer).

## Two layers

FUSE ships two layers over the same data:

### Statistical layer (default)

Bootstrap + permutation + 0-100 stability score. Works on any tabular data —
no physics assumed.

```python
from fusepoint import analyze
result = analyze(df, x="load", y="latency", current_x=5000)
print(result.score, result.grade)        # 87 STABLE
print(result.critical_x, result.ci)      # 7245.0 (6890, 7580)
```

### Theorem-anchored layer (`deep=True`)

Adds regime classification, gamma_O stability indicator, and theorem
citations — each output traceable to a proof in the
[foundation paper](https://doi.org/10.5281/zenodo.20548818).

```python
result = analyze(df, x="load", y="latency", deep=True)
print(result.regime)        # "I_geom"  (single mode, paper Thm 8.5)
print(result.gamma_O)       # 24.3      (strict-SOC indicator)
print(result.citations)     # ["def:sigmac", "thm:trichotomy-geometric", ...]
print(result.paper_doi)     # "10.5281/zenodo.20548818"
```

The statistical layer scores **how stable** the system is. The
theorem-anchored layer says **which regime** the system is in and which
paper theorem licenses that reading.

#### What `deep=True` surfaces on top of regime + gamma_O

With `sigma-c-framework >= 5.0`, the theorem-anchored layer reports
three additional pieces of provenance on the diagnosis string:

- **Branch-(ii) standing caveat for regime I.** A single visible mode
  in the chi-profile of a single probe does *not* exclude a slower
  mode that this probe suppressed below the visibility threshold.
  The framework's published two-probe verification protocol is the
  standard cross-check; FUSE surfaces this caveat by default whenever
  a regime-I reading is returned from a single probe.
- **Operational-floor flag.** When the spectral-attribution gate sits
  at the threshold, the regime call is reported as a soft reading
  rather than an asymptotic one.
- **Spectral type (gapped / III-pp / III-anom)** when the input is an
  autocorrelation profile rather than a parameter sweep. Parameter
  sweeps (FUSE's default mode) leave this field empty.

These appear in `result.diagnosis` and `result.regime_detail` only
when `deep=True`; the default statistical layer is unchanged.

## Quick start

### Array mode

```python
import numpy as np
from fusepoint import analyze

lr = np.linspace(1e-5, 0.1, 80)
loss = your_training_function(lr)

card = analyze(lr, loss, current_x=0.01,
               x_name="Learning Rate", y_name="Loss",
               label="Training Stability")
print(card.score)        # 87 -- you're safe
print(card.critical_x)   # 0.035 -- this is where it blows up
card.save("lr_report.png")
```

### DataFrame mode

```python
import pandas as pd
from fusepoint import analyze

df = pd.read_csv("server_metrics.csv")
card = analyze(df, x="concurrent_requests", y="response_time_ms",
               current_x=5000, label="Production Server")
card.save("server_report.png")
```

Column names become axis labels automatically.

### Scan mode

```python
from fusepoint import scan

results = scan("data.csv")                # auto-detect x, analyze every y column
results = scan(df, x="time", top_n=5)     # explicit x, top 5

for r in results:
    print(f"{r.y_name}: {r.score} ({r.grade})")
    r.save(f"{r.y_name}_report.png")
```

Accepts CSV, TSV, JSON (Plotly, Elasticsearch, pandas formats), Excel, Parquet.

### Compare mode

```python
from fusepoint import compare

delta = compare(x, y_before, x, y_after,
                current_x=0.2,
                label_before="Before Fix",
                label_after="After Fix")
print(delta.delta_score)  # +18 points
delta.save("improvement.png")
```

### Web UI

```bash
pip install "fusepoint[ui]"
fuse-ui
```

Drag-and-drop CSV upload, demo datasets included, scans every numeric column
and renders the cards side by side.

## The Stability Score

The Stability Score (0-100) combines four independently validated
statistical components:

| Component | Weight | What it measures |
|-----------|--------|------------------|
| **Detection** | 40% | Is the tipping point real? (permutation p-value) |
| **Clarity** | 20% | How sharp is it? (k = peak / mean ratio) |
| **Precision** | 15% | How precisely located? (CI width / range) |
| **Safety** | 25% | How far from the edge? (margin / range) |

The score is **self-calibrating**: detection is measured against your own
data's null distribution, not against arbitrary thresholds.

## What FUSE is *not*

- Not a curve fitter (use scipy for that).
- Not an anomaly detector (use isolation forests).
- Not a time-series tool (use `ruptures` for changepoint detection).

FUSE finds **parameter-space tipping points** — the critical value of a
knob where your system's behaviour qualitatively changes — and tells you
how confident it is.

## Version history

| Version | What changed |
|---------|--------------|
| **1.1.0** (current) | `deep=True` layer aligned to `sigma-c-framework` v5: surfaces the Branch-(ii) standing caveat on regime I, operational-floor flag, and spectral-type axis. Streamlit web UI forces dark theme. Explanation block visible on landing. |
| 1.0.x | Production release line. Streamlit UI bundled (`fuse-ui`). Theorem-anchored `deep=True` layer powered by `sigma-c-framework` v4. Dual licensure (AGPL + Commercial). |
| 0.1.0 | Initial PyPI release (renamed from `fusekit`). CLI / library only. |

## Citation

When `deep=True` results inform a publication, please cite the foundation
paper:

> Wurm, M. C. (2026). *Operational scale selection: axioms, spectral
> concentration, and a regime trichotomy.* Zenodo,
> [doi:10.5281/zenodo.20548818](https://doi.org/10.5281/zenodo.20548818).

## License

Copyright (c) 2026 Forgotten Forge — [forgottenforge.xyz](https://www.forgottenforge.xyz)

FUSE is **dual-licensed**: see [`LICENSE`](LICENSE).

- **AGPL-3.0-or-later** for open-source and academic use
  ([`license_AGPL.txt`](license_AGPL.txt)).
- **Commercial licence** available for use cases where AGPL obligations
  are inappropriate ([`license_COMMERCIAL.txt`](license_COMMERCIAL.txt)).
  Contact `nfo@forgottenforge.xyz`.

