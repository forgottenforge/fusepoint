# Fuse
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Commercial License](https://img.shields.io/badge/License-Commercial-orange.svg)](mailto:nfo@forgottenforge.xyz)
[![Version](https://img.shields.io/badge/version-0.1.0-green.svg)](https://pypi.org/project/fusekit/)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.9-blue.svg)](https://python.org)
[![Status](https://img.shields.io/badge/status-production-success.svg)]()

**Know your breaking point.**

One function. Two columns of data. A publication-quality Fuse Report.

```python
from fusekit import analyze

card = analyze(x, y, current_x=0.01)
print(card.score)   # 87
card.save("fuse_report.png")
```

## What it does

You have a parameter you turn and a result you measure. Fuse tells you:

- **Where the tipping point is** (and how sure it is — bootstrap CI)
- **Whether it's real or noise** (permutation test, not guessing)
- **How sharp it is** (k — is it a cliff or a gentle slope?)
- **How safe you are** (distance to the edge, as a percentage)
- **A single Stability Score** from 0 to 100

All of this in a beautiful Fuse Report you can screenshot, share, put in a presentation.

## Install

```bash
pip install fusekit
```

Dependencies: numpy, scipy, matplotlib, pandas. That's it.

## Quick Start

### Array mode

```python
import numpy as np
from fusekit import analyze

lr = np.linspace(1e-5, 0.1, 80)
loss = your_training_function(lr)

card = analyze(lr, loss, current_x=0.01,
               x_name="Learning Rate", y_name="Loss",
               label="Training Stability")
print(card.score)        # 87 — you're safe
print(card.critical_x)   # 0.035 — this is where it blows up
card.save("lr_report.png")
```

### DataFrame mode — the natural API

```python
import pandas as pd
from fusekit import analyze

df = pd.read_csv("server_metrics.csv")
card = analyze(df, x="concurrent_requests", y="response_time_ms",
               current_x=5000, label="Production Server")
card.save("server_report.png")
```

Column names become axis labels automatically.

### Scan mode — analyze everything at once

```python
from fusekit import scan

results = scan("data.csv")                # auto-detect x, analyze all y columns
results = scan(df, x="time", top_n=5)     # explicit x, top 5 results

for r in results:
    print(f"{r.y_name}: {r.score} ({r.grade})")
    r.save(f"{r.y_name}_report.png")
```

Accepts CSV, TSV, JSON (Plotly, Elasticsearch, Pandas formats), Excel, and Parquet.

### Before / After — the comparison

```python
from fusekit import compare

delta = compare(x, y_before, x, y_after,
                current_x=0.2,
                label_before="Before Fix",
                label_after="After Fix")
print(delta.delta_score)  # +18 points
delta.save("improvement.png")
```

## The Score

The Stability Score (0-100) is built from four independently validated
statistical components:

| Component | Weight | What it measures |
|-----------|--------|-----------------|
| **Detection** | 40% | Is the tipping point real? (permutation p-value) |
| **Clarity** | 20% | How sharp is it? (k = peak/mean ratio) |
| **Precision** | 15% | How precisely located? (CI width / range) |
| **Safety** | 25% | How far from the edge? (margin / range) |

The score is **self-calibrating**: Detection is measured against your own data's
null distribution, not against arbitrary thresholds.

## What it's NOT

- Not a curve fitter (use scipy for that)
- Not an anomaly detector (use isolation forests for that)
- Not a time-series tool (use ruptures for changepoint detection)

Fuse finds **parameter-space tipping points** — the critical value of a
knob where your system's behavior qualitatively changes. And it tells you how
confident it is.

## Built on

The mathematics behind Fuse come from [sigmacore](https://github.com/forgottenforge/sigmacore),
a peer-reviewed universal criticality analysis framework published in
[AVS Quantum Science](https://doi.org/10.1116/5.0254846).
Fuse is the simple door to that building.

## License

Copyright (c) 2026 Forgotten Forge — [forgottenforge.xyz](https://www.forgottenforge.xyz)

Dual-licensed: **AGPL-3.0** for open-source use, **commercial licenses** available.
Contact nfo@forgottenforge.xyz for commercial inquiries.
