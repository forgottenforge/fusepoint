---
title: FUSE
emoji: 🧨
colorFrom: gray
colorTo: blue
sdk: streamlit
sdk_version: 1.30.0
app_file: app.py
pinned: false
license: agpl-3.0
short_description: Know your breaking point.
---

# FUSE

**Know your breaking point.**

FUSE is an operational stability and tipping-point detector. Drop in a time series
(CSV, log file, or live stream) and FUSE returns the dominant operational scale and
a stability verdict grounded in the σ_c framework — telling you whether the system
is in a regime where small perturbations stay small, or where they are about to
blow up.

This Space runs the bundled Streamlit UI from the [`fusepoint`](https://pypi.org/project/fusepoint/)
PyPI package on top of [`sigma-c-framework`](https://pypi.org/project/sigma-c-framework/).

## Links

- **Source code**: <https://github.com/forgottenforge/fusepoint>
- **PyPI**: <https://pypi.org/project/fusepoint/>
- **Issue tracker**: <https://github.com/forgottenforge/fusepoint/issues>

## License

This demo is distributed under **AGPL-3.0**. If you run a modified version as a
network service, the AGPL requires you to make your modifications available to
the users of that service. A separate commercial license is available — see the
upstream repository for details.
