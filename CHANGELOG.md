# Changelog

All notable changes to FUSE (`fusepoint`) are recorded here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project adheres to [Semantic Versioning](https://semver.org/).

## [1.1.0] - 2026-06-27

### Added

- `deep=True` layer aligned to `sigma-c-framework` v5: the diagnosis
  string now surfaces three additional pieces of provenance from the
  theorem-anchored layer:
  - **Branch-(ii) standing caveat** for regime I: a single visible
    mode in the chi-profile of a single probe does not exclude a
    slower mode that this probe suppressed below the visibility
    threshold; the framework's two-probe verification protocol is
    the standard cross-check.
  - **Operational-floor flag**: surfaced when the spectral-attribution
    gate sits at the threshold (peak height at the eta_O threshold);
    treated as a soft reading rather than an asymptotic one.
  - **Spectral-type axis** (gapped / III-pp / III-anom): reported on
    autocorrelation-profile inputs; left empty for parameter sweeps.
- `analyze(..., deep=True)` now passes `preprocessing_scale_equivariant=True`
  to `sigma_c_v4.analyze` since FUSE's parameter-sweep preprocessing
  (dedup + smoothing) is scale-equivariant at the observable layer.

### Changed

- `fuse-ui` Streamlit web UI now forces the dark theme via env vars
  on subprocess launch (`STREAMLIT_THEME_BASE=dark` and matching
  background / text / primary colour env vars). The in-page colours
  (`#e0e0e0` / `#9ca3af` on muted-blue boxes) are tuned for dark
  backgrounds; on a light theme they collapsed onto white and became
  unreadable.
- Hugging Face Space and local web UI: the "What the results mean"
  block (grades, tested domains, recommendations) is now visible on
  landing without an extra click (`expanded=True`).
- Hugging Face Space `app.py`: restored the "What the results mean"
  reference block, which had been missing from the HF version.
- `sigma-c-framework` minimum required version bumped from `>=4.0.0`
  to `>=5.0.0`.

### Internal

- Streamlit configuration files for the forced dark theme are
  shipped at `.streamlit/config.toml` and `hf_space/.streamlit/config.toml`
  so the local-run-from-project-root path and the HF Space deployment
  both pick them up.

## [1.0.6] - prior

- Restored the original v1.0.0 fuse-ui look for local installs.

## [1.0.5] - prior

- Simplified `col_card` and added a global stderr marker while hunting
  the HF Space blank-page issue.

## [1.0.4] - prior

- Dropped the CSV-string cache key from the web layer to fix the HF
  Space blank-page issue.

## [1.0.3] - prior

- Visible try/except blocks and stderr diagnostics around card
  rendering.

## [1.0.2] - prior

- Dropped redundant `st.rerun()` after button clicks.

## [1.0.1] - prior

- Keep upload across reruns: click-to-view no longer drops the dataset.

## [1.0.0] - prior

- Polished release: theorem-anchored layer + Streamlit UI.

## [0.1.0] - prior

- Initial PyPI release (renamed from `fusekit`).
