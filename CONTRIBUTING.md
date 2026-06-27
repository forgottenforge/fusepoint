# Contributing to FUSE (fusepoint)

Thank you for your interest in contributing. The project is dual-licensed
(AGPL-3.0-or-later OR Commercial) and is developed at ForgottenForge.

## Before you start

- Read the **[Code of Conduct](CODE_OF_CONDUCT.md)** — it applies to all
  interactions on this repository.
- Skim the **[README](README.md)** to understand the two layers:
  - the **statistical layer** (bootstrap + permutation + 0-100 stability
    score), and
  - the **theorem-anchored layer** activated via `deep=True`, which calls
    [`sigma-c-framework`](https://github.com/forgottenforge/sigmacore) and
    is backed by the foundation paper:
    Wurm, M. C. (2026). *Operational scale selection: axioms, spectral
    concentration, and a regime trichotomy.* Zenodo,
    [doi:10.5281/zenodo.20548818](https://doi.org/10.5281/zenodo.20548818).

## How to contribute

### Bug reports

Use the **Bug report** issue template. Include:

- FUSE version (`python -c "import fusepoint; print(fusepoint.__version__)"`)
- Operating system and Python version
- Minimal reproducible example (≤ 30 lines)
- Expected vs observed behaviour

### Feature requests

Use the **Feature request** issue template. For changes that touch the
`deep=True` layer (regime classification, citations, gamma_O), specify
which theorem or proposition in the foundation paper the new behaviour
would cite.

### Pull requests

1. Fork the repo and create a feature branch from `main`:
   `git checkout -b feat/short-description`
2. Make your change. Keep it focused — one concern per PR.
3. Run the test suite locally:
   ```bash
   python -m pytest tests/ -v
   ```
4. For statistical-layer changes: preserve the public API of
   `analyze`, `scan`, `compare`. Changes to the stability-score formula
   are major-version-bumping.
5. For theorem-layer changes (`deep=True` path): match the upstream
   `sigma_c_v4` Result contract; do not silently mutate `regime` or
   `citations` between minor versions.
6. Commit with a clear message (see *Commit style* below).
7. Open the PR using the pull-request template.

### Commit style

- One concern per commit.
- First line: short imperative ≤ 72 chars, e.g.
  `engine: tighten bootstrap CI for low-N inputs`.
- Body (optional): why, not what.
- Sign-off if you can: `git commit -s` adds `Signed-off-by:`.
- **Do not add `Co-Authored-By:` trailers for AI assistants.**

### Tests

- New public surfaces require a test in `tests/`.
- The statistical layer (`compute_susceptibility`, `compute_kappa`,
  bootstrap, permutation) has reproducibility tests with fixed seeds —
  do not break them.
- Do not commit fixture data that includes private or unredistributable
  material.

### Code style

- Python: PEP 8, line length 100. Consistency with the surrounding file
  is appreciated.
- Type hints encouraged on public surfaces.
- Docstrings: short, with examples for user-facing surfaces.

## Development setup

```bash
git clone https://github.com/forgottenforge/fusepoint
cd fusepoint
python -m venv .venv
.venv\Scripts\activate              # Windows
# source .venv/bin/activate         # Linux/macOS
pip install -e ".[ui,dev]"
python -m pytest
```

## Reporting security issues

Do not open a public issue for security problems. See
[`SECURITY.md`](SECURITY.md) for responsible-disclosure instructions.

## License of contributions

By submitting a contribution, you agree to license it under the
dual-license model described in [`LICENSE`](LICENSE) (AGPL-3.0-or-later
OR Commercial; see [`license_AGPL.txt`](license_AGPL.txt) and
[`license_COMMERCIAL.txt`](license_COMMERCIAL.txt)). For substantial
contributions we may ask you to sign a contributor license agreement
(CLA) before merging.
