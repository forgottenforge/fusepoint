<!--
Thank you for your contribution! Please fill out the sections below.
Delete sections that do not apply.
-->

## What does this PR do?

A clear description of the change. One concern per PR is preferred.

## Layer

- [ ] Statistical layer (engine.py, score, bootstrap, permutation)
- [ ] Theorem-anchored layer (`deep=True` integration with `sigma_c_v4`)
- [ ] Web UI (`fuse-ui` / `fusepoint/web.py`)
- [ ] Public API (core.py: `analyze`, `scan`, `compare`)
- [ ] Tooling / docs / CI

## Theorem anchor (only if `deep=True` path changes)

If this PR modifies what `deep=True` returns (regime / citations /
gamma_O), state the foundation-paper label backing the change.

- Paper label: `thm:...` / `prop:...` / `def:...`
- Compatible with upstream `sigma_c_v4` >= version: 4.0.0

## Tests

- [ ] Added or updated tests for this change
- [ ] All tests pass locally (`python -m pytest tests/`)

## Backwards compatibility

- [ ] Public API unchanged
- [ ] Public API change is documented in the PR description and called
      out in the next release notes
- [ ] Breaking change requires a major-version bump

## Checklist

- [ ] I have read the [Contributing guidelines](../CONTRIBUTING.md)
- [ ] I have read the [Code of Conduct](../CODE_OF_CONDUCT.md)
- [ ] I have not added `Co-Authored-By:` trailers for AI assistants
- [ ] My contribution is licensed under the project dual-license
