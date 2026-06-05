---
name: Feature request
about: Propose a new feature or enhancement
title: "[feat] "
labels: enhancement
assignees: ''
---

## Problem / motivation

What's the underlying problem you'd like solved? Why does it matter?

## Proposed solution

A clear description of the feature you'd like to see. If possible, sketch
the API:

```python
# what calling the new feature would look like
```

## Layer

- [ ] Statistical layer (`compute_susceptibility`, score, bootstrap, permutation)
- [ ] Theorem-anchored layer (`deep=True` -> regime, citations, gamma_O)
- [ ] Web UI (`fuse-ui`)
- [ ] Tooling / docs / CI

## Theorem anchor (only if it touches `deep=True`)

If this would change the `deep=True` output (regime / citations /
gamma_O), which theorem or proposition from the foundation paper
([doi:10.5281/zenodo.20548818](https://doi.org/10.5281/zenodo.20548818))
should back it? Use the label form (e.g. `thm:trichotomy-geometric`).

## Alternatives considered

What other approaches have you tried or thought about?

## Additional context

References, related work, links to similar features in other tools.
