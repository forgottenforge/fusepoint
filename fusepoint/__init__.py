"""
Fuse — Find where your system breaks, before it does.

Usage:
    from fusepoint import analyze
    card = analyze(x, y, x_name="Load", y_name="Latency")
    card = analyze(df, x="load", y="latency")   # DataFrame mode
    print(card.score)
    card.save("stability.png")

https://www.forgottenforge.xyz
"""

from fusepoint.core import analyze, scan, compare
from fusepoint.result import StabilityResult
from fusepoint.parsers import parse_json, detect_x_column

__version__ = "1.1.0"
__paper_doi__ = "10.5281/zenodo.20548818"
__paper_url__ = "https://doi.org/10.5281/zenodo.20548818"
__all__ = [
    "analyze",
    "scan",
    "compare",
    "StabilityResult",
    "parse_json",
    "detect_x_column",
]
