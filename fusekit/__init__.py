"""
Fuse — Find where your system breaks, before it does.

Usage:
    from fusekit import analyze
    card = analyze(x, y, x_name="Load", y_name="Latency")
    card = analyze(df, x="load", y="latency")   # DataFrame mode
    print(card.score)
    card.save("stability.png")

https://www.forgottenforge.xyz
"""

from fusekit.core import analyze, scan, compare
from fusekit.result import StabilityResult
from fusekit.parsers import parse_json, detect_x_column

__version__ = "0.1.0"
__all__ = [
    "analyze",
    "scan",
    "compare",
    "StabilityResult",
    "parse_json",
    "detect_x_column",
]
