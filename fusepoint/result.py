"""
StabilityResult — the object returned by analyze().

Holds score, metrics, diagnosis, and provides .show() / .save() for the
Stability Card visualization.
"""

from __future__ import annotations
import numpy as np


class StabilityResult:
    """Result of a fusepoint analysis.

    Attributes
    ----------
    score : int
        Stability Score 0-100.
    grade : str
        STABLE / MODERATE / WARNING / CRITICAL
    diagnosis : str
        One-sentence human-readable diagnosis.
    critical_x : float
        Location of the tipping point.
    ci : tuple[float, float]
        95% bootstrap confidence interval for the tipping point.
    kappa : float
        Peak sharpness (chi_max / mean(chi)).
    p_value : float
        Permutation test p-value.
    safety_margin : float
        Distance to tipping point as fraction of parameter range.
    components : dict
        Score breakdown (detection, clarity, precision, safety).
    x, y : ndarray
        Original data.
    chi : ndarray
        Computed susceptibility.
    current_x : float | None
        Current operating point (if provided).
    x_name : str
        Human-readable name for x-axis.
    y_name : str
        Human-readable name for y-axis.
    """

    def __init__(self, *, score, grade, icon, diagnosis, critical_x,
                 ci_low, ci_high, kappa, p_value, safety_margin,
                 components, x, y, chi, current_x=None, label=None,
                 x_name=None, y_name=None):
        self.score = score
        self.grade = grade
        self.icon = icon
        self.diagnosis = diagnosis
        self.critical_x = critical_x
        self.ci = (ci_low, ci_high)
        self.kappa = kappa
        self.p_value = p_value
        self.safety_margin = safety_margin
        self.components = components
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.chi = np.asarray(chi)
        self.current_x = current_x
        self.label = label or "System"
        self.x_name = x_name or "Parameter"
        self.y_name = y_name or "Observable"

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            f"StabilityResult(score={self.score}, grade='{self.grade}', "
            f"critical_x={self.critical_x:.4g}, "
            f"kappa={self.kappa:.2f}, p={self.p_value:.4f})"
        )

    def summary(self) -> str:
        """Multi-line text summary."""
        xn = self.x_name
        lines = [
            f"  Stability Score:  {self.score} / 100  [{self.grade}]",
            f"  Tipping point:    {xn} = {self.critical_x:.4g}  "
            f"(95% CI: {self.ci[0]:.4g} – {self.ci[1]:.4g})",
            f"  Peak sharpness:   k = {self.kappa:.2f}",
            f"  Significance:     p = {self.p_value:.4f}  "
            f"({'significant' if self.p_value < 0.05 else 'not significant'})",
            f"  Safety margin:    {self.safety_margin * 100:.0f}%",
            f"",
            f"  {self.diagnosis}",
        ]
        return "\n".join(lines)

    def show(self, **kwargs):
        """Display the Stability Card interactively."""
        from fusepoint.card import render_card
        fig = render_card(self, **kwargs)
        import matplotlib.pyplot as plt
        plt.show()
        return fig

    def save(self, path: str = "stability_card.png", dpi: int = 200, **kwargs):
        """Save the Stability Card as PNG."""
        from fusepoint.card import render_card
        fig = render_card(self, **kwargs)
        fig.savefig(path, dpi=dpi,
                    facecolor=fig.get_facecolor(), edgecolor="none")
        import matplotlib.pyplot as plt
        plt.close(fig)
        return path
