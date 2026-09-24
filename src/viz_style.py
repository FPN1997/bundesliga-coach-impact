"""
Shared chart styling for every static PNG in outputs/ and docs/.

The palette is a validated categorical/diverging set (checked for colour-
blind separation and contrast against the light chart surface): categorical
slots 1-3 stay distinguishable in every pairing, and the diverging scale is
two hues with a neutral grey midpoint -- never a rainbow like red-yellow-
green, which is both hard to read for colour-blind viewers and implies a
"middle" colour that means something. Text always uses the ink colours,
never a series colour.
"""

from __future__ import annotations

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
DIVERGING = ("#e34948", "#f0efec", "#2a78d6")  # below average -> neutral -> above average


def style_axes(ax, grid_axis: str | None = "y") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=INK_2, labelsize=9)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def title(ax, text: str) -> None:
    ax.set_title(text, loc="left", fontsize=11.5, color=INK, fontweight="semibold")


def diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("bca_diverging", list(DIVERGING))


def centered_norm(values, center: float):
    """Symmetric diverging norm around `center` (e.g. the league average), so
    equal distances above and below it get equally strong colour."""
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm
    v = np.asarray(values, dtype=float)
    half = np.nanmax(np.abs(v - center))
    return TwoSlopeNorm(vcenter=center, vmin=center - half, vmax=center + half)
