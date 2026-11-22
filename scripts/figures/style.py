"""Shared plotting style for the knowledge-base figures.

Every figure is rendered twice -- once for a light page and once for a dark
page -- so the Markdown can serve the right one via <picture>.  Colours follow
a single validated palette (categorical order is fixed, never cycled).
"""
from __future__ import annotations

import functools
import os
import re
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(ROOT, "assets", "figures")


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    ink: str
    ink2: str
    muted: str
    grid: str
    axis: str
    series: tuple
    seq: tuple  # sequential ramp, "near zero" first
    div_mid: str
    good: str = "#0ca30c"
    critical: str = "#d03b3b"
    extra: dict = field(default_factory=dict)

    # categorical slots by name, in the fixed palette order
    @property
    def blue(self):
        return self.series[0]

    @property
    def orange(self):
        return self.series[1]

    @property
    def aqua(self):
        return self.series[2]

    @property
    def yellow(self):
        return self.series[3]

    @property
    def magenta(self):
        return self.series[4]

    @property
    def green(self):
        return self.series[5]

    @property
    def violet(self):
        return self.series[6]

    @property
    def red(self):
        return self.series[7]

    def cmap_seq(self):
        return LinearSegmentedColormap.from_list(f"seq-{self.name}", list(self.seq))

    def cmap_div(self):
        blue_end = "#104281" if self.name == "light" else "#86b6ef"
        red_end = "#a32424" if self.name == "light" else "#f09a9a"
        return LinearSegmentedColormap.from_list(
            f"div-{self.name}", [blue_end, self.series[0], self.div_mid, self.series[7], red_end]
        )


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    ink="#0b0b0b",
    ink2="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"),
    seq=("#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"),
    div_mid="#f0efec",
)

DARK = Theme(
    name="dark",
    surface="#1a1a19",
    ink="#f0efec",
    ink2="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    series=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"),
    seq=("#1a1a19", "#104281", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#b7d3f6"),
    div_mid="#383835",
)

THEMES = (LIGHT, DARK)

REGISTRY: dict[str, callable] = {}


def apply_rc(T: Theme):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.labelsize": 10.5,
            "axes.facecolor": T.surface,
            "figure.facecolor": T.surface,
            "savefig.facecolor": T.surface,
            "axes.edgecolor": T.axis,
            "axes.labelcolor": T.ink2,
            "axes.titlecolor": T.ink,
            "text.color": T.ink,
            "xtick.color": T.muted,
            "ytick.color": T.muted,
            "xtick.labelcolor": T.ink2,
            "ytick.labelcolor": T.ink2,
            "axes.grid": True,
            "grid.color": T.grid,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "legend.frameon": False,
            "legend.fontsize": 9.5,
            "legend.labelcolor": T.ink2,
            "mathtext.fontset": "dejavusans",
            "svg.fonttype": "path",
            "svg.hashsalt": "kreyszig-kb",
            "figure.dpi": 100,
        }
    )


def figure(name: str, size=(8, 4.2)):
    """Register a figure builder ``fn(fig, T)``; rendered once per theme."""

    def deco(fn):
        @functools.wraps(fn)
        def build():
            os.makedirs(OUT_DIR, exist_ok=True)
            for T in THEMES:
                apply_rc(T)
                fig = plt.figure(figsize=size)
                fn(fig, T)
                path = os.path.join(OUT_DIR, f"{name}-{T.name}.svg")
                fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0.18,
                            metadata={"Date": None, "Creator": None})
                plt.close(fig)
                _shrink(path)
            return name

        REGISTRY[name] = build
        return build

    return deco


_FLOAT = re.compile(r"(-?\d+\.\d{2,})")
_PATH_D = re.compile(r'(\sd=")([^"]*)(")')


def _shrink(path):
    """Round path coordinates to 0.1 pt -- invisible on screen, much smaller files.

    Only touches ``d="..."`` path data; transforms (e.g. glyph scale 0.015625)
    must keep full precision or the text disappears.
    """
    with open(path, encoding="utf-8") as f:
        svg = f.read()
    rnd = lambda m: f"{float(m.group(1)):.1f}"
    svg = _PATH_D.sub(lambda m: m.group(1) + _FLOAT.sub(rnd, m.group(2)) + m.group(3), svg)
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg)


def note(ax, x, y, text, T, **kw):
    """Annotation in secondary ink (text never wears the series colour)."""
    kw.setdefault("fontsize", 9.5)
    kw.setdefault("color", T.ink2)
    if kw.pop("box", False):
        kw["bbox"] = dict(boxstyle="round,pad=0.25", fc=T.surface, ec="none", alpha=0.92)
    return ax.text(x, y, text, **kw)


def clean(ax, T, grid=True, box=False):
    ax.grid(grid)
    if box:
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(T.axis)
    return ax


def bare(ax):
    """Diagram axes: no ticks, no grid, no spines, equal aspect."""
    ax.set_axis_off()
    ax.set_aspect("equal")
    return ax
