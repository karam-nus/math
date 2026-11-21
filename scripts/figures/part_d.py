"""Part D -- Complex Analysis (Chapters 13-18)."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Wedge

from style import figure, note


def _plane(ax, T, lim=2.0):
    ax.axhline(0, color=T.axis, lw=1); ax.axvline(0, color=T.axis, lw=1)
    ax.set_aspect("equal"); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)


# ---------------------------------------------------------------- Chapter 13
@figure("ch13-roots-domain-coloring", size=(11, 4.6))
def ch13(fig, T):
    ax1, ax2 = fig.subplots(1, 2)
    _plane(ax1, T, 1.5)
    th = np.linspace(0, 2 * np.pi, 400)
    ax1.plot(np.cos(th), np.sin(th), color=T.muted, lw=1.2)
    n = 6
    k = np.arange(n)
    z = np.exp(2j * np.pi * k / n)
    ax1.fill(z.real, z.imag, color=T.blue, alpha=0.08)
    ax1.plot(np.append(z.real, z.real[0]), np.append(z.imag, z.imag[0]), color=T.blue, lw=1.4)
    ax1.plot(z.real, z.imag, "o", ms=10, color=T.blue, mec=T.surface, mew=2)
    for kk, zz in zip(k, z):
        note(ax1, 1.2 * zz.real, 1.2 * zz.imag, ["1", "ω", "ω²", "ω³", "ω⁴", "ω⁵"][kk], T, ha="center", va="center", color=T.ink, fontsize=11)
    ax1.add_patch(FancyArrowPatch((0.55, 0), (0.55 * np.cos(np.pi / 3), 0.55 * np.sin(np.pi / 3)),
                                  connectionstyle="arc3,rad=0.45", arrowstyle="-|>", mutation_scale=14, color=T.orange, lw=2))
    note(ax1, -0.55, -0.3, "multiplying by ω\n= rotating by 60°", T, color=T.ink, box=True)
    ax1.set_title("6th roots of unity  zⁿ = 1:  ω = e^(2πi/6)")
    ax1.set_xlabel("Re z"); ax1.set_ylabel("Im z")

    g = np.linspace(-2.5, 2.5, 700)
    X, Y = np.meshgrid(g, g)
    Z = X + 1j * Y
    F = (Z**2 - 1) * (Z - 2 - 1j) ** 2 / (Z**2 + 2 + 2j)
    hue = (np.angle(F) + np.pi) / (2 * np.pi)
    rgb = plt.get_cmap("twilight_shifted")(hue)[..., :3]
    mag = np.log2(np.abs(F) + 1e-12)
    shade = 0.78 + 0.22 * (mag - np.floor(mag))
    ax2.imshow(rgb * shade[..., None], extent=(-2.5, 2.5, -2.5, 2.5), origin="lower", interpolation="bilinear")
    ax2.set_title("Domain colouring of f(z)  (hue = arg f)")
    for p, lab in ((1, "zero"), (-1, "zero"), (2 + 1j, "double\nzero")):
        note(ax2, p.real + 0.12, p.imag + 0.12, lab, T, color="#0b0b0b", fontsize=8.5, bbox=dict(boxstyle="round,pad=0.2", fc="#fcfcfb", ec="none", alpha=0.85))
    poles = np.sqrt(-2 - 2j) * np.array([1, -1])
    for p in poles:
        note(ax2, p.real + 0.12, p.imag - 0.3, "pole", T, color="#0b0b0b", fontsize=8.5, bbox=dict(boxstyle="round,pad=0.2", fc="#fcfcfb", ec="none", alpha=0.85))
    ax2.set_xlabel("Re z"); ax2.grid(False)
    note(ax2, -2.4, -2.4, "f = (z²−1)(z−2−i)² / (z²+2+2i)", T, color="#0b0b0b", fontsize=8.5, bbox=dict(boxstyle="round,pad=0.2", fc="#fcfcfb", ec="none", alpha=0.85))


# ---------------------------------------------------------------- Chapter 14
@figure("ch14-contour-deformation", size=(10.5, 4.3))
def ch14(fig, T):
    ax1, ax2 = fig.subplots(1, 2)
    th = np.linspace(0, 2 * np.pi, 400)
    r = 1.7 + 0.25 * np.cos(3 * th) + 0.15 * np.sin(2 * th)
    ax1.plot(r * np.cos(th), r * np.sin(th) * 0.8, color=T.blue, lw=2.4)
    ax1.add_patch(FancyArrowPatch((1.95 * np.cos(0.6), 1.95 * 0.8 * np.sin(0.6)),
                                  (1.62 * np.cos(1.0), 1.62 * 0.8 * np.sin(1.0)),
                                  arrowstyle="-|>", mutation_scale=16, color=T.blue, lw=0))
    sing = [(-0.6, 0.3), (0.7, -0.2)]
    for (a, b), lab in zip(sing, ("z₁", "z₂")):
        ax1.add_patch(Circle((a, b), 0.28, fill=False, ec=T.orange, lw=2))
        ax1.plot(a, b, "x", ms=9, mew=2.4, color=T.ink)
        note(ax1, a + 0.32, b + 0.2, lab, T, color=T.ink)
    note(ax1, 1.2, 1.45, "C", T, color=T.ink, fontsize=12)
    ax1.set_aspect("equal"); ax1.set_xlim(-2.3, 2.3); ax1.set_ylim(-1.9, 1.9); ax1.set_axis_off()
    ax1.set_title("Shrink C onto the singularities")
    note(ax1, 0, -1.85, "∮_C f dz = ∮_(around z₁) + ∮_(around z₂)", T, ha="center")

    _plane(ax2, T, 2.0)
    ax2.set_xticks([]); ax2.set_yticks([]); ax2.grid(False)
    A, B = (-1.5, 0.0), (1.5, 0.0)
    t = np.linspace(0, np.pi, 200)
    ax2.plot(1.5 * np.cos(np.pi - t), 1.1 * np.sin(t), color=T.blue, lw=2.4, label="upper path")
    ax2.plot(1.5 * np.cos(np.pi - t), -1.1 * np.sin(t), color=T.aqua, lw=2.4, label="lower path")
    ax2.plot(0, 0, "x", ms=10, mew=2.6, color=T.ink)
    note(ax2, 0.1, 0.12, "1/z blows up here", T)
    for P, lab in ((A, "A"), (B, "B")):
        ax2.plot(*P, "o", ms=9, color=T.ink, mec=T.surface, mew=2)
        note(ax2, P[0] - 0.05, P[1] - 0.32, lab, T, color=T.ink, ha="center", fontsize=11)
    ax2.legend(loc="upper right")
    ax2.set_title("Around a singularity, path matters")
    note(ax2, 0, -1.8, "∫ upper − ∫ lower = ∮ dz/z = −2πi  (clockwise)", T, ha="center")


# ---------------------------------------------------------------- Chapter 15
@figure("ch15-taylor-radius", size=(11.5, 3.8))
def ch15(fig, T):
    ax1, ax2, ax3 = fig.subplots(1, 3, gridspec_kw={"width_ratios": [1.2, 1.2, 0.85]})
    x = np.linspace(-2 * np.pi, 2 * np.pi, 600)
    ax1.plot(x, np.sin(x), color=T.ink, lw=2.6, label="sin x")
    from math import factorial
    for deg, col in ((1, T.orange), (3, T.aqua), (7, T.violet), (13, T.blue)):
        p = sum((-1) ** k * x ** (2 * k + 1) / factorial(2 * k + 1) for k in range((deg - 1) // 2 + 1))
        ax1.plot(x, p, color=col, lw=1.6, label=f"degree {deg}")
    ax1.set_ylim(-2.2, 2.2); ax1.set_xlim(-2 * np.pi, 2 * np.pi)
    ax1.legend(loc="lower left", fontsize=8.5, ncol=2, framealpha=0.9, frameon=True, facecolor=T.surface, edgecolor="none")
    ax1.set_title("Taylor polynomials of sin x")
    ax1.set_xlabel("x")

    xx = np.linspace(-2, 2, 600)
    ax2.plot(xx, 1 / (1 + xx**2), color=T.ink, lw=2.6, label="1/(1+x²)")
    for deg, col in ((4, T.orange), (10, T.aqua), (20, T.blue)):
        p = sum((-1) ** k * xx ** (2 * k) for k in range(deg // 2 + 1))
        ax2.plot(xx, p, color=col, lw=1.6, label=f"degree {deg}")
    ax2.axvspan(-2, -1, color=T.red, alpha=0.07); ax2.axvspan(1, 2, color=T.red, alpha=0.07)
    ax2.set_ylim(-0.5, 1.6)
    ax2.legend(loc="lower center", fontsize=8.5, ncol=2)
    note(ax2, -1.95, 1.45, "diverges for |x| > 1", T, box=True)
    ax2.set_title("…but this smooth function's series fails")
    ax2.set_xlabel("x")

    _plane(ax3, T, 1.6)
    ax3.add_patch(Circle((0, 0), 1, color=T.blue, alpha=0.12))
    ax3.add_patch(Circle((0, 0), 1, fill=False, ec=T.blue, lw=2))
    for s in (1, -1):
        ax3.plot(0, s, "x", ms=10, mew=2.6, color=T.red)
        note(ax3, 0.15, s + (0.1 if s > 0 else -0.3), "i" if s > 0 else "−i", T, color=T.ink, fontsize=11)
    ax3.plot([-1, 1], [0, 0], color=T.orange, lw=3)
    ax3.set_title("Why: poles at ±i", fontsize=11)
    ax3.set_xlabel("radius = distance to nearest pole")


# ---------------------------------------------------------------- Chapter 16
@figure("ch16-laurent-residues", size=(10.5, 4.4))
def ch16(fig, T):
    ax1, ax2 = fig.subplots(1, 2)
    _plane(ax1, T, 3.0)
    ax1.add_patch(Circle((0, 0), 1, color=T.blue, alpha=0.18, lw=0))
    ax1.add_patch(Wedge((0, 0), 2, 0, 360, width=1, color=T.aqua, alpha=0.18, lw=0))
    ax1.add_patch(Wedge((0, 0), 3.0, 0, 360, width=1, color=T.yellow, alpha=0.16, lw=0))
    for p in (1, 2):
        ax1.plot(p, 0, "x", ms=10, mew=2.6, color=T.red)
    note(ax1, 0, 0.45, "|z| < 1\nTaylor", T, ha="center", color=T.ink)
    note(ax1, -1.5, 0.15, "1 < |z| < 2\nLaurent", T, ha="center", color=T.ink)
    note(ax1, -2.35, 1.6, "|z| > 2\nLaurent", T, ha="center", color=T.ink)
    ax1.set_title("f = 1/((z−1)(z−2)): three series, three rings")
    ax1.set_xlabel("Re z"); ax1.set_ylabel("Im z")

    R = 2.0
    t = np.linspace(0, np.pi, 300)
    ax2.plot([-R, R], [0, 0], color=T.blue, lw=2.6)
    ax2.plot(R * np.cos(t), R * np.sin(t), color=T.blue, lw=2.6)
    ax2.add_patch(FancyArrowPatch((-0.4, 0), (0.4, 0), arrowstyle="-|>", mutation_scale=16, color=T.blue))
    ax2.add_patch(FancyArrowPatch((R * np.cos(0.9), R * np.sin(0.9)), (R * np.cos(1.1), R * np.sin(1.1)),
                                  arrowstyle="-|>", mutation_scale=16, color=T.blue))
    ax2.plot(0, 1, "x", ms=10, mew=2.6, color=T.red); note(ax2, 0.12, 1.1, "z = i (inside)", T, color=T.ink)
    ax2.plot(0, -1, "x", ms=10, mew=2.6, color=T.muted); note(ax2, 0.12, -1.25, "z = −i (outside)", T)
    ax2.axhline(0, color=T.axis, lw=1)
    ax2.set_aspect("equal"); ax2.set_xlim(-2.6, 2.6); ax2.set_ylim(-1.7, 2.5)
    ax2.set_xticks([]); ax2.set_yticks([]); ax2.grid(False)
    ax2.set_title("∫ dx/(1+x²) = 2πi·Res(i) = π")
    note(ax2, 0, 2.2, "semicircle contributes → 0 as R → ∞", T, ha="center")


# ---------------------------------------------------------------- Chapter 17
@figure("ch17-conformal-maps", size=(11.5, 3.9))
def ch17(fig, T):
    ax1, ax2, ax3 = fig.subplots(1, 3)
    s = np.linspace(0, 1.6, 300)
    for c in np.linspace(0.2, 1.6, 8):
        w = (c + 1j * s) ** 2
        ax1.plot(w.real, w.imag, color=T.blue, lw=1.3)
        w = (s + 1j * c) ** 2
        ax1.plot(w.real, w.imag, color=T.orange, lw=1.3)
    ax1.set_aspect("equal"); ax1.set_xlim(-2.6, 2.6); ax1.set_ylim(-0.2, 5.2)
    ax1.set_title("w = z²: grid → parabolas")

    y = np.linspace(-np.pi, np.pi, 400)
    xg = np.linspace(-1.2, 0.8, 300)
    for c in np.linspace(-1.2, 0.8, 7):
        w = np.exp(c + 1j * y)
        ax2.plot(w.real, w.imag, color=T.blue, lw=1.3)
    for c in np.linspace(-np.pi, np.pi, 13)[:-1]:
        w = np.exp(xg + 1j * c)
        ax2.plot(w.real, w.imag, color=T.orange, lw=1.3)
    ax2.set_aspect("equal"); ax2.set_xlim(-2.4, 2.4); ax2.set_ylim(-2.4, 2.4)
    ax2.set_title("w = eᶻ: lines → circles & rays")

    th = np.linspace(0, 2 * np.pi, 600)
    c0 = -0.1 + 0.12j
    rad = abs(1 - c0)
    for k, rr in enumerate(np.linspace(rad, 2.2 * rad, 6)):
        z = c0 + rr * np.exp(1j * th)
        w = z + 1 / z
        ax3.plot(w.real, w.imag, color=T.blue if k else T.ink, lw=1.2 if k else 0)
        if k == 0:
            ax3.fill(w.real, w.imag, color=T.orange, alpha=0.9)
    for ang in np.linspace(0, 2 * np.pi, 25)[:-1]:
        z = c0 + np.linspace(rad, 2.2 * rad, 60) * np.exp(1j * ang)
        w = z + 1 / z
        ax3.plot(w.real, w.imag, color=T.muted, lw=0.8)
    ax3.set_aspect("equal"); ax3.set_xlim(-3.2, 3.2); ax3.set_ylim(-2.6, 2.6)
    ax3.set_title("Joukowski w = z + 1/z: circle → airfoil")
    for ax in (ax1, ax2, ax3):
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(T.axis)
    note(ax1, -2.45, 4.75, "right angles survive", T, box=True)


# ---------------------------------------------------------------- Chapter 18
@figure("ch18-potential-flow", size=(11, 4.4))
def ch18(fig, T):
    ax1, ax2 = fig.subplots(1, 2)
    g = np.linspace(-3, 3, 500)
    X, Y = np.meshgrid(g, g)
    Z = X + 1j * Y
    with np.errstate(divide="ignore", invalid="ignore"):
        Fz = Z + 1 / Z
    inside = np.abs(Z) < 1
    psi = np.where(inside, np.nan, Fz.imag)
    phi = np.where(inside, np.nan, Fz.real)
    ax1.contour(X, Y, phi, levels=np.linspace(-6, 6, 25), colors=T.muted, linewidths=0.8, linestyles="--")
    ax1.contour(X, Y, psi, levels=np.linspace(-3, 3, 25), colors=T.blue, linewidths=1.4, linestyles="-")
    ax1.add_patch(Circle((0, 0), 1, color=T.orange, alpha=0.9))
    ax1.set_title("Flow past a cylinder:  F(z) = z + 1/z")
    note(ax1, -2.9, -2.85, "solid: streamlines Im F   dashed: equipotentials Re F", T, box=True, fontsize=8.8)

    a = 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        W = np.log((Z - a) / (Z + a))
    ax2.contour(X, Y, W.real, levels=np.linspace(-3, 3, 21), colors=T.orange, linewidths=1.3)
    ax2.contour(X, Y, W.imag, levels=np.linspace(-np.pi, np.pi, 25), colors=T.blue, linewidths=1.0)
    ax2.plot([a], [0], "o", ms=11, color=T.red, mec=T.surface, mew=2)
    ax2.plot([-a], [0], "o", ms=11, color=T.blue, mec=T.surface, mew=2)
    note(ax2, a + 0.15, 0.15, "+", T, color=T.ink, fontsize=12)
    note(ax2, -a - 0.35, 0.15, "−", T, color=T.ink, fontsize=12)
    ax2.set_title("Two line charges:  F(z) = log((z−1)/(z+1))")
    note(ax2, -2.9, -2.85, "orange: equipotentials   blue: field lines", T, box=True, fontsize=8.8)
    for ax in (ax1, ax2):
        ax.set_aspect("equal"); ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(T.axis)
