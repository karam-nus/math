"""Part B -- Linear Algebra, Vector Calculus (Chapters 7-10)."""
import numpy as np
from matplotlib.patches import FancyArrowPatch

from style import figure, note


# ---------------------------------------------------------------- Chapter 7
def _eq(a, b, c):
    ca = "" if a == 1 else str(a)
    sign = "−" if b < 0 else "+"
    cb = "" if abs(b) == 1 else str(abs(b))
    return f"{ca}x₁ {sign} {cb}x₂ = {c}"

@figure("ch07-three-fates", size=(11, 3.6))
def ch07(fig, T):
    axs = fig.subplots(1, 3)
    x = np.linspace(-1, 4, 50)
    cases = [
        ("Unique solution  (rank 2)", [(1, 1, 3), (1, -1, 1)], "det ≠ 0 → one point"),
        ("No solution  (inconsistent)", [(1, 1, 3), (1, 1, 1)], "parallel lines never meet"),
        ("Infinitely many  (rank 1)", [(1, 1, 2), (2, 2, 4)], "same line twice → a family"),
    ]
    for ax, (title, eqs, msg) in zip(axs, cases):
        cols = (T.blue, T.orange)
        for (a, b, c), col, lw, ls in zip(eqs, cols, (2.4, 2.4), ("-", "--" if title.startswith("Inf") else "-")):
            ax.plot(x, (c - a * x) / b, color=col, lw=lw, ls=ls, label=_eq(a, b, c))
        if title.startswith("Unique"):
            ax.plot(2, 1, "o", ms=10, color=T.ink, mec=T.surface, mew=2)
            note(ax, 2.2, 1.25, "(2, 1)", T, color=T.ink)
        ax.set_xlim(-1, 4); ax.set_ylim(-2, 4)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("x₁")
        ax.legend(loc="upper right", fontsize=8.8)
        note(ax, -0.85, -1.75, msg, T, box=True)
    axs[0].set_ylabel("x₂")
    fig.subplots_adjust(wspace=0.22)


# ---------------------------------------------------------------- Chapter 8
@figure("ch08-eigen-directions", size=(10, 4.6))
def ch08(fig, T):
    ax1, ax2 = fig.subplots(1, 2)
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    th = np.linspace(0, 2 * np.pi, 400)
    circ = np.vstack([np.cos(th), np.sin(th)])
    ell = A @ circ
    ax1.plot(*circ, color=T.muted, lw=1.5)
    ax2.plot(*ell, color=T.muted, lw=1.5)
    angles = np.deg2rad(np.arange(0, 360, 30))
    for a in angles:
        v = np.array([np.cos(a), np.sin(a)])
        w = A @ v
        ax1.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=T.axis, lw=1.2))
        ax2.annotate("", xy=w, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=T.axis, lw=1.2))
    e1 = np.array([1, 1]) / np.sqrt(2)
    e2 = np.array([1, -1]) / np.sqrt(2)
    for e, lam, col in ((e1, 3, T.orange), (e2, 1, T.aqua)):
        ax1.annotate("", xy=e, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=2.8, mutation_scale=16))
        ax2.annotate("", xy=lam * e, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=2.8, mutation_scale=16))
    note(ax1, 0.78, 0.95, "x₁ = (1, 1)/√2", T, color=T.ink)
    note(ax1, 0.78, -0.95, "x₂ = (1, −1)/√2", T, color=T.ink)
    note(ax2, 2.2, 2.35, "Ax₁ = 3x₁", T, color=T.ink)
    note(ax2, 0.85, -0.95, "Ax₂ = 1·x₂", T, color=T.ink)
    for ax, lim in ((ax1, 1.6), (ax2, 3.4)):
        ax.set_aspect("equal"); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.axhline(0, color=T.grid, lw=1); ax.axvline(0, color=T.grid, lw=1)
    ax1.set_title("Unit vectors x")
    ax2.set_title("After A = [[2, 1], [1, 2]]:  Ax")
    note(ax2, -3.3, -3.2, "grey arrows turn; eigenvectors\nonly stretch (by λ = 3 and λ = 1)", T)


# ---------------------------------------------------------------- Chapter 9
@figure("ch09-grad-div-curl", size=(11.5, 3.9))
def ch09(fig, T):
    ax1, ax2, ax3 = fig.subplots(1, 3)
    g = np.linspace(-2.5, 2.5, 200)
    X, Y = np.meshgrid(g, g)
    F = np.exp(-((X - 0.8) ** 2 + Y**2)) + 0.8 * np.exp(-((X + 1) ** 2 + (Y - 0.7) ** 2) / 0.8)
    ax1.contourf(X, Y, F, levels=12, cmap=T.cmap_seq())
    ax1.contour(X, Y, F, levels=12, colors=T.surface, linewidths=0.6)
    gs = np.linspace(-2.2, 2.2, 13)
    Xs, Ys = np.meshgrid(gs, gs)
    Fs = np.exp(-((Xs - 0.8) ** 2 + Ys**2)) + 0.8 * np.exp(-((Xs + 1) ** 2 + (Ys - 0.7) ** 2) / 0.8)
    Gy, Gx = np.gradient(Fs, gs, gs)
    ax1.quiver(Xs, Ys, Gx, Gy, color=T.orange, scale=9, width=0.006)
    ax1.set_title("grad f: uphill, across level curves")

    gq = np.linspace(-2, 2, 11)
    Xq, Yq = np.meshgrid(gq, gq)
    ax2.quiver(Xq, Yq, Xq, Yq, color=T.blue, scale=30, width=0.006)
    circ = np.linspace(0, 2 * np.pi, 200)
    ax2.plot(1.2 * np.cos(circ), 1.2 * np.sin(circ), color=T.muted, lw=1.2, ls="--")
    ax2.set_title("div v > 0: a source  (v = [x, y])")
    note(ax2, -2.3, -2.4, "more flows out than in", T, box=True)

    ax3.quiver(Xq, Yq, -Yq, Xq, color=T.violet, scale=30, width=0.006)
    ax3.add_patch(FancyArrowPatch((0.6, 0), (-0.42, 0.42), connectionstyle="arc3,rad=0.6",
                                  arrowstyle="-|>", mutation_scale=16, color=T.ink, lw=1.8))
    ax3.set_title("curl v ≠ 0: a whirl  (v = [−y, x])")
    note(ax3, -2.3, -2.4, "a paddle-wheel would spin", T, box=True)
    for ax in (ax1, ax2, ax3):
        ax.set_aspect("equal"); ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(T.axis)
    fig.subplots_adjust(wspace=0.08)


# ---------------------------------------------------------------- Chapter 10
@figure("ch10-why-green-works", size=(10.5, 4.4))
def ch10(fig, T):
    ax1, ax2 = fig.subplots(1, 2)
    n = 3
    arrow = dict(arrowstyle="-|>", mutation_scale=11, lw=1.4, shrinkA=0, shrinkB=0)
    for i in range(n):
        for j in range(n):
            x0, y0 = i, j
            pts = [(x0, y0), (x0 + 1, y0), (x0 + 1, y0 + 1), (x0, y0 + 1), (x0, y0)]
            for (a, b), (c, d) in zip(pts[:-1], pts[1:]):
                horiz = b == d
                if horiz:
                    edge_outer = (b == 0) or (b == n)
                else:
                    edge_outer = (a == 0) or (a == n)
                col = T.orange if edge_outer else T.muted
                off = 0.09
                # shift inward so the two opposite arrows on shared edges are both visible
                cx, cy = x0 + 0.5, y0 + 0.5
                sa = (a + (cx - a) * off * 2, b + (cy - b) * off * 2)
                sc = (c + (cx - c) * off * 2, d + (cy - d) * off * 2)
                mid = ((sa[0] + sc[0]) / 2, (sa[1] + sc[1]) / 2)
                start = ((sa[0] + mid[0]) / 2, (sa[1] + mid[1]) / 2)
                end = ((sc[0] + mid[0]) / 2, (sc[1] + mid[1]) / 2)
                ax1.add_patch(FancyArrowPatch(start, end, color=col, **arrow))
            ax1.add_patch(__import__("matplotlib").patches.Rectangle((x0, y0), 1, 1, fill=False, ec=T.grid, lw=1))
    ax1.set_xlim(-0.3, n + 0.3); ax1.set_ylim(-0.75, n + 0.3)
    ax1.set_aspect("equal"); ax1.set_axis_off()
    ax1.set_title("Green's theorem: inner edges cancel")
    note(ax1, n / 2, -0.5, "Σ tiny circulations = circulation round the boundary", T, ha="center")

    g = np.linspace(-2, 2, 200)
    X, Y = np.meshgrid(g, g)
    phi = X**2 / 2 + X * Y / 2 - Y**2 / 3
    ax2.contour(X, Y, phi, levels=14, colors=T.grid, linewidths=1.0)
    gq = np.linspace(-1.8, 1.8, 10)
    Xq, Yq = np.meshgrid(gq, gq)
    ax2.quiver(Xq, Yq, Xq + Yq / 2, Xq / 2 - 2 * Yq / 3, color=T.axis, scale=22, width=0.005)
    A, B = np.array([-1.5, -1.2]), np.array([1.4, 1.3])
    t = np.linspace(0, 1, 100)
    p1 = A[:, None] + (B - A)[:, None] * t
    p2 = np.vstack([A[0] + (B[0] - A[0]) * t, A[1] + (B[1] - A[1]) * t**3 + 1.2 * np.sin(np.pi * t)])
    p3 = np.vstack([A[0] + (B[0] - A[0]) * t**2 + 0.9 * np.sin(np.pi * t), A[1] + (B[1] - A[1]) * t])
    for p, col in ((p1, T.blue), (p2, T.orange), (p3, T.aqua)):
        ax2.plot(*p, color=col, lw=2.4)
    for P, lab in ((A, "A"), (B, "B")):
        ax2.plot(*P, "o", ms=9, color=T.ink, mec=T.surface, mew=2)
        note(ax2, P[0] + 0.1, P[1] - 0.25, lab, T, color=T.ink, fontsize=11)
    ax2.set_aspect("equal"); ax2.set_xlim(-2, 2); ax2.set_ylim(-2, 2)
    ax2.set_xticks([]); ax2.set_yticks([]); ax2.grid(False)
    ax2.set_title("F = grad φ: every path gives φ(B) − φ(A)")
