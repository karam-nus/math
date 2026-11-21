"""Part A -- Ordinary Differential Equations (Chapters 1-6)."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import eval_legendre, jv, yv, jn_zeros

from style import figure, note, bare


# ---------------------------------------------------------------- Chapter 1
@figure("ch01-direction-field-euler", size=(11, 4.2))
def ch01(fig, T):
    ax1, ax2 = fig.subplots(1, 2)
    # direction field of y' = y - x
    x, y = np.meshgrid(np.linspace(-2, 3, 21), np.linspace(-2, 4, 21))
    s = y - x
    L = np.sqrt(1 + s**2)
    ax1.quiver(x, y, 1 / L, s / L, color=T.muted, angles="xy", pivot="mid",
               headwidth=0, headlength=0, headaxislength=0, scale=28, width=0.004)
    xs = np.linspace(-2, 3, 400)
    for c in (-1.5, -0.6, -0.2, 0.2, 0.6, 1.5):
        ys = xs + 1 + c * np.exp(xs)
        m = (ys > -2.2) & (ys < 4.2)
        ax1.plot(xs[m], ys[m], color=T.blue, lw=1.8)
    ax1.plot(xs, xs + 1, color=T.orange, lw=2.6)
    note(ax1, -1.85, 3.5, "y′ = y − x", T, fontsize=11, color=T.ink, box=True)
    note(ax1, 0.55, 3.1, "y = x + 1: the one solution\nthat never escapes", T, box=True)
    ax1.set_xlim(-2, 3); ax1.set_ylim(-2, 4)
    ax1.set_title("Direction field: every curve follows the arrows")
    ax1.set_xlabel("x"); ax1.set_ylabel("y")

    # Euler's method on y' = y, y(0) = 1
    t = np.linspace(0, 2, 200)
    ax2.plot(t, np.exp(t), color=T.ink, lw=2.4, label="exact  eˣ")
    for h, col in ((0.5, T.orange), (0.25, T.aqua), (0.1, T.blue)):
        n = int(round(2 / h))
        tt = np.linspace(0, 2, n + 1)
        yy = (1 + h) ** np.arange(n + 1)
        ax2.plot(tt, yy, "-o", color=col, ms=4, lw=1.6, label=f"Euler, h = {h}")
    ax2.set_title("Euler's method: smaller steps hug the truth")
    ax2.set_xlabel("x"); ax2.set_ylabel("y")
    ax2.legend(loc="upper left")
    note(ax2, 1.05, 1.2, "global error ∝ h", T)


# ---------------------------------------------------------------- Chapter 2
@figure("ch02-damping-resonance", size=(11, 4.0))
def ch02(fig, T):
    ax1, ax2 = fig.subplots(1, 2)
    t = np.linspace(0, 15, 600)
    cases = ((0.4, "underdamped  c² < 4mk", T.blue),
             (2.0, "critically damped  c² = 4mk", T.orange),
             (4.0, "overdamped  c² > 4mk", T.aqua))
    for c, lab, col in cases:
        sol = solve_ivp(lambda _t, z: [z[1], -c * z[1] - z[0]], (0, 15), [1, 0], t_eval=t, rtol=1e-8)
        ax2_ = ax1.plot(t, sol.y[0], color=col, label=lab)
    env = np.exp(-0.2 * t)
    ax1.plot(t, env, color=T.muted, lw=1, ls="--")
    ax1.plot(t, -env, color=T.muted, lw=1, ls="--")
    note(ax1, 9.0, 0.28, "envelope e^(−ct/2m)", T)
    ax1.axhline(0, color=T.axis, lw=1)
    ax1.set_title("Free oscillation  my″ + cy′ + ky = 0")
    ax1.set_xlabel("t"); ax1.set_ylabel("displacement y(t)")
    ax1.legend(loc="upper right")

    w = np.linspace(0.01, 2.5, 800)
    for c, col in ((0.1, T.blue), (0.25, T.orange), (0.5, T.aqua), (1.0, T.yellow)):
        A = 1 / np.sqrt((1 - w**2) ** 2 + (c * w) ** 2)
        ax2.plot(w, A, color=col, label=f"c = {c}")
    ax2.axvline(1, color=T.muted, lw=1, ls="--")
    note(ax2, 1.04, 9.0, "ω₀ = √(k/m)", T)
    ax2.set_ylim(0, 10.5)
    ax2.set_title("Forced: amplitude vs driving frequency")
    ax2.set_xlabel("driving frequency ω"); ax2.set_ylabel("amplitude / F₀")
    ax2.legend(loc="upper right", title="damping", title_fontsize=9.5)


# ---------------------------------------------------------------- Chapter 3
@figure("ch03-roots-to-behaviour", size=(11, 4.0))
def ch03(fig, T):
    ax1, ax2 = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.5]})
    roots = [(-1.0, 0.0, "λ = −1  →  e^(−t)", T.blue),
             (0.35, 0.0, "λ = 0.35  →  growth", T.orange),
             (0.0, 2.0, "λ = ±2i  →  cos 2t", T.aqua),
             (-0.3, 2.0, "λ = −0.3 ± 2i  →  damped", T.violet)]
    ax1.axhline(0, color=T.axis, lw=1); ax1.axvline(0, color=T.axis, lw=1)
    ax1.axvspan(-1.6, 0, color=T.blue, alpha=0.06)
    note(ax1, -1.5, -2.6, "left half-plane:\neverything decays", T)
    for re, im, lab, col in roots:
        pts = [(re, im)] if im == 0 else [(re, im), (re, -im)]
        for p in pts:
            ax1.plot(*p, "o", ms=10, color=col, mec=T.surface, mew=2)
    ax1.set_xlim(-1.6, 1.0); ax1.set_ylim(-3, 3)
    ax1.set_xlabel("Re λ"); ax1.set_ylabel("Im λ")
    ax1.set_title("Roots of the characteristic polynomial")

    t = np.linspace(0, 10, 500)
    funcs = [np.exp(-t), np.exp(0.35 * t) / 8, np.cos(2 * t), np.exp(-0.3 * t) * np.cos(2 * t)]
    offs = [6.0, 3.0, 0.0, -3.0]
    for (re, im, lab, col), f, o in zip(roots, funcs, offs):
        ax2.plot(t, f + o, color=col)
        ax2.axhline(o, color=T.grid, lw=0.8)
        note(ax2, 10.2, o, lab, T, va="center")
    ax2.set_yticks([])
    ax2.set_xlim(0, 10)
    ax2.set_xlabel("t")
    ax2.set_title("…decide what every solution does")
    ax2.spines["left"].set_visible(False)


# ---------------------------------------------------------------- Chapter 4
def _portrait(ax, F, T, title, lim=2.0, eig=None):
    g = np.linspace(-lim, lim, 40)
    X, Y = np.meshgrid(g, g)
    U, V = F(X, Y)
    ax.streamplot(X, Y, U, V, color=T.blue, density=0.9, linewidth=1.0, arrowsize=0.9)
    if eig is not None:
        for v in eig:
            v = np.asarray(v, float)
            v = v / np.linalg.norm(v) * lim * 1.5
            ax.plot([-v[0], v[0]], [-v[1], v[1]], color=T.orange, lw=2)
    ax.plot(0, 0, "o", ms=8, color=T.ink, mec=T.surface, mew=1.5)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10.5)
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(T.axis)
    ax.grid(False)


@figure("ch04-phase-portraits", size=(11, 7.0))
def ch04(fig, T):
    axs = fig.subplots(2, 3).ravel()
    lin = lambda A: (lambda X, Y: (A[0][0] * X + A[0][1] * Y, A[1][0] * X + A[1][1] * Y))
    _portrait(axs[0], lin([[1, 0], [0, -1]]), T, "Saddle  (λ₁ < 0 < λ₂)", eig=[(1, 0), (0, 1)])
    _portrait(axs[1], lin([[-2, 0], [0, -0.6]]), T, "Stable node  (λ₁, λ₂ < 0)", eig=[(1, 0), (0, 1)])
    _portrait(axs[2], lin([[-0.3, -1], [1, -0.3]]), T, "Stable spiral  (Re λ < 0)")
    _portrait(axs[3], lin([[0, -1], [1, 0]]), T, "Center  (λ = ±iω)")
    _portrait(axs[4], lin([[0.25, -1], [1, 0.25]]), T, "Unstable spiral  (Re λ > 0)")
    ax = axs[5]
    X, Y = np.meshgrid(np.linspace(-2 * np.pi, 2 * np.pi, 60), np.linspace(-3, 3, 40))
    ax.streamplot(X, Y, Y, -np.sin(X), color=T.blue, density=0.9, linewidth=1.0, arrowsize=0.9)
    for xc in (-np.pi, np.pi):
        ax.plot(xc, 0, "o", ms=8, color=T.orange, mec=T.surface, mew=1.5)
    ax.plot(0, 0, "o", ms=8, color=T.ink, mec=T.surface, mew=1.5)
    note(ax, np.pi, -0.75, "saddle", T, ha="center", fontsize=8.5, box=True)
    note(ax, 0, -0.75, "center", T, ha="center", fontsize=8.5, box=True)
    ax.set_xlim(-2 * np.pi, 2 * np.pi); ax.set_ylim(-3, 3)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title("Nonlinear pendulum  θ″ = −sin θ", fontsize=10.5)
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(T.axis)
    fig.subplots_adjust(wspace=0.12, hspace=0.25)


@figure("ch04-trace-determinant", size=(7.5, 5.2))
def ch04_td(fig, T):
    ax = fig.subplots()
    p = np.linspace(-4, 4, 400)
    q = p**2 / 4
    ax.fill_between(p[p < 0], 0, 6.5, color=T.blue, alpha=0.07)
    ax.fill_between(p[p > 0], 0, 6.5, color=T.red, alpha=0.06)
    ax.fill_between(p, -3, 0, color=T.yellow, alpha=0.08)
    ax.plot(p, q, color=T.orange, lw=2.2)
    ax.axhline(0, color=T.ink2, lw=1.2)
    ax.plot([0, 0], [0, 6.5], color=T.aqua, lw=2.6)
    lab = dict(ha="center", va="center", fontsize=10.5, color=T.ink)
    ax.text(-2.9, 1.0, "stable\nnodes", **lab)
    ax.text(2.9, 1.0, "unstable\nnodes", **lab)
    ax.text(-1.3, 4.5, "stable\nspirals", **lab)
    ax.text(1.3, 4.5, "unstable\nspirals", **lab)
    ax.text(0, -1.6, "saddles  (det < 0, always unstable)", **lab)
    note(ax, 0.12, 6.0, "centers (tr = 0)", T)
    note(ax, 3.0, 2.6, "det = tr²/4\n(repeated λ)", T)
    ax.set_xlim(-4, 4); ax.set_ylim(-3, 6.5)
    ax.set_xlabel("p = trace A = λ₁ + λ₂")
    ax.set_ylabel("q = det A = λ₁λ₂")
    ax.set_title("One map for every 2×2 linear system  y′ = Ay")


# ---------------------------------------------------------------- Chapter 5
@figure("ch05-legendre-bessel", size=(11, 4.0))
def ch05(fig, T):
    ax1, ax2 = fig.subplots(1, 2)
    x = np.linspace(-1, 1, 400)
    for n in range(5):
        ax1.plot(x, eval_legendre(n, x), color=T.series[n], label=f"P{n}(x)".translate(str.maketrans("01234", "₀₁₂₃₄")))
    ax1.set_title("Legendre polynomials  Pₙ(x)")
    ax1.set_xlabel("x"); ax1.legend(ncol=5, loc="lower center", fontsize=8.8, handlelength=1.2, columnspacing=0.9)
    ax1.set_ylim(-1.25, 1.1)

    r = np.linspace(0.001, 20, 800)
    for n, col in zip(range(3), (T.blue, T.orange, T.aqua)):
        ax2.plot(r, jv(n, r), color=col, label=f"J{n}(x)".translate(str.maketrans("012", "₀₁₂")))
    yy = yv(0, r); yy[r < 0.4] = np.nan
    ax2.plot(r, yy, color=T.violet, ls="--", lw=1.6, label="Y₀(x)")
    z = jn_zeros(0, 6)
    ax2.plot(z, 0 * z, "o", ms=7, color=T.ink, mec=T.surface, mew=1.5)
    note(ax2, 7.5, -0.66, "● zeros of J₀ → drum frequencies", T, ha="left", box=True)
    ax2.axhline(0, color=T.axis, lw=1)
    ax2.set_ylim(-0.8, 1.05)
    ax2.set_title("Bessel functions: decaying waves")
    ax2.set_xlabel("x"); ax2.legend(loc="upper right", ncol=2)


# ---------------------------------------------------------------- Chapter 6
@figure("ch06-step-delta-response", size=(11.5, 3.6))
def ch06(fig, T):
    ax1, ax2, ax3 = fig.subplots(1, 3)
    t = np.linspace(0, 6, 1200)
    a = 2.0
    for k, col in ((1.0, T.blue), (0.5, T.orange), (0.2, T.aqua)):
        f = np.where((t >= a) & (t < a + k), 1 / k, 0)
        ax1.plot(t, f, color=col, label=f"width k = {k}")
    ax1.set_ylim(0, 5.5); ax1.set_xlim(0, 4)
    ax1.set_title("Unit-area pulses → δ(t − a)")
    ax1.legend(loc="upper right"); ax1.set_xlabel("t")
    note(ax1, 0.15, 4.0, "area is always 1", T)

    f = t**2
    ax2.plot(t, np.where(t < 4.5, f, np.nan), color=T.muted, lw=1.6, ls="--", label="f(t) = t²")
    g = np.where(t >= a, (t - a) ** 2, 0)
    ax2.plot(t, g, color=T.blue, label="f(t − a)·u(t − a)")
    ax2.axvline(a, color=T.axis, lw=1)
    ax2.set_ylim(0, 12); ax2.set_xlim(0, 6)
    ax2.set_title("t-shifting: delay a signal")
    ax2.legend(loc="upper left"); ax2.set_xlabel("t")
    note(ax2, a + 0.12, 0.5, "a = 2", T)

    tt = np.linspace(0, 20, 2000)
    # y'' + 0.4 y' + 4 y = delta(t - 2), y(0)=y'(0)=0
    wd = np.sqrt(4 - 0.04)
    y = np.where(tt >= a, np.exp(-0.2 * (tt - a)) * np.sin(wd * (tt - a)) / wd, 0)
    ax3.plot(tt, y, color=T.blue)
    ax3.annotate("hammer blow\nat t = 2", xy=(2, 0), xytext=(5.5, 0.38), color=T.ink2, fontsize=9.5,
                 arrowprops=dict(arrowstyle="->", color=T.muted))
    ax3.axhline(0, color=T.axis, lw=1)
    ax3.set_ylim(-0.5, 0.55)
    ax3.set_title("Response: y″ + 0.4y′ + 4y = δ(t − 2)")
    ax3.set_xlabel("t")
    fig.subplots_adjust(wspace=0.25)
