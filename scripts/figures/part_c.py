"""Part C -- Fourier Analysis, PDEs (Chapters 11-12)."""
import numpy as np
from scipy.special import jv, jn_zeros

from style import figure, note


# ---------------------------------------------------------------- Chapter 11
@figure("ch11-fourier-square-wave", size=(11.5, 3.7))
def ch11(fig, T):
    ax1, ax2, ax3 = fig.subplots(1, 3, gridspec_kw={"width_ratios": [1.35, 1, 1.1]})
    x = np.linspace(-np.pi, np.pi, 2000)
    sq = np.sign(np.sin(x))
    ax1.plot(x, sq, color=T.muted, lw=1.4)
    for N, col in ((1, T.orange), (3, T.aqua), (9, T.violet), (49, T.blue)):
        s = sum(4 / (np.pi * n) * np.sin(n * x) for n in range(1, N + 1, 2))
        ax1.plot(x, s, color=col, lw=1.6 if N < 49 else 1.8, label=f"up to n = {N}")
    ax1.set_xlim(-np.pi, np.pi); ax1.set_ylim(-1.45, 2.05)
    ax1.legend(loc="upper left", fontsize=8.5, ncol=2)
    note(ax1, 0.25, -0.35, "Gibbs overshoot ≈ 9%\nnever goes away", T)
    ax1.set_title("Square wave = sum of sines")
    ax1.set_xlabel("x")

    n = np.arange(1, 16)
    b = np.where(n % 2 == 1, 4 / (np.pi * n), 0)
    ax2.vlines(n, 0, b, color=T.blue, lw=2.4)
    ax2.plot(n[b > 0], b[b > 0], "o", ms=7, color=T.blue, mec=T.surface, mew=1.5)
    ax2.set_title("Spectrum  bₙ = 4/(nπ)")
    ax2.set_xlabel("harmonic n"); ax2.set_ylim(0, 1.4)

    rng = np.random.default_rng(3)
    fs, N = 256, 512
    t = np.arange(N) / fs
    sig = np.sin(2 * np.pi * 12 * t) + 0.6 * np.sin(2 * np.pi * 40 * t) + 0.8 * rng.standard_normal(N)
    F = np.abs(np.fft.rfft(sig)) / (N / 2)
    f = np.fft.rfftfreq(N, 1 / fs)
    ax3.plot(f, F, color=T.blue, lw=1.4)
    for f0, a in ((12, "12 Hz"), (40, "40 Hz")):
        note(ax3, f0 + 2, F[np.argmin(abs(f - f0))] + 0.02, a, T)
    ax3.set_xlim(0, 80)
    ax3.set_title("FFT: tones hidden in noise")
    ax3.set_xlabel("frequency (Hz)"); ax3.set_ylabel("|amplitude|")
    fig.subplots_adjust(wspace=0.32)


# ---------------------------------------------------------------- Chapter 12
@figure("ch12-heat-and-wave", size=(11, 4.0))
def ch12(fig, T):
    ax1, ax2 = fig.subplots(1, 2)
    x = np.linspace(0, np.pi, 400)
    f = np.where(x < np.pi / 2, x, np.pi - x)  # triangular initial temperature

    def heat(t, terms=60):
        u = np.zeros_like(x)
        for n in range(1, terms):
            bn = 2 / np.pi * np.trapezoid(f * np.sin(n * x), x)
            u += bn * np.sin(n * x) * np.exp(-(n**2) * t)
        return u

    times = (0, 0.05, 0.2, 0.5, 1.0, 2.0)
    shades = [T.seq[i] for i in (6, 5, 4, 3, 2, 2)]
    for t_, c in zip(times, shades):
        ax1.plot(x, heat(t_), color=c, lw=2, label=f"t = {t_}")
    ax1.set_title("Heat equation: sharp corners melt first")
    ax1.set_xlabel("x"); ax1.set_ylabel("temperature u(x, t)")
    ax1.legend(loc="upper right", fontsize=8.8)
    note(ax1, 0.05, 1.78, "harmonic n decays like e^(−n²t)", T)
    ax1.set_ylim(0, 1.95)

    xs = np.linspace(-6, 6, 800)
    bump = lambda s: np.exp(-4 * s**2)
    for k, t_ in enumerate((0, 0.75, 1.5, 3.0)):
        off = -1.3 * k
        u = 0.5 * (bump(xs - t_) + bump(xs + t_))
        ax2.plot(xs, u + off, color=T.blue)
        ax2.axhline(off, color=T.grid, lw=0.8)
        note(ax2, 6.2, off + 0.05, f"t = {t_}", T, va="bottom")
    ax2.annotate("", xy=(4.0, -3.7), xytext=(1.6, -3.7), arrowprops=dict(arrowstyle="-|>", color=T.orange, lw=2))
    ax2.annotate("", xy=(-4.0, -3.7), xytext=(-1.6, -3.7), arrowprops=dict(arrowstyle="-|>", color=T.orange, lw=2))
    ax2.set_yticks([]); ax2.spines["left"].set_visible(False)
    ax2.set_title("Wave equation: d'Alembert's split  ½[f(x−ct) + f(x+ct)]")
    ax2.set_xlabel("x")


@figure("ch12-drum-modes", size=(11, 3.2))
def ch12_drum(fig, T):
    axs = fig.subplots(1, 4)
    g = np.linspace(-1, 1, 320)
    X, Y = np.meshgrid(g, g)
    R, TH = np.hypot(X, Y), np.arctan2(Y, X)
    th = np.linspace(0, 2 * np.pi, 240)
    base = jn_zeros(0, 1)[0]
    cmap = T.cmap_div().copy()
    cmap.set_bad(T.surface)
    for ax, (n, m) in zip(axs, ((0, 1), (0, 2), (1, 1), (2, 1))):
        a = jn_zeros(n, m)[-1]
        U = np.where(R <= 1, jv(n, a * R) * np.cos(n * TH), np.nan)
        lim = np.nanmax(np.abs(U))
        ax.imshow(U, extent=(-1, 1, -1, 1), origin="lower", cmap=cmap, vmin=-lim, vmax=lim, interpolation="bilinear")
        ax.plot(np.cos(th), np.sin(th), color=T.ink2, lw=1.2)
        ax.set_aspect("equal"); ax.set_axis_off()
        ax.set_title(f"mode ({n},{m})   f/f₀₁ = {a / base:.2f}", fontsize=10, loc="center")
    fig.suptitle("Circular drum: Bessel modes  J_n(α_nm r) cos nθ  (overtones are not integer multiples)",
                 color=T.ink, fontsize=11, fontweight="bold", x=0.02, ha="left", y=1.02)
