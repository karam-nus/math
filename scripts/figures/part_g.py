"""Part G -- Probability, Statistics (Chapters 24-25)."""
import numpy as np
from scipy import stats

from style import figure, note


# ---------------------------------------------------------------- Chapter 24
@figure("ch24-distributions-lln", size=(11.5, 3.7))
def ch24(fig, T):
    ax1, ax2, ax3 = fig.subplots(1, 3)
    n, p = 20, 0.3
    k = np.arange(0, 17)
    ax1.bar(k - 0.2, stats.binom.pmf(k, n, p), width=0.4, color=T.blue, label="Binomial(20, 0.3)")
    ax1.bar(k + 0.2, stats.poisson.pmf(k, n * p), width=0.4, color=T.orange, label="Poisson(6)")
    xs = np.linspace(0, 16, 300)
    ax1.plot(xs, stats.norm.pdf(xs, n * p, np.sqrt(n * p * (1 - p))), color=T.ink, lw=1.8, label="Normal approx.")
    ax1.set_title("Three distributions, one shape")
    ax1.set_xlabel("k"); ax1.legend(loc="upper right", fontsize=8.6)

    x = np.linspace(-4, 4, 600)
    ax2.plot(x, stats.norm.pdf(x), color=T.ink, lw=2)
    for s, alpha, lab in ((3, 0.12, "99.7%"), (2, 0.2, "95%"), (1, 0.35, "68%")):
        m = np.abs(x) <= s
        ax2.fill_between(x[m], 0, stats.norm.pdf(x[m]), color=T.blue, alpha=alpha, lw=0)
    for s, lab, y in ((1, "68.3%", -0.025), (2, "95.4%", -0.055), (3, "99.7%", -0.085)):
        ax2.annotate("", xy=(s, y), xytext=(-s, y), arrowprops=dict(arrowstyle="<->", color=T.ink2, lw=1.2))
        note(ax2, s + 0.15, y, lab, T, va="center", color=T.ink, fontsize=9)
    ax2.axhline(0, color=T.axis, lw=1)
    ax2.set_ylim(-0.1, 0.43); ax2.set_xlim(-4, 4.6)
    ax2.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax2.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    ax2.set_xticklabels(["−3σ", "−2σ", "−σ", "μ", "σ", "2σ", "3σ"], fontsize=8.5)
    ax2.set_title("Normal: the 68–95–99.7 rule")

    rng = np.random.default_rng(7)
    N = 2000
    for col in (T.blue, T.orange, T.aqua, T.violet):
        flips = rng.integers(0, 2, N)
        ax3.plot(np.arange(1, N + 1), np.cumsum(flips) / np.arange(1, N + 1), color=col, lw=1.4)
    nn = np.arange(1, N + 1)
    ax3.plot(nn, 0.5 + 1 / np.sqrt(nn), color=T.muted, ls="--", lw=1)
    ax3.plot(nn, 0.5 - 1 / np.sqrt(nn), color=T.muted, ls="--", lw=1)
    ax3.set_xscale("log"); ax3.set_ylim(0, 1)
    ax3.axhline(0.5, color=T.ink2, lw=1)
    note(ax3, 60, 0.85, "band ±1/√n", T)
    ax3.set_title("Law of large numbers: coin flips")
    ax3.set_xlabel("number of flips n (log)"); ax3.set_ylabel("running mean")
    fig.subplots_adjust(wspace=0.3)


# ---------------------------------------------------------------- Chapter 25
@figure("ch25-ci-test-regression", size=(11.5, 3.9))
def ch25(fig, T):
    ax1, ax2, ax3 = fig.subplots(1, 3)
    rng = np.random.default_rng(11)
    mu, sigma, n, reps = 10.0, 2.0, 25, 40
    z = 1.96
    misses = 0
    for i in range(reps):
        s = rng.normal(mu, sigma, n)
        m = s.mean(); h = z * sigma / np.sqrt(n)
        hit = (m - h <= mu <= m + h)
        misses += not hit
        col = T.blue if hit else T.red
        ax1.plot([m - h, m + h], [i, i], color=col, lw=2.2 if not hit else 1.6)
        ax1.plot(m, i, "o", ms=3.5, color=col)
    ax1.axvline(mu, color=T.ink, lw=1.6)
    ax1.set_yticks([])
    ax1.set_title(f"40 samples → 40 95% CIs ({misses} miss)")
    ax1.set_xlabel("estimated mean (true μ = 10)")
    ax1.spines["left"].set_visible(False)

    x = np.linspace(-4, 7, 600)
    d = 2.5
    c = 1.645
    ax2.plot(x, stats.norm.pdf(x), color=T.blue, lw=2, label="H₀ true")
    ax2.plot(x, stats.norm.pdf(x, d), color=T.orange, lw=2, label="H₁ true")
    m = x >= c
    ax2.fill_between(x[m], 0, stats.norm.pdf(x[m]), color=T.blue, alpha=0.35, lw=0)
    m2 = x <= c
    ax2.fill_between(x[m2], 0, stats.norm.pdf(x[m2], d), color=T.orange, alpha=0.3, lw=0)
    ax2.axvline(c, color=T.ink, lw=1.4, ls="--")
    note(ax2, c + 0.1, 0.42, "reject H₀ →", T, color=T.ink)
    note(ax2, 2.1, 0.03, "α", T, color=T.ink, fontsize=12)
    note(ax2, 0.9, 0.08, "β", T, color=T.ink, fontsize=12)
    ax2.set_ylim(0, 0.47)
    ax2.legend(loc="upper left")
    ax2.set_title(f"Test: α = 5%, power 1−β = {1 - stats.norm.cdf(c, d):.0%}")
    ax2.set_xlabel("test statistic")

    xs = rng.uniform(0, 10, 30)
    ys = 1.2 + 0.8 * xs + rng.normal(0, 1.2, 30)
    b1, b0 = np.polyfit(xs, ys, 1)
    r = np.corrcoef(xs, ys)[0, 1]
    for xi, yi in zip(xs, ys):
        ax3.plot([xi, xi], [yi, b0 + b1 * xi], color=T.red, lw=1, alpha=0.7)
    ax3.plot(xs, ys, "o", ms=6, color=T.blue, mec=T.surface, mew=1)
    xx = np.linspace(0, 10, 10)
    ax3.plot(xx, b0 + b1 * xx, color=T.ink, lw=2)
    note(ax3, 0.3, 9.2, f"ŷ = {b0:.2f} + {b1:.2f}x\nr = {r:.2f}", T, color=T.ink)
    ax3.set_title("Least-squares line: minimise Σ residual²")
    ax3.set_xlabel("x"); ax3.set_ylabel("y")
    fig.subplots_adjust(wspace=0.3)
