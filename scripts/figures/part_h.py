"""Part H (1/2) -- Mathematics for AI: Chapters 26-31."""
import numpy as np
from scipy import stats
from scipy.special import erf

from style import figure, note


# ---------------------------------------------------------------- Chapter 26
@figure("ch26-jacobian-gradient-flow", size=(11, 4.3))
def ch26(fig, T):
    ax1, ax2 = fig.subplots(1, 2)
    F = lambda x, y: (x + 0.35 * np.sin(1.6 * y), y + 0.35 * np.sin(1.6 * x) + 0.1 * x**2)
    g = np.linspace(-2, 2, 9)
    s = np.linspace(-2, 2, 200)
    for c in g:
        ax1.plot(*F(np.full_like(s, c), s), color=T.axis, lw=1)
        ax1.plot(*F(s, np.full_like(s, c)), color=T.axis, lw=1)
    p = np.array([-0.55, -0.45]); h = 0.95
    sq = np.array([[0, 0], [h, 0], [h, h], [0, h], [0, 0]]) + p
    curved = []
    for (a0, b0), (a1, b1) in zip(sq[:-1], sq[1:]):
        t = np.linspace(0, 1, 30)
        curved.append(np.array(F(a0 + (a1 - a0) * t, b0 + (b1 - b0) * t)).T)
    curved = np.vstack(curved)
    ax1.fill(curved[:, 0], curved[:, 1], color=T.blue, alpha=0.3)
    ax1.plot(curved[:, 0], curved[:, 1], color=T.blue, lw=2, label="true image f(square)")
    eps = 1e-6
    f0 = np.array(F(*p))
    J = np.column_stack([(np.array(F(p[0] + eps, p[1])) - f0) / eps, (np.array(F(p[0], p[1] + eps)) - f0) / eps])
    lin = np.array([f0 + J @ (q - p) for q in sq])
    ax1.plot(lin[:, 0], lin[:, 1], color=T.orange, lw=2, ls="--", label="J·(square): linear guess")
    ax1.plot(*f0, "o", ms=7, color=T.ink)
    ax1.set_aspect("equal"); ax1.set_xticks([]); ax1.set_yticks([]); ax1.grid(False)
    for sp in ax1.spines.values():
        sp.set_visible(False)
    ax1.legend(loc="lower right", fontsize=9)
    ax1.set_title("The Jacobian is the best local linear map")

    depth = np.arange(0, 61)
    for gain, col in ((0.9, T.blue), (1.0, T.aqua), (1.1, T.orange)):
        ax2.semilogy(depth, gain**depth, color=col, label=f"per-layer gain {gain}")
    ax2.axhspan(1e-3, 1e3, color=T.grid, alpha=0.0)
    note(ax2, 32, 1e-2, "vanishing: 0.9⁶⁰ ≈ 0.002", T)
    note(ax2, 3, 2e2, "exploding: 1.1⁶⁰ ≈ 304", T)
    ax2.set_title("Backprop multiplies Jacobians: ∏ gains")
    ax2.set_xlabel("depth (layers)"); ax2.set_ylabel("gradient scale")
    ax2.legend(loc="lower left")
    ax2.set_ylim(1e-3, 1e3)


# ---------------------------------------------------------------- Chapter 27
def _f(x, y):
    return 0.5 * (x**2 + 25 * y**2) / 5


def _grad(x, y):
    return np.array([x / 5, 5 * y])


@figure("ch27-optimizers-lagrange", size=(11.5, 3.9))
def ch27(fig, T):
    ax1, ax2, ax3 = fig.subplots(1, 3, gridspec_kw={"width_ratios": [1.3, 1, 1]})
    X, Y = np.meshgrid(np.linspace(-6, 2, 300), np.linspace(-2, 2, 200))
    ax1.contour(X, Y, _f(X, Y), levels=np.geomspace(0.02, 12, 12), colors=T.axis, linewidths=0.9)
    start = np.array([-5.5, 1.5])

    def run(kind, steps=60):
        x = start.copy(); v = np.zeros(2); m = np.zeros(2); s = np.zeros(2); path = [x.copy()]
        for t in range(1, steps + 1):
            g = _grad(*x)
            if kind == "gd":
                x = x - 0.38 * g
            elif kind == "mom":
                v = 0.8 * v + g; x = x - 0.1 * v
            else:
                m = 0.9 * m + 0.1 * g; s = 0.999 * s + 0.001 * g**2
                mh = m / (1 - 0.9**t); sh = s / (1 - 0.999**t)
                x = x - 0.25 * mh / (np.sqrt(sh) + 1e-8)
            path.append(x.copy())
        return np.array(path)

    for kind, lab, col in (("gd", "gradient descent", T.orange), ("mom", "momentum", T.aqua), ("adam", "Adam", T.violet)):
        P = run(kind)
        ax1.plot(P[:, 0], P[:, 1], "-o", ms=2.5, lw=1.5, color=col, label=lab)
    ax1.plot(0, 0, "*", ms=13, color=T.ink)
    ax1.set_xlim(-6, 2); ax1.set_ylim(-2, 2)
    ax1.legend(loc="lower right", fontsize=8.8)
    ax1.set_title("Three optimisers, one narrow bowl")

    for lr, col, lab in ((0.05, T.blue, "η too small"), (0.38, T.aqua, "η about right"), (0.41, T.red, "η > 2/λ_max: diverges")):
        x = start.copy(); losses = []
        for _ in range(60):
            losses.append(_f(*x)); x = x - lr * _grad(*x)
        ax2.semilogy(losses, color=col, label=lab)
    ax2.set_title("Learning rate decides everything")
    ax2.set_xlabel("step"); ax2.set_ylabel("loss")
    ax2.legend(loc="lower left", fontsize=8.8)

    X, Y = np.meshgrid(np.linspace(-2, 2, 300), np.linspace(-2, 2, 300))
    ax3.contour(X, Y, X + Y, levels=np.linspace(-3, 3, 13), colors=T.axis, linewidths=0.9)
    th = np.linspace(0, 2 * np.pi, 300)
    ax3.plot(np.cos(th), np.sin(th), color=T.blue, lw=2.4)
    q = np.array([1, 1]) / np.sqrt(2)
    ax3.annotate("", xy=q + 0.6 * q, xytext=q, arrowprops=dict(arrowstyle="-|>", color=T.orange, lw=2.4, mutation_scale=14))
    ax3.annotate("", xy=q + 0.5 * np.array([1, 1]) / np.sqrt(2) * 1.4, xytext=q,
                 arrowprops=dict(arrowstyle="-|>", color=T.violet, lw=1.6, mutation_scale=12, ls="--"))
    ax3.plot(*q, "o", ms=9, color=T.ink, mec=T.surface, mew=2)
    note(ax3, 0.95, 1.55, "∇f ∥ ∇g", T, color=T.ink)
    note(ax3, -1.9, -1.85, "max f = x + y on the circle", T, box=True)
    ax3.set_aspect("equal"); ax3.set_xlim(-2, 2); ax3.set_ylim(-2, 2)
    ax3.set_title("Lagrange: ∇f = λ∇g at the optimum")
    note(ax3, -1.9, 1.65, "circle: constraint g = 0", T, box=True)
    fig.subplots_adjust(wspace=0.3)


# ---------------------------------------------------------------- Chapter 28
@figure("ch28-entropy-kl", size=(11.5, 3.7))
def ch28(fig, T):
    ax1, ax2, ax3 = fig.subplots(1, 3)
    p = np.linspace(1e-4, 1 - 1e-4, 400)
    H = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    ax1.plot(p, H, color=T.blue)
    ax1.plot(0.5, 1, "o", ms=8, color=T.blue, mec=T.surface, mew=2)
    note(ax1, 0.5, 1.05, "fair coin: 1 bit", T, color=T.ink, ha="center")
    ax1.set_ylim(0, 1.18)
    note(ax1, 0.05, 0.15, "certain:\n0 bits", T)
    ax1.set_title("Entropy of a coin  H(p)")
    ax1.set_xlabel("P(heads) = p"); ax1.set_ylabel("bits")

    x = np.linspace(-6, 6, 1200)
    target = 0.5 * stats.norm.pdf(x, -2, 0.7) + 0.5 * stats.norm.pdf(x, 2, 0.7)
    dx = x[1] - x[0]

    def kl(a, b):
        a = np.maximum(a, 1e-300); b = np.maximum(b, 1e-300)
        return np.sum(a * np.log(a / b)) * dx

    best_f, best_r = None, None
    for mu in np.linspace(-3, 3, 61):
        for sg in np.linspace(0.3, 3.5, 65):
            q = stats.norm.pdf(x, mu, sg)
            f_, r_ = kl(target, q), kl(q, target)
            if best_f is None or f_ < best_f[0]:
                best_f = (f_, mu, sg)
            if best_r is None or r_ < best_r[0]:
                best_r = (r_, mu, sg)
    ax2.fill_between(x, 0, target, color=T.muted, alpha=0.3, lw=0, label="target p (two modes)")
    ax2.plot(x, stats.norm.pdf(x, best_f[1], best_f[2]), color=T.blue, label="min KL(p‖q): covers both")
    ax2.plot(x, stats.norm.pdf(x, best_r[1], best_r[2]), color=T.orange, label="min KL(q‖p): picks one")
    ax2.set_ylim(0, 0.75)
    ax2.legend(loc="upper left", fontsize=8.4)
    ax2.set_title("KL is asymmetric")
    ax2.set_xlabel("x")

    q = np.linspace(0.005, 1, 400)
    ax3.plot(q, -np.log(q), color=T.blue)
    for qq, lab in ((0.9, "confident & right:\nloss 0.11"), (0.01, "confident & wrong:\nloss 4.6")):
        ax3.plot(qq, -np.log(qq), "o", ms=8, color=T.orange, mec=T.surface, mew=2)
    note(ax3, 0.62, 0.55, "confident & right: 0.11", T)
    note(ax3, 0.05, 4.7, "confident & wrong: 4.6", T)
    ax3.set_title("Cross-entropy loss  −log q(correct)")
    ax3.set_xlabel("probability given to the correct token"); ax3.set_ylim(0, 5.4)
    fig.subplots_adjust(wspace=0.28)


# ---------------------------------------------------------------- Chapter 29
@figure("ch29-bayes-montecarlo-mcmc", size=(11.5, 3.8))
def ch29(fig, T):
    ax1, ax2, ax3 = fig.subplots(1, 3)
    th = np.linspace(0, 1, 400)
    for (a, b), col, lab in (((2, 2), T.muted, "prior Beta(2,2)"), ((2 + 3, 2 + 1), T.orange, "after 3 of 4 ✓"),
                             ((2 + 30, 2 + 10), T.aqua, "after 30 of 40 ✓"), ((2 + 300, 2 + 100), T.blue, "after 300 of 400 ✓")):
        ax1.plot(th, stats.beta.pdf(th, a, b), color=col, label=lab)
    ax1.axvline(0.75, color=T.ink2, lw=1, ls=":")
    ax1.set_title("Bayesian updating")
    ax1.set_xlabel("success rate θ"); ax1.set_ylabel("density")
    ax1.legend(loc="upper left", fontsize=8.5)

    rng = np.random.default_rng(1)
    Ns = np.geomspace(10, 1e6, 30).astype(int)
    for col in (T.blue, T.orange, T.aqua):
        pts = rng.random((int(1e6), 2))
        inside = np.cumsum((pts**2).sum(1) <= 1)
        est = 4 * inside[Ns - 1] / Ns
        ax2.loglog(Ns, np.abs(est - np.pi) + 1e-9, color=col, lw=1.3, alpha=0.9)
    ax2.loglog(Ns, 1.6 / np.sqrt(Ns), color=T.ink, ls="--", lw=1.6, label="∝ 1/√N")
    ax2.set_title("Monte Carlo π: error ∝ 1/√N")
    ax2.set_xlabel("samples N"); ax2.set_ylabel("|estimate − π|")
    ax2.legend(loc="lower left")

    def logp(z):
        x, y = z
        return -0.5 * (x**2 / 4 + (y - 0.35 * x**2 + 1.0) ** 2 / 0.25)

    z = np.array([-3.0, 3.0]); chain = [z.copy()]; acc = 0
    for _ in range(6000):
        prop = z + rng.normal(0, 0.45, 2)
        if np.log(rng.random()) < logp(prop) - logp(z):
            z = prop; acc += 1
        chain.append(z.copy())
    C = np.array(chain)
    Xg, Yg = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-2.5, 6, 200))
    ax3.contour(Xg, Yg, np.exp(logp((Xg, Yg))), levels=6, colors=T.axis, linewidths=0.9)
    ax3.plot(C[200:, 0], C[200:, 1], ".", ms=2, color=T.blue, alpha=0.35)
    ax3.plot(C[:120, 0], C[:120, 1], "-", color=T.orange, lw=1.2, label="first 120 steps")
    ax3.plot(*C[0], "o", ms=7, color=T.orange)
    ax3.set_xlim(-5, 5); ax3.set_ylim(-2.5, 6)
    ax3.set_title(f"MCMC: {acc / 6000:.0%} of proposals accepted")
    ax3.legend(loc="upper right", fontsize=8.5)
    fig.subplots_adjust(wspace=0.3)


# ---------------------------------------------------------------- Chapter 30
def _image():
    n = 160
    y, x = np.mgrid[0:n, 0:n] / n
    img = 0.25 * np.sin(6 * np.pi * x) * np.cos(4 * np.pi * y)
    img += ((x - 0.33) ** 2 + (y - 0.6) ** 2 < 0.03).astype(float) * 0.9
    img += ((np.abs(x - 0.72) < 0.12) & (np.abs(y - 0.35) < 0.2)).astype(float) * 0.7
    img += (np.abs((x - 0.15) - (y - 0.1)) < 0.02).astype(float) * 0.8
    img += 0.6 * np.exp(-((x - 0.7) ** 2 + (y - 0.8) ** 2) / 0.01)
    return img


@figure("ch30-pca-svd", size=(11.5, 3.8))
def ch30(fig, T):
    ax1, ax2 = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.25]})
    rng = np.random.default_rng(4)
    C = np.array([[3.0, 1.6], [1.6, 1.3]])
    Xd = rng.multivariate_normal([0, 0], C, 600)
    ax1.plot(Xd[:, 0], Xd[:, 1], ".", ms=3.5, color=T.blue, alpha=0.45)
    lam, V = np.linalg.eigh(np.cov(Xd.T))
    for l, v, col, lab in zip(lam[::-1], V.T[::-1], (T.orange, T.aqua), ("PC1", "PC2")):
        ax1.annotate("", xy=2 * np.sqrt(l) * v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=3, mutation_scale=16))
        note(ax1, 2.15 * np.sqrt(l) * v[0], 2.15 * np.sqrt(l) * v[1], f"{lab}: {l / lam.sum():.0%} of variance", T, color=T.ink, fontsize=9.5, box=True)
    ax1.set_aspect("equal"); ax1.set_xlim(-5.5, 6.5); ax1.set_ylim(-4, 4)
    ax1.set_title("PCA: axes of greatest variance")

    img = _image()
    s = np.linalg.svd(img, compute_uv=False)
    k = np.arange(1, len(s) + 1)
    ax2.semilogy(k, s, color=T.blue, label="singular values σₖ")
    ax2b = ax2.twinx()
    energy = np.cumsum(s**2) / np.sum(s**2)
    ax2.set_title("Singular values decay fast → low rank works")
    ax2.set_xlabel("index k"); ax2.set_ylabel("σₖ (log)")
    ax2b.plot(k, energy, color=T.orange, lw=1.8, label="energy captured")
    ax2b.set_ylim(0, 1.05); ax2b.grid(False)
    ax2b.spines["right"].set_visible(True); ax2b.spines["right"].set_color(T.axis)
    ax2b.set_ylabel("cumulative energy", color=T.ink2)
    for r in (5, 20):
        note(ax2b, r + 3, energy[r - 1] - 0.08, f"rank {r}: {energy[r - 1]:.1%}", T, color=T.ink)
        ax2b.plot(r, energy[r - 1], "o", ms=6, color=T.orange)
    ax2.set_xlim(0, 80); ax2.set_ylim(1e-3, s[0] * 2)
    h1, l1 = ax2.get_legend_handles_labels(); h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="center right")


@figure("ch30-low-rank-image", size=(11.5, 3.0))
def ch30_img(fig, T):
    img = _image()
    U, s, Vt = np.linalg.svd(img)
    axs = fig.subplots(1, 5)
    cmap = T.cmap_seq()
    for ax, r in zip(axs, (1, 3, 8, 20, None)):
        A = img if r is None else (U[:, :r] * s[:r]) @ Vt[:r]
        ax.imshow(A, cmap=cmap, vmin=img.min(), vmax=img.max(), origin="lower", interpolation="bilinear")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(T.axis)
        if r is None:
            ax.set_title("original\nrank 160", fontsize=9.5, loc="center")
        else:
            ax.set_title(f"rank {r}\n{r * (2 * 160 + 1) / 160**2:.0%} of the numbers", fontsize=9.5, loc="center")
    fig.subplots_adjust(wspace=0.06)


# ---------------------------------------------------------------- Chapter 31
@figure("ch31-activations-softmax-rope-scaling", size=(11, 7.4))
def ch31(fig, T):
    (ax1, ax2), (ax3, ax4) = fig.subplots(2, 2)
    x = np.linspace(-4, 4, 400)
    gelu = 0.5 * x * (1 + erf(x / np.sqrt(2)))
    acts = [("ReLU", np.maximum(0, x), T.blue), ("GELU", gelu, T.orange), ("SiLU / Swish", x / (1 + np.exp(-x)), T.aqua),
            ("tanh", np.tanh(x), T.violet), ("sigmoid", 1 / (1 + np.exp(-x)), T.yellow)]
    for lab, y, col in acts:
        ax1.plot(x, y, color=col, label=lab)
    ax1.set_ylim(-1.5, 4); ax1.legend(loc="upper left", fontsize=9)
    ax1.set_title("Activation functions")
    ax1.axhline(0, color=T.axis, lw=1); ax1.axvline(0, color=T.axis, lw=1)

    logits = np.array([3.0, 2.0, 1.0, 0.2, -1.0])
    toks = ["Paris", "Lyon", "France", "the", "banana"]
    w = 0.26
    for i, (Tt, col) in enumerate(((0.5, T.blue), (1.0, T.orange), (2.0, T.aqua))):
        p = np.exp(logits / Tt); p /= p.sum()
        ax2.bar(np.arange(5) + (i - 1) * w, p, width=w * 0.92, color=col, label=f"T = {Tt}")
    ax2.set_xticks(range(5)); ax2.set_xticklabels(toks)
    ax2.set_ylabel("probability")
    ax2.legend(loc="upper right", title="temperature", title_fontsize=9)
    ax2.set_title("Softmax(logits / T): temperature reshapes")

    d = 128
    theta = 10000 ** (-2 * np.arange(d // 2) / d)
    m = np.arange(0, 512)
    val = np.abs(np.exp(1j * np.outer(m, theta)).sum(1)) / (d // 2)
    ax3.plot(m, val, color=T.blue, lw=1.4)
    ax3.set_title("RoPE: rotation sum decays with distance")
    ax3.set_xlabel("relative distance m − n"); ax3.set_ylabel("normalised |Σⱼ e^(i(m−n)θⱼ)|")
    note(ax3, 150, 0.7, "d = 128, base 10 000\nfar tokens interfere less", T)

    N = np.geomspace(1e7, 1e12, 200)
    E, A, B, al, be = 1.69, 406.4, 410.7, 0.34, 0.28
    for ratio, col in ((2, T.orange), (20, T.blue), (200, T.aqua)):
        D = ratio * N
        ax4.loglog(N, E + A / N**al + B / D**be, color=col, label=f"D = {ratio} tokens / param")
    ax4.axhline(E, color=T.muted, lw=1, ls="--")
    note(ax4, 1.3e7, E * 1.02, "irreducible loss E = 1.69", T)
    ax4.set_title("Scaling law  L = E + A/Nᵅ + B/Dᵝ")
    ax4.set_xlabel("parameters N"); ax4.set_ylabel("loss (nats/token)")
    ax4.legend(loc="upper right", fontsize=8.8)
    note(ax4, 1.3e7, 2.05, "constants: Chinchilla fit\n(Hoffmann et al., 2022)", T, fontsize=8.5)
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    ax4.yaxis.set_major_locator(FixedLocator([2, 3, 4, 5, 6]))
    ax4.yaxis.set_major_formatter(FixedFormatter(["2", "3", "4", "5", "6"]))
    ax4.yaxis.set_minor_locator(NullLocator())
    fig.subplots_adjust(hspace=0.38, wspace=0.25)
