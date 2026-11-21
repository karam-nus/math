"""Part H (2/2) -- Mathematics for AI: Chapters 32-36."""
import functools

import numpy as np
from scipy import stats

from style import figure, note


# ---------------------------------------------------------------- Chapter 32
def _e4m3_positive():
    vals = [m / 8 * 2.0**-6 for m in range(1, 8)]  # subnormals
    for e in range(1, 16):
        for m in range(8):
            if e == 15 and m == 7:
                continue  # NaN encoding
            vals.append((1 + m / 8) * 2.0 ** (e - 7))
    return np.array(vals)


@figure("ch32-quantization", size=(11, 7.2))
def ch32(fig, T):
    (ax1, ax2), (ax3, ax4) = fig.subplots(2, 2)
    fp8 = _e4m3_positive()
    int8 = np.arange(1, 128) * 448 / 127
    ax1.vlines(fp8, 1.1, 1.9, color=T.orange, lw=0.9)
    ax1.vlines(int8, 0.1, 0.9, color=T.blue, lw=0.9)
    ax1.set_xscale("log"); ax1.set_xlim(1e-3, 600)
    ax1.set_yticks([0.5, 1.5]); ax1.set_yticklabels(["INT8\n(scaled to 448)", "FP8 E4M3"])
    ax1.grid(axis="y", visible=False)
    ax1.set_title("Where the representable numbers live")
    ax1.set_xlabel("magnitude (log scale)")
    note(ax1, 1.3e-3, 2.05, "float: same relative precision at every scale", T)
    note(ax1, 1.3e-3, -0.4, "integer: uniform steps; tiny values all round to 0", T)
    ax1.set_ylim(-0.6, 2.35)

    x = np.linspace(-1.3, 1.3, 2000)
    s = 0.25
    q = s * np.clip(np.round(x / s), -4, 3)
    ax2.plot(x, x, color=T.muted, lw=1.2, ls="--", label="ideal  y = x")
    ax2.plot(x, q, color=T.blue, lw=2, label="3-bit quantize → dequantize")
    ax2.plot(x, (x - q) - 1.75, color=T.orange, lw=1.4, label="error x − Q(x)  (shifted down)")
    ax2.axvspan(-1.3, -1.0, color=T.red, alpha=0.08); ax2.axvspan(0.75, 1.3, color=T.red, alpha=0.08)
    note(ax2, 0.8, -0.55, "clipping\nregion", T)
    note(ax2, -0.55, -1.25, "rounding error: ≤ s/2,\nvariance s²/12", T)
    ax2.set_title("Uniform quantizer: round, then clip")
    ax2.set_xlabel("x"); ax2.legend(loc="upper left", fontsize=8.6)
    ax2.set_ylim(-2.15, 1.5)

    rng = np.random.default_rng(0)
    w = np.concatenate([rng.normal(0, 1, 20000), rng.choice([-1, 1], 12) * rng.uniform(9, 12, 12)])
    ax3.hist(w, bins=np.linspace(-12.5, 12.5, 251), color=T.blue, alpha=0.8)
    ax3.set_yscale("log")
    amax = np.abs(w).max()
    for lv in np.linspace(-amax, amax, 15):
        ax3.axvline(lv, color=T.orange, lw=0.9, alpha=0.9)
    ax3.set_title("Outliers stretch the scale: 4-bit absmax grid")
    ax3.set_xlabel("weight value"); ax3.set_ylabel("count (log)")
    note(ax3, -12.3, 60, "orange: the 15 levels.\nAlmost every weight\nlands on 3 of them.", T, box=True)

    g = rng.normal(0, 1, 200000)
    cs = np.linspace(0.8, 6, 80)
    for bits, col in ((2, T.orange), (3, T.aqua), (4, T.blue), (8, T.violet)):
        L = 2 ** (bits - 1) - 1
        mse = []
        for c in cs:
            sc = c / L
            mse.append(np.mean((g - sc * np.clip(np.round(g / sc), -L, L)) ** 2))
        mse = np.array(mse)
        ax4.semilogy(cs, mse, color=col, label=f"{bits}-bit")
        i = np.argmin(mse)
        ax4.plot(cs[i], mse[i], "o", ms=7, color=col, mec=T.surface, mew=1.5)
    ax4.set_title("Best clipping range is NOT the max")
    ax4.set_xlabel("clip threshold c  (in units of σ; Gaussian weights)")
    ax4.set_ylabel("MSE")
    ax4.legend(loc="lower left", ncol=2, fontsize=8.8)
    note(ax4, 4.15, 1.3e-3, "● = optimum: clipping a few\noutliers buys a finer step", T)
    fig.subplots_adjust(hspace=0.42, wspace=0.28)


# ---------------------------------------------------------------- Chapter 33
@figure("ch33-roofline-kv-speculative", size=(11.5, 3.9))
def ch33(fig, T):
    ax1, ax2, ax3 = fig.subplots(1, 3)
    peak, bw = 990.0, 3.35  # TFLOP/s (dense BF16), TB/s  -- H100-SXM-class, illustrative
    ai = np.geomspace(0.1, 1e4, 400)
    ax1.loglog(ai, np.minimum(peak, bw * ai), color=T.ink, lw=2.4)
    ridge = peak / bw
    ax1.axvline(ridge, color=T.muted, ls=":", lw=1)
    note(ax1, ridge * 1.15, 1.5, f"ridge ≈ {ridge:.0f}\nFLOP/B", T)
    pts = [("decode, batch 1", 1, T.red), ("decode, batch 32", 32, T.orange), ("prefill 2k tokens", 2048, T.blue)]
    for lab, x, col in pts:
        y = min(peak, bw * x)
        ax1.plot(x, y, "o", ms=9, color=col, mec=T.surface, mew=2)
        if x > 1000:
            note(ax1, x * 0.8, y * 1.5, lab, T, color=T.ink, fontsize=9, ha="right")
        else:
            note(ax1, x * 1.25, y * 0.6, lab, T, color=T.ink, fontsize=9)
    note(ax1, 0.13, 400, "memory-bound\n(slope = bandwidth)", T)
    note(ax1, 500, 3000, "compute-bound", T)
    ax1.set_ylim(0.3, 8000)
    ax1.set_title("Roofline (H100-class, BF16)")
    ax1.set_xlabel("arithmetic intensity (FLOP / byte)"); ax1.set_ylabel("attainable TFLOP/s")

    ctx = np.linspace(0, 131072, 300)
    L, dh = 32, 128
    for nkv, lab, col in ((32, "MHA (32 KV heads)", T.red), (8, "GQA (8 KV heads)", T.blue), (1, "MQA (1 KV head)", T.aqua)):
        gib = 2 * L * nkv * dh * 2 * ctx / 2**30
        ax2.plot(ctx / 1024, gib, color=col, label=lab)
    ax2.axhline(80, color=T.muted, ls="--", lw=1)
    note(ax2, 2, 82, "80 GB of HBM", T)
    ax2.set_title("KV cache (7B-class, FP16)")
    ax2.set_xlabel("context length (K tokens)"); ax2.set_ylabel("GiB")
    ax2.legend(loc="upper left", bbox_to_anchor=(0, 0.88), fontsize=8.8)

    a = np.linspace(0, 0.98, 200)
    for gm, col in ((2, T.aqua), (4, T.blue), (8, T.violet)):
        ax3.plot(a, (1 - a ** (gm + 1)) / (1 - a), color=col, label=f"draft length γ = {gm}")
    ax3.set_title("Speculative decoding")
    ax3.set_xlabel("acceptance rate α"); ax3.set_ylabel("tokens per target forward pass")
    ax3.legend(loc="upper left")
    note(ax3, 0.02, 5.6, "E = (1 − α^(γ+1)) / (1 − α)", T, box=True)
    fig.subplots_adjust(wspace=0.36)


# ---------------------------------------------------------------- Chapter 34
@figure("ch34-brownian-diffusion", size=(11, 4.1))
def ch34(fig, T):
    ax1, ax2 = fig.subplots(1, 2)
    rng = np.random.default_rng(5)
    n, tmax = 1000, 1.0
    t = np.linspace(0, tmax, n + 1)
    cols = [T.blue, T.orange, T.aqua, T.violet, T.yellow, T.magenta]
    for col in cols:
        W = np.concatenate([[0], np.cumsum(rng.normal(0, np.sqrt(tmax / n), n))])
        ax1.plot(t, W, color=col, lw=1.1)
    for k, ls in ((1, "--"), (2, ":")):
        ax1.plot(t, k * np.sqrt(t), color=T.ink2, ls=ls, lw=1.2)
        ax1.plot(t, -k * np.sqrt(t), color=T.ink2, ls=ls, lw=1.2)
    note(ax1, 0.82, 2.08, "±2√t", T)
    note(ax1, 0.82, 1.06, "±√t", T)
    ax1.set_title("Brownian motion: spread grows like √t")
    ax1.set_xlabel("t"); ax1.set_ylabel("W(t)")

    xs = np.linspace(-4, 4, 300)
    ts = np.linspace(0, 1, 300)
    abar = np.exp(-6 * ts)
    mus, sg = np.array([-2.0, 1.5]), 0.35
    wts = np.array([0.4, 0.6])
    P = np.zeros((len(xs), len(ts)))
    for j, ab in enumerate(abar):
        var = ab * sg**2 + (1 - ab)
        P[:, j] = sum(w * stats.norm.pdf(xs, np.sqrt(ab) * m, np.sqrt(var)) for w, m in zip(wts, mus))
    ax2.imshow(P, extent=(0, 1, -4, 4), origin="lower", aspect="auto", cmap=T.cmap_seq(), vmax=P.max() * 0.7)
    for k in range(7):
        x0 = rng.choice(mus, p=wts) + rng.normal(0, sg)
        eps = rng.normal(0, 1)
        path = [x0]
        for j in range(1, len(ts)):
            beta = 1 - abar[j] / abar[j - 1]
            path.append(np.sqrt(1 - beta) * path[-1] + np.sqrt(beta) * rng.normal())
        ax2.plot(ts, path, color=T.orange, lw=0.9, alpha=0.9)
    ax2.annotate("", xy=(0.95, 3.5), xytext=(0.05, 3.5), arrowprops=dict(arrowstyle="-|>", color=T.ink, lw=1.6))
    note(ax2, 0.3, 3.65, "forward: add noise", T, color=T.ink, box=True)
    ax2.annotate("", xy=(0.05, -3.5), xytext=(0.95, -3.5), arrowprops=dict(arrowstyle="-|>", color=T.ink, lw=1.6))
    note(ax2, 0.22, -3.35, "generate: reverse with learned score", T, color=T.ink, va="bottom", box=True)
    ax2.set_title("Diffusion: data density → pure Gaussian noise")
    ax2.set_xlabel("diffusion time t"); ax2.set_ylabel("x"); ax2.grid(False)


# ---------------------------------------------------------------- Chapter 35
@figure("ch35-gridworld-bandits", size=(11, 4.3))
def ch35(fig, T):
    ax1, ax2 = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.25]})
    H, W = 5, 6
    walls = {(1, 1), (2, 3), (3, 3)}
    term = {(0, 5): 1.0, (1, 5): -1.0}
    gamma, step = 0.95, -0.02
    V = np.zeros((H, W))
    acts = {"↑": (-1, 0), "↓": (1, 0), "←": (0, -1), "→": (0, 1)}

    def nxt(s, a):
        r, c = s[0] + a[0], s[1] + a[1]
        if 0 <= r < H and 0 <= c < W and (r, c) not in walls:
            return (r, c)
        return s

    for _ in range(200):
        Vn = V.copy()
        for r in range(H):
            for c in range(W):
                if (r, c) in walls or (r, c) in term:
                    continue
                Vn[r, c] = max(step + gamma * (term.get(nxt((r, c), a), 0) + (0 if nxt((r, c), a) in term else V[nxt((r, c), a)]))
                               for a in acts.values())
        V = Vn
    show = V.copy()
    for k, v in term.items():
        show[k] = v
    M = np.ma.array(show, mask=np.zeros_like(show, bool))
    for wll in walls:
        M.mask[wll] = True
    cmap = T.cmap_div().copy(); cmap.set_bad(T.ink2)
    ax1.imshow(M, cmap=cmap.reversed(), vmin=-1, vmax=1)
    for r in range(H):
        for c in range(W):
            if (r, c) in walls:
                continue
            if (r, c) in term:
                ax1.text(c, r, "+1\ngoal" if term[(r, c)] > 0 else "−1\npit", ha="center", va="center", fontsize=9, color="#ffffff", fontweight="bold")
                continue
            best = max(acts, key=lambda a: term.get(nxt((r, c), acts[a]), 0) + (0 if nxt((r, c), acts[a]) in term else V[nxt((r, c), acts[a])]))
            ax1.text(c, r - 0.12, best, ha="center", va="center", fontsize=14, color="#ffffff" if T.name == "light" else "#0b0b0b")
            ax1.text(c, r + 0.28, f"{V[r, c]:.2f}", ha="center", va="center", fontsize=8, color="#ffffff" if T.name == "light" else "#0b0b0b")
    ax1.set_xticks([]); ax1.set_yticks([]); ax1.grid(False)
    ax1.set_title("Value iteration: V(s) and greedy policy")

    for kind, lab, col in (("eps", "ε-greedy (ε = 0.1)", T.orange), ("ucb", "UCB1", T.blue), ("ts", "Thompson sampling", T.aqua)):
        ax2.plot(np.arange(1, BANDIT_STEPS + 1), _bandit(kind), color=col, label=lab)
    ax2.set_title("10-armed bandit: cumulative regret")
    ax2.set_xlabel("pulls t"); ax2.set_ylabel("regret (avg of 300 runs)")
    ax2.legend(loc="upper left")
    note(ax2, 1200, 12, "ε-greedy: grows linearly (explores forever)\nUCB1, Thompson: bend over (log-like)", T, box=True)


BANDIT_STEPS = 3000


@functools.lru_cache(maxsize=None)
def _bandit(kind):
    rng = np.random.default_rng(2)
    K, steps, runs = 10, BANDIT_STEPS, 300
    mu = rng.uniform(0.2, 0.8, K); best = mu.max()

    def simulate(kind):
        reg = np.zeros(steps)
        for _ in range(runs):
            n = np.zeros(K); s = np.zeros(K); a_b = np.ones(K); b_b = np.ones(K); cum = 0.0
            for t in range(steps):
                if kind == "eps":
                    a = rng.integers(K) if (rng.random() < 0.1 or t < K) else np.argmax(s / np.maximum(n, 1))
                elif kind == "ucb":
                    a = t if t < K else np.argmax(s / n + np.sqrt(2 * np.log(t + 1) / n))
                else:
                    a = np.argmax(rng.beta(a_b, b_b))
                rwd = rng.random() < mu[a]
                n[a] += 1; s[a] += rwd; a_b[a] += rwd; b_b[a] += 1 - rwd
                cum += best - mu[a]; reg[t] += cum
        return reg / runs

    return simulate(kind)


# ---------------------------------------------------------------- Chapter 36
@figure("ch36-agent-reliability", size=(11.5, 3.8))
def ch36(fig, T):
    ax1, ax2, ax3 = fig.subplots(1, 3)
    n = np.arange(1, 101)
    for p, col in ((0.90, T.red), (0.95, T.orange), (0.99, T.blue)):
        ax1.plot(n, p**n, color=col, label=f"p = {p} per step")
    ax1.plot(n, (1 - 0.1**3) ** n, color=T.aqua, ls="--", lw=2, label="p = 0.90 + verify & retry ×2")
    ax1.plot(20, 0.9**20, "o", ms=8, color=T.red, mec=T.surface, mew=2)
    note(ax1, 22, 0.9**20 + 0.03, f"20 steps at 90%: {0.9**20:.0%}", T, color=T.ink)
    ax1.set_title("Errors compound: pⁿ")
    ax1.set_xlabel("steps in the workflow n"); ax1.set_ylabel("P(whole task succeeds)")
    ax1.legend(loc="upper right", fontsize=8.2)
    ax1.set_ylim(0, 1.38); ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    ks = np.arange(1, 42, 2)
    for p, col in ((0.40, T.red), (0.55, T.orange), (0.65, T.aqua), (0.80, T.blue)):
        acc = [1 - stats.binom.cdf(k // 2, k, p) for k in ks]
        ax2.plot(ks, acc, "-o", ms=3.5, color=col, label=f"{p:.0%}")
    ax2.set_title("Majority vote of k samples")
    ax2.set_xlabel("samples k (odd)"); ax2.set_ylabel("P(majority is right)")
    ax2.legend(loc="center", bbox_to_anchor=(0.6, 0.47), ncol=2, fontsize=8.6, title="single-sample accuracy", title_fontsize=8.6)
    note(ax2, 2, 0.06, "below 50%, voting makes it worse", T)

    k = np.arange(1, 201)
    for p, col in ((0.01, T.violet), (0.05, T.orange), (0.2, T.blue)):
        ax3.semilogx(k, 1 - (1 - p) ** k, color=col, label=f"pass@1 = {p:.0%}")
    ax3.set_title("pass@k = 1 − (1 − p)ᵏ")
    ax3.set_xlabel("attempts k (needs a reliable checker)"); ax3.set_ylabel("P(at least one success)")
    ax3.legend(loc="upper left")
    fig.subplots_adjust(wspace=0.3)
