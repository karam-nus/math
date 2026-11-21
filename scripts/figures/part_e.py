"""Part E -- Numeric Analysis (Chapters 19-21)."""
import numpy as np
from scipy.interpolate import CubicSpline, BarycentricInterpolator

from style import figure, note


# ---------------------------------------------------------------- Chapter 19
@figure("ch19-newton-runge-quadrature", size=(11.5, 3.7))
def ch19(fig, T):
    ax1, ax2, ax3 = fig.subplots(1, 3)
    f = lambda x: x**2 - 2
    df = lambda x: 2 * x
    xs = np.linspace(0.8, 3.2, 300)
    ax1.plot(xs, f(xs), color=T.ink, lw=2.2)
    ax1.axhline(0, color=T.axis, lw=1)
    x = 3.0
    cols = (T.orange, T.aqua, T.violet)
    for k in range(3):
        xn = x - f(x) / df(x)
        ax1.plot([x, x], [0, f(x)], color=T.muted, lw=1, ls=":")
        ax1.plot([x, xn], [f(x), 0], color=cols[k], lw=1.8)
        ax1.plot(x, f(x), "o", ms=7, color=cols[k], mec=T.surface, mew=1.5)
        if k < 2:
            note(ax1, xn, -0.75, f"x{k + 1}".translate(str.maketrans("123", "₁₂₃")), T, ha="center")
        x = xn
    note(ax1, 3.0, -0.75, "x₀", T, ha="center")
    ax1.set_ylim(-1.2, 7.5); ax1.set_xlim(0.8, 3.2)
    ax1.set_title("Newton's method")
    ax1.set_xlabel("x")
    note(ax1, 0.9, 6.3, "digits double each step:\n3 → 1.83 → 1.462 → 1.4149", T)

    g = lambda x: 1 / (1 + 25 * x**2)
    xx = np.linspace(-1, 1, 600)
    ax2.plot(xx, g(xx), color=T.ink, lw=2.4, label="f = 1/(1+25x²)")
    n = 11
    xe = np.linspace(-1, 1, n)
    ax2.plot(xx, BarycentricInterpolator(xe, g(xe))(xx), color=T.red, lw=1.6, label="poly, equispaced")
    xc = np.cos(np.pi * (2 * np.arange(n) + 1) / (2 * n))
    ax2.plot(xx, BarycentricInterpolator(xc, g(xc))(xx), color=T.aqua, lw=1.6, label="poly, Chebyshev nodes")
    ax2.plot(xx, CubicSpline(xe, g(xe))(xx), color=T.blue, lw=1.6, ls="--", label="cubic spline")
    ax2.plot(xe, g(xe), "o", ms=5, color=T.ink2)
    ax2.set_ylim(-0.5, 2.1)
    ax2.legend(loc="upper center", fontsize=8.3)
    ax2.set_title("Runge's phenomenon")
    ax2.set_xlabel("x")

    ns = np.array([2, 4, 8, 16, 32, 64, 128, 256])
    exact = 1 - np.cos(1.0)
    fx = np.sin
    err_t, err_s = [], []
    for m in ns:
        x = np.linspace(0, 1, m + 1); h = 1 / m; y = fx(x)
        err_t.append(abs(h * (y.sum() - 0.5 * (y[0] + y[-1])) - exact))
        err_s.append(abs(h / 3 * (y[0] + y[-1] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum()) - exact))
    ax3.loglog(ns, err_t, "-o", color=T.orange, ms=5, label="trapezoidal  ∝ h²")
    ax3.loglog(ns, err_s, "-o", color=T.blue, ms=5, label="Simpson  ∝ h⁴")
    ax3.set_title("Quadrature error")
    ax3.set_xlabel("steps n  (h = 1/n)"); ax3.set_ylabel("|error|")
    ax3.legend(loc="lower left")
    fig.subplots_adjust(wspace=0.34)


# ---------------------------------------------------------------- Chapter 20
@figure("ch20-iteration-conditioning-power", size=(11.5, 3.7))
def ch20(fig, T):
    ax1, ax2, ax3 = fig.subplots(1, 3)
    n = 30
    A = 4 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    b = np.ones(n)
    xs = np.linalg.solve(A, b)
    for method, col in (("Jacobi", T.orange), ("Gauss–Seidel", T.blue)):
        x = np.zeros(n); errs = []
        for _ in range(40):
            if method == "Jacobi":
                x = (b - (A - np.diag(np.diag(A))) @ x) / np.diag(A)
            else:
                for i in range(n):
                    x[i] = (b[i] - A[i, :i] @ x[:i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i]
            errs.append(np.linalg.norm(x - xs))
        ax1.semilogy(range(1, 41), errs, color=col, label=method)
    ax1.set_title("Iterative solvers on Ax = b")
    ax1.set_xlabel("iteration"); ax1.set_ylabel("‖x − x*‖")
    ax1.legend(loc="upper right")
    note(ax1, 2, 3e-16, "Gauss–Seidel uses fresh\nvalues → ~2× faster here", T, va="bottom")

    x = np.linspace(0, 3, 50)
    ax2.plot(x, (2 - x) / 1.0, color=T.blue, lw=2, label="x₁ + x₂ = 2")
    ax2.plot(x, (2.01 - x) / 1.01, color=T.orange, lw=2, ls="--", label="x₁ + 1.01x₂ = 2.01")
    ax2.plot(x, (2.02 - x) / 1.01, color=T.violet, lw=1.6, ls=":", label="…= 2.02 (b nudged 0.5%)")
    ax2.plot(1, 1, "o", ms=9, color=T.orange, mec=T.surface, mew=2)
    ax2.plot(0, 2, "o", ms=9, color=T.violet, mec=T.surface, mew=2)
    note(ax2, 1.08, 1.12, "(1, 1)", T, color=T.ink)
    note(ax2, 0.08, 2.1, "(0, 2)", T, color=T.ink)
    ax2.set_xlim(-0.1, 2.6); ax2.set_ylim(-0.4, 2.6)
    ax2.set_title("Ill-conditioning: κ(A) ≈ 400")
    ax2.legend(loc="lower left", fontsize=8.3)
    ax2.set_xlabel("x₁"); ax2.set_ylabel("x₂")

    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.standard_normal((5, 5)))
    for ratio, col in ((0.5, T.blue), (0.8, T.aqua), (0.95, T.orange)):
        lam = np.array([1.0, ratio, 0.3, 0.2, 0.1])
        M = Q @ np.diag(lam) @ Q.T
        v = rng.standard_normal(5); errs = []
        for _ in range(60):
            v = M @ v; v /= np.linalg.norm(v)
            errs.append(abs(1 - v @ M @ v))
        ax3.semilogy(range(1, 61), np.maximum(errs, 1e-16), color=col, label=f"|λ₂/λ₁| = {ratio}")
    ax3.set_title("Power method: speed = |λ₂/λ₁|ᵏ")
    ax3.set_xlabel("iteration k"); ax3.set_ylabel("eigenvalue error")
    ax3.legend(loc="upper right", fontsize=8.8)
    ax3.set_ylim(1e-16, 1e4)
    fig.subplots_adjust(wspace=0.32)


# ---------------------------------------------------------------- Chapter 21
@figure("ch21-order-stability", size=(11.5, 3.8))
def ch21(fig, T):
    ax1, ax2, ax3 = fig.subplots(1, 3)
    hs = 1 / 2 ** np.arange(1, 9)
    f = lambda t, y: y

    def run(step, h):
        y, t = 1.0, 0.0
        for _ in range(int(round(1 / h))):
            y = step(t, y, h); t += h
        return abs(y - np.e)

    euler = lambda t, y, h: y + h * f(t, y)

    def heun(t, y, h):
        k1 = f(t, y); k2 = f(t + h, y + h * k1)
        return y + h / 2 * (k1 + k2)

    def rk4(t, y, h):
        k1 = f(t, y); k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2); k4 = f(t + h, y + h * k3)
        return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    for step, lab, col in ((euler, "Euler  (slope 1)", T.orange), (heun, "Improved Euler  (slope 2)", T.aqua),
                           (rk4, "RK4  (slope 4)", T.blue)):
        ax1.loglog(hs, [run(step, h) for h in hs], "-o", ms=5, color=col, label=lab)
    ax1.set_title("Order of accuracy: error ∝ hᵖ")
    ax1.set_xlabel("step size h"); ax1.set_ylabel("|error at t = 1|")
    ax1.legend(loc="lower right", fontsize=8.8)

    g = np.linspace(-3.5, 1.5, 500)
    X, Y = np.meshgrid(g, np.linspace(-3.2, 3.2, 500))
    Z = X + 1j * Y
    R_e = np.abs(1 + Z)
    R_4 = np.abs(1 + Z + Z**2 / 2 + Z**3 / 6 + Z**4 / 24)
    ax2.contourf(X, Y, R_4, levels=[0, 1], colors=[T.blue], alpha=0.25)
    ax2.contour(X, Y, R_4, levels=[1], colors=[T.blue], linewidths=2)
    ax2.contourf(X, Y, R_e, levels=[0, 1], colors=[T.orange], alpha=0.35)
    ax2.contour(X, Y, R_e, levels=[1], colors=[T.orange], linewidths=2)
    ax2.axhline(0, color=T.axis, lw=1); ax2.axvline(0, color=T.axis, lw=1)
    note(ax2, -1.6, 0.15, "Euler", T, color=T.ink, ha="center")
    note(ax2, -2.2, 2.2, "RK4", T, color=T.ink, ha="center")
    ax2.set_aspect("equal")
    ax2.set_title("Stability regions in the hλ-plane")
    ax2.set_xlabel("Re hλ"); ax2.set_ylabel("Im hλ")

    nx = 41
    x = np.linspace(0, 1, nx)
    u0 = np.sin(np.pi * x) + 0.0 * x
    u0[nx // 2] += 0.05
    for r, col, lab in ((0.45, T.blue, "r = 0.45  (stable)"), (0.55, T.red, "r = 0.55  (unstable)")):
        u = u0.copy()
        for _ in range(24):
            u[1:-1] = u[1:-1] + r * (u[2:] - 2 * u[1:-1] + u[:-2])
        ax3.plot(x, u, color=col, label=lab)
    ax3.set_ylim(-0.35, 1.5)
    ax3.set_title("Explicit heat scheme: r = kc²/h² ≤ ½")
    ax3.set_xlabel("x"); ax3.legend(loc="upper left", fontsize=8.8, ncol=1)
    note(ax3, 0.02, -0.28, "after 24 steps; a 5% bump at x = ½", T)
    fig.subplots_adjust(wspace=0.32)
