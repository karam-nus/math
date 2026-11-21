# Appendix · Formula Sheet (Auxiliary Material)

> *The formulas you would otherwise look up repeatedly, collected on one page. This follows Kreyszig's Appendix 3.*

[← Agentic Workflows](../part-h-math-for-ai/36-agentic-workflows.md) · [Index](../../README.md)

---

## A3.1 · Special functions

### Exponential, logarithm, hyperbolic

| | |
|---|---|
| $e^{a+b} = e^ae^b$, $\ \ln(ab) = \ln a + \ln b$, $\ a^x = e^{x\ln a}$ | $\cosh x = \tfrac12(e^x + e^{-x})$, $\ \sinh x = \tfrac12(e^x - e^{-x})$ |
| $\cosh^2 x - \sinh^2 x = 1$ | $\cos(ix) = \cosh x$, $\ \sin(ix) = i\sinh x$ |

### Trigonometric identities

| | |
|---|---|
| $\sin^2 x + \cos^2 x = 1$ | $e^{ix} = \cos x + i\sin x$ |
| $\sin(x\pm y) = \sin x\cos y \pm \cos x\sin y$ | $\cos(x\pm y) = \cos x\cos y \mp \sin x\sin y$ |
| $\sin 2x = 2\sin x\cos x$ | $\cos 2x = \cos^2 x - \sin^2 x = 1 - 2\sin^2 x$ |
| $\sin^2 x = \tfrac12(1 - \cos 2x)$ | $\cos^2 x = \tfrac12(1 + \cos 2x)$ |
| $\sin x\cos y = \tfrac12[\sin(x+y) + \sin(x-y)]$ | $\cos x\cos y = \tfrac12[\cos(x+y) + \cos(x-y)]$ |
| $A\cos x + B\sin x = R\cos(x - \delta)$ | $R = \sqrt{A^2 + B^2},\ \tan\delta = B/A$ |

### Gamma, Beta, error function

| Function | Definition | Key facts |
|---|---|---|
| Gamma | $\Gamma(\alpha) = \int_0^\infty e^{-t}t^{\alpha-1}\,dt$ | $\Gamma(\alpha+1) = \alpha\Gamma(\alpha)$, $\ \Gamma(n+1) = n!$, $\ \Gamma(\tfrac12) = \sqrt\pi$ |
| Beta | $B(x, y) = \int_0^1 t^{x-1}(1-t)^{y-1}\,dt$ | $B(x,y) = \frac{\Gamma(x)\Gamma(y)}{\Gamma(x+y)}$ |
| Error function | $\operatorname{erf}x = \frac{2}{\sqrt\pi}\int_0^x e^{-t^2}\,dt$ | $\Phi(z) = \tfrac12\left[1 + \operatorname{erf}(z/\sqrt2)\right]$ (normal CDF) |
| Stirling | $n! \approx \sqrt{2\pi n}\,(n/e)^n$ | relative error ≈ $1/(12n)$ |

### Bessel & Legendre recurrences

| | |
|---|---|
| $J_{\nu-1}(x) + J_{\nu+1}(x) = \frac{2\nu}{x}J_\nu(x)$ | $[x^\nu J_\nu(x)]' = x^\nu J_{\nu-1}(x)$ |
| $J_{1/2}(x) = \sqrt{\frac{2}{\pi x}}\sin x$ | $J_{-1/2}(x) = \sqrt{\frac{2}{\pi x}}\cos x$ |
| $(n+1)P_{n+1} = (2n+1)xP_n - nP_{n-1}$ | generating function $\frac{1}{\sqrt{1 - 2xu + u^2}} = \sum P_n(x)u^n$ |

## A3.2 · Partial derivatives

| Rule | Formula |
|---|---|
| Chain rule, $w = f(x(t), y(t))$ | $\dfrac{dw}{dt} = f_x\dfrac{dx}{dt} + f_y\dfrac{dy}{dt}$ |
| Chain rule, $w = f(x(u,v), y(u,v))$ | $w_u = f_x x_u + f_y y_u$ |
| Total differential | $dw = f_x\,dx + f_y\,dy + f_z\,dz$ |
| Implicit function $F(x, y) = 0$ | $\dfrac{dy}{dx} = -\dfrac{F_x}{F_y}$ |
| Mixed partials (Schwarz) | $f_{xy} = f_{yx}$ if continuous |
| Taylor (2 variables) | $f(\mathbf{x} + \mathbf{h}) \approx f + \nabla f^T\mathbf{h} + \tfrac12\mathbf{h}^TH\mathbf{h}$ |

## A3.3 · Sequences & series

| Series | Sum / behaviour |
|---|---|
| Geometric $\sum_{n=0}^\infty q^n$ | $\frac{1}{1-q}$ for $\vert q\vert < 1$ |
| Finite geometric $\sum_{n=0}^{N-1} q^n$ | $\frac{1 - q^N}{1 - q}$ |
| $\sum_{n=1}^{N} n$, $\ \sum n^2$ | $\frac{N(N+1)}{2}$, $\ \frac{N(N+1)(2N+1)}{6}$ |
| Harmonic $\sum 1/n$ | diverges; $\sum_{n=1}^{N} 1/n \approx \ln N + 0.5772$ |
| $p$-series $\sum 1/n^p$ | converges iff $p > 1$; $\sum 1/n^2 = \pi^2/6$ |
| Alternating harmonic | $\sum (-1)^{n+1}/n = \ln 2$ |
| $\sum q^n/n!$ | $e^q$ |

**Tests:** ratio $\left\vert a_{n+1}/a_n\right\vert \to L < 1$ converges; root $\sqrt[n]{\vert a_n\vert} \to L < 1$ converges; comparison, integral, alternating-series (Leibniz) tests. Absolute convergence implies convergence.

## A3.4 · Grad, div, curl, ∇² in curvilinear coordinates

### Cylindrical $(r, \theta, z)$ with $x = r\cos\theta,\ y = r\sin\theta$

```math
\nabla f = f_r\,\mathbf{e}_r + \frac{1}{r}f_\theta\,\mathbf{e}_\theta + f_z\,\mathbf{e}_z, \qquad
\nabla\cdot\mathbf{v} = \frac{1}{r}\frac{\partial(r v_r)}{\partial r} + \frac{1}{r}\frac{\partial v_\theta}{\partial\theta} + \frac{\partial v_z}{\partial z}
```

```math
\nabla^2 u = u_{rr} + \frac{1}{r}u_r + \frac{1}{r^2}u_{\theta\theta} + u_{zz}
```

```math
\nabla\times\mathbf{v} = \left(\frac{1}{r}\frac{\partial v_z}{\partial\theta} - \frac{\partial v_\theta}{\partial z}\right)\mathbf{e}_r + \left(\frac{\partial v_r}{\partial z} - \frac{\partial v_z}{\partial r}\right)\mathbf{e}_\theta + \frac{1}{r}\left(\frac{\partial(r v_\theta)}{\partial r} - \frac{\partial v_r}{\partial\theta}\right)\mathbf{e}_z
```

### Spherical $(r, \theta, \phi)$: $\theta$ = polar angle from $+z$, $\phi$ = azimuth

```math
\nabla f = f_r\,\mathbf{e}_r + \frac{1}{r}f_\theta\,\mathbf{e}_\theta + \frac{1}{r\sin\theta}f_\phi\,\mathbf{e}_\phi, \qquad
\nabla\cdot\mathbf{v} = \frac{1}{r^2}\frac{\partial(r^2 v_r)}{\partial r} + \frac{1}{r\sin\theta}\frac{\partial(\sin\theta\,v_\theta)}{\partial\theta} + \frac{1}{r\sin\theta}\frac{\partial v_\phi}{\partial\phi}
```

```math
\nabla^2 u = \frac{1}{r^2}\frac{\partial}{\partial r}\left(r^2\frac{\partial u}{\partial r}\right) + \frac{1}{r^2\sin\theta}\frac{\partial}{\partial\theta}\left(\sin\theta\frac{\partial u}{\partial\theta}\right) + \frac{1}{r^2\sin^2\theta}\frac{\partial^2 u}{\partial\phi^2}
```

```math
\nabla\times\mathbf{v} = \frac{1}{r\sin\theta}\left(\frac{\partial(\sin\theta\,v_\phi)}{\partial\theta} - \frac{\partial v_\theta}{\partial\phi}\right)\mathbf{e}_r + \frac{1}{r}\left(\frac{1}{\sin\theta}\frac{\partial v_r}{\partial\phi} - \frac{\partial(r v_\phi)}{\partial r}\right)\mathbf{e}_\theta + \frac{1}{r}\left(\frac{\partial(r v_\theta)}{\partial r} - \frac{\partial v_r}{\partial\theta}\right)\mathbf{e}_\phi
```

**Volume elements:** cylindrical $dV = r\,dr\,d\theta\,dz$; spherical $dV = r^2\sin\theta\,dr\,d\theta\,d\phi$.

## Useful constants & quick conversions

| | |
|---|---|
| $e \approx 2.71828$, $\ \pi \approx 3.14159$, $\ \gamma_{\text{Euler}} \approx 0.57722$ | $1\ \text{nat} = 1.4427$ bits, $\ \ln 2 = 0.6931$ |
| $\Phi^{-1}(0.975) = 1.960$, $\ \Phi^{-1}(0.95) = 1.645$ | 68.3% / 95.4% / 99.7% within 1σ / 2σ / 3σ |
| $2^{10} = 1024 \approx 10^3$ | 1 dB = $10\log_{10}$(power ratio); 1 bit of quantization ≈ 6.02 dB |

## About the book's other appendices

Kreyszig's **Appendix 1** (references), **Appendix 2** (answers to odd-numbered problems), **Appendix 4** (additional proofs) and **Appendix 5** (tables of Bessel, Gamma, normal, t, χ² and F distributions) are meant to be used alongside the book. For the tables, `scipy.special` and `scipy.stats` give any value to machine precision, for example `scipy.stats.t.ppf(0.975, df=10)`.

---
[← Agentic Workflows](../part-h-math-for-ai/36-agentic-workflows.md) · [Index](../../README.md)
