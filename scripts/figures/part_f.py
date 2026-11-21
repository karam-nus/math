"""Part F -- Optimization, Graphs (Chapters 22-23)."""
import heapq

import numpy as np

from style import figure, note


# ---------------------------------------------------------------- Chapter 22
@figure("ch22-descent-and-lp", size=(11, 4.3))
def ch22(fig, T):
    ax1, ax2 = fig.subplots(1, 2)
    g = np.linspace(-6.5, 6.5, 300)
    X, Y = np.meshgrid(g, np.linspace(-3, 3, 200))
    F = X**2 + 8 * Y**2
    ax1.contour(X, Y, F, levels=np.geomspace(0.3, 80, 12), colors=T.grid if T.name == "dark" else T.axis, linewidths=1)
    p = np.array([-6.0, 2.0]); path = [p.copy()]
    for _ in range(14):
        gr = np.array([2 * p[0], 16 * p[1]])
        Hg = np.array([2 * gr[0], 16 * gr[1]])
        t = gr @ gr / (gr @ Hg)  # exact line search
        p = p - t * gr; path.append(p.copy())
    path = np.array(path)
    ax1.plot(path[:, 0], path[:, 1], "-o", color=T.orange, ms=4.5, lw=1.8, label="steepest descent")
    ax1.plot([-6, 0], [2, 0], "--", color=T.blue, lw=2, label="Newton (1 step)")
    ax1.plot(0, 0, "*", ms=14, color=T.ink)
    ax1.set_aspect("equal"); ax1.set_xlim(-6.5, 6.5); ax1.set_ylim(-3, 3)
    ax1.legend(loc="upper right")
    ax1.set_title("Steepest descent zig-zags in a valley")
    ax1.set_xlabel("f = x² + 8y²   (condition number 8)")

    verts = np.array([[0, 0], [4, 0], [4, 2], [3, 3], [0, 4]])
    ax2.fill(verts[:, 0], verts[:, 1], color=T.blue, alpha=0.12)
    ax2.plot(*np.vstack([verts, verts[:1]]).T, color=T.blue, lw=2)
    xs = np.linspace(-0.5, 5.5, 50)
    for c, a in ((4, 0.5), (8, 0.7), (12, 1.0)):
        ax2.plot(xs, (c - 2 * xs) / 2 if False else (c - 3 * xs) / 2, color=T.muted, lw=1.2, ls="--")
    # objective z = 3x1 + 2x2 ; optimum at (3,3)? z(4,2)=16, z(3,3)=15, -> (4,2)
    zs = {tuple(v): 3 * v[0] + 2 * v[1] for v in verts}
    route = [(0, 0), (4, 0), (4, 2)]
    for a, b in zip(route[:-1], route[1:]):
        ax2.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=T.orange, lw=2.4, mutation_scale=16))
    for v in verts:
        ax2.plot(*v, "o", ms=8, color=T.ink if tuple(v) != (4, 2) else T.orange, mec=T.surface, mew=1.5)
        note(ax2, v[0] + 0.1, v[1] + 0.12, f"z={zs[tuple(v)]}", T, fontsize=8.5)
    xx = np.linspace(2.5, 5.2, 10)
    ax2.plot(xx, (16 - 3 * xx) / 2, color=T.orange, lw=1.4)
    note(ax2, 4.25, 3.0, "maximise\nz = 3x₁ + 2x₂", T, color=T.ink)
    ax2.set_xlim(-0.5, 5.5); ax2.set_ylim(-0.5, 4.8)
    ax2.set_title("Linear programming: the optimum sits at a corner")
    ax2.set_xlabel("x₁"); ax2.set_ylabel("x₂")
    note(ax2, 1.6, 4.3, "dashed: level lines of z\narrows: simplex walk", T, fontsize=8.8, box=True)


# ---------------------------------------------------------------- Chapter 23
NODES = {"s": (0, 1.5), "a": (1.2, 2.8), "b": (1.3, 0.3), "c": (2.6, 1.7), "d": (3.0, 3.2),
         "e": (3.2, 0.0), "f": (4.3, 1.6), "t": (5.4, 2.6)}
EDGES = [("s", "a", 4), ("s", "b", 2), ("a", "b", 5), ("a", "c", 3), ("a", "d", 6), ("b", "c", 4),
         ("b", "e", 7), ("c", "d", 2), ("c", "e", 3), ("c", "f", 5), ("d", "t", 4), ("e", "f", 2),
         ("f", "t", 3), ("d", "f", 4)]


def _dijkstra(src="s"):
    adj = {n: [] for n in NODES}
    for u, v, w in EDGES:
        adj[u].append((v, w)); adj[v].append((u, w))
    dist = {n: np.inf for n in NODES}; prev = {}
    dist[src] = 0; pq = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            if d + w < dist[v]:
                dist[v] = d + w; prev[v] = u; heapq.heappush(pq, (dist[v], v))
    return dist, {(prev[v], v) for v in prev}


def _kruskal():
    parent = {n: n for n in NODES}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    tree = set()
    for u, v, w in sorted(EDGES, key=lambda e: e[2]):
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv; tree.add((u, v))
    return tree


def _draw(ax, T, chosen, title, labels=None):
    chosen = {frozenset(e) for e in chosen}
    for u, v, w in EDGES:
        (x1, y1), (x2, y2) = NODES[u], NODES[v]
        on = frozenset((u, v)) in chosen
        ax.plot([x1, x2], [y1, y2], color=T.orange if on else T.axis, lw=3.2 if on else 1.4, zorder=1)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my, str(w), fontsize=9, ha="center", va="center", color=T.ink2, zorder=3,
                bbox=dict(boxstyle="circle,pad=0.15", fc=T.surface, ec="none"))
    for n, (x, y) in NODES.items():
        ax.scatter([x], [y], s=520, color=T.blue, edgecolors=T.surface, linewidths=2, zorder=4)
        ax.text(x, y, n, ha="center", va="center", color="#ffffff", fontsize=10.5, fontweight="bold", zorder=5)
        if labels:
            ax.text(x, y + 0.38, labels[n], ha="center", fontsize=9, color=T.ink, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc=T.surface, ec="none", alpha=0.9))
    ax.set_aspect("equal"); ax.set_axis_off()
    ax.set_xlim(-0.5, 5.9); ax.set_ylim(-0.5, 3.75)
    ax.set_title(title)


@figure("ch23-dijkstra-mst", size=(11, 3.9))
def ch23(fig, T):
    ax1, ax2 = fig.subplots(1, 2)
    dist, tree = _dijkstra()
    _draw(ax1, T, tree, "Dijkstra: shortest-path tree from s", {k: f"d={int(v)}" for k, v in dist.items()})
    mst = _kruskal()
    total = sum(w for u, v, w in EDGES if (u, v) in mst or (v, u) in mst)
    _draw(ax2, T, mst, f"Minimum spanning tree (Kruskal/Prim): weight {total}")
