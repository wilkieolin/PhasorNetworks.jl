# routing_layers.py — wiring the radius-3 stencil with two routing layers and vias.
#
# Straight delay lines (length = delay, so no detours) between every pair of
# cells within three pitches: crossings per line, spacing between consecutive
# crossings, and the split of the 14 line directions over two default layers
# that minimises same-layer crossings (each costs one hop-under, <= 4 vias).
# Run: python3 demos/device_studies/routing_layers.py
import math, numpy as np
R = 3; P = 129.0
offs = [(dy, dx) for dy in range(-R, R+1) for dx in range(-R, R+1) if 0 < dy*dy+dx*dx <= R*R]
cls = sorted({max(o, (-o[0], -o[1])) for o in offs})
n = 8
S = [((y, x), (y+dy, x+dx)) for y in range(-n, n+1) for x in range(-n, n+1) for dy, dx in cls]
A = np.array([s[0] for s in S], float); B = np.array([s[1] for s in S], float)
def crossings(i):
    a, b = A[i], B[i]; d = b - a
    e = B - A; w = A - a
    den = d[0]*e[:, 1] - d[1]*e[:, 0]
    ok = np.abs(den) > 1e-12
    t = np.where(ok, (w[:, 0]*e[:, 1] - w[:, 1]*e[:, 0]) / np.where(ok, den, 1), np.nan)
    u = np.where(ok, (w[:, 0]*d[1] - w[:, 1]*d[0]) / np.where(ok, den, 1), np.nan)
    m = ok & (t > 1e-9) & (t < 1-1e-9) & (u > 1e-9) & (u < 1-1e-9)
    # also lines passing through an intermediate lattice node count as crossing that node's lines
    return np.sort(t[m]) * np.linalg.norm(d) * P          # positions along the line, µm
counts, gaps, through = [], [], 0
for i, (a, b) in enumerate(S):
    if a != (0, 0): continue
    pos = crossings(i)
    up = np.unique(np.round(pos, 6))
    counts.append((b, len(pos), len(up)))
    if len(up) > 1: gaps += list(np.diff(up))
L = sum(math.hypot(*o) for o in cls)
print("per cell (14 line pairs):")
for (b, c, u) in counts:
    print(f"  offset {b}: length {math.hypot(*b)*P:5.0f} µm, crosses {c:3d} lines at {u:3d} distinct points")
tot = sum(c for _, c, _ in counts)
print("total crossings per cell (each counted once per pair of lines):", tot // 2 * 1, "line-pairs; x2 lines per pair ->", tot*2, "wire crossings")
g = np.array(gaps)
print(f"gap between consecutive crossing points along a line: min {g.min():.1f} µm, 5th pct {np.percentile(g,5):.1f}, median {np.median(g):.1f} µm")

# crossings between direction classes, per cell (translation invariant)
K = len(cls); W = np.zeros((K, K))
for i, (a, b) in enumerate(S):
    if a != (0, 0): continue
    ki = cls.index((b[0]-a[0], b[1]-a[1]))
    d = B[i]-A[i]; e = B-A; w = A-A[i]
    den = d[0]*e[:, 1]-d[1]*e[:, 0]; ok = np.abs(den) > 1e-12
    t = np.where(ok, (w[:, 0]*e[:, 1]-w[:, 1]*e[:, 0])/np.where(ok, den, 1), np.nan)
    u = np.where(ok, (w[:, 0]*d[1]-w[:, 1]*d[0])/np.where(ok, den, 1), np.nan)
    m = ok & (t > 1e-9) & (t < 1-1e-9) & (u > 1e-9) & (u < 1-1e-9)
    for j in np.where(m)[0]:
        kj = cls.index((B[j][0]-A[j][0], B[j][1]-A[j][1])); W[ki, kj] += 1
W = (W + W.T) / 2          # crossings per cell between class pairs (each crossing counted once per cell)
total = W[np.triu_indices(K, 1)].sum() + np.trace(W) / 2
best = None
for mask in range(1 << (K-1)):
    g = np.array([(mask >> k) & 1 for k in range(K)])
    same = sum(W[i, j] for i in range(K) for j in range(i+1, K) if g[i] == g[j]) + np.trace(W)/2
    if best is None or same < best[0]: best = (same, g)
same, g = best
print(f"pair-crossings per cell: {total:.0f}; best split into two default layers leaves {same:.0f} same-layer crossings")
print("  layer A:", [cls[k] for k in range(K) if g[k] == 0])
print("  layer B:", [cls[k] for k in range(K) if g[k] == 1])
print(f"  each same-layer crossing: one line hops under (2 vias); both lines of a pair -> x4 at most: <= {4*same:.0f} vias per cell")
