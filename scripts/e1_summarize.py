#!/usr/bin/env python3
"""Summarize results/ep_trained_vs_rescaled/trained_vs_rescaled.csv.

E1 asks one question: at MATCHED weight norm, does a trained network give a
better EP gradient than a randomly-initialized one? Three arms answer it:

  trained           the snapshot as trained
  rescaled          random init rescaled to the snapshot's norms
  trained_rescaled  epoch-1 trained weights rescaled to the snapshot's norms

If `trained` holds up where `rescaled` collapses, the published probe overstates
the effect and the 83% run is not a contradiction. If `trained_rescaled` tracks
`rescaled`, the norm is what matters; if it tracks `trained`, the structure is.

Rows where the FD bracket did not converge as the step shrank (fd_trusted=false)
are excluded from every conclusion and counted separately -- at those points the
loss is not locally linear and no cosine against it means anything.
"""
import csv, sys, collections, statistics as st

PATH = sys.argv[1] if len(sys.argv) > 1 else \
    "results/ep_trained_vs_rescaled/trained_vs_rescaled.csv"
rows = list(csv.DictReader(open(PATH)))
F = lambda r, k: float(r[k])
trusted = [r for r in rows if r["fd_trusted"] == "true"]

print(f"{len(rows)} rows | {len(trusted)} trusted "
      f"({len(rows)-len(trusted)} excluded: FD bracket did not converge)\n")

# ---- headline: cos at the operating beta, by arm and weight norm -----------
print("=" * 78)
print("A. Gradient fidelity vs weight norm, at matched norm")
print("=" * 78)
for run in ("nodecay", "wd1e-4"):
    sub = [r for r in trusted if r["run"] == run]
    if not sub:
        continue
    print(f"\n-- {run} --")
    print(f"{'|W1|':>7} {'ep':>3} {'arm':17} {'acc':>6} "
          f"{'1side b=.1':>11} {'cent b=.1':>10} {'lockin':>8} {'hi-agree':>9}")
    key = lambda r: (float(r["w1_norm"]), r["tag"])
    for (w1, tag) in sorted({key(r) for r in sub}):
        cell = [r for r in sub if key(r) == (w1, tag)]
        pick = lambda est, b=None: next(
            (F(c, "cos_fd") for c in cell
             if c["estimator"] == est and (b is None or abs(F(c, "beta") - b) < 1e-9)),
            float("nan"))
        c0 = cell[0]
        print(f"{w1:7.1f} {c0['epoch']:>3} {tag:17} {F(c0,'acc'):6.3f} "
              f"{pick('static_onesided', 0.1):11.4f} "
              f"{pick('static_centered', 0.1):10.4f} "
              f"{pick('lockin'):8.4f} {F(c0,'fd_agree_hi'):9.4f}")

# ---- the 1/beta diagnostic -------------------------------------------------
print("\n" + "=" * 78)
print("B. The 1/beta signature (one-sided). Relative error rising as beta falls")
print("   means a beta-independent term survives -> different fixed points.")
print("=" * 78)
for arm in ("trained", "rescaled", "trained_rescaled"):
    sub = [r for r in trusted if r["tag"] == arm and r["estimator"] == "static_onesided"]
    if not sub:
        continue
    betas = sorted({F(r, "beta") for r in sub}, reverse=True)
    print(f"\n-- {arm} -- (median relerr over snapshots)")
    print("   |W1| bin " + "".join(f"{b:>11.3f}" for b in betas) + "   slope")
    bins = collections.defaultdict(list)
    for r in sub:
        bins[round(F(r, "w1_norm") / 20) * 20].append(r)
    for b0 in sorted(bins):
        vals = []
        for b in betas:
            v = [F(r, "relerr_fd") for r in bins[b0] if abs(F(r, "beta") - b) < 1e-9]
            vals.append(st.median(v) if v else float("nan"))
        ok = [(b, v) for b, v in zip(betas, vals) if v == v and v > 0]
        slope = ""
        if len(ok) >= 3:
            import math
            xs = [math.log(b) for b, _ in ok]; ys = [math.log(v) for _, v in ok]
            n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
            d = sum((x-mx)**2 for x in xs)
            if d > 0:
                slope = f"{sum((x-mx)*(y-my) for x,y in zip(xs,ys))/d:>7.2f}"
        print(f"   ~{b0:>4}    " + "".join(f"{v:>11.3f}" for v in vals) + f"  {slope}")
print("\n   slope ~ -1 is the basin-hopping fingerprint; ~0 means the estimator")
print("   is clean and shrinking beta costs nothing.")

# ---- centered vs one-sided -------------------------------------------------
print("\n" + "=" * 78)
print("C. Does `centered` rescue it? (median cos over all trusted snapshots)")
print("=" * 78)
print(f"{'arm':18}{'estimator':18}{'median cos':>12}{'p10 cos':>10}{'n':>5}")
for arm in ("trained", "rescaled", "trained_rescaled"):
    for est in ("static_onesided", "static_centered", "lockin"):
        v = sorted(F(r, "cos_fd") for r in trusted
                   if r["tag"] == arm and r["estimator"] == est
                   and (est == "lockin" or abs(F(r, "beta") - 0.1) < 1e-9))
        if v:
            p10 = v[max(0, int(0.1 * len(v)) - 1)]
            print(f"{arm:18}{est:18}{st.median(v):12.4f}{p10:10.4f}{len(v):5d}")
