#!/usr/bin/env python3
"""Compare final test accuracy across ReLU / feedforward-backprop / lock-in EP.

Reads the three arms' CSVs, reports the distributions, then runs:

  * Levene, because the arms are not expected to have equal variance -- lock-in
    EP's spread is several times backprop's, and classical ANOVA assumes it is not.
  * classical one-way ANOVA and Welch's ANOVA. Where they disagree, believe Welch.
  * pairwise Welch t-tests with Holm correction.
  * TOST equivalence against a stated margin. This is the test that can support
    "these are the same"; a non-significant ANOVA cannot, it only fails to reject.

    python3 scripts/e8_analysis.py [--margin 0.01] [--emit]

--emit additionally writes the tidy tables Figure 5 reads, into the manuscript's
figures/data/. Note that this is a SECOND bridge into figures/data/ alongside
export_figure_data.jl, which is otherwise the only one. It is here rather than
there because these are scipy statistics -- Welch intervals, TOST, Levene,
Alexander-Govern -- and reimplementing them in Julia to honour the convention
would create two versions of the tests to keep in agreement. One audited place
per number is the point of the convention; this keeps it.
"""
import argparse, csv, glob as _glob, itertools, math, os, sys
from collections import defaultdict

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MARGIN_DEFAULT = 0.01          # 1 accuracy point
FINAL_EPOCH = 8                # runs are 8 epochs; anything short of that is in flight

# Filtering on "the largest epoch present in this file" is WRONG while a shard is
# still running -- an in-progress file then contributes its current epoch as if it
# were the final one, silently mixing half-trained networks into the sample. Match
# on FINAL_EPOCH explicitly instead.


def final_epoch_rows(path, arm_col, acc_col, epoch_col, seed_col, want=None):
    """Last-epoch accuracy per (arm, seed). Duplicate keys keep the first."""
    out, seen = defaultdict(dict), set()
    if not os.path.exists(path):
        return out
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return out
    for r in rows:
        if int(r[epoch_col]) != FINAL_EPOCH:
            continue
        arm = r[arm_col] if want is None else want
        key = (arm, r[seed_col])
        if key in seen:
            continue
        seen.add(key)
        out[arm][r[seed_col]] = float(r[acc_col])
    return out


groups = defaultdict(dict)
# arms may have been run on either machine; glob both trees
import glob
for path in sorted(glob.glob(f"{ROOT}/results/e8_baselines*/*.csv")):
    for k, v in final_epoch_rows(path, "arm", "test_acc", "epoch", "seed").items():
        for seed, acc in v.items():
            groups[k].setdefault(seed, acc)
# Point B was run at two learning rates and they behave differently, so they are
# separate arms: 0.003 is the rate the basin sweep used, 0.001 matches the
# backprop baselines. Keying both into one group would average a bimodal arm
# together with a clean one.
for p in sorted(glob.glob(f"{ROOT}/results/e6_basin_training/e6_basin_*.csv")):
    for r in csv.DictReader(open(p)):
        if r["point"] != "B" or int(r["epoch"]) != FINAL_EPOCH:
            continue
        arm = "lockin_B" if float(r["lr"]) > 0.002 else "lockin_B_lr001"
        groups[arm].setdefault(r["seed"], float(r["test_acc"]))

ap = argparse.ArgumentParser()
ap.add_argument("--margin", type=float, default=MARGIN_DEFAULT)
ap.add_argument("--arms", default=None,
                help="comma-separated subset to analyse, e.g. to exclude a control "
                     "so the reported omnibus matches what a figure plots")
ap.add_argument("--emit", action="store_true",
                help="write figures/data/fig5_*.dat and fig5_macros.tex")
args = ap.parse_args()

ORDER = [a for a in ("relu", "phasor_bp", "phasor_bp_flip",
                     "lockin_B", "lockin_B_lr001")
         if len(groups.get(a, {})) > 1]
if args.arms:
    want = [t.strip() for t in args.arms.split(",") if t.strip()]
    missing = [a for a in want if a not in ORDER]
    if missing:
        sys.exit(f"--arms names arms with no data: {missing}; have {ORDER}")
    ORDER = want
if len(ORDER) < 2:
    sys.exit("need at least two arms with >1 seed; have " +
             str({k: len(v) for k, v in groups.items()}))

print("FINAL-EPOCH TEST ACCURACY\n")
data = {}
for a in ORDER:
    v = np.array(sorted(groups[a].values()))
    data[a] = v
    print(f"  {a:10s} n={len(v):3d}  mean {v.mean():.4f}  sd {v.std(ddof=1):.4f}  "
          f"median {np.median(v):.4f}  min {v.min():.4f}  max {v.max():.4f}")
    print(f"             {np.round(v, 4).tolist()}")

# A large sd can mean a shifted-but-tight arm or a bimodal one, and those call
# for completely different summaries. Detect a gap in the sorted values: if the
# biggest jump is several times the typical spacing, report the two modes and the
# failure rate rather than pretending a single mean describes the arm.
SPLIT = {}                     # arm -> (threshold, n_collapsed, wilson_lo, wilson_hi)
print("\nBIMODALITY CHECK  (a large sd may be a shifted arm or a split one)")
for a in ORDER:
    v = data[a]
    if len(v) < 5:
        continue
    gaps = np.diff(v)
    k = int(np.argmax(gaps))
    typical = np.median(gaps[gaps > 0]) if np.any(gaps > 0) else 0.0
    if typical > 0 and gaps[k] > 6 * typical and 0 < k + 1 < len(v):
        lo, hi = v[:k + 1], v[k + 1:]
        n, f = len(v), len(lo)
        z = 1.96
        c = (f / n + z**2 / (2 * n)) / (1 + z**2 / n)
        h = z * math.sqrt((f / n) * (1 - f / n) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
        SPLIT[a] = (0.5 * (v[k] + v[k + 1]), f, max(0.0, c - h), min(1.0, c + h))
        print(f"  {a}: SPLIT at a gap of {gaps[k]:.4f} ({gaps[k]/typical:.0f}x typical)")
        print(f"     collapsed  n={len(lo):2d}  mean {lo.mean():.4f}  {np.round(lo,4).tolist()}")
        print(f"     healthy    n={len(hi):2d}  mean {hi.mean():.4f}  sd {hi.std(ddof=1):.4f}")
        print(f"     failure rate {f}/{n} = {f/n*100:.0f}%  (Wilson 95% CI "
              f"{max(0,c-h)*100:.0f}-{min(1,c+h)*100:.0f}%)")
        print(f"     -> quote the rate and the healthy mean; the pooled mean "
              f"{v.mean():.4f} describes no actual run")
    else:
        print(f"  {a}: unimodal (largest gap {gaps[k]:.4f})")

print("\nVARIANCE HOMOGENEITY")
W, pl = stats.levene(*[data[a] for a in ORDER], center="median")
print(f"  Levene  W={W:.3f}  p={pl:.4g}   " +
      ("variances differ -- classical ANOVA is not appropriate" if pl < 0.05
       else "no evidence against equal variance"))

print("\nOMNIBUS")
F, pf = stats.f_oneway(*[data[a] for a in ORDER])
print(f"  one-way ANOVA   F={F:.3f}  p={pf:.4g}")
try:
    Fw, pw = stats.alexandergovern(*[data[a] for a in ORDER]).statistic, \
             stats.alexandergovern(*[data[a] for a in ORDER]).pvalue
    print(f"  Alexander-Govern (unequal variance)  A={Fw:.3f}  p={pw:.4g}")
except Exception as e:
    print(f"  (unequal-variance omnibus unavailable: {e})")

print(f"\nPAIRWISE (Welch), Holm-corrected, and TOST at +/-{args.margin:.3f}")
pairs = list(itertools.combinations(ORDER, 2))
raw = []
for a, c in pairs:
    t, p = stats.ttest_ind(data[a], data[c], equal_var=False)
    raw.append(p)
order = np.argsort(raw)
holm = [0.0] * len(raw)
for rank, i in enumerate(order):
    holm[i] = min(1.0, raw[i] * (len(raw) - rank))
for i in range(1, len(order)):
    holm[order[i]] = max(holm[order[i]], holm[order[i - 1]])

CONTRASTS = []
for (a, c), p_raw, p_adj in zip(pairs, raw, holm):
    x, y = data[a], data[c]
    d = x.mean() - y.mean()
    se = math.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
    df = se**4 / (x.var(ddof=1)**2 / (len(x)**2 * (len(x) - 1)) +
                  y.var(ddof=1)**2 / (len(y)**2 * (len(y) - 1)))
    tcrit = stats.t.ppf(0.95, df)                       # 90% CI, the TOST interval
    lo, hi = d - tcrit * se, d + tcrit * se
    p_tost = max(stats.t.sf((d + args.margin) / se, df),
                 stats.t.cdf((d - args.margin) / se, df))
    equiv = "EQUIVALENT" if p_tost < 0.05 else "not shown equivalent"
    diff = "differ" if p_adj < 0.05 else "no sig. difference"
    print(f"  {a:10s} vs {c:10s}  diff {d:+.4f}  90% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"      Welch p={p_raw:.4g}  Holm p={p_adj:.4g}  -> {diff}")
    print(f"      TOST  p={p_tost:.4g}  -> {equiv}")
    CONTRASTS.append((a, c, d, lo, hi, p_raw, p_adj, p_tost))

print(f"\nNote: 'no significant difference' is failure to reject, not evidence of")
print(f"equivalence. Only a significant TOST supports sameness, and then only to")
print(f"within the stated +/-{args.margin:.3f} margin.")


# --------------------------------------------------------------- figure data
if args.emit:
    import os as _os
    OUT = _os.environ.get("LIEP_FIGDATA", _os.path.join(
        _os.path.expanduser("~"), "Documents", "ICRC-LIEP-manuscript", "figures", "data"))
    _os.makedirs(OUT, exist_ok=True)
    prov = (f"# generated by scripts/e8_analysis.py --emit "
            f"(margin {args.margin}, final epoch {FINAL_EPOCH})\n"
            f"# sources: results/e8_baselines*/*.csv, "
            f"results/e6_basin_training/e6_basin_*.csv (point B)\n")

    with open(_os.path.join(OUT, "fig5_runs.dat"), "w") as fh:
        fh.write(prov)
        fh.write("# one row per training run; collapsed=1 marks the low mode of a split arm\n")
        fh.write("arm idx acc collapsed\n")
        for a in ORDER:
            thr = SPLIT.get(a, (None,))[0]
            for i, val in enumerate(data[a]):
                fh.write(f"{a} {i} {val:.6f} {int(thr is not None and val < thr)}\n")

    with open(_os.path.join(OUT, "fig5_contrasts.dat"), "w") as fh:
        fh.write(prov)
        fh.write("# Welch pairwise contrasts; ci_lo/ci_hi are the 90% (TOST) interval\n")
        fh.write("a b delta ci_lo ci_hi p_welch p_holm p_tost\n")  # not "diff": DataFrame.diff
        for a, c, d, lo, hi, pr, pa, pt in CONTRASTS:
            fh.write(f"{a} {c} {d:.6f} {lo:.6f} {hi:.6f} {pr:.6g} {pa:.6g} {pt:.6g}\n")

    # Two protocol facts that must reach the caption, derived rather than asserted:
    # the learning rates differ, and the evaluation set sizes differ. The test-set
    # size is not recorded in either CSV, so recover it from the quantisation of the
    # accuracies -- k/n can only land on multiples of 1/n.
    def _denom(v):
        for n in (500, 1000, 2000, 2500, 5000, 10000):
            if all(abs(x * n - round(x * n)) < 1e-6 for x in v):
                return n
        return None

    lrs = {}
    for path in sorted(_glob.glob(f"{ROOT}/results/e8_baselines*/*.csv")):
        for r in csv.DictReader(open(path)):
            lrs.setdefault(r["arm"], r["lr"])
    for pth in sorted(_glob.glob(f"{ROOT}/results/e6_basin_training/e6_basin_*.csv")):
        if True:
            for r in csv.DictReader(open(pth)):
                if r["point"] == "B":
                    k = "lockin_B" if float(r["lr"]) > 0.002 else "lockin_B_lr001"
                    lrs.setdefault(k, r["lr"])

    TAGS = {"relu": "Relu", "phasor_bp": "Phasorbp", "phasor_bp_flip": "Phasorbpflip",
            "lockin_B": "Lockinb", "lockin_B_lr001": "Lockinlo"}

    def _tag(a):
        # digits are illegal in a LaTeX control sequence
        return TAGS.get(a, "".join(c for c in a.replace("_", "").title()
                                   if not c.isdigit()))

    def _mac(fh, name, val):
        # \providecommand, not \newcommand: these numbers are cited by both the
        # baselines figure and Fig. 1's comparison panel, so the file can be
        # \input twice in one document and \newcommand would abort on the second.
        fh.write("\\providecommand{\\%s}{%s}\n" % (name, val))

    with open(_os.path.join(OUT, "fig5_macros.tex"), "w") as fh:
        fh.write("% generated by scripts/e8_analysis.py --emit -- do not edit\n")
        _mac(fh, "FigBlMargin", f"{args.margin:.3f}")
        for a in ORDER:
            tag = _tag(a)
            _mac(fh, f"FigBlLr{tag}", f"{float(lrs[a]):g}")
            d = _denom(data[a])
            _mac(fh, f"FigBlNtest{tag}", str(d) if d else "?")
        # the extra, common uncertainty a smaller eval subset puts on an arm's level
        _dl, _db = _denom(data["lockin_B"]), _denom(data["relu"])
        if _dl and _db and _dl < _db:
            _p = float(data["lockin_B"].mean())
            _mac(fh, "FigBlSubsetSd",
                 f"{math.sqrt(_p * (1 - _p) * (1 / _dl - 1 / _db)) * 100:.1f}")
        _mac(fh, "FigBlLevenePl", f"{pl:.1g}")
        _mac(fh, "FigBlAG", f"{Fw:.1f}")
        _mac(fh, "FigBlAGp", f"{pw:.0e}".replace("e-", "\\times10^{-").replace("+", "") + "}")
        for a in ORDER:
            v, tag = data[a], _tag(a)
            _mac(fh, f"FigBlN{tag}", str(len(v)))
            _mac(fh, f"FigBlMean{tag}", f"{v.mean():.4f}")
            _mac(fh, f"FigBlSd{tag}", f"{v.std(ddof=1):.4f}")
        for a, (thr, f, wlo, whi) in SPLIT.items():
            v, tag = data[a], _tag(a)
            _mac(fh, f"FigBlFail{tag}", f"{f}/{len(v)}")
            _mac(fh, f"FigBlFailPct{tag}", f"{f / len(v) * 100:.0f}")
            _mac(fh, f"FigBlWilson{tag}", f"{wlo * 100:.0f}--{whi * 100:.0f}")
            _mac(fh, f"FigBlHealthy{tag}", f"{v[v >= thr].mean():.4f}")
            _mac(fh, f"FigBlHealthySd{tag}", f"{v[v >= thr].std(ddof=1):.4f}")
            _mac(fh, f"FigBlCollapsed{tag}", f"{v[v < thr].mean():.4f}")
        # The lower-rate arm is the direct test of whether the collapses at
        # eta=0.003 were a step-size effect and whether the LEVEL moved with it.
        # Both need saying, and they have different answers.
        if "lockin_B" in SPLIT and "lockin_B_lr001" in data:
            thr = SPLIT["lockin_B"][0]
            lo_arm = data["lockin_B_lr001"]
            f, n = int((lo_arm < thr).sum()), len(lo_arm)
            z = 1.96
            cc = (f / n + z**2 / (2 * n)) / (1 + z**2 / n)
            hh = z * math.sqrt((f / n) * (1 - f / n) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
            _mac(fh, "FigBlFailLockinlo", f"{f}/{n}")
            _mac(fh, "FigBlWilsonLockinlo",
                 f"{max(0.0, cc - hh) * 100:.0f}--{min(1.0, cc + hh) * 100:.0f}")
            healthy = data["lockin_B"][data["lockin_B"] >= thr]
            dd = lo_arm.mean() - healthy.mean()
            tt, pp2 = stats.ttest_ind(lo_arm, healthy, equal_var=False)
            _mac(fh, "FigBlLoVsHealthy", f"{dd:+.4f}")
            _mac(fh, "FigBlLoVsHealthyP", f"{pp2:.2f}")

        for a, c, d, lo, hi, pr, pa, pt in CONTRASTS:
            tag = _tag(a) + _tag(c)
            _mac(fh, f"FigBlDiff{tag}", f"{d:+.4f}")
            _mac(fh, f"FigBlCI{tag}", f"[{lo:+.4f}, {hi:+.4f}]")
    print(f"\nwrote fig5_runs.dat, fig5_contrasts.dat, fig5_macros.tex -> {OUT}")
