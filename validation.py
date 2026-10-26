

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

BENCH_RNG_SEED   = 12345
BENCH_N_CORES    = 8           # synthetic cores per configuration
BENCH_CORE_CM    = 400.0
BENCH_STEP_CM    = 0.5
BENCH_LADDER_CM  = [5.0, 15.0, 45.0, 135.0]
BENCH_DELTA_TOL  = 2.0
BENCH_N_MIN      = 5
BENCH_MIN_FAM    = 2
BENCH_MAD_FLOOR  = 0.01
BENCH_E_FALSE    = 1.0          # expected false flags per core, for the corrected tau

# two families of two columns each, as in a typical admitted core
BENCH_FAMILIES   = {"grain": ["grain_a", "grain_b"],
              "physical": ["phys_a", "phys_b"]}


# ------------------------------------------------------------------ synthetic
def bench_ar1(n, rho, sd, rng):
    """Autocorrelated background, normalised to standard deviation `sd`.

    Real sediment properties are NOT independent; white noise would flatter the
    detector badly. The sqrt(1 - rho^2) factor is essential: without it an
    AR(1) process with rho = 0.85 has variance 1/(1 - rho^2) = 3.6 times the
    innovation variance, so a layer described as "4 sd" would in fact be about
    2.5 sd and every amplitude on the detection-limit curve would be mislabelled.
    """
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + rng.normal(0, 1.0)
    return x * sd * np.sqrt(max(1e-9, 1.0 - rho ** 2))


def bench_make_core(rng, n_events=6, thickness_cm=(2, 30), amp_sd=4.0,
              sharpness=1.0, rho=0.85):
    """One synthetic core with planted event layers.

    sharpness: 1.0 = step contacts; larger = more gradational (contact smeared
    over sharpness * thickness / 4).
    """
    n = int(BENCH_CORE_CM / BENCH_STEP_CM)
    depth = np.arange(n) * BENCH_STEP_CM
    cols = {}
    for fam, cs in BENCH_FAMILIES.items():
        base = bench_ar1(n, rho, 1.0, rng)
        for c in cs:
            cols[c] = base * 0.8 + bench_ar1(n, rho, 1.0, rng) * 0.2

    truth = []
    margin = 30.0
    for _ in range(n_events):
        t = rng.uniform(*thickness_cm)
        top = rng.uniform(margin, BENCH_CORE_CM - margin - t)
        bot = top + t
        truth.append((top, bot))
        centre = (depth >= top) & (depth <= bot)
        if sharpness <= 1.0:
            shape = centre.astype(float)
        else:                      # gradational: taper the contacts
            edge = max(BENCH_STEP_CM, sharpness * t / 4)
            shape = np.clip(np.minimum((depth - top) / edge,
                                       (bot - depth) / edge), 0, 1)
            shape[~((depth >= top - edge) & (depth <= bot + edge))] = 0
        for c in cols:
            sign = 1 if c.startswith("grain") else -1
            cols[c] = cols[c] + sign * amp_sd * shape
    df = pd.DataFrame({"depth": depth, **cols})
    return df, sorted(truth)


# ------------------------------------------------------------------- detector
def bench_robust_z(v, n_win):
    s = pd.Series(v, dtype=float)
    minp = max(BENCH_N_MIN, n_win // 2)
    med = s.rolling(n_win, center=True, min_periods=minp).median()
    r = s - med
    mad = r.abs().rolling(n_win, center=True, min_periods=minp).median()
    g = float((s - s.median()).abs().median())
    floor = BENCH_MAD_FLOOR * g if g > 0 else np.inf
    ok = mad >= floor
    z = pd.Series(np.nan, index=s.index)
    z[ok] = r[ok] / (1.4826 * mad[ok])
    return z


def bench_effective_tests(df):
    """Non-overlapping windows per column per resolvable scale."""
    n_eff = 0
    for c in df.columns[1:]:
        nv = int(df[c].notna().sum())
        for w in BENCH_LADDER_CM:
            nw = int(round(w / BENCH_STEP_CM))
            nw += (nw % 2 == 0)
            if nw < BENCH_N_MIN or nw > nv:
                continue
            n_eff += max(1, nv / nw)
    return n_eff


def bench_corrected_tau(df, e_false=BENCH_E_FALSE):
    n_eff = bench_effective_tests(df)
    return float(norm.ppf(1 - e_false / (2 * n_eff))), n_eff


def bench_estimate_rho(v):
    """Lag-1 autocorrelation of the background. Events are a minority of a core,
    so the bulk statistic is dominated by background structure."""
    x = pd.Series(v).dropna().values
    x = x - np.median(x)
    if len(x) < 20:
        return 0.0
    r = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    return float(np.clip(r, 0.0, 0.98))


def bench_events_at(df, zmaps, tau):
    """Flags -> family agreement -> merged intervals, for a precomputed
    z-map. Used by the calibration so the null is evaluated through the WHOLE
    pipeline: what matters is the rate of spurious EVENTS, not of flags."""
    rows = []
    for (fam, c, w), z in zmaps.items():
        hit = np.abs(np.nan_to_num(z, nan=0.0)) > tau
        if hit.any():
            rows.append(pd.DataFrame({"depth": df.depth.values[hit],
                                      "family": fam}))
    if not rows:
        return 0
    fl = pd.concat(rows, ignore_index=True)
    cand = [d for d in np.sort(fl.depth.unique())
            if fl[(fl.depth - d).abs() <= BENCH_DELTA_TOL].family.nunique() >= BENCH_MIN_FAM]
    if not cand:
        return 0
    cand = np.sort(np.array(cand))
    n = 1
    for a, b in zip(cand[:-1], cand[1:]):
        if b - a > 2 * BENCH_DELTA_TOL:
            n += 1
    return n


def bench_zmap(df):
    out = {}
    for fam, cs in BENCH_FAMILIES.items():
        for c in cs:
            v = df[c].values
            for w in BENCH_LADDER_CM:
                nw = int(round(w / BENCH_STEP_CM)); nw += (nw % 2 == 0)
                if nw < BENCH_N_MIN or nw > len(v):
                    continue
                out[(fam, c, w)] = bench_robust_z(v, nw)
    return out


def bench_surrogate_tau(df, e_false=BENCH_E_FALSE, n_surr=6, rng=None):
    rng = rng or np.random.default_rng(0)
    n = len(df)
    rhos = {c: bench_estimate_rho(df[c].values)
            for cs in BENCH_FAMILIES.values() for c in cs}
    sets = []
    for _ in range(n_surr):
        sur = pd.DataFrame({"depth": df.depth.values})
        for cs in BENCH_FAMILIES.values():
            for c in cs:
                sur[c] = bench_ar1(n, rhos[c], 1.0, rng)
        sets.append((sur, bench_zmap(sur)))
    for tau in np.arange(3.0, 9.01, 0.25):
        m = float(np.mean([bench_events_at(sd, zm, tau) for sd, zm in sets]))
        if m <= e_false:
            return float(tau)
    return 9.0


def bench_detect(df, tau):
    """tau may be a scalar or a dict keyed by (column, scale)."""
    flags = []
    for fam, cs in BENCH_FAMILIES.items():
        for c in cs:
            v = df[c].values
            for w in BENCH_LADDER_CM:
                nw = int(round(w / BENCH_STEP_CM)); nw += (nw % 2 == 0)
                if nw < BENCH_N_MIN or nw > len(v):
                    continue
                t = tau[(c, w)] if isinstance(tau, dict) else tau
                z = bench_robust_z(v, nw)
                hit = np.abs(z.values) > t
                for d in df.depth.values[hit]:
                    flags.append((float(d), fam))
    if not flags:
        return []
    fl = pd.DataFrame(flags, columns=["depth", "family"])
    cand = [d for d in np.sort(fl.depth.unique())
            if fl[(fl.depth - d).abs() <= BENCH_DELTA_TOL].family.nunique() >= BENCH_MIN_FAM]
    if not cand:
        return []
    cand = np.sort(np.array(cand))
    groups, cur = [], [cand[0]]
    for a, b in zip(cand[:-1], cand[1:]):
        if b - a <= 2 * BENCH_DELTA_TOL:
            cur.append(b)
        else:
            groups.append(cur); cur = [b]
    groups.append(cur)
    return [(float(min(g)), float(max(g))) for g in groups]


# ----------------------------------------------------------------- evaluation
def bench_overlaps(a, b):
    return not (a[1] < b[0] or b[1] < a[0])


def bench_score(detected, truth, pad=2.0):
    """An event is recalled if any detection bench_overlaps it (padded); a detection
    is a true positive if it bench_overlaps any true event."""
    t_pad = [(a - pad, b + pad) for a, b in truth]
    tp = sum(any(bench_overlaps(d, t) for t in t_pad) for d in detected)
    fp = len(detected) - tp
    rec = sum(any(bench_overlaps(d, t) for d in detected) for t in t_pad)
    return tp, fp, rec, len(truth)


def bench_coverage(detected):
    return sum(b - a for a, b in detected) / BENCH_CORE_CM


def bench_run(label, tau_mode, **kw):
    rng = np.random.default_rng(BENCH_RNG_SEED)
    TP = FP = REC = TOT = 0
    covs, taus = [], []
    for _ in range(BENCH_N_CORES):
        df, truth = bench_make_core(rng, **kw)
        if tau_mode == "fixed":
            tau = 3.0
        elif tau_mode == "corrected":
            tau = bench_corrected_tau(df)[0]
        else:
            tau = bench_surrogate_tau(df, rng=rng)
        det = bench_detect(df, tau)
        tp, fp, rec, tot = bench_score(det, truth)
        TP += tp; FP += fp; REC += rec; TOT += tot
        covs.append(bench_coverage(det)); taus.append(tau)
    prec = TP / (TP + FP) if TP + FP else float("nan")
    recall = REC / TOT
    f1 = 2 * prec * recall / (prec + recall) if prec + recall else float("nan")
    return dict(label=label, tau=float(np.mean(taus)), precision=prec,
                recall=recall,
                f1=f1, bench_coverage=np.mean(covs), tp=TP, fp=FP)


def run_synthetic_benchmark():
    print(f"synthetic benchmark: {BENCH_N_CORES} cores per row, {BENCH_CORE_CM:.0f} cm at "
          f"{BENCH_STEP_CM} cm, AR(1) background rho=0.85\n")

    rows = []
    for mode, name in [("fixed", "tau = 3.0 (fixed)"),
                       ("corrected", "tau multiplicity-corrected"),
                       ("surrogate", "tau surrogate-calibrated")]:
        rows.append(bench_run(name, mode))
    r = pd.DataFrame(rows)
    print("=== EFFECT OF THE MULTIPLICITY CORRECTION ===")
    print(r[["label", "tau", "precision", "recall", "f1", "bench_coverage",
             "tp", "fp"]].to_string(index=False,
                                    float_format=lambda x: f"{x:.3f}"))

    print("\n=== DETECTION LIMITS (corrected tau) ===")
    lim = []
    for amp in [3.0, 4.0, 6.0, 8.0, 12.0, 20.0]:
        lim.append(bench_run(f"amplitude {amp} sd", "surrogate", amp_sd=amp))
    for th in [(2, 4), (4, 8), (8, 16), (16, 32)]:
        lim.append(bench_run(f"thickness {th[0]}-{th[1]} cm", "surrogate",
                       thickness_cm=th))
    for sh in [1.0, 2.0, 4.0]:
        lim.append(bench_run(f"sharpness {sh} (1=step)", "surrogate", sharpness=sh))
    L = pd.DataFrame(lim)
    print(L[["label", "precision", "recall", "f1", "bench_coverage"]].to_string(
        index=False, float_format=lambda x: f"{x:.3f}"))

    pd.concat([r, L]).to_csv("ev_benchmark.csv", index=False)
    print("\nwritten: ev_benchmark.csv")


import glob
import os

import numpy as np
import pandas as pd

MONDSEE_CORE_GLOB  = "ev_clean/*818922*_proxies.csv"
MONDSEE_LAYER_GLOB = "ev_cores/*818922*/*.csv"
MONDSEE_PROXY      = "geochem_Ca_Ti"

MONDSEE_LADDER_CM  = [5.0, 15.0, 45.0, 135.0]     # unchanged
MONDSEE_N_MIN      = 5
MONDSEE_MAD_FLOOR  = 0.01
MONDSEE_DELTA_TOL  = 2.0
MONDSEE_TOL_CM     = 2.0                          # hit criterion
MONDSEE_TAU_GRID   = np.arange(3.0, 9.01, 0.25)
MONDSEE_E_EVENTS   = 1.0                          # expected false events per core
MONDSEE_N_SURR     = 6
MONDSEE_N_PERM     = 5000
MONDSEE_RNG        = np.random.default_rng(0)


# ------------------------------------------------------------------ scoring
def mondsee_robust_z(v, n_win):
    s = pd.Series(v, dtype=float)
    mp = max(MONDSEE_N_MIN, n_win // 2)
    med = s.rolling(n_win, center=True, min_periods=mp).median()
    r = s - med
    mad = r.abs().rolling(n_win, center=True, min_periods=mp).median()
    g = float((s - s.median()).abs().median())
    floor = MONDSEE_MAD_FLOOR * g if g > 0 else np.inf
    ok = mad >= floor
    z = pd.Series(np.nan, index=s.index)
    z[ok] = r[ok] / (1.4826 * mad[ok])
    return z.values


def mondsee_ar1(n, rho, rng):
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + rng.normal(0, 1)
    return x * np.sqrt(max(1e-9, 1 - rho ** 2))


def mondsee_estimate_rho(v):
    x = pd.Series(v).dropna().values
    x = x - np.median(x)
    r = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    return 0.0 if not np.isfinite(r) else float(np.clip(r, 0.0, 0.98))


def mondsee_windows(dstep, n, length):
    out = []
    for w in MONDSEE_LADDER_CM:
        nw = int(round(w / dstep)); nw += (nw % 2 == 0)
        if nw < MONDSEE_N_MIN or nw > n or w > length:
            continue
        out.append((w, nw))
    return out


def mondsee_events_from(depth, zs, tau):
    """Flagged depths -> merged intervals. No family rule: one proxy."""
    flagged = np.zeros(len(depth), bool)
    for z in zs.values():
        flagged |= np.abs(np.nan_to_num(z, nan=0.0)) > tau
    if not flagged.any():
        return []
    d = depth[flagged]
    groups, cur = [], [d[0]]
    for a, b in zip(d[:-1], d[1:]):
        if b - a <= 2 * MONDSEE_DELTA_TOL:
            cur.append(b)
        else:
            groups.append(cur); cur = [b]
    groups.append(cur)
    return [(float(min(g)), float(max(g))) for g in groups]


def mondsee_calibrate(depth, v, wins, rng):
    rho = mondsee_estimate_rho(v)
    sets = []
    for _ in range(MONDSEE_N_SURR):
        sur = mondsee_ar1(len(v), rho, rng)
        sets.append({w: mondsee_robust_z(sur, nw) for w, nw in wins})
    for tau in MONDSEE_TAU_GRID:
        m = float(np.mean([len(mondsee_events_from(depth, z, tau)) for z in sets]))
        if m <= MONDSEE_E_EVENTS:
            return float(tau), m
    return float(MONDSEE_TAU_GRID[-1]), m


# ------------------------------------------------------------- ground truth
def mondsee_load_layers():
    best, best_n = None, -1
    for f in glob.glob(MONDSEE_LAYER_GLOB):
        try:
            h = pd.read_csv(f, nrows=5)
        except Exception:
            continue
        c = " ".join(str(x).lower() for x in h.columns)
        if "thick" in c and ("lay" in c or "layer" in c):
            n = len(pd.read_csv(f))
            if n > best_n:
                best, best_n = f, n
    if best is None:
        raise SystemExit("layer-thickness file not found")
    df = pd.read_csv(best)
    dcol = next(c for c in df.columns if "depth" in str(c).lower())
    depth = pd.to_numeric(df[dcol], errors="coerce")
    if depth.max() <= 30:
        depth = depth * 100

    def col(*w):
        for c in df.columns:
            cl = str(c).lower()
            if all(x in cl for x in w):
                return c
        return None

    rows = []
    for lab, c in [("flood", col("flood", "thick")),
                   ("debris_flow", col("debris", "thick"))]:
        if c is None:
            continue
        t = pd.to_numeric(df[c], errors="coerce")
        m = t.notna() & (t > 0) & depth.notna()
        for d, th in zip(depth[m], t[m]):
            rows.append({"kind": lab, "depth": float(d),
                         "thickness_raw": float(th)})
    return os.path.basename(best), pd.DataFrame(rows).sort_values("depth")


# ------------------------------------------------------------- mondsee_permutation
def mondsee_gaps(targets, iv):
    t = np.asarray(targets, float)[:, None]
    a = np.asarray([x for x, _ in iv], float)[None, :]
    b = np.asarray([y for _, y in iv], float)[None, :]
    return np.where((t >= a) & (t <= b), 0.0,
                    np.minimum(np.abs(t - a), np.abs(t - b))).min(axis=1)


def mondsee_permutation(targets, iv, lo, hi, tol, n=MONDSEE_N_PERM):
    t = np.asarray(targets, float)[:, None]
    w = np.asarray([b - a for a, b in iv], float)
    obs = int((mondsee_gaps(targets, iv) <= tol).sum())
    hits = np.empty(n, int)
    for k in range(n):
        s = MONDSEE_RNG.uniform(lo, hi, size=len(w))
        lo_e = (s - tol)[None, :]
        hi_e = (np.minimum(s + w, hi) + tol)[None, :]
        hits[k] = int(((t >= lo_e) & (t <= hi_e)).any(axis=1).sum())
    return obs, hits.mean(), np.percentile(hits, 95), float((hits >= obs).mean())


# ---------------------------------------------------------------------- main
def run_mondsee_single_proxy():
    f = glob.glob(MONDSEE_CORE_GLOB)
    if not f:
        raise SystemExit(f"core not found: {MONDSEE_CORE_GLOB}")
    df = pd.read_csv(f[0])
    if MONDSEE_PROXY not in df.columns:
        raise SystemExit(f"{MONDSEE_PROXY} not in {os.path.basename(f[0])}")

    sub = df.loc[df[MONDSEE_PROXY].notna(), ["depth", MONDSEE_PROXY]].sort_values("depth")
    depth = sub.depth.values
    v = sub[MONDSEE_PROXY].values
    dstep = float(np.median(np.diff(depth)))
    length = depth.max() - depth.min()
    print(f"core   : {os.path.basename(f[0])}")
    print(f"proxy  : {MONDSEE_PROXY}   n={len(v)}   "
          f"{depth.min():.1f}-{depth.max():.1f} cm   step={dstep:.3f} cm")

    wins = mondsee_windows(dstep, len(v), length)
    print(f"scales : {[int(w) for w, _ in wins]} cm "
          f"(= {[nw for _, nw in wins]} samples)")

    lf, lay = mondsee_load_layers()
    lay = lay[(lay.depth >= depth.min()) & (lay.depth <= depth.max())]
    print(f"layers : {lf}")
    print(f"         {len(lay)} published in this interval "
          f"({', '.join(f'{k}={v_}' for k, v_ in lay.kind.value_counts().items())})")

    rng = np.random.default_rng(7)
    tau, exp_false = mondsee_calibrate(depth, v, wins, rng)
    print(f"tau    : {tau:.2f}  (surrogate-calibrated, "
          f"{exp_false:.2f} expected false events)\n")

    zs = {w: mondsee_robust_z(v, nw) for w, nw in wins}
    iv = mondsee_events_from(depth, zs, tau)
    lo, hi = float(depth.min()), float(depth.max())
    cov = sum(b - a for a, b in iv) / (hi - lo)
    print(f"detected events : {len(iv)}   coverage {100*cov:.2f}%")
    if not iv:
        raise SystemExit("no events detected")

    print("\n=== RESULT ===")
    obs, mean, p95, p = mondsee_permutation(lay.depth.values, iv, lo, hi, MONDSEE_TOL_CM)
    rec = obs / len(lay)
    mids = np.array([(a + b) / 2 for a, b in iv])
    prec = float((mondsee_gaps(mids, [(d, d) for d in lay.depth.values])
                  <= MONDSEE_TOL_CM).mean())
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else float("nan")
    print(f"  recall    : {obs}/{len(lay)} = {rec:.3f}")
    print(f"  precision : {prec:.3f}")
    print(f"  F1        : {f1:.3f}")
    print(f"  chance    : {mean:.1f} ({100*mean/len(lay):.1f}%), 95th pct {p95:.0f}")
    print(f"  p-value   : {p:.5f}  -> "
          f"{'ABOVE CHANCE' if p < 0.05 else 'not distinguishable from chance'}")

    print("\n=== BY LAYER TYPE ===")
    for k, g in lay.groupby("kind"):
        if len(g) < 5:
            continue
        o, m, _, pp = mondsee_permutation(g.depth.values, iv, lo, hi, MONDSEE_TOL_CM, n=3000)
        print(f"  {k:12s}: {o}/{len(g)} = {o/len(g):.3f}   "
              f"chance {m/len(g):.3f}   p={pp:.4f}")

    print("\n=== DETECTION LIMIT (recall vs published layer thickness) ===")
    qs = [0, .25, .5, .75, .9, 1.0]
    e = lay.thickness_raw.quantile(qs).values
    print(f"{'band (archive units)':>26} {'n':>5} {'recall':>8} {'chance':>8}")
    for i in range(len(e) - 1):
        b = lay[(lay.thickness_raw >= e[i]) & (lay.thickness_raw <= e[i + 1])]
        if len(b) < 5:
            continue
        o, m, _, _ = mondsee_permutation(b.depth.values, iv, lo, hi, MONDSEE_TOL_CM, n=2500)
        print(f"{e[i]:11.2f}-{e[i+1]:<13.2f} {len(b):5d} "
              f"{o/len(b):8.3f} {m/len(b):8.3f}")

    g = mondsee_gaps(lay.depth.values, iv)
    out = lay.copy(); out["gap_cm"] = g; out["hit"] = g <= MONDSEE_TOL_CM
    out.to_csv("ev_validation_mondsee_singleproxy.csv", index=False)
    pd.DataFrame(iv, columns=["depth_top", "depth_bottom"]).assign(
        tau=tau).to_csv("ev_events_mondsee_singleproxy.csv", index=False)
    print("\nwritten: ev_validation_mondsee_singleproxy.csv, "
          "ev_events_mondsee_singleproxy.csv")


#!/usr/bin/env python3
"""
ev11_validate_published.py -- validation against published layer descriptions
for the two cores whose ground truth is numeric depths in the paper text.

  Lake Lucerne, core LU18-2  (Nigg et al. 2021, G-cubed 22, e2021GC009753)
      1601 CE tsunami deposit, Unit 3, 37.5-101.5 cm, 0.64 m thick.
      Two contacts to test.

  Laguna Meseta, core LME-CP (Franco et al. 2024, Clim. Past 20, 817-839)
      Nine lithological units, eight numeric boundaries.
      Five of them have their character (sharp / gradational) stated in the
      publication, which permits a test of WHICH boundaries are recovered.

EVERY hit rate is reported against a PERMUTATION BASELINE: the same detected
intervals, preserving their number and widths, placed at random depths in the
same core. Without it a hit count is uninterpretable -- Meseta's apparent 4/8
at tau = 3 came with a chance expectation of 3.99 (p = 0.65), i.e. nothing.

IN : ev_events_strict.csv (from ev5_explore.py)
RUN: python3 ev11_validate_published.py
"""
import sys

import numpy as np
import pandas as pd

PUB_EVENTS  = "PRL_final_results.csv"
PUB_N_PERM  = 20000
PUB_RNG     = np.random.default_rng(0)

# ---- ground truth, transcribed from the publications ------------------------
PUB_LUCERNE = {
    "key": "938716",
    "name": "Lake Lucerne LU18-2",
    "ref": "Nigg et al. 2021, G-cubed 22, e2021GC009753",
    "lo": 0.0, "hi": 287.5,
    "targets": {101.5: "Unit 3 basal contact (sharp)",
                37.5:  "Unit 3 top contact"},
    "character": {},
}

PUB_MESETA = {
    "key": "961940",
    "name": "Laguna Meseta LME-CP",
    "ref": "Franco et al. 2024, Clim. Past 20, 817-839",
    "lo": 0.0, "hi": 292.5,
    "targets": {24.1: "Unit A/B", 78.6: "Unit B/C", 139.7: "Unit C/D",
                184.0: "Unit D/E", 199.6: "Unit E/F", 214.5: "Unit F/T5",
                229.0: "T5 tephra base", 240.5: "Unit G/H"},
    # character AS STATED IN THE PUBLICATION -- never assigned by us after
    # seeing which boundaries were recovered
    "character": {24.1: "sharp",        # MS and sand drop synchronously
                  229.0: "sharp",       # sharp lower tephra boundary
                  139.7: "gradational",  # transitional unit
                  199.6: "gradational",  # upper boundary undeterminable
                  214.5: "gradational"},  # tephra grades into Unit E
}


def pub_gaps(targets, iv):
    t = np.asarray(targets, float)[:, None]
    a = np.asarray([x for x, _ in iv], float)[None, :]
    b = np.asarray([y for _, y in iv], float)[None, :]
    return np.where((t >= a) & (t <= b), 0.0,
                    np.minimum(np.abs(t - a), np.abs(t - b))).min(axis=1)


def pub_permutation(targets, iv, lo, hi, tol, n=PUB_N_PERM):
    t = np.asarray(targets, float)[:, None]
    w = np.asarray([b - a for a, b in iv], float)
    obs = int((pub_gaps(targets, iv) <= tol).sum())
    hits = np.empty(n, int)
    for k in range(n):
        s = PUB_RNG.uniform(lo, hi, size=len(w))
        lo_e = (s - tol)[None, :]
        hi_e = (np.minimum(s + w, hi) + tol)[None, :]
        hits[k] = int(((t >= lo_e) & (t <= hi_e)).any(axis=1).sum())
    return obs, float(hits.mean()), float((hits >= obs).mean())


def pub_report(spec, ev):
    sub = ev[ev.core.str.startswith(spec["key"])]
    print("=" * 68)
    print(f"{spec['name']}   ({spec['ref']})")
    if not len(sub):
        print("  no events for this core -- excluded, or none detected\n")
        return
    iv = list(zip(sub.depth_top, sub.depth_bottom))
    tau = sub.tau.iloc[0] if "tau" in sub.columns else float("nan")
    cov = sum(b - a for a, b in iv) / (spec["hi"] - spec["lo"])
    print(f"  {len(iv)} events, tau={tau}, coverage {100*cov:.1f}% of core")
    print()

    tg = sorted(spec["targets"])
    g = pub_gaps(tg, iv)
    print(f"  {'depth':>7} {'gap':>7}  {'':<5} description")
    for d, gg in zip(tg, g):
        mark = "HIT" if gg <= 5.0 else "miss"
        print(f"  {d:7.1f} {gg:7.1f}  {mark:<5} {spec['targets'][d]}")

    print()
    for tol, lab in [(5.0, "within 5 cm"), (0.0, "containment")]:
        obs, ch, p = pub_permutation(tg, iv, spec["lo"], spec["hi"], tol)
        star = " *" if p < 0.05 else ""
        print(f"  [{lab:12s}] {obs}/{len(tg)} observed, {ch:.2f} expected by "
              f"chance, p = {p:.4f}{star}")

    # sharp vs gradational, where the publication states it
    ch_map = spec["character"]
    if ch_map:
        try:
            from scipy.stats import fisher_exact
        except ImportError:
            print("\n  (scipy not installed -- skipping Fisher test)")
            return
        hit = {d: gg <= 5.0 for d, gg in zip(tg, g)}
        sh = [hit[d] for d in ch_map if ch_map[d] == "sharp"]
        gr = [hit[d] for d in ch_map if ch_map[d] == "gradational"]
        if sh and gr:
            _, p = fisher_exact([[sum(sh), len(sh) - sum(sh)],
                                 [sum(gr), len(gr) - sum(gr)]],
                                alternative="greater")
            print(f"\n  boundary character (publication-stated only, "
                  f"n={len(sh)+len(gr)}):")
            print(f"    sharp       {sum(sh)}/{len(sh)} recovered")
            print(f"    gradational {sum(gr)}/{len(gr)} recovered")
            print(f"    Fisher exact, one-tailed: p = {p:.3f}")
    print()


def run_published_validation():
    try:
        ev = pd.read_csv(PUB_EVENTS)
    except FileNotFoundError:
        sys.exit(f"{PUB_EVENTS} not found -- run event_detector.py first")

    # The refactored detector writes one consolidated table containing both
    # strict and relaxed results. Published-record validation uses the strict
    # event intervals only, matching the original validation intent.
    if "record_type" in ev.columns:
        ev = ev[ev["record_type"].astype(str).str.lower().eq("event")].copy()
    if "tier" in ev.columns:
        ev = ev[ev["tier"].astype(str).str.lower().eq("strict")].copy()

    required = {"core", "depth_top", "depth_bottom"}
    missing = required.difference(ev.columns)
    if missing:
        sys.exit(
            "PRL_final_results.csv is missing required columns: "
            + ", ".join(sorted(missing))
        )

    # Keep only the fields used by pub_report; preserve tau when available.
    keep = [c for c in ["core", "tier", "tau", "depth_top", "depth_bottom"] if c in ev.columns]
    ev = ev[keep].copy()
    ev = ev.rename(columns={"depth_top": "depth_top", "depth_bottom": "depth_bottom"})

    for spec in (PUB_LUCERNE, PUB_MESETA):
        pub_report(spec, ev)
    print("Hit counts are meaningless without the chance column: a detector")
    print("flagging a large fraction of a core will hit any target by luck.")


import argparse
import glob
import os
from itertools import combinations

import numpy as np
import pandas as pd

# ---- must match ev5_explore.py --------------------------------------
IND_STEP_DEFAULT = 0.5
IND_LADDER = [5.0, 15.0, 45.0, 135.0]
IND_NMIN = 5
IND_DTOL = 2.0
IND_MADFLOOR = 0.01
IND_RHO_BG = 0.85
IND_TAU_GRID = np.arange(3.0, 9.01, 0.25)
IND_E_TARGET = 1.0

IND_FAMILY_PATTERNS = [
    ("organic",  ["organic", "loi", "toc", "tc_", "tn_", "corg"]),
    ("grain",    ["grain", "sand", "silt", "clay", "mean_um", "d50", "frac"]),
    ("geochem",  ["geochem", "_ratio", "ca_ti", "fe_ti", "mn_fe", "k_ti",
                  "zr_rb", "si_ti", "caco3", "tic", "sio2", "_ts"]),
    ("physical", ["density", "water", "porosity", "mag_susc", "mags",
                  "suscept", "kappa", "shear"]),
    ("biogenic", ["biogenic", "bsio2", "bsi", "diatom"]),
    ("colour",   ["lstar", "lightness", "reflect", "grey", "gray"]),
]


def ind_family_of(col):
    head = str(col).split("_", 1)[0].lower()
    if head in {"organic", "grain", "geochem", "physical", "biogenic", "colour"}:
        return head
    c = str(col).lower()
    for fam, pats in IND_FAMILY_PATTERNS:
        if any(p in c for p in pats):
            return fam
    return None


# ---- robust score, shared with the detector -------------------------
def ind_robust_z(v, nw):
    s = pd.Series(v, dtype=float)
    mp = max(IND_NMIN, nw // 2)
    med = s.rolling(nw, center=True, min_periods=mp).median()
    r = s - med
    mad = r.abs().rolling(nw, center=True, min_periods=mp).median()
    g = float((s - s.median()).abs().median())
    fl = IND_MADFLOOR * g if g > 0 else np.inf
    ok = mad >= fl
    z = np.full(len(s), np.nan)
    z[ok.values] = (r[ok] / (1.4826 * mad[ok])).values
    return z


# =====================================================================
#  PART A -- synthetic: how much correlation would matter
# =====================================================================
IND_FAM_SYN = {"grain": ["grain_a", "grain_b"], "physical": ["phys_a", "phys_b"]}


def ind_ar1(n, rho, rng):
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + rng.normal(0, 1)
    return x * np.sqrt(max(1e-9, 1 - rho ** 2))


def ind_make_null(rng, r_cross, core_cm=400.0, step=0.5):
    n = int(core_cm / step)
    depth = np.arange(n) * step
    g = ind_ar1(n, IND_RHO_BG, rng)
    p = ind_ar1(n, IND_RHO_BG, rng)
    p = r_cross * g + np.sqrt(max(1e-9, 1 - r_cross ** 2)) * p
    cols = {"depth": depth}
    for c in IND_FAM_SYN["grain"]:
        cols[c] = g * 0.8 + ind_ar1(n, IND_RHO_BG, rng) * 0.2
    for c in IND_FAM_SYN["physical"]:
        cols[c] = p * 0.8 + ind_ar1(n, IND_RHO_BG, rng) * 0.2
    return pd.DataFrame(cols), step


def ind_zmap(df, step, fams):
    out = {}
    for fam, cs in fams.items():
        for c in cs:
            if c not in df:
                continue
            v = df[c].values
            for w in IND_LADDER:
                nw = int(round(w / step)); nw += (nw % 2 == 0)
                if IND_NMIN <= nw <= len(v):
                    out[(fam, c, w)] = ind_robust_z(v, nw)
    return out


def ind_n_events(df, zm, tau):
    rows = []
    for (fam, c, w), z in zm.items():
        hit = np.abs(np.nan_to_num(z, nan=0.0)) > tau
        if hit.any():
            rows.append(pd.DataFrame({"depth": df.depth.values[hit],
                                      "family": fam}))
    if not rows:
        return 0
    fl = pd.concat(rows, ignore_index=True)
    cand = [d for d in np.sort(fl.depth.unique())
            if fl[(fl.depth - d).abs() <= IND_DTOL].family.nunique() >= 2]
    if not cand:
        return 0
    cand = np.sort(np.array(cand))
    n = 1
    for a, b in zip(cand[:-1], cand[1:]):
        if b - a > 2 * IND_DTOL:
            n += 1
    return n


def ind_calibrate_percolumn(df, step, rng, nsurr=6):
    rhos = {}
    for cs in IND_FAM_SYN.values():
        for c in cs:
            x = df[c].values - np.median(df[c].values)
            rhos[c] = float(np.clip(np.corrcoef(x[:-1], x[1:])[0, 1], 0, 0.98))
    sets = []
    for _ in range(nsurr):
        sur = pd.DataFrame({"depth": df.depth.values})
        for cs in IND_FAM_SYN.values():
            for c in cs:
                sur[c] = ind_ar1(len(df), rhos[c], rng)
        sets.append((sur, ind_zmap(sur, step, IND_FAM_SYN)))
    for tau in IND_TAU_GRID:
        if np.mean([ind_n_events(s, z, tau) for s, z in sets]) <= IND_E_TARGET:
            return float(tau)
    return 9.0


def ind_part_a(seed=4, n_eval=15):
    rng = np.random.default_rng(seed)
    print("PART A -- how much inter-family correlation would matter")
    print("Event-free synthetic cores; realised false events after per-column")
    print(f"calibration (target <= {IND_E_TARGET}).\n")
    print(f"  {'r_cross':>8} {'tau':>6} {'false_events':>13}")
    for r in [0.0, 0.3, 0.5, 0.7, 0.9]:
        cal, step = ind_make_null(rng, r)
        tau = ind_calibrate_percolumn(cal, step, rng)
        fp = []
        for _ in range(n_eval):
            c, _ = ind_make_null(rng, r)
            fp.append(ind_n_events(c, ind_zmap(c, step, IND_FAM_SYN), tau))
        print(f"  {r:8.1f} {tau:6.2f} {np.mean(fp):13.2f}")
    print()


# =====================================================================
#  PART B -- real cores: how much correlation is present
# =====================================================================
def ind_family_series(df, window_cm=15.0):
    depth = df[df.columns[0]].astype(float).values
    step = float(np.median(np.diff(np.sort(depth))))
    if not np.isfinite(step) or step <= 0:
        step = IND_STEP_DEFAULT
    nw = int(round(window_cm / step)); nw += (nw % 2 == 0)
    nw = max(nw, IND_NMIN)
    fam_z = {}
    for col in df.columns[1:]:
        fam = ind_family_of(col)
        if fam is None or df[col].notna().sum() < 20:
            continue
        # rolling z on the column's OWN non-null grid, then placed back on the
        # full depth index -- so a family series is defined only where that
        # family was actually measured (matches the verified inline analysis)
        sub = df.loc[df[col].notna(), [df.columns[0], col]]
        z_sub = ind_robust_z(sub[col].values, nw)
        z_full = pd.Series(np.nan, index=df.index)
        z_full.loc[sub.index] = z_sub
        fam_z.setdefault(fam, []).append(z_full.values)
    out = pd.DataFrame({"depth": depth})
    for fam, zs in fam_z.items():
        M = np.vstack(zs)
        # nanmean across the family's columns; all-NaN rows stay NaN
        with np.errstate(invalid="ignore", all="ignore"):
            import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)
            out[fam] = np.where(np.all(np.isnan(M), axis=0),
                                np.nan, np.nanmean(M, axis=0))
    return out


def ind_part_b(paths, masks):
    print("PART B -- inter-family background correlation in real cores")
    print("(event/boundary depths masked; only pairs with >=15 shared "
          "background depths reported)\n")
    rows = []
    for p in paths:
        name = os.path.basename(p)
        df = pd.read_csv(p).sort_values(pd.read_csv(p, nrows=0).columns[0])
        Z = ind_family_series(df)
        # apply mask
        keep = np.ones(len(Z), bool)
        for key, ranges in masks.items():
            if key in name:
                for lo, hi in ranges:
                    keep &= ~((Z.depth >= lo) & (Z.depth <= hi))
        bg = Z[keep]
        fams = [c for c in Z.columns if c != "depth"]
        print(f"  {name}")
        for a, b in combinations(fams, 2):
            sub = bg[[a, b]].dropna()
            if len(sub) >= 15:
                r = float(np.corrcoef(sub[a], sub[b])[0, 1])
                flag = "" if abs(r) < 0.5 else "  <-- |r|>=0.5"
                print(f"     {a:9s} vs {b:9s}: r = {r:+.3f}  (n={len(sub)}){flag}")
                rows.append((name, a, b, round(r, 3), len(sub)))
            else:
                print(f"     {a:9s} vs {b:9s}: n={len(sub)} (too few)")
        print()
    if rows:
        out = pd.DataFrame(rows, columns=["core", "family_a", "family_b",
                                          "r", "n"])
        out.to_csv("ev_family_independence.csv", index=False)
        mx = out.r.abs().max()
        print(f"max |r| across all measurable pairs: {mx:.3f}  "
              f"({'all < 0.5' if mx < 0.5 else 'SOME >= 0.5'})")
        print("written: ev_family_independence.csv")


def ind_parse_masks(mask_args):
    masks = {}
    for m in mask_args or []:
        key, spec = m.split("=", 1)
        ranges = []
        for part in spec.split(","):
            lo, hi = part.split(":")
            ranges.append((float(lo), float(hi)))
        masks[key] = ranges
    return masks


# ---------------------------------------------------------------------------
def run_family_independence(core_paths=None, masks=None, seed=4):
    ind_part_a(seed=seed)
    if core_paths:
        ind_part_b(core_paths, masks or {})
    else:
        print("PART B skipped (no core files given).")

def main():
    parser = argparse.ArgumentParser(description="PRL validation suite")
    parser.add_argument("stage", choices=["benchmark", "published", "mondsee", "independence", "all"])
    parser.add_argument("cores", nargs="*", help="optional harmonised core CSVs for independence Part B")
    args = parser.parse_args()

    if args.stage in ("benchmark", "all"):
        run_synthetic_benchmark()
    if args.stage in ("published", "all"):
        run_published_validation()
    if args.stage in ("mondsee", "all"):
        run_mondsee_single_proxy()
    if args.stage in ("independence", "all"):
        run_family_independence(args.cores or None)

if __name__ == "__main__":
    main()
