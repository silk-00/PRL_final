
import argparse
import os, glob
import time
import numpy as np
import pandas as pd

# ----------------------------------------------------------------- CONSTANTS
# Fixed (physical or definitional) -- identical for every core, every tier.
# tau is NOT a fixed constant. |z| > 3 is an ASYMPTOTIC threshold: it assumes
# the rolling MAD is a good estimate of scale. At a window of 11 samples the MAD
# is computed from ~11 points and is itself noisy, so the robust score has much
# heavier tails than a standard normal. Measured on pure iid noise, |z| > 3
# occurs at 2.99% of depths at w = 5 cm against the nominal 0.27% -- an 11-fold
# inflation -- rising to 4.44% when the background is autocorrelated (rho=0.85).
# Family agreement then compounds it: two families with f false flags each in a
# core of length L coincide within +/-delta about f_A*f_B*2*delta/L times.
# The consequence is severe: at tau = 3 a synthetic core containing NO events
# yields 14 spurious "events". tau is therefore calibrated per core against
# surrogate data preserving that core's own autocorrelation.
TAU_FALLBACK      = 3.0    # used only if calibration cannot run
TAU_GRID          = np.arange(3.0, 9.01, 0.25)
E_EVENTS          = 1.0    # expected spurious events per core (declared)
N_SURROGATE       = 6      # surrogate realisations per core
DELTA_TOL         = 2.0    # cm, family co-location tolerance
N_MIN             = 5      # min real points in a window for a stable MAD
MIN_FAMILIES      = 2      # independent confirmations required
MIN_LENGTH_CM     = 50.0   # shortest usable core
MAD_FLOOR_FRAC    = 0.01   # local MAD must exceed this x column global MAD
CENSOR_FRAC       = 0.10   # >=10% of values pinned to column min => censored
SATURATION_FRAC   = 0.25   # event longer than this x core length = artefact
# THE SCALE LADDER IS FIXED ACROSS TIERS, in cm.
# It must span both ends: fine rungs resolve sharp contacts (a step scores
# |z|=4.9 at 5cm but only 3.0 at 25cm -- coarse windows straddle the step, the
# local MAD inflates, and the contact suppresses its own detection), while
# coarse rungs are required for thick deposits (a 64cm unit is invisible to any
# window shorter than itself). Tiers differ ONLY in which cores are admitted;
# if the ladder varied by tier the two runs would be different detectors and
# the comparison would confound admission with scale.
LADDER_CM = [5.0, 15.0, 45.0, 135.0]

# Tiers. T_min = thinnest layer the study claims to detect.
#   DELTA_MAX = T_min / 2   (a layer needs >=2 samples to be a layer)
TIERS = {"strict":  {"T_min": 2.0,  "delta_max": 1.0},
         "relaxed": {"T_min": 10.0, "delta_max": 5.0}}

IN_DIR = "ev_clean"
FINAL_OUTPUT = "PRL_final_results.csv"
DEFAULT_RETRIES = max(1, int(os.environ.get("PRL_RETRIES", "3")))
RETRY_WAIT_SECONDS = max(0.1, float(os.environ.get("PRL_RETRY_WAIT", "2.0")))


def retry_call(label, fn, *args, attempts=DEFAULT_RETRIES, **kwargs):
    """Retry a failed local detector operation without restarting earlier cores."""
    last = None
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last = exc
            if attempt >= attempts:
                raise
            wait = RETRY_WAIT_SECONDS * (2 ** (attempt - 1))
            print(f"    retry {label} ({attempt}/{attempts - 1}) after error: {exc}; waiting {wait:.1f}s", flush=True)
            time.sleep(wait)
    raise last

# magnetic folded into physical: susceptibility is density-dependent
FAMILY_PATTERNS = [
    ("organic",  ["organic", "loi", "toc", "tc_", "tn_", "corg"]),
    ("grain",    ["grain", "sand", "silt", "clay", "mean_um", "d50"]),
    ("geochem",  ["_ratio", "ca_ti", "fe_ti", "mn_fe", "k_ti", "zr_rb",
                  "si_ti", "ti_ca", "ratio_"]),
    ("physical", ["density", "water_pct", "porosity", "wet_bulk",
                  "mag_susc", "mags", "susceptibility", "kappa"]),
    ("colour",   ["lstar", "astar", "bstar", "colour", "color", "reflect"]),
    ("biogenic", ["diatom", "bsi", "biogenic", "pollen", "chironomid"]),
]


FAMILY_NAMES = {"organic", "grain", "geochem", "physical", "colour",
                "biogenic", "magnetic"}


def _prefix_family(col):
    head = str(col).split("_", 1)[0].lower()
    if head in FAMILY_NAMES:
        return "physical" if head == "magnetic" else head
    return None


def family_of(col):
    fam = _prefix_family(col)
    if fam:
        return fam
    c = col.lower()
    for fam, pats in FAMILY_PATTERNS:
        if any(p in c for p in pats):
            return fam
    return None


def ladder_for(_delta_max=None):
    """The fixed physical scale ladder. Identical for every core and tier.
    Which rungs a given column can actually support is derived at runtime from
    that column's sampling step -- never chosen."""
    return list(LADDER_CM)


def median_step(depths):
    d = np.diff(np.sort(np.asarray(depths, float)))
    d = d[d > 0]
    return float(np.median(d)) if len(d) else np.nan


def apply_censor_mask(series):
    """Blank detection-limit floors. Returns (masked series, n_masked)."""
    v = series.dropna()
    if len(v) < 10:
        return series, 0
    lo = v.min()
    n_at_min = int((v <= lo + 1e-9).sum())
    if n_at_min / len(v) >= CENSOR_FRAC and n_at_min >= 3:
        out = series.mask(series <= lo + 1e-9)
        return out, n_at_min
    return series, 0


def ar1_surrogate(n, rho, rng):
    """Null series preserving lag-1 autocorrelation, unit variance."""
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + rng.normal(0, 1)
    return x * np.sqrt(max(1e-9, 1.0 - rho ** 2))


def estimate_rho(v):
    """Lag-1 autocorrelation of the background. Event layers are a minority of
    any core, so this bulk statistic is dominated by background structure."""
    x = pd.Series(v).dropna().values
    if len(x) < 20:
        return 0.0
    x = x - np.median(x)
    r = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    return 0.0 if not np.isfinite(r) else float(np.clip(r, 0.0, 0.98))


def robust_z(values, n_win):
    """Rolling median/MAD z-score on a column's own (non-null) grid.

    Returns z with NaN wherever the local MAD is below the floor, so a
    degenerate window yields 'not evaluated' rather than an infinite score.
    """
    s = pd.Series(values, dtype=float)
    minp = max(N_MIN, n_win // 2)
    med = s.rolling(n_win, center=True, min_periods=minp).median()
    resid = s - med
    mad = resid.abs().rolling(n_win, center=True, min_periods=minp).median()

    global_mad = float((s - s.median()).abs().median())
    floor = MAD_FLOOR_FRAC * global_mad if global_mad > 0 else np.inf

    mad_ok = mad >= floor
    z = pd.Series(np.nan, index=s.index)
    z[mad_ok] = resid[mad_ok] / (1.4826 * mad[mad_ok])
    return z


def _events_from_flagsets(flag_frames, tau):
    """Flags -> family agreement -> merged intervals. Used by the calibration
    so the null is evaluated through the WHOLE pipeline, not just per depth:
    what matters is the rate of spurious EVENTS, not of spurious flags."""
    rows = []
    for fam, dep, absz in flag_frames:
        hit = absz > tau
        if hit.any():
            rows.append(pd.DataFrame({"depth": dep[hit], "family": fam}))
    if not rows:
        return 0
    fl = pd.concat(rows, ignore_index=True)
    cand = [d for d in np.sort(fl.depth.unique())
            if fl[(fl.depth - d).abs() <= DELTA_TOL].family.nunique()
            >= MIN_FAMILIES]
    if not cand:
        return 0
    cand = np.sort(np.array(cand))
    n = 1
    for a, b in zip(cand[:-1], cand[1:]):
        if b - a > 2 * DELTA_TOL:
            n += 1
    return n


def calibrate_tau(df, depth, fam_cols, ladder, length, rng):
    """Smallest tau on TAU_GRID for which the expected number of spurious
    events per core, under a null preserving each column's autocorrelation,
    is at most E_EVENTS.

    One mechanical rule, derived per core from that core's own noise structure.
    Nothing is fitted to the validation cores.
    """
    rhos = {}
    for fam, cols in fam_cols.items():
        for c in cols:
            rhos[c] = estimate_rho(df[c].values)

    surrogate_sets = []
    for _ in range(N_SURROGATE):
        frames = []
        for fam, cols in fam_cols.items():
            for c in cols:
                sub = df.loc[df[c].notna(), [depth.name, c]]
                nv = len(sub)
                if nv < N_MIN:
                    continue
                dstep = median_step(sub[depth.name].values)
                if not np.isfinite(dstep) or dstep <= 0:
                    continue
                sur = ar1_surrogate(nv, rhos[c], rng)
                dep = sub[depth.name].values
                for w_cm in ladder:
                    n_win = int(round(w_cm / dstep))
                    n_win += (n_win % 2 == 0)
                    if n_win < N_MIN or n_win > nv or w_cm > length:
                        continue
                    z = robust_z(sur, n_win).values
                    ok = ~np.isnan(z)
                    frames.append((fam, dep[ok], np.abs(z[ok])))
        surrogate_sets.append(frames)

    if not any(surrogate_sets):
        return TAU_FALLBACK, np.nan

    for tau in TAU_GRID:
        counts = [_events_from_flagsets(fs, tau) for fs in surrogate_sets]
        m = float(np.mean(counts))
        if m <= E_EVENTS:
            return float(tau), m
    return float(TAU_GRID[-1]), m


def stable_seed(name):
    import hashlib
    return int.from_bytes(hashlib.sha256(str(name).encode("utf-8")).digest()[:8], "little") % (2 ** 32)

# ------------------------------------------------------------------ per core
def analyse_core(path, tier_name, tier):
    df = pd.read_csv(path)
    core = os.path.basename(path).replace("_proxies.csv", "")
    depth_col = df.columns[0]
    df = df.sort_values(depth_col).reset_index(drop=True)
    depth = df[depth_col].astype(float)
    length = float(depth.max() - depth.min())

    info = {"core": core, "tier": tier_name, "length_cm": round(length, 1)}

    # ---- 1. censoring mask (before any statistic is computed) -------------
    censored = []
    for col in df.columns[1:]:
        if family_of(col) is None:
            continue
        df[col], n_masked = apply_censor_mask(df[col])
        if n_masked:
            censored.append(f"{col}:{n_masked}")
    info["censored"] = ";".join(censored)

    # ---- 2. per-family sampling step --------------------------------------
    fam_cols, fam_step = {}, {}
    for col in df.columns[1:]:
        fam = family_of(col)
        if fam is None:
            continue
        d = depth[df[col].notna()].values
        if len(d) < N_MIN:
            continue
        st = median_step(d)
        if not np.isfinite(st):
            continue
        fam_cols.setdefault(fam, []).append(col)
        fam_step[fam] = min(fam_step.get(fam, np.inf), st)

    fams = sorted(fam_step, key=lambda f: fam_step[f])
    d2 = fam_step[fams[1]] if len(fams) > 1 else np.nan
    info["n_families"] = len(fams)
    info["families"] = ";".join(fams)
    info["delta_2fam_cm"] = round(d2, 2) if np.isfinite(d2) else np.nan

    # ---- 3. admissibility gate -------------------------------------------
    reasons = []
    if len(fams) < MIN_FAMILIES:
        reasons.append(f"<{MIN_FAMILIES} families")
    if length < MIN_LENGTH_CM:
        reasons.append(f"core {length:.0f}cm < {MIN_LENGTH_CM:.0f}cm")
    if np.isfinite(d2) and d2 > tier["delta_max"]:
        reasons.append(f"delta_2fam={d2:.2f}cm > {tier['delta_max']}cm")
    if reasons:
        info["admitted"] = False
        info["reason"] = "; ".join(reasons)
        return info, [], []
    info["admitted"] = True
    info["reason"] = ""

    # ---- 4. multi-scale scoring on resolvable rungs only -------------------
    ladder = ladder_for()

    rng = np.random.default_rng(stable_seed(core))
    tau, exp_false = calibrate_tau(df, depth, fam_cols, ladder, length, rng)
    info["tau"] = round(tau, 2)
    info["expected_false_events"] = (round(exp_false, 2)
                                     if np.isfinite(exp_false) else np.nan)

    flags = []          # (depth, family, column, w_cm)
    rungs_used = set()

    for fam, cols in fam_cols.items():
        for col in cols:
            sub = df.loc[df[col].notna(), [depth_col, col]]
            if len(sub) < N_MIN:
                continue
            dstep = median_step(sub[depth_col].values)
            if not np.isfinite(dstep) or dstep <= 0:
                continue
            vals = sub[col].values
            dep = sub[depth_col].values

            for w_cm in ladder:
                n_win = int(round(w_cm / dstep))
                if n_win % 2 == 0:
                    n_win += 1
                # rung resolvable? enough real points, and fits in the core
                if n_win < N_MIN or n_win > len(vals) or w_cm > length:
                    continue
                z = robust_z(vals, n_win)
                hit = np.abs(z.values) > tau
                if hit.any():
                    rungs_used.add(w_cm)
                    for dd in dep[hit]:
                        flags.append((float(dd), fam, col, w_cm))

    info["rungs_resolvable"] = ";".join(str(int(w)) for w in sorted(rungs_used))
    if not flags:
        info["n_events"] = 0
        return info, [], []

    fl = pd.DataFrame(flags, columns=["depth", "family", "column", "w_cm"])

    # ---- 5. family agreement within DELTA_TOL ----------------------------
    cand = []
    uniq = np.sort(fl.depth.unique())
    for d0 in uniq:
        near = fl[(fl.depth - d0).abs() <= DELTA_TOL]
        if near.family.nunique() >= MIN_FAMILIES:
            cand.append(d0)
    if not cand:
        info["n_events"] = 0
        return info, [], []

    # ---- 6. merge into intervals (gap <= 2 x DELTA_TOL) -------------------
    cand = np.sort(np.array(cand))
    groups, cur = [], [cand[0]]
    for a, b in zip(cand[:-1], cand[1:]):
        if b - a <= 2 * DELTA_TOL:
            cur.append(b)
        else:
            groups.append(cur); cur = [b]
    groups.append(cur)

    events, saturated = [], []
    for g in groups:
        top, bot = float(min(g)), float(max(g))
        thick = bot - top
        near = fl[(fl.depth >= top - DELTA_TOL) & (fl.depth <= bot + DELTA_TOL)]
        # Edge test uses the FINEST firing scale: if the event is seen at a
        # scale whose window fits clear of the core ends, the detection stands
        # even though a coarser window also fired and is edge-affected.
        half_min = min(near.w_cm) / 2.0
        rec = {
            "core": core, "tier": tier_name, "tau": round(tau, 2),
            "depth_top": round(top, 2), "depth_bottom": round(bot, 2),
            "thickness_cm": round(thick, 2),
            "n_families": int(near.family.nunique()),
            "families": ";".join(sorted(near.family.unique())),
            "scales_cm": ";".join(str(int(w)) for w in sorted(near.w_cm.unique())),
            "columns": ";".join(sorted(near.column.unique())),
            # edge effect: rolling stats unreliable within half a window of the ends
            "edge": bool(top - depth.min() < half_min or depth.max() - bot < half_min),
        }
        # ---- 7. saturation guard -----------------------------------------
        if thick > SATURATION_FRAC * length:
            rec["saturation_frac"] = round(thick / length, 3)
            saturated.append(rec)
        else:
            events.append(rec)

    info["n_events"] = len(events)
    info["n_saturated"] = len(saturated)
    return info, events, saturated


# ---------------------------------------------------------------------- main
def main(output_file=FINAL_OUTPUT, retries=DEFAULT_RETRIES):
    if not os.path.isdir(IN_DIR):
        raise SystemExit(f"ERROR: '{IN_DIR}/' not found.")
    files = sorted(glob.glob(os.path.join(IN_DIR, "*.csv")))
    if not files:
        raise SystemExit(f"ERROR: no CSVs in {IN_DIR}/")

    all_events = []
    tier_summaries = []

    for tier_name, tier in TIERS.items():
        print(f"\n=== TIER '{tier_name}'  T_min={tier['T_min']}cm  "
              f"delta_max={tier['delta_max']}cm  "
              f"ladder={[int(w) for w in LADDER_CM]} cm (fixed) ===")

        infos, tier_events, all_sat = [], [], []
        for f in files:
            try:
                info, ev, sat = retry_call(f"detect {os.path.basename(f)} [{tier_name}]", analyse_core, f, tier_name, tier, attempts=retries)
            except Exception as e:
                print(f"  !! {os.path.basename(f)}: {e}")
                continue
            infos.append(info)
            tier_events.extend(ev)
            all_sat.extend(sat)

        idf = pd.DataFrame(infos)
        edf = pd.DataFrame(tier_events)

        # The public result is one consolidated table. Diagnostics stay in memory
        # so users do not have to manage a collection of CSVs for a normal run.
        if len(edf):
            summary_cols = [c for c in ["core", "tier", "expected_false_events", "delta_2fam_cm", "length_cm"]
                            if c in idf.columns]
            if summary_cols:
                edf = edf.merge(idf[summary_cols], on=["core", "tier"], how="left")
            edf.insert(0, "record_type", "event")
            all_events.append(edf)

        adm = int(idf.admitted.sum()) if len(idf) else 0
        print(f"  cores admitted        : {adm}/{len(idf)}")
        if "tau" in idf.columns and idf.tau.notna().any():
            t = idf.loc[idf.admitted, "tau"]
            if len(t):
                print(f"  calibrated tau        : median {t.median():.2f} "
                      f"(range {t.min():.2f}-{t.max():.2f}); "
                      f"target <= {E_EVENTS} false events/core")
        print(f"  events detected       : {len(edf)}"
              f" across {edf.core.nunique() if len(edf) else 0} cores")
        if len(edf):
            print(f"  median thickness (cm) : {edf.thickness_cm.median():.1f}")
            print(f"  events at core edges  : {int(edf.edge.sum())}")
        print(f"  saturated artefacts   : {len(all_sat)} (kept out of final event table)")

    if all_events:
        final = pd.concat(all_events, ignore_index=True, sort=False)
        final = final.sort_values(["core", "tier", "depth_top"]).reset_index(drop=True)
    else:
        final = pd.DataFrame(columns=[
            "record_type", "core", "tier", "tau", "expected_false_events",
            "depth_top", "depth_bottom", "thickness_cm", "n_families",
            "families", "scales_cm", "columns", "edge", "saturation_frac",
            "delta_2fam_cm", "length_cm"
        ])

    final.to_csv(output_file, index=False)
    print(f"\nFINAL OUTPUT: {output_file}")
    print(f"  detected event intervals: {len(final)}")
    if len(final):
        print(f"  cores with events        : {final.core.nunique()}")
        print(f"  strict                  : {int((final.tier == 'strict').sum())}")
        print(f"  relaxed                 : {int((final.tier == 'relaxed').sum())}")
    return output_file

# ---------------------------------------------------------------------------
# Public programmatic API used by validation.py
def detect_core(path, tier_name="strict"):
    tier = TIERS[tier_name]
    return analyse_core(path, tier_name, tier)

def detect_dataframe(df, tier_name="strict"):
    """Run the detector on an in-memory dataframe with core semantics."""
    import tempfile
    fd, path = tempfile.mkstemp(suffix="_proxies.csv")
    os.close(fd)
    try:
        df.to_csv(path, index=False)
        return detect_core(path, tier_name=tier_name)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PRL multi-scale multi-proxy event detector")
    parser.add_argument("--input", default=IN_DIR, help="directory containing *_proxies.csv files")
    parser.add_argument("--output", default=FINAL_OUTPUT, help="single final event-results CSV")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="retries after a core-level detector error (default: 3)")
    args = parser.parse_args()
    IN_DIR = args.input
    main(args.output, retries=max(1, args.retries))
