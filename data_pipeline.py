import argparse
import csv
import json
import io
import logging
import os
import re
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from contextlib import redirect_stdout
from pathlib import Path

try:
    from pangaeapy import PanQuery, PanDataSet
except ImportError:  # PANGAEA stages only; harmonisation/profile remain usable
    PanQuery = None
    PanDataSet = None

# ---------------------------------------------------------------------------
# Stage 1: PANGAEA candidate search
QUERIES = [
    "lake sediment flood layer grain size magnetic susceptibility",
    "turbidite lake sediment core grain size",
    "tephra layer lake sediment core geochemistry",
    "varved lake sediment core high resolution",
    "seismite lake sediment core event layer",
    "extreme event lake sediment record grain size",
    "storm deposit coastal lagoon sediment core",
    "debris flow lacustrine sediment core",
    "lake sediment core multiproxy loss on ignition magnetic susceptibility",
    "lake sediment geochemistry grain size organic carbon core",
    "peat bog core organic carbon magnetic susceptibility",
    "XRF core scanning lake sediment high resolution",
    "lacustrine sediment core physical properties density",
    "lake sediment core magnetic susceptibility grain size Holocene",
    "alpine lake sediment core multiproxy",
    "Arctic lake sediment core geochemistry",
    "tropical lake sediment core multiproxy",
    "Mediterranean lake sediment core event",
    "Andes lake sediment core grain size",
    "Asia lake sediment core multiproxy record",
    # Expanded discovery terms: still screened later for >=2 independent families.
    "lake sediment earthquake deposit core multiproxy",
    "lake sediment landslide deposit core multiproxy",
    "lake sediment tsunami deposit core multiproxy",
    "lake sediment storm deposit core multiproxy",
    "lake sediment flood deposit XRF grain size",
    "lake sediment turbidite XRF magnetic susceptibility",
    "lake sediment mass movement core grain size organic",
    "lake sediment event deposit magnetic susceptibility organic",
    "lacustrine sediment core XRF TOC grain size",
    "lake sediment core XRF magnetic susceptibility",
    "lake sediment core XRF organic carbon",
    "lake sediment core grain size magnetic susceptibility",
    "lake sediment core grain size loss on ignition",
    "lake sediment core bulk density magnetic susceptibility",
    "lake sediment core biogenic silica grain size",
    "lake sediment tephra XRF magnetic susceptibility",
    "lake sediment paleoflood multiproxy core",
]
TARGET_UNIQUE_CANDIDATES = 1000   # total across the whole search, not per query
PER_QUERY_LIMIT = 150          # request size sent to PANGAEA for each query
WORK_DIR = ".prl_work"
os.makedirs(WORK_DIR, exist_ok=True)
CANDIDATE_FILE = os.path.join(WORK_DIR, "ev_candidates.csv")
SEARCH_STATE_FILE = os.path.join(WORK_DIR, "search_state.json")
SEARCH_EXCLUDE = ["compilation", "eu project", "oceanograph", "ctd", "cruise",
                  "polarstern", "expedition", "database", "archive of", "iodp",
                  "odp ", "drill", "deep-sea", "deep sea", "pore water", "abyssal"]
DEFAULT_RETRIES = max(1, int(os.environ.get("PRL_RETRIES", "5")))
RETRY_WAIT_SECONDS = max(0.5, float(os.environ.get("PRL_RETRY_WAIT", "3.0")))
PANGAEA_CONNECT_TIMEOUT = max(5.0, float(os.environ.get("PRL_PANGAEA_CONNECT_TIMEOUT", "20")))
PANGAEA_READ_TIMEOUT = max(15.0, float(os.environ.get("PRL_PANGAEA_READ_TIMEOUT", "120")))


def _patch_pangaeapy_http():
    """Give pangaeapy a more tolerant HTTP layer for archive-scale runs.

    The pangaeapy client currently uses short connect/read timeouts internally
    for its metadata/data requests. That is fine for small interactive calls,
    but DOI-backed PANGAEA metadata can occasionally take longer. The wrapper
    below keeps pangaeapy's public API and parsing intact while using a longer
    timeout and HTTP-level retries for transient 429/5xx failures.
    """
    try:
        import pangaeapy._core as _core
        import pangaeapy.pandataset as _pandataset
        import pangaeapy.panquery as _panquery
    except Exception:
        return

    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))

    try:
        from pangaeapy._core import CURRENT_VERSION
    except Exception:
        CURRENT_VERSION = "prl"

    def robust_get_request(url, accepted_type=None, auth_token=None, **kwargs):
        headers = dict(kwargs.pop("headers", {}) or {})
        headers.setdefault("User-Agent", f"pangaeapy/{CURRENT_VERSION} PRL-event-pipeline")
        if accepted_type:
            headers["Accept"] = accepted_type
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        return session.get(
            url,
            headers=headers,
            timeout=(PANGAEA_CONNECT_TIMEOUT, PANGAEA_READ_TIMEOUT),
            allow_redirects=True,
            **kwargs,
        )

    # PanDataSet imports get_request into its module namespace, so both aliases
    # need replacing. The core module is patched as well for consistency.
    _core.get_request = robust_get_request
    _pandataset.get_request = robust_get_request

    # PanQuery uses requests.get directly and supplies its own short timeout.
    # Wrap only that module's requests.get so search calls get the same resilient
    # network behaviour without changing requests globally.
    _orig_panquery_get = _panquery.requests.get
    def robust_query_get(url, *args, **kwargs):
        kwargs["timeout"] = (PANGAEA_CONNECT_TIMEOUT, PANGAEA_READ_TIMEOUT)
        return session.get(url, *args, **kwargs)
    _panquery.requests.get = robust_query_get


def retry_call(label, fn, *args, attempts=DEFAULT_RETRIES, **kwargs):
    """Retry a transient operation; raise only after the final attempt."""
    last = None
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last = exc
            if attempt >= attempts:
                raise
            wait = RETRY_WAIT_SECONDS * (2 ** (attempt - 1))
            print(f"      retry {label} ({attempt}/{attempts - 1}) after error: {exc}; waiting {wait:.1f}s", flush=True)
            time.sleep(wait)
    raise last

# Apply the tolerant HTTP layer after retry helpers are defined.
if PanDataSet is not None:
    _patch_pangaeapy_http()

def atomic_write_csv(df, path):
    """Atomic CSV write so an interrupted run never leaves a partial result."""
    tmp = f"{path}.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

def save_candidate_rows(rows, out_file):
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    with open(out_file, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["title", "doi", "link", "query"])
        w.writeheader(); w.writerows(rows)

def clean_search_title(html):
    t = re.sub(r"<[^>]+>", " ", html or "")
    return re.sub(r"\s+", " ", t).strip()[:180]

def search_candidates(out_file=CANDIDATE_FILE, target=TARGET_UNIQUE_CANDIDATES, per_query_limit=PER_QUERY_LIMIT):
    """Discover up to `target` UNIQUE collection datasets TOTAL across all queries.

    Progress is checkpointed after every successful query. Failed queries are
    retried automatically and left eligible for a later run. Rerunning the
    search stage resumes from the existing candidate/state files.
    """
    if PanQuery is None:
        raise RuntimeError("pangaeapy is required for the PANGAEA search stage")

    seen = {}
    if os.path.exists(out_file):
        try:
            with open(out_file, newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    doi = (r.get("doi") or "").strip()
                    if doi:
                        seen[doi] = r
        except Exception as exc:
            print(f"Could not resume {out_file}: {exc}; rebuilding candidate index.")

    done = set()
    if os.path.exists(SEARCH_STATE_FILE):
        try:
            with open(SEARCH_STATE_FILE, encoding="utf-8") as fh:
                done = set(json.load(fh).get("completed_queries", []))
        except Exception:
            done = set()

    skipped = 0
    t0 = time.time()
    for i, q in enumerate(QUERIES, 1):
        if len(seen) >= target:
            break
        if q in done:
            print(f"[{i}/{len(QUERIES)}] already completed: {q[:56]}")
            continue
        print(f"[{i}/{len(QUERIES)}] {q[:56]} ...", flush=True)
        try:
            res = retry_call(f"search query {i}", PanQuery, q, limit=per_query_limit)
            if getattr(res, "error", None):
                raise RuntimeError(res.error)
        except Exception as exc:
            print(f"      query failed after {DEFAULT_RETRIES} attempts: {exc}")
            continue

        new = 0
        for hit in (res.result or []):
            if hit.get("type") != "collection":
                continue
            doi = (hit.get("URI") or "").replace("doi:", "")
            if not doi or doi in seen:
                continue
            title = clean_search_title(hit.get("html", ""))
            if any(w in title.lower() for w in SEARCH_EXCLUDE):
                skipped += 1
                continue
            seen[doi] = {"title": title, "doi": doi,
                         "link": "https://doi.pangaea.de/" + doi, "query": q}
            new += 1
            if len(seen) >= target:
                break

        done.add(q)
        save_candidate_rows(list(seen.values())[:target], out_file)
        with open(SEARCH_STATE_FILE, "w", encoding="utf-8") as fh:
            json.dump({"target": target, "per_query_limit": per_query_limit,
                       "completed_queries": sorted(done)}, fh, indent=2)
        print(f"      +{new} (unique: {len(seen)}/{target}, {time.time()-t0:.0f}s)", flush=True)

    save_candidate_rows(list(seen.values())[:target], out_file)
    print(f"Saved {min(len(seen), target)} unique candidates to {out_file} ({skipped} excluded).")
    if len(seen) < target:
        print(f"WARNING: only {len(seen)} unique candidates were found after all queries.")
    return out_file

# ---------------------------------------------------------------------------
# Stage 2: proxy-family screening
MAX_CHILDREN = 50
MIN_PROXIES = 2
PANGAEA_PAUSE = 0.2
PROXIES = {
    "organic": ["loss on ignition", "loi", "toc", "organic carbon", "organic matter",
                "c org", "corg", "total carbon", "tc ", "tn ", "c/n", "nitrogen"],
    "mag_susc": ["magnetic susceptibility", "suscept", "mags", "kappa", "chi lf", "magnetic"],
    "grain": ["grain size", "grainsize", "sand", "silt", "clay", "mud", "µm",
              "particle size", "d50", "sorting"],
    "geochem": ["xrf", "count", "cps", "sio2", "al2o3", "k2o", "cao", "tio2",
                "mgo", "na2o", "fe2o3", "mno", "p2o5", "/ti", "/al", "/ca",
                "element", "ppm", "ca ", "ti ", "fe ", "k ", "si ", "sr ", "zr ",
                "rb ", "mn ", "geochem"],
    "density": ["bulk density", "dbd", "dry density", "water content", "porosity",
                "wet density", "gamma density"],
    "colour": ["l*", "a*", "b*", "colour", "color", "reflectance", "brightness"],
    "biogenic": ["biogenic silica", "bsi", "opal", "diatom", "carbonate", "caco3"],
}
logging.getLogger("pangaeapy").setLevel(logging.CRITICAL)

def pangaea_quiet(doi):
    if PanDataSet is None:
        raise RuntimeError("pangaeapy is required for this stage")
    with redirect_stdout(io.StringIO()):
        return PanDataSet(doi, include_data=False, enable_cache=True)

def pangaea_quiet_retry(doi):
    return retry_call(f"PANGAEA {doi}", pangaea_quiet, doi)

def params_of(doi):
    ds = pangaea_quiet_retry(doi)
    title = ds.title or doi
    members = getattr(ds, "collection_members", None)
    if getattr(ds, "isCollection", False) and members:
        if len(members) > MAX_CHILDREN:
            return title, "", [], f"mega-bundle ({len(members)} datasets)"
        names = []
        for ch in members:
            try:
                names += [p.name for p in (pangaea_quiet_retry(ch).params or {}).values()]
            except Exception as exc:
                print(f"      child metadata failed {ch}: {exc}")
            time.sleep(PANGAEA_PAUSE)
        return title, " | ".join(names).lower(), names, ""
    names = [p.name for p in (ds.params or {}).values()]
    return title, " | ".join(names).lower(), names, ""

def check_coverage(in_file=CANDIDATE_FILE, out_file=None):
    """Screen candidates. Successful rows are checkpointed; failed DOIs are retried next run."""
    out_file = out_file or os.path.join(WORK_DIR, "ev_coverage.csv")
    with open(in_file, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    previous = {}
    if os.path.exists(out_file):
        try:
            with open(out_file, newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    if r.get("doi"):
                        previous[r["doi"].strip()] = r
        except Exception:
            previous = {}

    out = dict(previous)
    print(f"Checking {len(rows)} candidates. KEEP => >= {MIN_PROXIES} proxy families")
    for i, r in enumerate(rows, 1):
        doi = (r.get("doi") or "").strip()
        if not doi or doi in out:
            continue
        full = doi if doi.startswith("doi:") else f"doi:{doi}"
        try:
            title, blob, names, note = params_of(full)
        except Exception as exc:
            print(f"[{i}/{len(rows)}] {doi} FAILED after retries: {exc}")
            continue
        if note:
            rec = {"doi": doi, "title": title, "n_proxies": 0, "proxies": "",
                   "verdict": "skip", "note": note, "sample_params": ""}
        else:
            found = {k: any(w in blob for w in ws) for k, ws in PROXIES.items()}
            n = sum(found.values())
            uniq, seen_names = [], set()
            for nm in names:
                key = re.sub(r"\s*\[.*?\]", "", str(nm)).strip().lower()
                if key and key not in seen_names:
                    seen_names.add(key); uniq.append(str(nm))
                if len(uniq) >= 14:
                    break
            rec = {"doi": doi, "title": title, "n_proxies": n,
                   "proxies": ";".join(k for k, v in found.items() if v),
                   "verdict": "KEEP" if n >= MIN_PROXIES else "skip",
                   "note": "", "sample_params": " | ".join(uniq)}
        out[doi] = rec
        atomic_write_csv(pd.DataFrame(out.values()), out_file)
        print(f"[{i}/{len(rows)}] {doi} -> {rec['verdict']}", flush=True)
        time.sleep(PANGAEA_PAUSE)
    atomic_write_csv(pd.DataFrame(out.values()), out_file)
    print(f"Saved coverage table: {out_file} ({len(out)} checked successfully)")
    return out_file



import csv, os, re, time

DOWNLOAD_DIR, COVERAGE_FILE = "ev_cores", os.path.join(WORK_DIR, "ev_coverage.csv")
MAX_CHILDREN, PAUSE = 50, 0.3
EXCLUDE = ["compilation", "eu project", "oceanograph", "ctd", "cruise",
           "polarstern", "expedition", "database", "iodp", "odp ", "drill"]


def san(s, n=48):
    s = re.sub(r"[^A-Za-z0-9]+", "_", s or "").strip("_")
    return s[:n] or "untitled"


def did(doi):
    m = re.search(r"PANGAEA\.(\d+)", doi or "")
    return m.group(1) if m else san(doi)


def process_download(doi):
    """Download one candidate bundle.

    A bundle is marked complete only when every child dataset downloads and the
    completion marker is written. Partial bundles are deliberately left
    incomplete so the same candidate is retried on the next attempt/run.
    """
    full = doi if doi.startswith("doi:") else f"doi:{doi}"
    parent = retry_call(f"parent download {doi}", PanDataSet, full,
                        include_data=False, enable_cache=True)
    title = parent.title or doi
    core = f"{did(doi)}_{san(title, 40)}"
    folder = os.path.join(DOWNLOAD_DIR, core)
    complete_marker = os.path.join(folder, "_complete.ok")
    if os.path.isdir(folder) and os.path.exists(complete_marker):
        print(f"  have  {core[:52]}")
        return "have"

    hit = next((w for w in EXCLUDE if w in title.lower()), None)
    if hit:
        print(f"  SKIP  {core[:44]} (keyword '{hit}')")
        return "skip"

    members = getattr(parent, "collection_members", None)
    children = members if (getattr(parent, "isCollection", False) and members) else [full]
    if len(children) > MAX_CHILDREN:
        print(f"  SKIP  {core[:44]} ({len(children)} datasets)")
        return "skip"

    os.makedirs(folder, exist_ok=True)
    idx = [f"parent DOI : {doi}", f"title      : {title}", ""]
    failures = 0
    print(f"\n{core}  ({len(children)} datasets)")
    for ch in children:
        try:
            cds = retry_call(f"child download {ch}", PanDataSet, ch,
                             include_data=True, enable_cache=True)
            csv_path = os.path.join(folder, f"{did(ch)}_{san(cds.title, 44)}.csv")
            cds.data.to_csv(csv_path, index=False)
            line = f"{os.path.basename(csv_path)}  ({len(cds.data)} rows)"
            print("    " + line)
            idx.append(line)
        except Exception as exc:
            failures += 1
            print(f"    child FAILED after retries: {ch}: {exc}")
            idx.append(f"FAILED: {ch} :: {exc}")
        time.sleep(PAUSE)

    with open(os.path.join(folder, "_index.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(idx) + "\n")

    if failures == 0:
        Path(complete_marker).write_text("complete\n", encoding="utf-8")
        return "downloaded"

    # Remove an old completion marker if a rerun discovered a new child failure.
    try:
        if os.path.exists(complete_marker):
            os.remove(complete_marker)
    except OSError:
        pass
    print(f"    {failures} child download(s) failed; bundle marked incomplete for retry.")
    return "failed"


import glob, json, os, re
import numpy as np
import pandas as pd

CORES_DIR, OUT_DIR = "ev_cores", "ev_clean"
AUDIT_FILE = os.path.join(WORK_DIR, "ev4_audit.csv")
MIN_PROXIES, MIN_ROWS = 2, 25

# ----------------------------------------------------------- proxy vocabulary
ALIASES = {
    "organic":         ["loss on ignition", "loi", "toc", "organic carbon",
                        "organic matter", "c org", "corg", "tot org c"],
    "mag_susc":        ["magnetic susceptibility", "suscept", "mags", "kappa",
                        "susc"],
    "grain_size_mean": ["grain size mean", "mean grain", "grain size median",
                        "median grain", "d50", "grain size average",
                        "average grain size", "mean grain size"],
    "sand_pct":        ["sand"],
    "silt_pct":        ["silt"],
    "clay_pct":        ["clay"],
    "density":         ["bulk density", "dbd", "dry density", "gamma density",
                        "wet bulk density", "density"],
    "water_pct":       ["water content", "water wm", "water"],
    "porosity":        ["porosity"],
    "lightness":       ["l*", "lightness", "reflectance", "greyscale",
                        "grayscale", "grey", "gray"],
}

# Short archive codes are matched EXACTLY (after stripping units), never as
# substrings -- "TC" or "TS" as a substring would collide with dozens of words.
# Names are emitted family-prefixed so the detector need not guess the family.
EXACT_MAP = {
    "wbd": "density", "dbd": "density", "gra": "density",
    "poros": "porosity", "porosity": "porosity",
    "shear str": "physical_shear_strength",
    "sort p": "grain_sorting", "sorting": "grain_sorting",
    "tc": "organic_TC", "tn": "organic_TN",
    "tic": "geochem_TIC", "caco3": "geochem_CaCO3",
    "ts": "geochem_TS", "sio2": "geochem_SiO2",
    "bsio2": "biogenic_SiO2", "bsi": "biogenic_SiO2",
    "grey": "lightness", "gray": "lightness",
    "kappa": "mag_susc",
}

# Water-column profiles ("Depth water", O2/pH/conductivity) are NOT sediment
# records. If one shares an Event value with sediment files its measurements
# would be merged onto the sediment depth axis.
HYDRO_MARKERS = {"o2", "o2 sat", "ph", "cond", "temp", "sal", "eh", "orp",
                 "turbidity", "chl a"}

# PANGAEA grain-size fractions: "<2 µm", "63-20 µm gU", ">63 µm", "250-125 µm".
# Each fraction is kept as its OWN column. Collapsing the silt sub-fractions
# into one "silt" would either average them (meaningless) or sum them
# (inventing a total that was never measured).
FRACTION_RE = re.compile(
    r"^\s*(?:(<|>)\s*([\d.]+)|([\d.]+)\s*-\s*([\d.]+))\s*(?:µm|um|mm)\b",
    re.I)

# FIX 2: matched on word boundaries, never as bare substrings
SKIP_WORDS = ["age", "latitude", "longitude", "date", "time", "event",
              "sample id", "label", "comment", "core id", "depth top",
              "depth bot", "id", "flag", "method", "reference",
              "elevation", "210pb", "137cs", "226ra", "cal age"]
SKIP_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in SKIP_WORDS) + r")\b")

# FIX 1: real elements, and every unit token PANGAEA actually uses for XRF
XRF_ELEMENTS = {
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "K", "Ca", "Sc", "Ti", "V", "Cr",
    "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Rb",
    "Sr", "Y", "Zr", "Nb", "Mo", "Ag", "Cd", "Sn", "Sb", "Te", "I", "Ba",
    "La", "Ce", "Hf", "Ta", "W", "Au", "Hg", "Tl", "Pb", "Bi", "Th", "U",
}
COUNT_TOKENS = ["cps", "kcps", "counts", "count", "cts", "area", "peak area",
                "total counts", "#", "intensity"]
CONC_TOKENS  = ["%", "wt%", "ppm", "mg/kg", "g/kg", "µg/g", "ug/g"]

RATIO_PAIRS = [("Ca", "Ti"), ("Fe", "Ti"), ("Mn", "Fe"), ("K", "Ti"),
               ("Zr", "Rb"), ("Si", "Ti")]

CORE_RE = re.compile(
    r"(?:core|profile)[_\s-]+"
    r"([A-Za-z]{1,5}[-_]?\d{1,6}(?:[-_](?:\d{1,3}|[A-Za-z](?![A-Za-z])))?)(?!\d)",
    re.I)

AUDIT = []


def log(core, kind, detail):
    AUDIT.append({"core": core, "kind": kind, "detail": detail})


def core_id_from(fname):
    m = CORE_RE.search(os.path.basename(fname))
    return m.group(1).upper().replace("-", "_") if m else None


def is_skipped(colname):
    return bool(SKIP_RE.search(str(colname).lower()))


def normalise(colname):
    """Lowercase, drop bracketed units, collapse whitespace."""
    c = re.sub(r"\[[^\]]*\]", " ", str(colname))
    return re.sub(r"\s+", " ", c).strip().lower()


def _fmt(x):
    xi = int(round(x))
    return str(xi) if abs(x - xi) < 1e-9 else str(x).replace(".", "p")


def fraction_name(colname):
    """Grain-size fraction column -> its own grain_frac_* name, or None."""
    m = FRACTION_RE.match(str(colname))
    if not m:
        return None
    if m.group(1):
        v = float(m.group(2))
        return f"grain_frac_0_{_fmt(v)}" if m.group(1) == "<" \
            else f"grain_frac_{_fmt(v)}_plus"
    a, b = float(m.group(3)), float(m.group(4))
    lo, hi = min(a, b), max(a, b)
    return f"grain_frac_{_fmt(lo)}_{_fmt(hi)}"


def is_water_column(df):
    """True if the frame is a lake-water profile rather than a sediment core."""
    norm = {normalise(c) for c in df.columns}
    if any("depth water" in n for n in norm):
        return True
    return len(norm & HYDRO_MARKERS) >= 3


# ------------------------------------------------------------- depth handling
UNIT_RE = re.compile(r"\[\s*(m|cm|mm)\s*\]", re.I)


def declared_unit(colname):
    """Depth unit taken from the header where the archive declares one."""
    m = UNIT_RE.search(str(colname))
    if m:
        return m.group(1).lower()
    cl = str(colname).lower()
    if "metre" in cl or "meter" in cl:
        return "m"
    return None


def find_depth(cols):
    low = {c: str(c).lower() for c in cols
           if "water" not in str(c).lower()}      # never "Depth water"
    for c, cl in low.items():
        if "depth sed" in cl:
            return c
    for c, cl in low.items():
        if cl.strip() in ("depth", "depth [m]", "depth [cm]", "depth [mm]"):
            return c
    for c, cl in low.items():
        if "depth" in cl and "top" not in cl and "bot" not in cl:
            return c
    return None


def resolve_depths(frames, core):
    """FIX 3: decide depth units ONCE PER CORE, then apply to every file.

    PANGAEA exports the standard sediment-depth parameter as a bare
    "Depth sed" column with the unit stripped, so no unit can be read from the
    header. The core-wide maximum is therefore the only evidence available --
    but it is evidence about the CORE, not about each file. Deciding per file
    (v1) inflated any proxy measured over a shallow interval by 100x, because
    its own max fell under the threshold while the core's did not. One decision
    per core removes that failure entirely: a shallow file inherits the core's
    unit instead of guessing its own.
    """
    entries = []          # (frame_index, depth_col, raw values, declared unit)
    for i, df in enumerate(frames):
        dcol = find_depth(df.columns)
        if dcol is None:
            continue
        v = pd.to_numeric(df[dcol], errors="coerce")
        if v.dropna().empty:
            continue
        entries.append([i, dcol, v, declared_unit(dcol)])
    if not entries:
        return {}

    scale = {"m": 100.0, "cm": 1.0, "mm": 0.1}

    # declared units, where any exist, are authoritative for those files
    undeclared = [e for e in entries if e[3] not in scale]
    core_max = max(float(v.max()) for _, _, v, u in entries if u not in scale) \
        if undeclared else None

    core_unit = None
    if core_max is not None:
        core_unit = "m" if core_max <= 30 else "cm"
        log(core, "depth_unit_core",
            f"{core_unit} for all {len(undeclared)} undeclared files "
            f"(core-wide max {core_max:.3f})")
        if core_unit == "cm" and core_max > 3000:
            log(core, "depth_suspect", f"core-wide max {core_max:.1f} as cm")

    out = {}
    for i, dcol, v, unit in entries:
        if unit in scale:
            out[i] = (v * scale[unit], f"declared {unit}")
        else:
            out[i] = (v * scale[core_unit], f"core rule {core_unit}")
    return out


# --------------------------------------------------------------- geochemistry
def element_columns(df):
    """FIX 1: element -> series, across the unit spellings actually in use."""
    els, kinds = {}, {}
    for c in df.columns:
        s = str(c).strip()
        m = re.match(r"^([A-Z][a-z]?)\b(.*)$", s)
        if not m:
            continue
        el, rest = m.group(1), m.group(2).lower()
        if el not in XRF_ELEMENTS:
            continue
        rest_clean = rest.strip(" []()/")
        kind = None
        if any(t in rest for t in COUNT_TOKENS):
            kind = "counts"
        elif any(t in rest for t in CONC_TOKENS):
            kind = "conc"
        elif rest_clean == "":
            kind = "bare"
        if kind is None:
            continue
        v = pd.to_numeric(df[c], errors="coerce")
        if v.notna().sum() < 5:
            continue
        # prefer counts over concentration over bare
        rank = {"counts": 3, "conc": 2, "bare": 1}
        if el not in els or rank[kind] > rank[kinds[el]]:
            els[el], kinds[el] = v, kind
    # a lone bare-element column is more likely a coincidence than an XRF table
    if all(k == "bare" for k in kinds.values()) and len(els) < 3:
        return {}, {}
    return els, kinds


def geochem_series(df):
    """Element ratios, plus any ratio the archive already provides."""
    out, note = {}, []
    els, kinds = element_columns(df)
    for a, b in RATIO_PAIRS:
        if a in els and b in els:
            r = (els[a] / els[b]).replace([np.inf, -np.inf], np.nan)
            if r.notna().sum() >= 5:
                out[f"geochem_{a}_{b}"] = r
                note.append(f"{a}/{b}[{kinds[a]}]")
    for c in df.columns:
        m = re.match(r"^([A-Z][a-z]?)\s*/\s*([A-Z][a-z]?)\b", str(c).strip())
        if m and m.group(1) in XRF_ELEMENTS and m.group(2) in XRF_ELEMENTS:
            name = f"geochem_{m.group(1)}_{m.group(2)}"
            if name not in out:
                v = pd.to_numeric(df[c], errors="coerce")
                if v.notna().sum() >= 5:
                    out[name] = v
                    note.append(f"{m.group(1)}/{m.group(2)}[given]")
    return out, note


# --------------------------------------------------------------------- build
def find_alias_cols(cols, words):
    """All columns matching a proxy (FIX 5: no longer first-wins)."""
    hits = []
    for c in cols:
        cl = str(c).lower()
        if is_skipped(c):
            continue
        if any(w in cl for w in words):
            hits.append(c)
    return hits


def build(frames, core):
    """Long-form gather, then one value per (depth, proxy). FIX 5 and FIX 6."""
    depths = resolve_depths(frames, core)
    records = []

    for i, df in enumerate(frames):
        if i not in depths:
            continue
        if is_water_column(df):
            log(core, "water_column_skipped", f"frame {i}")
            continue
        depth = depths[i][0].round(2)
        claimed = set()

        # exact archive codes and grain-size fractions first -- both are
        # unambiguous, and matching them before the substring aliases stops a
        # loose alias from claiming a column that has a precise meaning
        for col in df.columns:
            if is_skipped(col):
                continue
            pname = EXACT_MAP.get(normalise(col)) or fraction_name(col)
            if not pname:
                continue
            v = pd.to_numeric(df[col], errors="coerce")
            ok = v.notna().values & depth.notna().values
            if ok.sum() < 5:
                continue
            records.append(pd.DataFrame({
                "depth": depth.values[ok], "proxy": pname,
                "value": v.values[ok]}))
            log(core, "map", f"{pname} <- {col}")
            claimed.add(col)

        for name, words in ALIASES.items():
            for col in find_alias_cols(df.columns, words):
                if col in claimed:
                    continue
                v = pd.to_numeric(df[col], errors="coerce")
                if v.notna().sum() < 5:
                    continue
                ok = v.notna().values & depth.notna().values
                if ok.sum() < 5:
                    continue
                records.append(pd.DataFrame({
                    "depth": depth.values[ok],
                    "proxy": name,
                    "value": v.values[ok]}))
                log(core, "map", f"{name} <- {col}")
                break            # one column per proxy per file

        # record numeric columns that matched nothing, so alias gaps can be
        # triaged from evidence in one batch rather than guessed at
        mapped = set()
        for _n, _w in ALIASES.items():
            mapped.update(find_alias_cols(df.columns, _w))
        unmapped = [str(c) for c in df.columns
                    if c not in mapped and c not in claimed
                    and not is_skipped(c)
                    and str(c).lower() not in ("depth sed", "depth")
                    and pd.to_numeric(df[c], errors="coerce").notna().sum() >= 5]
        if unmapped:
            log(core, "unmapped", ";".join(unmapped[:12]))

        gser, gnote = geochem_series(df)
        for gname, s in gser.items():
            ok = s.notna().values & depth.notna().values
            if ok.sum() < 5:
                continue
            records.append(pd.DataFrame({
                "depth": depth.values[ok], "proxy": gname,
                "value": s.values[ok]}))
        if gnote:
            log(core, "geochem", ";".join(gnote))

    if not records:
        return None

    long = pd.concat(records, ignore_index=True)
    # FIX 6: duplicates averaged rather than joined combinatorially
    wide = (long.groupby(["depth", "proxy"], as_index=False)["value"].mean()
                .pivot(index="depth", columns="proxy", values="value")
                .reset_index())
    wide.columns.name = None
    return wide.sort_values("depth").dropna(how="all", axis=1).reset_index(drop=True)


# --------------------------------------------------------------------- output
def bundle_meta(folder):
    doi = title = ""
    idx = os.path.join(folder, "_index.txt")
    if os.path.exists(idx):
        # Index files may have been created by an older run with a
        # non-UTF-8 Windows encoding. Read defensively so one bad
        # character cannot stop the entire harmonization run.
        try:
            with open(idx, "r", encoding="utf-8-sig", errors="replace") as f:
                for ln in f:
                    if ln.lower().startswith("parent doi"):
                        doi = ln.split(":", 1)[1].strip()
                    if ln.lower().startswith("title"):
                        title = ln.split(":", 1)[1].strip()
        except OSError as e:
            log(os.path.basename(folder), "index_read_fail",
                f"{os.path.basename(idx)}: {e}")
    return doi, title


def latlon_from(frames):
    for d in frames:
        la = [c for c in d.columns if "latitude" in str(c).lower()]
        lo = [c for c in d.columns if "longitude" in str(c).lower()]
        if la and lo:
            try:
                return (float(d[la[0]].dropna().iloc[0]),
                        float(d[lo[0]].dropna().iloc[0]))
            except Exception:
                pass
    return None, None


def write_core(name, frames, folder, doi, btitle):
    merged = build(frames, name)
    if merged is None:
        print(f"   SKIP {name[:50]} (no proxy columns)")
        log(name, "skip", "no proxy columns")
        return 0
    proxies = [c for c in merged.columns if c != "depth"]
    if len(proxies) < MIN_PROXIES or len(merged) < MIN_ROWS:
        print(f"   SKIP {name[:50]} ({len(proxies)} proxies, {len(merged)} rows)")
        log(name, "skip", f"{len(proxies)} proxies, {len(merged)} rows")
        return 0
    os.makedirs(OUT_DIR, exist_ok=True)
    merged.to_csv(os.path.join(OUT_DIR, f"{name}_proxies.csv"), index=False)
    lat, lon = latlon_from(frames)
    json.dump({"core_id": name, "bundle_title": btitle, "source_doi": doi,
               "latitude": lat, "longitude": lon, "proxies": proxies,
               "n_depths": int(len(merged)), "n_files": len(frames),
               "depth_min_cm": float(merged.depth.min()),
               "depth_max_cm": float(merged.depth.max())},
              open(os.path.join(OUT_DIR, f"{name}_meta.json"), "w"), indent=2)
    print(f"   OK   {name[:50]}  {len(merged):5d} depths, {len(proxies)} proxies"
          f"  [{merged.depth.min():.1f}-{merged.depth.max():.1f} cm]")
    log(name, "written", f"{len(merged)} depths, {len(proxies)} proxies, "
                         f"{merged.depth.min():.1f}-{merged.depth.max():.1f}cm")
    return 1


def event_column(df):
    for c in df.columns:
        if str(c).strip().lower() in ("event", "event label", "station"):
            return c
    return None


def consolidate_ids(groups, bundle, authoritative):
    """FIX 4: consolidation repairs FILENAME TRUNCATION only.

    Download filenames are length-capped ("...profile_PG111" for PG1111), so a
    filename-derived id can be a truncated form of another. The PANGAEA Event
    column is never truncated, so Event-derived ids are authoritative and are
    NEVER merged -- without this, LAKE850_INLET, LAKE850_OUTLET,
    LAKE850_CORE2_2019 and LAKE850_CORE3_1999 all collapse onto one depth axis,
    which is the exact corruption this script exists to prevent.
    """
    ids = sorted([k for k in groups if k], key=len)
    for long_id in list(ids)[::-1]:
        if long_id not in groups:
            continue
        for short_id in ids:
            if short_id == long_id or short_id not in groups:
                continue
            if not long_id.startswith(short_id):
                continue
            if long_id in authoritative or short_id in authoritative:
                log(bundle, "kept_distinct",
                    f"{long_id} vs {short_id} (Event-derived)")
                break
            rest = long_id[len(short_id):]
            if rest and rest[0] in "_-":
                groups[short_id].extend(groups.pop(long_id))
                log(bundle, "consolidate", f"{long_id} -> {short_id} (filename)")
            else:
                log(bundle, "kept_distinct", f"{long_id} vs {short_id}")
            break
    return groups


def process_bundle(folder):
    bundle = os.path.basename(folder)
    doi, btitle = bundle_meta(folder)
    paths = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not paths:
        return 0

    groups, authoritative = {}, set()
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            log(bundle, "read_fail", f"{os.path.basename(p)}: {e}")
            continue
        ecol = event_column(df)
        if ecol is not None and df[ecol].notna().any():
            for val, sub in df.groupby(df[ecol].astype(str).str.strip()):
                if val and val.lower() not in ("nan", ""):
                    cid = re.sub(r"[^A-Za-z0-9]+", "_", val).strip("_").upper()
                    authoritative.add(cid)
                    groups.setdefault(cid, []).append(sub.reset_index(drop=True))
        else:
            groups.setdefault(core_id_from(p), []).append(df)

    groups = consolidate_ids(groups, bundle, authoritative)
    real = {k: v for k, v in groups.items() if k is not None}
    unlabelled = groups.get(None, [])

    if len(real) <= 1:
        if unlabelled and real:
            log(bundle, "unlabelled_merged",
                f"{len(unlabelled)} files with no core id merged into "
                f"{list(real)[0]}")
        allframes = [f for v in groups.values() for f in v]
        return write_core(bundle, allframes, folder, doi, btitle)

    print(f"   bundle has {len(real)} cores: {', '.join(sorted(real))}"
          + (f"  ({len(unlabelled)} unlabelled files skipped)" if unlabelled else ""))
    log(bundle, "split", f"{len(real)} cores: {','.join(sorted(real))}")
    n = 0
    for cid, frames in sorted(real.items()):
        n += write_core(f"{bundle[:26]}__{cid}", frames, folder, doi, btitle)
    return n


import os, glob, csv
import numpy as np
import pandas as pd

IN_DIR   = "ev_clean"
OUT_FILE = os.path.join(WORK_DIR, "ev_profile.csv")

DELTA_TOL = 2.0     # cm, family co-location tolerance (fixed, physical)
N_MIN     = 5       # min real points inside a window for a stable MAD

# Tiers to evaluate. T_min = thinnest layer claimed detectable.
# Delta_max = T_min / 2  (a layer needs >=2 samples to be a layer, not a point)
TIERS = {"strict_T2cm": 1.0, "relaxed_T10cm": 5.0}
FIXED_LADDER_CM = [5.0, 15.0, 45.0, 135.0]

# column-name substring -> proxy family (magnetic+physical merged: density
# dependence of volume-specific susceptibility makes them non-independent)
FAMILY_PATTERNS = [
    ("organic",  ["organic", "loi", "toc", "toc_", "tc_", "tn_", "corg"]),
    ("grain",    ["grain", "sand", "silt", "clay", "mean_um", "d50"]),
    ("geochem",  ["_ratio", "ca_ti", "fe_ti", "mn_fe", "k_ti", "zr_rb",
                  "si_ti", "ti_ca", "ratio_"]),
    ("physical", ["density", "water_pct", "porosity", "wet_bulk",
                  "mag_susc", "mags", "susceptibility", "kappa"]),
    ("colour",   ["lstar", "astar", "bstar", "colour", "color", "rgb",
                  "reflect"]),
    ("biogenic", ["diatom", "bsi", "biogenic", "pollen", "chironomid"]),
]




# Columns emitted by ev4 are family-prefixed ("geochem_CaCO3",
# "physical_shear_strength"). An explicit prefix is authoritative -- pattern
# matching is only the fallback for names that come straight from an archive.
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


def median_step(depths):
    """Median spacing between consecutive depths carrying a real value."""
    if len(depths) < 3:
        return np.nan
    d = np.diff(np.sort(np.asarray(depths, dtype=float)))
    d = d[d > 0]
    return float(np.median(d)) if len(d) else np.nan


def censored_fraction(v):
    """Fraction of values sitting exactly on the column minimum.

    A detection-limit floor shows up as many values pinned to the minimum.
    Constant runs also drive rolling MAD to zero -> infinite z -> false flags.
    """
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 5:
        return 0.0
    return float((v <= v.min() + 1e-9).sum() / len(v))


def profile_core(path):
    df = pd.read_csv(path)
    name = os.path.basename(path).replace("_proxies.csv", "")

    depth_col = df.columns[0]
    depth = df[depth_col].astype(float)

    row = {
        "core": name,
        "n_rows": len(df),
        "depth_min_cm": round(float(depth.min()), 1),
        "depth_max_cm": round(float(depth.max()), 1),
        "length_cm": round(float(depth.max() - depth.min()), 1),
    }

    # ---- per-family sampling step (finest column within each family) -------
    fam_step, fam_depths, censored = {}, {}, []
    for col in df.columns[1:]:
        fam = family_of(col)
        if fam is None:
            continue
        d = depth[df[col].notna()].values
        if len(d) < 3:
            continue
        step = median_step(d)
        if np.isnan(step):
            continue
        if fam not in fam_step or step < fam_step[fam]:
            fam_step[fam] = step
            fam_depths[fam] = d
        if censored_fraction(df[col].values) >= 0.10:
            censored.append(col)

    fams = sorted(fam_step, key=lambda f: fam_step[f])
    row["n_families"] = len(fams)
    row["families"] = ";".join(fams)
    row["censored_cols"] = ";".join(censored)
    row["n_censored"] = len(censored)

    # binding constraint for TWO-family agreement = coarser of the two finest
    row["delta_best_cm"] = round(fam_step[fams[0]], 2) if fams else np.nan
    row["delta_2fam_cm"] = round(fam_step[fams[1]], 2) if len(fams) > 1 else np.nan

    # ---- co-location: can the two finest families actually agree? ---------
    coloc = np.nan
    if len(fams) > 1:
        a, b = fam_depths[fams[0]], fam_depths[fams[1]]
        b_sorted = np.sort(b)
        idx = np.searchsorted(b_sorted, a)
        near = []
        for ai, i in zip(a, idx):
            cand = []
            if i < len(b_sorted):
                cand.append(abs(b_sorted[i] - ai))
            if i > 0:
                cand.append(abs(b_sorted[i - 1] - ai))
            near.append(min(cand) if cand else np.inf)
        coloc = float(np.mean(np.array(near) <= DELTA_TOL))
    row["coloc_frac"] = round(coloc, 3) if coloc == coloc else np.nan

    # ---- tier admissibility + resolvable scale rungs ----------------------
    for tier, dmax in TIERS.items():
        d2 = row["delta_2fam_cm"]
        ok = (len(fams) >= 2 and d2 == d2 and d2 <= dmax
              and row["length_cm"] >= 50)
        row[f"adm_{tier}"] = bool(ok)

        if len(fams) >= 2 and d2 == d2:
            ladder = FIXED_LADDER_CM
            # The detector ladder is fixed across strict and relaxed tiers;
            # tiers differ only in admissibility (the paper explicitly states
            # that changing the ladder would create different detectors).
            res = sum(1 for w in ladder
                      if (w / d2) >= N_MIN and w <= row["length_cm"])
            row[f"rungs_{tier}"] = res
            row[f"ladder_{tier}"] = ";".join(str(int(w)) for w in ladder)
        else:
            row[f"rungs_{tier}"] = 0
            row[f"ladder_{tier}"] = ""

    # ---- reason for exclusion (strict tier), for the archive audit --------
    reasons = []
    if len(fams) < 2:
        reasons.append("<2 families")
    if row["length_cm"] < 50:
        reasons.append("core <50cm")
    d2 = row["delta_2fam_cm"]
    if len(fams) >= 2 and d2 == d2 and d2 > TIERS["strict_T2cm"]:
        reasons.append(f"delta_2fam={d2}cm")
    if coloc == coloc and coloc < 0.5:
        reasons.append(f"coloc={coloc:.2f}")
    row["exclude_reason_strict"] = "; ".join(reasons)

    return row


def main():
    if not os.path.isdir(IN_DIR):
        raise SystemExit(f"ERROR: '{IN_DIR}/' not found. "
                         f"Run this from the directory holding ev_clean/")

    files = sorted(glob.glob(os.path.join(IN_DIR, "*.csv")))
    if not files:
        raise SystemExit(f"ERROR: no CSVs in {IN_DIR}/")

    rows = []
    for f in files:
        try:
            rows.append(profile_core(f))
        except Exception as e:
            print(f"  !! {os.path.basename(f)}: {e}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_FILE, index=False)

    n = len(out)
    print(f"\nProfiled {n} cores -> {OUT_FILE}\n")
    for tier in TIERS:
        k = int(out[f"adm_{tier}"].sum())
        print(f"  {tier:16s} admits {k:3d}/{n} cores "
              f"({100*k/n:.0f}%)  Delta_max={TIERS[tier]} cm")
    print(f"\n  cores with censored columns : "
          f"{int((out.n_censored > 0).sum())}")
    print(f"  median delta_2fam (cm)      : "
          f"{out.delta_2fam_cm.median():.2f}")
    print(f"\nSend {OUT_FILE} back -- it is small and contains no measurements.")



# ---------------------------------------------------------------------------
# Consolidated orchestration
def harmonize_corpus():
    """Harmonize every bundle without deleting successful cores on rerun."""
    folders = sorted(d for d in glob.glob(os.path.join(CORES_DIR, "*")) if os.path.isdir(d))
    if not folders:
        raise SystemExit(f"No folders in {CORES_DIR}/. Run the download stage first.")
    global AUDIT
    AUDIT = []
    total = 0
    print(f"Harmonizing {len(folders)} bundles (need >={MIN_PROXIES} proxies, >={MIN_ROWS} depths)\n")
    for folder in folders:
        label = os.path.basename(folder)[:56]
        try:
            total += retry_call(f"harmonize {label}", process_bundle, folder)
        except Exception as exc:
            print(f"  !! harmonize failed after retries: {label}: {exc}")
            log(label, "bundle_fail", str(exc))
    atomic_write_csv(pd.DataFrame(AUDIT), AUDIT_FILE)
    print(f"Wrote/updated {total} cores into {OUT_DIR}/")
    return total

def profile_corpus():
    files = sorted(glob.glob(os.path.join(IN_DIR, "*.csv")))
    if not files:
        raise SystemExit(f"ERROR: no CSVs in {IN_DIR}/")
    rows = []
    for f in files:
        try:
            rows.append(retry_call(f"profile {os.path.basename(f)}", profile_core, f))
        except Exception as exc:
            print(f"  !! {os.path.basename(f)} failed after retries: {exc}")
    out = pd.DataFrame(rows)
    atomic_write_csv(out, OUT_FILE)
    for tier in TIERS:
        k = int(out[f"adm_{tier}"].sum()) if len(out) else 0
        print(f"{tier}: admits {k}/{len(out)} cores")
    return out

def download_keep(attempts=DEFAULT_RETRIES):
    """Download KEEP candidates with per-candidate retries and resume support."""
    if not os.path.exists(COVERAGE_FILE):
        raise SystemExit(f"ERROR: {COVERAGE_FILE} not found. Run the check stage first.")
    with open(COVERAGE_FILE, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    dois = [r["doi"] for r in rows if r.get("verdict", "").upper() == "KEEP"]
    if not dois:
        print(f"No KEEP rows in {COVERAGE_FILE}.")
        return {"downloaded": 0, "have": 0, "skip": 0, "failed": 0}
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    stats = {"downloaded": 0, "have": 0, "skip": 0, "failed": 0}
    for i, doi in enumerate(dois, 1):
        print(f"[{i}/{len(dois)}] {doi}", flush=True)
        try:
            def _one():
                status = process_download(doi)
                if status == "failed":
                    raise RuntimeError("candidate bundle did not complete")
                return status
            result = retry_call(f"download candidate {doi}", _one, attempts=attempts)
        except Exception as exc:
            print(f"  FAILED after {attempts} attempts: {doi}: {exc}")
            result = "failed"
        stats[result] = stats.get(result, 0) + 1
    print(f"downloaded={stats['downloaded']} had={stats['have']} skipped={stats['skip']} failed={stats['failed']}")
    return stats

# Imports used by the copied harmonisation/profile code.
import glob
import json
import numpy as np
import pandas as pd

def run_stage_with_retries(label, fn, *args, attempts=DEFAULT_RETRIES, **kwargs):
    return retry_call(label, fn, *args, attempts=attempts, **kwargs)

def run_pipeline(args):
    if args.stage == "search":
        return run_stage_with_retries("search stage", search_candidates,
                                      target=args.target_candidates,
                                      per_query_limit=args.per_query_limit,
                                      attempts=args.retries)
    if args.stage == "check":
        return run_stage_with_retries("check stage", check_coverage, attempts=args.retries)
    if args.stage == "download":
        return run_stage_with_retries("download stage", download_keep, attempts=args.retries)
    if args.stage == "harmonize":
        return run_stage_with_retries("harmonize stage", harmonize_corpus, attempts=args.retries)
    if args.stage == "profile":
        return run_stage_with_retries("profile stage", profile_corpus, attempts=args.retries)
    if args.stage == "all":
        # Each stage retries/resumes independently; download failure therefore
        # never forces search/check to start over.
        for stage in ["search", "check", "download", "harmonize", "profile"]:
            run_pipeline(argparse.Namespace(stage=stage,
                target_candidates=args.target_candidates,
                per_query_limit=args.per_query_limit,
                retries=args.retries))
        return True
    raise SystemExit(f"Unknown stage: {args.stage}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PRL consolidated data pipeline. Intermediate audit/checkpoint data live in .prl_work/."
    )
    parser.add_argument("stage", choices=["search", "check", "download", "harmonize", "profile", "all"])
    parser.add_argument("--target-candidates", type=int, default=TARGET_UNIQUE_CANDIDATES,
                        help="maximum UNIQUE PANGAEA candidate datasets across all queries (default: 1000)")
    parser.add_argument("--per-query-limit", type=int, default=PER_QUERY_LIMIT,
                        help="PANGAEA results requested per query (default: 150)")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES,
                        help="retries after a transient stage/record error (default: 3)")
    run_pipeline(parser.parse_args())

