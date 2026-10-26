"""Download the two published Lake Mondsee Mo05 validation inputs from PANGAEA.

Run from the root of the PRL project in the active (prl) environment:
    python download_mondsee_inputs.py

Outputs are placed under ev_cores/818922_Mondsee_Mo05/ so that the existing
create_mondsee_validation.py and validation.py scripts can find them.
"""
from pathlib import Path
import re
import sys

try:
    from pangaeapy import PanDataSet
except ImportError as exc:
    raise SystemExit(
        "pangaeapy is not available in this Python environment. Run:\n"
        "  python -m pip install pangaeapy"
    ) from exc

OUT_DIR = Path("ev_cores") / "818922_Mondsee_Mo05"
DATASETS = {
    "10.1594/PANGAEA.818921": "geochemistry input (Ca and Ti)",
    "10.1594/PANGAEA.818919": "published layer-thickness reference data",
}


def safe_name(value: str, max_len: int = 100) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", value or "dataset").strip("_")
    return (value[:max_len].rstrip("_") or "dataset")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for doi, purpose in DATASETS.items():
        print(f"Downloading {doi} ({purpose}) ...", flush=True)
        try:
            ds = PanDataSet(f"doi:{doi}", include_data=True, enable_cache=True)
            df = getattr(ds, "data", None)
            if df is None or df.empty:
                raise RuntimeError("PANGAEA returned no tabular rows")
            numeric_id = doi.rsplit(".", 1)[-1]
            title = safe_name(getattr(ds, "title", ""))
            destination = OUT_DIR / f"{numeric_id}_{title}.csv"
            df.to_csv(destination, index=False)
            print(f"  Saved: {destination} ({len(df)} rows)")
            print(f"  Columns: {', '.join(map(str, df.columns))}")
        except Exception as exc:
            raise SystemExit(
                f"Could not download {doi}: {exc}\n"
                "Check your internet connection and try again."
            ) from exc

    geochem_files = list(OUT_DIR.glob("818921_*.csv"))
    layer_files = list(OUT_DIR.glob("818919_*.csv"))
    if not geochem_files or not layer_files:
        raise SystemExit(
            "Download finished, but the expected 818921 geochemistry file or "
            "818919 layer-thickness file is missing from the output folder."
        )
    print("\nMondsee inputs are ready for the existing project scripts.")
    print("Next run: python create_mondsee_validation.py")
    print("Then run: python validation.py mondsee")


if __name__ == "__main__":
    main()
