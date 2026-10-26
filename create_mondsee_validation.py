import glob
import os
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# Find Mondsee geochemistry file
# ------------------------------------------------------------
files = glob.glob(
    r"ev_cores\818922_*\*818921*.csv"
)

if not files:
    raise SystemExit("Mondsee 818921 geochemistry file not found.")

src = files[0]

print("Source:")
print(src)

df = pd.read_csv(src)

# ------------------------------------------------------------
# Required columns
# ------------------------------------------------------------
required = ["Depth sed", "Ca", "Ti"]

missing = [c for c in required if c not in df.columns]

if missing:
    raise SystemExit(
        f"Missing required columns: {missing}\n"
        f"Available columns: {df.columns.tolist()}"
    )

# ------------------------------------------------------------
# Numeric conversion
# ------------------------------------------------------------
depth = pd.to_numeric(df["Depth sed"], errors="coerce")
ca = pd.to_numeric(df["Ca"], errors="coerce")
ti = pd.to_numeric(df["Ti"], errors="coerce")

# ------------------------------------------------------------
# Calculate Ca/Ti
# ------------------------------------------------------------
ratio = ca / ti.replace(0, np.nan)

out = pd.DataFrame({
    "depth": depth,
    "geochem_Ca_Ti": ratio
})

out = out.dropna(subset=["depth", "geochem_Ca_Ti"])
out = out.sort_values("depth").drop_duplicates("depth")

# ------------------------------------------------------------
# Convert depth to cm when it appears to be in metres.
# This follows the core-wide depth rule used by the pipeline.
# ------------------------------------------------------------
max_depth = float(out["depth"].max())

if max_depth <= 30:
    print(f"Depth maximum = {max_depth:.3f}; assuming metres and converting to cm.")
    out["depth"] = out["depth"] * 100.0
else:
    print(f"Depth maximum = {max_depth:.3f}; keeping depth as cm.")

# ------------------------------------------------------------
# Write special Mondsee validation file
# ------------------------------------------------------------
os.makedirs("ev_clean", exist_ok=True)

output = r"ev_clean\818922_Mondsee_Mo05_proxies.csv"

out.to_csv(output, index=False)

print()
print("Created:")
print(output)
print(f"Rows: {len(out)}")
print(f"Depth range: {out.depth.min():.2f} - {out.depth.max():.2f} cm")
print("Columns:", out.columns.tolist())