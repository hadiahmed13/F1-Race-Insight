#!/usr/bin/env python
"""
Download lap-by-lap timing data for five F1 seasons with FastF1,
add a binary safety-car flag, and save to raw_laps.parquet.

Requires:
    pip install --upgrade fastf1 pandas pyarrow
"""

import pathlib, sys
import pandas as pd
import fastf1 as ff

# --------------------------------------------------------------------- #
# CONFIG                                                                #
# --------------------------------------------------------------------- #
SEASONS = range(2020, 2025)                # 2020–2024 inclusive
CACHE_DIR = pathlib.Path("cache")          # where .ff1pkl files live
OUT_FILE  = pathlib.Path("raw_laps.parquet")

# --------------------------------------------------------------------- #
# INIT                                                                  #
# --------------------------------------------------------------------- #
CACHE_DIR.mkdir(exist_ok=True)
ff.Cache.enable_cache(CACHE_DIR)
print(f"FastF1 {ff.__version__} • cache → {CACHE_DIR.resolve()}")
print("-" * 60)

all_laps: list[pd.DataFrame] = []

def add_sc_flag(df: pd.DataFrame) -> pd.DataFrame:
    """SC = 1 if TrackStatus string contains code '4' (Safety Car)."""
    df["SC"] = (
        df["TrackStatus"].fillna("")        # may be NaN
        .astype(str)
        .str.contains("4")
        .astype(int)
    )
    return df

# --------------------------------------------------------------------- #
# MAIN LOOP                                                             #
# --------------------------------------------------------------------- #
for year in SEASONS:
    try:
        schedule = ff.get_event_schedule(year, include_testing=False)
    except Exception as e:
        print(f"⚠️  Could not fetch schedule for {year}: {e}")
        continue

    for _, evt in schedule.iterrows():
        try:
            sess = ff.get_session(year, evt["RoundNumber"], "R")
            sess.load()                                   # download or read cache

            # if FastF1 says “0 drivers” it means no timing data → skip
            if not getattr(sess, "drivers", None):
                raise RuntimeError("no timing data")

            laps = sess.laps.copy()                       # will raise DataNotLoadedError if empty
        except Exception as e:
            print(f"⚠️  {year} {evt['EventName']} – skipped ({e})")
            continue

        # ---- add our safety-car flag & meta columns ----
        laps["SC"] = laps["TrackStatus"].astype(str).str.contains("4").astype(int)
        laps["Season"] = year
        laps["EventName"] = evt["EventName"]
        all_laps.append(laps)

        print(f"✓  {year} {evt['EventName']} ({len(laps)} laps)")

# --------------------------------------------------------------------- #
# SAVE                                                                  #
# --------------------------------------------------------------------- #
if not all_laps:
    print("No sessions downloaded – nothing to save.")
    sys.exit(1)

df_final = pd.concat(all_laps, ignore_index=True)
df_final.to_parquet(OUT_FILE, index=False)
print("-" * 60)
print(f"✅  Saved {len(df_final):,} rows → {OUT_FILE.resolve()}")
