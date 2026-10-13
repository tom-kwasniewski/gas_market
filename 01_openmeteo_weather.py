#!/usr/bin/env python3
"""
Download daily weather history from Open-Meteo and build one population-weighted
weather index PER COUNTRY, including Heating Degree Days (HDD).

Default countries: DE, NL, PL, IT, FR, BE (Germany, Netherlands, Poland, Italy,
France, Belgium) - the markets behind TTF, THE/EGSI and the Polish TGE.

Source: Open-Meteo Historical Weather API (archive-api.open-meteo.com), no API key.
Free for NON-COMMERCIAL use; data is CC BY 4.0, so credit it, e.g.
"Weather data by Open-Meteo.com". Do not use this in anything for your employer
without checking their commercial terms.

What it does
  1. Downloads daily data for every location (cities) of the chosen countries.
  2. Caches the raw per-city data in  data/raw/openmeteo/  (reruns on the same day
     do not call the API again).
  3. Computes HDD per city, then takes a weighted average across the cities of each
     country (weights = approximate population, or your own weights from a CSV).
  4. Writes two files to data/processed/:
       weather_by_country_long.csv   one row per date and country (tidy format)
       weather_by_country_wide.csv   one row per date, columns like PL_hdd, DE_temp_mean_c

HDD definition (Eurostat style):  HDD = base - T_mean  if T_mean <= threshold else 0
  defaults: base 18 C, threshold 15 C. Change with --hdd-base / --hdd-threshold.

Usage
    pip install pandas numpy requests
    python openmeteo_weather.py                              # all six countries
    python openmeteo_weather.py --start 2021-10-01
    python openmeteo_weather.py --countries PL,DE            # subset
    python openmeteo_weather.py --locations places.csv       # columns: country,name,lat,lon[,weight]

Notes
  * Archive data (ERA5-based reanalysis) lags real time by a few days, so the last
    days of a run may be empty. The script trims all countries to the last date that
    every country has, and tells you which date that is.
  * Output units: temperature C, wind m/s (daily max at 10 m), precipitation mm,
    sunshine hours.
  * City weights are rough population figures (thousands) and only a proxy for gas
    demand. For example, Italy's heating demand is concentrated in the north and
    France's Paris region is far larger than the city proper - edit the weights, or
    pass your own CSV, if you want a better proxy.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

API_URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY_VARS = [
    "temperature_2m_mean",
    "temperature_2m_min",
    "temperature_2m_max",
    "wind_speed_10m_max",
    "precipitation_sum",
    "sunshine_duration",  # seconds per day
]
TIMEZONE = "Europe/Berlin"  # all default countries share CET/CEST, so local days line up

# country, city, latitude, longitude, weight (approx. population in thousands - edit freely)
DEFAULT_LOCATIONS = [
    # Germany
    ("DE", "Berlin", 52.520, 13.405, 3700),
    ("DE", "Hamburg", 53.551, 9.994, 1900),
    ("DE", "Munich", 48.137, 11.575, 1500),
    ("DE", "Cologne", 50.938, 6.960, 1100),
    ("DE", "Frankfurt", 50.110, 8.682, 770),
    ("DE", "Stuttgart", 48.775, 9.182, 630),
    ("DE", "Dusseldorf", 51.227, 6.774, 620),
    ("DE", "Leipzig", 51.340, 12.375, 620),
    ("DE", "Dortmund", 51.514, 7.468, 590),
    ("DE", "Essen", 51.456, 7.012, 580),
    ("DE", "Bremen", 53.079, 8.802, 570),
    ("DE", "Dresden", 51.050, 13.738, 560),
    ("DE", "Hanover", 52.375, 9.732, 550),
    ("DE", "Nuremberg", 49.452, 11.077, 520),
    # Netherlands
    ("NL", "Amsterdam", 52.370, 4.895, 920),
    ("NL", "Rotterdam", 51.925, 4.479, 660),
    ("NL", "The Hague", 52.078, 4.288, 550),
    ("NL", "Utrecht", 52.090, 5.121, 360),
    ("NL", "Eindhoven", 51.442, 5.478, 240),
    # Poland
    ("PL", "Warsaw", 52.230, 21.012, 1860),
    ("PL", "Krakow", 50.065, 19.945, 800),
    ("PL", "Wroclaw", 51.108, 17.039, 670),
    ("PL", "Lodz", 51.759, 19.456, 660),
    ("PL", "Poznan", 52.407, 16.929, 540),
    ("PL", "Gdansk", 54.352, 18.646, 470),
    ("PL", "Katowice", 50.264, 19.023, 300),
    # Italy
    ("IT", "Rome", 41.903, 12.496, 2750),
    ("IT", "Milan", 45.464, 9.190, 1400),
    ("IT", "Naples", 40.852, 14.268, 910),
    ("IT", "Turin", 45.070, 7.686, 850),
    ("IT", "Bologna", 44.494, 11.343, 390),
    ("IT", "Verona", 45.438, 10.992, 260),
    # France
    ("FR", "Paris", 48.857, 2.352, 2100),
    ("FR", "Marseille", 43.296, 5.370, 870),
    ("FR", "Lyon", 45.764, 4.836, 520),
    ("FR", "Toulouse", 43.605, 1.444, 490),
    ("FR", "Lille", 50.629, 3.057, 230),
    # Belgium
    ("BE", "Brussels", 50.850, 4.352, 1200),
    ("BE", "Antwerp", 51.219, 4.402, 530),
    ("BE", "Ghent", 51.054, 3.722, 265),
    ("BE", "Liege", 50.633, 5.567, 200),
]


# --------------------------------------------------------------------------- #
# Locations
# --------------------------------------------------------------------------- #
def load_locations(path: Path | None, countries: list[str] | None) -> pd.DataFrame:
    if path is None:
        df = pd.DataFrame(DEFAULT_LOCATIONS, columns=["country", "name", "lat", "lon", "weight"])
    else:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        missing = {"country", "name", "lat", "lon"} - set(df.columns)
        if missing:
            raise SystemExit(f"{path} is missing columns: {sorted(missing)}")
        if "weight" not in df.columns:
            df["weight"] = 1.0
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0)
        df["country"] = df["country"].astype(str).str.strip().str.upper()
        df = df[["country", "name", "lat", "lon", "weight"]]

    if countries:
        unknown = sorted(set(countries) - set(df["country"]))
        if unknown:
            raise SystemExit(f"No locations for countries {unknown}. Available: {sorted(set(df['country']))}")
        df = df[df["country"].isin(countries)]
    if df.empty:
        raise SystemExit("No locations selected.")
    return df.reset_index(drop=True)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def _get_json(params: dict, retries: int = 4) -> dict:
    """GET with retry/backoff (separate function so it is easy to mock in tests)."""
    import requests

    wait = 5
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(API_URL, params=params, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            data = r.json()
            if r.status_code >= 400 or data.get("error"):
                # a real request error (bad variable, bad dates...): do not retry
                raise ValueError(f"Open-Meteo rejected the request: {data.get('reason', r.text[:200])}")
            return data
        except ValueError:
            raise
        except Exception as exc:
            last_err = exc
            if attempt < retries:
                print(f"    retry {attempt}/{retries - 1} after error: {exc}", file=sys.stderr)
                time.sleep(wait)
                wait *= 3
    raise RuntimeError(f"Open-Meteo request failed after {retries} attempts: {last_err}")


def fetch_city(country: str, name: str, lat: float, lon: float, start: str, end: str,
               raw_dir: Path, refresh: bool) -> pd.DataFrame:
    """Daily data for one location; cached per (country, city, start, end)."""
    cache = raw_dir / f"{country}_{slug(name)}_{start}_{end}.csv"
    label = f"{country} {name}"
    if cache.exists() and not refresh:
        print(f"  {label:<18} cached")
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": ",".join(DAILY_VARS),
        "timezone": TIMEZONE,          # aggregate to local calendar days
        "wind_speed_unit": "ms",
    }
    data = _get_json(params)
    daily = data.get("daily")
    if not daily or "time" not in daily:
        raise RuntimeError(f"Unexpected response for {label}: no 'daily' block")

    df = pd.DataFrame(daily)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()
    df.index.name = "date"
    for col in DAILY_VARS:
        if col not in df.columns:
            df[col] = np.nan
    df["sunshine_h"] = df["sunshine_duration"] / 3600.0
    df = df.drop(columns="sunshine_duration")

    raw_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache)
    print(f"  {label:<18} downloaded {len(df)} days")
    return df


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def add_hdd(df: pd.DataFrame, base: float, threshold: float) -> pd.DataFrame:
    t = df["temperature_2m_mean"]
    df = df.copy()
    df["hdd"] = np.where(t <= threshold, base - t, 0.0)
    df.loc[t.isna(), "hdd"] = np.nan
    return df


def weighted_index(frames: dict[str, pd.DataFrame], weights: dict[str, float]) -> pd.DataFrame:
    """Weighted mean across cities; weights renormalised over cities with data that day."""
    cols = {
        "temperature_2m_mean": "temp_mean_c",
        "temperature_2m_min": "temp_min_c",
        "temperature_2m_max": "temp_max_c",
        "wind_speed_10m_max": "wind_max_ms",
        "precipitation_sum": "precip_mm",
        "sunshine_h": "sunshine_h",
        "hdd": "hdd",
    }
    names = list(frames)
    out = pd.DataFrame(index=frames[names[0]].index)
    w = pd.Series({n: weights[n] for n in names})
    for src, dst in cols.items():
        vals = pd.concat({n: frames[n][src] for n in names}, axis=1)
        mask = vals.notna()
        num = (vals.fillna(0) * w).sum(axis=1)
        den = (mask * w).sum(axis=1)
        out[dst] = (num / den).where(den > 0)
    out["n_locations"] = pd.concat(
        {n: frames[n]["temperature_2m_mean"].notna() for n in names}, axis=1
    ).sum(axis=1)
    return out.round(3)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--start", default="2021-01-01", help="Start date YYYY-MM-DD (default 2021-01-01)")
    p.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: yesterday)")
    p.add_argument("--countries", default=None,
                   help="Comma-separated country codes (default: all, e.g. DE,NL,PL,IT,FR,BE)")
    p.add_argument("--locations", type=Path, help="CSV with columns country,name,lat,lon[,weight]")
    p.add_argument("--outdir", type=Path, default=Path("data"), help="Output folder (default ./data)")
    p.add_argument("--hdd-base", type=float, default=18.0, help="HDD base temperature in C (default 18)")
    p.add_argument("--hdd-threshold", type=float, default=15.0,
                   help="Heating threshold in C: at or below this temp HDD > 0 (default 15)")
    p.add_argument("--refresh", action="store_true", help="Ignore cached downloads")
    p.add_argument("--pause", type=float, default=1.0, help="Seconds between requests (default 1)")
    args = p.parse_args()

    end = args.end or (date.today() - timedelta(days=1)).isoformat()
    if pd.Timestamp(args.start) > pd.Timestamp(end):
        raise SystemExit("--start must not be after --end")

    wanted = [c.strip().upper() for c in args.countries.split(",")] if args.countries else None
    locs = load_locations(args.locations, wanted)
    raw_dir = args.outdir / "raw" / "openmeteo"
    proc_dir = args.outdir / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    countries = list(dict.fromkeys(locs["country"]))
    print(f"Downloading {len(locs)} locations in {len(countries)} countries "
          f"({', '.join(countries)}), {args.start} -> {end}")

    results: dict[str, pd.DataFrame] = {}
    n_done = 0
    for cc in countries:
        group = locs[locs["country"] == cc]
        frames: dict[str, pd.DataFrame] = {}
        for row in group.itertuples(index=False):
            if n_done:
                time.sleep(args.pause)
            n_done += 1
            try:
                df = fetch_city(cc, row.name, row.lat, row.lon, args.start, end, raw_dir, args.refresh)
            except Exception as exc:
                print(f"  {cc} {row.name:<14} FAILED: {exc}", file=sys.stderr)
                continue
            frames[row.name] = add_hdd(df, args.hdd_base, args.hdd_threshold)

        if not frames:
            print(f"WARNING: no data for {cc} - country skipped.", file=sys.stderr)
            continue
        failed = sorted(set(group["name"]) - set(frames))
        if failed:
            print(f"WARNING: {cc} continues without {failed}; weights renormalised.", file=sys.stderr)
        results[cc] = weighted_index(frames, dict(zip(group["name"], group["weight"])))

    if not results:
        raise SystemExit("No country could be downloaded.")

    # trim to the last date for which EVERY country has data (archive lag)
    last_valid = {}
    for cc, df in results.items():
        v = df["temp_mean_c"].dropna()
        if v.empty:
            print(f"WARNING: {cc} has no temperature data - dropped.", file=sys.stderr)
            continue
        last_valid[cc] = v.index.max()
    results = {cc: results[cc] for cc in last_valid}
    if not results:
        raise SystemExit("All temperature values are empty - nothing to save.")
    common_last = min(last_valid.values())
    if common_last < max(df.index.max() for df in results.values()):
        print(f"\nArchive is complete for all countries only up to {common_last.date()}; "
              f"later days were trimmed.")
    results = {cc: df.loc[:common_last] for cc, df in results.items()}

    long_df = pd.concat(
        {cc: df.reset_index() for cc, df in results.items()}, names=["country", "_"]
    ).reset_index(level="country").reset_index(drop=True)
    long_df = long_df[["date", "country"] + [c for c in long_df.columns if c not in ("date", "country")]]
    long_df = long_df.sort_values(["date", "country"])

    wide = pd.concat({cc: df for cc, df in results.items()}, axis=1)
    wide.columns = [f"{cc}_{col}" for cc, col in wide.columns]
    wide.index.name = "date"

    long_path = proc_dir / "weather_by_country_long.csv"
    wide_path = proc_dir / "weather_by_country_wide.csv"
    long_df.to_csv(long_path, index=False)
    wide.to_csv(wide_path)

    first = min(df.index.min() for df in results.values())
    print(f"\nSaved {len(wide)} days ({first.date()} -> {common_last.date()}) for "
          f"{len(results)} countries:\n  {long_path}\n  {wide_path}")

    miss = {cc: int(df.drop(columns='n_locations').isna().sum().sum()) for cc, df in results.items()}
    print("Missing values per country:", miss)

    temp = pd.DataFrame({cc: df["temp_mean_c"].groupby(df.index.year).mean() for cc, df in results.items()})
    hdd = pd.DataFrame({cc: df["hdd"].groupby(df.index.year).sum() for cc, df in results.items()})
    print("\nSanity check - mean temperature by year (C; first/last year may be partial):\n",
          temp.round(1).to_string())
    print("\nSanity check - HDD sum by year:\n", hdd.round(0).to_string())
    print("\nCredit: Weather data by Open-Meteo.com (CC BY 4.0).")


if __name__ == "__main__":
    main()