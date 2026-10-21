#!/usr/bin/env python3
"""
Download daily gas storage (AGSI+) and LNG terminal (ALSI) data from Gas
Infrastructure Europe (GIE) for several countries plus the EU aggregate.

API facts (GIE user manual v007, Oct 2022):
  * Endpoints:  https://agsi.gie.eu/api   and   https://alsi.gie.eu/api
  * Free, but you must register on the website to get a personal API key
    (one key works for both platforms if you tick both when registering).
  * The key goes in the HTTP header  x-key.
  * Results are paginated: size is capped at 300 and every response has last_page.
  * Rate limit: more than 60 calls per minute puts you in a 60-second waiting queue,
    and ignoring that can get an account blacklisted. This script waits between calls.
  * Data for a gas day is published the following evening (19:30 and 23:00 CET),
    and operators may correct history retroactively - so re-download rather than
    trusting an old copy.
  * Credit GIE as the data source (mandatory), e.g. "Source: GIE AGSI/ALSI".

Usage
    pip install pandas requests
    export GIE_API_KEY=your_key_here            # Windows: set GIE_API_KEY=your_key_here
    python gie_agsi_alsi.py                     # DE,NL,PL,IT,FR,BE + EU, 2021-01-01 -> today
    python gie_agsi_alsi.py --start 2021-10-01 --countries PL,DE
    python gie_agsi_alsi.py --only agsi         # storage only (or: --only alsi)

Output
    data/raw/gie/<agsi|alsi>_<COUNTRY>.csv      per country, all fields
    data/processed/agsi_storage_daily.csv       tidy: one row per date and country
    data/processed/alsi_lng_daily.csv           tidy: one row per date and country
    data/processed/gie_wide.csv                 key columns, one row per date
                                                (e.g. DE_storage_full_pct, PL_lng_sendout_gwh)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

BASES = {"agsi": "https://agsi.gie.eu/api", "alsi": "https://alsi.gie.eu/api"}
DEFAULT_COUNTRIES = ["DE", "NL", "PL", "IT", "FR", "BE"]
PAGE_SIZE = 300  # API maximum

# raw field -> output column
AGSI_FIELDS = {
    "gasInStorage": "storage_twh",
    "injection": "injection_gwh",
    "withdrawal": "withdrawal_gwh",
    "netWithdrawal": "net_withdrawal_gwh",
    "workingGasVolume": "wgv_twh",
    "injectionCapacity": "inj_capacity_gwh",
    "withdrawalCapacity": "wd_capacity_gwh",
    "full": "full_pct",
    "trend": "trend_pct",
    "consumption": "consumption_twh",
    "consumptionFull": "consumption_full_pct",
}
ALSI_FIELDS = {
    "sendOut": "lng_sendout_gwh",
    "dtmi": "lng_dtmi_1000m3",
    "dtrs": "lng_dtrs_gwh",
}


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def _get_json(url: str, params: dict, key: str, retries: int = 4) -> dict:
    """GET with the x-key header, retry on rate limits / server errors."""
    import requests

    wait = 60
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, headers={"x-key": key}, timeout=60)
            if r.status_code == 429:
                raise RuntimeError("HTTP 429 (rate limit)")
            if r.status_code >= 500:
                raise RuntimeError(f"HTTP {r.status_code}")
            if r.status_code in (401, 403):
                raise PermissionError(
                    f"HTTP {r.status_code}: key rejected. Check GIE_API_KEY and that your "
                    f"account has access to this platform (AGSI and/or ALSI)."
                )
            r.raise_for_status()
            return r.json()
        except PermissionError:
            raise
        except Exception as exc:
            last_err = exc
            if attempt < retries:
                print(f"    retry {attempt}/{retries - 1} in {wait}s ({exc})", file=sys.stderr)
                time.sleep(wait)
    raise RuntimeError(f"GIE request failed after {retries} attempts: {last_err}")


def fetch_scope(db: str, key: str, scope: str, start: str, end: str, pause: float) -> list[dict]:
    """All pages for one scope: a country code like 'DE', or 'EU' for the aggregate."""
    params = {"from": start, "to": end, "size": PAGE_SIZE, "page": 1}
    if scope.upper() == "EU":
        params["type"] = "eu"
    else:
        params["country"] = scope.lower()

    rows: list[dict] = []
    page, last_page = 1, 1
    while page <= last_page:
        params["page"] = page
        resp = _get_json(BASES[db], params, key)
        if not isinstance(resp, dict) or "data" not in resp:
            raise RuntimeError(f"Unexpected response (no 'data'); likely a wrong or missing key: {str(resp)[:150]}")
        rows.extend(resp["data"])
        last_page = int(resp.get("last_page") or 1)
        page += 1
        if page <= last_page:
            time.sleep(pause)
    return rows


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace({"-": np.nan, "": np.nan}), errors="coerce")


def normalise(db: str, scope: str, rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["gasDayStart"], errors="coerce")
    out = pd.DataFrame({"date": df["date"], "country": scope.upper()})

    if db == "agsi":
        for raw, col in AGSI_FIELDS.items():
            out[col] = _num(df[raw]) if raw in df.columns else np.nan
        out["status"] = df["status"] if "status" in df.columns else np.nan  # E/C/N
    else:
        inv = df["inventory"] if "inventory" in df.columns else pd.Series([None] * len(df))
        if inv.map(lambda v: isinstance(v, dict)).any():  # inventory may be {lng, gwh}
            out["lng_inventory_1000m3"] = _num(inv.map(lambda v: v.get("lng") if isinstance(v, dict) else np.nan))
            out["lng_inventory_gwh"] = _num(inv.map(lambda v: v.get("gwh") if isinstance(v, dict) else np.nan))
        else:
            out["lng_inventory_1000m3"] = _num(inv)
            out["lng_inventory_gwh"] = np.nan
        for raw, col in ALSI_FIELDS.items():
            out[col] = _num(df[raw]) if raw in df.columns else np.nan

    out = out.dropna(subset=["date"]).sort_values("date")
    # keep the last row if the API ever returns a date twice
    return out.drop_duplicates("date", keep="last").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--api-key", default=os.environ.get("GIE_API_KEY"),
                   help="GIE API key (or set the GIE_API_KEY environment variable)")
    p.add_argument("--start", default="2021-01-01", help="Start gas day YYYY-MM-DD (default 2021-01-01)")
    p.add_argument("--end", default=date.today().isoformat(), help="End gas day (default today)")
    p.add_argument("--countries", default=",".join(DEFAULT_COUNTRIES),
                   help="Comma-separated country codes; EU aggregate is always added")
    p.add_argument("--only", choices=["agsi", "alsi"], help="Download only one platform")
    p.add_argument("--outdir", type=Path, default=Path("data"), help="Output folder (default ./data)")
    p.add_argument("--pause", type=float, default=1.2,
                   help="Seconds between API calls (default 1.2; the limit is 60 calls/min)")
    args = p.parse_args()

    if not args.api_key:
        raise SystemExit("No API key. Register at https://agsi.gie.eu (API page) and set GIE_API_KEY.")
    if args.pause < 1.0:
        print("WARNING: pause below 1 s risks the 60 calls/min limit.", file=sys.stderr)

    scopes = [c.strip().upper() for c in args.countries.split(",") if c.strip()]
    scopes = [s for s in scopes if s != "EU"] + ["EU"]
    platforms = [args.only] if args.only else ["agsi", "alsi"]

    raw_dir = args.outdir / "raw" / "gie"
    proc_dir = args.outdir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    tidy: dict[str, pd.DataFrame] = {}
    n_calls = 0
    for db in platforms:
        frames = []
        print(f"\n{db.upper()}: {len(scopes)} scopes, {args.start} -> {args.end}")
        for scope in scopes:
            if n_calls:
                time.sleep(args.pause)
            try:
                rows = fetch_scope(db, args.api_key, scope, args.start, args.end, args.pause)
            except PermissionError as exc:
                raise SystemExit(str(exc))
            except Exception as exc:
                print(f"  {scope}: FAILED - {exc}", file=sys.stderr)
                n_calls += 1
                continue
            n_calls += 1
            df = normalise(db, scope, rows)
            if df.empty:
                print(f"  {scope}: no data returned")
                continue
            df.to_csv(raw_dir / f"{db}_{scope}.csv", index=False)
            frames.append(df)
            print(f"  {scope}: {len(df)} days ({df['date'].min().date()} -> {df['date'].max().date()})")
        if frames:
            tidy[db] = pd.concat(frames, ignore_index=True).sort_values(["date", "country"])

    if not tidy:
        raise SystemExit("Nothing was downloaded.")

    names = {"agsi": "agsi_storage_daily.csv", "alsi": "alsi_lng_daily.csv"}
    for db, df in tidy.items():
        df.to_csv(proc_dir / names[db], index=False)

    # wide file with the columns most useful for modelling
    pieces = []
    if "agsi" in tidy:
        a = tidy["agsi"].pivot(index="date", columns="country",
                               values=["storage_twh", "full_pct", "net_withdrawal_gwh", "wgv_twh"])
        a.columns = [f"{cc}_{v}" for v, cc in a.columns]
        pieces.append(a)
    if "alsi" in tidy:
        l = tidy["alsi"].pivot(index="date", columns="country",
                               values=["lng_sendout_gwh", "lng_inventory_1000m3"])
        l.columns = [f"{cc}_{v}" for v, cc in l.columns]
        pieces.append(l)
    wide = pd.concat(pieces, axis=1).sort_index()
    wide.index.name = "date"
    wide.to_csv(proc_dir / "gie_wide.csv")

    print(f"\nSaved to {proc_dir}: " + ", ".join([names[d] for d in tidy] + ["gie_wide.csv"]))
    print(f"Wide file: {wide.shape[0]} days x {wide.shape[1]} columns, "
          f"{wide.index.min().date()} -> {wide.index.max().date()}")
    if "agsi" in tidy:
        last = tidy["agsi"][tidy["agsi"]["date"] == tidy["agsi"]["date"].max()]
        print("\nLatest storage fill (%):", last.set_index("country")["full_pct"].round(1).to_dict())
        est = (tidy["agsi"]["status"] == "E").sum()
        print(f"Rows flagged 'E' (estimated, may be revised): {est}")
    print("\nSource: GIE (Gas Infrastructure Europe) AGSI/ALSI - credit this in your README.")


if __name__ == "__main__":
    main()