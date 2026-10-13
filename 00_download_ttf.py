#!/usr/bin/env python3
"""
Daily TTF gas price series from two free sources.

  * Yahoo Finance TTF=F  - front-month TTF future (via yfinance), long history.
  * EEX EGSI TTF         - EEX European Gas Spot Index (day-ahead style spot
                           index), from EEX's public market-data endpoint.

IMPORTANT about EGSI
  The free EEX endpoint serves only about the last ~45 days. This script
  therefore ACCUMULATES it: every run merges the fresh pull into
  data/raw/egsi_old_energy_map.csv, so your own history grows over time (run it
  daily, e.g. with GitHub Actions). If you obtain older EGSI values elsewhere,
  seed them with --egsi-seed (CSV: date,price).

  The endpoint is the one EEX's own website uses; it is not a documented public
  API and EEX can change it or restrict it. Check EEX's licensing/terms before
  republishing the data, and credit EEX as the source.

Usage:
    pip install pandas yfinance requests
    python ttf_prices.py
    python ttf_prices.py --egsi-seed egsi_old_energy_map.csv

Output: data/processed/ttf_daily.csv with columns
    ttf_front_month, ttf_is_filled, egsi, egsi_is_filled
EGSI is indexed by DELIVERY day. It is determined the evening before, so when
you build forecasting features treat it as information known on day D-1.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

YAHOO_TICKER = "TTF=F"
EEX_URL = "https://api.eex-group.com/pub/market-data/table-data"
EEX_HEADERS = {
    "Accept": "application/json",
    "Origin": "https://www.eex.com",
    "Referer": "https://www.eex.com/",
    "User-Agent": "Mozilla/5.0 (compatible; personal-research-script)",
}
FILL_LIMIT_DAYS = 4  # forward-fill weekends/holidays only, never long gaps


# --------------------------------------------------------------------------- #
# Yahoo
# --------------------------------------------------------------------------- #
def load_yahoo(start: str, cache: Path) -> pd.Series:
    """Download TTF=F daily closes; fall back to the local cache if offline."""
    try:
        import yfinance as yf

        raw = yf.download(YAHOO_TICKER, start=start, auto_adjust=False, progress=False)
        if raw is None or raw.empty:
            raise RuntimeError("Yahoo returned no data")
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):  # newer yfinance: MultiIndex columns
            close = close.iloc[:, 0]
        s = close.dropna().astype(float)
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        s.name = "ttf_front_month"
        cache.parent.mkdir(parents=True, exist_ok=True)
        s.to_csv(cache, header=True)
        print(f"[yahoo] downloaded {len(s)} rows, cached to {cache}")
        return s
    except Exception as exc:  # network, rate limit, yfinance change, ...
        print(f"[yahoo] download failed: {exc}", file=sys.stderr)
        if cache.exists():
            print(f"[yahoo] using cached copy {cache}", file=sys.stderr)
            s = pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
            s.name = "ttf_front_month"
            return s
        raise SystemExit("No Yahoo data available (download failed and no cache).")


# --------------------------------------------------------------------------- #
# EEX EGSI
# --------------------------------------------------------------------------- #
def _eex_get(params: dict) -> dict:
    """One HTTP call to the EEX table-data endpoint (separate for easy mocking)."""
    import requests

    r = requests.get(EEX_URL, params=params, headers=EEX_HEADERS, timeout=30)
    if r.status_code == 401:
        raise PermissionError("EEX returned 401 (this product/period may need a subscription)")
    r.raise_for_status()
    if "json" not in (r.headers.get("content-type") or "").lower():
        raise ValueError(f"EEX returned non-JSON content: {r.text[:200]!r}")
    return r.json()


def _parse_eex(payload: dict) -> pd.Series:
    """Turn an EEX table-data payload into a Series indexed by delivery day."""
    header = payload.get("header") or []
    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError("Unexpected EEX response shape (no 'data' list)")

    d_idx = header.index("deliveryDay") if "deliveryDay" in header else 5
    p_idx = next((header.index(c) for c in ("settlPx", "close", "lastPrice") if c in header), 4)

    values: dict[pd.Timestamp, float] = {}
    for row in data:  # served newest first, so setdefault keeps the latest value
        try:
            day = pd.Timestamp(str(row[d_idx])[:10])
            px = row[p_idx]
        except (IndexError, TypeError, ValueError):
            continue
        if px is None:
            continue
        values.setdefault(day, float(px))

    s = pd.Series(
        list(values.values()), index=pd.DatetimeIndex(list(values.keys())),
        name="egsi", dtype=float,
    )
    return s.sort_index()


def fetch_egsi(lookback_days: int) -> pd.Series:
    """Pull EGSI TTF Day (required) and Weekend (best effort) from EEX."""
    start = date.today() - timedelta(days=lookback_days)
    end = date.today() + timedelta(days=2)
    parts = []
    for kind in ("Day", "Weekend"):
        params = {
            "shortCode": f"EEX EGSI TTF {kind}",
            "commodity": "NATGAS",
            "pricing": "I",
            "area": "TTF",
            "product": "EGSI",
            "isRolling": "true",
            "startDate": start.isoformat(),
            "endDate": end.isoformat(),
        }
        try:
            parts.append(_parse_eex(_eex_get(params)))
        except Exception as exc:
            if kind == "Day":
                raise
            print(f"[egsi] weekend product skipped: {exc}", file=sys.stderr)

    s = pd.concat(parts)
    s = s[~s.index.duplicated(keep="first")].sort_index()
    s.name = "egsi"
    return s


def read_history(path: Path) -> pd.Series:
    if not path.exists():
        return pd.Series(dtype=float, name="egsi", index=pd.DatetimeIndex([]))
    s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0].astype(float)
    s.name = "egsi"
    return s


def merge_into_history(new: pd.Series, path: Path) -> pd.Series:
    """Merge new values into the stored history (new values win on conflicts)."""
    combined = pd.concat([read_history(path), new])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined.name = "egsi"
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, header=True, index_label="date")
    return combined


def load_seed(path: Path) -> pd.Series:
    """Read a manual EGSI CSV: first column = date (ISO, YYYY-MM-DD), second = price."""
    df = pd.read_csv(path, sep=None, engine="python")
    dates = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    prices = pd.to_numeric(df.iloc[:, 1].astype(str).str.replace(",", ".", regex=False),
                           errors="coerce")
    s = pd.Series(prices.to_numpy(), index=dates, name="egsi").dropna()
    s = s[~s.index.isna()]
    if s.empty:
        raise ValueError(f"No valid rows in seed file {path}")
    return s


# --------------------------------------------------------------------------- #
# Combine and report
# --------------------------------------------------------------------------- #
def build_daily(yahoo: pd.Series, egsi: pd.Series) -> pd.DataFrame:
    """Daily calendar with both series; short gaps forward-filled and flagged."""
    last = max(yahoo.index.max(), egsi.index.max()) if len(egsi) else yahoo.index.max()
    cal = pd.date_range(yahoo.index.min(), last, freq="D")
    out = pd.DataFrame(index=cal)
    out.index.name = "date"

    for col, s in (("ttf_front_month", yahoo), ("egsi", egsi)):
        raw = s.reindex(cal)
        filled = raw.ffill(limit=FILL_LIMIT_DAYS)
        filled[cal > s.index.max()] = float("nan")  # never invent values past the last real one
        filled[cal < s.index.min()] = float("nan")
        flag = "ttf_is_filled" if col == "ttf_front_month" else "egsi_is_filled"
        out[col] = filled.round(3)
        out[flag] = raw.isna() & filled.notna()
    return out[["ttf_front_month", "ttf_is_filled", "egsi", "egsi_is_filled"]]


def report(yahoo: pd.Series, egsi: pd.Series) -> None:
    """Compare EGSI(D) with the previous day's TTF=F close (EGSI is set on D-1)."""
    print("\n=== EGSI vs Yahoo TTF=F ===")
    if egsi.empty:
        print("No EGSI data yet.")
        return
    print(f"EGSI history: {egsi.index.min().date()} -> {egsi.index.max().date()} "
          f"({len(egsi)} days)")

    cal = pd.date_range(yahoo.index.min(), yahoo.index.max(), freq="D")
    prev_close = yahoo.reindex(cal).ffill().shift(1)  # last close available on D-1
    both = pd.concat([egsi, prev_close.rename("ttf_prev")], axis=1, join="inner").dropna()
    if len(both) < 5:
        print("Too few overlapping days to compare.")
        return
    diff = both["egsi"] - both["ttf_prev"]
    print(f"Overlap days:       {len(both)}")
    print(f"Mean diff (E - Y):  {diff.mean():+.2f} EUR/MWh")
    print(f"MAE:                {diff.abs().mean():.2f} EUR/MWh")
    print(f"MAPE:               {(diff.abs() / both['ttf_prev']).mean() * 100:.2f} %")
    print(f"Level correlation:  {both['egsi'].corr(both['ttf_prev']):.4f}")
    print("Note: EGSI is a spot (day-ahead) index and TTF=F a front-month future, "
          "so a persistent gap is normal.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--start", default="2021-01-01", help="Yahoo start date (default 2021-01-01)")
    p.add_argument("--outdir", type=Path, default=Path("data"), help="Output folder (default ./data)")
    p.add_argument("--egsi-lookback", type=int, default=50,
                   help="Days of EGSI to request from EEX (endpoint serves ~45; default 50)")
    p.add_argument("--egsi-seed", type=Path,
                   help="Optional CSV (date,price) with older EGSI values to add to the history")
    args = p.parse_args()

    raw_dir = args.outdir / "raw"
    proc_dir = args.outdir / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)
    hist_path = raw_dir / "egsi_old_energy_map.csv"

    yahoo = load_yahoo(args.start, raw_dir / "yahoo_ttf.csv")

    if args.egsi_seed:
        seed = load_seed(args.egsi_seed)
        merge_into_history(seed, hist_path)
        print(f"[egsi] seeded {len(seed)} rows from {args.egsi_seed}")

    try:
        fresh = fetch_egsi(args.egsi_lookback)
        egsi = merge_into_history(fresh, hist_path)
        print(f"[egsi] fetched {len(fresh)} days from EEX; history now {len(egsi)} days "
              f"({egsi.index.min().date()} -> {egsi.index.max().date()})")
    except Exception as exc:
        print(f"[egsi] EEX fetch failed: {exc}", file=sys.stderr)
        egsi = read_history(hist_path)
        if egsi.empty:
            print("[egsi] no stored history either - output will have an empty EGSI column.",
                  file=sys.stderr)
        else:
            print(f"[egsi] using stored history ({len(egsi)} days).", file=sys.stderr)

    report(yahoo, egsi)

    daily = build_daily(yahoo, egsi)
    out_path = proc_dir / "ttf_egsi_daily.csv"
    daily.to_csv(out_path)
    print(f"\nSaved {len(daily)} daily rows ({daily.index.min().date()} -> "
          f"{daily.index.max().date()}) to {out_path}")
    print("Credit sources in your README: Yahoo Finance (TTF=F) and EEX (EGSI).")


if __name__ == "__main__":
    main()