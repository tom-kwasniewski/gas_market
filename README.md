# Gas Market Data

Python scripts for collecting TTF gas prices and weather indicators for
European gas-market analysis. Both scripts write their data beneath `data/`.

## Requirements

- Python 3.10 or newer
- Internet access to Yahoo Finance, EEX, and/or Open-Meteo, depending on which
  scripts you run

Create and activate a virtual environment, then install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Scripts

### TTF prices and EGSI

Run:

```powershell
python 00_download_ttf.py
```

This downloads Yahoo Finance's `TTF=F` front-month futures prices and EEX's
European Gas Spot Index (EGSI) for TTF. Yahoo prices are cached in
`data/raw/yahoo_ttf.csv`. EGSI values are accumulated in
`data/raw/egsi_old_energy_map.csv`, because the EEX endpoint only serves a
limited recent history. If the EEX request fails, the script uses the stored
EGSI history when available.

The daily output is `data/processed/ttf_egsi_daily.csv`, with columns:

- `ttf_front_month` and `egsi`: price series
- `ttf_is_filled` and `egsi_is_filled`: flags for short gaps filled from the
  preceding value

EGSI is indexed by delivery day and is determined the evening before. Treat it
as information available on the preceding day when building forecasting
features.

Options:

```text
--start DATE              Yahoo history start date (default: 2021-01-01)
--outdir PATH             Data directory (default: data)
--egsi-lookback DAYS      EEX request lookback (default: 50)
--egsi-seed PATH          Add older EGSI values from a CSV to the history
```

Seed CSVs should have a date column and a price column, for example:

```csv
date,egsi
2021-10-01,90.269
2021-10-02,87.915
```

The seed file's first two columns are read. Newer values replace stored values
for duplicate dates.

### Weather indicators

Run for all configured locations and countries:

```powershell
python 01_openmeteo_weather.py
```

This downloads daily Open-Meteo archive data for configured cities, calculates
population-weighted country averages, and derives heating degree days (HDD).
Raw city data is cached under `data/raw/openmeteo/`; cached files are reused
unless `--refresh` is specified.

The script creates:

- `data/processed/weather_by_country_long.csv`: one row per date and country
- `data/processed/weather_by_country_wide.csv`: one row per date, with
  country-prefixed columns

The weather outputs include weighted mean/minimum/maximum temperature, maximum
wind speed, precipitation, sunshine hours, HDD, and the number of locations
with temperature data. HDD uses a base of 18 °C and a threshold of 15 °C by
default. Recent dates may be missing because the archive data is delayed.

Examples:

```powershell
python 01_openmeteo_weather.py --start 2021-10-01
python 01_openmeteo_weather.py --countries PL,DE
python 01_openmeteo_weather.py --start 2021-10-01 --end 2025-12-31
python 01_openmeteo_weather.py --locations places.csv
python 01_openmeteo_weather.py --refresh
```

Custom location CSVs require `country,name,lat,lon` columns and may include a
`weight` column. If omitted, each location has equal weight.

Options:

```text
--start DATE              Start date (default: 2021-01-01)
--end DATE                End date (default: yesterday)
--countries CODES         Comma-separated country codes
--locations PATH          Custom location CSV
--outdir PATH             Data directory (default: data)
--hdd-base NUMBER         HDD base temperature in °C (default: 18)
--hdd-threshold NUMBER    HDD threshold in °C (default: 15)
--refresh                 Redownload data instead of using cached files
--pause SECONDS           Delay between requests (default: 1)
```

## Data and version control

Downloaded Open-Meteo city caches, the Yahoo cache, generated processed
outputs, the virtual environment, and Python bytecode are excluded by
`.gitignore`. EGSI history files are not ignored, so they can be retained or
shared as project inputs.

## Data sources and usage

- **Yahoo Finance:** TTF futures data for ticker `TTF=F`. Follow Yahoo Finance's
  terms for access and redistribution.
- **EEX:** EGSI data from EEX's public market-data endpoint. The endpoint is
  undocumented and may change or restrict access. Check EEX's licensing and
  terms before redistributing the data; credit EEX as the source.
- **Open-Meteo:** Historical weather data, licensed CC BY 4.0. Credit
  “Weather data by Open-Meteo.com.” The free service is for non-commercial use;
  check its current terms before commercial use.
