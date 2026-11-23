"""Download hourly weather for the four corridors from Open-Meteo (ERA5 reanalysis).

Needs data/sensors_100.csv. Writes data/weather_hourly.csv with columns
time, precip_mm, temp_c, wind, corridor (26,280 rows per corridor: 2017-2019, hourly).

One point per corridor: the mean latitude and longitude of its 25 sensors, rounded to 3 decimals.
Times are local (America/Los_Angeles), the same clock as the sensor timestamps.
precip_mm at hour h is the sum over the hour that ENDS at h (Open-Meteo convention), so the
project's weather feature uses the floor-hour of the forecast origin and never looks ahead.

    python scripts/fetch_weather.py --sensors data/sensors_100.csv --out data/weather_hourly.csv
"""
from __future__ import annotations

import argparse
import time

import pandas as pd
import requests

URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch(sensors: str, out: str) -> pd.DataFrame:
    s = pd.read_csv(sensors)
    pts = s.groupby("corridor")[["Lat", "Lng"]].mean().round(3)
    frames = []
    for name, r in pts.iterrows():
        params = dict(latitude=r["Lat"], longitude=r["Lng"],
                      start_date="2017-01-01", end_date="2019-12-31",
                      hourly="precipitation,temperature_2m,wind_speed_10m",
                      timezone="America/Los_Angeles")
        for _ in range(4):
            resp = requests.get(URL, params=params, timeout=120)
            if resp.status_code == 200:
                break
            print(name, "HTTP", resp.status_code, resp.text[:200])
            time.sleep(20)
        resp.raise_for_status()
        j = resp.json()
        print(name, "| requested", (r["Lat"], r["Lng"]), "| ERA5 grid point", (j["latitude"], j["longitude"]),
              "| units", j["hourly_units"])
        df = pd.DataFrame(j["hourly"])
        df["corridor"] = name
        frames.append(df)
        time.sleep(2)             # stay below the free-tier rate limit
    w = pd.concat(frames, ignore_index=True)
    w["time"] = pd.to_datetime(w["time"])
    w = w.rename(columns={"precipitation": "precip_mm", "temperature_2m": "temp_c", "wind_speed_10m": "wind"})
    w.to_csv(out, index=False)
    print(w.groupby("corridor").agg(rows=("time", "size"), first=("time", "min"), last=("time", "max"),
                                    nan_precip=("precip_mm", lambda x: int(x.isna().sum()))).to_string())
    return w


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sensors", default="data/sensors_100.csv")
    ap.add_argument("--out", default="data/weather_hourly.csv")
    a = ap.parse_args()
    fetch(a.sensors, a.out)
