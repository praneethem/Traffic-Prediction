import numpy as np, pandas as pd

flow = pd.read_parquet("data/flow_5min_100.parquet")
sensors = pd.read_csv("data/sensors_100.csv")
weather = pd.read_csv("data/weather_hourly.csv", parse_dates=["time"])

print("Flow:", flow.shape, "|", flow.index.min(), "->", flow.index.max(), "| dtype", flow.dtypes.iloc[0])
print("Columns match sensor list:", set(flow.columns) == set(sensors["ID2"].astype(str)))
print("Roles:", sensors["role"].value_counts().to_dict())
for y in (2017, 2018, 2019):
    x = flow.loc[str(y)].to_numpy()
    print(y, "NaN share", round(float(np.isnan(x).mean()), 4), "| zero share", round(float((x == 0).mean()), 4))
print("Weather rows per corridor:", weather.groupby("corridor").size().to_dict())
one = flow[str(sensors.loc[0, "ID2"])]
day = one.loc["2018-03-06"]
print("First sensor, Tue 2018-03-06, mean flow per hour (00 to 23):")
print(day.groupby(day.index.hour).mean().round(0).tolist())
