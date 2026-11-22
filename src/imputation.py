import numpy as np, pandas as pd

WEEK = 672         # 15-minute bins in 7 days
FFILL_LIMIT = 4    # 1 hour


def causal_fill(F):
    """F: DataFrame (bin start x sensor), vehicles per 15 min, NaN = missing.
    Returns (G, stats). Every filled value uses only data from earlier bins."""
    G1 = F.ffill(limit=FFILL_LIMIT)
    G2 = G1.fillna(F.shift(WEEK))
    G3 = G2.fillna(F.rolling(WEEK, min_periods=48).median().shift(1))
    stats = dict(
        missing=int(F.isna().sum().sum()),
        by_ffill=int((F.isna() & G1.notna()).sum().sum()),
        by_last_week=int((G1.isna() & G2.notna()).sum().sum()),
        by_median=int((G2.isna() & G3.notna()).sum().sum()),
        still_missing=int(G3.isna().sum().sum()),
    )
    return G3, stats


def scale_factors(G):
    """Each sensor's own scale: trailing 7-day mean, else trailing 28-day median, else NaN.
    S at bin i uses data up to and including bin i."""
    s7 = G.rolling(WEEK, min_periods=WEEK // 2).mean()
    s28 = G.rolling(4 * WEEK, min_periods=WEEK).median()
    return s7.where(s7 >= 5).fillna(s28.where(s28 >= 5))


if __name__ == "__main__":
    F = pd.read_parquet("data/flow_15min_100.parquet")
    G, stats = causal_fill(F)
    print("Fill statistics (counts):", stats)
    print("Still missing from the train start (2017-01-29):",
          round(float(G.loc["2017-01-29":].isna().to_numpy().mean()), 6))

    S = scale_factors(G)
    for y in (2017, 2018, 2019):
        print(y, "scale missing share:", round(float(S.loc[str(y)].isna().to_numpy().mean()), 5))
    print("Scale missing share from train start:",
          round(float(S.loc["2017-01-29":].isna().to_numpy().mean()), 6))

    k = 50000
    F2 = F.astype("float64")
    F2.iloc[k + 1:] = np.random.default_rng(0).uniform(0, 600, size=F2.iloc[k + 1:].shape)
    G2, _ = causal_fill(F2)
    S2 = scale_factors(G2)
    print("Gap filling is causal (scrambled future changes nothing before it):",
          bool(np.allclose(G.iloc[:k + 1].to_numpy(), G2.iloc[:k + 1].to_numpy(), equal_nan=True)))
    print("Scaling is causal:",
          bool(np.allclose(S.iloc[:k + 1].to_numpy(), S2.iloc[:k + 1].to_numpy(), equal_nan=True)))
