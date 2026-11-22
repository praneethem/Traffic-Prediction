"""Data-quality mask: days on which a sensor only repeats last week's (or next week's) values.

Why: some PeMS detectors were filled with a weekly template. On those days there is nothing to forecast, so the
last-week baseline is perfect and every model looks better than it is. The rule below is fixed (no tuning) and is
applied identically to every model and baseline.

Rule: a (sensor, day) is a TEMPLATE day if at least THRESHOLD of that day's valid 15-minute bins are exactly equal to
the same bin 7 days earlier OR 7 days later. Ordinary sensors do this for about 0.8% of bins.
"""
import numpy as np
import pandas as pd

THRESHOLD = 0.90
LAG = 7 * 96


def template_days(F):
    """Bool DataFrame (rows = calendar days, columns = sensor id as str): True on template days."""
    rep = ((F == F.shift(LAG)) | (F == F.shift(-LAG))) & F.notna()
    day = F.index.normalize()
    share = rep.groupby(day).sum() / F.notna().groupby(day).sum().replace(0, np.nan)
    return (share >= THRESHOLD).fillna(False)


def row_mask(df, flags):
    """True for feature-table rows whose TARGET day is NOT a template day for that sensor (the rows kept as 'clean')."""
    s = flags.stack()
    s.index = s.index.set_names(["day", "sid"])
    key = pd.MultiIndex.from_arrays([df["target_time"].dt.normalize().to_numpy(), df["ID2"].astype(str).to_numpy()],
                                    names=["day", "sid"])
    tpl = s.reindex(key).fillna(False).to_numpy(bool)
    return ~tpl
