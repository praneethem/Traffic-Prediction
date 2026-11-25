"""Build every table and number of the report from the result files, so the PDF can never disagree with the repo.

  python report/make_tables.py        (run from the repository root)

Writes report/tables/*.tex and report/tables/numbers.tex (LaTeX macros for numbers quoted in the text) and copies the
figures used by the report to report/figures/."""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import evaluate as ev           # noqa: E402
from src import features as fe           # noqa: E402
from src import models as mo             # noqa: E402

R = ROOT / "results"
T = ROOT / "report" / "tables"
FIG = ROOT / "report" / "figures"
T.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
NUM = {}                                  # macro name -> text


def tex(s):
    return str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


def write(name, body):
    (T / f"{name}.tex").write_text(body, encoding="utf8")


def table(cols, rows, align=None, header=None):
    align = align or "l" + "r" * (len(cols) - 1)
    out = [rf"\begin{{tabular}}{{{align}}}", r"\toprule", " & ".join(header or cols) + r" \\", r"\midrule"]
    for r in rows:
        out.append(" & ".join(r) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def bold_best(vals, fmt, lower_is_better=True):
    v = np.asarray(vals, float)
    best = np.nanmin(v) if lower_is_better else np.nanmax(v)
    return [(rf"\textbf{{{fmt(x)}}}" if x == best else fmt(x)) for x in v]


def macro(name, value):
    NUM[name] = value


ORDER = ["Ridge", "PCA + Ridge", "Decision Tree", "Random Forest", "AdaBoost", "Gradient Boosting", "XGBoost",
         "LightGBM", "CatBoost"]
LABELS = [mo.LABEL[m] for m in mo.MODELS]

# ----------------------------------------------------------------------------------------------- data tables
S = pd.read_csv(ROOT / "data" / "sensors_100.csv")
rows = []
for c, g in S.groupby("corridor", sort=False):
    rows.append([tex(c.split("_", 1)[1]), tex(g["Fwy"].iloc[0]), tex("/".join(sorted(g["County"].unique()))),
                 f"{g['Lanes'].min()}--{g['Lanes'].max()}", f"{g['ID2'].min()}--{g['ID2'].max()}",
                 f"{g['usable_min'].min() * 100:.1f}"])
write("sensors", table(["c"] * 6, rows, "lllccc", ["Corridor", "Freeway", "County", "Lanes", "Sensor IDs", "Min. usable (\\%)"]))

DESC = {"Lags": "flow now and 15, 30, 45, 60, 120 min ago; share of the last 3 h missing",
        "Rolling": "mean and std of the last 1 h and 3 h, 24 h mean, sensor scale (7-day mean)",
        "Seasonal": "same time last week; mean of the same time over the last 4 weeks",
        "Calendar": "time of day and weekday (sin, cos), month, holiday, weekend",
        "Weather": "precipitation, temperature, wind, rain in the last 3 h",
        "Neighbours": "flow of the sensor behind and ahead on the freeway, and whether they exist",
        "Context": "number of lanes"}
rows = [[g, str(len(f)), tex(DESC[g])] for g, f in fe.FEATURE_GROUPS.items()]
rows.append([r"\textbf{Total}", rf"\textbf{{{sum(len(f) for f in fe.FEATURE_GROUPS.values())}}}", ""])
write("features", table(["c"] * 3, rows, "lrp{9.2cm}", ["Group", "Count", "Content"]))
macro("nFeatures", sum(len(f) for f in fe.FEATURE_GROUPS.values()))

tab = ev.load_table(15, columns=["split", "role", "time"])
sp = tab.groupby("split").agg(rows=("time", "size"), first=("time", "min"), last=("time", "max"))
rolecount = tab.groupby(["split", "role"]).size().unstack(fill_value=0)
rows = []
for s_, name in (("train", "Train"), ("val", "Validation"), ("test", "Test")):
    rows.append([name, f"{sp.loc[s_, 'first']:%Y-%m-%d}", f"{sp.loc[s_, 'last']:%Y-%m-%d}", f"{sp.loc[s_, 'rows']:,}",
                 f"{rolecount.loc[s_].get('warm', 0):,}", f"{rolecount.loc[s_].get('cold', 0):,}"])
write("splits", table(["c"] * 6, rows, "lllrrr", ["Split", "From", "To", "Rows (15 min)", "Warm-sensor rows", "Cold-sensor rows"]))
macro("rowsTest", f"{sp.loc['test', 'rows']:,}")
macro("rowsTrain", f"{sp.loc['train', 'rows']:,}")

eda = pd.read_csv(R / "eda_facts.csv", index_col=0)["value"]
macro("missingPct", f"{float(eda['missing_pct_2017_2018']):.2f}")
macro("templateShare", f"{float(eda['template_sensor_day_share']) * 100:.1f}")
macro("weekendRatio", f"{float(eda['weekend_vs_weekday_ratio']) * 100:.0f}")
macro("gapMedian", f"{float(eda['gap_median_hours']):.0f}")
macro("gapLong", f"{float(eda['gap_share_longer_than_1h']) * 100:.0f}")
macro("nbCorr", f"{float(eda['neighbour_corr_gap1']):.2f}")

# ----------------------------------------------------------------------------------------------- headline tables
H = pd.read_csv(R / "final" / "headline_test_table.csv")
for h in (15, 60):
    for view, tag in (("all sensors", "all"), ("clean rows", "clean")):
        d = H[(H["horizon_min"] == h) & (H["view"] == view)].copy()
        d["model"] = d["model"].str.replace("baseline: ", "", regex=False)
        order = ["Persistence", "Last week", "4-week average"] + ORDER
        d = d.set_index("model").loc[order]
        mae = bold_best(d["MAE"], lambda x: f"{x:.2f}")
        rmse = bold_best(d["RMSE"], lambda x: f"{x:.2f}")
        wape = bold_best(d["WAPE_%"], lambda x: f"{x:.2f}")
        rows = []
        for i, (m, r) in enumerate(d.iterrows()):
            if i == 3:
                rows.append([r"\midrule"])
            rows.append([tex(m), mae[i], rf"{{\small [{r['MAE_lo']:.2f}, {r['MAE_hi']:.2f}]}}", rmse[i], wape[i],
                         f"{r['Skill'] * 100:+.1f}"])
        body = ["\\begin{tabular}{lrrrrr}", "\\toprule",
                "Model & MAE & 95\\% interval & RMSE & WAPE (\\%) & Skill (\\%) \\\\", "\\midrule"]
        for r in rows:
            body.append(r[0] if len(r) == 1 else " & ".join(r) + " \\\\")
        body += ["\\bottomrule", "\\end{tabular}"]
        write(f"headline_{tag}_h{h}", "\n".join(body) + "\n")
    d = H[(H["horizon_min"] == h) & (H["view"] == "all sensors")].set_index("model")
    macro(f"lgbmMAE{h}", f"{d.loc['LightGBM', 'MAE']:.2f}")
    macro(f"lgbmWAPE{h}", f"{d.loc['LightGBM', 'WAPE_%']:.2f}")
    macro(f"lgbmSkill{h}", f"{d.loc['LightGBM', 'Skill'] * 100:.1f}")
    macro(f"persMAE{h}", f"{d.loc['baseline: Persistence', 'MAE']:.2f}")
    macro(f"bestBaseName{h}", {15: "persistence", 60: "the four-week average"}[h])
    macro(f"ridgeSkill{h}", f"{d.loc['Ridge', 'Skill'] * 100:.1f}")
    macro(f"ridgeMAE{h}", f"{d.loc['Ridge', 'MAE']:.2f}")
    macro(f"lastweekMAE{h}", f"{d.loc['baseline: Last week', 'MAE']:.2f}")
    c = H[(H["horizon_min"] == h) & (H["view"] == "clean rows")].set_index("model")
    macro(f"lgbmCleanMAE{h}", f"{c.loc['LightGBM', 'MAE']:.2f}")
    macro(f"keptPct", f"{c['rows_kept_%'].iloc[0]:.1f}")

WC = pd.read_csv(R / "final" / "test_warm_vs_cold.csv")
rows = []
for m in ORDER:
    cells = [tex(m)]
    for h in (15, 60):
        g = WC[(WC["horizon_min"] == h) & (WC["model"] == m)].set_index("role")
        cells += [f"{g.loc['warm', 'WAPE_%']:.2f}", f"{g.loc['cold', 'WAPE_%']:.2f}",
                  f"{(g.loc['cold', 'WAPE_%'] - g.loc['warm', 'WAPE_%']):+.2f}"]
    rows.append(cells)
write("warm_cold", table(["c"] * 7, rows, "lrrrrrr",
                         ["Model", "Warm", "Cold", "Gap", "Warm", "Cold", "Gap"]))

# ----------------------------------------------------------------------------------------------- cost and tuning
C = pd.read_csv(R / "final" / "cost_table.csv")
rows = []
for m in ORDER:
    cells = [tex(m)]
    for h in (15, 60):
        r = C[(C["model"] == m) & (C["h"] == h)].iloc[0]
        cells += [f"{r['fit_s']:.0f}", f"{r['predict_s_per_100k']:.2f}", f"{r['model_MB']:.2f}"]
    rows.append(cells)
write("cost", table(["c"] * 7, rows, "lrrrrrr",
                    ["Model", "Fit (s)", "Predict (s/100k)", "Size (MB)", "Fit (s)", "Predict (s/100k)", "Size (MB)"]))

rows = []
for n in mo.MODELS:
    for h in (15, 60):
        b = ev.read_best(n, h)
        p = {k: v for k, v in b["params"].items()}
        ps = ", ".join(f"{k}={v}" for k, v in p.items())
        rows.append([tex(mo.LABEL[n]), str(h), tex(ps), "" if b["best_iter"] is None else str(b["best_iter"]),
                     f"{b['val']['MAE']:.2f}"])
write("tuning", table(["c"] * 5, rows, "llp{7.0cm}rr", ["Model", "h", "Selected settings", "Trees", "Val. MAE"]))

# validation leaderboard (what the choice of settings was based on)
rows = []
for n in mo.MODELS:
    cells = [tex(mo.LABEL[n])]
    for h in (15, 60):
        cells.append(f"{ev.read_best(n, h)['val']['MAE']:.2f}")
    rows.append(cells)

# ----------------------------------------------------------------------------------------------- ablation
for h in (15, 60):
    A = pd.read_csv(R / "ablation" / f"ablation_delta_pct_h{h}.csv").set_index("model").loc[ORDER]
    cols = [c for c in A.columns]
    rows = []
    for m, r in A.iterrows():
        rows.append([tex(m)] + [f"{r[c]:+.1f}" for c in cols])
    write(f"ablation_h{h}", table(["c"] * (1 + len(cols)), rows, "l" + "r" * len(cols), ["Model"] + cols))
A15 = pd.read_csv(R / "ablation" / "ablation_delta_pct_h15.csv").set_index("model")
macro("ablLagsLgbm", f"{A15.loc['LightGBM', 'Lags']:.1f}")
macro("ablCalLgbm", f"{A15.loc['LightGBM', 'Calendar']:.1f}")
macro("ablSeasLgbm", f"{A15.loc['LightGBM', 'Seasonal']:.1f}")
macro("ablWeatherLgbm", f"{A15.loc['LightGBM', 'Weather']:.2f}")

# ----------------------------------------------------------------------------------------------- robustness
rows = []
RBDIR = Path(os.environ.get("ROBUST_DIR", R / "robustness"))
RB = {h: pd.read_csv(RBDIR / f"robust_h{h}.csv") for h in (15, 60)}
for m in mo.MODELS:
    cells = [tex(mo.LABEL[m])]
    for h in (15, 60):
        d = RB[h][RB[h]["model"] == m]
        if d.empty:
            cells += ["--"] * 4
            continue
        g = lambda ex, st: d[(d["experiment"] == ex) & (d["setting"] == st)]["MAE_change_pct"]
        un = d[(d["experiment"] == "sensors") & (d["setting"] == "trained on half (unseen)")]["MAE_change_pct"]
        hist = d[(d["experiment"] == "history") & (d["setting"] == "3 mo")]["MAE_change_pct"]
        cells += [f"{g('noise', '10%').iloc[0]:+.1f}", f"{g('missing', '30%').iloc[0]:+.1f}",
                  f"{un.iloc[0]:+.1f}", (f"{hist.iloc[0]:+.1f}" if len(hist) else "--")]
    rows.append(cells)
write("robust", table(["c"] * 9, rows, "lrrrrrrrr",
                      ["Model", "Noise 10\\%", "Miss. 30\\%", "Unseen", "3 mo", "Noise 10\\%", "Miss. 30\\%", "Unseen", "3 mo"]))

# ----------------------------------------------------------------------------------------------- error analysis (LightGBM)
def seg(h, name, key):
    d = pd.read_csv(R / f"error_test_h{h}_{name}.csv").set_index(key)
    return d["LightGBM"]


segs = [("peak", "peak", "Peak hours"), ("peak", "peak", None), ("weekend", "weekend", None), ("rain", "rain", None),
        ("change", "change", None), ("level", "level", None), ("role", "role", None)]
rows = []
for name, key, _ in dict.fromkeys(segs):
    pass
spec = [("peak", "peak", ["off-peak", "peak"]), ("weekend", "weekend", ["weekday", "weekend", "holiday"]),
        ("rain", "rain", ["dry", "rain"]), ("change", "change", None), ("level", "level", ["low flow", "mid flow", "high flow"]),
        ("role", "role", ["warm", "cold"])]
nice = {"change": "Change since last reading"}
for name, key, order in spec:
    a, b = seg(15, name, key), seg(60, name, key)
    for lab in (order or list(a.index)):
        rows.append([tex(nice.get(name, name.capitalize()) if lab == (order or list(a.index))[0] else ""), tex(lab),
                     f"{a[lab]:.2f}", f"{b[lab]:.2f}"])
    rows.append([r"\midrule"])
rows = rows[:-1]
body = ["\\begin{tabular}{llrr}", "\\toprule", "Segment & Level & 15 min & 60 min \\\\", "\\midrule"]
for r in rows:
    body.append(r[0] if len(r) == 1 else " & ".join(r) + " \\\\")
body += ["\\bottomrule", "\\end{tabular}"]
write("segments", "\n".join(body) + "\n")
for h in (15, 60):
    ch = seg(h, "change", "change")
    macro(f"chgSteady{h}", f"{ch.iloc[0]:.2f}")
    macro(f"chgSudden{h}", f"{ch.iloc[-1]:.2f}")
    macro(f"rain{h}", f"{seg(h, 'rain', 'rain')['rain']:.2f}")
    macro(f"dry{h}", f"{seg(h, 'rain', 'rain')['dry']:.2f}")
    macro(f"holiday{h}", f"{seg(h, 'weekend', 'weekend')['holiday']:.2f}")
    macro(f"weekday{h}", f"{seg(h, 'weekend', 'weekend')['weekday']:.2f}")
    macro(f"lowFlow{h}", f"{seg(h, 'level', 'level')['low flow']:.2f}")
    macro(f"highFlow{h}", f"{seg(h, 'level', 'level')['high flow']:.2f}")

# ----------------------------------------------------------------------------------------------- paired and published
rows = []
for h in (15, 60):
    P = pd.read_csv(R / "final" / f"paired_test_h{h}.csv")
    P = P[P["view"] == "all rows"].set_index("other")
    for o in ["XGBoost", "CatBoost", "Random Forest", "Gradient Boosting", "AdaBoost", "Decision Tree", "Ridge"]:
        r = P.loc[o]
        rows.append([str(h), tex(o), f"{r['MAE_diff']:.2f}", f"[{r['lo']:.2f}, {r['hi']:.2f}]", f"{r['ref_wins_share'] * 100:.0f}"])
write("paired", table(["c"] * 5, rows, "llrcr", ["h", "Other model", "MAE diff.", "95\\% interval", "LightGBM wins (\\%)"]))
Pm = {h: pd.read_csv(R / "final" / f"paired_test_h{h}.csv") for h in (15, 60)}
for h in (15, 60):
    r = Pm[h][(Pm[h]["view"] == "all rows") & (Pm[h]["other"] == "XGBoost")].iloc[0]
    macro(f"xgbGap{h}", f"{r['MAE_diff']:.2f}")

PUB = pd.read_csv(R / "final" / "published_comparison.csv")
rows = []
for _, r in PUB.iterrows():
    ours = r["source"].startswith("this")
    name = ("Ours: " if ours else "") + r["method"]
    rows.append([tex(r["horizon"]), tex(name), f"{r['MAE']:.2f}", f"{r['MAPE_pct']:.1f}",
                 f"{r['MAE_vs_persistence']:.2f}", f"{r['MAPE_vs_persistence']:.2f}"])
write("published", table(["c"] * 6, rows, "llrrrr", ["Horizon", "Method", "MAE", "MAPE (\\%)", "MAE / persist.", "MAPE / persist."]))
mine = PUB[(PUB["horizon"] == "60 min") & (PUB["method"] == "LightGBM")].iloc[0]
macro("ratio60", f"{mine['MAE_vs_persistence']:.2f}")
macro("mape60", f"{mine['MAPE_pct']:.1f}")
macro("mapePers60", f"{PUB[(PUB['horizon'] == '60 min') & (PUB['method'] == 'persistence')].iloc[0]['MAPE_pct']:.1f}")

# ----------------------------------------------------------------------------------------------- sensitivity (optional)
f = R / "tuning_sensitivity" / "summary.csv"
if f.exists():
    d = pd.read_csv(f)
    rows = [[tex(mo.LABEL[r["model"]]), str(int(r["h"])), f"{r['frozen_val_MAE']:.2f}", f"{r['wider_search_val_MAE']:.2f}",
             f"{r['change_pct']:+.2f}"] for _, r in d.iterrows()]
    write("sensitivity", table(["c"] * 5, rows, "llrrr", ["Model", "h", "Frozen", "Wider search", "Change (\\%)"]))
    macro("sensBest", f"{d['change_pct'].min():.2f}")


# ----------------------------------------------------------------------------------------------- template sensors
from src import quality as Q             # noqa: E402
F15 = pd.read_parquet(ROOT / "data" / "flow_15min_100.parquet")
flags = Q.template_days(F15)
tpl_sensors = sorted(int(c) for c in flags.columns[flags.mean() > 0.95])
macro("nTemplateSensors", len(tpl_sensors))
macro("templateSensorList", ", ".join(str(x) for x in tpl_sensors))
for h in (15, 60):
    d = ev.load_preds("test", "lgbm", h)
    y, p_ = d["y_raw"].to_numpy("float64"), d["pred"].to_numpy("float64")
    t = d["ID2"].isin(tpl_sensors).to_numpy()
    wape = lambda m: float(np.abs(y[m] - p_[m]).sum() / np.abs(y[m]).sum() * 100)
    mae = lambda m: float(np.abs(y[m] - p_[m]).mean())
    macro(f"tplRows", f"{t.mean() * 100:.1f}")
    macro(f"tplWape{h}", f"{wape(t):.2f}")
    macro(f"otherWape{h}", f"{wape(~t):.2f}")
    macro(f"otherMAE{h}", f"{mae(~t):.2f}")


# ------------------------------------------------------------------------------------------------------ PCA study
ps = pd.read_csv(R / "pca_study" / "pca_study.csv")
ps["k"] = ps["k"].astype(str)
KCOLS = ["4", "8", "10", "15", "20", "31"]
LAB = {"ridge": "Ridge", "dt": "Decision Tree", "rf": "Random Forest", "lgbm": "LightGBM"}
rows = []
for h in (15, 60):
    for m in ("ridge", "dt", "rf", "lgbm"):
        d = ps[(ps["h"] == h) & (ps["model"] == m)].set_index("k")["val_MAE"]
        cells = [f"{d[k]:.1f}" if k in d.index else "--" for k in ["raw"] + KCOLS]
        rows.append([f"{h} min" if m == "ridge" else "", LAB[m]] + cells)
write("pca_study", table(None, rows, align="llrrrrrrr",
                         header=["Horizon", "Model", "No PCA"] + [f"$k={k}$" for k in KCOLS]))
for h in (15, 60):
    e = pd.read_csv(R / "pca_study" / f"explained_variance_h{h}.csv")
    macro(f"pcaVar15comp{h}", f"{100 * e['cumulative'][14]:.1f}")
    macro(f"pcaCompNinety{h}", int(np.searchsorted(e["cumulative"], 0.90) + 1))
    macro(f"pcaCompNinetyNine{h}", int(np.searchsorted(e["cumulative"], 0.99) + 1))
macro("pcaFirstComp", f"{100 * pd.read_csv(R / 'pca_study' / 'explained_variance_h15.csv')['cumulative'][0]:.0f}")

# ----------------------------------------------------------------------------------------------- numbers + figures
word = lambda k: k.replace("15", "Fifteen").replace("60", "Sixty")
(T / "numbers.tex").write_text("".join(f"\\newcommand{{\\{word(k)}}}{{{v}}}\n" for k, v in NUM.items()), encoding="utf8")
for f in ["eda01_daily_profile", "eda02_weekly_heatmap", "eda03_day_types", "eda05_rain_effect", "eda06_spatial_dependence",
          "eda07_temporal_dependence", "eda09_missing_data", "eda13_template_days", "test_headline", "test_per_sensor",
          "test_quarter_corridor", "ablation_h15", "ablation_h60", "robust_h15", "robust_h60", "err_segments_test",
          "err_by_hour_test", "err_per_sensor_test", "err_worst_day_test", "val_feature_drivers", "model_pca_curve",
          "model_tree_depth", "model_boosting_trials", "pca_study"]:
    src = R / "figures" / f"{f}.png"
    if src.exists():
        shutil.copy(src, FIG / src.name)
    else:
        print("missing figure:", f)
print("tables:", len(list(T.glob('*.tex'))), "| macros:", len(NUM))
