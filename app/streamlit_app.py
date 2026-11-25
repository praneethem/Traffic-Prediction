"""Bay Area freeway traffic forecast: Streamlit app on top of the trained pipeline.

Run from the repo root:   streamlit run app/streamlit_app.py

What it does
  * The user picks a sensor, a date and time (the "origin") and a horizon (15 or 60 minutes).
  * src/live.py rebuilds the 31 features from ONLY the flow history that ended before the origin
    (tests/test_live_parity.py proves it equals the training table), and the trained LightGBM model predicts.
  * The page shows the forecast next to the recent trend, the naive baselines, a validation-based 80 % interval,
    the real value when the origin lies in the past, a corridor view, a per-prediction explanation, a stress test
    (noisy or missing readings, Analysis D), and the benchmark tables and figures.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from src.features import ALL_FEATURES, FEATURE_GROUPS, H_OFFSET          # noqa: E402
from src.live import live_features, WINDOW_BINS, MIN_BINS, BIN            # noqa: E402

GROUP_OF = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
LAGS = ["flow_t", "flow_t15", "flow_t30", "flow_t45", "flow_t60", "flow_t120"]
WORDS = {"flow_t": "latest reading", "flow_t15": "reading 15 min earlier", "flow_t30": "reading 30 min earlier",
         "flow_t45": "reading 45 min earlier", "flow_t60": "reading 1 h earlier", "flow_t120": "reading 2 h earlier",
         "same_slot_last_week": "same time last week", "same_slot_4wk_mean": "same time, 4-week average",
         "tod_sin": "time of day (sin)", "tod_cos": "time of day (cos)", "dow_sin": "day of week (sin)",
         "dow_cos": "day of week (cos)", "mean_1h": "mean of last hour", "mean_3h": "mean of last 3 h",
         "mean_24h": "mean of last day", "std_1h": "spread of last hour", "std_3h": "spread of last 3 h",
         "is_holiday": "holiday", "is_weekend": "weekend", "precip_mm": "rain now (mm)", "rain_3h_mm": "rain, last 3 h (mm)",
         "temp_c": "temperature", "wind_kmh": "wind", "nb_back": "upstream neighbour", "nb_ahead": "downstream neighbour",
         "has_back": "has upstream neighbour", "has_ahead": "has downstream neighbour", "lanes": "number of lanes",
         "scale_7d": "sensor's usual flow", "miss_3h": "missing readings, last 3 h", "month": "month"}
TRAIN_END = pd.Timestamp("2019-01-01")

st.set_page_config(page_title="Bay Area traffic forecast", layout="wide")


# ---------------------------------------------------------------------------------------------- cached loaders
@st.cache_resource(show_spinner="Loading the traffic data ...")
def load_data():
    F = pd.read_parquet(ROOT / "data" / "flow_15min_100.parquet")
    S = pd.read_csv(ROOT / "data" / "sensors_100.csv")
    W = pd.read_csv(ROOT / "data" / "weather_hourly.csv", parse_dates=["time"])
    return F, S, W


@st.cache_resource(show_spinner="Loading the model ...")
def load_model(h):
    import joblib
    for tag, label in (("final", "trained on 2017 to 2018 (train + validation)"),
                       ("trainonly", "trained on 2017 to mid-2018 only")):
        p = ROOT / "models" / f"lgbm_h{h}_{tag}.joblib"
        if p.exists():
            return joblib.load(p), label
    return None, None


@st.cache_data(show_spinner="Rebuilding the features from the history before this time ...")
def features_at(origin_iso, h):
    F, S, W = load_data()
    o = pd.Timestamp(origin_iso)
    hist = F.loc[F.index < o].iloc[-WINDOW_BINS:]
    return live_features(hist, S, W, o, h)


@st.cache_data
def read_json(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data
def read_csv(path, **kw):
    p = Path(path)
    return pd.read_csv(p, **kw) if p.exists() else None


# ------------------------------------------------------------------------------------------------- core logic
def stressed_inputs(feat, missing, noise_pct, seed=0):
    """Feature matrix after the optional stress test (Analysis D): the latest reading missing (replaced by the
    reading before it) and/or multiplicative noise on the six lag readings."""
    X = feat[ALL_FEATURES].copy()
    if missing:
        X["flow_t"] = X["flow_t15"]
    if noise_pct > 0:
        rng = np.random.default_rng(seed)
        for c in LAGS:
            X[c] = X[c] * (1 + noise_pct / 100 * rng.standard_normal(len(X)))
    return X


def predict_all(model, feat, X):
    ok = feat["ok"].to_numpy()
    pred = np.full(len(feat), np.nan)
    if ok.any():
        pred[ok] = model.predict(X[ok].to_numpy("float32")) * feat.loc[ok, "sc"].to_numpy("float64")
    return pred


def explain(model, X_row, sc):
    """Per-prediction contributions (LightGBM's built-in tree-SHAP) in vehicles per 15 min."""
    c = model.predict(X_row.to_numpy("float32"), pred_contrib=True)[0] * sc
    return pd.Series(c[:-1], index=ALL_FEATURES), float(c[-1])


def origin_range(F):
    return F.index[MIN_BINS], F.index[-1] + BIN


# ---------------------------------------------------------------------------------------------------- page
REQUIRED_FILES = ["data/flow_15min_100.parquet", "data/sensors_100.csv", "data/weather_hourly.csv"]


def missing_files():
    """Files the app needs that are not on disk (data are not in Git unless they were added to the repository)."""
    need = list(REQUIRED_FILES)
    for h in (15, 60):
        if not any((ROOT / "models" / f"lgbm_h{h}_{tag}.joblib").exists() for tag in ("final", "trainonly")):
            need.append(f"models/lgbm_h{h}_final.joblib")
    return [f for f in need if not (ROOT / f).exists()]


def tidy(df):
    """Fixed decimals for the benchmark tables (WAPE shown in %); returns (frame, column_config)."""
    df = df.copy()
    if "WAPE" in df.columns:                                       # leaderboards store WAPE as a fraction
        df["WAPE %"] = df.pop("WAPE") * 100
    cfg = {}
    for c in df.columns:
        if c in ("MAE", "RMSE", "MAE_lo", "MAE_hi", "WAPE %", "WAPE_%", "rows_kept_%"):
            cfg[c] = st.column_config.NumberColumn(format="%.2f")
        elif c.startswith("Skill"):
            cfg[c] = st.column_config.NumberColumn(format="%.3f")
    return df, cfg


def main():
    gone = missing_files()
    if gone:
        st.title("Bay Area freeway traffic forecast")
        st.error("The app cannot start because some data or model files are missing:\n\n"
                 + "\n".join(f"* `{f}`" for f in gone))
        st.markdown("These files are large, so they are created locally rather than stored in Git. Build them by "
                    "following **Get the data** and **How to reproduce** in the README (steps 0 and 1 for the data, "
                    "then `python scripts/run_model.py refit --models lgbm --h 15 60` for the models), "
                    "then run `python scripts/build_app_assets.py` once and restart the app.")
        st.stop()
    F, S, W = load_data()
    first, last = origin_range(F)
    S = S.copy()
    S["ID2"] = S["ID2"].astype(int)

    st.title("Bay Area freeway traffic forecast")
    st.caption("Predicts the traffic flow (vehicles per 15 minutes) for 100 sensors, 15 or 60 minutes ahead, using only "
               "what was known before the chosen time.  Data: LargeST (Caltrans PeMS), 2017 to 2019.")

    # --------------------------------------------------------------------------------------------- sidebar
    with st.sidebar:
        st.header("Choose a forecast")
        h = st.radio("Horizon", [15, 60], format_func=lambda x: f"{x} minutes ahead", horizontal=True)
        corridors = sorted(S["corridor"].unique())
        corridor = st.selectbox("Corridor", corridors)
        sc_tab = S[S["corridor"] == corridor].sort_values("position")
        sid = st.selectbox("Sensor", sc_tab["ID2"].tolist(),
                           format_func=lambda i: f"{i}  (position {int(sc_tab.set_index('ID2').at[i, 'position'])}, "
                                                 f"{sc_tab.set_index('ID2').at[i, 'role']})")
        presets = {"Choose my own time": None,
                   "Evening rush, 12 Mar 2019": pd.Timestamp("2019-03-12 17:00"),
                   "Weekend midday, 16 Mar 2019": pd.Timestamp("2019-03-16 12:00"),
                   "Holiday morning, 25 Dec 2019": pd.Timestamp("2019-12-25 08:00"),
                   "Weekday in training data, 13 Mar 2018": pd.Timestamp("2018-03-13 08:00")}
        preset = st.selectbox("Quick examples", list(presets))
        default = presets[preset] or pd.Timestamp("2019-03-12 17:00")
        default = min(max(default, first), last)
        d = st.date_input("Date", value=default.date(), min_value=first.date(), max_value=last.date())
        t = st.time_input("Time", value=default.time(), step=900)
        origin = pd.Timestamp.combine(d, t).floor("15min")
        if presets[preset] is not None:
            origin = presets[preset] if first <= presets[preset] <= last else origin
        origin = min(max(origin, first), last)
        with st.expander("Stress test (what if the data are bad?)"):
            missing = st.checkbox("Latest reading missing (use the one before)")
            noise = st.slider("Noise on the recent readings (%)", 0, 30, 0, step=5)

    model, model_label = load_model(h)
    if model is None:
        st.error("No trained model found in models/. Run the refit first: "
                 "python scripts/run_model.py refit --models lgbm --h 15 60")
        return
    try:
        feat = features_at(str(origin), h)
    except Exception as e:                                         # noqa: BLE001
        st.error(f"Cannot build features for {origin}: {e}")
        return

    X = stressed_inputs(feat, missing, noise)
    pred_all = predict_all(model, feat, X)
    feat = feat.assign(pred=pred_all)
    row = feat[feat["ID2"] == sid].iloc[0]
    idx = int(feat.index[feat["ID2"] == sid][0])
    sc = float(row["sc"])
    target = origin + H_OFFSET[h] * BIN
    pred = float(row["pred"])
    base = {"Persistence (latest reading)": float(row["flow_t"] * sc),
            "Same time last week": float(row["same_slot_last_week"] * sc),
            "4-week average": float(row["same_slot_4wk_mean"] * sc)}
    q = (read_json(ROOT / "results" / "app" / "error_quantiles.json") or {}).get(str(h))
    lo, hi = (pred + sc * q["q10"], pred + sc * q["q90"]) if q else (None, None)
    actual = float(F.at[target, str(sid)]) if target in F.index and str(sid) in F.columns else np.nan
    in_sample = target < TRAIN_END

    if not row["ok"]:
        st.warning("This sensor has no usable history at the chosen time (missing data), so no forecast is shown.")
        return
    if in_sample:
        st.info(f"This time is inside the period the model was trained on ({model_label}), so the match with the real value "
                "is optimistic. Pick a time in 2019 for a fair test.")
    st.subheader(f"Sensor {sid} on {corridor}: forecast for {target:%a %d %b %Y, %H:%M}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted flow", f"{pred:,.0f}", help="vehicles per 15 minutes")
    c2.metric("80 % interval", f"{lo:,.0f} to {hi:,.0f}" if q else "n/a",
              help="from the spread of the model's errors on the validation year")
    c3.metric("Latest reading", f"{base['Persistence (latest reading)']:,.0f}")
    if np.isfinite(actual):
        c4.metric("Actual flow", f"{actual:,.0f}", delta=f"model error {pred - actual:+,.0f}", delta_color="off")
    else:
        c4.metric("Actual flow", "not available", help="the origin is at the end of the data")

    tabs = st.tabs(["Forecast", "Corridor view", "Why this forecast", "Model performance", "About"])

    # ------------------------------------------------------------------------------------------ tab: forecast
    with tabs[0]:
        t0 = origin - 96 * BIN
        obs = F.loc[(F.index >= t0) & (F.index < origin), str(sid)]
        wk = F.loc[(F.index >= t0 - 7 * 96 * BIN) & (F.index < origin - 7 * 96 * BIN + (H_OFFSET[h] + 4) * BIN), str(sid)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=obs.index, y=obs.values, name="Observed (last 24 h)", line=dict(color="#6C8EBF", width=2.5)))
        fig.add_trace(go.Scatter(x=wk.index + pd.Timedelta(days=7), y=wk.values, name="Same hours one week earlier",
                                 line=dict(color="#999", dash="dot")))
        fig.add_trace(go.Scatter(x=[target], y=[pred], mode="markers", name="Forecast", marker=dict(size=13, color="#0072B2"),
                                 error_y=dict(type="data", symmetric=False, array=[hi - pred] if q else [0],
                                              arrayminus=[pred - lo] if q else [0], color="#0072B2")))
        if np.isfinite(actual):
            fig.add_trace(go.Scatter(x=[target], y=[actual], mode="markers", name="Actual",
                                     marker=dict(size=12, color="#D55E00", symbol="diamond")))
        fig.add_vline(x=origin, line_dash="dash", line_color="#888")
        fig.add_annotation(x=origin, y=1.04, yref="paper", text="forecast made here", showarrow=False, font=dict(size=11))
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="vehicles per 15 min",
                          legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, width="stretch")
        st.markdown("**The model against the simple baselines**")
        rows = [{"Method": "LightGBM forecast", "Prediction": pred}] + [{"Method": k, "Prediction": v} for k, v in base.items()]
        tb = pd.DataFrame(rows)
        if np.isfinite(actual):
            tb["Error vs actual"] = tb["Prediction"] - actual
        st.dataframe(tb.round(0), hide_index=True, width="stretch")
        if missing or noise:
            clean = predict_all(model, feat, stressed_inputs(feat, False, 0))[idx]
            st.caption(f"Stress test is on. Without it the forecast would be {clean:,.0f} "
                       f"(change {pred - clean:+,.0f}).")

    # ---------------------------------------------------------------------------------------- tab: corridor
    with tabs[1]:
        cv = feat[feat["corridor"] == corridor].merge(S[["ID2", "position"]], on="ID2").sort_values("position")
        cv["Latest reading"] = cv["flow_t"] * cv["sc"]
        cv["Actual"] = [F.at[target, str(i)] if target in F.index else np.nan for i in cv["ID2"]]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=cv["position"], y=cv["Latest reading"], name="Latest reading", mode="lines+markers",
                                  line=dict(color="#999", dash="dot")))
        fig2.add_trace(go.Scatter(x=cv["position"], y=cv["pred"], name="Forecast", mode="lines+markers", line=dict(color="#0072B2")))
        if cv["Actual"].notna().any():
            fig2.add_trace(go.Scatter(x=cv["position"], y=cv["Actual"], name="Actual", mode="lines+markers",
                                      line=dict(color="#D55E00")))
        fig2.add_vline(x=float(S.set_index("ID2").at[sid, "position"]), line_dash="dash", line_color="#888")
        fig2.update_layout(height=380, xaxis_title=f"sensor position along {corridor}", yaxis_title="vehicles per 15 min",
                           margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig2, width="stretch")
        show = cv[["ID2", "position", "role", "Latest reading", "pred", "Actual"]].rename(columns={"pred": "Forecast"})
        show["Change vs latest (%)"] = (show["Forecast"] / show["Latest reading"] - 1) * 100
        st.dataframe(show.round(1), hide_index=True, width="stretch")
        st.caption("The dashed line marks the sensor you selected. 'cold' sensors were never used for training.")

    # ------------------------------------------------------------------------------------ tab: explanation
    with tabs[2]:
        contrib, bias = explain(model, X.loc[[idx]], sc)
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(10)[::-1]
        fig3 = go.Figure(go.Bar(x=top.values, y=[WORDS.get(f, f) for f in top.index], orientation="h",
                                marker_color=["#0072B2" if v >= 0 else "#D55E00" for v in top.values]))
        fig3.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="effect on this forecast (vehicles per 15 min)",
                           title=f"Starting from the model's average of {bias:,.0f}, these inputs moved the forecast to {pred:,.0f}")
        st.plotly_chart(fig3, width="stretch")
        grp = contrib.groupby(GROUP_OF).sum().sort_values()
        fig4 = go.Figure(go.Bar(x=grp.values, y=grp.index, orientation="h",
                                marker_color=["#0072B2" if v >= 0 else "#D55E00" for v in grp.values]))
        fig4.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10), title="The same effects added up by feature group")
        st.plotly_chart(fig4, width="stretch")
        st.caption("Blue pushes the forecast up, orange pushes it down. Values come from LightGBM's built-in tree explanation "
                   "(SHAP values) for this one prediction.")
        with st.expander("All 31 input values for this sensor"):
            vals = X.loc[idx]
            st.dataframe(pd.DataFrame({"Feature": [WORDS.get(f, f) for f in ALL_FEATURES], "Group": [GROUP_OF[f] for f in ALL_FEATURES],
                                       "Value": vals.values}).round(3), hide_index=True, width="stretch")

    # ------------------------------------------------------------------------------------------ tab: results
    with tabs[3]:
        st.markdown("**Validation year (2018-07 to 2018-12), all nine models**")
        a, b = st.columns(2)
        for col, hh in zip((a, b), (15, 60)):
            lb = read_csv(ROOT / "results" / f"val_leaderboard_h{hh}.csv")
            if lb is not None:
                col.markdown(f"{hh}-minute horizon")
                lb, cfg = tidy(lb.drop(columns=[c for c in ("n",) if c in lb.columns]))
                col.dataframe(lb, hide_index=True, width="stretch", column_config=cfg)
        ht = read_csv(ROOT / "results" / "final" / "headline_test_table.csv")
        if ht is not None:
            st.markdown("**Test year 2019 (scored once)**")
            ht, cfg = tidy(ht)
            st.dataframe(ht, hide_index=True, width="stretch", column_config=cfg)
        for title, name in [("Validation leaderboard against the baselines", "val_leaderboard"),
                            ("Test headline with confidence intervals", "test_headline"),
                            ("Feature groups: what happens when one is removed (15 min)", "ablation_h15"),
                            ("Feature groups: what happens when one is removed (60 min)", "ablation_h60"),
                            ("Robustness to noise, missing data and fewer sensors (15 min)", "robust_h15"),
                            ("Robustness to noise, missing data and fewer sensors (60 min)", "robust_h60")]:
            p = ROOT / "results" / "figures" / f"{name}.png"
            if p.exists():
                st.markdown(f"**{title}**")
                st.image(str(p))

    # --------------------------------------------------------------------------------------------- tab: about
    with tabs[4]:
        st.markdown(f"""
**Task.** Predict the traffic flow (vehicles per 15 minutes) 15 and 60 minutes ahead for 100 Bay Area freeway sensors.

**Data.** LargeST (Caltrans PeMS), 2017 to 2019, 15-minute flow. Original source: Caltrans Performance Measurement System. Licence CC BY-NC 4.0.

**Split.** Chronological: train 2017-01 to 2018-06, validation 2018-07 to 2018-12, test 2019. 80 sensors are used for training and
20 are held back as "cold" sensors that the models never see.

**Leakage prevention.** Every feature uses data from before the forecast time only. This page rebuilds the features from the
history that ended before your chosen time, and a test confirms the result equals the table used for training.

**Model in this app.** LightGBM, {model_label}. It is the best of nine models compared (Ridge, PCA + Ridge, Decision Tree, Random Forest,
AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost).

**Limitations.** The model forecasts flow, not speed or congestion. It cannot know about incidents or events that are not
visible in the recent readings. The 80 % interval comes from the validation year and is a rough guide, not a guarantee.
Forecasts for times before 2019 are in-sample.
""")


main()
