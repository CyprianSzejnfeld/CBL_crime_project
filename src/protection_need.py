from __future__ import annotations

import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance

from src import config
from src.crime_features import add_harm_weighted_serious_crime_score

MODEL_DIR = config.MODELS_DIR / "protection_forecast"
WARD_LOOKUP = config.INTERIM_DIR / "lsoa21_ward24_lookup_london.csv"
WARD_CLUSTERS = config.PROCESSED_DIR / "ward_clusters_latest.parquet"
OUT_CLUSTER = config.PROCESSED_DIR / "cluster_protection_need_forecasts.csv"
OUT_WARD = config.PROCESSED_DIR / "ward_protection_need_forecasts.csv"

SERIOUS = ["violence_count", "robbery_count", "possession_of_weapons_count"]
HARM_TARGET = config.HARM_WEIGHTED_SERIOUS_TARGET
TEST_QUARTERS = ["2025Q1", "2025Q2", "2025Q3", "2025Q4"]
LATEST_QUARTER = "2025Q4"


def _q(month: pd.Series) -> pd.Series:
    return pd.PeriodIndex(month.astype(str).str[:7], freq="M").asfreq("Q").astype(str)


def ward_quarter_truth() -> pd.DataFrame:
    p = pd.read_parquet(config.MODELLING_PANEL_PATH)
    p["month"] = p["month"].astype(str).str[:7]
    lookup = pd.read_csv(WARD_LOOKUP)[["lsoa21cd", "ward_code"]]
    if "ward_code" not in p.columns:
        p = p.merge(lookup, on="lsoa21cd", how="left")
    p = add_harm_weighted_serious_crime_score(p)
    p["serious_harm"] = p[[c for c in SERIOUS if c in p.columns]].sum(axis=1)
    pop_col = "total_population" if "total_population" in p.columns else "population"
    p["population"] = pd.to_numeric(p.get(pop_col, 1.0), errors="coerce").replace(0, np.nan).fillna(1.0)
    p["quarter"] = _q(p["month"])
    crime = p.groupby(["ward_code", "quarter"], as_index=False).agg(
        serious_harm=("serious_harm", "sum"),
        harm_weighted_serious_crime_score=(HARM_TARGET, "sum"),
    )
    pop = (
        p.groupby(["ward_code", "quarter", "lsoa21cd"], as_index=False)["population"].max()
        .groupby(["ward_code", "quarter"], as_index=False)["population"].sum()
    )
    return crime.merge(pop, on=["ward_code", "quarter"], how="left")


def strategy_a() -> pd.DataFrame:


    v2 = config.PROCESSED_DIR / "met_lsoa_month_crime_forecasts_v2_2025.csv"
    fc = pd.read_csv(v2 if v2.exists() else config.CRIME_FORECASTS_PATH)
    pred_cols = [f"pred_{c}" for c in SERIOUS if f"pred_{c}" in fc.columns]
    fc["pred_serious"] = fc[pred_cols].sum(axis=1)
    harm_pred_col = f"pred_{HARM_TARGET}"
    if harm_pred_col in fc.columns:
        fc["pred_harm_weighted"] = fc[harm_pred_col]
    else:
        weights = config.HARM_WEIGHTED_SERIOUS_CRIME_WEIGHTS
        pred_violence = pd.to_numeric(fc.get("pred_violence_count", 0), errors="coerce").fillna(0)
        pred_robbery = pd.to_numeric(fc.get("pred_robbery_count", 0), errors="coerce").fillna(0)
        pred_weapons_proxy = pd.to_numeric(fc.get("pred_possession_of_weapons_count", 0), errors="coerce").fillna(0)
        pred_weapons_exclusive = (pred_weapons_proxy - pred_robbery).clip(lower=0)
        fc["pred_harm_weighted"] = (
            weights["violence_count"] * pred_violence
            + weights["robbery_count"] * pred_robbery
            + weights["weapons_exclusive_proxy_count"] * pred_weapons_exclusive
        )
    fc["target_month"] = (
        pd.PeriodIndex(fc["month"].astype(str).str[:7], freq="M") + 1
    ).astype(str)
    fc["quarter"] = pd.PeriodIndex(fc["target_month"], freq="M").asfreq("Q").astype(str)
    lookup = pd.read_csv(WARD_LOOKUP)[["lsoa21cd", "ward_code"]]
    fc = fc.merge(lookup, on="lsoa21cd", how="left")
    agg = fc.groupby(["ward_code", "quarter"], as_index=False)[["pred_serious", "pred_harm_weighted"]].sum()
    return agg[agg["quarter"].isin(TEST_QUARTERS)].rename(
        columns={"pred_serious": "pred_A", "pred_harm_weighted": "pred_A_harm"}
    )


def _b_features(truth: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df = truth.sort_values(["ward_code", "quarter"]).copy()
    df["target_value"] = pd.to_numeric(df[target_col], errors="coerce").fillna(0)
    g = df.groupby("ward_code")["target_value"]
    df["lag1"] = g.shift(1)
    df["lag2"] = g.shift(2)
    df["lag4"] = g.shift(4)
    df["rollmean4"] = g.shift(1).rolling(4, min_periods=1).mean().reset_index(level=0, drop=True)
    df["log_pop"] = np.log1p(df["population"])
    df["qnum"] = df["quarter"].str[-1].astype(int)
    for q in (2, 3, 4):
        df[f"q{q}"] = (df["qnum"] == q).astype(int)
    return df


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, float)
    y_pred = np.clip(np.asarray(y_pred, float), 1e-6, None)
    n = len(y_true)
    dec = max(int(round(n * 0.10)), 1)
    order_pred = np.argsort(-y_pred)
    order_true = np.argsort(-y_true)
    top_pred = set(order_pred[:dec])
    top_true = set(order_true[:dec])
    captured = y_true[list(top_pred)].sum() / max(y_true.sum(), 1e-9)
    precision = len(top_pred & top_true) / dec
    false_neg = len(top_true - top_pred) / max(len(top_true), 1)
    sp = spearmanr(y_true, y_pred).correlation
    return {
        "mae": round(mean_absolute_error(y_true, y_pred), 4),
        "rmse": round(float(np.sqrt(np.mean((y_true - y_pred) ** 2))), 4),
        "poisson_deviance": round(mean_poisson_deviance(y_true + 1e-6, y_pred), 4),
        "spearman": round(float(sp), 4),
        "top_decile_capture": round(float(captured), 4),
        "precision_top_decile": round(float(precision), 4),
        "false_negative_high_harm": round(float(false_neg), 4),
        "n": n,
    }


def strategy_b(truth: pd.DataFrame, target_col: str, pred_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = _b_features(truth, target_col)
    feat = ["lag1", "lag2", "lag4", "rollmean4", "log_pop", "q2", "q3", "q4"]
    train = df[(~df["quarter"].isin(TEST_QUARTERS)) & df[feat].notna().all(axis=1)]
    test = df[df["quarter"].isin(TEST_QUARTERS) & df[feat].notna().all(axis=1)].copy()
    yt = train["target_value"].values
    Xtr, Xte = train[feat], test[feat]

    preds = {"naive_last_quarter": test["lag1"].values, "seasonal_lag4": test["lag4"].values}
    try:
        po = sm.GLM(yt, sm.add_constant(Xtr), family=sm.families.Poisson()).fit()
        preds["poisson_glm"] = po.predict(sm.add_constant(Xte, has_constant="add"))
    except Exception:
        pass
    try:
        nb = sm.GLM(yt, sm.add_constant(Xtr), family=sm.families.NegativeBinomial(alpha=1.0)).fit()
        preds["negbin_glm"] = nb.predict(sm.add_constant(Xte, has_constant="add"))
    except Exception:
        pass
    try:
        hgb = HistGradientBoostingRegressor(loss="poisson", max_depth=4, learning_rate=0.08, max_iter=300)
        hgb.fit(Xtr, np.clip(yt, 0, None))
        preds["hgb_poisson"] = hgb.predict(Xte)
    except Exception:
        pass

    rows, best_name, best_key = [], None, -1
    for name, p in preds.items():
        m = _metrics(test["target_value"].values, p)
        m["model"] = f"B_{name}"
        m["target"] = target_col
        rows.append(m)
        key = m["top_decile_capture"] - m["false_negative_high_harm"]
        if key > best_key:
            best_key, best_name = key, name
    test[pred_col] = preds[best_name]
    bmetrics = pd.DataFrame(rows)
    bmetrics.attrs["best"] = f"B_{best_name}"
    return test[["ward_code", "quarter", pred_col]], bmetrics


def run() -> pd.DataFrame:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    truth = ward_quarter_truth()
    a = strategy_a()
    b_pred, bmetrics = strategy_b(truth, "serious_harm", "pred_B")
    b_harm_pred, b_harm_metrics = strategy_b(truth, HARM_TARGET, "pred_B_harm")

    test_truth = truth[truth["quarter"].isin(TEST_QUARTERS)]
    merged = test_truth.merge(a, on=["ward_code", "quarter"], how="inner").merge(
        b_pred, on=["ward_code", "quarter"], how="inner"
    ).merge(
        b_harm_pred, on=["ward_code", "quarter"], how="inner"
    )
    a_m = _metrics(merged["serious_harm"].values, merged["pred_A"].values)
    a_m["model"] = "A_aggregated_lsoa_monthly"
    a_m["target"] = "serious_harm"
    a_harm_m = _metrics(merged[HARM_TARGET].values, merged["pred_A_harm"].values)
    a_harm_m["model"] = "A_aggregated_lsoa_monthly"
    a_harm_m["target"] = HARM_TARGET
    best_b = bmetrics.attrs["best"]
    b_row = bmetrics[bmetrics["model"] == best_b].iloc[0].to_dict()
    best_harm_b = b_harm_metrics.attrs["best"]
    b_harm_row = b_harm_metrics[b_harm_metrics["model"] == best_harm_b].iloc[0].to_dict()

    all_metrics = pd.concat([pd.DataFrame([a_m, a_harm_m]), bmetrics, b_harm_metrics], ignore_index=True)

    a_key = a_m["top_decile_capture"] - a_m["false_negative_high_harm"]
    b_key = b_row["top_decile_capture"] - b_row["false_negative_high_harm"]
    selected = "A_aggregated_lsoa_monthly" if a_key >= b_key else best_b
    sel_pred_col = "pred_A" if selected.startswith("A") else "pred_B"
    a_harm_key = a_harm_m["top_decile_capture"] - a_harm_m["false_negative_high_harm"]
    b_harm_key = b_harm_row["top_decile_capture"] - b_harm_row["false_negative_high_harm"]
    selected_harm = "A_aggregated_lsoa_monthly" if a_harm_key >= b_harm_key else best_harm_b
    sel_harm_pred_col = "pred_A_harm" if selected_harm.startswith("A") else "pred_B_harm"

    bands = _cluster_bands(merged, sel_pred_col, sel_harm_pred_col)
    bands.to_csv(OUT_CLUSTER, index=False)

    meta = {
        "selected_strategy": selected,
        "selected_harm_weighted_strategy": selected_harm,
        "strategy_A_metrics": a_m,
        "strategy_B_best": best_b,
        "strategy_B_metrics": {k: b_row[k] for k in a_m if k in b_row},
        "strategy_A_harm_weighted_metrics": a_harm_m,
        "strategy_B_harm_weighted_best": best_harm_b,
        "strategy_B_harm_weighted_metrics": {k: b_harm_row[k] for k in a_harm_m if k in b_harm_row},
        "serious_harm_target": SERIOUS,
        "harm_weighted_serious_crime_target": HARM_TARGET,
        "harm_weighted_serious_crime_note": config.HARM_WEIGHTED_SERIOUS_CRIME_NOTE,
        "excluded_predictors": "ethnicity (prop_*), fairness scores/pathways, trust context, racial disproportionality",
        "test_quarters": TEST_QUARTERS,
        "all_model_metrics": all_metrics.to_dict("records"),
        "selection_criterion": "top_decile_capture minus false_negative_high_harm (decision-relevant), then rank correlation",
        "banding_rule": "cluster protection band uses max of serious-crime-per-1,000 rank and harm-weighted-per-1,000 rank against all London wards, plus aggregate guardrail floor",
    }
    (MODEL_DIR / "protection_forecast_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return bands


def _cluster_bands(merged: pd.DataFrame, pred_col: str, harm_pred_col: str) -> pd.DataFrame:
    latest = merged[merged["quarter"] == LATEST_QUARTER][["ward_code", pred_col, harm_pred_col]]
    reference = merged[merged["quarter"] == LATEST_QUARTER][["ward_code", pred_col, harm_pred_col, "population"]].copy()
    ref_pop = pd.to_numeric(reference["population"], errors="coerce").replace(0, np.nan)
    reference["serious_per_1000"] = pd.to_numeric(reference[pred_col], errors="coerce").fillna(0) / ref_pop * 1000
    reference["harm_per_1000"] = pd.to_numeric(reference[harm_pred_col], errors="coerce").fillna(0) / ref_pop * 1000
    serious_ref = reference["serious_per_1000"].replace([np.inf, -np.inf], np.nan).dropna()
    harm_ref = reference["harm_per_1000"].replace([np.inf, -np.inf], np.nan).dropna()
    serious_ref_avg = float(serious_ref.mean()) if not serious_ref.empty else 0.0
    harm_ref_avg = float(harm_ref.mean()) if not harm_ref.empty else 0.0

    s1k = reference["serious_per_1000"].replace([np.inf, -np.inf], np.nan)
    h1k = reference["harm_per_1000"].replace([np.inf, -np.inf], np.nan)
    ward_tbl = pd.DataFrame({
        "ward_code": reference["ward_code"],
        "predicted_serious_harm_per_1000_residents": s1k.round(4),
        "predicted_harm_weighted_serious_crime_score_per_1000_residents": h1k.round(4),
        "london_serious_harm_avg_per_1000_residents": round(serious_ref_avg, 4),
        "london_harm_weighted_serious_crime_score_avg_per_1000_residents": round(harm_ref_avg, 4),
        "predicted_serious_harm_rank_pct": s1k.apply(lambda v: _rank_against_reference(v, serious_ref) if pd.notna(v) else None),
        "predicted_harm_weighted_serious_crime_score_rank_pct": h1k.apply(lambda v: _rank_against_reference(v, harm_ref) if pd.notna(v) else None),
    })
    ward_tbl.to_csv(OUT_WARD, index=False)

    clusters = pd.read_parquet(WARD_CLUSTERS)
    panel = pd.read_parquet(config.MODELLING_PANEL_PATH, columns=["lsoa21cd", "month", "total_population", "population"])
    pop_col = "total_population" if "total_population" in panel.columns else "population"
    latest_pop = panel.sort_values(["lsoa21cd", "month"]).groupby("lsoa21cd", as_index=False).tail(1)
    population_by_lsoa = pd.to_numeric(latest_pop[pop_col], errors="coerce").fillna(0)
    pop_lookup = dict(zip(latest_pop["lsoa21cd"], population_by_lsoa))
    rows = []
    for _, cl in clusters.iterrows():
        wards = [w for w in str(cl["member_wards"]).split(";") if w]
        lsoas = [x for x in str(cl.get("member_lsoas", "")).split(";") if x and x != "nan"]
        population = float(sum(pop_lookup.get(x, 0.0) for x in lsoas))
        sub = latest[latest["ward_code"].isin(wards)]
        pred = float(sub[pred_col].sum())
        harm_pred = float(sub[harm_pred_col].sum())
        has_population = population > 0
        rows.append({
            "cluster_id": cl["cluster_id"],
            "member_wards": cl["member_wards"],
            "cluster_population": round(population, 0),
            "aggregate_crime_guardrail": cl.get("aggregate_crime_guardrail", "Low"),
            "predicted_serious_harm_next_period": round(pred, 2),
            "predicted_harm_weighted_serious_crime_score_next_period": round(harm_pred, 2),
            "predicted_serious_harm_per_1000_residents": round((pred / population) * 1000, 4) if has_population else 0.0,
            "predicted_harm_weighted_serious_crime_score_per_1000_residents": round((harm_pred / population) * 1000, 4) if has_population else 0.0,
            "london_serious_harm_avg_per_1000_residents": round(serious_ref_avg, 4),
            "london_harm_weighted_serious_crime_score_avg_per_1000_residents": round(harm_ref_avg, 4),
        })
    bands = pd.DataFrame(rows)
    bands["predicted_serious_harm_count_rank_pct"] = bands["predicted_serious_harm_next_period"].rank(pct=True)
    bands["predicted_harm_weighted_serious_crime_score_count_rank_pct"] = bands[
        "predicted_harm_weighted_serious_crime_score_next_period"
    ].rank(pct=True)
    bands["predicted_serious_harm_rank_pct"] = bands["predicted_serious_harm_per_1000_residents"].apply(
        lambda v: _rank_against_reference(v, serious_ref)
    )
    bands["predicted_harm_weighted_serious_crime_score_rank_pct"] = bands[
        "predicted_harm_weighted_serious_crime_score_per_1000_residents"
    ].apply(lambda v: _rank_against_reference(v, harm_ref))

    def _band(r):


        g = str(r["aggregate_crime_guardrail"])
        p = max(r["predicted_serious_harm_rank_pct"], r["predicted_harm_weighted_serious_crime_score_rank_pct"])
        if g == "Severe" or p >= 0.97:
            return "Critical"
        if p >= 0.92:
            return "High"
        if p >= 0.70:
            return "Medium"
        return "Low"

    bands["protection_need_band"] = bands.apply(_band, axis=1)
    impl = {
        "Low": "May receive review packages where fairness evidence is strong.",
        "Medium": "May receive review or training packages with outcome monitoring.",
        "High": "Fairness intervention with protected harm-focused presence.",
        "Critical": "Critical safety level: use protective or monitoring packages only.",
    }
    bands["eligibility_implication"] = bands["protection_need_band"].map(impl)
    return bands


def _rank_against_reference(value: float, reference: pd.Series) -> float:
    ref = pd.to_numeric(reference, errors="coerce").dropna()
    if ref.empty:
        return 0.0
    return float((ref <= float(value)).mean())


if __name__ == "__main__":
    run()
