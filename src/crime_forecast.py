from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .crime_features import build_features, feature_columns

warnings.filterwarnings("ignore")

TARGETS = config.FORECAST_TARGETS
SOURCE = {
    **config.CRIME_TARGET_SOURCE,
    config.HARM_WEIGHTED_SERIOUS_TARGET: config.HARM_WEIGHTED_SERIOUS_TARGET,
}


def _splits(months: pd.Series) -> dict[str, pd.Series]:
    return {
        "train": months <= config.TRAIN_END,
        "valid": (months >= config.VALID_START) & (months <= config.VALID_END),
        "test": (months >= config.TEST_START) & (months <= config.TEST_END),
    }


def _poisson_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", PoissonRegressor(alpha=1e-3, max_iter=400)),
        ]
    )


def _hgb_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="poisson",
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=1.0,
        random_state=0,
    )


def _rf_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=60, max_depth=14, min_samples_leaf=20,
                    n_jobs=-1, random_state=0,
                ),
            ),
        ]
    )


def _precision_at_k(frame: pd.DataFrame, k: int) -> float:
    vals = []
    for _, grp in frame.groupby("month"):
        if len(grp) < k:
            continue
        top_pred = set(grp.nlargest(k, "pred")["lsoa21cd"])
        top_true = set(grp.nlargest(k, "true")["lsoa21cd"])
        vals.append(len(top_pred & top_true) / k)
    return float(np.mean(vals)) if vals else np.nan


def _top_decile_capture(frame: pd.DataFrame) -> float:
    vals = []
    for _, grp in frame.groupby("month"):
        n = max(1, int(round(len(grp) * 0.1)))
        top_pred = set(grp.nlargest(n, "pred")["lsoa21cd"])
        captured = grp[grp["lsoa21cd"].isin(top_pred)]["true"].sum()
        denom = grp["true"].sum()
        if denom > 0:
            vals.append(captured / denom)
    return float(np.mean(vals)) if vals else np.nan


def _spearman_monthly(frame: pd.DataFrame) -> float:
    vals = []
    for _, grp in frame.groupby("month"):
        if grp["true"].nunique() > 1 and grp["pred"].nunique() > 1:
            vals.append(grp["pred"].corr(grp["true"], method="spearman"))
    return float(np.nanmean(vals)) if vals else np.nan


def _evaluate(y_true, y_pred, months, lsoa, borough) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    frame = pd.DataFrame(
        {"true": y_true, "pred": y_pred, "month": np.asarray(months),
         "lsoa21cd": np.asarray(lsoa), "borough": np.asarray(borough)}
    )
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }
    try:
        metrics["poisson_deviance"] = float(
            mean_poisson_deviance(y_true, np.clip(y_pred, 1e-6, None))
        )
    except ValueError:
        metrics["poisson_deviance"] = np.nan
    met_month = frame.groupby("month").agg(true=("true", "sum"), pred=("pred", "sum"))
    metrics["met_monthly_agg_abs_error"] = float((met_month["true"] - met_month["pred"]).abs().mean())
    bm = frame.groupby(["borough", "month"]).agg(true=("true", "sum"), pred=("pred", "sum"))
    metrics["borough_month_agg_abs_error"] = float((bm["true"] - bm["pred"]).abs().mean())
    metrics["spearman_monthly"] = _spearman_monthly(frame)
    metrics["precision_at_10"] = _precision_at_k(frame, 10)
    metrics["precision_at_25"] = _precision_at_k(frame, 25)
    metrics["top_decile_capture"] = _top_decile_capture(frame)
    return metrics


def _baseline_preds(df: pd.DataFrame, target: str) -> dict[str, pd.Series]:


    src = target
    seasonal = df.groupby("lsoa21cd")[src].shift(11)
    return {
        "baseline_last_month": df[src],
        "baseline_rolling_3m": df.get(f"{src}_rolling_3m_mean", df[src]),
        "baseline_seasonal_lag12": seasonal,
    }


def train_target(df: pd.DataFrame, target: str, feat_cols: list[str], run_rf: bool) -> dict:

    ycol = f"target_{target}_next_month"
    src = SOURCE[target]
    work = df.dropna(subset=[ycol]).copy()
    masks = _splits(work["month"])

    candidates: dict[str, object] = {}
    preds_valid: dict[str, np.ndarray] = {}


    base = _baseline_preds(work, target)
    for name, series in base.items():
        candidates[name] = ("baseline", series)
        preds_valid[name] = series[masks["valid"]].fillna(0).to_numpy()

    X = work[feat_cols]
    y = work[ycol]
    Xtr, ytr = X[masks["train"]], y[masks["train"]]
    Xva = X[masks["valid"]]

    fitted: dict[str, object] = {}
    model_specs = [("poisson", _poisson_pipeline()), ("hist_gbm_poisson", _hgb_model())]
    if run_rf:
        model_specs.append(("random_forest", _rf_pipeline()))
    for name, model in model_specs:
        model.fit(Xtr, ytr)
        fitted[name] = model
        candidates[name] = ("model", model)
        preds_valid[name] = np.clip(model.predict(Xva), 0, None)


    yva = y[masks["valid"]].to_numpy()
    val_mae = {name: float(mean_absolute_error(yva, np.nan_to_num(p))) for name, p in preds_valid.items()}
    best_name = min(val_mae, key=val_mae.get)


    test_metrics = {}
    test_frame = work[masks["test"]]
    for name, (kind, obj) in candidates.items():
        if kind == "baseline":
            p = obj[masks["test"]].fillna(0).to_numpy()
        else:
            p = np.clip(obj.predict(test_frame[feat_cols]), 0, None)
        test_metrics[name] = _evaluate(
            test_frame[ycol].to_numpy(), p, test_frame["month"],
            test_frame["lsoa21cd"], test_frame["borough"],
        )

    best_obj = candidates[best_name]
    return {
        "best_name": best_name,
        "best_kind": best_obj[0],
        "best_model": fitted.get(best_name),
        "best_baseline_source": src if best_obj[0] == "baseline" else None,
        "val_mae": val_mae,
        "test_metrics": test_metrics,
    }


def _band_from_percentile(p: float) -> str:
    for band, (lo, hi) in config.RISK_PERCENTILE_BANDS.items():
        if (p >= lo and p < hi) or (band == "Very High" and p >= lo):
            return band
    return "Low"


def build_guardrails(forecasts: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:

    df = forecasts.copy()
    relevant = ["total_crime_count"] + [t for t in config.STOP_SEARCH_RELEVANT_TARGETS]
    relevant = list(dict.fromkeys(relevant))


    for t in relevant:
        col = f"pred_{t}"
        if col not in df.columns:
            continue
        pct = df.groupby("month")[col].rank(pct=True) * 100
        df[f"risk_pct_{t}"] = pct
        df[f"risk_band_{t}"] = pct.apply(_band_from_percentile)


    feat_idx = features.set_index(["lsoa21cd", "month"])
    for t in relevant:
        src = SOURCE[t]
        tcol = f"{src}_trend_3m_vs_12m"
        if tcol in features.columns:
            mapped = df.set_index(["lsoa21cd", "month"]).index.map(feat_idx[tcol])
            df[f"rising_{t}"] = (pd.Series(mapped, index=df.index) > config.RISING_TREND_RATIO).fillna(False)
        else:
            df[f"rising_{t}"] = False

    def level(row) -> str:
        weap = row.get("risk_band_possession_of_weapons_count", "Low")
        viol = row.get("risk_band_violence_count", "Low")
        robb = row.get("risk_band_robbery_count", "Low")
        harm = row.get(f"risk_band_{config.HARM_WEIGHTED_SERIOUS_TARGET}", "Low")
        total = row.get("risk_band_total_crime_count", "Low")
        weap_rise = row.get("rising_possession_of_weapons_count", False)
        viol_rise = row.get("rising_violence_count", False)
        harm_rise = row.get(f"rising_{config.HARM_WEIGHTED_SERIOUS_TARGET}", False)
        total_rise = row.get("rising_total_crime_count", False)
        if (weap == "Very High" and weap_rise) or (viol == "Very High" and viol_rise) or (harm == "Very High" and harm_rise):
            return "Severe"
        if weap == "Very High" or viol == "Very High" or robb == "Very High" or harm == "Very High" or (total == "Very High" and total_rise):
            return "High"
        if any(row.get(f"risk_band_{t}") == "High" for t in relevant):
            return "Medium"
        return "Low"

    df["crime_guardrail_level"] = df.apply(level, axis=1)
    df["max_reduction_cap_from_crime_guardrail"] = df["crime_guardrail_level"].map(config.CRIME_GUARDRAIL_CAPS)

    def explain(row) -> str:
        lvl = row["crime_guardrail_level"]
        cap = int(row["max_reduction_cap_from_crime_guardrail"] * 100)
        bands = ", ".join(
            f"{t.replace('_count','').replace('_',' ')}: {row.get(f'risk_band_{t}','Low')}"
            for t in relevant
        )
        note = (
            " (possession-of-weapons uses a weapons+robbery street-crime proxy; harm-weighted score is CCHI-inspired, not true offence-code CCHI)"
            if "possession_of_weapons_count" in relevant else ""
        )
        return (
            f"Crime guardrail {lvl}: maximum reduction {cap}%. Predicted next-month "
            f"risk bands - {bands}.{note}"
        )

    df["crime_guardrail_explanation"] = df.apply(explain, axis=1)
    return df


def run() -> None:
    config.CRIME_FORECAST_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    feats = build_features()
    config.CRIME_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(config.CRIME_FEATURES_PATH, index=False)
    cols = feature_columns(feats)

    metadata = {"targets": {}, "splits": {
        "train_end": config.TRAIN_END, "valid": [config.VALID_START, config.VALID_END],
        "test": [config.TEST_START, config.TEST_END]}}
    metrics_rows = []

    test_mask = (feats["month"] >= config.TEST_START) & (feats["month"] <= config.TEST_END)
    forecast = feats.loc[test_mask, ["lsoa21cd", "month", "borough"]].copy()

    for i, target in enumerate(TARGETS):
        if SOURCE[target] not in feats.columns:
            continue

        run_rf = target == config.PRIMARY_CRIME_TARGET
        pi = train_target(feats, target, cols["policy_independent"], run_rf=run_rf)
        ws = train_target(feats, target, cols["with_stop_search"], run_rf=False)

        metadata["targets"][target] = {
            "is_proxy": target in config.PROXY_TARGETS,
            "is_harm_weighted_proxy": target == config.HARM_WEIGHTED_SERIOUS_TARGET,
            "policy_independent_best": pi["best_name"],
            "with_stop_search_best": ws["best_name"],
            "policy_independent_val_mae": pi["val_mae"],
        }
        for setname, res in [("policy_independent", pi), ("with_stop_search", ws)]:
            for model_name, m in res["test_metrics"].items():
                metrics_rows.append({
                    "target": target, "feature_set": setname, "model": model_name,
                    "selected": model_name == res["best_name"], **m,
                })


        work = feats.dropna(subset=[f"target_{target}_next_month"]).copy()
        tmask = (work["month"] >= config.TEST_START) & (work["month"] <= config.TEST_END)
        tw = work[tmask]
        if pi["best_kind"] == "model":
            preds = np.clip(pi["best_model"].predict(tw[cols["policy_independent"]]), 0, None)
        else:
            preds = _baseline_preds(work, target)[pi["best_name"]][tmask].fillna(0).to_numpy()


        op_mask = test_mask & feats[f"target_{target}_next_month"].isna()
        if pi["best_kind"] == "model" and op_mask.any():
            op_preds = np.clip(pi["best_model"].predict(feats.loc[op_mask, cols["policy_independent"]]), 0, None)
        elif op_mask.any():
            op_preds = _baseline_preds(feats, target)[pi["best_name"]][op_mask].fillna(0).to_numpy()
        else:
            op_preds = np.array([])


        combined = pd.Series(np.nan, index=feats.index)
        combined.loc[tw.index] = preds
        if op_mask.any():
            combined.loc[feats.loc[op_mask].index] = op_preds
        forecast[f"pred_{target}"] = combined.reindex(forecast.index).values
        forecast[f"actual_next_{target}"] = feats.loc[forecast.index, f"target_{target}_next_month"].values


    guardrails = build_guardrails(forecast, feats)


    forecast.to_csv(config.CRIME_FORECASTS_PATH, index=False)
    guard_cols = ["lsoa21cd", "month", "borough", "crime_guardrail_level",
                  "max_reduction_cap_from_crime_guardrail", "crime_guardrail_explanation"]
    guard_cols += [c for c in guardrails.columns if c.startswith("risk_band_") or c.startswith("rising_") or c.startswith("risk_pct_")]
    guardrails[guard_cols].to_csv(config.CRIME_GUARDRAILS_PATH, index=False)

    metrics_df = pd.DataFrame(metrics_rows)
    metadata["test_metrics"] = metrics_df.to_dict("records")
    metadata["guardrail_caps"] = config.CRIME_GUARDRAIL_CAPS
    metadata["weapons_note"] = (
        "possession_of_weapons_count is a weapons+robbery street-crime PROXY; "
        "theft_from_person_count uses broader theft. Documented limitation."
    )
    metadata["harm_weighted_serious_crime_note"] = config.HARM_WEIGHTED_SERIOUS_CRIME_NOTE
    (config.CRIME_FORECAST_MODELS_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8")



if __name__ == "__main__":
    run()
