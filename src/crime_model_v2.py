from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance
from sklearn.neighbors import NearestNeighbors

from src import config
from src.crime_features import add_harm_weighted_serious_crime_score, feature_columns

MODEL_DIR = config.MODELS_DIR / "crime_forecast_v2"
OUT_FORECASTS = config.PROCESSED_DIR / "met_lsoa_month_crime_forecasts_v2_2025.csv"
BOUNDARIES = config.LONDON_LSOA_BOUNDARIES_PATH

SERIOUS = [
    "violence_count",
    "robbery_count",
    "possession_of_weapons_count",
    config.HARM_WEIGHTED_SERIOUS_TARGET,
    "total_crime_count",
]
SPATIAL_BASES = [
    "violence_count",
    "robbery_count",
    "possession_of_weapons_count",
    config.HARM_WEIGHTED_SERIOUS_TARGET,
    "total_crime_count",
    "drugs_count",
]
K_NEIGHBOURS = 8
TRAIN_END, VALID_END, TEST_START = "2023-12", "2024-12", "2025-01"


def _centroids() -> pd.DataFrame:
    import json as _json

    from shapely.geometry import shape

    payload = _json.loads(BOUNDARIES.read_text())
    rows = []
    for f in payload["features"]:
        props = f.get("properties") or {}
        code = props.get("LSOA21CD") or props.get("lsoa21cd")
        geom = f.get("geometry")
        if code and geom:
            c = shape(geom).centroid
            rows.append({"lsoa21cd": code, "cx": c.x, "cy": c.y})
    return pd.DataFrame(rows)


def _spatial_adjacency(lsoa_order: list[str]) -> sparse.csr_matrix:
    cent = _centroids().set_index("lsoa21cd").reindex(lsoa_order).dropna()
    coords = cent[["cx", "cy"]].to_numpy()
    nn = NearestNeighbors(n_neighbors=K_NEIGHBOURS + 1).fit(coords)
    _, idx = nn.kneighbors(coords)
    n = len(lsoa_order)
    pos = {c: i for i, c in enumerate(lsoa_order)}
    rows, cols = [], []
    cent_codes = list(cent.index)
    for i, neigh in enumerate(idx):
        src = pos[cent_codes[i]]
        for j in neigh[1:]:
            rows.append(src)
            cols.append(pos[cent_codes[j]])
    A = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1
    return sparse.diags(1.0 / deg) @ A


def add_spatial_features(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel[["lsoa21cd", "month", *SPATIAL_BASES]].copy()
    lsoa_order = sorted(df["lsoa21cd"].unique())
    months = sorted(df["month"].unique())
    A = _spatial_adjacency(lsoa_order)
    out = df[["lsoa21cd", "month"]].copy()
    for base in SPATIAL_BASES:
        M = (df.pivot(index="lsoa21cd", columns="month", values=base)
             .reindex(index=lsoa_order, columns=months).fillna(0.0))
        neigh = A @ M.values
        nbr = pd.DataFrame(neigh, index=lsoa_order, columns=months)
        long = nbr.reset_index().melt(id_vars="index", var_name="month", value_name=f"nbr_{base}").rename(columns={"index": "lsoa21cd"})

        long = long.sort_values(["lsoa21cd", "month"])
        long[f"nbr_{base}_roll3"] = long.groupby("lsoa21cd")[f"nbr_{base}"].transform(lambda s: s.rolling(3, min_periods=1).mean())
        out = out.merge(long, on=["lsoa21cd", "month"], how="left")
    return out


def _splits(df: pd.DataFrame):
    m = df["month"]
    return m.le(TRAIN_END), m.gt(TRAIN_END) & m.le(VALID_END), m.ge(TEST_START)


def _metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, float)
    y_pred = np.clip(np.asarray(y_pred, float), 1e-6, None)
    dec = max(int(len(y_true) * 0.1), 1)
    top = np.argsort(-y_pred)[:dec]
    return {
        "mae": round(mean_absolute_error(y_true, y_pred), 4),
        "rmse": round(float(np.sqrt(np.mean((y_true - y_pred) ** 2))), 4),
        "poisson_deviance": round(mean_poisson_deviance(y_true + 1e-6, y_pred), 4),
        "spearman": round(float(spearmanr(y_true, y_pred).correlation), 4),
        "top_decile_capture": round(float(y_true[top].sum() / max(y_true.sum(), 1e-9)), 4),
    }


def run() -> pd.DataFrame:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    feats = pd.read_parquet(config.CRIME_FEATURES_PATH)
    feats["month"] = feats["month"].astype(str).str[:7]
    feats = add_harm_weighted_serious_crime_score(feats)
    harm_target = f"target_{config.HARM_WEIGHTED_SERIOUS_TARGET}_next_month"
    if harm_target not in feats.columns:
        feats[harm_target] = feats.groupby("lsoa21cd", sort=False)[config.HARM_WEIGHTED_SERIOUS_TARGET].shift(-1)
    panel = pd.read_parquet(config.MODELLING_PANEL_PATH)
    panel["month"] = panel["month"].astype(str).str[:7]
    panel = add_harm_weighted_serious_crime_score(panel)
    spatial = add_spatial_features(panel)
    df = feats.merge(spatial, on=["lsoa21cd", "month"], how="left")

    safe = feature_columns(feats)["policy_independent"]
    spatial_cols = [c for c in df.columns if c.startswith("nbr_")]
    base_feats = [c for c in safe if c in df.columns]
    tr, va, te = _splits(df)

    metrics_rows = []
    forecast = df.loc[te, ["lsoa21cd", "month", "borough"]].copy()
    meta = {"targets": {}, "splits": {"train_end": TRAIN_END, "valid_end": VALID_END, "test_start": TEST_START},
            "spatial_features": spatial_cols, "excluded": "stop-search, ethnicity, fairness, trust",
            "harm_weighted_serious_crime_note": config.HARM_WEIGHTED_SERIOUS_CRIME_NOTE}

    for target in SERIOUS:
        tcol = f"target_{target}_next_month"
        if tcol not in df.columns:
            continue
        valid_rows = df[tcol].notna()
        def fit_eval(feat_list):
            models = {}
            Xtr, ytr = df.loc[tr & valid_rows, feat_list].fillna(0), df.loc[tr & valid_rows, tcol]
            Xva, yva = df.loc[va & valid_rows, feat_list].fillna(0), df.loc[va & valid_rows, tcol]
            results = {}

            results["baseline_last_month"] = (df.loc[va & valid_rows, f"{target}_lag_1"].fillna(0) if f"{target}_lag_1" in df else pd.Series(0, index=Xva.index))

            try:
                glm = PoissonRegressor(alpha=1e-3, max_iter=300).fit(Xtr, ytr)
                results["poisson_glm"] = pd.Series(glm.predict(Xva), index=Xva.index); models["poisson_glm"] = glm
            except Exception:
                pass

            rf = RandomForestRegressor(n_estimators=120, max_depth=12, n_jobs=-1, random_state=0).fit(Xtr, ytr)
            results["random_forest"] = pd.Series(rf.predict(Xva), index=Xva.index); models["random_forest"] = rf

            best, best_mae, best_params = None, np.inf, None
            for lr in (0.05, 0.1):
                for depth in (3, 5):
                    h = HistGradientBoostingRegressor(loss="poisson", learning_rate=lr, max_depth=depth,
                                                      max_iter=400, l2_regularization=1.0, random_state=0).fit(Xtr, np.clip(ytr, 0, None))
                    mae = mean_absolute_error(yva, np.clip(h.predict(Xva), 0, None))
                    if mae < best_mae:
                        best, best_mae, best_params = h, mae, {"lr": lr, "depth": depth}
            results["hgb_poisson"] = pd.Series(best.predict(Xva), index=Xva.index); models["hgb_poisson"] = best
            return results, models, best_params

        res_base, models_base, params = fit_eval(base_feats)
        res_sp, models_sp, params_sp = fit_eval(base_feats + spatial_cols)

        val_mae = {f"{k}": round(mean_absolute_error(df.loc[va & valid_rows, tcol], np.clip(v, 0, None)), 4) for k, v in res_base.items()}
        val_mae["hgb_poisson_spatial"] = round(mean_absolute_error(df.loc[va & valid_rows, tcol], np.clip(res_sp["hgb_poisson"], 0, None)), 4)


        candidates = {**{k: ("base", k) for k in res_base}, "hgb_poisson_spatial": ("spatial", "hgb_poisson")}
        best_name = min(val_mae, key=val_mae.get)
        kind, mkey = candidates[best_name]
        chosen_models, chosen_feats = (models_sp, base_feats + spatial_cols) if kind == "spatial" else (models_base, base_feats)


        Xte = df.loc[te & valid_rows, chosen_feats].fillna(0)
        if mkey in chosen_models:
            pred_te = np.clip(chosen_models[mkey].predict(Xte), 0, None)
        else:
            pred_te = df.loc[te & valid_rows, f"{target}_lag_1"].fillna(0).values
        ytrue_te = df.loc[te & valid_rows, tcol]
        tm = _metrics(ytrue_te, pred_te)
        tm.update({"target": target, "selected_model": best_name, "val_mae": val_mae[best_name]})
        metrics_rows.append(tm)


        pser = pd.Series(pred_te, index=Xte.index)
        forecast[f"pred_{target}"] = pser.reindex(forecast.index)
        forecast[f"actual_next_{target}"] = df.loc[te, tcol].reindex(forecast.index)
        meta["targets"][target] = {
            "selected_model": best_name,
            "val_mae": val_mae,
            "best_hgb_params": params_sp if kind == "spatial" else params,
            "is_harm_weighted_proxy": target == config.HARM_WEIGHTED_SERIOUS_TARGET,
        }

    forecast.to_csv(OUT_FORECASTS, index=False)
    metrics = pd.DataFrame(metrics_rows)
    meta["test_metrics"] = metrics.to_dict("records")
    (MODEL_DIR / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    run()
