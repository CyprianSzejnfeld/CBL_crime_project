from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config
from .data_interfaces import load_modelling_panel

SIMPLIFY_TOLERANCE = 0.0004


def _clean(v):
    if isinstance(v, (np.floating,)):
        v = float(v)
    if isinstance(v, (np.integer,)):
        v = int(v)
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def _records(df: pd.DataFrame) -> list[dict]:
    return [{k: _clean(v) for k, v in row.items()} for row in df.to_dict(orient="records")]





def build_reduced_geometry() -> dict:
    from shapely.geometry import mapping, shape
    from shapely.validation import make_valid

    raw = json.loads(config.LONDON_LSOA_BOUNDARIES_PATH.read_text(encoding="utf-8"))
    features = []
    invalid = 0
    for feat in raw.get("features", []):
        props = feat.get("properties") or {}
        code = props.get("LSOA21CD")
        geom = feat.get("geometry")
        if not code or not geom:
            continue
        g = shape(geom)
        if not g.is_valid:
            g = make_valid(g)
        s = g.simplify(SIMPLIFY_TOLERANCE, preserve_topology=True)
        if s.is_empty:
            s = g
            invalid += 1
        features.append({
            "type": "Feature",
            "properties": {"lsoa21cd": code, "lsoa21nm": props.get("LSOA21NM")},
            "geometry": mapping(s),
        })
    return {
        "type": "FeatureCollection",
        "features": features,
        "_meta": {"simplify_tolerance_deg": SIMPLIFY_TOLERANCE, "invalid_repaired": invalid,
                  "n_features": len(features)},
    }





def build_latest_metrics(panel: pd.DataFrame, month: str, guardrails: pd.DataFrame) -> pd.DataFrame:
    latest = panel[panel["month"] == month].copy()
    keep = [
        "lsoa21cd", "lsoa21nm", "borough",
        "stop_burden_score_0_100", "low_yield_score_0_100", "over_search_score_0_100",
        "racial_disproportionality_score_0_100", "deprivation_exposure_score_0_100",
        "excess_stop_share", "data_reliability", "borough_low_trust_flag", "trust_context_band",
        "trust_context_score_0_100", "stops_total", "rolling_stops_12m", "stop_search_no_result_rate",
        "stop_search_success_rate",
    ]
    keep = [c for c in keep if c in latest.columns]
    metrics = latest[keep].copy()

    band_cols = [c for c in guardrails.columns if c.startswith("risk_band_")]
    g = guardrails[["lsoa21cd", "crime_guardrail_level", "max_reduction_cap_from_crime_guardrail",
                    "crime_guardrail_explanation"] + band_cols]
    metrics = metrics.merge(g, on="lsoa21cd", how="left")
    return metrics


def run() -> None:
    config.APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    panel = load_modelling_panel()
    month = panel["month"].max()

    guardrails = pd.read_csv(config.CRIME_GUARDRAILS_PATH)
    guardrails = guardrails[guardrails["month"] == month]


    geo = build_reduced_geometry()
    (config.APP_DATA_DIR / "london_lsoa_geometry_reduced.geojson").write_text(json.dumps(geo))


    metrics = build_latest_metrics(panel, month, guardrails)
    (config.APP_DATA_DIR / "london_lsoa_metrics_latest.json").write_text(
        json.dumps(_records(metrics)))



if __name__ == "__main__":
    run()
