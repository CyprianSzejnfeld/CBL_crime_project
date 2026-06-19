from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from . import paths

MAP_METRIC_FIELDS = [
    "borough",
    "low_yield_score_0_100", "over_search_score_0_100", "stop_burden_score_0_100",
    "racial_disproportionality_score_0_100", "crime_guardrail_level",
    "max_reduction_cap_from_crime_guardrail", "borough_low_trust_flag",
    "trust_context_band", "data_reliability",
    "risk_band_total_crime_count", "risk_band_violence_count", "risk_band_drugs_count",
    "risk_band_robbery_count", "risk_band_possession_of_weapons_count",
    "overall_review_priority", "fairness_review_status", "intervention_actionability_status",
    "criticalness_score", "criticalness_level",
    "actionable_fairness_score_0_100", "fairness_reduction_priority",
    "v2_reduction_candidate_flag", "dominant_unfairness_pattern", "v2_score_explanation",
    "deprivation_oversearch_path_flag", "deprivation_oversearch_low_yield_flag", "racial_oversearch_low_yield_flag",
    "extreme_oversearch_low_yield_flag",
    "any_fairness_pathway_flag", "number_of_pathways_flagged", "fairness_pathway_labels",
    "substantial_oversearch_flag", "much_oversearch_flag", "deprivation_trait_flag",
    "oversearch_trait_flag", "much_oversearch_trait_flag", "low_yield_trait_flag", "racial_trait_flag",
    "very_low_yield_trait_flag", "monitor_trait_count", "monitor_trait_labels",
    "independent_flag_count", "any_independent_flag",
    "any_racial_pathway_flag",
    "excess_burden_score_0_100", "excess_burden_flag",
    "deprivation_burden_score_0_100", "deprivation_burden_flag",
    "racial_pathway_black_score_0_100", "racial_pathway_black_flag",
    "racial_pathway_asian_score_0_100", "racial_pathway_asian_flag",
    "racial_pathway_mixed_score_0_100", "racial_pathway_mixed_flag",
    "racial_pathway_other_score_0_100", "racial_pathway_other_flag",
    "racial_pathway_white_score_0_100", "racial_pathway_white_flag",
    "low_yield_actionability_score_0_100", "low_yield_actionability_flag", "very_low_yield_actionability_flag",
    "trust_context_warning_flag", "resident_denominator_caution_flag",
    "eligible_reduction_categories",
    "very_low_yield_categories",
    "stop_rate_per_1000", "london_avg_lsoa_stop_rate_per_1000", "london_normal_lsoa_stop_rate_per_1000",
    "stop_rate_vs_london_lsoa_avg_ratio", "excess_searches_to_london_lsoa_normal_annual",
    "excess_searches_to_london_lsoa_normal_month",
]


def _mtime_ns(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except FileNotFoundError:
        return 0


@lru_cache(maxsize=1)
def geometry() -> dict:
    return json.loads(paths.GEOMETRY.read_text())


def metrics_by_lsoa() -> dict[str, dict]:
    return _metrics_by_lsoa_cached(
        _mtime_ns(paths.METRICS_JSON),
        _mtime_ns(paths.FAIRNESS_V2_LSOA_PATHWAYS),
    )


@lru_cache(maxsize=8)
def _metrics_by_lsoa_cached(_metrics_mtime: int, _fairness_mtime: int) -> dict[str, dict]:
    rows = json.loads(paths.METRICS_JSON.read_text())
    metrics = {r["lsoa21cd"]: r for r in rows}
    if paths.FAIRNESS_V2_LSOA_PATHWAYS.exists():
        v2 = pd.read_csv(paths.FAIRNESS_V2_LSOA_PATHWAYS, low_memory=False)
        for rec in v2.to_dict(orient="records"):
            code = rec.get("lsoa21cd")
            if code in metrics:
                metrics[code].update(_strip_deprivation_only_monitor(_clean_record(rec)))
    return metrics


def _strip_deprivation_only_monitor(rec: dict) -> dict:
    labels_raw = rec.get("monitor_trait_labels")
    if labels_raw and isinstance(labels_raw, str):
        parts = [p.strip() for p in labels_raw.split(";") if p.strip()]
        kept = [p for p in parts if p != "High deprivation"]
        if len(kept) != len(parts):
            rec["monitor_trait_labels"] = ";".join(kept) if kept else None
            count = int(rec.get("monitor_trait_count") or 0) - (len(parts) - len(kept))
            rec["monitor_trait_count"] = max(count, 0)
            ind = rec.get("independent_flag_count")
            if ind is not None:
                rec["independent_flag_count"] = max(int(ind) - (len(parts) - len(kept)), 0)
                rec["any_independent_flag"] = bool(rec["independent_flag_count"])
            pathways = int(rec.get("number_of_pathways_flagged") or 0)
            new_count = rec["monitor_trait_count"]
            if pathways == 0:
                if new_count >= 1:
                    rec["criticalness_level"] = "Monitor trait"
                    rec["criticalness_score"] = 0.5
                else:
                    rec["criticalness_level"] = "No signal"
                    rec["criticalness_score"] = 0.0


                    rec["overall_review_priority"] = "No signal"
                    rec["fairness_review_status"] = "No signal"
    return rec


def map_feature_collection() -> dict:
    return _map_feature_collection_cached(
        _mtime_ns(paths.GEOMETRY),
        _mtime_ns(paths.METRICS_JSON),
        _mtime_ns(paths.FAIRNESS_V2_LSOA_PATHWAYS),
    )


@lru_cache(maxsize=4)
def _map_feature_collection_cached(
    _geometry_mtime: int,
    _metrics_mtime: int,
    _fairness_mtime: int,
) -> dict:
    geo = geometry()
    metrics = metrics_by_lsoa()
    features = []
    for feat in geo["features"]:
        code = feat["properties"]["lsoa21cd"]
        m = metrics.get(code, {})
        props = {"lsoa21cd": code, "lsoa21nm": feat["properties"].get("lsoa21nm")}
        for k in MAP_METRIC_FIELDS:
            props[k] = m.get(k)
        features.append({"type": "Feature", "properties": props, "geometry": feat["geometry"]})
    return {"type": "FeatureCollection", "features": features}


def lsoa_detail(lsoa21cd: str) -> dict | None:
    m = metrics_by_lsoa().get(lsoa21cd)
    if m is None:
        return None
    return dict(m)


def _clean_record(rec: dict) -> dict:
    out = {}
    for key, val in rec.items():
        if pd.isna(val):
            out[key] = None
        elif hasattr(val, "item"):
            out[key] = val.item()
        else:
            out[key] = val
    return out
