from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .smoothing import band_from_score, reliability_from_volume


def add_excess_burden_pathway(latest: pd.DataFrame) -> pd.DataFrame:
    out = latest.copy()
    for col in ["rolling_stops_12m", "population", "stop_burden_percentile", "no_result_burden_percentile", "over_search_score_0_100"]:
        if col not in out.columns:
            out[col] = 0.0
    if "stop_rate_per_1000" not in out.columns:
        population = pd.to_numeric(out["population"], errors="coerce").replace(0, np.nan)
        out["stop_rate_per_1000"] = pd.to_numeric(out["rolling_stops_12m"], errors="coerce").fillna(0) / population * 1000
        out["stop_rate_per_1000"] = out["stop_rate_per_1000"].replace([np.inf, -np.inf], np.nan).fillna(0)
    if "stop_rate_vs_london_lsoa_avg_ratio" not in out.columns:
        valid = out["stop_rate_per_1000"].replace([np.inf, -np.inf], np.nan).dropna()
        london_avg = float(valid.mean()) if not valid.empty else 0.0
        out["london_avg_lsoa_stop_rate_per_1000"] = london_avg
        out["london_normal_lsoa_stop_rate_per_1000"] = london_avg * config.OVERSEARCH_NORMAL_TOLERANCE
        out["stop_rate_vs_london_lsoa_avg_ratio"] = out["stop_rate_per_1000"] / london_avg if london_avg else 0.0
    if "excess_searches_to_london_lsoa_normal_annual" not in out.columns:
        normal_rate = out.get("london_normal_lsoa_stop_rate_per_1000", 0)
        normal_annual = normal_rate * out.get("population", 0) / 1000
        out["excess_searches_to_london_lsoa_normal_annual"] = (out["rolling_stops_12m"] - normal_annual).clip(lower=0)

    out["excess_burden_score_0_100"] = (
        0.35 * out["stop_burden_percentile"].fillna(0)
        + 0.35 * out["no_result_burden_percentile"].fillna(0)
        + 0.30 * out["over_search_score_0_100"].fillna(0)
    ).clip(0, 100)
    out["excess_burden_band"] = band_from_score(out["excess_burden_score_0_100"])
    out["excess_burden_reliability"] = reliability_from_volume(
        out["rolling_stops_12m"], config.MIN_LSOA_DISPLAY_STOPS_12M, config.MIN_LSOA_STRONG_STOPS_12M
    )
    sufficient = out["rolling_stops_12m"].ge(config.MIN_LSOA_STRONG_STOPS_12M)
    out["substantial_oversearch_flag"] = (
        out["stop_rate_vs_london_lsoa_avg_ratio"].ge(config.SUBSTANTIAL_OVERSEARCH_RATIO)
        & out["stop_burden_percentile"].ge(75)
        & out["excess_searches_to_london_lsoa_normal_annual"].ge(config.MIN_CATEGORY_STOPS_12M)
        & sufficient
    )
    out["much_oversearch_flag"] = (
        out["stop_rate_vs_london_lsoa_avg_ratio"].ge(config.MUCH_OVERSEARCH_RATIO)
        & out["stop_burden_percentile"].ge(85)
        & out["excess_searches_to_london_lsoa_normal_annual"].ge(config.MIN_CATEGORY_STOPS_12M)
        & sufficient
    )
    out["excess_burden_flag"] = out["substantial_oversearch_flag"]
    moderate = out["excess_burden_score_0_100"].ge(65) & out["rolling_stops_12m"].ge(config.MIN_LSOA_DISPLAY_STOPS_12M)
    out["excess_burden_moderate_flag"] = moderate & ~out["substantial_oversearch_flag"]

    reason = np.select(
        [
            out["much_oversearch_flag"],
            out["substantial_oversearch_flag"],
            out["excess_burden_moderate_flag"],
            out["rolling_stops_12m"].lt(config.MIN_LSOA_DISPLAY_STOPS_12M),
        ],
        [
            "LSOA search rate is at least 1.50x the London LSOA average, with enough volume.",
            "LSOA search rate is at least 1.25x the London LSOA average, with enough volume.",
            "Elevated search pressure; monitor or aggregate with neighbours.",
            "Insufficient stop volume for reliable LSOA-level excess-burden flag.",
        ],
        default="No strong excess-burden pathway signal.",
    )
    out["excess_burden_reason"] = reason
    return out
