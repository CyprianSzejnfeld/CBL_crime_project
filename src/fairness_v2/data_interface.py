from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config as base_config
from src.data_interfaces import load_modelling_panel
from . import config
from .smoothing import percentile_0_100, safe_divide


def _quarter_label(month: pd.Series) -> pd.Series:
    periods = pd.PeriodIndex(month.astype(str).str[:7], freq="M")
    return periods.asfreq("Q").astype(str)


def ensure_v2_dirs() -> None:
    base_config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _load_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def add_category_rollups(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["month"] = out["month"].astype(str).str[:7]
    out["month_dt"] = pd.PeriodIndex(out["month"], freq="M").to_timestamp()
    out = out.sort_values(["lsoa21cd", "month_dt"]).reset_index(drop=True)

    for category in config.ALL_CATEGORY_BUCKETS:
        for metric in ["stops", "arrests", "positive_outcomes", "no_result_stops"]:
            col = f"{metric}_{category}"
            if col not in out.columns:
                out[col] = 0
            roll = f"rolling_12m_{col}"
            out[roll] = (
                out.groupby("lsoa21cd", observed=True)[col]
                .transform(lambda s: pd.to_numeric(s, errors="coerce").fillna(0).rolling(12, min_periods=1).sum())
                .astype(float)
            )




    for group in config.RACIAL_GROUPS:
        for source in [f"positive_stops_{group}", f"arrest_stops_{group}"]:
            if source not in out.columns:
                out[source] = 0
            target = f"rolling_12m_{source}"
            out[target] = (
                out.groupby("lsoa21cd", observed=True)[source]
                .transform(lambda s: pd.to_numeric(s, errors="coerce").fillna(0).rolling(12, min_periods=1).sum())
                .astype(float)
            )


    for metric in ["stops", "arrests", "positive_outcomes", "no_result_stops"]:
        parts = [f"{metric}_{c}" for c in config.ATOMIC_REDUCIBLE_CATEGORIES]
        out[f"{metric}_low_yield_non_weapon"] = out[parts].sum(axis=1)
        rparts = [f"rolling_12m_{metric}_{c}" for c in config.ATOMIC_REDUCIBLE_CATEGORIES]
        out[f"rolling_12m_{metric}_low_yield_non_weapon"] = out[rparts].sum(axis=1)

    out["quarter"] = _quarter_label(out["month"])
    return out


def add_latest_percentiles(latest: pd.DataFrame) -> pd.DataFrame:
    out = latest.copy()
    out["stop_rate_per_1000"] = safe_divide(out["rolling_stops_12m"], out["population"]) * 1000
    out["no_result_stop_rate_per_1000"] = safe_divide(out["rolling_no_result_stops_12m"], out["population"]) * 1000
    out["stop_burden_percentile"] = percentile_0_100(out["stop_rate_per_1000"])
    out["no_result_burden_percentile"] = percentile_0_100(out["no_result_stop_rate_per_1000"])

    valid_rates = out.loc[out["population"].gt(0), "stop_rate_per_1000"].replace([np.inf, -np.inf], np.nan).dropna()
    london_avg_lsoa_stop_rate = float(valid_rates.mean()) if not valid_rates.empty else 0.0
    london_normal_lsoa_stop_rate = london_avg_lsoa_stop_rate * config.OVERSEARCH_NORMAL_TOLERANCE
    out["london_avg_lsoa_stop_rate_per_1000"] = london_avg_lsoa_stop_rate
    out["london_normal_lsoa_stop_rate_per_1000"] = london_normal_lsoa_stop_rate
    out["stop_rate_vs_london_lsoa_avg_ratio"] = safe_divide(out["stop_rate_per_1000"], london_avg_lsoa_stop_rate).fillna(0)
    normal_annual_stops = london_normal_lsoa_stop_rate * out["population"] / 1000
    out["excess_searches_to_london_lsoa_normal_annual"] = (out["rolling_stops_12m"] - normal_annual_stops).clip(lower=0)
    out["excess_searches_to_london_lsoa_normal_month"] = out["excess_searches_to_london_lsoa_normal_annual"] / 12

    expected = pd.to_numeric(out.get("expected_stops", 0), errors="coerce").fillna(0)
    stops_12m = pd.to_numeric(out["rolling_stops_12m"], errors="coerce").fillna(0)
    out["over_search_residual"] = (stops_12m - expected).clip(lower=0)
    out["excess_stop_share"] = safe_divide(out["over_search_residual"], stops_12m).fillna(0)
    out["over_search_residual_rate_per_1000"] = safe_divide(out["over_search_residual"], out["population"]) * 1000
    out["over_search_score_0_100"] = percentile_0_100(out["over_search_residual_rate_per_1000"]).fillna(0)

    dep_parts = []
    for col in ["imd_score", "income_score", "education_score", "deprivation_component"]:
        if col in out.columns and pd.to_numeric(out[col], errors="coerce").notna().any():
            dep_parts.append(percentile_0_100(out[col]))
    out["deprivation_percentile"] = pd.concat(dep_parts, axis=1).mean(axis=1) if dep_parts else np.nan
    return out


def load_inputs() -> pd.DataFrame:
    ensure_v2_dirs()
    panel = load_modelling_panel()
    panel = add_category_rollups(panel)

    guardrails = _load_optional_csv(base_config.CRIME_GUARDRAILS_PATH)
    if not guardrails.empty:
        guardrails["month"] = guardrails["month"].astype(str).str[:7]
        keep = [
            "lsoa21cd",
            "month",
            "crime_guardrail_level",
            "max_reduction_cap_from_crime_guardrail",
            "crime_guardrail_explanation",
        ] + [c for c in guardrails.columns if c.startswith("risk_band_")]
        panel = panel.merge(guardrails[keep], on=["lsoa21cd", "month"], how="left", suffixes=("", "_guard"))
    else:
        panel["crime_guardrail_level"] = "Low"
        panel["max_reduction_cap_from_crime_guardrail"] = 0.20

    forecasts = _load_optional_csv(base_config.CRIME_FORECASTS_PATH)
    if not forecasts.empty:
        forecasts["month"] = forecasts["month"].astype(str).str[:7]
        pred_cols = [c for c in forecasts.columns if c.startswith("pred_")]
        panel = panel.merge(
            forecasts[["lsoa21cd", "month"] + pred_cols],
            on=["lsoa21cd", "month"],
            how="left",
            suffixes=("", "_forecast"),
        )

    latest = panel.loc[panel["month"].eq(config.LATEST_MONTH)].copy()
    if latest.empty:
        latest = panel.loc[panel["month"].eq(panel["month"].max())].copy()
    latest = add_latest_percentiles(latest)
    return latest
