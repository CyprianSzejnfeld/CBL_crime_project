from __future__ import annotations

import json
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd
import requests
from shapely.geometry import mapping, shape
from shapely.ops import unary_union
from shapely.strtree import STRtree

from src import config
from src.fairness_v2.data_interface import add_category_rollups
from src.fairness_v2.smoothing import beta_posterior_summary, percentile_0_100, safe_divide


LOOKUP_SERVICE = (
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
    "LSOA21_WD24_LAD24_EW_LU/FeatureServer/0/query"
)
WARD_BOUNDARY_SERVICE = (
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
    "Wards_May_2024_Boundaries_UK_BGC/FeatureServer/0/query"
)

WARD_LOOKUP_PATH = config.INTERIM_DIR / "lsoa21_ward24_lookup_london.csv"
WARD_BOUNDARIES_PATH = config.RAW_GEO_DIR / "london_ward24_boundaries.geojson"
WARD_METRICS_CSV = config.PROCESSED_DIR / "ward_quarter_intervention_metrics.csv"
WARD_METRICS_PARQUET = config.PROCESSED_DIR / "ward_quarter_intervention_metrics.parquet"
WARD_CLUSTERS_CSV = config.PROCESSED_DIR / "ward_intervention_clusters.csv"

MIN_QUARTERLY_SEARCHES_REDUCED = config.WARD_CLUSTER_MIN_QUARTERLY_SEARCHES_REDUCED
MIN_QUARTERLY_NO_RESULT_STOPS_AVOIDED = config.WARD_CLUSTER_MIN_QUARTERLY_NO_RESULT_STOPS_AVOIDED
PROTECTED_CATEGORY = config.WARD_CLUSTER_PROTECTED_CATEGORY
REDUCIBLE_CATEGORIES = ["drugs", "stolen_property", "other_non_weapon", "offensive_weapons", "low_yield_non_weapon"]
WARD_MIN_DISPLAY_STOPS_12M = max(30, config.FAIRNESS_V2_MIN_LSOA_DISPLAY_STOPS_12M * 2)
WARD_MIN_STRONG_STOPS_12M = max(60, config.FAIRNESS_V2_MIN_LSOA_STRONG_STOPS_12M * 3)
WARD_MIN_CATEGORY_STOPS_12M = max(40, config.FAIRNESS_V2_MIN_CATEGORY_STOPS_12M * 2)
WARD_MIN_KNOWN_ETHNICITY_STOPS_12M = max(60, config.FAIRNESS_V2_MIN_KNOWN_ETHNICITY_STOPS_12M * 2)
WARD_MIN_GROUP_STOPS_12M = max(30, config.FAIRNESS_V2_MIN_GROUP_STOPS_12M * 2)
WARD_MIN_GROUP_POP_SHARE = config.FAIRNESS_V2_MIN_GROUP_POP_SHARE
WARD_OVERSEARCH_NORMAL_TOLERANCE = 1.25
WARD_SUBSTANTIAL_OVERSEARCH_RATIO = 1.25
WARD_MUCH_OVERSEARCH_RATIO = 1.50
WARD_LOW_YIELD_MARGIN = 0.05
WARD_VERY_LOW_YIELD_MARGIN = 0.10
RACIAL_GROUPS = ["black", "asian", "mixed", "other", "white"]
RACIAL_GROUP_LABELS = {
    "black": "Black",
    "asian": "Asian",
    "mixed": "Mixed",
    "other": "Other",
    "white": "White",
}


def _quarter_label(month: pd.Series) -> pd.Series:
    return pd.PeriodIndex(month.astype(str).str[:7], freq="M").asfreq("Q").astype(str)


def _download_lookup(lad_codes: list[str]) -> pd.DataFrame:
    WARD_LOOKUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    if WARD_LOOKUP_PATH.exists():
        return pd.read_csv(WARD_LOOKUP_PATH)

    chunks = []
    for i in range(0, len(lad_codes), 8):
        subset = lad_codes[i : i + 8]
        where = "LAD24CD IN (" + ",".join([f"'{x}'" for x in subset]) + ")"
        offset = 0
        while True:
            params = {
                "where": where,
                "outFields": "LSOA21CD,LSOA21NM,WD24CD,WD24NM,LAD24CD,LAD24NM",
                "returnGeometry": "false",
                "f": "json",
                "resultOffset": offset,
                "resultRecordCount": 2000,
                "orderByFields": "ObjectId",
            }
            payload = requests.get(LOOKUP_SERVICE, params=params, timeout=60).json()
            feats = payload.get("features", [])
            chunks.extend([f["attributes"] for f in feats])
            if not payload.get("exceededTransferLimit") or not feats:
                break
            offset += len(feats)
    if not chunks:
        raise RuntimeError("Ward lookup missing: ONS LSOA21->WD24 lookup download returned no rows.")
    df = pd.DataFrame(chunks).rename(
        columns={
            "LSOA21CD": "lsoa21cd",
            "LSOA21NM": "lsoa21nm_lookup",
            "WD24CD": "ward_code",
            "WD24NM": "ward_name",
            "LAD24CD": "lad24cd",
            "LAD24NM": "borough_lookup",
        }
    )
    df.to_csv(WARD_LOOKUP_PATH, index=False)
    return df


def _download_ward_boundaries(ward_codes: list[str]) -> dict:
    WARD_BOUNDARIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    if WARD_BOUNDARIES_PATH.exists():
        return json.loads(WARD_BOUNDARIES_PATH.read_text())

    features = []
    for i in range(0, len(ward_codes), 80):
        subset = ward_codes[i : i + 80]
        where = "WD24CD IN (" + ",".join([f"'{x}'" for x in subset]) + ")"
        params = {
            "where": where,
            "outFields": "WD24CD,WD24NM",
            "returnGeometry": "true",
            "outSR": 4326,
            "f": "geojson",
        }
        payload = requests.get(WARD_BOUNDARY_SERVICE, params=params, timeout=90).json()
        features.extend(payload.get("features", []))
    if not features:
        raise RuntimeError("Ward boundary missing: ONS WD24 boundary download returned no rows.")
    fc = {"type": "FeatureCollection", "features": features}
    WARD_BOUNDARIES_PATH.write_text(json.dumps(fc), encoding="utf-8")
    return fc


def _mode(values: pd.Series, default: str = "") -> str:
    vals = [v for v in values.dropna().astype(str) if v and v != "nan"]
    return Counter(vals).most_common(1)[0][0] if vals else default


def _guardrail(levels: pd.Series) -> tuple[str, float]:
    vals = levels.fillna("Low").astype(str)
    if vals.eq("Severe").any():
        return "Severe", 0.0
    if vals.eq("High").mean() >= 0.25:
        return "High", 0.05
    if vals.isin(["High", "Medium"]).any():
        return "Medium", 0.10
    return "Low", 0.20


def _category_for_group(grp: pd.DataFrame, eligible_categories: str = "") -> tuple[str | None, float, float, float, float, float]:
    best = None
    eligible = {x.strip() for x in str(eligible_categories or "").split(";") if x.strip()}
    for cat in REDUCIBLE_CATEGORIES:
        stops = float(grp[f"stops_{cat}"].sum()) if f"stops_{cat}" in grp else 0.0
        no_result = float(grp[f"no_result_stops_{cat}"].sum()) if f"no_result_stops_{cat}" in grp else 0.0
        positive = float(grp[f"positive_outcomes_{cat}"].sum()) if f"positive_outcomes_{cat}" in grp else 0.0
        arrests = float(grp[f"arrests_{cat}"].sum()) if f"arrests_{cat}" in grp else 0.0
        if cat == "low_yield_non_weapon":
            stops = float(grp[["stops_drugs", "stops_stolen_property", "stops_other_non_weapon"]].sum().sum())
            no_result = float(grp[["no_result_stops_drugs", "no_result_stops_stolen_property", "no_result_stops_other_non_weapon"]].sum().sum())
            positive = float(grp[["positive_outcomes_drugs", "positive_outcomes_stolen_property", "positive_outcomes_other_non_weapon"]].sum().sum())
            arrests = float(grp[["arrests_drugs", "arrests_stolen_property", "arrests_other_non_weapon"]].sum().sum())
        if stops <= 0:
            continue
        low_yield_rate = no_result / max(stops, 1e-9)
        score = (1000 if cat in eligible else 0) + no_result + stops * low_yield_rate
        item = (score, cat, stops, no_result, positive, arrests, low_yield_rate)
        if best is None or item > best:
            best = item
    if best is None:
        return None, 0, 0, 0, 0, 0
    _, cat, stops, no_result, positive, arrests, rate = best
    return cat, stops, no_result, positive, arrests, rate


def _merge_group_no_result_counts(panel: pd.DataFrame) -> pd.DataFrame:
    group_path = config.PROCESSED_DIR / "london_lsoa_month_ethnicity_outcomes_2021_2025.parquet"
    if not group_path.exists():
        return panel
    gpanel = pd.read_parquet(group_path)
    gpanel["month"] = gpanel["month"].astype(str).str[:7]
    nr_cols = [c for c in gpanel.columns if c.startswith("no_result_stops_")]
    if not nr_cols:
        return panel
    out = panel.copy()
    out["month"] = out["month"].astype(str).str[:7]
    out = out.merge(gpanel[["lsoa21cd", "month", *nr_cols]], on=["lsoa21cd", "month"], how="left")
    for col in nr_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    return out


def _positive_percentile(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0)
    out = pd.Series(0.0, index=values.index)
    pos = values.gt(0)
    if pos.any():
        out.loc[pos] = percentile_0_100(values.loc[pos]).fillna(0)
    return out


def _weighted_average_by_ward(latest: pd.DataFrame, value_col: str, weight_col: str = "population") -> pd.Series:
    if value_col not in latest.columns:
        return pd.Series(dtype="float64")
    values = pd.to_numeric(latest[value_col], errors="coerce")
    weights = pd.to_numeric(latest.get(weight_col, 1), errors="coerce").fillna(0)
    work = pd.DataFrame({"ward_code": latest["ward_code"], "value": values, "weight": weights})
    work = work[work["value"].notna()]
    if work.empty:
        return pd.Series(dtype="float64")
    numerator = (work["value"] * work["weight"]).groupby(work["ward_code"]).sum()
    denominator = work["weight"].groupby(work["ward_code"]).sum().replace(0, np.nan)
    return numerator / denominator


def _ward_direct_fairness(latest: pd.DataFrame) -> pd.DataFrame:
    work = latest.copy()
    for col in ["population", "total_population"]:
        if col not in work.columns:
            work[col] = 0
    if pd.to_numeric(work["population"], errors="coerce").fillna(0).sum() <= 0:
        work["population"] = pd.to_numeric(work["total_population"], errors="coerce").fillna(0)

    sum_cols = [
        "population",
        "total_population",
        "rolling_stops_12m",
        "rolling_no_result_stops_12m",
        "rolling_arrests_12m",
        "rolling_positive_outcomes_12m",
        "rolling_stops_known_ethnicity_12m",
        "expected_stops",
    ]
    for group in RACIAL_GROUPS:
        sum_cols.extend(
            [
                f"{group}_population",
                f"rolling_stops_{group}_12m",
                f"rolling_12m_positive_stops_{group}",
                f"rolling_12m_arrest_stops_{group}",
                f"rolling_12m_no_result_stops_{group}",
            ]
        )
    for category in REDUCIBLE_CATEGORIES:
        for metric in ["stops", "positive_outcomes", "arrests", "no_result_stops"]:
            sum_cols.append(f"rolling_12m_{metric}_{category}")
    for col in sum_cols:
        if col not in work.columns:
            work[col] = 0.0
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)
    for col in ["borough_low_trust_flag", "resident_denominator_caution_flag"]:
        if col not in work.columns:
            work[col] = False
    if "trust_context_band" not in work.columns:
        work["trust_context_band"] = ""

    agg = {col: (col, "sum") for col in sum_cols}
    agg.update(
        {
            "ward_name": ("ward_name", "first"),
            "borough": ("borough", "first"),
            "borough_low_trust_flag": ("borough_low_trust_flag", "max"),
            "trust_context_band": ("trust_context_band", "first"),
            "resident_denominator_caution_flag": ("resident_denominator_caution_flag", "max"),
        }
    )
    ward = work.groupby("ward_code", observed=True).agg(**agg).reset_index()
    ward["population"] = ward["population"].where(ward["population"].gt(0), ward["total_population"])
    ward["stop_rate_per_1000"] = safe_divide(ward["rolling_stops_12m"], ward["population"]) * 1000
    ward["no_result_stop_rate_per_1000"] = safe_divide(ward["rolling_no_result_stops_12m"], ward["population"]) * 1000
    ward["stop_burden_percentile"] = percentile_0_100(ward["stop_rate_per_1000"]).fillna(0)
    ward["no_result_burden_percentile"] = percentile_0_100(ward["no_result_stop_rate_per_1000"]).fillna(0)

    dep_parts = []
    for col in ["imd_score", "income_score", "education_score", "deprivation_component"]:
        avg = _weighted_average_by_ward(work, col)
        if not avg.empty:
            ward[col] = ward["ward_code"].map(avg)
            dep_parts.append(percentile_0_100(ward[col]))
    ward["deprivation_percentile"] = pd.concat(dep_parts, axis=1).mean(axis=1).fillna(0) if dep_parts else 0.0

    valid_rates = ward.loc[ward["population"].gt(0), "stop_rate_per_1000"].replace([np.inf, -np.inf], np.nan).dropna()
    london_avg_ward_stop_rate = float(valid_rates.mean()) if not valid_rates.empty else 0.0
    london_normal_ward_stop_rate = london_avg_ward_stop_rate * WARD_OVERSEARCH_NORMAL_TOLERANCE
    ward["london_avg_ward_stop_rate_per_1000"] = london_avg_ward_stop_rate
    ward["london_normal_ward_stop_rate_per_1000"] = london_normal_ward_stop_rate
    ward["stop_rate_vs_london_avg_ratio"] = safe_divide(ward["stop_rate_per_1000"], london_avg_ward_stop_rate).fillna(0)
    normal_annual_stops = london_normal_ward_stop_rate * ward["population"] / 1000
    ward["excess_searches_to_london_normal_annual"] = (ward["rolling_stops_12m"] - normal_annual_stops).clip(lower=0)
    ward["excess_searches_to_london_normal_quarter"] = ward["excess_searches_to_london_normal_annual"] / 4

    expected = pd.to_numeric(ward["expected_stops"], errors="coerce").fillna(0)
    stops_12m = pd.to_numeric(ward["rolling_stops_12m"], errors="coerce").fillna(0)
    ward["over_search_residual"] = (stops_12m - expected).clip(lower=0)
    ward["over_search_ratio"] = safe_divide(stops_12m, expected).fillna(0)
    ward["excess_stop_share"] = safe_divide(ward["over_search_residual"], stops_12m).fillna(0)
    ward["over_search_residual_rate_per_1000"] = safe_divide(ward["over_search_residual"], ward["population"]) * 1000
    ward["over_search_score_0_100"] = _positive_percentile(ward["over_search_residual_rate_per_1000"]).fillna(0)

    sufficient = stops_12m.ge(WARD_MIN_STRONG_STOPS_12M)
    ward["excess_burden_score_0_100"] = (
        0.35 * ward["stop_burden_percentile"]
        + 0.35 * ward["no_result_burden_percentile"]
        + 0.30 * ward["over_search_score_0_100"]
    ).clip(0, 100)
    ward["substantial_oversearch_flag"] = (
        ward["stop_rate_vs_london_avg_ratio"].ge(WARD_SUBSTANTIAL_OVERSEARCH_RATIO)
        & ward["stop_burden_percentile"].ge(75)
        & ward["excess_searches_to_london_normal_annual"].ge(WARD_MIN_CATEGORY_STOPS_12M)
        & sufficient
    )
    ward["much_oversearch_flag"] = (
        ward["stop_rate_vs_london_avg_ratio"].ge(WARD_MUCH_OVERSEARCH_RATIO)
        & ward["stop_burden_percentile"].ge(85)
        & ward["excess_searches_to_london_normal_annual"].ge(WARD_MIN_CATEGORY_STOPS_12M)
        & sufficient
    )
    ward["excess_burden_flag"] = ward["substantial_oversearch_flag"]
    ward["excess_burden_moderate_flag"] = (
        ward["excess_burden_score_0_100"].ge(65)
        & stops_12m.ge(WARD_MIN_DISPLAY_STOPS_12M)
        & ~ward["substantial_oversearch_flag"]
    )

    highly_deprived = ward["deprivation_percentile"].ge(80)
    ward["deprivation_burden_score_0_100"] = (
        0.35 * ward["deprivation_percentile"]
        + 0.30 * ward["stop_burden_percentile"]
        + 0.20 * ward["no_result_burden_percentile"]
        + 0.15 * ward["over_search_score_0_100"]
    ).clip(0, 100)
    ward["deprivation_burden_flag"] = (
        highly_deprived
        & ward["substantial_oversearch_flag"]
    )
    ward["deprivation_burden_moderate_flag"] = (
        highly_deprived
        & ward["deprivation_burden_score_0_100"].ge(65)
        & stops_12m.ge(WARD_MIN_DISPLAY_STOPS_12M)
        & ~ward["deprivation_burden_flag"]
    )

    low_yield_scores = []
    low_yield_flags = []
    very_low_yield_flags = []
    for category in REDUCIBLE_CATEGORIES:
        stops = pd.to_numeric(ward[f"rolling_12m_stops_{category}"], errors="coerce").fillna(0)
        positives = pd.to_numeric(ward[f"rolling_12m_positive_outcomes_{category}"], errors="coerce").fillna(0)
        no_result = pd.to_numeric(ward[f"rolling_12m_no_result_stops_{category}"], errors="coerce").fillna(0)
        benchmark_success = positives.sum() / stops.sum() if stops.sum() else np.nan
        benchmark_no_result = no_result.sum() / stops.sum() if stops.sum() else np.nan
        benchmark_success_clean = 0.0 if pd.isna(benchmark_success) else float(benchmark_success)
        benchmark_no_result_clean = 0.0 if pd.isna(benchmark_no_result) else float(benchmark_no_result)
        success_rate = safe_divide(positives, stops).fillna(0)
        no_result_rate = safe_divide(no_result, stops).fillna(0)
        success_post = beta_posterior_summary(positives, stops, benchmark_success, alpha=config.FAIRNESS_V2_EMPIRICAL_BAYES_ALPHA)
        no_result_post = beta_posterior_summary(no_result, stops, benchmark_no_result, alpha=config.FAIRNESS_V2_EMPIRICAL_BAYES_ALPHA)
        evidence = np.maximum(
            success_post["probability_rate_below_benchmark"].fillna(0),
            no_result_post["probability_rate_above_benchmark"].fillna(0),
        )
        low_yield_margin = (
            no_result_rate.ge(benchmark_no_result_clean + WARD_LOW_YIELD_MARGIN)
            | success_rate.le(max(benchmark_success_clean - WARD_LOW_YIELD_MARGIN, 0))
        )
        very_low_yield_margin = (
            no_result_rate.ge(benchmark_no_result_clean + WARD_VERY_LOW_YIELD_MARGIN)
            | success_rate.le(max(benchmark_success_clean - WARD_VERY_LOW_YIELD_MARGIN, 0))
        )
        volume_pct = percentile_0_100(stops).fillna(0)
        no_result_volume_pct = percentile_0_100(no_result).fillna(0)
        score = (0.40 * no_result_volume_pct + 0.25 * volume_pct + 0.35 * evidence * 100).clip(0, 100)
        flag = stops.ge(WARD_MIN_CATEGORY_STOPS_12M) & evidence.ge(config.FAIRNESS_V2_LOW_YIELD_PROB_THRESHOLD) & low_yield_margin
        very_low_flag = stops.ge(WARD_MIN_CATEGORY_STOPS_12M) & evidence.ge(0.90) & very_low_yield_margin
        ward[f"{category}_low_yield_probability"] = evidence
        ward[f"{category}_no_result_rate"] = no_result_rate
        ward[f"{category}_london_no_result_rate"] = benchmark_no_result_clean
        ward[f"{category}_success_rate"] = success_rate
        ward[f"{category}_london_success_rate"] = benchmark_success_clean
        ward[f"{category}_low_yield_actionability_score_0_100"] = score
        ward[f"{category}_eligible_reduction_category"] = flag
        ward[f"{category}_very_low_yield_flag"] = very_low_flag
        low_yield_scores.append(score.rename(category))
        low_yield_flags.append(flag.rename(category))
        very_low_yield_flags.append(very_low_flag.rename(category))

    score_matrix = pd.concat(low_yield_scores, axis=1) if low_yield_scores else pd.DataFrame(index=ward.index)
    flag_matrix = pd.concat(low_yield_flags, axis=1) if low_yield_flags else pd.DataFrame(index=ward.index)
    very_flag_matrix = pd.concat(very_low_yield_flags, axis=1) if very_low_yield_flags else pd.DataFrame(index=ward.index)
    ward["eligible_reduction_categories"] = [
        ";".join([cat for cat in flag_matrix.columns if bool(flag_matrix.loc[idx, cat])])
        for idx in ward.index
    ]
    ward["very_low_yield_categories"] = [
        ";".join([cat for cat in very_flag_matrix.columns if bool(very_flag_matrix.loc[idx, cat])])
        for idx in ward.index
    ]
    ward["low_yield_actionability_score_0_100"] = score_matrix.max(axis=1).fillna(0) if not score_matrix.empty else 0.0
    ward["low_yield_actionability_flag"] = ward["eligible_reduction_categories"].astype(str).ne("")
    ward["very_low_yield_actionability_flag"] = ward["very_low_yield_categories"].astype(str).ne("")

    racial_flags = []
    racial_moderates = []
    racial_scores = []
    known = pd.to_numeric(ward["rolling_stops_known_ethnicity_12m"], errors="coerce").fillna(0)
    for group in RACIAL_GROUPS:
        label = RACIAL_GROUP_LABELS[group]
        group_stops = pd.to_numeric(ward[f"rolling_stops_{group}_12m"], errors="coerce").fillna(0)
        group_pop = pd.to_numeric(ward[f"{group}_population"], errors="coerce").fillna(0)
        positives = pd.to_numeric(ward[f"rolling_12m_positive_stops_{group}"], errors="coerce").fillna(0)
        group_no_result = pd.to_numeric(ward[f"rolling_12m_no_result_stops_{group}"], errors="coerce").fillna(0)
        if group_no_result.sum() <= 0:
            group_no_result = (group_stops - positives).clip(lower=0)
        group_share = safe_divide(group_pop, ward["population"]).fillna(0)
        stop_share = safe_divide(group_stops, known).fillna(0)
        ratio = safe_divide(stop_share, group_share).clip(upper=5).fillna(0)
        group_rate = safe_divide(group_stops, group_pop) * 1000
        group_burden_pct = percentile_0_100(group_rate).fillna(0)
        benchmark_success = positives.sum() / group_stops.sum() if group_stops.sum() else np.nan
        benchmark_no_result = group_no_result.sum() / group_stops.sum() if group_stops.sum() else np.nan
        group_success_rate = safe_divide(positives, group_stops).fillna(0)
        group_no_result_rate = safe_divide(group_no_result, group_stops).fillna(0)
        success_post = beta_posterior_summary(positives, group_stops, benchmark_success, alpha=config.FAIRNESS_V2_EMPIRICAL_BAYES_ALPHA)
        no_result_post = beta_posterior_summary(group_no_result, group_stops, benchmark_no_result, alpha=config.FAIRNESS_V2_EMPIRICAL_BAYES_ALPHA)
        evidence_score = np.maximum(
            success_post["probability_rate_below_benchmark"].fillna(0),
            no_result_post["probability_rate_above_benchmark"].fillna(0),
        )
        benchmark_success_clean = 0.0 if pd.isna(benchmark_success) else float(benchmark_success)
        benchmark_no_result_clean = 0.0 if pd.isna(benchmark_no_result) else float(benchmark_no_result)
        low_yield_gap = (
            group_no_result_rate.ge(benchmark_no_result_clean + WARD_LOW_YIELD_MARGIN)
            | group_success_rate.le(max(benchmark_success_clean - WARD_LOW_YIELD_MARGIN, 0))
        )
        evidence = evidence_score.ge(config.FAIRNESS_V2_LOW_YIELD_PROB_THRESHOLD) & low_yield_gap
        eligible = group_share.ge(WARD_MIN_GROUP_POP_SHARE) & known.ge(WARD_MIN_KNOWN_ETHNICITY_STOPS_12M) & group_stops.ge(
            WARD_MIN_GROUP_STOPS_12M
        )
        flag = eligible & ratio.ge(1.5) & group_burden_pct.ge(75) & evidence
        moderate = eligible & ratio.ge(1.25) & group_burden_pct.ge(60) & ~flag
        score = (
            0.40 * percentile_0_100(ratio).fillna(0)
            + 0.35 * group_burden_pct
            + 0.25
            * (
                evidence_score * 100
            )
        ).clip(0, 100)
        prefix = f"racial_pathway_{group}"
        ward[f"{prefix}_score_0_100"] = score
        ward[f"{prefix}_flag"] = flag
        ward[f"{prefix}_moderate_flag"] = moderate
        ward[f"{prefix}_low_yield_probability"] = evidence_score
        ward[f"{prefix}_low_yield_gap_flag"] = low_yield_gap
        ward[f"{prefix}_excess_known_ethnicity_stops_annual"] = (
            group_stops - (group_share * known * WARD_OVERSEARCH_NORMAL_TOLERANCE)
        ).clip(lower=0)
        ward[f"{group}_racial_disproportionality_ratio_capped"] = ratio
        ward[f"{group}_group_stop_burden_percentile"] = group_burden_pct
        ward[f"{prefix}_reason"] = np.where(
            flag,
            f"{label} stop exposure is disproportionate at ward level and yield is not clearly higher than benchmark.",
            f"No strong {label} ward-level racial pathway signal.",
        )
        racial_flags.append(flag.rename(group))
        racial_moderates.append(moderate.rename(group))
        racial_scores.append(score.rename(group))

    racial_flag_matrix = pd.concat(racial_flags, axis=1) if racial_flags else pd.DataFrame(index=ward.index)
    racial_moderate_matrix = pd.concat(racial_moderates, axis=1) if racial_moderates else pd.DataFrame(index=ward.index)
    racial_score_matrix = pd.concat(racial_scores, axis=1) if racial_scores else pd.DataFrame(index=ward.index)
    racial_score = racial_score_matrix.max(axis=1).fillna(0) if not racial_score_matrix.empty else pd.Series(0, index=ward.index)
    ward["any_racial_pathway_flag"] = racial_flag_matrix.any(axis=1) if not racial_flag_matrix.empty else False
    ward["any_racial_pathway_moderate_flag"] = racial_moderate_matrix.any(axis=1) if not racial_moderate_matrix.empty else False
    ward["racial_oversearch_low_yield_groups"] = [
        "; ".join([RACIAL_GROUP_LABELS.get(group, group.title()) for group in racial_flag_matrix.columns if bool(racial_flag_matrix.loc[idx, group])])
        for idx in ward.index
    ] if not racial_flag_matrix.empty else ""
    ward["racial_monitor_groups"] = [
        "; ".join([RACIAL_GROUP_LABELS.get(group, group.title()) for group in racial_moderate_matrix.columns if bool(racial_moderate_matrix.loc[idx, group])])
        for idx in ward.index
    ] if not racial_moderate_matrix.empty else ""

    ward["oversearch_low_yield_path_flag"] = (
        ward["much_oversearch_flag"] & ward["very_low_yield_actionability_flag"] & sufficient
    )
    ward["deprivation_oversearch_path_flag"] = highly_deprived & ward["substantial_oversearch_flag"] & sufficient
    ward["racial_oversearch_low_yield_path_flag"] = ward["any_racial_pathway_flag"] & sufficient
    ward["deprivation_trait_flag"] = highly_deprived
    ward["oversearch_trait_flag"] = ward["substantial_oversearch_flag"]
    ward["very_low_yield_trait_flag"] = ward["very_low_yield_actionability_flag"]
    ward["low_yield_trait_flag"] = ward["very_low_yield_trait_flag"]
    ward["racial_trait_flag"] = ward["any_racial_pathway_flag"] | ward["any_racial_pathway_moderate_flag"]



    ward["deprivation_oversearch_low_yield_flag"] = ward["deprivation_oversearch_path_flag"]
    ward["extreme_oversearch_low_yield_flag"] = ward["oversearch_low_yield_path_flag"]
    ward["racial_oversearch_low_yield_flag"] = ward["racial_oversearch_low_yield_path_flag"]

    pathway_cols = [
        "oversearch_low_yield_path_flag",
        "deprivation_oversearch_path_flag",
        "racial_oversearch_low_yield_path_flag",
    ]
    moderate_cols = ["excess_burden_moderate_flag", "deprivation_burden_moderate_flag", "any_racial_pathway_moderate_flag"]
    ward["number_of_pathways_flagged"] = ward[pathway_cols].sum(axis=1).astype(int)
    trait_cols = ["oversearch_trait_flag", "very_low_yield_trait_flag"]
    ward["monitor_trait_count"] = ward[trait_cols].sum(axis=1).astype(int)
    ward["monitor_trait_labels"] = ward.apply(_ward_monitor_trait_labels, axis=1)
    ward["any_fairness_pathway_flag"] = ward["number_of_pathways_flagged"].gt(0)
    ward["fairness_pathway_labels"] = ward.apply(_ward_pathway_labels, axis=1)
    ward["criticalness_score"] = np.select(
        [
            ward["number_of_pathways_flagged"].ge(3),
            ward["number_of_pathways_flagged"].eq(2),
            ward["number_of_pathways_flagged"].eq(1),
            ward["monitor_trait_count"].ge(1),
        ],
        [3.0, 2.0, 1.0, 0.5],
        default=0.0,
    )
    ward["criticalness_level"] = np.select(
        [
            ward["number_of_pathways_flagged"].ge(3),
            ward["number_of_pathways_flagged"].eq(2),
            ward["number_of_pathways_flagged"].eq(1),
            ward["monitor_trait_count"].ge(1),
        ],
        ["Three multipaths", "Two multipaths", "One multipath", "Monitor trait"],
        default="No signal",
    )
    ward["overall_review_priority"] = np.select(
        [
            ward["number_of_pathways_flagged"].ge(2),
            ward["number_of_pathways_flagged"].eq(1),
            ward[moderate_cols].any(axis=1),
        ],
        ["Critical Review", "Priority Review", "Monitor"],
        default="No signal",
    )

    deprivation_combo = (
        0.45 * ward["deprivation_percentile"] + 0.35 * ward["stop_burden_percentile"] + 0.20 * ward["over_search_score_0_100"]
    )
    racial_combo = 0.45 * racial_score + 0.30 * ward["low_yield_actionability_score_0_100"] + 0.25 * ward["stop_burden_percentile"]
    extreme_combo = (
        0.45 * ward["over_search_score_0_100"] + 0.35 * ward["low_yield_actionability_score_0_100"] + 0.20 * ward["no_result_burden_percentile"]
    )
    base = pd.concat(
        [
            deprivation_combo.where(ward["deprivation_oversearch_low_yield_flag"], 0),
            racial_combo.where(ward["racial_oversearch_low_yield_flag"], 0),
            extreme_combo.where(ward["extreme_oversearch_low_yield_flag"], 0),
            ward["excess_burden_score_0_100"].where(ward["excess_burden_flag"], 0) * 0.80,
        ],
        axis=1,
    ).max(axis=1)
    combo_count = ward[["deprivation_oversearch_low_yield_flag", "racial_oversearch_low_yield_flag", "extreme_oversearch_low_yield_flag"]].sum(axis=1)
    trust_flag = ward["borough_low_trust_flag"].fillna(False).astype(bool) if "borough_low_trust_flag" in ward.columns else pd.Series(False, index=ward.index)
    synergy = (
        ward["deprivation_oversearch_low_yield_flag"].astype(int) * 8
        + ward["racial_oversearch_low_yield_flag"].astype(int) * 10
        + ward["extreme_oversearch_low_yield_flag"].astype(int) * 8
        + combo_count.ge(2).astype(int) * 8
        + trust_flag.astype(int) * 4
    )
    ward["actionable_fairness_score_0_100"] = (base + synergy).clip(0, 100)
    ward["fairness_reduction_priority"] = np.select(
        [
            ward["actionable_fairness_score_0_100"].ge(90),
            ward["actionable_fairness_score_0_100"].ge(80),
            ward["actionable_fairness_score_0_100"].ge(65),
            ward["actionable_fairness_score_0_100"].ge(50),
        ],
        ["Critical reduction review", "Priority reduction review", "Review candidate", "Monitor"],
        default="No reduction signal",
    )
    ward["v2_reduction_candidate_flag"] = ward["actionable_fairness_score_0_100"].ge(65)
    ward["dominant_unfairness_pattern"] = ward.apply(_ward_dominant_pattern, axis=1)
    ward["v2_score_explanation"] = ward.apply(_ward_score_explanation, axis=1)
    return ward


def _ward_pathway_labels(row: pd.Series) -> str:
    labels = []
    if row.get("oversearch_low_yield_path_flag"):
        labels.append("Over-search + very low yield")
    if row.get("deprivation_oversearch_path_flag"):
        labels.append("Deprivation + over-search")
    if row.get("racial_oversearch_low_yield_path_flag"):
        labels.append("Racial over-search + low yield")
    return "; ".join(labels)


def _ward_monitor_trait_labels(row: pd.Series) -> str:
    labels = []
    if row.get("oversearch_trait_flag"):
        labels.append("Over-search")
    if row.get("very_low_yield_trait_flag"):
        labels.append("Very low yield")
    return "; ".join(labels)


def _ward_dominant_pattern(row: pd.Series) -> str:
    if row.get("oversearch_low_yield_path_flag"):
        return "Over-search + very low yield"
    if row.get("racial_oversearch_low_yield_path_flag"):
        return "Racial over-search + low yield"
    if row.get("deprivation_oversearch_path_flag"):
        return "Deprivation + over-search"
    labels = str(row.get("fairness_pathway_labels", "") or "")
    return labels if labels else "No strong unfairness pattern"


def _ward_score_explanation(row: pd.Series) -> str:
    parts = []
    if row.get("oversearch_low_yield_path_flag"):
        parts.append("much higher search rate than the London ward average plus very low-result evidence")
    if row.get("deprivation_oversearch_path_flag"):
        parts.append("high deprivation plus a search rate above the London-normal range")
    if row.get("racial_oversearch_low_yield_path_flag"):
        parts.append("racial over-exposure plus low-result evidence for the affected group")
    if not parts:
        return "No strong ward-level fairness pathway is active."
    return "Flagged because " + "; ".join(parts) + "."


def _status(row: pd.Series) -> str:
    if not bool(row["has_reliable_fairness_pathway"]):
        return "No current intervention signal"
    if row["aggregate_crime_guardrail"] == "Severe" or row["max_reduction_cap"] <= 0:
        return "Review only due to safety/reliability constraint"
    if row["target_reduction_category"] == PROTECTED_CATEGORY or not row["target_reduction_category"]:
        return "Review only due to safety/reliability constraint"
    if (
        row["expected_searches_reduced_at_cap"] >= MIN_QUARTERLY_SEARCHES_REDUCED
        or row["expected_no_result_avoided_at_cap"] >= MIN_QUARTERLY_NO_RESULT_STOPS_AVOIDED
    ):
        return "Actionable quarterly trial candidate"
    return "Contributes to wider ward cluster"


def build_ward_metrics() -> tuple[pd.DataFrame, dict]:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    london_lsoa = pd.read_csv(config.INTERIM_DIR / "london_lsoa21_lookup.csv")
    lad_codes = sorted(london_lsoa["lad24cd"].dropna().unique())
    lookup = _download_lookup(lad_codes)
    missing = sorted(set(london_lsoa["lsoa21cd"]) - set(lookup["lsoa21cd"]))
    if missing:
        raise RuntimeError(f"Ward lookup missing {len(missing)} London LSOAs; first={missing[:5]}")

    ward_geo = _download_ward_boundaries(sorted(lookup["ward_code"].dropna().unique()))
    lsoa_latest = pd.read_csv(config.FAIRNESS_V2_LSOA_PATHWAYS_CSV, low_memory=False)
    panel = pd.read_parquet(config.MODELLING_PANEL_PATH)
    panel = _merge_group_no_result_counts(panel)
    panel = add_category_rollups(panel)
    panel["month"] = panel["month"].astype(str).str[:7]
    if "quarter" not in panel.columns:
        panel["quarter"] = _quarter_label(panel["month"])
    latest_quarter = pd.Period(config.OUTPUT_LATEST_MONTH, freq="M").asfreq("Q").strftime("%YQ%q")
    q = panel.loc[panel["quarter"].eq(latest_quarter)].copy()
    q = q.merge(lookup[["lsoa21cd", "ward_code", "ward_name"]], on="lsoa21cd", how="left")
    latest_panel = panel.loc[panel["month"].eq(config.OUTPUT_LATEST_MONTH)].copy()
    if latest_panel.empty:
        latest_panel = panel.loc[panel["month"].eq(panel["month"].max())].copy()
    latest_panel = latest_panel.merge(lookup[["lsoa21cd", "ward_code", "ward_name"]], on="lsoa21cd", how="left")
    ward_fairness = _ward_direct_fairness(latest_panel)
    ward_fairness_idx = ward_fairness.set_index("ward_code")
    latest = lsoa_latest.merge(lookup[["lsoa21cd", "ward_code", "ward_name"]], on="lsoa21cd", how="left")

    latest_cols = [
        "lsoa21cd",
        "ward_code",
        "overall_review_priority",
        "fairness_reduction_priority",
        "dominant_unfairness_pattern",
        "fairness_pathway_labels",
        "v2_reduction_candidate_flag",
        "trust_context_warning_flag",
        "resident_denominator_caution_flag",
        "eligible_reduction_categories",
        "crime_guardrail_level",
        "actionable_fairness_score_0_100",
        "over_search_score_0_100",
        "low_yield_actionability_score_0_100",
        "racial_pathway_black_score_0_100",
        "racial_pathway_asian_score_0_100",
    ]
    q = q.merge(latest[[c for c in latest_cols if c in latest.columns]], on=["lsoa21cd", "ward_code"], how="left")


    for metric in ["stops", "arrests", "positive_outcomes", "no_result_stops"]:
        parts = [f"{metric}_{c}" for c in ["drugs", "stolen_property", "other_non_weapon"]]
        q[f"{metric}_low_yield_non_weapon"] = q[parts].sum(axis=1)

    rows = []
    for (ward_code, quarter), grp in q.groupby(["ward_code", "quarter"], observed=True):
        if ward_code not in ward_fairness_idx.index:
            continue
        ward_direct = ward_fairness_idx.loc[ward_code]
        eligible_categories = str(ward_direct.get("eligible_reduction_categories", "") or "")
        guardrail, cap = _guardrail(grp["crime_guardrail_level"])
        target_cat, target_stops, target_no, target_pos, target_arr, target_no_rate = _category_for_group(grp, eligible_categories)
        pct = min(cap, 0.20)
        dominant_pathway = str(ward_direct.get("dominant_unfairness_pattern", "") or "")
        if not dominant_pathway:
            dominant_pathway = str(ward_direct.get("fairness_pathway_labels", "") or "No strong unfairness pattern")
        priority_count = int(
            grp.loc[
                grp["overall_review_priority"].isin(["Priority Review", "Critical Review"]),
                "lsoa21cd",
            ].nunique()
        )
        flagged_stops = grp.loc[grp["v2_reduction_candidate_flag"].fillna(False), "stops_total"].sum()
        total_stops = float(grp["stops_total"].sum())
        ward_any_flag = bool(ward_direct.get("any_fairness_pathway_flag", False))
        ward_priority = str(ward_direct.get("overall_review_priority", "No signal") or "No signal")
        ward_priority_flag = ward_priority in {"Priority Review", "Critical Review"}
        row = {
            "ward_code": ward_code,
            "ward_name": grp["ward_name"].dropna().iloc[0] if grp["ward_name"].notna().any() else ward_code,
            "borough": _mode(grp["borough"], ""),
            "bcu": "",
            "quarter": quarter,
            "ward_fairness_source": "direct_ward_level",
            "number_of_lsoas": int(grp["lsoa21cd"].nunique()),
            "number_of_priority_or_critical_review_lsoas": priority_count,
            "share_of_ward_stops_from_flagged_lsoas": flagged_stops / total_stops if total_stops else 0.0,
            "number_of_wards": 1,
            "number_of_fairness_flagged_wards": int(ward_any_flag),
            "number_of_priority_or_critical_review_wards": int(ward_priority_flag),
            "share_of_ward_stops_from_flagged_wards": 1.0 if ward_any_flag and total_stops else 0.0,
            "ward_overall_review_priority": ward_priority,
            "ward_fairness_pathways": str(ward_direct.get("fairness_pathway_labels", "") or ""),
            "ward_v2_score_explanation": str(ward_direct.get("v2_score_explanation", "") or ""),
            "ward_number_of_pathways_flagged": int(ward_direct.get("number_of_pathways_flagged", 0) or 0),
            "ward_monitor_trait_count": int(ward_direct.get("monitor_trait_count", 0) or 0),
            "ward_monitor_trait_labels": str(ward_direct.get("monitor_trait_labels", "") or ""),
            "ward_criticalness_score": float(ward_direct.get("criticalness_score", 0) or 0),
            "ward_criticalness_level": str(ward_direct.get("criticalness_level", "No signal") or "No signal"),
            "ward_oversearch_low_yield_path_flag": bool(ward_direct.get("oversearch_low_yield_path_flag", False)),
            "ward_deprivation_oversearch_path_flag": bool(ward_direct.get("deprivation_oversearch_path_flag", False)),
            "ward_racial_oversearch_low_yield_path_flag": bool(ward_direct.get("racial_oversearch_low_yield_path_flag", False)),
            "ward_racial_oversearch_groups": str(ward_direct.get("racial_oversearch_low_yield_groups", "") or ""),
            "ward_racial_monitor_groups": str(ward_direct.get("racial_monitor_groups", "") or ""),
            "ward_deprivation_trait_flag": bool(ward_direct.get("deprivation_trait_flag", False)),
            "ward_oversearch_trait_flag": bool(ward_direct.get("oversearch_trait_flag", False)),
            "ward_low_yield_trait_flag": bool(ward_direct.get("low_yield_trait_flag", False)),
            "ward_very_low_yield_trait_flag": bool(ward_direct.get("very_low_yield_trait_flag", False)),
            "ward_racial_trait_flag": bool(ward_direct.get("racial_trait_flag", False)),
            "ward_excess_burden_flag": bool(ward_direct.get("oversearch_low_yield_path_flag", False)),
            "ward_deprivation_burden_flag": bool(ward_direct.get("deprivation_oversearch_path_flag", False)),
            "ward_racial_pathway_flag": bool(ward_direct.get("racial_oversearch_low_yield_path_flag", False)),
            "ward_low_yield_actionability_flag": bool(ward_direct.get("oversearch_low_yield_path_flag", False)),
            "ward_substantial_oversearch_flag": bool(ward_direct.get("substantial_oversearch_flag", False)),
            "ward_much_oversearch_flag": bool(ward_direct.get("much_oversearch_flag", False)),
            "ward_standalone_low_yield_actionability_flag": bool(ward_direct.get("low_yield_actionability_flag", False)),
            "ward_very_low_yield_actionability_flag": bool(ward_direct.get("very_low_yield_actionability_flag", False)),
            "ward_low_yield_categories": eligible_categories,
            "ward_very_low_yield_categories": str(ward_direct.get("very_low_yield_categories", "") or ""),
            "ward_stop_burden_percentile": float(ward_direct.get("stop_burden_percentile", 0) or 0),
            "ward_no_result_burden_percentile": float(ward_direct.get("no_result_burden_percentile", 0) or 0),
            "ward_deprivation_percentile": float(ward_direct.get("deprivation_percentile", 0) or 0),
            "ward_over_search_score_0_100": float(ward_direct.get("over_search_score_0_100", 0) or 0),
            "ward_stop_rate_vs_london_avg_ratio": float(ward_direct.get("stop_rate_vs_london_avg_ratio", 0) or 0),
            "ward_london_avg_stop_rate_per_1000": float(ward_direct.get("london_avg_ward_stop_rate_per_1000", 0) or 0),
            "ward_london_normal_stop_rate_per_1000": float(ward_direct.get("london_normal_ward_stop_rate_per_1000", 0) or 0),
            "ward_excess_searches_to_london_normal_annual": float(ward_direct.get("excess_searches_to_london_normal_annual", 0) or 0),
            "ward_excess_searches_to_london_normal_quarter": float(ward_direct.get("excess_searches_to_london_normal_quarter", 0) or 0),
            "ward_low_yield_actionability_score_0_100": float(ward_direct.get("low_yield_actionability_score_0_100", 0) or 0),
            "total_stops": total_stops,
            "non_weapon_stops": float(grp[["stops_drugs", "stops_stolen_property", "stops_other_non_weapon"]].sum().sum()),
            "drug_stops": float(grp["stops_drugs"].sum()),
            "stolen_property_stops": float(grp["stops_stolen_property"].sum()),
            "offensive_weapon_stops": float(grp["stops_offensive_weapons"].sum()),
            "no_result_stops": float(grp["no_result_stops_total"].sum()),
            "positive_outcomes": float(grp["positive_outcomes_total"].sum()),
            "arrests": float(grp["arrests_total"].sum()),
            "expected_reducible_non_weapon_stops": float(
                grp[["stops_drugs", "stops_stolen_property", "stops_other_non_weapon"]].sum().sum()
            ),
            "expected_no_result_stops_in_reducible_categories": float(
                grp[["no_result_stops_drugs", "no_result_stops_stolen_property", "no_result_stops_other_non_weapon"]].sum().sum()
            ),
            "dominant_fairness_pathway": dominant_pathway,
            "trust_context_warning": bool(grp["trust_context_warning_flag"].fillna(False).any()),
            "resident_denominator_caution": bool(grp["resident_denominator_caution_flag"].fillna(False).any()),
            "aggregate_crime_guardrail": guardrail,
            "max_reduction_cap": cap,
            "target_reduction_category": target_cat or "",
            "quarterly_baseline_reducible_searches": target_stops,
            "target_category_no_result_rate": target_no_rate,
            "expected_searches_reduced_at_cap": target_stops * pct,
            "expected_no_result_avoided_at_cap": target_no * pct,
            "expected_positive_outcomes_forgone_at_cap": target_pos * pct,
            "expected_arrests_forgone_at_cap": target_arr * pct,
            "has_reliable_fairness_pathway": ward_any_flag or bool(ward_direct.get("v2_reduction_candidate_flag", False)),
            "mean_actionable_fairness_score_0_100": float(ward_direct.get("actionable_fairness_score_0_100", 0) or 0),
            "max_actionable_fairness_score_0_100": float(ward_direct.get("actionable_fairness_score_0_100", 0) or 0),
            "member_lsoas": ";".join(sorted(grp["lsoa21cd"].dropna().unique())),
        }
        row["ward_actionability_status"] = _status(pd.Series(row))
        rows.append(row)

    wards = pd.DataFrame(rows)
    wards.to_csv(WARD_METRICS_CSV, index=False)
    wards.to_parquet(WARD_METRICS_PARQUET, index=False)
    return wards, ward_geo


def _feature_geometries(ward_geo: dict) -> dict[str, object]:
    geoms = {}
    for feat in ward_geo["features"]:
        code = feat["properties"].get("WD24CD")
        if code:
            geoms[code] = shape(feat["geometry"])
    return geoms


def build_clusters(wards: pd.DataFrame, ward_geo: dict) -> tuple[pd.DataFrame, dict]:
    geoms = _feature_geometries(ward_geo)
    candidates = wards.loc[
        wards["ward_actionability_status"].isin(
            ["Actionable quarterly trial candidate", "Contributes to wider ward cluster", "Priority review location"]
        )
        & wards["target_reduction_category"].ne("")
        & wards["aggregate_crime_guardrail"].ne("Severe")
        & wards["ward_fairness_pathways"].fillna("").astype(str).ne("")
    ].copy()
    if candidates.empty:
        empty = pd.DataFrame()
        fc = {"type": "FeatureCollection", "features": []}
        empty.to_csv(WARD_CLUSTERS_CSV, index=False)
        return empty, fc

    features = []
    cluster_rows = []
    idx_to_code = [c for c in list(candidates["ward_code"]) if c in geoms]
    geom_list = [geoms[c] for c in idx_to_code]
    tree = STRtree(geom_list)

    graph = nx.Graph()
    graph.add_nodes_from(idx_to_code)
    ward_lookup = candidates.set_index("ward_code").to_dict(orient="index")
    geom_to_code = {i: c for i, c in enumerate(idx_to_code)}
    for geom_idx, geom in enumerate(geom_list):
        code = geom_to_code[geom_idx]
        row = ward_lookup[code]
        for hit in tree.query(geom):
            if isinstance(hit, (int, np.integer)):
                other_idx = int(hit)
                other = geom_list[other_idx]
                other_code = geom_to_code[other_idx]
            else:
                other = hit
                other_code = idx_to_code[geom_list.index(other)]
            if other_code == code:
                continue
            other_row = ward_lookup[other_code]
            if str(row["ward_fairness_pathways"] or "") != str(other_row["ward_fairness_pathways"] or ""):
                continue
            if row["aggregate_crime_guardrail"] == "Severe" or other_row["aggregate_crime_guardrail"] == "Severe":
                continue
            if geom.touches(other) or geom.intersects(other):
                graph.add_edge(code, other_code)

    cluster_no = 1
    for component in nx.connected_components(graph):
        comp = sorted(component)
        grp = candidates.loc[candidates["ward_code"].isin(comp)].copy()
        if grp.empty:
            continue
        guardrail, cap = _guardrail(grp["aggregate_crime_guardrail"])
        pct = min(cap, 0.20)
        baseline = float(grp["quarterly_baseline_reducible_searches"].sum())
        total_stops = float(grp["total_stops"].sum())
        flagged_stops = float((grp["share_of_ward_stops_from_flagged_lsoas"] * grp["total_stops"]).sum())
        flagged_ward_stops = float((grp["share_of_ward_stops_from_flagged_wards"] * grp["total_stops"]).sum())
        no_rate = float(
            np.average(
                grp["target_category_no_result_rate"].fillna(0),
                weights=grp["quarterly_baseline_reducible_searches"].clip(lower=0.01),
            )
        )
        expected_reduced = baseline * pct
        no_avoided = expected_reduced * no_rate
        if expected_reduced < MIN_QUARTERLY_SEARCHES_REDUCED and no_avoided < MIN_QUARTERLY_NO_RESULT_STOPS_AVOIDED:
            continue
        unfair_excess_annual = float(grp["ward_excess_searches_to_london_normal_annual"].sum())
        unfair_excess_quarter = float(grp["ward_excess_searches_to_london_normal_quarter"].sum())
        oversearch_low_yield_wards = int(grp["ward_oversearch_low_yield_path_flag"].fillna(False).sum())
        deprivation_oversearch_wards = int(grp["ward_deprivation_oversearch_path_flag"].fillna(False).sum())
        racial_oversearch_low_yield_wards = int(grp["ward_racial_oversearch_low_yield_path_flag"].fillna(False).sum())
        n_strong_pathways = int(
            sum(
                [
                    oversearch_low_yield_wards > 0,
                    deprivation_oversearch_wards > 0,
                    racial_oversearch_low_yield_wards > 0,
                ]
            )
        )
        union = unary_union([geoms[c] for c in comp if c in geoms])
        cluster_id = f"WCL{cluster_no:03d}"
        cluster_no += 1
        row = {
            "cluster_id": cluster_id,
            "cluster_name": f"{_mode(grp['borough'])} {_mode(grp['target_reduction_category'])} ward cluster {cluster_no - 1}",
            "member_wards": ";".join(comp),
            "member_ward_names": ";".join(grp["ward_name"].astype(str)),
            "member_lsoas": ";".join(sorted(set(";".join(grp["member_lsoas"]).split(";")))),
            "boroughs": ";".join(sorted(grp["borough"].dropna().unique())),
            "bcu": "",
            "fairness_source": "direct_ward_level",
            "dominant_fairness_pathway": _mode(grp["ward_fairness_pathways"]),
            "target_reduction_category": _mode(grp["target_reduction_category"]),
            "number_of_wards": len(comp),
            "number_of_fairness_flagged_wards": int(grp["number_of_fairness_flagged_wards"].sum()),
            "number_of_priority_or_critical_review_wards": int(grp["number_of_priority_or_critical_review_wards"].sum()),
            "share_of_stops_from_flagged_wards": flagged_ward_stops / total_stops if total_stops else 0.0,
            "ward_fairness_pathways": "; ".join(
                sorted(
                    {
                        part.strip()
                        for value in grp["ward_fairness_pathways"].fillna("").astype(str)
                        for part in value.split(";")
                        if part.strip()
                    }
                )
            ),
            "ward_review_priorities": ";".join(sorted(set(grp["ward_overall_review_priority"].dropna().astype(str)) - {""})),
            "ward_level_score_explanation": " ".join(
                sorted(set(grp["ward_v2_score_explanation"].dropna().astype(str)) - {""})
            ),
            "oversearch_low_yield_flagged_wards": oversearch_low_yield_wards,
            "deprivation_oversearch_flagged_wards": deprivation_oversearch_wards,
            "racial_oversearch_low_yield_flagged_wards": racial_oversearch_low_yield_wards,
            "racial_oversearch_groups": "; ".join(
                sorted(
                    {
                        part.strip()
                        for value in grp["ward_racial_oversearch_groups"].fillna("").astype(str)
                        for part in value.split(";")
                        if part.strip()
                    }
                )
            ),
            "excess_burden_flagged_wards": oversearch_low_yield_wards,
            "deprivation_burden_flagged_wards": deprivation_oversearch_wards,
            "racial_pathway_flagged_wards": racial_oversearch_low_yield_wards,
            "low_yield_flagged_wards": oversearch_low_yield_wards,
            "n_strong_pathways": n_strong_pathways,
            "criticalness_score": float(grp["ward_criticalness_score"].max()),
            "criticalness_level": (
                "Three multipaths"
                if n_strong_pathways >= 3
                else "Two multipaths"
                if n_strong_pathways == 2
                else "One multipath"
                if n_strong_pathways == 1
                else "Monitor trait"
                if int(grp["ward_monitor_trait_count"].fillna(0).sum()) > 0
                else "No signal"
            ),
            "monitor_trait_wards": int(grp["ward_monitor_trait_count"].fillna(0).gt(0).sum()),
            "monitor_trait_labels": "; ".join(
                sorted(
                    {
                        part.strip()
                        for value in grp["ward_monitor_trait_labels"].fillna("").astype(str)
                        for part in value.split(";")
                        if part.strip()
                    }
                )
            ),
            "ward_stop_burden_percentile": float(grp["ward_stop_burden_percentile"].max()),
            "ward_no_result_burden_percentile": float(grp["ward_no_result_burden_percentile"].max()),
            "ward_deprivation_percentile": float(grp["ward_deprivation_percentile"].max()),
            "ward_over_search_score_0_100": float(grp["ward_over_search_score_0_100"].max()),
            "ward_stop_rate_vs_london_avg_ratio": float(grp["ward_stop_rate_vs_london_avg_ratio"].max()),
            "london_avg_ward_stop_rate_per_1000": float(grp["ward_london_avg_stop_rate_per_1000"].mean()),
            "london_normal_ward_stop_rate_per_1000": float(grp["ward_london_normal_stop_rate_per_1000"].mean()),
            "excess_searches_to_london_normal_annual": unfair_excess_annual,
            "expected_quarterly_unfair_searches_to_london_normal": unfair_excess_quarter,
            "ward_low_yield_actionability_score_0_100": float(grp["ward_low_yield_actionability_score_0_100"].max()),
            "number_of_lsoas": int(sum(len(str(x).split(";")) for x in grp["member_lsoas"] if str(x))),
            "number_of_priority_or_critical_review_lsoas": int(grp["number_of_priority_or_critical_review_lsoas"].sum()),
            "share_of_stops_from_flagged_lsoas": flagged_stops / total_stops if total_stops else 0.0,
            "total_stops": total_stops,
            "no_result_stops": float(grp["no_result_stops"].sum()),
            "positive_outcomes": float(grp["positive_outcomes"].sum()),
            "arrests": float(grp["arrests"].sum()),
            "expected_no_result_stops_in_reducible_categories": float(
                grp["expected_no_result_stops_in_reducible_categories"].sum()
            ),
            "target_category_no_result_rate": no_rate,
            "mean_actionable_fairness_score_0_100": float(grp["mean_actionable_fairness_score_0_100"].mean()),
            "max_actionable_fairness_score_0_100": float(grp["max_actionable_fairness_score_0_100"].max()),
            "quarterly_baseline_reducible_searches": baseline,
            "proposed_reduction_pct": pct,
            "expected_quarterly_searches_reduced": expected_reduced,
            "expected_quarterly_no_result_stops_avoided": no_avoided,
            "expected_quarterly_positive_outcomes_forgone": float(grp["expected_positive_outcomes_forgone_at_cap"].sum()),
            "expected_quarterly_arrests_forgone": float(grp["expected_arrests_forgone_at_cap"].sum()),
            "aggregate_crime_guardrail": guardrail,
            "estimated_crime_response_lower_bound": 0.0,
            "estimated_crime_response_upper_bound": baseline * pct * 0.05,
            "trust_context_warning": bool(grp["trust_context_warning"].any()),
            "resident_denominator_caution": bool(grp["resident_denominator_caution"].any()),
            "recommendation_explanation": (
                f"Recommended monitored quarterly trial: {int(pct*100)}% reduction in historically low-yield "
                f"{_mode(grp['target_reduction_category'])} searches across adjacent compatible wards."
            ),
            "cluster_geometry": json.dumps(mapping(union)),
        }
        cluster_rows.append(row)
        props = {k: v for k, v in row.items() if k not in ["cluster_geometry"]}
        features.append({"type": "Feature", "properties": props, "geometry": mapping(union)})
    clusters = pd.DataFrame(cluster_rows)
    clusters.to_csv(WARD_CLUSTERS_CSV, index=False)
    fc = {"type": "FeatureCollection", "features": features}
    return clusters, fc


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    wards, ward_geo = build_ward_metrics()
    clusters, _ = build_clusters(wards, ward_geo)
    return wards, clusters


def main() -> None:
    run()


if __name__ == "__main__":
    main()
