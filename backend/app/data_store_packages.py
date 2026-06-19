from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from src import config
from src.fairness_v2.smoothing import beta_posterior_summary, empirical_bayes_rate

from . import paths

SEARCH_REVIEW_GUARDRAIL_CAPS = {"Low": 0.45, "Medium": 0.30, "High": 0.15, "Severe": 0.05}
SEARCH_REVIEW_PROTECTION_BAND_CAPS = {"Low": 0.45, "Medium": 0.30, "High": 0.15, "Critical": 0.05}
MIN_PRACTICAL_SEARCHES_REDUCED = 5.0
SEARCH_REGIMES: dict[str, list[str]] = {
    "drugs": ["drugs"],
    "stolen_property": ["stolen_property"],
    "other_non_weapon": ["other_non_weapon"],
    "combined_non_weapon": ["drugs", "stolen_property", "other_non_weapon"],
    "offensive_weapons": ["offensive_weapons"],
}


def _records(df: pd.DataFrame) -> list[dict]:
    clean = df.astype(object).where(pd.notnull(df), None)
    return clean.to_dict("records")


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _safe_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    try:
        return bool(value) and not pd.isna(value)
    except TypeError:
        return bool(value)


def _safe_str(value, default: str = "") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    text = str(value).strip()
    return text if text and text.lower() != "nan" else default


def _jsonable(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _split_pathways(value) -> list[str]:
    if value is None or pd.isna(value):
        return []
    parts = []
    for piece in str(value).replace("|", ";").split(";"):
        clean = piece.strip()
        if clean:
            parts.append(clean)
    return parts


def _package_name(package_id: str) -> str:
    labels = {
        "P0": "Monitor Only",
        "P1": "Search-Practice Review",
        "P2": "Procedural-Justice Training and Grounds/BWV Audit Priority",
        "P3": "Community Scrutiny and Confidence Intervention",
        "P4": "Fairness-Safeguarded Precision Protection",
        "P5": "Combined High-Priority Fairness Intervention",
    }
    return labels.get(str(package_id), str(package_id))


def _band_from_guardrail(value: str) -> str:
    guardrail = str(value or "Low")
    if guardrail == "Severe":
        return "Critical"
    if guardrail in {"High", "Medium", "Low"}:
        return guardrail
    return "Low"


def _ward_id(value) -> str:
    return str(value or "").strip()


def _cluster_membership() -> pd.DataFrame:
    rows = []
    for _, row in clusters().iterrows():
        for ward_code in str(row.get("member_wards", "")).split(";"):
            ward_code = ward_code.strip()
            if ward_code:
                rows.append(
                    {
                        "ward_code": ward_code,
                        "source_cluster_id": row["cluster_id"],
                        "source_cluster_name": row.get("cluster_name", row["cluster_id"]),
                    }
                )
    return pd.DataFrame(rows)


def _ward_metrics_with_cluster() -> pd.DataFrame:
    wards = ward_metrics().copy()
    membership = _cluster_membership()
    if not membership.empty:
        wards = wards.merge(membership, on="ward_code", how="left")
    else:
        wards["source_cluster_id"] = None
        wards["source_cluster_name"] = None
    return wards


def _ward_allocation_lookup(strategy_id: str, allocation_df: pd.DataFrame | None = None) -> dict[str, dict]:
    alloc = allocations() if allocation_df is None else allocation_df
    sub = alloc[alloc["strategy_id"].eq(strategy_id)].copy() if "strategy_id" in alloc.columns else alloc.copy()
    if sub.empty:
        return {}
    cluster_to_pkg = sub.set_index("cluster_id").to_dict(orient="index")
    out: dict[str, dict] = {}
    for _, row in _cluster_membership().iterrows():
        cid = row["source_cluster_id"]
        if cid in cluster_to_pkg:
            out[row["ward_code"]] = cluster_to_pkg[cid]
    return out


def _ward_protection_lookup() -> dict[str, dict]:
    prot = protection().set_index("cluster_id").to_dict(orient="index")
    out: dict[str, dict] = {}
    for _, row in _cluster_membership().iterrows():
        cid = row["source_cluster_id"]
        if cid in prot:
            out[row["ward_code"]] = prot[cid]
    return out


@lru_cache(maxsize=1)
def _ward_protection_all() -> dict[str, dict]:
    if not paths.WARD_PROTECTION.exists():
        return {}
    df = pd.read_csv(paths.WARD_PROTECTION, low_memory=False)
    return {_safe_str(r.get("ward_code")): r.to_dict() for _, r in df.iterrows()}


def _fallback_ward_protection(row: pd.Series) -> dict:
    guardrail = _safe_str(row.get("aggregate_crime_guardrail"), "Low")
    band = _band_from_guardrail(guardrail)
    wp = _ward_protection_all().get(_safe_str(row.get("ward_code")), {})

    def harm(key):
        return _jsonable(wp.get(key))

    return {
        "cluster_id": row.get("ward_code"),
        "cluster_population": None,
        "predicted_serious_harm_next_period": None,
        "predicted_serious_harm_per_1000_residents": harm("predicted_serious_harm_per_1000_residents"),
        "london_serious_harm_avg_per_1000_residents": harm("london_serious_harm_avg_per_1000_residents"),
        "predicted_serious_harm_rank_pct": harm("predicted_serious_harm_rank_pct"),
        "predicted_harm_weighted_serious_crime_score_next_period": None,
        "predicted_harm_weighted_serious_crime_score_per_1000_residents": harm("predicted_harm_weighted_serious_crime_score_per_1000_residents"),
        "london_harm_weighted_serious_crime_score_avg_per_1000_residents": harm("london_harm_weighted_serious_crime_score_avg_per_1000_residents"),
        "predicted_harm_weighted_serious_crime_score_rank_pct": harm("predicted_harm_weighted_serious_crime_score_rank_pct"),
        "aggregate_crime_guardrail": guardrail,
        "protection_need_band": band,
        "eligibility_implication": "Ward-level serious-crime estimate.",
    }


def _ward_fairness_indicators(row: pd.Series) -> list[dict]:
    indicators: list[dict] = []
    for pathway in _split_pathways(row.get("ward_fairness_pathways")):
        detail = "This ward has this active ward-level fairness path."
        if pathway == "Over-search + very low yield":
            detail = "Search rate is much higher than the London ward average and at least one search type has very low-result evidence."
        elif pathway == "Deprivation + over-search":
            detail = "The ward is highly deprived and its search rate is above the London-normal range after tolerance."
        elif pathway == "Racial over-search + low yield":
            groups = _safe_str(row.get("ward_racial_oversearch_groups"))
            detail = "A resident group is over-exposed relative to ward population share and has low search yield."
            if groups:
                detail += f" Group(s): {groups}."
        indicators.append(
            {
                "label": pathway,
                "value": "Ward path flagged",
                "detail": detail,
                "kind": "pathway",
                "flagged": True,
            }
        )
    if not indicators:
        for trait in _split_pathways(row.get("ward_monitor_trait_labels")):
            indicators.append(
                {
                    "label": trait,
                    "value": "Monitor signal",
                    "detail": "Standalone ward signal only; no combined active fairness path.",
                    "kind": "monitor",
                    "flagged": False,
                }
            )
    return indicators


def _ward_context_flags(row: pd.Series) -> list[dict]:
    flags = [
        (
            "Over-search",
            _safe_bool(row.get("ward_substantial_oversearch_flag")),
            "Search rate above London-normal ward range.",
        ),
        (
            "Much over-search",
            _safe_bool(row.get("ward_much_oversearch_flag")),
            "Search rate at least 1.50x London ward average with enough volume.",
        ),
        (
            "High deprivation",
            _safe_bool(row.get("ward_deprivation_trait_flag")),
            "Ward sits in high-deprivation range.",
        ),
        (
            "Low-result category",
            _safe_bool(row.get("ward_low_yield_actionability_flag")),
            f"Category: {_safe_str(row.get('ward_low_yield_categories'), 'none')}.",
        ),
        (
            "Very-low-result category",
            _safe_bool(row.get("ward_very_low_yield_actionability_flag")),
            f"Category: {_safe_str(row.get('ward_very_low_yield_categories'), 'none')}.",
        ),
        (
            "Racial over-search + low yield",
            _safe_bool(row.get("ward_racial_pathway_flag")),
            f"Group(s): {_safe_str(row.get('ward_racial_oversearch_groups'), 'none')}.",
        ),
    ]
    return [
        {"label": label, "flagged": flagged, "detail": detail}
        for label, flagged, detail in flags
        if flagged
    ]


def _cluster_member_ward_context(cluster_id: str) -> list[dict]:
    wards = _ward_metrics_with_cluster()
    rows = wards[wards["source_cluster_id"].eq(cluster_id)].copy()
    if rows.empty:
        rows = wards[wards["ward_code"].eq(cluster_id)].copy()
    out = []
    for _, row in rows.sort_values(["borough", "ward_name"]).iterrows():
        out.append(
            {
                "ward_code": _safe_str(row.get("ward_code")),
                "ward_name": _safe_str(row.get("ward_name"), row.get("ward_code")),
                "borough": _safe_str(row.get("borough")),
                "criticalness_level": _safe_str(row.get("ward_criticalness_level")),
                "fairness_pathways": _safe_str(row.get("ward_fairness_pathways")),
                "monitor_trait_labels": _safe_str(row.get("ward_monitor_trait_labels")),
                "racial_oversearch_groups": _safe_str(row.get("ward_racial_oversearch_groups")),
                "flagged_characteristics": _ward_context_flags(row),
            }
        )
    return out


def _fairness_indicators(detail: dict, search_rows: list[dict]) -> list[dict]:

    indicators: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def add(label: str, value: str, detail_text: str, kind: str, flagged: bool = True) -> None:
        key = (label, value)
        if key in seen:
            return
        seen.add(key)
        indicators.append({
            "label": label,
            "value": value,
            "detail": detail_text,
            "kind": kind,
            "flagged": flagged,
        })

    _ = search_rows
    total_wards = _safe_float(detail.get("number_of_wards"))
    ward_suffix = f" of {int(total_wards)} ward(s)" if total_wards else " ward(s)"
    excess_qtr = _safe_float(detail.get("expected_quarterly_unfair_searches_to_london_normal"))
    excess_text = f" About {excess_qtr:.0f} searches per quarter sit above the London-normal ward rate." if excess_qtr > 0 else ""
    racial_groups = _safe_str(detail.get("racial_oversearch_groups"))
    racial_group_text = f" Group(s): {racial_groups}." if racial_groups else ""
    pathway_types = [
        (
            "Over-search + very low yield",
            "oversearch_low_yield_flagged_wards",
            "Ward search rate is much higher than the London ward average, and at least one search type has very low-result evidence."
            + excess_text,
        ),
        (
            "Deprivation + over-search",
            "deprivation_oversearch_flagged_wards",
            "Ward is in the high-deprivation range and its search rate is above the London-normal range after tolerance."
            + excess_text,
        ),
        (
            "Racial over-search + low yield",
            "racial_oversearch_low_yield_flagged_wards",
            "A resident group is over-exposed relative to its ward population share, and that group's search yield is low against the London benchmark.",
        ),
    ]
    for label, key, detail_text in pathway_types:
        value = _safe_float(detail.get(key))
        if value > 0:
            if key == "racial_oversearch_low_yield_flagged_wards":
                detail_text += racial_group_text
            add(label, f"{int(value)}{ward_suffix}", detail_text, "pathway")

    if not indicators:
        for pathway in _split_pathways(detail.get("ward_fairness_pathways")):
            if pathway in {"Over-search + very low yield", "Deprivation + over-search", "Racial over-search + low yield"}:
                add(pathway, "Ward path flagged", "Selected ward-level unfairness pathway recorded for this cluster.", "pathway")

    return indicators


@lru_cache(maxsize=1)
def _ward_search_volume() -> pd.DataFrame:
    cats = pd.read_parquet(config.STOP_SEARCH_CATEGORIES_PATH)
    cats["month"] = cats["month"].astype(str).str[:7]
    months = sorted(cats["month"].unique())[-12:]
    cats = cats[cats["month"].isin(months)]
    lookup = pd.read_csv(paths.WARD_LOOKUP)[["lsoa21cd", "ward_code"]]
    cats = cats.merge(lookup, on="lsoa21cd", how="inner")
    metrics = ["stops", "no_result_stops", "positive_outcomes", "arrests"]
    categories = ["drugs", "stolen_property", "other_non_weapon", "offensive_weapons"]
    cols = [f"{metric}_{cat}" for metric in metrics for cat in categories if f"{metric}_{cat}" in cats.columns]
    return cats.groupby("ward_code", as_index=False)[cols].sum()


def _ward_search_review_frame() -> pd.DataFrame:
    wards = _ward_metrics_with_cluster()
    vol = _ward_search_volume()
    vol_idx = vol.set_index("ward_code")
    prot_lookup = _ward_protection_lookup()
    rows = []
    for _, ward in wards.iterrows():
        ward_code = _ward_id(ward.get("ward_code"))
        sub = vol_idx.loc[ward_code] if ward_code in vol_idx.index else pd.Series(dtype=float)
        protection_row = prot_lookup.get(ward_code, _fallback_ward_protection(ward))
        cluster_total = sum(float(sub.get(f"stops_{cat}", 0) or 0) for cat in ["drugs", "stolen_property", "other_non_weapon", "offensive_weapons"])
        quarterly_total = cluster_total / 4.0
        stop_rate_ratio = _safe_float(ward.get("ward_stop_rate_vs_london_avg_ratio"))
        london_avg_quarterly_stops = quarterly_total / stop_rate_ratio if stop_rate_ratio > 0 else 0.0
        excess_to_london_avg_qtr = max(quarterly_total - london_avg_quarterly_stops, 0.0)
        for regime, parts in SEARCH_REGIMES.items():
            stops = float(sum(float(sub.get(f"stops_{cat}", 0) or 0) for cat in parts))
            no_result = float(sum(float(sub.get(f"no_result_stops_{cat}", 0) or 0) for cat in parts))
            positive = float(sum(float(sub.get(f"positive_outcomes_{cat}", 0) or 0) for cat in parts))
            arrests = float(sum(float(sub.get(f"arrests_{cat}", 0) or 0) for cat in parts))
            rows.append(
                {
                    "cluster_id": ward_code,
                    "ward_code": ward_code,
                    "source_cluster_id": _safe_str(ward.get("source_cluster_id")),
                    "cluster_name": f"{_safe_str(ward.get('ward_name'), ward_code)} ward",
                    "member_ward_names": _safe_str(ward.get("ward_name"), ward_code),
                    "boroughs": _safe_str(ward.get("borough")),
                    "dominant_fairness_pathway": _safe_str(ward.get("ward_fairness_pathways")),
                    "primary_search_regime": _safe_str(ward.get("target_reduction_category")),
                    "resident_denominator_caution": _safe_bool(ward.get("resident_denominator_caution")),
                    "search_regime": regime,
                    "is_protected_context": False,
                    "annual_stops": stops,
                    "quarterly_stops": stops / 4.0,
                    "share_of_cluster_stops": stops / cluster_total if cluster_total else 0.0,
                    "no_result_count_annual": no_result,
                    "positive_outcome_count_annual": positive,
                    "arrest_count_annual": arrests,
                    "package_relevance": "",
                    "aggregate_crime_guardrail": _safe_str(protection_row.get("aggregate_crime_guardrail"), ward.get("aggregate_crime_guardrail") or "Low"),
                    "protection_need_band": _safe_str(protection_row.get("protection_need_band"), _band_from_guardrail(ward.get("aggregate_crime_guardrail"))),
                    "predicted_serious_harm_next_period": protection_row.get("predicted_serious_harm_next_period"),
                    "predicted_harm_weighted_serious_crime_score_next_period": protection_row.get("predicted_harm_weighted_serious_crime_score_next_period"),
                    "predicted_serious_harm_rank_pct": protection_row.get("predicted_serious_harm_rank_pct"),
                    "predicted_harm_weighted_serious_crime_score_rank_pct": protection_row.get("predicted_harm_weighted_serious_crime_score_rank_pct"),
                    "expected_quarterly_unfair_searches_to_london_normal": _safe_float(ward.get("ward_excess_searches_to_london_normal_quarter")),
                    "expected_quarterly_excess_searches_to_london_avg": excess_to_london_avg_qtr,
                    "excess_searches_to_london_normal_annual": _safe_float(ward.get("ward_excess_searches_to_london_normal_annual")),
                    "london_avg_ward_stop_rate_per_1000": _safe_float(ward.get("ward_london_avg_stop_rate_per_1000")),
                    "london_normal_ward_stop_rate_per_1000": _safe_float(ward.get("ward_london_normal_stop_rate_per_1000")),
                    "ward_stop_rate_vs_london_avg_ratio": _safe_float(ward.get("ward_stop_rate_vs_london_avg_ratio")),
                    "ward_number_of_pathways_flagged": int(_safe_float(ward.get("ward_number_of_pathways_flagged"))),
                }
            )
    reg = pd.DataFrame(rows)
    if reg.empty:
        return reg

    for regime, parts in SEARCH_REGIMES.items():
        mask = reg["search_regime"].eq(regime)
        total_stops = float(reg.loc[mask, "annual_stops"].sum())
        bench_no_result = float(reg.loc[mask, "no_result_count_annual"].sum()) / total_stops if total_stops else np.nan
        bench_positive = float(reg.loc[mask, "positive_outcome_count_annual"].sum()) / total_stops if total_stops else np.nan
        bench_arrest = float(reg.loc[mask, "arrest_count_annual"].sum()) / total_stops if total_stops else np.nan
        reg.loc[mask, "london_category_no_result_rate"] = bench_no_result
        reg.loc[mask, "smoothed_no_result_rate"] = empirical_bayes_rate(
            reg.loc[mask, "no_result_count_annual"],
            reg.loc[mask, "annual_stops"],
            bench_no_result,
            alpha=config.EMPIRICAL_BAYES_ALPHA,
        ).values
        reg.loc[mask, "smoothed_positive_outcome_rate"] = empirical_bayes_rate(
            reg.loc[mask, "positive_outcome_count_annual"],
            reg.loc[mask, "annual_stops"],
            bench_positive,
            alpha=config.EMPIRICAL_BAYES_ALPHA,
        ).values
        reg.loc[mask, "smoothed_arrest_rate"] = empirical_bayes_rate(
            reg.loc[mask, "arrest_count_annual"],
            reg.loc[mask, "annual_stops"],
            bench_arrest,
            alpha=config.EMPIRICAL_BAYES_ALPHA,
        ).values
        post = beta_posterior_summary(
            reg.loc[mask, "no_result_count_annual"],
            reg.loc[mask, "annual_stops"],
            bench_no_result,
            alpha=config.EMPIRICAL_BAYES_ALPHA,
        )
        reg.loc[mask, "posterior_prob_low_yield"] = post["probability_rate_above_benchmark"].values

    reg["is_rollup"] = reg["search_regime"].eq("combined_non_weapon")
    reg["quarterly_no_result_searches"] = reg["no_result_count_annual"] / 4
    reg["quarterly_positive_outcomes"] = reg["positive_outcome_count_annual"] / 4
    reg["quarterly_arrests"] = reg["arrest_count_annual"] / 4
    reg["low_result_signal"] = reg["posterior_prob_low_yield"].fillna(0)
    reg["raw_no_result_rate"] = np.where(
        reg["annual_stops"].gt(0),
        reg["no_result_count_annual"] / reg["annual_stops"],
        np.nan,
    )
    reg["no_result_rate_gap_vs_london"] = reg["smoothed_no_result_rate"] - reg["london_category_no_result_rate"]
    reg["no_result_rate_ratio_vs_london"] = np.where(
        reg["london_category_no_result_rate"].gt(0),
        reg["smoothed_no_result_rate"] / reg["london_category_no_result_rate"],
        np.nan,
    )
    reg["above_london_category_no_result_rate"] = reg["no_result_rate_gap_vs_london"].fillna(-np.inf).ge(0)
    reg["is_reducible_search_type"] = reg["search_regime"].isin(config.REDUCIBLE_SEARCH_CATEGORIES)
    guardrail_cap = reg["aggregate_crime_guardrail"].map(SEARCH_REVIEW_GUARDRAIL_CAPS).fillna(0.30)
    band_cap = reg["protection_need_band"].map(SEARCH_REVIEW_PROTECTION_BAND_CAPS).fillna(0.30)
    reg["safety_reduction_cap"] = np.minimum(guardrail_cap, band_cap)
    reg["candidate_reduction_pct_uncapped"] = reg["safety_reduction_cap"].clip(lower=0)
    reg["candidate_expected_searches_reduced_uncapped"] = (
        reg["quarterly_stops"] * reg["candidate_reduction_pct_uncapped"]
    )
    has_fairness_path = reg["ward_number_of_pathways_flagged"].gt(0)
    preliminary_candidate = (
        (~reg["is_rollup"])
        & reg["is_reducible_search_type"]
        & has_fairness_path
        & reg["quarterly_stops"].ge(config.WARD_CLUSTER_MIN_QUARTERLY_SEARCHES_REDUCED)
        & reg["above_london_category_no_result_rate"]
        & reg["safety_reduction_cap"].gt(0)
        & reg["expected_quarterly_excess_searches_to_london_avg"].gt(0)
        & reg["candidate_expected_searches_reduced_uncapped"].ge(MIN_PRACTICAL_SEARCHES_REDUCED)
    )
    reg["uncapped_candidate_sum_by_ward"] = (
        reg["candidate_expected_searches_reduced_uncapped"].where(preliminary_candidate, 0.0)
        .groupby(reg["cluster_id"])
        .transform("sum")
    )
    reg["london_average_excess_scale"] = np.where(
        reg["uncapped_candidate_sum_by_ward"].gt(0),
        np.minimum(
            1.0,
            reg["expected_quarterly_excess_searches_to_london_avg"]
            / reg["uncapped_candidate_sum_by_ward"].replace(0, np.nan),
        ),
        0.0,
    )
    reg["london_average_excess_scale"] = reg["london_average_excess_scale"].fillna(0.0).clip(lower=0, upper=1)
    reg["candidate_expected_searches_reduced"] = np.where(
        preliminary_candidate,
        reg["candidate_expected_searches_reduced_uncapped"] * reg["london_average_excess_scale"],
        0.0,
    )
    reg["candidate_reduction_pct"] = np.where(
        reg["quarterly_stops"].gt(0),
        reg["candidate_expected_searches_reduced"] / reg["quarterly_stops"],
        0.0,
    )
    reg["candidate_reduction_pct"] = reg["candidate_reduction_pct"].clip(lower=0)
    reg["candidate_expected_no_result_avoided"] = reg["quarterly_no_result_searches"] * reg["candidate_reduction_pct"]
    candidate = preliminary_candidate & reg["candidate_expected_searches_reduced"].ge(MIN_PRACTICAL_SEARCHES_REDUCED)
    reg["is_reduction_candidate"] = candidate
    reg["is_reduction_target"] = candidate
    reg["recommended_reduction_pct"] = np.where(candidate, reg["candidate_reduction_pct"], 0.0)
    reg["expected_searches_reduced_if_applied"] = reg["quarterly_stops"] * reg["recommended_reduction_pct"]
    reg["expected_no_result_avoided_if_applied"] = reg["quarterly_no_result_searches"] * reg["recommended_reduction_pct"]
    reg["expected_positive_outcomes_at_risk_if_applied"] = reg["quarterly_positive_outcomes"] * reg["recommended_reduction_pct"]
    reg["expected_arrests_at_risk_if_applied"] = reg["quarterly_arrests"] * reg["recommended_reduction_pct"]

    states, actions, reasons = [], [], []
    for _, row in reg.iterrows():
        if bool(row["is_rollup"]):
            states.append("Context roll-up")
            actions.append("Context only")
            reasons.append("This combines the non-weapon rows. Use the individual search types for any action.")
        elif not bool(row["is_reducible_search_type"]):
            states.append("No reduction suggested")
            actions.append("No change")
            reasons.append("This search type is counted, but it is not one of the defined reducible categories.")
        elif not bool(row["ward_number_of_pathways_flagged"]):
            states.append("No reduction suggested")
            actions.append("No change")
            reasons.append("This ward is not active on a combined fairness pathway, so no search cut is suggested.")
        elif bool(row["is_reduction_target"]):
            pct = int(round(float(row["recommended_reduction_pct"]) * 100))
            ward_rate = float(row["smoothed_no_result_rate"])
            london_rate = float(row["london_category_no_result_rate"])
            avg_cap = float(row["expected_quarterly_excess_searches_to_london_avg"])
            cut = float(row["expected_searches_reduced_if_applied"])
            cap_note = (
                f" The cut is capped at {avg_cap:.1f} searches per quarter, the amount needed to bring this "
                "ward back to the London average search rate."
                if float(row["london_average_excess_scale"]) < 0.999
                else ""
            )
            states.append("Suggested reduction")
            actions.append(f"Reduce by {pct}%")
            reasons.append(
                f"This ward's NFA/no-result rate for this search type is {ward_rate:.0%}, at or above the London "
                f"average for the same type ({london_rate:.0%}). It also has enough quarterly volume and the safety "
                f"cap allows up to {float(row['safety_reduction_cap']):.0%}.{cap_note} "
                f"If applied, about {cut:.1f} searches and {float(row['expected_no_result_avoided_if_applied']):.1f} "
                f"no-result searches would be avoided per quarter, with about "
                f"{float(row['expected_positive_outcomes_at_risk_if_applied']):.1f} positive outcomes at risk."
            )
        elif not bool(row["above_london_category_no_result_rate"]):
            states.append("No reduction suggested")
            actions.append("No change")
            reasons.append(
                "This search type's NFA/no-result rate is below the London average for the same search type."
            )
        elif float(row["quarterly_stops"]) < config.WARD_CLUSTER_MIN_QUARTERLY_SEARCHES_REDUCED:
            states.append("No reduction suggested")
            actions.append("No change")
            reasons.append("Too few quarterly searches in this type for a stable ward-level percentage.")
        elif float(row["safety_reduction_cap"]) <= 0:
            states.append("Blocked by safety")
            actions.append("No cut")
            reasons.append("Current safety level blocks search reduction in this ward.")
        elif float(row["expected_quarterly_excess_searches_to_london_avg"]) <= 0:
            states.append("No reduction suggested")
            actions.append("No change")
            reasons.append("This ward is not above the London average search volume, so no reduction is needed to reach average.")
        elif float(row["candidate_expected_searches_reduced"]) < MIN_PRACTICAL_SEARCHES_REDUCED:
            states.append("No reduction suggested")
            actions.append("No change")
            reasons.append("The safe percentage would change too few searches per quarter to be a practical ward-level action.")
        else:
            states.append("No reduction suggested")
            actions.append("No change")
            reasons.append("No reduction is suggested for this row after evidence, volume and safety checks.")
    reg["recommendation_state"] = states
    reg["recommended_action"] = actions
    reg["recommendation_reason"] = reasons
    reg["review_signal"] = reg["recommendation_state"]
    return reg


@lru_cache(maxsize=1)
def clusters() -> pd.DataFrame:
    return pd.read_parquet(paths.WARD_CLUSTERS_LATEST)


@lru_cache(maxsize=1)
def protection() -> pd.DataFrame:
    return pd.read_csv(paths.CLUSTER_PROTECTION)


@lru_cache(maxsize=1)
def regimes() -> pd.DataFrame:
    return pd.read_csv(paths.CLUSTER_SEARCH_REGIMES)


@lru_cache(maxsize=1)
def ward_metrics() -> pd.DataFrame:
    return pd.read_csv(paths.WARD_QUARTER_METRICS)


@lru_cache(maxsize=1)
def eligibility() -> pd.DataFrame:
    return pd.read_csv(paths.CLUSTER_PACKAGE_ELIGIBILITY)


@lru_cache(maxsize=1)
def strategies() -> pd.DataFrame:
    return pd.read_csv(paths.PKG_STRATEGIES)


@lru_cache(maxsize=1)
def allocations() -> pd.DataFrame:
    return pd.read_csv(paths.PKG_ALLOCATIONS)


@lru_cache(maxsize=1)
def ward_criticalness_geojson() -> dict:
    wards = pd.read_csv(paths.WARD_QUARTER_METRICS)
    bounds = json.loads(paths.WARD_BOUNDARIES.read_text())
    ward_by_code = wards.drop_duplicates("ward_code").set_index("ward_code")
    low_trust_by_borough: dict[str, bool] = {}
    if config.MOPAC_TRUST_CONTEXT_PARQUET_PATH.exists():
        trust = pd.read_parquet(config.MOPAC_TRUST_CONTEXT_PARQUET_PATH)
        low_trust_by_borough = {
            str(b).strip(): bool(f)
            for b, f in zip(trust["borough"], trust["borough_low_trust_flag"].fillna(False))
        }
    features = []
    for feat in bounds.get("features", []):
        props = feat.get("properties", {})
        code = props.get("WD24CD") or props.get("ward_code")
        if code not in ward_by_code.index:
            continue
        row = ward_by_code.loc[code]
        path_count = int(_safe_float(row.get("ward_number_of_pathways_flagged")))
        trait_count = int(_safe_float(row.get("ward_monitor_trait_count")))
        criticalness_score = _safe_float(row.get("ward_criticalness_score"))
        if path_count >= 3:
            level = "Three multipaths"
        elif path_count == 2:
            level = "Two multipaths"
        elif path_count == 1:
            level = "One multipath"
        elif trait_count >= 1:
            level = "Monitor trait"
        else:
            level = "No signal"
        total_stops = _safe_float(row.get("total_stops"))
        no_result = _safe_float(row.get("no_result_stops"))
        out_props = {
            "ward_code": code,
            "ward_name": _safe_str(row.get("ward_name"), props.get("WD24NM") or code),
            "borough": _safe_str(row.get("borough")),
            "borough_low_trust": low_trust_by_borough.get(_safe_str(row.get("borough")).strip(), False),
            "criticalness_level": level,
            "criticalness_score": criticalness_score,
            "multipath_count": path_count,
            "monitor_trait_count": trait_count,
            "fairness_pathways": _safe_str(row.get("ward_fairness_pathways")),
            "monitor_trait_labels": _safe_str(row.get("ward_monitor_trait_labels")),
            "racial_oversearch_groups": _safe_str(row.get("ward_racial_oversearch_groups")),
            "low_yield_categories": _safe_str(row.get("ward_low_yield_categories")),
            "very_low_yield_categories": _safe_str(row.get("ward_very_low_yield_categories")),
            "substantial_oversearch_flag": _safe_bool(row.get("ward_substantial_oversearch_flag")),
            "much_oversearch_flag": _safe_bool(row.get("ward_much_oversearch_flag")),
            "deprivation_trait_flag": _safe_bool(row.get("ward_deprivation_trait_flag")),
            "low_yield_actionability_flag": _safe_bool(row.get("ward_low_yield_actionability_flag")),
            "very_low_yield_actionability_flag": _safe_bool(row.get("ward_very_low_yield_actionability_flag")),
            "racial_pathway_flag": _safe_bool(row.get("ward_racial_pathway_flag")),
            "overall_review_priority": _safe_str(row.get("ward_overall_review_priority")),
            "total_stops_qtr": total_stops,
            "no_result_stops_qtr": no_result,
            "no_result_rate": no_result / total_stops if total_stops else 0,
            "stop_rate_vs_london_avg_ratio": _safe_float(row.get("ward_stop_rate_vs_london_avg_ratio")),
            "london_avg_ward_stop_rate_per_1000": _safe_float(row.get("ward_london_avg_stop_rate_per_1000")),
            "london_normal_ward_stop_rate_per_1000": _safe_float(row.get("ward_london_normal_stop_rate_per_1000")),
            "excess_searches_to_london_normal_qtr": _safe_float(row.get("ward_excess_searches_to_london_normal_quarter")),
        }
        features.append({"type": "Feature", "properties": out_props, "geometry": feat.get("geometry")})
    return {"type": "FeatureCollection", "features": features}


def package_library() -> list[dict]:
    from src.packages.library import PACKAGES, package_cost
    out = []
    for p in PACKAGES:
        out.append({
            "package_id": p.id,
            "name": p.name,
            "components": p.components,
            "includes_reduction": p.includes_reduction,
            "reduction_only": p.reduction_only,
            "resources": package_cost(p.id),
        })
    return out


def budgets() -> dict:
    from src.packages.library import quarterly_budgets
    return quarterly_budgets()


def cluster_detail(cluster_id: str) -> dict | None:
    ward_detail = _ward_detail(cluster_id)
    if ward_detail is not None:
        return ward_detail
    cl = clusters()
    row = cl[cl["cluster_id"] == cluster_id]
    if row.empty:
        return None
    detail = _records(row)[0]
    prot = protection()
    pr = prot[prot["cluster_id"] == cluster_id]
    detail["protection"] = _records(pr)[0] if not pr.empty else None
    reg = regimes()
    search_rows = _records(reg[reg["cluster_id"] == cluster_id])
    detail["search_regimes"] = search_rows
    detail["fairness_indicators"] = _fairness_indicators(detail, search_rows)
    detail["member_ward_context"] = _cluster_member_ward_context(cluster_id)
    elig = eligibility()
    detail["packages"] = _records(elig[elig["cluster_id"] == cluster_id])
    detail["limitations"] = (
        "Human-reviewed intervention-allocation scenario. Does not prove "
        "discrimination, does not measure trust locally, and does not guarantee "
        "future trust or crime outcomes."
    )
    return detail


def _ward_detail(ward_code: str) -> dict | None:
    code = _ward_id(ward_code)
    wards = _ward_metrics_with_cluster()
    row = wards[wards["ward_code"].eq(code)]
    if row.empty:
        return None
    ward = row.iloc[0]
    source_cluster_id = _safe_str(ward.get("source_cluster_id"))
    prot_lookup = _ward_protection_lookup()
    protection_row = prot_lookup.get(code, _fallback_ward_protection(ward))
    protection_row = dict(protection_row)
    protection_row["cluster_id"] = code

    search_rows = _records(_ward_search_review_frame()[_ward_search_review_frame()["cluster_id"].eq(code)])
    packages = []
    if source_cluster_id:
        elig = eligibility()
        packages = _records(elig[elig["cluster_id"].eq(source_cluster_id)])
        for pkg in packages:
            pkg["cluster_id"] = code
            pkg["source_cluster_id"] = source_cluster_id

    return {
        "cluster_id": code,
        "ward_code": code,
        "source_cluster_id": source_cluster_id,
        "cluster_name": f"{_safe_str(ward.get('ward_name'), code)} ward",
        "member_ward_names": _safe_str(ward.get("ward_name"), code),
        "boroughs": _safe_str(ward.get("borough")),
        "dominant_fairness_pathway": _safe_str(ward.get("ward_fairness_pathways")),
        "secondary_pathways": "",
        "relevant_group_concern": _safe_str(ward.get("ward_racial_oversearch_groups")),
        "primary_search_regime": _safe_str(ward.get("target_reduction_category")),
        "intervention_eligibility": _safe_str(ward.get("ward_actionability_status")),
        "trust_context_warning": _safe_bool(ward.get("trust_context_warning")),
        "resident_denominator_caution": _safe_bool(ward.get("resident_denominator_caution")),
        "fairness_indicators": _ward_fairness_indicators(ward),
        "member_ward_context": _cluster_member_ward_context(code),
        "protection": protection_row,
        "search_regimes": search_rows,
        "packages": packages,
        "expected_quarterly_unfair_searches_to_london_normal": _safe_float(ward.get("ward_excess_searches_to_london_normal_quarter")),
        "expected_quarterly_excess_searches_to_london_avg": _safe_float(
            search_rows[0].get("expected_quarterly_excess_searches_to_london_avg") if search_rows else 0
        ),
        "london_avg_ward_stop_rate_per_1000": _safe_float(ward.get("ward_london_avg_stop_rate_per_1000")),
        "london_normal_ward_stop_rate_per_1000": _safe_float(ward.get("ward_london_normal_stop_rate_per_1000")),
        "ward_stop_rate_vs_london_avg_ratio": _safe_float(ward.get("ward_stop_rate_vs_london_avg_ratio")),
        "limitations": (
            "Ward-level decision-support view. It supports human review and programme planning; it does not prove "
            "discrimination or guarantee future trust or crime outcomes."
        ),
    }


def search_review() -> dict:
    reg = _ward_search_review_frame()
    atomic = reg[~reg["is_rollup"]].copy()
    total_searches = float(pd.to_numeric(atomic["quarterly_stops"], errors="coerce").fillna(0).sum())
    total_no_result = float(atomic["quarterly_no_result_searches"].sum())
    total_positive = float(atomic["quarterly_positive_outcomes"].sum())
    weapons = atomic[atomic["search_regime"].eq("offensive_weapons")]
    targets = reg[reg["is_reduction_target"]].copy()
    cluster_once = reg.drop_duplicates("cluster_id").copy()
    unfair_qtr = float(
        pd.to_numeric(
            cluster_once.loc[cluster_once["ward_number_of_pathways_flagged"].gt(0)].get(
                "expected_quarterly_unfair_searches_to_london_normal", pd.Series(dtype=float)
            ),
            errors="coerce",
        ).fillna(0).sum()
    )
    avg_excess_qtr = float(
        pd.to_numeric(
            cluster_once.loc[cluster_once["ward_number_of_pathways_flagged"].gt(0)].get(
                "expected_quarterly_excess_searches_to_london_avg", pd.Series(dtype=float)
            ),
            errors="coerce",
        ).fillna(0).sum()
    )

    summary = {
        "clusters": int(reg["cluster_id"].nunique()),
        "search_type_rows": int(len(atomic)),
        "rollup_rows": int(reg["is_rollup"].sum()),
        "total_quarterly_searches": round(total_searches, 1),
        "total_quarterly_no_result_searches": round(total_no_result, 1),
        "total_quarterly_positive_outcomes": round(total_positive, 1),
        "overall_no_result_rate": round(total_no_result / total_searches, 4) if total_searches else 0,
        "weapons_quarterly_searches": round(
            float(pd.to_numeric(weapons["quarterly_stops"], errors="coerce").fillna(0).sum()), 1
        ),
        "protected_weapons_quarterly_searches": 0.0,
        "strong_low_result_rows": int((atomic["low_result_signal"] >= config.FAIRNESS_V2_LOW_YIELD_PROB_THRESHOLD).sum()),
        "rows_at_or_above_london_no_result_rate": int(atomic["above_london_category_no_result_rate"].sum()),
        "total_unfair_searches_detected": round(unfair_qtr, 1),
        "total_excess_searches_to_london_avg": round(avg_excess_qtr, 1),
        "clusters_with_suggested_reduction": int(targets["cluster_id"].nunique()),
        "search_type_rows_with_suggested_reduction": int(len(targets)),
        "total_suggested_searches_reduced": round(float(targets["expected_searches_reduced_if_applied"].sum()), 1),
        "total_suggested_no_result_avoided": round(float(targets["expected_no_result_avoided_if_applied"].sum()), 1),
        "total_positive_outcomes_at_risk": round(float(targets["expected_positive_outcomes_at_risk_if_applied"].sum()), 1),
    }
    rows = reg.sort_values(
        ["is_rollup", "is_reduction_target", "expected_no_result_avoided_if_applied", "quarterly_no_result_searches"],
        ascending=[True, False, False, False],
    )
    return {"summary": summary, "rows": _records(rows)}


def map_search_review_clusters() -> dict:
    reg = _ward_search_review_frame()
    atomic = reg[~reg["is_rollup"]].copy()
    grouped = atomic.groupby("cluster_id", as_index=True).agg(
        quarterly_total_searches=("quarterly_stops", "sum"),
        quarterly_no_result_searches=("quarterly_no_result_searches", "sum"),
        quarterly_positive_outcomes=("quarterly_positive_outcomes", "sum"),
        max_low_result_signal=("posterior_prob_low_yield", "max"),
    )
    grouped["overall_no_result_rate"] = (
        grouped["quarterly_no_result_searches"] / grouped["quarterly_total_searches"].replace(0, np.nan)
    ).fillna(0)
    primary = (
        atomic.sort_values(["cluster_id", "quarterly_no_result_searches"], ascending=[True, False])
        .groupby("cluster_id")
        .first()
    )
    targets = reg[reg["is_reduction_target"]].copy()
    if not targets.empty:
        target_summary = targets.groupby("cluster_id", as_index=True).agg(
            recommended_reduction_pct=("recommended_reduction_pct", "max"),
            search_review_target_type=("search_regime", lambda s: "; ".join(s.astype(str))),
            search_review_expected_searches_reduced=("expected_searches_reduced_if_applied", "sum"),
            search_review_expected_no_result_avoided=("expected_no_result_avoided_if_applied", "sum"),
            search_review_expected_positives_at_risk=("expected_positive_outcomes_at_risk_if_applied", "sum"),
            target_count=("search_regime", "count"),
        )
    else:
        target_summary = pd.DataFrame()
    prot_lookup = _ward_protection_lookup()
    unfair_cols = [
        "expected_quarterly_unfair_searches_to_london_normal",
        "expected_quarterly_excess_searches_to_london_avg",
        "excess_searches_to_london_normal_annual",
        "london_avg_ward_stop_rate_per_1000",
        "london_normal_ward_stop_rate_per_1000",
        "ward_stop_rate_vs_london_avg_ratio",
    ]
    unfair = reg.drop_duplicates("cluster_id").set_index("cluster_id")

    fc = _ward_base_geojson()
    for feat in fc["features"]:
        cid = feat["properties"].get("cluster_id")
        p = feat["properties"]
        if cid in grouped.index:
            g = grouped.loc[cid]
            p["search_review_total_qtr"] = float(g["quarterly_total_searches"])
            p["search_review_no_result_qtr"] = float(g["quarterly_no_result_searches"])
            p["search_review_no_result_rate"] = float(g["overall_no_result_rate"])
            p["search_review_positive_qtr"] = float(g["quarterly_positive_outcomes"])
            p["search_review_low_result_signal"] = float(g["max_low_result_signal"])
            p["search_review_main_type"] = str(primary.loc[cid, "search_regime"]) if cid in primary.index else ""
        else:
            p["search_review_total_qtr"] = 0
            p["search_review_no_result_qtr"] = 0
            p["search_review_no_result_rate"] = 0
            p["search_review_positive_qtr"] = 0
            p["search_review_low_result_signal"] = 0
            p["search_review_main_type"] = ""
        if cid in target_summary.index:
            t = target_summary.loc[cid]
            p["search_review_recommended_pct"] = float(t["recommended_reduction_pct"])
            p["search_review_target_type"] = str(t["search_review_target_type"])
            count = int(t["target_count"])
            p["search_review_action"] = (
                "Reduce type above London NFA average"
                if count == 1
                else f"Reduce {count} types above London NFA average"
            )
            p["search_review_recommendation_state"] = "Suggested reduction"
            p["search_review_expected_searches_reduced"] = float(t["search_review_expected_searches_reduced"])
            p["search_review_expected_no_result_avoided"] = float(t["search_review_expected_no_result_avoided"])
            p["search_review_expected_positives_at_risk"] = float(t["search_review_expected_positives_at_risk"])
        else:
            p["search_review_recommended_pct"] = 0
            p["search_review_target_type"] = ""
            p["search_review_action"] = "No reduction suggested"
            p["search_review_recommendation_state"] = "No reduction suggested"
            p["search_review_expected_searches_reduced"] = 0
            p["search_review_expected_no_result_avoided"] = 0
            p["search_review_expected_positives_at_risk"] = 0
        if cid in unfair.index:
            for col in unfair_cols:
                p[col] = float(unfair.loc[cid, col]) if col in unfair.columns and pd.notna(unfair.loc[cid, col]) else 0.0
            p["search_review_unfair_searches_qtr"] = p["expected_quarterly_unfair_searches_to_london_normal"]
            p["search_review_london_average_excess_qtr"] = p["expected_quarterly_excess_searches_to_london_avg"]
        else:
            for col in unfair_cols:
                p[col] = 0.0
            p["search_review_unfair_searches_qtr"] = 0.0
            p["search_review_london_average_excess_qtr"] = 0.0
        protection_row = prot_lookup.get(cid)
        if protection_row:
            p["aggregate_crime_guardrail"] = _safe_str(protection_row.get("aggregate_crime_guardrail"), p.get("aggregate_crime_guardrail"))
            p["predicted_serious_harm_rank_pct"] = _jsonable(protection_row.get("predicted_serious_harm_rank_pct"))
            p["predicted_harm_weighted_serious_crime_score_rank_pct"] = _jsonable(
                protection_row.get("predicted_harm_weighted_serious_crime_score_rank_pct")
            )
            p["protection_need_band"] = _safe_str(protection_row.get("protection_need_band"), p.get("protection_need_band"))
    return fc


def optimise(strategy_id: str, budget_scale: float) -> dict:


    from src.optimisation import optimisation_reporting, weighted_ilp
    from src.optimisation.resource_costs import STRATEGY_WEIGHTS, budgets, decision_table

    table = decision_table()
    table = table[table["package_id"] != "P0"].copy()
    weights = STRATEGY_WEIGHTS.get(strategy_id, STRATEGY_WEIGHTS["High-Volume Fairness Coverage"])
    over = {k: v * float(budget_scale) for k, v in budgets().items()}
    alloc = weighted_ilp.solve(table, weights, strategy_id, budget_overrides=over)
    summ = optimisation_reporting.summarise(alloc)
    fc = _map_package_wards(strategy_id, alloc)
    return {
        "strategy_id": strategy_id,
        "budget_scale": budget_scale,
        "summary": _records(summ)[0] if not summ.empty else {},
        "map": fc,
    }


def _ward_base_geojson() -> dict:
    wards = _ward_metrics_with_cluster()
    ward_by_code = wards.drop_duplicates("ward_code").set_index("ward_code")
    bounds = json.loads(paths.WARD_BOUNDARIES.read_text())
    features = []
    for feat in bounds.get("features", []):
        props = feat.get("properties", {})
        code = _ward_id(props.get("WD24CD") or props.get("ward_code"))
        if code not in ward_by_code.index:
            continue
        row = ward_by_code.loc[code]
        path_count = int(_safe_float(row.get("ward_number_of_pathways_flagged")))
        if path_count <= 0:
            continue
        guardrail = _safe_str(row.get("aggregate_crime_guardrail"), "Low")
        out_props = {
            "cluster_id": code,
            "ward_code": code,
            "source_cluster_id": _safe_str(row.get("source_cluster_id")),
            "cluster_name": f"{_safe_str(row.get('ward_name'), props.get('WD24NM') or code)} ward",
            "member_ward_names": _safe_str(row.get("ward_name"), props.get("WD24NM") or code),
            "boroughs": _safe_str(row.get("borough")),
            "dominant_fairness_pathway": _safe_str(row.get("ward_fairness_pathways")),
            "target_reduction_category": _safe_str(row.get("target_reduction_category")),
            "quarterly_total_encounters": _safe_float(row.get("total_stops")),
            "total_stops": _safe_float(row.get("total_stops")),
            "no_result_stops": _safe_float(row.get("no_result_stops")),
            "positive_outcomes": _safe_float(row.get("positive_outcomes")),
            "aggregate_crime_guardrail": guardrail,
            "protection_need_band": _band_from_guardrail(guardrail),
            "expected_quarterly_unfair_searches_to_london_normal": _safe_float(
                row.get("ward_excess_searches_to_london_normal_quarter")
            ),
            "excess_searches_to_london_normal_annual": _safe_float(row.get("ward_excess_searches_to_london_normal_annual")),
            "london_avg_ward_stop_rate_per_1000": _safe_float(row.get("ward_london_avg_stop_rate_per_1000")),
            "london_normal_ward_stop_rate_per_1000": _safe_float(row.get("ward_london_normal_stop_rate_per_1000")),
            "ward_stop_rate_vs_london_avg_ratio": _safe_float(row.get("ward_stop_rate_vs_london_avg_ratio")),
            "ward_number_of_pathways_flagged": path_count,
            "ward_monitor_trait_count": int(_safe_float(row.get("ward_monitor_trait_count"))),
            "criticalness_level": _safe_str(row.get("ward_criticalness_level")),
        }
        features.append({"type": "Feature", "properties": out_props, "geometry": feat.get("geometry")})
    return {"type": "FeatureCollection", "features": features}


def _map_package_wards(strategy_id: str, allocation_df: pd.DataFrame | None = None) -> dict:
    fc = _ward_base_geojson()
    alloc_lookup = _ward_allocation_lookup(strategy_id, allocation_df)
    prot_lookup = _ward_protection_lookup()
    prot_cols = [
        "aggregate_crime_guardrail",
        "predicted_serious_harm_per_1000_residents",
        "predicted_serious_harm_rank_pct",
        "predicted_harm_weighted_serious_crime_score_per_1000_residents",
        "predicted_harm_weighted_serious_crime_score_rank_pct",
        "london_serious_harm_avg_per_1000_residents",
        "london_harm_weighted_serious_crime_score_avg_per_1000_residents",
        "protection_need_band",
    ]
    for feat in fc["features"]:
        code = feat["properties"].get("ward_code")
        p = feat["properties"]
        alloc_row = alloc_lookup.get(code)
        has_multipath = int(_safe_float(p.get("ward_number_of_pathways_flagged"))) > 0
        if alloc_row and has_multipath:
            p["allocated_package_id"] = str(alloc_row.get("package_id", "P0"))
            p["allocated_package_name"] = str(alloc_row.get("package_name", _package_name(p["allocated_package_id"])))
        else:
            p["allocated_package_id"] = "P0"
            p["allocated_package_name"] = "Monitor Only"
        prot_row = prot_lookup.get(code)
        if prot_row:
            for col in prot_cols:
                p[col] = _jsonable(prot_row.get(col))
        else:
            p["protection_need_band"] = p.get("protection_need_band", "Low")
    return fc


def map_package_clusters(strategy_id: str) -> dict:

    return _map_package_wards(strategy_id)
