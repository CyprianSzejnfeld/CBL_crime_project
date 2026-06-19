from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _bool_col(df: pd.DataFrame, col: str) -> pd.Series:
    return df.get(col, pd.Series(False, index=df.index)).fillna(False).astype(bool)


def _num_col(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(df.get(col, pd.Series(default, index=df.index)), errors="coerce").fillna(default)


def _ensure_selected_pathways(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    sufficient = _num_col(out, "rolling_stops_12m").ge(config.MIN_LSOA_STRONG_STOPS_12M)
    substantial = _bool_col(out, "substantial_oversearch_flag")
    much = _bool_col(out, "much_oversearch_flag")
    very_low = _bool_col(out, "very_low_yield_actionability_flag")
    racial = _bool_col(out, "any_racial_pathway_flag")
    highly_deprived = _num_col(out, "deprivation_percentile").ge(80)
    stop_ratio = _num_col(out, "stop_rate_vs_london_lsoa_avg_ratio")
    above_normal_rate = stop_ratio.ge(config.SUBSTANTIAL_OVERSEARCH_RATIO) & _num_col(
        out, "excess_searches_to_london_lsoa_normal_annual"
    ).gt(0)
    much_above_avg_rate = stop_ratio.ge(config.MUCH_OVERSEARCH_RATIO) & _num_col(
        out, "excess_searches_to_london_lsoa_normal_annual"
    ).gt(0)

    out["oversearch_low_yield_path_flag"] = much & very_low & sufficient
    out["deprivation_oversearch_path_flag"] = highly_deprived & substantial & sufficient
    out["racial_oversearch_low_yield_path_flag"] = racial & sufficient
    out["deprivation_burden_flag"] = out["deprivation_oversearch_path_flag"]
    out["deprivation_oversearch_low_yield_flag"] = out["deprivation_oversearch_path_flag"]
    out["extreme_oversearch_low_yield_flag"] = out["oversearch_low_yield_path_flag"]
    out["racial_oversearch_low_yield_flag"] = out["racial_oversearch_low_yield_path_flag"]
    path_cols = ["oversearch_low_yield_path_flag", "deprivation_oversearch_path_flag", "racial_oversearch_low_yield_path_flag"]
    path_count = out[path_cols].sum(axis=1).astype(int)
    out["number_of_pathways_flagged"] = path_count
    out["any_fairness_pathway_flag"] = path_count.gt(0)
    out["deprivation_trait_flag"] = highly_deprived
    out["oversearch_trait_flag"] = substantial | above_normal_rate
    out["much_oversearch_trait_flag"] = much | much_above_avg_rate
    out["very_low_yield_trait_flag"] = very_low
    out["low_yield_trait_flag"] = very_low
    out["racial_trait_flag"] = out["racial_oversearch_low_yield_path_flag"]
    independent_cols = [
        "oversearch_trait_flag",
        "much_oversearch_trait_flag",
        "deprivation_trait_flag",
        "low_yield_actionability_flag",
        "very_low_yield_trait_flag",
        "racial_trait_flag",
    ]
    out["independent_flag_count"] = out[independent_cols].sum(axis=1).astype(int)
    out["any_independent_flag"] = out["independent_flag_count"].gt(0)
    out["monitor_trait_count"] = out["independent_flag_count"]
    out["monitor_trait_labels"] = out.apply(_monitor_trait_labels, axis=1)
    out["criticalness_score"] = np.select(
        [
            path_count.ge(3),
            path_count.eq(2),
            path_count.eq(1),
            out["any_independent_flag"],
        ],
        [3.0, 2.0, 1.0, 0.5],
        default=0.0,
    )
    out["criticalness_level"] = np.select(
        [
            path_count.ge(3),
            path_count.eq(2),
            path_count.eq(1),
            out["any_independent_flag"],
        ],
        ["Three multipaths", "Two multipaths", "One multipath", "Monitor trait"],
        default="No signal",
    )
    return out


def add_review_priority(latest: pd.DataFrame) -> pd.DataFrame:
    out = _ensure_selected_pathways(latest)
    pathway_cols = [
        "oversearch_low_yield_path_flag",
        "deprivation_oversearch_path_flag",
        "racial_oversearch_low_yield_path_flag",
    ]
    for col in pathway_cols:
        if col not in out.columns:
            out[col] = False
    out["number_of_pathways_flagged"] = out[pathway_cols].sum(axis=1).astype(int)
    out["any_fairness_pathway_flag"] = out["number_of_pathways_flagged"].gt(0)
    out["fairness_pathway_labels"] = out.apply(_labels_for_row, axis=1)
    out["overall_review_priority"] = np.select(
        [
            out["number_of_pathways_flagged"].ge(2),
            out["number_of_pathways_flagged"].eq(1),
            out.get("any_independent_flag", pd.Series(False, index=out.index)).fillna(False).astype(bool),
        ],
        ["Critical Review", "Priority Review", "Monitor"],
        default="No signal",
    )
    out["fairness_review_status"] = out["overall_review_priority"]
    out["intervention_actionability_status"] = np.select(
        [
            out["crime_guardrail_level"].eq("Severe") & out["any_fairness_pathway_flag"],
            out["any_fairness_pathway_flag"],
        ],
        [
            "Review only due to safety guardrail",
            "Contributes to spatial cluster",
        ],
        default="Not actionable individually",
    )
    return out


def add_actionable_fairness_score(latest: pd.DataFrame) -> pd.DataFrame:
    out = _ensure_selected_pathways(latest)

    racial_cols = [f"racial_pathway_{g}_score_0_100" for g in config.RACIAL_GROUPS]
    racial_cols = [c for c in racial_cols if c in out.columns]
    racial_score = out[racial_cols].max(axis=1).fillna(0) if racial_cols else pd.Series(0, index=out.index)

    deprivation_combo = (
        0.34 * out["deprivation_percentile"].fillna(0)
        + 0.33 * out["over_search_score_0_100"].fillna(0)
        + 0.33 * out["low_yield_actionability_score_0_100"].fillna(0)
    )
    racial_combo = (
        0.45 * racial_score
        + 0.30 * out["low_yield_actionability_score_0_100"].fillna(0)
        + 0.25 * out["stop_burden_percentile"].fillna(0)
    )
    extreme_combo = (
        0.45 * out["over_search_score_0_100"].fillna(0)
        + 0.35 * out["low_yield_actionability_score_0_100"].fillna(0)
        + 0.20 * out["no_result_burden_percentile"].fillna(0)
    )
    base = pd.concat(
        [
            deprivation_combo.where(out["deprivation_oversearch_low_yield_flag"], 0),
            racial_combo.where(out["racial_oversearch_low_yield_flag"], 0),
            extreme_combo.where(out["extreme_oversearch_low_yield_flag"], 0),
        ],
        axis=1,
    ).max(axis=1)

    combo_count = out[
        [
            "deprivation_oversearch_low_yield_flag",
            "racial_oversearch_low_yield_flag",
            "extreme_oversearch_low_yield_flag",
        ]
    ].sum(axis=1)
    synergy = (
        out["deprivation_oversearch_low_yield_flag"].astype(int) * 8
        + out["racial_oversearch_low_yield_flag"].astype(int) * 10
        + out["extreme_oversearch_low_yield_flag"].astype(int) * 8
        + combo_count.ge(2).astype(int) * 8
        + _bool_col(out, "trust_context_warning_flag").astype(int) * 4
    )
    out["actionable_fairness_score_0_100"] = (base + synergy).clip(0, 100)
    out["fairness_reduction_priority"] = np.select(
        [
            out["actionable_fairness_score_0_100"].ge(90),
            out["actionable_fairness_score_0_100"].ge(80),
            out["actionable_fairness_score_0_100"].ge(65),
            out["actionable_fairness_score_0_100"].ge(50),
        ],
        ["Critical reduction review", "Priority reduction review", "Review candidate", "Monitor"],
        default="No reduction signal",
    )
    out["v2_reduction_candidate_flag"] = out["actionable_fairness_score_0_100"].ge(65)
    out["dominant_unfairness_pattern"] = out.apply(_dominant_pattern, axis=1)
    out["v2_score_explanation"] = out.apply(_score_explanation, axis=1)
    return out


def _dominant_pattern(row: pd.Series) -> str:
    if row.get("extreme_oversearch_low_yield_flag"):
        return "Over-search + very low yield"
    if row.get("racial_oversearch_low_yield_flag"):
        return "Racial over-search + low yield"
    if row.get("deprivation_oversearch_low_yield_flag"):
        return "Deprivation + over-search"
    return "No strong unfairness pattern"


def _score_explanation(row: pd.Series) -> str:
    parts = []
    if row.get("deprivation_oversearch_low_yield_flag"):
        parts.append("high deprivation plus search rate above London-normal LSOA range")
    if row.get("racial_oversearch_low_yield_flag"):
        parts.append("racial over-exposure plus low-result evidence for affected group")
    if row.get("extreme_oversearch_low_yield_flag"):
        parts.append("much higher search rate than London LSOA average plus very low-result evidence")
    if not parts:
        return "No strong V2 reduction pathway is active."
    return "Flagged because " + "; ".join(parts) + "."


def _labels_for_row(row: pd.Series) -> str:
    labels = []
    if row.get("extreme_oversearch_low_yield_flag"):
        labels.append("Over-search + very low yield")
    if row.get("deprivation_oversearch_low_yield_flag"):
        labels.append("Deprivation + over-search")
    if row.get("racial_oversearch_low_yield_flag"):
        labels.append("Racial over-search + low yield")
    return "; ".join(labels)


def _monitor_trait_labels(row: pd.Series) -> str:
    labels = []
    if row.get("oversearch_trait_flag") and not row.get("any_fairness_pathway_flag"):
        labels.append("Over-search")
    if row.get("much_oversearch_trait_flag") and not row.get("any_fairness_pathway_flag"):
        labels.append("Much over-search")
    if row.get("deprivation_trait_flag") and not row.get("any_fairness_pathway_flag"):
        labels.append("High deprivation")
    if row.get("low_yield_actionability_flag") and not row.get("any_fairness_pathway_flag"):
        labels.append("Low yield")
    if row.get("very_low_yield_trait_flag") and not row.get("any_fairness_pathway_flag"):
        labels.append("Very low yield")
    if row.get("racial_trait_flag") and not row.get("any_fairness_pathway_flag"):
        labels.append("Racial signal")
    return ";".join(labels)

