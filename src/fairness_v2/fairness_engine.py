from __future__ import annotations

import pandas as pd

from src import config as base_config
from .actionability import (
    add_actionable_fairness_score,
    add_review_priority,
)
from .burden_pathway import add_excess_burden_pathway
from .data_interface import load_inputs
from .deprivation_pathway import add_deprivation_burden_pathway
from .low_yield_pathway import add_low_yield_actionability
from .racial_pathway import add_racial_pathways
from .trust_context import add_trust_context_warning


def build_lsoa_pathways(latest: pd.DataFrame) -> pd.DataFrame:
    out = add_excess_burden_pathway(latest)
    out = add_deprivation_burden_pathway(out)
    out = add_racial_pathways(out)
    out = add_low_yield_actionability(out)
    out = add_review_priority(out)
    out = add_trust_context_warning(out)
    out = add_actionable_fairness_score(out)
    return out


def save_lsoa_outputs(latest: pd.DataFrame) -> None:

    base_cols = [
        "lsoa21cd",
        "lsoa21nm",
        "borough",
        "month",
        "overall_review_priority",
        "fairness_review_status",
        "intervention_actionability_status",
        "any_fairness_pathway_flag",
        "number_of_pathways_flagged",
        "fairness_pathway_labels",
        "criticalness_score",
        "criticalness_level",
        "resident_denominator_caution_flag",
        "rolling_stops_12m",
        "rolling_no_result_stops_12m",
        "stop_rate_per_1000",
        "no_result_stop_rate_per_1000",
        "london_avg_lsoa_stop_rate_per_1000",
        "london_normal_lsoa_stop_rate_per_1000",
        "stop_rate_vs_london_lsoa_avg_ratio",
        "excess_searches_to_london_lsoa_normal_annual",
        "excess_searches_to_london_lsoa_normal_month",
        "stop_burden_percentile",
        "no_result_burden_percentile",
        "deprivation_percentile",
        "over_search_score_0_100",
        "over_search_residual",
        "over_search_residual_rate_per_1000",
        "excess_stop_share",
        "substantial_oversearch_flag",
        "much_oversearch_flag",
        "deprivation_trait_flag",
        "oversearch_trait_flag",
        "much_oversearch_trait_flag",
        "low_yield_trait_flag",
        "racial_trait_flag",
        "very_low_yield_trait_flag",
        "monitor_trait_count",
        "monitor_trait_labels",
        "independent_flag_count",
        "any_independent_flag",
        "oversearch_low_yield_path_flag",
        "deprivation_oversearch_path_flag",
        "racial_oversearch_low_yield_path_flag",
        "eligible_reduction_categories",
        "very_low_yield_categories",
        "low_yield_actionability_score_0_100",
        "low_yield_actionability_flag",
        "very_low_yield_actionability_flag",
        "low_yield_actionability_reliability",
        "low_yield_actionability_reason",
        "actionable_fairness_score_0_100",
        "estimated_post_reduction_fairness_score_0_100",
        "fairness_reduction_priority",
        "v2_reduction_candidate_flag",
        "dominant_unfairness_pattern",
        "v2_score_explanation",
        "deprivation_oversearch_low_yield_flag",
        "racial_oversearch_low_yield_flag",
        "extreme_oversearch_low_yield_flag",
        "trust_context_warning_flag",
        "trust_context_warning_text",
        "any_racial_pathway_flag",
        "borough_low_trust_flag",
        "trust_context_band",
        "crime_guardrail_level",
        "max_reduction_cap_from_crime_guardrail",
    ]
    pathway_cols = [
        c
        for c in latest.columns
        if c.startswith("excess_burden_")
        or c.startswith("deprivation_burden_")
        or c.startswith("racial_pathway_")
        or c.endswith("_racial_disproportionality_ratio_raw")
        or c.endswith("_racial_disproportionality_ratio_capped")
        or c.endswith("_group_stop_burden_percentile")
        or c.endswith("_low_yield_probability")
        or c.endswith("_eligible_reduction_category")
    ]
    pathway_cols = [c for c in pathway_cols if not c.endswith("_moderate_flag")]
    keep = [c for c in base_cols + pathway_cols if c in latest.columns]
    out = latest[keep].copy()
    out.to_csv(base_config.FAIRNESS_V2_LSOA_PATHWAYS_CSV, index=False)


def run() -> pd.DataFrame:
    latest = build_lsoa_pathways(load_inputs())
    save_lsoa_outputs(latest)
    return latest


def main() -> None:
    run()


if __name__ == "__main__":
    main()
