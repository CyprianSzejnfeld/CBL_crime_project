from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .smoothing import beta_posterior_summary, empirical_bayes_rate, percentile_0_100, safe_divide


def add_low_yield_actionability(latest: pd.DataFrame) -> pd.DataFrame:
    out = latest.copy()
    category_flags = []
    very_low_flags = []
    category_labels = []
    action_scores = []

    for category in config.REDUCIBLE_CATEGORIES:
        stops = pd.to_numeric(out.get(f"rolling_12m_stops_{category}", 0), errors="coerce").fillna(0)
        positives = pd.to_numeric(out.get(f"rolling_12m_positive_outcomes_{category}", 0), errors="coerce").fillna(0)
        arrests = pd.to_numeric(out.get(f"rolling_12m_arrests_{category}", 0), errors="coerce").fillna(0)
        no_result = pd.to_numeric(out.get(f"rolling_12m_no_result_stops_{category}", 0), errors="coerce").fillna(0)

        benchmark_success = positives.sum() / stops.sum() if stops.sum() else np.nan
        benchmark_no_result = no_result.sum() / stops.sum() if stops.sum() else np.nan
        benchmark_success_clean = 0.0 if pd.isna(benchmark_success) else float(benchmark_success)
        benchmark_no_result_clean = 0.0 if pd.isna(benchmark_no_result) else float(benchmark_no_result)
        out[f"{category}_raw_success_rate"] = safe_divide(positives, stops)
        out[f"{category}_raw_arrest_rate"] = safe_divide(arrests, stops)
        out[f"{category}_raw_no_result_rate"] = safe_divide(no_result, stops)
        out[f"{category}_success_rate"] = out[f"{category}_raw_success_rate"].fillna(0)
        out[f"{category}_no_result_rate"] = out[f"{category}_raw_no_result_rate"].fillna(0)
        out[f"{category}_london_success_rate"] = benchmark_success_clean
        out[f"{category}_london_no_result_rate"] = benchmark_no_result_clean
        out[f"{category}_smoothed_success_rate"] = empirical_bayes_rate(
            positives, stops, benchmark_success, alpha=config.ALPHA
        )
        out[f"{category}_smoothed_arrest_rate"] = empirical_bayes_rate(
            arrests, stops, arrests.sum() / stops.sum() if stops.sum() else np.nan, alpha=config.ALPHA
        )
        out[f"{category}_smoothed_no_result_rate"] = empirical_bayes_rate(
            no_result, stops, benchmark_no_result, alpha=config.ALPHA
        )
        success_post = beta_posterior_summary(positives, stops, benchmark_success, alpha=config.ALPHA)
        no_result_post = beta_posterior_summary(no_result, stops, benchmark_no_result, alpha=config.ALPHA)
        p_low_success = success_post["probability_rate_below_benchmark"].fillna(0)
        p_high_no_result = no_result_post["probability_rate_above_benchmark"].fillna(0)
        evidence_prob = np.maximum(p_low_success, p_high_no_result)
        volume_pct = percentile_0_100(stops)
        no_result_volume_pct = percentile_0_100(no_result)
        score = (0.40 * no_result_volume_pct.fillna(0) + 0.25 * volume_pct.fillna(0) + 0.35 * evidence_prob * 100).clip(0, 100)
        low_yield_margin = (
            out[f"{category}_no_result_rate"].ge(benchmark_no_result_clean + config.LOW_YIELD_MARGIN)
            | out[f"{category}_success_rate"].le(max(benchmark_success_clean - config.LOW_YIELD_MARGIN, 0))
        )
        very_low_yield_margin = (
            out[f"{category}_no_result_rate"].ge(benchmark_no_result_clean + config.VERY_LOW_YIELD_MARGIN)
            | out[f"{category}_success_rate"].le(max(benchmark_success_clean - config.VERY_LOW_YIELD_MARGIN, 0))
        )
        eligible = (
            stops.ge(config.MIN_CATEGORY_STOPS_12M)
            & evidence_prob.ge(config.LOW_YIELD_PROB_THRESHOLD)
            & low_yield_margin
        )
        very_low = (
            stops.ge(config.MIN_CATEGORY_STOPS_12M)
            & evidence_prob.ge(config.STRONG_LOW_YIELD_PROB_THRESHOLD)
            & very_low_yield_margin
        )
        if category in config.PROTECTED_CATEGORIES:
            eligible = pd.Series(False, index=out.index)
            very_low = pd.Series(False, index=out.index)
        out[f"{category}_low_yield_probability"] = evidence_prob
        out[f"{category}_low_yield_actionability_score_0_100"] = score
        out[f"{category}_eligible_reduction_category"] = eligible
        out[f"{category}_very_low_yield_flag"] = very_low
        category_flags.append(eligible)
        very_low_flags.append(very_low)
        action_scores.append(score)
        category_labels.append(category)

    flag_matrix = pd.concat(category_flags, axis=1)
    very_low_matrix = pd.concat(very_low_flags, axis=1)
    score_matrix = pd.concat(action_scores, axis=1)
    flag_matrix.columns = category_labels
    very_low_matrix.columns = category_labels
    score_matrix.columns = category_labels
    out["eligible_reduction_categories"] = [
        ";".join([cat for cat in category_labels if bool(flag_matrix.loc[idx, cat])])
        for idx in out.index
    ]
    out["very_low_yield_categories"] = [
        ";".join([cat for cat in category_labels if bool(very_low_matrix.loc[idx, cat])])
        for idx in out.index
    ]
    out["low_yield_actionability_score_0_100"] = score_matrix.max(axis=1).fillna(0)
    out["low_yield_actionability_flag"] = out["eligible_reduction_categories"].astype(str).ne("")
    out["very_low_yield_actionability_flag"] = out["very_low_yield_categories"].astype(str).ne("")
    out["low_yield_actionability_reliability"] = np.select(
        [
            out["low_yield_actionability_flag"],
            out[[f"rolling_12m_stops_{c}" for c in config.REDUCIBLE_CATEGORIES if f"rolling_12m_stops_{c}" in out.columns]]
            .sum(axis=1)
            .lt(config.MIN_CATEGORY_STOPS_12M),
        ],
        ["reliable category evidence", "insufficient category-specific volume"],
        default="no strong low-yield category evidence",
    )
    out["low_yield_actionability_reason"] = np.where(
        out["low_yield_actionability_flag"],
        "One or more known search categories has sufficient volume and low-yield evidence.",
        "No known search category meets low-yield evidence and volume thresholds.",
    )
    return out
