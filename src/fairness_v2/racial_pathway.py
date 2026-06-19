from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .smoothing import beta_posterior_summary, empirical_bayes_rate, percentile_0_100, safe_divide


def _central_denominator_caution(df: pd.DataFrame) -> pd.Series:
    central = df["borough"].isin(config.CENTRAL_BOROUGH_DENOMINATOR_CAUTION)
    very_high_stops_per_resident = df["stop_rate_per_1000"].ge(df["stop_rate_per_1000"].quantile(0.95))
    return central & very_high_stops_per_resident


def add_racial_pathways(latest: pd.DataFrame) -> pd.DataFrame:
    out = latest.copy()
    known = pd.to_numeric(out["rolling_stops_known_ethnicity_12m"], errors="coerce").fillna(0)
    out["resident_denominator_caution_flag"] = _central_denominator_caution(out)
    group_any_flags = []
    group_any_moderate = []

    for group in config.RACIAL_GROUPS:
        label = config.RACIAL_GROUP_LABELS[group]
        stops_col = f"rolling_stops_{group}_12m"
        pop_col = f"{group}_population"
        prop_col = f"prop_{group}"
        positive_col = f"rolling_12m_positive_stops_{group}"

        group_stops = pd.to_numeric(out.get(stops_col, 0), errors="coerce").fillna(0)
        group_pop = pd.to_numeric(out.get(pop_col, 0), errors="coerce").fillna(0)
        fallback_share = pd.to_numeric(out.get(prop_col, 0), errors="coerce").fillna(0)
        if "population" in out.columns:
            group_share = safe_divide(group_pop, pd.to_numeric(out["population"], errors="coerce").fillna(0)).fillna(0)
        else:
            group_share = fallback_share
        stop_share = safe_divide(group_stops, known)
        ratio_raw = safe_divide(stop_share, group_share)
        ratio_capped = ratio_raw.clip(upper=5)
        group_rate = safe_divide(group_stops, group_pop) * 1000
        group_burden_pct = percentile_0_100(group_rate)




        positives = pd.to_numeric(out.get(positive_col, 0), errors="coerce").fillna(0)
        approx_no_result = (group_stops - positives).clip(lower=0)

        met_success = positives.sum() / group_stops.sum() if group_stops.sum() else np.nan
        met_no_result = approx_no_result.sum() / group_stops.sum() if group_stops.sum() else np.nan
        smoothed_success = empirical_bayes_rate(positives, group_stops, met_success, alpha=config.ALPHA)
        smoothed_no_result = empirical_bayes_rate(approx_no_result, group_stops, met_no_result, alpha=config.ALPHA)

        success_post = beta_posterior_summary(positives, group_stops, met_success, alpha=config.ALPHA)
        no_result_post = beta_posterior_summary(approx_no_result, group_stops, met_no_result, alpha=config.ALPHA)
        p_success_not_higher = success_post["probability_rate_below_benchmark"].fillna(0)
        p_no_result_high = no_result_post["probability_rate_above_benchmark"].fillna(0)

        eligible = (
            group_share.ge(config.MIN_GROUP_POP_SHARE)
            & known.ge(config.MIN_KNOWN_ETHNICITY_STOPS_12M)
            & group_stops.ge(config.MIN_GROUP_STOPS_12M)
        )
        evidence_score = np.maximum(p_success_not_higher, p_no_result_high)
        group_no_result_rate = safe_divide(approx_no_result, group_stops).fillna(0)
        group_success_rate = safe_divide(positives, group_stops).fillna(0)
        met_no_result_clean = 0.0 if pd.isna(met_no_result) else float(met_no_result)
        met_success_clean = 0.0 if pd.isna(met_success) else float(met_success)
        low_yield_gap = (
            group_no_result_rate.ge(met_no_result_clean + config.LOW_YIELD_MARGIN)
            | group_success_rate.le(max(met_success_clean - config.LOW_YIELD_MARGIN, 0))
        )
        evidence = evidence_score.ge(config.LOW_YIELD_PROB_THRESHOLD) & low_yield_gap
        flag = eligible & ratio_capped.ge(1.5) & group_burden_pct.ge(75) & evidence
        moderate = eligible & ratio_capped.ge(1.25) & group_burden_pct.ge(60) & ~flag

        score = (
            0.40 * percentile_0_100(ratio_capped).fillna(0)
            + 0.35 * group_burden_pct.fillna(0)
            + 0.25 * (evidence_score * 100)
        ).clip(0, 100)
        reliability = np.select(
            [
                ~eligible,
                eligible & out["resident_denominator_caution_flag"],
                eligible,
            ],
            [
                "insufficient LSOA-level evidence",
                "resident-denominator caution; use aggregate/category interpretation",
                "reliable",
            ],
            default="insufficient LSOA-level evidence",
        )
        reason = np.select(
            [
                flag & out["resident_denominator_caution_flag"],
                flag,
                moderate,
                ~eligible,
            ],
            [
                f"{label} stop exposure is disproportionate, but resident denominator is weak in high-footfall/central context.",
                f"{label} stop exposure is disproportionate and yield is not clearly higher than benchmark.",
                f"{label} disproportionality signal is elevated but below strong pathway threshold.",
                f"{label} racial indicator has insufficient LSOA-level sample size or population share.",
            ],
            default=f"No strong {label} racial pathway signal.",
        )

        prefix = f"racial_pathway_{group}"
        out[f"{prefix}_score_0_100"] = score
        out[f"{prefix}_flag"] = flag
        out[f"{prefix}_moderate_flag"] = moderate
        out[f"{prefix}_reliability"] = reliability
        out[f"{prefix}_reason"] = reason
        out[f"{prefix}_low_yield_probability"] = evidence_score
        out[f"{prefix}_low_yield_gap_flag"] = low_yield_gap
        out[f"{group}_racial_disproportionality_ratio_raw"] = ratio_raw
        out[f"{group}_racial_disproportionality_ratio_capped"] = ratio_capped
        out[f"{group}_group_stop_burden_percentile"] = group_burden_pct
        out[f"{group}_group_success_rate"] = group_success_rate
        out[f"{group}_group_no_result_rate"] = group_no_result_rate
        out[f"{group}_london_group_success_rate"] = met_success_clean
        out[f"{group}_london_group_no_result_rate"] = met_no_result_clean
        out[f"{group}_smoothed_group_success_rate"] = smoothed_success
        out[f"{group}_smoothed_group_no_result_rate"] = smoothed_no_result
        out[f"{group}_prob_success_below_benchmark"] = p_success_not_higher
        out[f"{group}_prob_no_result_above_benchmark"] = p_no_result_high
        group_any_flags.append(flag)
        group_any_moderate.append(moderate)

    out["any_racial_pathway_flag"] = pd.concat(group_any_flags, axis=1).any(axis=1)
    out["any_racial_pathway_moderate_flag"] = pd.concat(group_any_moderate, axis=1).any(axis=1)
    return out
