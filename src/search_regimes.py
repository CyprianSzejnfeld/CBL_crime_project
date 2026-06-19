from __future__ import annotations

import pandas as pd

from src import config
from src.fairness_v2.smoothing import beta_posterior_summary, empirical_bayes_rate

OUT_CSV = config.PROCESSED_DIR / "cluster_search_regime_profiles.csv"

WINDOW_MONTHS = 12
REGIMES: dict[str, list[str]] = {
    "drugs": ["drugs"],
    "stolen_property": ["stolen_property"],
    "other_non_weapon": ["other_non_weapon"],
    "combined_non_weapon": ["drugs", "stolen_property", "other_non_weapon"],
    "offensive_weapons": ["offensive_weapons"],
}
PROTECTED: set[str] = set(config.PROTECTED_SEARCH_CATEGORIES)
MIN_MEANINGFUL_QUARTERLY = config.WARD_CLUSTER_MIN_QUARTERLY_SEARCHES_REDUCED


def _cluster_membership() -> pd.DataFrame:
    latest = config.PROCESSED_DIR / "ward_clusters_latest.parquet"
    if latest.exists():
        cl = pd.read_parquet(latest)
    else:
        cl = pd.read_csv(config.PROCESSED_DIR / "ward_intervention_clusters.csv")
    keep = [c for c in ["cluster_id", "dominant_fairness_pathway", "member_lsoas"] if c in cl.columns]
    return cl[keep].copy()


def _lsoa_window_volume() -> pd.DataFrame:
    cats = pd.read_parquet(config.STOP_SEARCH_CATEGORIES_PATH)
    cats["month"] = cats["month"].astype(str).str[:7]
    months = sorted(cats["month"].unique())[-WINDOW_MONTHS:]
    cats = cats[cats["month"].isin(months)]
    metrics = ["stops", "no_result_stops", "positive_outcomes", "arrests"]
    cols = [f"{m}_{c}" for m in metrics for c in ["drugs", "stolen_property", "other_non_weapon", "offensive_weapons"]]
    cols = [c for c in cols if c in cats.columns]
    return cats.groupby("lsoa21cd", as_index=False)[cols].sum()


def _met_benchmarks(vol: pd.DataFrame) -> dict[str, dict[str, float]]:
    bench: dict[str, dict[str, float]] = {}
    for regime, parts in REGIMES.items():
        stops = sum(vol.get(f"stops_{c}", 0).sum() for c in parts)
        nr = sum(vol.get(f"no_result_stops_{c}", 0).sum() for c in parts)
        pos = sum(vol.get(f"positive_outcomes_{c}", 0).sum() for c in parts)
        arr = sum(vol.get(f"arrests_{c}", 0).sum() for c in parts)
        s = max(float(stops), 1.0)
        bench[regime] = {
            "no_result": float(nr) / s,
            "positive": float(pos) / s,
            "arrest": float(arr) / s,
        }
    return bench


def _package_relevance(no_result_rate: float, quarterly_stops: float) -> str:
    if quarterly_stops < MIN_MEANINGFUL_QUARTERLY:
        return "Insufficient volume for a monitored review component"
    if no_result_rate >= 0.70:
        return "Low-yield review candidate (P1 / P5 component)"
    return "Monitor"


def build() -> pd.DataFrame:
    members = _cluster_membership()
    vol = _lsoa_window_volume()
    bench = _met_benchmarks(vol)
    vol_idx = vol.set_index("lsoa21cd")

    rows = []
    for _, cl in members.iterrows():
        lsoas = [x for x in str(cl["member_lsoas"]).split(";") if x]
        sub = vol_idx.reindex(lsoas).fillna(0.0)
        cluster_total = sum(
            sub.get(f"stops_{c}", pd.Series(0.0)).sum()
            for c in ["drugs", "stolen_property", "other_non_weapon", "offensive_weapons"]
        )
        for regime, parts in REGIMES.items():
            stops = float(sum(sub.get(f"stops_{c}", pd.Series(0.0)).sum() for c in parts))
            nr = float(sum(sub.get(f"no_result_stops_{c}", pd.Series(0.0)).sum() for c in parts))
            pos = float(sum(sub.get(f"positive_outcomes_{c}", pd.Series(0.0)).sum() for c in parts))
            arr = float(sum(sub.get(f"arrests_{c}", pd.Series(0.0)).sum() for c in parts))
            rows.append({
                "cluster_id": cl["cluster_id"],
                "dominant_fairness_pathway": cl.get("dominant_fairness_pathway", ""),
                "search_regime": regime,
                "is_protected_context": regime in PROTECTED,
                "annual_stops": stops,
                "quarterly_stops": stops / 4.0,
                "share_of_cluster_stops": stops / cluster_total if cluster_total else 0.0,
                "no_result_count_annual": nr,
                "positive_outcome_count_annual": pos,
                "arrest_count_annual": arr,
                "_bench_nr": bench[regime]["no_result"],
                "_bench_pos": bench[regime]["positive"],
                "_bench_arr": bench[regime]["arrest"],
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["smoothed_no_result_rate"] = empirical_bayes_rate(
        df["no_result_count_annual"], df["annual_stops"], df["_bench_nr"], alpha=config.EMPIRICAL_BAYES_ALPHA
    )
    df["smoothed_positive_outcome_rate"] = empirical_bayes_rate(
        df["positive_outcome_count_annual"], df["annual_stops"], df["_bench_pos"], alpha=config.EMPIRICAL_BAYES_ALPHA
    )
    df["smoothed_arrest_rate"] = empirical_bayes_rate(
        df["arrest_count_annual"], df["annual_stops"], df["_bench_arr"], alpha=config.EMPIRICAL_BAYES_ALPHA
    )
    post = beta_posterior_summary(
        df["no_result_count_annual"], df["annual_stops"], df["_bench_nr"], alpha=config.EMPIRICAL_BAYES_ALPHA
    )
    df["posterior_prob_low_yield"] = post["probability_rate_above_benchmark"].values
    df["package_relevance"] = [
        _package_relevance(r.smoothed_no_result_rate, r.quarterly_stops)
        for r in df.itertuples()
    ]
    df = df.drop(columns=["_bench_nr", "_bench_pos", "_bench_arr"])
    return df


def run() -> pd.DataFrame:
    df = build()
    df.to_csv(OUT_CSV, index=False)
    return df


if __name__ == "__main__":
    run()
