from __future__ import annotations

import pandas as pd

from src import config

WARD_METRICS = config.PROCESSED_DIR / "ward_quarter_intervention_metrics.parquet"
EXISTING_CLUSTERS = config.PROCESSED_DIR / "ward_intervention_clusters.csv"
REGIME_PROFILES = config.PROCESSED_DIR / "cluster_search_regime_profiles.csv"

OUT_PARQUET = config.PROCESSED_DIR / "ward_clusters_latest.parquet"


def _primary_regime() -> dict[str, str]:
    if not REGIME_PROFILES.exists():
        return {}
    prof = pd.read_csv(REGIME_PROFILES)
    out = {}
    for cid, grp in prof.groupby("cluster_id"):
        nz = grp[(~grp["search_regime"].eq("combined_non_weapon")) & (grp["quarterly_stops"] >= config.WARD_CLUSTER_MIN_QUARTERLY_SEARCHES_REDUCED)]
        if nz.empty:
            out[cid] = "none (insufficient search-type volume)"
        else:
            best = nz.sort_values("posterior_prob_low_yield", ascending=False).iloc[0]
            out[cid] = str(best["search_regime"])
    return out


def _secondary_pathways(member_wards: str, ward_path: dict[str, str], dominant: str) -> str:
    codes = [w for w in str(member_wards).split(";") if w]
    paths = {ward_path.get(c, "") for c in codes}
    paths.discard(dominant)
    paths.discard("")
    paths.discard("No strong unfairness pattern")
    return ";".join(sorted(paths))


def _group_concern(pathway: str) -> str:
    p = str(pathway).lower()
    if "black" in p:
        return "Black"
    if "asian" in p:
        return "Asian"
    if "racial" in p or "disproportion" in p:
        return "Group-specific (see racial pathway)"
    return ""


def run() -> pd.DataFrame:
    wards = pd.read_parquet(WARD_METRICS)


    clusters = pd.read_csv(EXISTING_CLUSTERS)
    ward_path = wards.set_index("ward_code")["dominant_fairness_pathway"].to_dict()
    regime = _primary_regime()
    clusters["secondary_pathways"] = clusters.apply(
        lambda r: _secondary_pathways(r["member_wards"], ward_path, r["dominant_fairness_pathway"]), axis=1
    )
    clusters["relevant_group_concern"] = clusters["dominant_fairness_pathway"].apply(_group_concern)
    clusters["primary_search_regime"] = clusters["cluster_id"].map(regime).fillna("none")
    clusters["quarterly_total_encounters"] = clusters["total_stops"]
    clusters["quarterly_no_result_burden"] = clusters["no_result_stops"]
    clusters["intervention_eligibility"] = clusters.apply(
        lambda r: "Review/monitor only (safety or volume)"
        if (r["aggregate_crime_guardrail"] == "Severe" or r["quarterly_baseline_reducible_searches"] <= 0)
        else "Eligible for intervention package",
        axis=1,
    )

    out = clusters.drop(columns=["cluster_geometry"]).copy()
    out.to_parquet(OUT_PARQUET, index=False)

    return out


if __name__ == "__main__":
    run()
