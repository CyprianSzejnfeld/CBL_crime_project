from __future__ import annotations

import pandas as pd

from src import config
from src.packages.library import PACKAGES, RESOURCE_TYPES, package_cost

OUT_CSV = config.PROCESSED_DIR / "cluster_package_eligibility.csv"
CLUSTERS = config.PROCESSED_DIR / "ward_clusters_latest.parquet"
PROTECTION = config.PROCESSED_DIR / "cluster_protection_need_forecasts.csv"
REGIMES = config.PROCESSED_DIR / "cluster_search_regime_profiles.csv"

MEANINGFUL_VOLUME = config.WARD_CLUSTER_MIN_QUARTERLY_SEARCHES_REDUCED
ACTION_VOLUME = 50.0
LOW_YIELD_PROB = config.FAIRNESS_V2_LOW_YIELD_PROB_THRESHOLD


def cluster_features() -> pd.DataFrame:
    clusters = pd.read_parquet(CLUSTERS)
    protection = pd.read_csv(PROTECTION)[["cluster_id", "protection_need_band"]]
    static = pd.read_csv(config.STATIC_FEATURES_PATH, usecols=["lsoa21cd", "total_population"])
    population_by_lsoa = static.set_index("lsoa21cd")["total_population"].to_dict()

    regimes = pd.read_csv(REGIMES) if REGIMES.exists() else pd.DataFrame()
    low_yield_by_cluster = {}
    if not regimes.empty:
        for cid, grp in regimes.groupby("cluster_id"):
            nz = grp[(~grp["search_regime"].eq("combined_non_weapon")) & (grp["quarterly_stops"] >= MEANINGFUL_VOLUME)]
            low_yield_by_cluster[cid] = bool((nz["posterior_prob_low_yield"] >= LOW_YIELD_PROB).any())

    rows = []
    for _, cl in clusters.iterrows():
        lsoas = [x for x in str(cl["member_lsoas"]).split(";") if x]
        wards = [x for x in str(cl.get("member_wards", "")).split(";") if x]
        cluster_population = float(sum(population_by_lsoa.get(x, 0) for x in lsoas))
        excess_strong = float(cl.get("oversearch_low_yield_flagged_wards", cl.get("excess_burden_flagged_wards", 0)) or 0) > 0
        deprivation_strong = float(
            cl.get("deprivation_oversearch_flagged_wards", cl.get("deprivation_burden_flagged_wards", 0)) or 0
        ) > 0
        racial_strong = float(
            cl.get("racial_oversearch_low_yield_flagged_wards", cl.get("racial_pathway_flagged_wards", 0)) or 0
        ) > 0
        low_yield_actionable = bool(low_yield_by_cluster.get(cl["cluster_id"], False)) or float(
            cl.get("low_yield_flagged_wards", 0) or 0
        ) > 0
        n_from_cluster = int(cl.get("n_strong_pathways", 0) or 0)
        has_multipath = n_from_cluster > 0
        if not has_multipath:
            excess_strong = False
            deprivation_strong = False
            racial_strong = False
            low_yield_actionable = False
        n_strong = max(n_from_cluster, sum([excess_strong, deprivation_strong, racial_strong, low_yield_actionable]))
        rows.append({
            "cluster_id": cl["cluster_id"],
            "cluster_name": cl.get("cluster_name", cl["cluster_id"]),
            "fairness_source": cl.get("fairness_source", "direct_ward_level"),
            "excess_strong": excess_strong,
            "deprivation_strong": deprivation_strong,
            "racial_strong": racial_strong,
            "low_yield_actionable": low_yield_actionable,
            "has_multipath": has_multipath,
            "n_strong_pathways": n_strong,
            "trust_warning": bool(cl.get("trust_context_warning", False)),
            "meaningful_volume": float(cl.get("quarterly_baseline_reducible_searches", 0)) >= MEANINGFUL_VOLUME,
            "action_volume": float(cl.get("quarterly_total_encounters", 0)) >= ACTION_VOLUME,
            "quarterly_total_encounters": float(cl.get("quarterly_total_encounters", 0)),
            "quarterly_no_result_burden": float(cl.get("quarterly_no_result_burden", 0)),
            "cluster_population": cluster_population,
            "number_of_lsoas": int(cl.get("number_of_lsoas", len(lsoas)) or len(lsoas)),
            "number_of_wards": len(wards),
            "resident_denominator_caution": bool(cl.get("resident_denominator_caution", False)),
            "mean_actionable_fairness_score": float(cl.get("mean_actionable_fairness_score_0_100", 0) or 0),
            "expected_positive_outcomes_forgone": float(cl.get("expected_quarterly_positive_outcomes_forgone", 0) or 0),
            "crime_response_lower": float(cl.get("estimated_crime_response_lower_bound", 0) or 0),
            "crime_response_upper": float(cl.get("estimated_crime_response_upper_bound", 0) or 0),
            "primary_search_regime": cl.get("primary_search_regime", "none"),
        })
    feats = pd.DataFrame(rows).merge(protection, on="cluster_id", how="left")
    feats["protection_need_band"] = feats["protection_need_band"].fillna("Low")
    return feats


def _eligible(pid: str, r: pd.Series) -> tuple[bool, str]:
    band = r["protection_need_band"]
    critical = band == "Critical"
    if pid == "P0":
        return True, "Routine monitoring fallback (always available)."
    if not bool(r.get("has_multipath", False)):
        return False, "No combined multi-path fairness flag; monitor only."
    if pid == "P1":
        if critical:
            return False, "Blocked: Critical safety level requires protective or monitoring packages."
        if not r["low_yield_actionable"]:
            return False, "No reliable low-result search regime with meaningful volume."
        if not r["meaningful_volume"]:
            return False, "Insufficient reducible search volume for a monitored review."
        if band == "High":
            return True, "Reliable low-result search evidence; review must keep harm-focused safety monitoring."
        return True, "Reliable low-result search evidence with enough volume for a monitored review."
    if pid == "P2":
        if not (r["racial_strong"] or r["n_strong_pathways"] >= 2):
            return False, "No reliable racial disproportionality and fewer than two strong pathways."
        if not r["action_volume"]:
            return False, "Insufficient encounter volume for training/audit allocation."
        return True, ("Reliable racial disproportionality pathway." if r["racial_strong"]
                      else "Repeated fairness concern (>=2 strong pathways) involving encounter quality.")
    if pid == "P3":
        if critical:
            return False, "Blocked: Critical protection need (no community-confidence-only allocation)."
        if not (r["n_strong_pathways"] >= 1 and r["trust_warning"]):
            return False, "Requires a strong fairness pathway AND a low-trust context (trust alone is insufficient)."
        return True, "Strong fairness pathway plus low-trust borough context."
    if pid == "P4":
        if r["n_strong_pathways"] < 1:
            return False, "No strong fairness pathway to safeguard."
        if band not in ("High", "Critical"):
            return False, "Reserved for High/Critical protection need."
        return True, "Strong fairness concern with high/critical serious-harm protection need."
    if pid == "P5":
        if critical:
            return False, "Blocked: Critical safety level requires protective or monitoring packages."
        if r["n_strong_pathways"] < 2:
            return False, "Fewer than two high-confidence fairness pathways."
        if not (r["meaningful_volume"] and r["action_volume"]):
            return False, "Insufficient interaction volume for a combined high-priority package."
        return True, "Multiple high-confidence pathways with meaningful volume and manageable protection need."
    return False, "Unknown package."


def evaluate() -> pd.DataFrame:
    feats = cluster_features()
    rows = []
    for _, r in feats.iterrows():
        for pkg in PACKAGES:
            ok, reason = _eligible(pkg.id, r)
            cost = package_cost(pkg.id, r)
            includes_reduction = pkg.includes_reduction
            rows.append({
                "cluster_id": r["cluster_id"],
                "cluster_name": r["cluster_name"],
                "package_id": pkg.id,
                "package_name": pkg.name,
                "eligibility_status": "Eligible" if ok else "Not eligible",
                "eligibility_reason": reason,
                "protection_need_band": r["protection_need_band"],
                "n_strong_pathways": int(r["n_strong_pathways"]),
                "cluster_population": round(float(r["cluster_population"]), 0),
                "number_of_lsoas": int(r["number_of_lsoas"]),
                "number_of_wards": int(r["number_of_wards"]),
                **{f"cost_{k}": cost[k] for k in RESOURCE_TYPES},
                "fairness_pathway_severity_covered": round(r["mean_actionable_fairness_score"], 2) if ok else 0.0,
                "no_result_encounters_covered": round(r["quarterly_no_result_burden"], 1) if (ok and pkg.id != "P0") else 0.0,
                "racial_disproportionality_concern_covered": bool(r["racial_strong"]) and ok,
                "deprivation_burden_covered": bool(r["deprivation_strong"]) and ok,
                "low_trust_context_covered": bool(r["trust_warning"]) and ok,
                "serious_harm_protection_maintained": ok,
                "protective_presence_required": bool(ok and includes_reduction and r["protection_need_band"] == "High"),
                "interaction_volume_reached": round(r["quarterly_total_encounters"], 1),
                "expected_positive_outcomes_at_risk": round(r["expected_positive_outcomes_forgone"], 2) if (ok and includes_reduction) else 0.0,
                "estimated_crime_response_lower": r["crime_response_lower"] if (ok and includes_reduction) else 0.0,
                "estimated_crime_response_upper": r["crime_response_upper"] if (ok and includes_reduction) else 0.0,
                "evaluation_requirements": (
                    "Matched comparison cluster + difference-in-differences plan; monitor process and safety outcomes."
                    if (ok and cost["evaluation_slots"] > 0)
                    else "Routine quarterly monitoring."
                ),
            })
    return pd.DataFrame(rows)


def run() -> pd.DataFrame:
    df = evaluate()
    df.to_csv(OUT_CSV, index=False)
    elig = df[df["eligibility_status"] == "Eligible"]
    return df


if __name__ == "__main__":
    run()
