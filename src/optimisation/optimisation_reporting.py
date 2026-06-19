from __future__ import annotations

import pandas as pd

from src.optimisation.resource_costs import cost_columns
from src.packages.library import RESOURCE_TYPES


def summarise(allocations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, g in allocations.groupby("strategy_id"):
        critical = g[g["protection_need_band"] == "Critical"]["cluster_id"].nunique()
        row = {
            "strategy_id": sid,
            "clusters_treated": int(g["cluster_id"].nunique()),
            "fairness_burden_covered": round(float(g["fairness_pathway_severity_covered"].sum()), 2),
            "no_result_burden_covered": round(float(g["no_result_encounters_covered"].sum()), 1),
            "racial_concern_covered": int(g["racial_disproportionality_concern_covered"].sum()),
            "deprivation_burden_covered": int(g["deprivation_burden_covered"].sum()),
            "protection_coverage": round(float(g["benefit_protection"].sum()), 3),
            "critical_clusters_reached": int(critical),
            "encounters_covered": round(float(g["interaction_volume_reached"].sum()), 0),
            "operational_uncertainty": round(float(g["estimated_crime_response_upper"].sum()), 2),
            "total_resource_cost": round(float(g[cost_columns()].sum().sum()), 1),
        }
        for r in RESOURCE_TYPES:
            row[f"used_{r}"] = round(float(g[f"cost_{r}"].sum()), 1)

        for pid in ["P1", "P2", "P3", "P4", "P5"]:
            row[f"n_{pid}"] = int((g["package_id"] == pid).sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("strategy_id").reset_index(drop=True)
