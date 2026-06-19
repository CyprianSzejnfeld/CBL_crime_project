from __future__ import annotations

import pandas as pd

from src import config
from src.optimisation import optimisation_reporting, weighted_ilp
from src.optimisation.resource_costs import STRATEGY_WEIGHTS, decision_table

SCEN_DIR = config.SCENARIOS_DIR
ALLOC_COLS = [
    "strategy_id", "cluster_id", "cluster_name", "package_id", "package_name",
    "protection_need_band", "eligibility_reason", "evaluation_requirements",
    "fairness_pathway_severity_covered", "no_result_encounters_covered",
    "racial_disproportionality_concern_covered", "deprivation_burden_covered",
    "low_trust_context_covered", "benefit_protection", "interaction_volume_reached",
    "cluster_population", "number_of_lsoas", "number_of_wards",
    "estimated_crime_response_upper",
]


def _alloc_view(df: pd.DataFrame) -> pd.DataFrame:
    cost_cols = [c for c in df.columns if c.startswith("cost_")]
    cols = [c for c in ALLOC_COLS if c in df.columns] + cost_cols
    return df[cols]


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    SCEN_DIR.mkdir(parents=True, exist_ok=True)
    table = decision_table()
    table = table[table["package_id"] != "P0"].copy()

    allocations = [
        weighted_ilp.solve(table, weights, name) for name, weights in STRATEGY_WEIGHTS.items()
    ]
    all_alloc = pd.concat([a for a in allocations if not a.empty], ignore_index=True)

    summary = optimisation_reporting.summarise(all_alloc)

    summary.to_csv(SCEN_DIR / "intervention_package_all_strategies.csv", index=False)
    _alloc_view(all_alloc).to_csv(SCEN_DIR / "intervention_package_all_allocations.csv", index=False)

    return summary, all_alloc


if __name__ == "__main__":
    run()
