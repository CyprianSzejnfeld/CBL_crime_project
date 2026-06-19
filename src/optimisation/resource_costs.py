from __future__ import annotations

import pandas as pd

from src import config
from src.packages.library import RESOURCE_TYPES, quarterly_budgets

ELIGIBILITY = config.PROCESSED_DIR / "cluster_package_eligibility.csv"

BENEFIT_COMPONENTS = [
    "fairness", "no_result", "racial", "deprivation", "trust", "protection", "volume",
]

_BAND_WEIGHT = {"Low": 0.25, "Medium": 0.5, "High": 0.8, "Critical": 1.0}
_PROTECTION_FACTOR = {"P4": 1.0, "P2": 0.5, "P3": 0.5, "P0": 0.3, "P1": 0.1, "P5": 0.1}



STRATEGY_WEIGHTS: dict[str, dict[str, float]] = {
    "High-Volume Fairness Coverage":  {"fairness": 0.8, "no_result": 1.6, "racial": 0.8, "deprivation": 0.8, "trust": 0.5, "protection": 0.5, "volume": 1.5},
}


def _norm(s: pd.Series) -> pd.Series:
    m = s.max()
    return s / m if m and m > 0 else s * 0.0


def decision_table() -> pd.DataFrame:
    df = pd.read_csv(ELIGIBILITY)
    df = df[df["eligibility_status"] == "Eligible"].copy()
    df["benefit_fairness"] = _norm(df["fairness_pathway_severity_covered"])
    df["benefit_no_result"] = _norm(df["no_result_encounters_covered"])
    df["benefit_racial"] = _norm(
        df["racial_disproportionality_concern_covered"].astype(float) * df["fairness_pathway_severity_covered"]
    )
    df["benefit_deprivation"] = _norm(
        df["deprivation_burden_covered"].astype(float) * df["fairness_pathway_severity_covered"]
    )
    df["benefit_trust"] = _norm(
        df["low_trust_context_covered"].astype(float) * df["fairness_pathway_severity_covered"]
    )
    df["benefit_protection"] = (
        df["protection_need_band"].map(_BAND_WEIGHT).fillna(0.25)
        * df["package_id"].map(_PROTECTION_FACTOR).fillna(0.1)
    )
    df["benefit_volume"] = _norm(df["interaction_volume_reached"])
    return df


def strategy_benefit(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    total = pd.Series(0.0, index=df.index)
    for comp in BENEFIT_COMPONENTS:
        total = total + weights.get(comp, 0.0) * df[f"benefit_{comp}"]
    return total


def budgets() -> dict[str, float]:
    return quarterly_budgets()


def cost_columns() -> list[str]:
    return [f"cost_{r}" for r in RESOURCE_TYPES]
