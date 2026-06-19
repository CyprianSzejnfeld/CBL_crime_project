from __future__ import annotations

import pandas as pd
import pulp

from src.optimisation.resource_costs import (
    budgets,
    strategy_benefit,
)
from src.packages.library import RESOURCE_TYPES


def solve(
    table: pd.DataFrame,
    weights: dict,
    strategy_id: str,
    budget_overrides: dict | None = None,
) -> pd.DataFrame:

    df = table.reset_index(drop=True).copy()
    df["benefit"] = strategy_benefit(df, weights)
    budget = budgets().copy()
    if budget_overrides:
        budget.update(budget_overrides)

    prob = pulp.LpProblem(f"alloc_{strategy_id}".replace(" ", "_"), pulp.LpMaximize)
    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in df.index}
    prob += pulp.lpSum(df.loc[i, "benefit"] * x[i] for i in df.index)


    for _, idx in df.groupby("cluster_id").groups.items():
        prob += pulp.lpSum(x[i] for i in idx) <= 1


    for r in RESOURCE_TYPES:
        col = f"cost_{r}"
        prob += pulp.lpSum(df.loc[i, col] * x[i] for i in df.index) <= budget[r]


    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob.status] != "Optimal":
        return df.iloc[0:0].assign(strategy_id=strategy_id)
    chosen = [i for i in df.index if x[i].value() is not None and x[i].value() > 0.5]
    out = df.loc[chosen].copy()
    out.insert(0, "strategy_id", strategy_id)
    return out
