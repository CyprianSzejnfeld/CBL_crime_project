from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .smoothing import band_from_score, reliability_from_volume


def add_deprivation_burden_pathway(latest: pd.DataFrame) -> pd.DataFrame:
    out = latest.copy()
    out["deprivation_burden_score_0_100"] = (
        0.35 * out["deprivation_percentile"].fillna(0)
        + 0.30 * out["stop_burden_percentile"].fillna(0)
        + 0.20 * out["no_result_burden_percentile"].fillna(0)
        + 0.15 * out["over_search_score_0_100"].fillna(0)
    ).clip(0, 100)
    out["deprivation_burden_band"] = band_from_score(out["deprivation_burden_score_0_100"])
    out["deprivation_burden_reliability"] = reliability_from_volume(
        out["rolling_stops_12m"], config.MIN_LSOA_DISPLAY_STOPS_12M, config.MIN_LSOA_STRONG_STOPS_12M
    )
    highly_deprived = out["deprivation_percentile"].ge(80)
    substantial = out.get("substantial_oversearch_flag", pd.Series(False, index=out.index)).fillna(False)
    strong = highly_deprived & substantial
    out["deprivation_oversearch_path_flag"] = strong
    out["deprivation_burden_flag"] = strong
    moderate = (
        highly_deprived
        & out["deprivation_burden_score_0_100"].ge(65)
        & out["rolling_stops_12m"].ge(config.MIN_LSOA_DISPLAY_STOPS_12M)
    )
    out["deprivation_burden_moderate_flag"] = moderate & ~strong
    out["deprivation_burden_reason"] = np.select(
        [
            out["deprivation_burden_flag"],
            out["deprivation_burden_moderate_flag"],
            highly_deprived,
        ],
        [
            "High deprivation plus a search rate above the London-normal LSOA range.",
            "Elevated stop-search burden in a highly deprived area; monitor or aggregate.",
            "High deprivation context, but stop/no-result burden trigger not met.",
        ],
        default="No deprivation-concentrated burden signal.",
    )
    return out
