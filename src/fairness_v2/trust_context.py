from __future__ import annotations

import numpy as np
import pandas as pd


def add_trust_context_warning(latest: pd.DataFrame) -> pd.DataFrame:
    out = latest.copy()
    high_confidence_pathway = out.get("any_fairness_pathway_flag", False)
    low_trust = out.get("borough_low_trust_flag", False)
    if not isinstance(high_confidence_pathway, pd.Series):
        high_confidence_pathway = pd.Series(bool(high_confidence_pathway), index=out.index)
    out["trust_context_warning_flag"] = high_confidence_pathway.fillna(False).astype(bool) & low_trust.fillna(False).astype(bool)
    out["trust_context_warning_text"] = np.where(
        out["trust_context_warning_flag"],
        "High fairness-risk pattern located in a low-trust borough context; prioritise human review.",
        "",
    )
    return out
