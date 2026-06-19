from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


CRIME_SOURCE = config.CRIME_TARGET_SOURCE


TRUST_KEEP = [
    "borough",
    "trust_context_score_0_100",
    "borough_low_trust_flag",
    "trust_context_band",
    "trust_context_reliability",
    "suggested_low_trust_warning_text",
]






def load_search_metrics() -> pd.DataFrame:
    path = config.SEARCH_METRICS_PATH
    if not path.exists():
        raise FileNotFoundError(f"Search metrics not found at {path}")
    df = pd.read_csv(path, low_memory=False)
    df["lsoa21cd"] = df["lsoa21cd"].astype(str).str.strip()
    df["month"] = df["month"].astype(str).str[:7]
    return df


def load_trust_context() -> pd.DataFrame:
    path = config.MOPAC_TRUST_CONTEXT_PARQUET_PATH
    if not path.exists():
        return pd.DataFrame(columns=TRUST_KEEP)
    df = pd.read_parquet(path)
    keep = [c for c in TRUST_KEEP if c in df.columns]
    df = df[keep].drop_duplicates("borough").copy()
    if "borough_low_trust_flag" in df.columns:
        df["borough_low_trust_flag"] = df["borough_low_trust_flag"].astype(bool)
    return df


def load_stop_search_categories() -> pd.DataFrame | None:
    path = config.STOP_SEARCH_CATEGORIES_PATH
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["lsoa21cd"] = df["lsoa21cd"].astype(str).str.strip()
    df["month"] = df["month"].astype(str).str[:7]
    return df



def standardise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()


    for canonical, source in CRIME_SOURCE.items():
        if source in df.columns and canonical not in df.columns:
            df[canonical] = df[source]


    if "stops_total" not in df.columns and "stop_search_count" in df.columns:
        df["stops_total"] = df["stop_search_count"]
    if "no_result_stops_total" not in df.columns and "stop_search_no_result_count" in df.columns:
        df["no_result_stops_total"] = df["stop_search_no_result_count"]
    if "arrests_total" not in df.columns and "stop_search_arrest_count" in df.columns:
        df["arrests_total"] = df["stop_search_arrest_count"]
    if "positive_outcomes_total" not in df.columns and "stop_search_positive_outcome_count" in df.columns:
        df["positive_outcomes_total"] = df["stop_search_positive_outcome_count"]



    if "excess_stop_share" not in df.columns:
        share = pd.Series(np.nan, index=df.index, dtype="float64")
        if "over_search_ratio" in df.columns:
            ratio = pd.to_numeric(df["over_search_ratio"], errors="coerce")
            share = np.where(ratio > 1, 1.0 - 1.0 / ratio, 0.0)
            share = pd.Series(share, index=df.index)
        elif {"positive_over_search_residual", "rolling_stops_12m"} <= set(df.columns):
            resid = pd.to_numeric(df["positive_over_search_residual"], errors="coerce")
            stops = pd.to_numeric(df["rolling_stops_12m"], errors="coerce")
            share = (resid / stops).clip(lower=0, upper=1)
        df["excess_stop_share"] = pd.Series(share, index=df.index).clip(0, 1).fillna(0.0)


    if "data_reliability" not in df.columns:
        stops12 = pd.to_numeric(df.get("rolling_stops_12m", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
        rel = np.where(
            stops12 < config.MIN_ANNUAL_STOPS_FOR_REDUCTION,
            "low",
            np.where(stops12 < 60, "medium", "high"),
        )
        df["data_reliability"] = rel

    return df


def merge_existing_trust_context_if_needed(panel: pd.DataFrame, trust: pd.DataFrame) -> pd.DataFrame:
    if trust.empty or "borough" not in panel.columns:
        panel["borough_low_trust_flag"] = panel.get("borough_low_trust_flag", False)
        return panel
    return panel.merge(trust, on="borough", how="left", suffixes=("", "_trust"))


def merge_stop_search_categories(panel: pd.DataFrame, cats: pd.DataFrame | None) -> tuple[pd.DataFrame, bool]:
    if cats is None:
        return panel, False
    cat_cols = [c for c in cats.columns if c not in {"lsoa21cd", "month"}]
    merged = panel.merge(cats, on=["lsoa21cd", "month"], how="left")
    merged[cat_cols] = merged[cat_cols].fillna(0)
    return merged, True





def validate_lsoa_month_panel(df: pd.DataFrame) -> dict:
    issues = []
    dup = df.duplicated(["lsoa21cd", "month"]).sum()
    if dup:
        issues.append(f"{dup} duplicate (lsoa21cd, month) rows")
    n_lsoa = df["lsoa21cd"].nunique()
    n_month = df["month"].nunique()
    if df["rolling_stops_12m"].isna().all():
        issues.append("rolling_stops_12m entirely missing")
    summary = {
        "rows": int(len(df)),
        "n_lsoa": int(n_lsoa),
        "n_month": int(n_month),
        "months": [df["month"].min(), df["month"].max()],
        "duplicate_rows": int(dup),
        "issues": issues,
    }
    return summary





def build_modelling_panel() -> tuple[pd.DataFrame, dict]:
    metrics = load_search_metrics()
    trust = load_trust_context()
    cats = load_stop_search_categories()

    panel = standardise_column_names(metrics)
    panel = merge_existing_trust_context_if_needed(panel, trust)
    panel, category_available = merge_stop_search_categories(panel, cats)

    if "borough_low_trust_flag" in panel.columns:
        panel["borough_low_trust_flag"] = panel["borough_low_trust_flag"].fillna(False).astype(bool)

    panel = panel.sort_values(["lsoa21cd", "month"]).reset_index(drop=True)
    summary = validate_lsoa_month_panel(panel)
    summary["category_data_available"] = category_available
    summary["weapons_target_is_proxy"] = True
    return panel, summary


def save_modelling_panel(panel: pd.DataFrame) -> None:
    config.MODELLING_PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(config.MODELLING_PANEL_PATH, index=False)


def load_modelling_panel() -> pd.DataFrame:
    if not config.MODELLING_PANEL_PATH.exists():
        panel, _ = build_modelling_panel()
        save_modelling_panel(panel)
        return panel
    return pd.read_parquet(config.MODELLING_PANEL_PATH)


def main() -> None:
    panel, _ = build_modelling_panel()
    save_modelling_panel(panel)


if __name__ == "__main__":
    main()
