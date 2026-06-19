from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .data_interfaces import load_modelling_panel


CRIME_COLS = list(config.FORECAST_TARGETS)
LO_O_BOROUGH_CRIMES = [
    "total_crime_count",
    "violence_count",
    "drugs_count",
    "robbery_count",
    config.HARM_WEIGHTED_SERIOUS_TARGET,
]
LAGS = [1, 2, 3, 6, 12]
ROLL_WINDOWS = [3, 6, 12]


def _add_lag_roll(df: pd.DataFrame, base_cols: list[str]) -> pd.DataFrame:

    g = df.groupby("lsoa21cd", sort=False)
    new = {}
    for col in base_cols:
        if col not in df.columns:
            continue
        s = df[col]
        gcol = g[col]
        for lag in LAGS:
            new[f"{col}_lag_{lag}"] = gcol.shift(lag)
        for w in ROLL_WINDOWS:


            roll = gcol.transform(lambda x: x.rolling(w, min_periods=max(2, w // 2)).mean())
            new[f"{col}_rolling_{w}m_mean"] = roll
            new[f"{col}_rolling_{w}m_sum"] = gcol.transform(
                lambda x: x.rolling(w, min_periods=max(2, w // 2)).sum()
            )
        r3 = new[f"{col}_rolling_3m_mean"]
        r12 = new[f"{col}_rolling_12m_mean"]
        new[f"{col}_trend_3m_vs_12m"] = r3 / r12.replace({0: np.nan})
        new[f"{col}_change_1m"] = s - new[f"{col}_lag_1"]
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df["month"] + "-01")
    df["year"] = dt.dt.year
    df["month_number"] = dt.dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month_number"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_number"] / 12)
    start = pd.to_datetime(config.START_MONTH + "-01")
    df["months_since_start"] = (dt.dt.year - start.year) * 12 + (dt.dt.month - start.month)

    df["covid_recovery_flag"] = (df["month"] <= "2021-12").astype(int)
    return df


def _add_borough_loo(df: pd.DataFrame) -> pd.DataFrame:

    new = {}
    for crime in LO_O_BOROUGH_CRIMES:
        for w in (3, 12):
            base = f"{crime}_rolling_{w}m_mean"
            if base not in df.columns:
                continue
            grp = df.groupby(["borough", "month"])[base]
            bsum = grp.transform("sum")
            bcnt = grp.transform("count")
            loo = (bsum - df[base].fillna(0)) / (bcnt - 1).replace({0: np.nan})
            new[f"borough_other_lsoas_{crime}_rolling_{w}m_mean"] = loo
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def _add_stop_search_benchmark_features(df: pd.DataFrame) -> pd.DataFrame:

    g = df.groupby("lsoa21cd", sort=False)
    new = {}
    if "stops_total" in df.columns:
        new["stops_total_lag_1"] = g["stops_total"].shift(1)
        new["stops_total_rolling_3m_mean"] = g["stops_total"].transform(
            lambda x: x.rolling(3, min_periods=2).mean()
        )
        new["stops_total_rolling_12m_mean"] = g["stops_total"].transform(
            lambda x: x.rolling(12, min_periods=6).mean()
        )
    for rate, out in [
        ("stop_search_success_rate", "historical_success_rate_rolling_12m"),
        ("stop_search_arrest_rate", "historical_arrest_rate_rolling_12m"),
        ("stop_search_no_result_rate", "historical_no_result_rate_rolling_12m"),
    ]:
        if rate in df.columns:
            new[out] = g[rate].transform(lambda x: x.rolling(12, min_periods=6).mean())
    for cat in ["drugs", "offensive_weapons", "other_non_weapon"]:
        col = f"stops_{cat}"
        if col in df.columns:
            new[f"{col}_rolling_12m_mean"] = g[col].transform(
                lambda x: x.rolling(12, min_periods=6).mean()
            )
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def _add_targets(df: pd.DataFrame) -> pd.DataFrame:

    g = df.groupby("lsoa21cd", sort=False)
    for col in CRIME_COLS:
        if col in df.columns:
            df[f"target_{col}_next_month"] = g[col].shift(-1)
    return df


def add_harm_weighted_serious_crime_score(df: pd.DataFrame) -> pd.DataFrame:


    target = config.HARM_WEIGHTED_SERIOUS_TARGET
    if target in df.columns:
        return df
    weights = config.HARM_WEIGHTED_SERIOUS_CRIME_WEIGHTS
    violence = pd.to_numeric(df.get("violence_count", 0), errors="coerce").fillna(0)
    robbery = pd.to_numeric(df.get("robbery_count", 0), errors="coerce").fillna(0)
    weapons_proxy = pd.to_numeric(df.get("possession_of_weapons_count", 0), errors="coerce").fillna(0)
    weapons_exclusive = (weapons_proxy - robbery).clip(lower=0)
    df[target] = (
        weights["violence_count"] * violence
        + weights["robbery_count"] * robbery
        + weights["weapons_exclusive_proxy_count"] * weapons_exclusive
    )
    return df


def feature_columns(df: pd.DataFrame) -> dict[str, list[str]]:

    context = [
        "population",
        "imd_score",
        "imd_decile",
        "income_score",
        "income_decile",
        "education_score",
        "education_decile",
        "employment_score",
        "employment_decile",
    ]
    time_feats = [
        "year",
        "month_number",
        "month_sin",
        "month_cos",
        "months_since_start",
        "covid_recovery_flag",
    ]
    crime_derived = [
        c
        for c in df.columns
        if any(c.startswith(f"{base}_") for base in CRIME_COLS)
        and not c.startswith("target_")
    ]
    borough_loo = [c for c in df.columns if c.startswith("borough_other_lsoas_")]
    policy_independent = [
        c for c in (crime_derived + borough_loo + context + time_feats) if c in df.columns
    ]

    stop_feats = [
        c
        for c in df.columns
        if c.startswith("stops_total")
        or c.startswith("historical_")
        or (c.startswith("stops_") and c.endswith("_rolling_12m_mean"))
    ]
    with_stop_search = policy_independent + [c for c in stop_feats if c in df.columns]
    return {
        "policy_independent": policy_independent,
        "with_stop_search": sorted(set(with_stop_search)),
    }


def build_features() -> pd.DataFrame:
    panel = load_modelling_panel()
    panel = panel.sort_values(["lsoa21cd", "month"]).reset_index(drop=True)
    panel = add_harm_weighted_serious_crime_score(panel)
    panel = _add_lag_roll(panel, CRIME_COLS)
    panel = _add_time_features(panel)
    panel = _add_borough_loo(panel)
    panel = _add_stop_search_benchmark_features(panel)
    panel = _add_targets(panel)
    return panel


def main() -> None:
    feats = build_features()
    config.CRIME_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(config.CRIME_FEATURES_PATH, index=False)
    cols = feature_columns(feats)
    targets = [c for c in feats.columns if c.startswith("target_")]


if __name__ == "__main__":
    main()
