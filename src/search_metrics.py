from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

from . import config
from .utils import ensure_dirs, percentile_0_100, safe_divide


ROLLING_MAP = {
    "stop_search_count": "rolling_stops_12m",
    "total_crime_count": "rolling_crimes_12m",
    "stop_search_arrest_count": "rolling_arrests_12m",
    "stop_search_positive_outcome_count": "rolling_positive_outcomes_12m",
    "stop_search_no_result_count": "rolling_no_result_stops_12m",
    "stops_white": "rolling_stops_white_12m",
    "stops_black": "rolling_stops_black_12m",
    "stops_asian": "rolling_stops_asian_12m",
    "stops_mixed": "rolling_stops_mixed_12m",
    "stops_other": "rolling_stops_other_12m",
    "stops_unknown_ethnicity": "rolling_stops_unknown_ethnicity_12m",
    "violent_crime_count": "rolling_violent_crimes_12m",
    "drugs_count": "rolling_drugs_12m",
    "robbery_count": "rolling_robbery_12m",
    "weapon_relevant_proxy_count": "rolling_weapon_relevant_proxy_12m",
    "burglary_count": "rolling_burglary_12m",
    "theft_count": "rolling_theft_12m",
    "vehicle_crime_count": "rolling_vehicle_crime_12m",
    "public_order_count": "rolling_public_order_12m",
    "anti_social_behaviour_count": "rolling_anti_social_behaviour_12m",
}


def add_rolling_features(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    df["month_dt"] = pd.PeriodIndex(df["month"], freq="M").to_timestamp()
    df = df.sort_values(["lsoa21cd", "month_dt"]).reset_index(drop=True)
    window = config.ROLLING_WINDOW_MONTHS
    for source, target in ROLLING_MAP.items():
        if source not in df.columns:
            df[source] = 0
        df[target] = (
            df.groupby("lsoa21cd", observed=True)[source]
            .transform(lambda s: s.fillna(0).rolling(window, min_periods=1).sum())
            .astype(float)
        )

    df["rolling_stops_known_ethnicity_12m"] = (
        df["rolling_stops_white_12m"]
        + df["rolling_stops_black_12m"]
        + df["rolling_stops_asian_12m"]
        + df["rolling_stops_mixed_12m"]
        + df["rolling_stops_other_12m"]
    )
    df["rolling_stops_non_white_12m"] = (
        df["rolling_stops_black_12m"]
        + df["rolling_stops_asian_12m"]
        + df["rolling_stops_mixed_12m"]
        + df["rolling_stops_other_12m"]
    )
    df["rolling_stop_rate_per_1000"] = safe_divide(df["rolling_stops_12m"], df["population"]) * 1000
    df["stop_rate_per_1000"] = df["rolling_stop_rate_per_1000"]
    df["rolling_crime_rate_per_1000"] = safe_divide(df["rolling_crimes_12m"], df["population"]) * 1000
    df["rolling_arrest_rate"] = safe_divide(df["rolling_arrests_12m"], df["rolling_stops_12m"])
    df["rolling_success_rate"] = safe_divide(df["rolling_positive_outcomes_12m"], df["rolling_stops_12m"])
    df["rolling_no_result_rate"] = safe_divide(df["rolling_no_result_stops_12m"], df["rolling_stops_12m"])
    return df


def add_city_averages(df: pd.DataFrame) -> pd.DataFrame:
    city = df.groupby("month", observed=True).agg(
        met_rolling_stops=("rolling_stops_12m", "sum"),
        met_rolling_crimes=("rolling_crimes_12m", "sum"),
        met_rolling_arrests=("rolling_arrests_12m", "sum"),
        met_rolling_positive=("rolling_positive_outcomes_12m", "sum"),
        met_rolling_no_result=("rolling_no_result_stops_12m", "sum"),
        met_population=("population", "sum"),
    )
    city["met_avg_stop_rate_per_1000"] = safe_divide(city["met_rolling_stops"], city["met_population"]) * 1000
    city["met_avg_crime_rate_per_1000"] = safe_divide(city["met_rolling_crimes"], city["met_population"]) * 1000
    city["met_avg_success_rate"] = safe_divide(city["met_rolling_positive"], city["met_rolling_stops"])
    city["met_avg_arrest_rate"] = safe_divide(city["met_rolling_arrests"], city["met_rolling_stops"])
    city["met_avg_no_result_rate"] = safe_divide(city["met_rolling_no_result"], city["met_rolling_stops"])
    keep = [
        "met_avg_stop_rate_per_1000",
        "met_avg_success_rate",
        "met_avg_arrest_rate",
        "met_avg_no_result_rate",
        "met_avg_crime_rate_per_1000",
    ]
    return df.merge(city[keep].reset_index(), on="month", how="left")


def _static_percentile_components(df: pd.DataFrame) -> pd.DataFrame:
    static = df.drop_duplicates("lsoa21cd").copy()

    dep_parts: list[pd.Series] = []
    for col, fallback in [
        ("imd_score", "deprivation_intensity"),
        ("income_score", "income_deprivation_intensity"),
        ("education_score", "education_deprivation_intensity"),
    ]:
        source = col if col in static.columns and static[col].notna().any() else fallback
        if source in static.columns:
            dep_parts.append(percentile_0_100(pd.to_numeric(static[source], errors="coerce")))
    static["deprivation_component"] = pd.concat(dep_parts, axis=1).mean(axis=1) if dep_parts else np.nan

    static["ethnicity_exposure_component"] = percentile_0_100(pd.to_numeric(static.get("prop_non_white"), errors="coerce"))
    static["black_exposure_component"] = percentile_0_100(pd.to_numeric(static.get("prop_black"), errors="coerce"))
    static["asian_exposure_component"] = percentile_0_100(pd.to_numeric(static.get("prop_asian"), errors="coerce"))
    static["deprivation_exposure_score_0_100"] = static[
        ["deprivation_component", "ethnicity_exposure_component"]
    ].mean(axis=1)
    return static[
        [
            "lsoa21cd",
            "deprivation_component",
            "ethnicity_exposure_component",
            "black_exposure_component",
            "asian_exposure_component",
            "deprivation_exposure_score_0_100",
        ]
    ]


def _positive_percentile_0_100(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    result = pd.Series(np.nan, index=series.index, dtype="float64")
    non_positive = numeric.notna() & numeric.le(0)
    result.loc[non_positive] = 0.0
    positive = numeric.gt(0)
    if positive.any():
        result.loc[positive] = percentile_0_100(numeric.loc[positive])
    return result


def add_percentile_components(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    static_components = _static_percentile_components(out)
    out = out.merge(static_components, on="lsoa21cd", how="left")

    out["stop_burden_ratio"] = safe_divide(out["stop_rate_per_1000"], out["met_avg_stop_rate_per_1000"])
    out["stop_burden_component"] = out.groupby("month", group_keys=False)["stop_rate_per_1000"].apply(percentile_0_100)

    out["share_stops_black"] = safe_divide(out["rolling_stops_black_12m"], out["rolling_stops_known_ethnicity_12m"])
    out["share_stops_asian"] = safe_divide(out["rolling_stops_asian_12m"], out["rolling_stops_known_ethnicity_12m"])
    out["share_stops_non_white"] = safe_divide(out["rolling_stops_non_white_12m"], out["rolling_stops_known_ethnicity_12m"])
    out["black_stop_disproportionality"] = safe_divide(out["share_stops_black"], out["prop_black"])
    out["asian_stop_disproportionality"] = safe_divide(out["share_stops_asian"], out["prop_asian"])
    out["non_white_stop_disproportionality"] = safe_divide(out["share_stops_non_white"], out["prop_non_white"])

    out.loc[out["prop_black"].lt(0.02), "black_stop_disproportionality"] = np.nan
    out.loc[out["prop_asian"].lt(0.02), "asian_stop_disproportionality"] = np.nan
    out.loc[out["prop_non_white"].lt(0.02), "non_white_stop_disproportionality"] = np.nan
    ratio_cols = ["black_stop_disproportionality", "asian_stop_disproportionality", "non_white_stop_disproportionality"]
    out[ratio_cols] = out[ratio_cols].clip(upper=5)
    out["max_capped_stop_disproportionality"] = out[ratio_cols].max(axis=1)
    out.loc[
        out["rolling_stops_known_ethnicity_12m"].lt(config.MIN_STOPS_FOR_RATE),
        "max_capped_stop_disproportionality",
    ] = np.nan
    out["racial_disproportionality_component"] = out.groupby("month", group_keys=False)[
        "max_capped_stop_disproportionality"
    ].apply(percentile_0_100)

    out["arrest_rate_gap"] = (out["met_avg_arrest_rate"] - out["rolling_arrest_rate"]).clip(lower=0)
    out["success_rate_gap"] = (out["met_avg_success_rate"] - out["rolling_success_rate"]).clip(lower=0)
    out["no_result_rate_gap"] = (out["rolling_no_result_rate"] - out["met_avg_no_result_rate"]).clip(lower=0)
    for col in ["arrest_rate_gap", "success_rate_gap", "no_result_rate_gap"]:
        out[f"{col}_component"] = out.groupby("month", group_keys=False)[col].apply(_positive_percentile_0_100)
    out["low_yield_component"] = out[
        ["arrest_rate_gap_component", "success_rate_gap_component", "no_result_rate_gap_component"]
    ].mean(axis=1)
    out.loc[out["rolling_stops_12m"].lt(config.MIN_STOPS_FOR_RATE), "low_yield_component"] = np.nan
    return out


def _fallback_expected_stops(df: pd.DataFrame) -> pd.Series:
    city = df.groupby("month", observed=True).agg(
        stops=("rolling_stops_12m", "sum"),
        crimes=("rolling_crimes_12m", "sum"),
        population=("population", "sum"),
    )
    city["stops_per_crime"] = safe_divide(city["stops"], city["crimes"]).fillna(0)
    city["stop_rate"] = safe_divide(city["stops"], city["population"]).fillna(0)
    tmp = df.merge(city[["stops_per_crime", "stop_rate"]].reset_index(), on="month", how="left")
    expected = tmp["rolling_crimes_12m"] * tmp["stops_per_crime"]
    no_crime = tmp["rolling_crimes_12m"].le(0) | expected.isna()
    expected.loc[no_crime] = tmp.loc[no_crime, "population"] * tmp.loc[no_crime, "stop_rate"]
    return expected.clip(lower=0.1)


def add_over_search_model(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    feature_cols = [
        "rolling_crimes_12m",
        "population",
        "rolling_violent_crimes_12m",
        "rolling_drugs_12m",
        "rolling_robbery_12m",
        "rolling_weapon_relevant_proxy_12m",
        "rolling_burglary_12m",
        "rolling_theft_12m",
        "rolling_vehicle_crime_12m",
        "rolling_public_order_12m",
        "rolling_anti_social_behaviour_12m",
    ]
    for col in feature_cols:
        if col not in out.columns:
            out[col] = 0
    valid = out["rolling_stops_12m"].notna() & out["population"].gt(0)
    try:
        x_num = out.loc[valid, feature_cols].fillna(0).astype(float)
        x_num = np.log1p(x_num)
        x_cat = pd.get_dummies(out.loc[valid, ["borough", "month"]].fillna("Unknown"), dtype=float)
        x_train = pd.concat([x_num.reset_index(drop=True), x_cat.reset_index(drop=True)], axis=1)
        y_train = out.loc[valid, "rolling_stops_12m"].astype(float).values
        model = PoissonRegressor(alpha=0.001, max_iter=300)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(x_train, y_train)

        all_num = np.log1p(out[feature_cols].fillna(0).astype(float))
        all_cat = pd.get_dummies(out[["borough", "month"]].fillna("Unknown"), dtype=float)
        all_cat = all_cat.reindex(columns=x_cat.columns, fill_value=0.0)
        x_all = pd.concat([all_num.reset_index(drop=True), all_cat.reset_index(drop=True)], axis=1)
        out["expected_stops"] = model.predict(x_all).clip(min=0.1)
        out["over_search_model"] = "poisson"
    except Exception as exc:
        out["expected_stops"] = _fallback_expected_stops(out)
        out["over_search_model"] = "citywide_stops_per_crime_fallback"

    out["over_search_residual"] = out["rolling_stops_12m"] - out["expected_stops"]
    out["over_search_ratio"] = safe_divide(out["rolling_stops_12m"], out["expected_stops"])
    out["positive_over_search_residual"] = out["over_search_residual"].clip(lower=0)
    out["over_search_component"] = out.groupby("month", group_keys=False)["positive_over_search_residual"].apply(
        _positive_percentile_0_100
    )
    return out


def add_component_scores(df: pd.DataFrame) -> pd.DataFrame:


    out = df.copy()
    out["over_search_score_0_100"] = out["over_search_component"]
    out["low_yield_score_0_100"] = out["low_yield_component"]
    out["racial_disproportionality_score_0_100"] = out["racial_disproportionality_component"]
    out["stop_burden_score_0_100"] = out["stop_burden_component"]
    return out


def main() -> None:
    ensure_dirs()
    panel = pd.read_parquet(config.PANEL_PARQUET_PATH)

    metrics = add_rolling_features(panel)
    metrics = add_city_averages(metrics)
    metrics = add_percentile_components(metrics)
    metrics = add_over_search_model(metrics)
    metrics = add_component_scores(metrics)
    metrics.to_csv(config.SEARCH_METRICS_PATH, index=False)


if __name__ == "__main__":
    main()
