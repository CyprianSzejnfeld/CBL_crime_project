from __future__ import annotations

import glob
from collections import Counter

import numpy as np
import pandas as pd

from . import config
from .utils import read_csv_flexible

OUTPUT_PATH = (
    config.PROCESSED_DIR
    / "london_lsoa_month_stop_search_categories_2021_2025.parquet"
)






CATEGORIES = [
    "drugs",
    "offensive_weapons",
    "stolen_property",
    "other_non_weapon",
    "unknown_object",
]


def classify_object(value: object) -> str:
    if value is None:
        return "unknown_object"
    text = str(value).strip().lower()
    if text in {"", "nan", "none"}:
        return "unknown_object"
    if "drug" in text:
        return "drugs"
    if (
        "weapon" in text
        or "firearm" in text
        or "threaten or harm" in text
        or "threaten or harm anyone" in text
    ):
        return "offensive_weapons"
    if "stolen" in text:
        return "stolen_property"

    return "other_non_weapon"


def _is_arrest(outcome: str) -> bool:
    return "arrest" in outcome


def _is_positive(outcome: str) -> bool:
    text = outcome
    if text in {"", "nan"}:
        return False
    negative = (
        "no further action" in text
        or "nothing found" in text
        or text.startswith("a no further")
    )
    return not negative


def _is_no_result(outcome: str) -> bool:
    if outcome in {"", "nan"}:
        return True
    return "no further action" in outcome or "nothing found" in outcome


def _load_boundary_arrays():
    import json

    import shapely
    from shapely.geometry import shape
    from shapely.strtree import STRtree

    payload = json.loads(
        config.LONDON_LSOA_BOUNDARIES_PATH.read_text(encoding="utf-8")
    )
    geoms = []
    codes = []
    for feature in payload.get("features", []):
        geom = feature.get("geometry")
        code = (feature.get("properties") or {}).get("LSOA21CD")
        if geom and code:
            geoms.append(shape(geom))
            codes.append(code)
    if not geoms:
        raise RuntimeError(
            f"No geometries loaded from {config.LONDON_LSOA_BOUNDARIES_PATH}"
        )
    geoms_arr = np.array(geoms, dtype=object)
    tree = STRtree(geoms_arr)
    return shapely, tree, geoms_arr, np.array(codes, dtype=object)


def _assign_lsoa_vectorised(lon: pd.Series, lat: pd.Series, boundary) -> pd.Series:
    shapely, tree, geoms_arr, codes = boundary
    result = pd.Series(pd.NA, index=lon.index, dtype="object")
    lon_v = pd.to_numeric(lon, errors="coerce").to_numpy(dtype="float64")
    lat_v = pd.to_numeric(lat, errors="coerce").to_numpy(dtype="float64")
    valid = np.isfinite(lon_v) & np.isfinite(lat_v)
    if not valid.any():
        return result
    points = shapely.points(lon_v[valid], lat_v[valid])



    pairs = tree.query(points, predicate="covered_by")
    if pairs.size == 0:
        return result
    valid_positions = np.flatnonzero(valid)
    assigned_codes = np.array([pd.NA] * len(points), dtype=object)

    seen = np.zeros(len(points), dtype=bool)
    for input_pos, tree_idx in zip(pairs[0], pairs[1]):
        if not seen[input_pos]:
            assigned_codes[input_pos] = codes[tree_idx]
            seen[input_pos] = True
    result.iloc[valid_positions] = assigned_codes
    return result


def build_category_panel() -> pd.DataFrame:
    files = sorted(
        glob.glob(str(config.RAW_POLICE_DIR / "*stop-and-search.csv"))
    )
    if not files:
        raise RuntimeError(
            f"No stop-and-search files found in {config.RAW_POLICE_DIR}"
        )

    london = read_csv_flexible(config.LONDON_LSOA_LOOKUP_PATH)
    london_codes = set(london["lsoa21cd"].astype(str).str.strip())

    boundary = _load_boundary_arrays()
    qa = Counter()
    parts: list[pd.DataFrame] = []

    for path in files:
        raw = read_csv_flexible(path)
        if raw.empty:
            continue
        qa["raw_rows"] += len(raw)
        work = pd.DataFrame(index=raw.index)

        date_col = "Date" if "Date" in raw.columns else "Month"
        month = pd.to_datetime(raw.get(date_col), errors="coerce", utc=True)
        work["month"] = month.dt.to_period("M").astype(str)
        in_range = work["month"].between(config.START_MONTH, config.END_MONTH)
        work = work.loc[in_range].copy()
        raw = raw.loc[in_range]
        if work.empty:
            continue

        if "Longitude" in raw.columns and "Latitude" in raw.columns:
            work["lsoa21cd"] = _assign_lsoa_vectorised(
                raw["Longitude"], raw["Latitude"], boundary
            )
        else:
            continue
        work = work.loc[work["lsoa21cd"].isin(london_codes)].copy()
        raw = raw.loc[work.index]
        if work.empty:
            continue
        qa["assigned_london"] += len(work)

        obj_col = "Object of search"
        work["category"] = (
            raw[obj_col].map(classify_object)
            if obj_col in raw.columns
            else "unknown_object"
        )
        outcome = (
            raw["Outcome"].fillna("").astype(str).str.strip().str.lower()
            if "Outcome" in raw.columns
            else pd.Series("", index=work.index)
        )
        work["is_arrest"] = outcome.map(_is_arrest).astype(int)
        work["is_positive"] = outcome.map(_is_positive).astype(int)
        work["is_no_result"] = outcome.map(_is_no_result).astype(int)
        parts.append(work)

    if not parts:
        raise RuntimeError("No stop-and-search rows assigned to London LSOAs")

    combined = pd.concat(parts, ignore_index=True)
    combined["category"] = pd.Categorical(combined["category"], categories=CATEGORIES)


    grouped = combined.groupby(["lsoa21cd", "month", "category"], observed=False).agg(
        stops=("is_arrest", "size"),
        arrests=("is_arrest", "sum"),
        positive_outcomes=("is_positive", "sum"),
        no_result_stops=("is_no_result", "sum"),
    )


    wide = grouped.unstack("category")
    wide.columns = [f"{metric}_{cat}" for metric, cat in wide.columns]
    wide = wide.fillna(0).reset_index()


    rename = {}
    for cat in CATEGORIES:
        rename[f"stops_{cat}"] = f"stops_{cat}"
        rename[f"arrests_{cat}"] = f"arrests_{cat}"
        rename[f"positive_outcomes_{cat}"] = f"positive_outcomes_{cat}"
        rename[f"no_result_stops_{cat}"] = f"no_result_stops_{cat}"
    wide = wide.rename(columns=rename)


    stop_cols = [f"stops_{c}" for c in CATEGORIES if f"stops_{c}" in wide.columns]
    wide["stops_total_from_categories"] = wide[stop_cols].sum(axis=1)

    count_cols = [c for c in wide.columns if c not in {"lsoa21cd", "month"}]
    wide[count_cols] = wide[count_cols].astype(int)
    wide = wide.sort_values(["lsoa21cd", "month"]).reset_index(drop=True)

    return wide


def main() -> None:
    panel = build_category_panel()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
