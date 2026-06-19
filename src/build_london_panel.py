from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from . import config
from .utils import (
    ensure_dirs,
    find_column,
    first_existing_column,
    log_dropped,
    month_range,
    numeric,
    read_csv_flexible,
    require_columns,
    safe_divide,
    write_json,
)


COUNT_COLUMNS = [
    "total_crime_count",
    "violent_crime_count",
    "drugs_count",
    "robbery_count",
    "weapon_relevant_proxy_count",
    "burglary_count",
    "theft_count",
    "vehicle_crime_count",
    "public_order_count",
    "anti_social_behaviour_count",
    "stop_search_count",
    "stop_search_arrest_count",
    "stop_search_positive_outcome_count",
    "stop_search_no_result_count",
    "stop_search_no_further_action_count",
    "stops_white",
    "stops_black",
    "stops_asian",
    "stops_mixed",
    "stops_other",
    "stops_unknown_ethnicity",
    "arrest_stops_white",
    "arrest_stops_black",
    "arrest_stops_asian",
    "arrest_stops_mixed",
    "arrest_stops_other",
    "positive_stops_white",
    "positive_stops_black",
    "positive_stops_asian",
    "positive_stops_mixed",
    "positive_stops_other",
]

LONDON_BOROUGH_LAD24_CODES = {
    "Barking and Dagenham": "E09000002",
    "Barnet": "E09000003",
    "Bexley": "E09000004",
    "Brent": "E09000005",
    "Bromley": "E09000006",
    "Camden": "E09000007",
    "Croydon": "E09000008",
    "Ealing": "E09000009",
    "Enfield": "E09000010",
    "Greenwich": "E09000011",
    "Hackney": "E09000012",
    "Hammersmith and Fulham": "E09000013",
    "Haringey": "E09000014",
    "Harrow": "E09000015",
    "Havering": "E09000016",
    "Hillingdon": "E09000017",
    "Hounslow": "E09000018",
    "Islington": "E09000019",
    "Kensington and Chelsea": "E09000020",
    "Kingston upon Thames": "E09000021",
    "Lambeth": "E09000022",
    "Lewisham": "E09000023",
    "Merton": "E09000024",
    "Newham": "E09000025",
    "Redbridge": "E09000026",
    "Richmond upon Thames": "E09000027",
    "Southwark": "E09000028",
    "Sutton": "E09000029",
    "Tower Hamlets": "E09000030",
    "Waltham Forest": "E09000031",
    "Wandsworth": "E09000032",
    "Westminster": "E09000033",
}


def _london_lsoa_lookup_from_boundaries() -> pd.DataFrame:
    payload = json.loads(config.LONDON_LSOA_BOUNDARIES_PATH.read_text(encoding="utf-8"))
    rows: list[dict[str, str]] = []
    missing_boroughs: set[str] = set()

    for feature in payload.get("features", []):
        props = feature.get("properties", {})
        lsoa_code = str(props.get("LSOA21CD") or "").strip()
        lsoa_name = str(props.get("LSOA21NM") or "").strip()
        borough = re.sub(r"\s+\d+[A-Z]$", "", lsoa_name).strip()
        lad_code = LONDON_BOROUGH_LAD24_CODES.get(borough)
        if not lsoa_code or not lsoa_name:
            continue
        if not lad_code:
            missing_boroughs.add(borough or "<blank>")
            continue
        rows.append(
            {
                "lsoa21cd": lsoa_code,
                "lsoa21nm": lsoa_name,
                "lad24cd": lad_code,
                "borough": borough,
            }
        )

    if missing_boroughs:
        raise RuntimeError(
            "Could not infer borough LAD codes from boundary names: "
            + ", ".join(sorted(missing_boroughs))
        )
    if not rows:
        raise RuntimeError(f"No London LSOA features found in {config.LONDON_LSOA_BOUNDARIES_PATH}")

    london = pd.DataFrame(rows).drop_duplicates("lsoa21cd").sort_values("lsoa21cd").reset_index(drop=True)
    london.to_csv(config.LONDON_LSOA_LOOKUP_PATH, index=False)
    return london


def build_london_lsoa_universe() -> pd.DataFrame:
    if config.LONDON_LSOA_LOOKUP_PATH.exists():
        london = read_csv_flexible(config.LONDON_LSOA_LOOKUP_PATH)
    elif config.LSOA_LAD_LOOKUP_PATH.exists():
        lookup = read_csv_flexible(config.LSOA_LAD_LOOKUP_PATH)
        lsoa_col = first_existing_column(lookup.columns, ["LSOA21CD"])
        lsoa_name_col = first_existing_column(lookup.columns, ["LSOA21NM"])
        lad_col = first_existing_column(lookup.columns, ["LAD24CD", "LAD23CD", "LAD22CD"])
        lad_name_col = first_existing_column(lookup.columns, ["LAD24NM", "LAD23NM", "LAD22NM"])
        if not lsoa_col or not lad_col or not lad_name_col:
            raise RuntimeError(f"Could not detect LSOA/LAD columns in {config.LSOA_LAD_LOOKUP_PATH}")
        london = lookup.loc[lookup[lad_col].astype(str).str.startswith(config.LONDON_LAD_CODE_PREFIX)].copy()
        if not config.INCLUDE_CITY_OF_LONDON:
            london = london.loc[london[lad_col].ne(config.CITY_OF_LONDON_LAD_CODE)]
        rename = {lsoa_col: "lsoa21cd", lad_col: "lad24cd", lad_name_col: "borough"}
        if lsoa_name_col:
            rename[lsoa_name_col] = "lsoa21nm"
        london = london[list(rename)].rename(columns=rename)
        london = london.drop_duplicates("lsoa21cd").sort_values("lsoa21cd").reset_index(drop=True)
        london.to_csv(config.LONDON_LSOA_LOOKUP_PATH, index=False)
    else:
        london = _london_lsoa_lookup_from_boundaries()
    require_columns(london, ["lsoa21cd", "lad24cd", "borough"], config.LONDON_LSOA_LOOKUP_PATH)
    return london


def _police_files(kind: str) -> list[Path]:
    months = {str(month) for month in month_range()}
    files: list[Path] = []
    for path in sorted(config.RAW_POLICE_DIR.glob(f"*-{kind}.csv")):
        name = path.name.lower()
        month = name[:7]
        if month not in months:
            continue
        if any(f"-{slug}-{kind}.csv" in name for slug in config.POLICE_FORCE_SLUGS):
            files.append(path)
    return files


def _read_police_chunks(path: Path, chunksize: int = 200_000):
    yield from pd.read_csv(path, dtype=str, chunksize=chunksize, low_memory=False, encoding_errors="replace")


def process_crime_data(london_codes: set[str]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    dropped = Counter()
    files = _police_files("street")
    if not files:
        raise RuntimeError(f"No street crime files found in {config.RAW_POLICE_DIR}")

    for path in tqdm(files, desc="crime files", unit="file"):
        for chunk in _read_police_chunks(path):
            month_col = first_existing_column(chunk.columns, ["Month"])
            lsoa_col = first_existing_column(chunk.columns, ["LSOA code", "LSOA21CD"])
            crime_col = first_existing_column(chunk.columns, ["Crime type"])
            if not month_col or not lsoa_col or not crime_col:
                raise RuntimeError(f"{path} missing Month/LSOA code/Crime type")
            work = chunk[[month_col, lsoa_col, crime_col]].copy()
            work.columns = ["month", "lsoa21cd", "crime_type"]
            work["month"] = work["month"].astype(str).str[:7]
            work = work.loc[work["month"].between(config.START_MONTH, config.END_MONTH)]
            before = len(work)
            work = work.loc[work["lsoa21cd"].isin(london_codes)]
            dropped["crime_not_london_or_missing_lsoa"] += before - len(work)
            if work.empty:
                continue
            work["crime_type_norm"] = work["crime_type"].astype(str).str.strip().str.lower()

            base = work.groupby(["lsoa21cd", "month"], observed=True).size().rename("total_crime_count")
            grouped = base.to_frame()
            for out_col, labels in config.CRIME_CATEGORY_COLUMNS.items():
                mask = work["crime_type_norm"].isin(labels)
                counts = work.loc[mask].groupby(["lsoa21cd", "month"], observed=True).size()
                grouped[out_col] = counts
            parts.append(grouped.reset_index())

    if not parts:
        return pd.DataFrame(columns=["lsoa21cd", "month"] + list(config.CRIME_CATEGORY_COLUMNS) + ["total_crime_count"])
    out = pd.concat(parts, ignore_index=True)
    numeric_cols = [col for col in out.columns if col not in {"lsoa21cd", "month"}]
    out[numeric_cols] = out[numeric_cols].fillna(0)
    out = out.groupby(["lsoa21cd", "month"], as_index=False)[numeric_cols].sum()
    log_dropped(dict(dropped))
    return out


def _positive_outcome(outcome: object) -> bool:
    text = str(outcome or "").strip().lower()
    if not text or text in {"nan", "none", "null"}:
        return False
    positives = [
        "arrest",
        "caution",
        "community resolution",
        "summons",
        "charged",
        "postal requisition",
        "penalty notice",
        "cannabis warning",
        "khat warning",
    ]
    return any(token in text for token in positives)


def _arrest_outcome(outcome: object) -> bool:
    return "arrest" in str(outcome or "").strip().lower()


def _no_result_outcome(outcome: object) -> bool:
    text = str(outcome or "").strip().lower()
    if not text or text in {"nan", "none", "null"}:
        return False
    no_result = ["no further action", "nothing found", "no action", "no outcome"]
    return any(token in text for token in no_result)


def _ethnicity_group(value: object) -> str:
    text = str(value or "").strip().lower()
    if not text or text in {"nan", "none", "null", "not known", "not stated", "unknown", "other ethnic group - not stated"}:
        return "unknown"
    if "white" in text:
        return "white"
    if "black" in text:
        return "black"
    if "asian" in text or "chinese" in text:
        return "asian"
    if "mixed" in text or "multiple" in text:
        return "mixed"
    if "other" in text or "arab" in text:
        return "other"
    return "unknown"


def _load_boundary_index():
    from shapely.geometry import Point, shape
    from shapely.strtree import STRtree

    payload = json.loads(config.LONDON_LSOA_BOUNDARIES_PATH.read_text(encoding="utf-8"))
    geometries = []
    codes = []
    for feature in payload.get("features", []):
        geom = shape(feature.get("geometry"))
        code = feature.get("properties", {}).get("LSOA21CD")
        if geom and code:
            geometries.append(geom)
            codes.append(code)
    if not geometries:
        raise RuntimeError(f"No geometries loaded from {config.LONDON_LSOA_BOUNDARIES_PATH}")
    tree = STRtree(geometries)
    return Point, tree, geometries, codes


def _spatial_assign_lsoa(lon: pd.Series, lat: pd.Series, boundary_index) -> pd.Series:
    Point, tree, geometries, codes = boundary_index
    assigned: list[str | None] = []
    for x, y in zip(lon, lat):
        if pd.isna(x) or pd.isna(y):
            assigned.append(None)
            continue
        point = Point(float(x), float(y))
        code = None
        for idx in tree.query(point):
            i = int(idx)
            if geometries[i].covers(point):
                code = codes[i]
                break
        assigned.append(code)
    return pd.Series(assigned, index=lon.index, dtype="object")


def process_stop_search_data(london_codes: set[str]) -> pd.DataFrame:
    files = _police_files("stop-and-search")
    if not files:
        raise RuntimeError(f"No stop-and-search files found in {config.RAW_POLICE_DIR}")

    parts: list[pd.DataFrame] = []
    dropped = Counter()
    outcome_counter = Counter()
    qa = Counter()
    boundary_index = None

    for path in tqdm(files, desc="stop-search files", unit="file"):
        for chunk in _read_police_chunks(path, chunksize=100_000):
            qa["raw_stop_search_rows"] += len(chunk)
            date_col = first_existing_column(chunk.columns, ["Date", "Month"])
            lsoa_col = first_existing_column(chunk.columns, ["LSOA code", "LSOA21CD"])
            lon_col = first_existing_column(chunk.columns, ["Longitude"])
            lat_col = first_existing_column(chunk.columns, ["Latitude"])
            outcome_col = first_existing_column(chunk.columns, ["Outcome"])
            officer_col = first_existing_column(chunk.columns, ["Officer-defined ethnicity"])
            self_col = first_existing_column(chunk.columns, ["Self-defined ethnicity"])

            if not date_col:
                raise RuntimeError(f"{path} missing Date/Month")
            work = pd.DataFrame()
            work["month"] = pd.to_datetime(chunk[date_col], errors="coerce", utc=True).dt.to_period("M").astype(str)
            if work["month"].eq("NaT").all():
                work["month"] = chunk[date_col].astype(str).str[:7]
            valid_month = work["month"].between(config.START_MONTH, config.END_MONTH)
            work = work.loc[valid_month].copy()
            chunk = chunk.loc[valid_month].copy()
            if work.empty:
                continue

            if lsoa_col:
                work["lsoa21cd"] = chunk[lsoa_col].astype(str).str.strip()
            else:
                work["lsoa21cd"] = pd.NA

            if lon_col and lat_col:
                lon = numeric(chunk[lon_col])
                lat = numeric(chunk[lat_col])
                has_coords = lon.notna() & lat.notna()
                qa["rows_with_coordinates"] += int(has_coords.sum())
                need_spatial = ~work["lsoa21cd"].isin(london_codes) & has_coords
                if need_spatial.any():
                    if boundary_index is None:
                        boundary_index = _load_boundary_index()
                    work.loc[need_spatial, "lsoa21cd"] = _spatial_assign_lsoa(
                        lon.loc[need_spatial], lat.loc[need_spatial], boundary_index
                    )
            else:
                dropped["stop_search_missing_coordinate_columns"] += len(work)

            before_london = len(work)
            work = work.loc[work["lsoa21cd"].isin(london_codes)].copy()
            dropped["stop_search_not_assigned_london_lsoa"] += before_london - len(work)
            qa["rows_assigned_london_lsoa"] += len(work)
            if work.empty:
                continue

            if outcome_col:
                outcome = chunk.loc[work.index, outcome_col].fillna("")
            else:
                outcome = pd.Series("", index=work.index)
            outcome_counter.update(outcome.astype(str).str.strip().replace({"": "(missing)"}).tolist())
            work["is_arrest"] = outcome.map(_arrest_outcome).astype(int)
            work["is_positive"] = outcome.map(_positive_outcome).astype(int)
            work["is_no_result"] = outcome.map(_no_result_outcome).astype(int)
            work["is_no_further_action"] = work["is_no_result"]

            if officer_col:
                officer = chunk.loc[work.index, officer_col]
                qa["rows_with_officer_ethnicity"] += int(officer.notna().sum() - officer.astype(str).str.strip().isin(["", "nan"]).sum())
            else:
                officer = pd.Series(pd.NA, index=work.index)
            if self_col:
                self_defined = chunk.loc[work.index, self_col]
                qa["rows_with_self_ethnicity"] += int(
                    self_defined.notna().sum() - self_defined.astype(str).str.strip().isin(["", "nan"]).sum()
                )
            else:
                self_defined = pd.Series(pd.NA, index=work.index)

            primary_ethnicity = officer.where(officer.astype(str).str.strip().ne("") & officer.notna(), self_defined)
            work["ethnicity_group"] = primary_ethnicity.map(_ethnicity_group)

            grouped = work.groupby(["lsoa21cd", "month"], observed=True).agg(
                stop_search_count=("lsoa21cd", "size"),
                stop_search_arrest_count=("is_arrest", "sum"),
                stop_search_positive_outcome_count=("is_positive", "sum"),
                stop_search_no_result_count=("is_no_result", "sum"),
                stop_search_no_further_action_count=("is_no_further_action", "sum"),
            )

            eth = work.pivot_table(
                index=["lsoa21cd", "month"],
                columns="ethnicity_group",
                values="is_positive",
                aggfunc="size",
                fill_value=0,
            )
            for group in ["white", "black", "asian", "mixed", "other", "unknown"]:
                grouped[f"stops_{group if group != 'unknown' else 'unknown_ethnicity'}"] = eth.get(group, 0)

            for metric, prefix in [("is_arrest", "arrest_stops"), ("is_positive", "positive_stops")]:
                metric_eth = work.loc[work[metric].eq(1)].pivot_table(
                    index=["lsoa21cd", "month"],
                    columns="ethnicity_group",
                    values=metric,
                    aggfunc="size",
                    fill_value=0,
                )
                for group in ["white", "black", "asian", "mixed", "other"]:
                    grouped[f"{prefix}_{group}"] = metric_eth.get(group, 0)

            parts.append(grouped.reset_index())

    if not parts:
        return pd.DataFrame(columns=["lsoa21cd", "month"] + COUNT_COLUMNS)

    out = pd.concat(parts, ignore_index=True)
    numeric_cols = [col for col in out.columns if col not in {"lsoa21cd", "month"}]
    out[numeric_cols] = out[numeric_cols].fillna(0)
    out = out.groupby(["lsoa21cd", "month"], as_index=False)[numeric_cols].sum()
    out["stop_search_success_rate"] = safe_divide(out["stop_search_positive_outcome_count"], out["stop_search_count"])
    out["stop_search_arrest_rate"] = safe_divide(out["stop_search_arrest_count"], out["stop_search_count"])
    out["stop_search_no_result_rate"] = safe_divide(out["stop_search_no_result_count"], out["stop_search_count"])

    qa_payload = dict(qa)
    qa_payload["outcome_frequencies"] = dict(outcome_counter.most_common(100))
    write_json(config.POLICE_PROCESS_QA_PATH, qa_payload)
    log_dropped(dict(dropped))
    return out


def _optional_metric(frame: pd.DataFrame, include: list[str], exclude: list[str] | None = None) -> pd.Series | None:
    col = find_column(frame.columns, include, exclude=exclude, required=False)
    if not col:
        return None
    return numeric(frame[col])


def process_imd_data(london_codes: set[str]) -> pd.DataFrame:
    imd = read_csv_flexible(config.IMD_FILE7_PATH)
    lsoa_col = first_existing_column(imd.columns, ["LSOA code (2021)", "LSOA21CD", "LSOA code"])
    if not lsoa_col:
        lsoa_col = find_column(imd.columns, ["lsoa", "code"])
    out = pd.DataFrame({"lsoa21cd": imd[lsoa_col].astype(str).str.strip()})
    metrics = {
        "imd_score": (["index", "multiple", "deprivation", "score"], []),
        "imd_rank": (["index", "multiple", "deprivation", "rank"], []),
        "imd_decile": (["index", "multiple", "deprivation", "decile"], []),
        "income_score": (["income", "score"], ["children", "older"]),
        "income_rank": (["income", "rank"], ["children", "older"]),
        "income_decile": (["income", "decile"], ["children", "older"]),
        "employment_score": (["employment", "score"], []),
        "employment_decile": (["employment", "decile"], []),
        "education_score": (["education", "score"], []),
        "education_decile": (["education", "decile"], []),
        "crime_domain_score": (["crime", "score"], []),
        "crime_domain_decile": (["crime", "decile"], []),
        "health_score": (["health", "score"], []),
        "health_decile": (["health", "decile"], []),
        "living_environment_score": (["living", "environment", "score"], []),
        "living_environment_decile": (["living", "environment", "decile"], []),
        "imd_population": (["total", "population"], []),
    }
    for out_col, (include, exclude) in metrics.items():
        series = _optional_metric(imd, include, exclude)
        if series is not None:
            out[out_col] = series

    for decile_col, intensity_col in [
        ("imd_decile", "deprivation_intensity"),
        ("income_decile", "income_deprivation_intensity"),
        ("education_decile", "education_deprivation_intensity"),
    ]:
        if decile_col in out.columns:
            out[intensity_col] = 11 - out[decile_col]

    out = out.loc[out["lsoa21cd"].isin(london_codes)].drop_duplicates("lsoa21cd")
    return out


def _wide_census_ethnicity(frame: pd.DataFrame, lsoa_col: str) -> pd.DataFrame:
    out = pd.DataFrame({"lsoa21cd": frame[lsoa_col].astype(str).str.strip()})
    labels = {col: str(col).strip().lower() for col in frame.columns if col != lsoa_col}

    def matching_columns(tokens: list[str]) -> list[str]:
        broad: list[str] = []
        detail: list[str] = []
        for col, label in labels.items():
            if all(token in label for token in tokens):
                if label.count(":") <= 1:
                    broad.append(col)
                else:
                    detail.append(col)
        return broad or detail

    total_cols = matching_columns(["total", "usual residents"])
    group_cols = {
        "white": matching_columns(["ethnic group", "white"]),
        "black": matching_columns(["ethnic group", "black"]),
        "asian": matching_columns(["ethnic group", "asian"]),
        "mixed": matching_columns(["ethnic group", "mixed"]),
        "other": matching_columns(["ethnic group", "other ethnic"]),
    }
    if total_cols:
        out["total_population"] = numeric(frame[total_cols[0]]).fillna(0)
    else:
        out["total_population"] = 0
    for key, cols in group_cols.items():
        if cols:
            out[f"{key}_population"] = sum((numeric(frame[col]).fillna(0) for col in cols), start=0)
        else:
            out[f"{key}_population"] = 0
    return out


def _long_census_ethnicity(frame: pd.DataFrame, lsoa_col: str, category_col: str, value_col: str) -> pd.DataFrame:
    work = frame[[lsoa_col, category_col, value_col]].copy()
    work.columns = ["lsoa21cd", "ethnicity", "value"]
    work["lsoa21cd"] = work["lsoa21cd"].astype(str).str.strip()
    work["value"] = numeric(work["value"]).fillna(0)
    work["label"] = work["ethnicity"].astype(str).str.lower()

    rows = []
    for code, group in work.groupby("lsoa21cd", observed=True):
        labels = group["label"]
        values = group["value"]
        total_mask = labels.str.contains("total") & (labels.str.contains("usual residents") | labels.str.contains("all"))
        white = values.loc[labels.str.contains("white")].sum()
        black = values.loc[labels.str.contains("black")].sum()
        asian = values.loc[labels.str.contains("asian") | labels.str.contains("chinese")].sum()
        mixed = values.loc[labels.str.contains("mixed") | labels.str.contains("multiple")].sum()
        other = values.loc[labels.str.contains("other ethnic") | labels.str.contains("arab")].sum()
        total = values.loc[total_mask].max() if total_mask.any() else white + black + asian + mixed + other
        rows.append(
            {
                "lsoa21cd": code,
                "total_population": total,
                "white_population": white,
                "black_population": black,
                "asian_population": asian,
                "mixed_population": mixed,
                "other_population": other,
            }
        )
    return pd.DataFrame(rows)


def process_census_ethnicity_data(london_codes: set[str]) -> pd.DataFrame:
    census = read_csv_flexible(config.CENSUS_ETHNICITY_PATH)
    lsoa_col = first_existing_column(census.columns, ["geography code", "GeographyCode", "LSOA21CD", "LSOA code"])
    if not lsoa_col:
        lsoa_col = find_column(census.columns, ["geography", "code"])

    category_col = find_column(census.columns, ["ethnic", "group"], required=False)
    value_col = first_existing_column(census.columns, ["observation", "Observation", "OBS_VALUE", "Value", "value"])
    if category_col and value_col and category_col != value_col:
        out = _long_census_ethnicity(census, lsoa_col, category_col, value_col)
    else:
        out = _wide_census_ethnicity(census, lsoa_col)

    out = out.loc[out["lsoa21cd"].isin(london_codes)].drop_duplicates("lsoa21cd")
    for col in ["white_population", "black_population", "asian_population", "mixed_population", "other_population"]:
        if col not in out.columns:
            out[col] = 0
    out["non_white_population"] = (
        out["black_population"] + out["asian_population"] + out["mixed_population"] + out["other_population"]
    )
    if "total_population" not in out.columns:
        out["total_population"] = out["white_population"] + out["non_white_population"]
    out["prop_white"] = safe_divide(out["white_population"], out["total_population"])
    out["prop_black"] = safe_divide(out["black_population"], out["total_population"])
    out["prop_asian"] = safe_divide(out["asian_population"], out["total_population"])
    out["prop_mixed"] = safe_divide(out["mixed_population"], out["total_population"])
    out["prop_other"] = safe_divide(out["other_population"], out["total_population"])
    out["prop_non_white"] = safe_divide(out["non_white_population"], out["total_population"])
    return out


def complete_panel(
    london: pd.DataFrame,
    crime: pd.DataFrame,
    stops: pd.DataFrame,
    imd: pd.DataFrame,
    census: pd.DataFrame,
) -> pd.DataFrame:
    months = pd.DataFrame({"month": [str(month) for month in month_range()]})
    grid = london[["lsoa21cd"]].merge(months, how="cross")
    static = london.merge(imd, on="lsoa21cd", how="left").merge(census, on="lsoa21cd", how="left")
    static.to_csv(config.STATIC_FEATURES_PATH, index=False)

    panel = grid.merge(crime, on=["lsoa21cd", "month"], how="left").merge(stops, on=["lsoa21cd", "month"], how="left")
    panel = panel.merge(static, on="lsoa21cd", how="left")
    for col in COUNT_COLUMNS:
        if col not in panel.columns:
            panel[col] = 0
    existing_count_cols = [col for col in COUNT_COLUMNS if col in panel.columns]
    panel[existing_count_cols] = panel[existing_count_cols].fillna(0).astype("int64")

    for col in ["stop_search_success_rate", "stop_search_arrest_rate", "stop_search_no_result_rate"]:
        if col in panel.columns:
            panel = panel.drop(columns=[col])
    panel["stop_search_success_rate"] = safe_divide(panel["stop_search_positive_outcome_count"], panel["stop_search_count"])
    panel["stop_search_arrest_rate"] = safe_divide(panel["stop_search_arrest_count"], panel["stop_search_count"])
    panel["stop_search_no_result_rate"] = safe_divide(panel["stop_search_no_result_count"], panel["stop_search_count"])
    panel["population"] = panel["total_population"].where(panel["total_population"].notna(), panel.get("imd_population"))
    panel = panel.sort_values(["lsoa21cd", "month"]).reset_index(drop=True)
    return panel


def main() -> None:
    ensure_dirs()
    london = build_london_lsoa_universe()
    london_codes = set(london["lsoa21cd"].astype(str))
    crime = process_crime_data(london_codes)
    stops = process_stop_search_data(london_codes)
    imd = process_imd_data(london_codes)
    census = process_census_ethnicity_data(london_codes)
    panel = complete_panel(london, crime, stops, imd, census)
    panel.to_parquet(config.PANEL_PARQUET_PATH, index=False)


if __name__ == "__main__":
    main()
