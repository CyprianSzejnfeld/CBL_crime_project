from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .utils import normalize_column_name, percentile_0_100, read_csv_flexible


LONDON_BOROUGHS_32 = [
    "Barking and Dagenham",
    "Barnet",
    "Bexley",
    "Brent",
    "Bromley",
    "Camden",
    "Croydon",
    "Ealing",
    "Enfield",
    "Greenwich",
    "Hackney",
    "Hammersmith and Fulham",
    "Haringey",
    "Harrow",
    "Havering",
    "Hillingdon",
    "Hounslow",
    "Islington",
    "Kensington and Chelsea",
    "Kingston upon Thames",
    "Lambeth",
    "Lewisham",
    "Merton",
    "Newham",
    "Redbridge",
    "Richmond upon Thames",
    "Southwark",
    "Sutton",
    "Tower Hamlets",
    "Waltham Forest",
    "Wandsworth",
    "Westminster",
]

AGGREGATE_AREA_VALUES = {
    "london",
    "all london",
    "greater london",
    "met total",
    "mps total",
    "mps",
    "england",
    "national",
    "unknown",
    "total",
}

STANDARD_INDICATOR_COLUMNS = [
    "confidence_local_police_pct",
    "trust_met_police_pct",
    "police_fair_treatment_pct",
    "police_listen_concerns_pct",
    "police_do_good_job_pct",
    "feel_safe_local_area_pct",
]

PDF_BOROUGH_TABLE_COLUMNS = [
    "police_do_good_job_pct",
    "trust_met_police_pct",
    "police_fair_treatment_pct",
    "police_use_stop_search_fairly_pct",
]

TRUST_KEYWORDS = [
    "confidence",
    "trust",
    "fair",
    "fairly",
    "treatment",
    "listen",
    "concerns",
    "good job",
    "effective",
    "safety",
    "safe",
    "local police",
    "met police",
    "metropolitan police",
]


def ensure_mopac_dirs() -> None:
    for path in [
        config.MOPAC_RAW_DIR,
        config.MOPAC_INTERIM_DIR,
        config.MOPAC_PROCESSED_DIR,
    ]:
        Path(path).mkdir(parents=True, exist_ok=True)


def standardize_borough_name(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    if not text:
        return None
    text = text.replace("&", "and")
    text = re.sub(r"\bLondon Borough of\b", "", text, flags=re.I).strip()
    text = re.sub(r"\bCouncil\b", "", text, flags=re.I).strip()
    replacements = {
        "Barking and dagenham": "Barking and Dagenham",
        "Hammersmith and fulham": "Hammersmith and Fulham",
        "Kensington and chelsea": "Kensington and Chelsea",
        "Kingston upon thames": "Kingston upon Thames",
        "Richmond upon thames": "Richmond upon Thames",
        "City of westminster": "Westminster",
        "Westminster city": "Westminster",
        "Westminster city council": "Westminster",
    }
    title = text.title()
    title = title.replace(" And ", " and ").replace(" Upon ", " upon ")
    return replacements.get(title, title)


def expected_boroughs() -> list[str]:
    if config.LONDON_LSOA_LOOKUP_PATH.exists():
        try:
            lookup = read_csv_flexible(config.LONDON_LSOA_LOOKUP_PATH)
            if "borough" in lookup.columns:
                boroughs = sorted({standardize_borough_name(v) for v in lookup["borough"].dropna()})
                boroughs = [b for b in boroughs if b and b != "City of London"]
                if boroughs:
                    return boroughs
        except Exception:
            pass
    return LONDON_BOROUGHS_32


def is_aggregate_area(value: object) -> bool:
    standard = standardize_borough_name(value)
    if standard is None:
        return True
    return normalize_column_name(standard) in {normalize_column_name(v) for v in AGGREGATE_AREA_VALUES}


def parse_percentage(value: object) -> tuple[float | None, str | None]:
    if value is None or pd.isna(value):
        return None, None
    raw = str(value).strip()
    if raw.lower() in {"", "nan", "none", "null", "n/a", "na", "suppressed", "*", "-"}:
        return None, None
    had_percent = "%" in raw
    cleaned = re.sub(r"[^0-9.\-]", "", raw)
    if not cleaned:
        return None, f"could not parse percentage: {raw!r}"
    try:
        number = float(cleaned)
    except ValueError:
        return None, f"could not parse percentage: {raw!r}"
    if not had_percent and 0 <= number <= 1:
        number *= 100
    if number < 0 or number > 100:
        return number, f"percentage outside 0-100: {raw!r} -> {number}"
    return number, None


def detect_borough_column(frame: pd.DataFrame) -> str | None:
    candidates = [
        "borough",
        "local_authority",
        "local authority",
        "geography",
        "geography_name",
        "area",
    ]
    norm_map = {normalize_column_name(col): col for col in frame.columns}
    for candidate in candidates:
        match = norm_map.get(normalize_column_name(candidate))
        if match is not None:
            return match
    borough_set = set(expected_boroughs())
    best_col = None
    best_hits = 0
    for col in frame.columns:
        values = {standardize_borough_name(v) for v in frame[col].dropna().head(500)}
        hits = len(values & borough_set)
        if hits > best_hits:
            best_col = col
            best_hits = hits
    return best_col if best_hits >= 5 else None


def detect_period_column(frame: pd.DataFrame) -> str | None:
    candidates = ["year", "quarter", "wave", "period", "date", "survey_year", "financial_year"]
    norm_map = {normalize_column_name(col): col for col in frame.columns}
    for candidate in candidates:
        match = norm_map.get(normalize_column_name(candidate))
        if match is not None:
            return match
    return None


def indicator_standard_name(column: str) -> str | None:
    norm = normalize_column_name(column)
    if "trust" in norm and ("mps" in norm or "met" in norm or "police" in norm):
        return "trust_met_police_pct"
    if "treat" in norm and "fair" in norm:
        return "police_fair_treatment_pct"
    if "listen" in norm and ("concern" in norm or "local" in norm):
        return "police_listen_concerns_pct"
    if "goodjob" in norm or ("good" in norm and "job" in norm):
        return "police_do_good_job_pct"
    if ("safe" in norm or "safety" in norm) and ("local" in norm or "area" in norm or "afterdark" in norm):
        return "feel_safe_local_area_pct"
    if "confidence" in norm and ("localpolice" in norm or "police" in norm):
        return "confidence_local_police_pct"
    return None


def detect_trust_columns(frame: pd.DataFrame) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for col in frame.columns:
        standard = indicator_standard_name(str(col))
        if standard and standard not in mapping.values():
            mapping[str(col)] = standard
            continue
        norm = normalize_column_name(col)
        if any(normalize_column_name(keyword) in norm for keyword in TRUST_KEYWORDS):
            guessed = indicator_standard_name(str(col))
            if guessed and guessed not in mapping.values():
                mapping[str(col)] = guessed
    return mapping


def period_sort_key(value: object) -> tuple[int, int, int, str]:
    text = str(value or "")
    lowered = text.lower()
    match = re.search(r"q([1-4])[_\s-]*(\d{2})[_\s/-]*(\d{2})", lowered)
    if match:
        quarter = int(match.group(1))
        start = 2000 + int(match.group(2))
        end = 2000 + int(match.group(3))
        return (end, start, quarter, text)
    match = re.search(r"q([1-4]).*?(20\d{2})", lowered)
    if match:
        return (int(match.group(2)), int(match.group(2)), int(match.group(1)), text)
    match = re.search(r"(20\d{2})", lowered)
    if match:
        year = int(match.group(1))
        return (year, year, 0, text)
    return (0, 0, 0, text)


def period_from_filename(path: Path) -> str:
    name = path.name
    match = re.search(r"Q([1-4])[_\s-]*(\d{2})[_\s-]*(\d{2})", name, flags=re.I)
    if match:
        return f"Q{match.group(1)} {match.group(2)}-{match.group(3)}"
    match = re.search(r"Q([1-4]).*?(20\d{2})", name, flags=re.I)
    if match:
        return f"Q{match.group(1)} {match.group(2)}"
    return path.stem


def candidate_files() -> list[Path]:
    extensions = {".csv", ".xlsx", ".xls", ".zip", ".pdf"}
    return sorted(path for path in config.MOPAC_RAW_DIR.glob("*") if path.suffix.lower() in extensions)


def pdf_text(path: Path) -> str:
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", str(path), "-"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout
    except FileNotFoundError as exc:
        raise RuntimeError("pdftotext is required to parse MOPAC PDF result packs") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"pdftotext failed for {path}: {exc.stderr}") from exc


def parse_public_voice_pdf(path: Path) -> pd.DataFrame:
    text = pdf_text(path)
    start = text.find("Barking and Dagenham")
    end_marker = "CSEW data shows"
    end = text.find(end_marker, start)
    table_text = text[start:end if end > start else None]
    rows: list[dict[str, object]] = []
    period = period_from_filename(path)
    boroughs = expected_boroughs()
    pattern = re.compile(
        rf"^\s*({'|'.join(re.escape(b) for b in boroughs)})\s+"
        r"([0-9]+%)\s+([0-9]+%)\s+([0-9]+%)\s+([0-9]+%)",
        flags=re.M,
    )
    for match in pattern.finditer(table_text):
        values: dict[str, object] = {
            "borough": standardize_borough_name(match.group(1)),
            "period_label": period,
            "source_file": path.name,
        }
        for raw_value, col in zip(match.groups()[1:], PDF_BOROUGH_TABLE_COLUMNS):
            parsed, _ = parse_percentage(raw_value)
            values[col] = parsed
        rows.append(values)

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"No borough-level trust table parsed from {path}")
    return frame


def read_tabular_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return read_csv_flexible(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise RuntimeError(f"unsupported tabular file: {path}")


def tabular_to_borough_context(path: Path) -> pd.DataFrame | None:
    frame = read_tabular_file(path)
    borough_col = detect_borough_column(frame)
    period_col = detect_period_column(frame)
    mapping = detect_trust_columns(frame)
    if not borough_col or not mapping:
        return None

    work = frame.copy()
    work["borough"] = work[borough_col].map(standardize_borough_name)
    work = work.loc[~work[borough_col].map(is_aggregate_area)].copy()
    if period_col:
        work["period_label"] = work[period_col].astype(str)
    else:
        work["period_label"] = path.stem
    work["source_file"] = path.name

    for original, standard in mapping.items():
        parsed = work[original].map(parse_percentage)
        work[standard] = parsed.map(lambda item: item[0])

    keep = ["borough", "period_label", "source_file"] + sorted(set(mapping.values()))
    result = work[keep].copy()
    result = result.loc[result["borough"].isin(expected_boroughs()) | result["borough"].eq("City of London")]
    return result


def choose_latest_period(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    if "period_label" not in work.columns:
        work["period_label"] = "unknown"
    period_scores = (
        work.groupby("period_label", observed=True)
        .agg(row_count=("borough", "nunique"))
        .reset_index()
    )
    period_scores["sort_key"] = period_scores["period_label"].map(period_sort_key)
    if period_scores["sort_key"].map(lambda x: x[0]).max() > 0:
        selected_period = period_scores.sort_values(["sort_key", "row_count"], ascending=[False, False]).iloc[0][
            "period_label"
        ]
    else:
        selected_period = period_scores.sort_values("row_count", ascending=False).iloc[0]["period_label"]
    latest = work.loc[work["period_label"].astype(str).eq(str(selected_period))].copy()
    return latest.drop_duplicates("borough", keep="last")


def finalize_trust_context(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in STANDARD_INDICATOR_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    score_cols = [col for col in STANDARD_INDICATOR_COLUMNS if out[col].notna().any()]
    if not score_cols:
        raise RuntimeError("No usable trust/confidence indicators found in MOPAC source files")
    out["indicators_used_count"] = out[score_cols].notna().sum(axis=1)
    out["trust_context_score_0_100"] = out[score_cols].mean(axis=1)
    out["low_trust_context_score_0_100"] = 100 - out["trust_context_score_0_100"]
    out["trust_context_percentile"] = percentile_0_100(out["trust_context_score_0_100"])
    out["borough_low_trust_flag"] = out["trust_context_percentile"].le(config.LOW_TRUST_PERCENTILE_CUTOFF)
    out["borough_high_trust_flag"] = out["trust_context_percentile"].ge(config.HIGH_TRUST_PERCENTILE_CUTOFF)
    out["trust_context_band"] = np.select(
        [out["borough_low_trust_flag"], out["borough_high_trust_flag"]],
        ["Low trust context", "High trust context"],
        default="Medium trust context",
    )
    out["trust_context_reliability"] = np.select(
        [out["indicators_used_count"].ge(2), out["indicators_used_count"].eq(1)],
        ["standard", "low"],
        default="missing",
    )
    warning = "High fairness-risk LSOA is located in a low-trust borough; prioritise review."
    out["suggested_low_trust_warning_text"] = np.where(out["borough_low_trust_flag"], warning, "")
    out["suggested_priority_uplift_points_if_high_fairness"] = np.where(
        out["borough_low_trust_flag"], config.LOW_TRUST_PRIORITY_UPLIFT_POINTS, 0
    )
    out["suggested_priority_multiplier_if_high_fairness"] = np.where(
        out["borough_low_trust_flag"], config.LOW_TRUST_PRIORITY_MULTIPLIER, 1.0
    )
    final_cols = [
        "borough",
        "period_label",
        "confidence_local_police_pct",
        "trust_met_police_pct",
        "police_fair_treatment_pct",
        "police_listen_concerns_pct",
        "police_do_good_job_pct",
        "feel_safe_local_area_pct",
        "trust_context_score_0_100",
        "low_trust_context_score_0_100",
        "trust_context_percentile",
        "trust_context_band",
        "borough_low_trust_flag",
        "borough_high_trust_flag",
        "trust_context_reliability",
        "indicators_used_count",
        "suggested_low_trust_warning_text",
        "suggested_priority_uplift_points_if_high_fairness",
        "suggested_priority_multiplier_if_high_fairness",
        "source_file",
    ]
    return out[final_cols].sort_values("borough").reset_index(drop=True)

