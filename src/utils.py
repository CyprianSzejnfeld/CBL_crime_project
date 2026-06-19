from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def ensure_dirs() -> None:
    for path in [
        config.RAW_POLICE_DIR,
        config.RAW_GEO_DIR,
        config.RAW_IMD_DIR,
        config.RAW_CENSUS_DIR,
        config.INTERIM_DIR,
        config.PROCESSED_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def month_range(start: str | None = None, end: str | None = None) -> pd.PeriodIndex:
    return pd.period_range(start or config.START_MONTH, end or config.END_MONTH, freq="M")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def normalize_column_name(name: object) -> str:
    text = str(name).strip().lower()
    text = text.replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", "", text)


def clean_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(col).strip() for col in frame.columns]
    return frame


def read_csv_flexible(path: Path, **kwargs) -> pd.DataFrame:
    errors: list[str] = []
    for encoding in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            return clean_columns(pd.read_csv(path, encoding=encoding, low_memory=False, **kwargs))
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise RuntimeError(f"Could not read {path}; tried encodings: {errors}")


def find_column(
    columns: Iterable[object],
    include: Iterable[str],
    exclude: Iterable[str] | None = None,
    required: bool = True,
) -> str | None:
    normalized = {str(col): normalize_column_name(col) for col in columns}
    include_norm = [normalize_column_name(tok) for tok in include]
    exclude_norm = [normalize_column_name(tok) for tok in (exclude or [])]
    for original, norm in normalized.items():
        if all(tok in norm for tok in include_norm) and not any(tok in norm for tok in exclude_norm):
            return original
    if required:
        raise RuntimeError(f"Missing column containing {list(include)} excluding {list(exclude or [])}")
    return None


def first_existing_column(columns: Iterable[object], candidates: Iterable[str]) -> str | None:
    col_list = [str(col) for col in columns]
    normalized = {normalize_column_name(col): col for col in col_list}
    for candidate in candidates:
        if candidate in col_list:
            return candidate
        match = normalized.get(normalize_column_name(candidate))
        if match is not None:
            return match
    return None


def require_columns(frame: pd.DataFrame, columns: Iterable[str], source: Path | str) -> None:
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise RuntimeError(f"{source} missing columns: {missing}")


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace({0: pd.NA})
    return numerator / denominator


def percentile_0_100(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    result = pd.Series(np.nan, index=series.index, dtype="float64")
    if valid.empty:
        return result
    if len(valid) == 1:
        result.loc[valid.index] = 100.0
        return result
    ranks = valid.rank(method="average")
    result.loc[valid.index] = (ranks - 1) / (len(valid) - 1) * 100
    return result


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")


def log_dropped(reason_counts: dict[str, int]) -> None:
    if not reason_counts:
        return
    rows = [{"reason": reason, "row_count": count} for reason, count in reason_counts.items() if count]
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(
        config.DROPPED_ROWS_LOG_PATH,
        mode="a",
        header=not config.DROPPED_ROWS_LOG_PATH.exists(),
        index=False,
    )
