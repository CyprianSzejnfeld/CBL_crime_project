from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config
from .mopac_trust_context_utils import (
    candidate_files,
    choose_latest_period,
    ensure_mopac_dirs,
    finalize_trust_context,
    parse_public_voice_pdf,
    period_sort_key,
    tabular_to_borough_context,
)


def _source_priority(path: Path) -> tuple[int, tuple[int, int, int, str]]:
    name = path.name.lower()
    if path.suffix.lower() == ".pdf" and "public" in name and "voice" in name:
        return (3, period_sort_key(name))
    if path.suffix.lower() in {".csv", ".xlsx", ".xls"}:
        return (2, period_sort_key(name))
    return (1, period_sort_key(name))


def _try_london_wide_save(path: Path) -> bool:
    if path.suffix.lower() != ".csv":
        return False
    try:
        frame = pd.read_csv(path, low_memory=False)
    except Exception:
        return False
    cols = {str(col).lower() for col in frame.columns}
    if {"date", "measure", "proportion"}.issubset(cols):
        frame.to_csv(config.MOPAC_LONDON_WIDE_CONTEXT_PATH, index=False)
        return True
    return False


def _try_bcu_save(path: Path) -> bool:
    if path.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
        return False
    try:
        frame = pd.read_csv(path, low_memory=False) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    except Exception:
        return False
    if any("bcu" in str(col).lower() or "basic command unit" in str(col).lower() for col in frame.columns):
        frame.to_csv(config.MOPAC_BCU_CONTEXT_PATH, index=False)
        return True
    return False


def select_and_parse_source(files: list[Path]) -> pd.DataFrame:
    parsed_candidates: list[tuple[tuple[int, tuple[int, int, int, str]], pd.DataFrame]] = []

    for path in sorted(files, key=_source_priority, reverse=True):
        _try_london_wide_save(path)
        _try_bcu_save(path)
        try:
            if path.suffix.lower() == ".pdf":
                frame = parse_public_voice_pdf(path)
            elif path.suffix.lower() in {".csv", ".xlsx", ".xls"}:
                frame = tabular_to_borough_context(path)
                if frame is None:
                    continue
            else:
                continue
            borough_count = frame["borough"].nunique() if "borough" in frame.columns else 0
            if borough_count < 5:
                continue
            parsed_candidates.append((_source_priority(path), frame))
        except Exception:
            continue

    if not parsed_candidates:
        raise RuntimeError(
            "No borough-level MOPAC trust context source parsed. "
            f"Place a borough-level CSV/XLSX file in {config.MOPAC_RAW_DIR}, or a MOPAC Public Voice PDF result pack."
        )
    parsed_candidates.sort(key=lambda item: item[0], reverse=True)
    _, frame = parsed_candidates[0]
    return frame


def main() -> None:
    ensure_mopac_dirs()
    files = candidate_files()
    if not files:
        raise RuntimeError(
            f"No MOPAC CSV/XLSX/PDF files found in {config.MOPAC_RAW_DIR}. "
            "Place the MOPAC source files there manually."
        )
    frame = select_and_parse_source(files)
    latest = choose_latest_period(frame)
    output = finalize_trust_context(latest)
    output.to_parquet(config.MOPAC_TRUST_CONTEXT_PARQUET_PATH, index=False)


if __name__ == "__main__":
    main()

