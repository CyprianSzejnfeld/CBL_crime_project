from __future__ import annotations

from .. import data_store


def map_lsoas() -> dict:
    return data_store.map_feature_collection()


def lsoa_detail(lsoa21cd: str) -> dict | None:
    return data_store.lsoa_detail(lsoa21cd)
