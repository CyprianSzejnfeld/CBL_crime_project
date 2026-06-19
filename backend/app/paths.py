from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_APP = REPO_ROOT / "data" / "app"
DATA_SCENARIOS = REPO_ROOT / "data" / "scenarios"

GEOMETRY = DATA_APP / "london_lsoa_geometry_reduced.geojson"
METRICS_JSON = DATA_APP / "london_lsoa_metrics_latest.json"

FAIRNESS_V2_LSOA_PATHWAYS = REPO_ROOT / "data" / "processed" / "fairness_v2_lsoa_pathways_latest.csv"
WARD_LOOKUP = REPO_ROOT / "data" / "interim" / "lsoa21_ward24_lookup_london.csv"
WARD_BOUNDARIES = REPO_ROOT / "data" / "raw" / "geo" / "london_ward24_boundaries.geojson"
WARD_QUARTER_METRICS = REPO_ROOT / "data" / "processed" / "ward_quarter_intervention_metrics.csv"

PROCESSED = REPO_ROOT / "data" / "processed"
WARD_CLUSTERS_LATEST = PROCESSED / "ward_clusters_latest.parquet"
CLUSTER_PROTECTION = PROCESSED / "cluster_protection_need_forecasts.csv"
WARD_PROTECTION = PROCESSED / "ward_protection_need_forecasts.csv"
CLUSTER_SEARCH_REGIMES = PROCESSED / "cluster_search_regime_profiles.csv"
CLUSTER_PACKAGE_ELIGIBILITY = PROCESSED / "cluster_package_eligibility.csv"

PKG_STRATEGIES = DATA_SCENARIOS / "intervention_package_all_strategies.csv"
PKG_ALLOCATIONS = DATA_SCENARIOS / "intervention_package_all_allocations.csv"


def outputs_available() -> bool:
    return GEOMETRY.exists() and METRICS_JSON.exists()
