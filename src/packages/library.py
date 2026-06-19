from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import math

import yaml

from src import config

BUDGETS_PATH = config.ROOT_DIR / "config" / "intervention_resource_budgets.yaml"

RESOURCE_TYPES = [
    "encounter_quality_audits",
    "procedural_justice_training_places",
    "community_scrutiny_sessions",
    "precision_protection_packages",
    "monitored_search_regime_reviews",
    "evaluation_slots",
]


@dataclass(frozen=True)
class Package:
    id: str
    cost_key: str
    name: str
    components: list[str]
    includes_reduction: bool = False
    reduction_only: bool = False


PACKAGES: list[Package] = [
    Package(
        "P0", "P0_monitor_only", "Monitor Only",
        ["routine quarterly monitoring"],
    ),
    Package(
        "P1", "P1_low_yield_review", "Search-Practice Review",
        ["supervisory review of low-result search practice",
         "category-specific grounds and policy review",
         "quarterly reassessment",
         "evaluation-ready monitoring plan"],
        includes_reduction=True, reduction_only=True,
    ),
    Package(
        "P2", "P2_training_and_audit", "Procedural-Justice Training and Grounds/BWV Audit Priority",
        ["procedural-justice training allocation",
         "supervisor quality-audit capacity",
         "grounds review where grounds text exists; BWV review as a proposed activity",
         "quarterly monitoring"],
    ),
    Package(
        "P3", "P3_community_confidence", "Community Scrutiny and Confidence Intervention",
        ["community/public encounter scrutiny session",
         "local reporting/engagement allocation",
         "quality-review follow-up"],
    ),
    Package(
        "P4", "P4_precision_protection", "Fairness-Safeguarded Precision Protection",
        ["protected harm-focused presence maintained",
         "procedural-justice training priority",
         "supervisory/BWV/grounds audit priority",
         "community scrutiny where relevant"],
    ),
    Package(
        "P5", "P5_combined_high_priority", "Combined High-Priority Fairness Intervention",
        ["training", "audit", "community scrutiny",
         "search-practice review where supported",
         "post-intervention evaluation plan"],
        includes_reduction=True,
    ),
]

PACKAGE_BY_ID = {p.id: p for p in PACKAGES}


@lru_cache(maxsize=1)
def load_budgets() -> dict:
    with open(BUDGETS_PATH) as fh:
        return yaml.safe_load(fh)


def quarterly_budgets() -> dict[str, float]:
    return {k: float(v) for k, v in load_budgets()["quarterly_budgets"].items()}


def package_cost(package_id: str, cluster_features: dict | None = None) -> dict[str, float]:


    costs = load_budgets()["package_costs"]
    pkg = PACKAGE_BY_ID[package_id]
    raw = costs.get(pkg.cost_key, {})
    base = {r: float(raw.get(r, 0)) for r in RESOURCE_TYPES}
    if cluster_features is None or package_id == "P0":
        return base
    return _scaled_cost(package_id, base, cluster_features)


def _feature(features: dict, key: str, default: float = 0.0) -> float:
    try:
        value = features.get(key, default)
    except AttributeError:
        value = default
    if value is None:
        return default
    try:
        n = float(value)
    except (TypeError, ValueError):
        return default
    return n if math.isfinite(n) else default


def _tier(value: float, threshold: float, step: float, cap: int) -> int:
    if value <= threshold:
        return 0
    return int(min(cap, math.ceil((value - threshold) / step)))


def _scaled_cost(package_id: str, base: dict[str, float], features: dict) -> dict[str, float]:
    out = base.copy()
    encounters = _feature(features, "quarterly_total_encounters")
    population = _feature(features, "cluster_population")
    lsoas = _feature(features, "number_of_lsoas")
    wards = _feature(features, "number_of_wards")
    resident_caution = bool(features.get("resident_denominator_caution", False)) if hasattr(features, "get") else False
    band = str(features.get("protection_need_band", "")) if hasattr(features, "get") else ""

    volume_tier = _tier(encounters, threshold=150, step=150, cap=4)
    spread_tier = max(_tier(wards, threshold=2, step=2, cap=3), _tier(lsoas, threshold=20, step=20, cap=3))
    population_tier = _tier(population, threshold=50000, step=50000, cap=3)

    if out["encounter_quality_audits"] > 0:
        out["encounter_quality_audits"] += 5 * volume_tier
    if out["procedural_justice_training_places"] > 0:
        out["procedural_justice_training_places"] += 2 * spread_tier
    if out["community_scrutiny_sessions"] > 0:
        out["community_scrutiny_sessions"] += population_tier
        if resident_caution:
            out["community_scrutiny_sessions"] += 1
    if out["precision_protection_packages"] > 0:
        if band in {"High", "Critical"} and (encounters >= 350 or population >= 65000):
            out["precision_protection_packages"] += 1
        if band == "Critical" and (encounters >= 600 or population >= 100000):
            out["precision_protection_packages"] += 1
    if out["monitored_search_regime_reviews"] > 0:
        if encounters >= 300 or lsoas >= 35:
            out["monitored_search_regime_reviews"] += 1
        if encounters >= 700:
            out["monitored_search_regime_reviews"] += 1
    if out["evaluation_slots"] > 0 and package_id == "P5" and (encounters >= 700 or population >= 100000):
        out["evaluation_slots"] += 1

    return {k: float(v) for k, v in out.items()}
