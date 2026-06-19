from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from .. import data_store_packages as store

router = APIRouter(tags=["packages"])


@router.get("/api/packages")
def packages() -> dict:
    return {
        "packages": store.package_library(),
        "quarterly_budgets": store.budgets(),
        "note": "Configurable planning-budget assumptions, not official Met capacity.",
    }


@router.get("/api/package-scenarios/{strategy_id}")
def package_scenario(strategy_id: str) -> dict:
    df = store.strategies()
    row = df[df["strategy_id"] == strategy_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="strategy not found")
    return store._records(row)[0]


@router.get("/api/map/package-clusters")
def map_package_clusters(scenario_id: str = Query("High-Volume Fairness Coverage")) -> dict:
    return store.map_package_clusters(scenario_id)


@router.get("/api/search-review")
def search_review() -> dict:
    return store.search_review()


@router.get("/api/map/search-review-clusters")
def map_search_review_clusters() -> dict:
    return store.map_search_review_clusters()


@router.get("/api/map/ward-criticalness")
def map_ward_criticalness() -> dict:
    return store.ward_criticalness_geojson()


@router.post("/api/packages/optimise")
def packages_optimise(
    scenario_id: str = Query("High-Volume Fairness Coverage"),
    budget_scale: float = Query(1.0, ge=0.1, le=6.0),
) -> dict:
    return store.optimise(scenario_id, budget_scale)


@router.get("/api/clusters/{cluster_id}/detail")
def cluster_detail(cluster_id: str) -> dict:
    detail = store.cluster_detail(cluster_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="cluster not found")
    return detail
