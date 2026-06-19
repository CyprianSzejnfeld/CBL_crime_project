from fastapi import APIRouter, HTTPException

from ..services import lsoa_service

router = APIRouter(prefix="/api", tags=["lsoas"])


@router.get("/map/lsoas")
def map_lsoas():
    return lsoa_service.map_lsoas()


@router.get("/lsoas/{lsoa21cd}")
def lsoa_detail(lsoa21cd: str):
    detail = lsoa_service.lsoa_detail(lsoa21cd)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"LSOA {lsoa21cd} not found")
    return detail
