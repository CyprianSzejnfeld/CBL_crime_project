from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import paths
from .routers import lsoas, packages
from .schemas import Health

app = FastAPI(
    title="Met FairSearch API",
    description="FairStop London deliverable API.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:4173", "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=Health, tags=["health"])
def health() -> Health:
    return Health(status="ok", data_available=paths.outputs_available(), latest_period=None)


app.include_router(lsoas.router)
app.include_router(packages.router)
