from __future__ import annotations

from pydantic import BaseModel


class Health(BaseModel):
    status: str
    data_available: bool
    latest_period: str | None = None
