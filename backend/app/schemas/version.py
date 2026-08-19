from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class VersionResponse(BaseModel):
    id: UUID
    version_number: int
    status: str
    created_at: datetime
    activated_at: datetime | None

    class Config:
        from_attributes = True