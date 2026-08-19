from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class KnowledgeBaseCreate(BaseModel):
    name: str
    description: str | None = None


class KnowledgeBaseResponse(BaseModel):
    id: UUID
    name: str
    description: str | None
    active_version_id: UUID | None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True