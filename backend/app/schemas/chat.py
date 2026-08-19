from uuid import UUID

from pydantic import BaseModel


class ChatRequest(BaseModel):
    knowledge_base_id: UUID
    persona : str
    message: str


class SourceResponse(BaseModel):
    filename: str
    chunk_index: int


class ChatResponse(BaseModel):
    route: str
    answer: str
    sources: list[SourceResponse]