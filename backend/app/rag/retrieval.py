from uuid import UUID

from sqlalchemy.orm import Session

from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.knowledge_base import KnowledgeBase
from app.rag.vector_store import search_similar


def retrieve_chunks(
    db: Session,
    knowledge_base_id: UUID,
    query: str,
    top_k: int = 5,
) -> list[DocumentChunk]:
    knowledge_base = db.get(
        KnowledgeBase,
        knowledge_base_id,
    )

    if knowledge_base is None:
        raise ValueError("Knowledge base not found")

    if knowledge_base.active_version_id is None:
        raise ValueError("Knowledge base has no active version")

    active_version_id = knowledge_base.active_version_id

    results = search_similar(
        query=query,
        top_k=top_k,
        version_id=active_version_id,
    )

    valid_chunk_ids = []

    for match in results.matches:
        metadata = match.metadata or {}

        if metadata.get("version_id") != str(active_version_id):
            continue

        valid_chunk_ids.append(UUID(match.id))

    if not valid_chunk_ids:
        return []

    chunks = (
        db.query(DocumentChunk)
        .join(
            Document,
            Document.id == DocumentChunk.document_id,
        )
        .filter(
            DocumentChunk.id.in_(valid_chunk_ids),
            Document.kb_version_id == active_version_id,
        )
        .all()
    )

    chunk_map = {
        chunk.id: chunk
        for chunk in chunks
    }

    return [
        chunk_map[chunk_id]
        for chunk_id in valid_chunk_ids
        if chunk_id in chunk_map
    ]