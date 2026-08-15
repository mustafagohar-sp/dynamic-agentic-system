from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.orm import Session

from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.knowledge_base import KnowledgeBase
from app.rag.vector_store import search_similar


@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: UUID
    document_id: UUID
    chunk_index: int
    content: str
    filename: str


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


def retrieve(
    db: Session,
    knowledge_base_id: UUID,
    query: str,
    top_k: int = 5,
) -> list[RetrievalResult]:
    """
    Application-facing retrieval service.

    Uses the existing version-aware retrieval implementation
    and enriches the retrieved chunks with document metadata.
    """

    chunks = retrieve_chunks(
        db=db,
        knowledge_base_id=knowledge_base_id,
        query=query,
        top_k=top_k,
    )

    if not chunks:
        return []

    document_ids = {
        chunk.document_id
        for chunk in chunks
    }

    documents = (
        db.query(Document)
        .filter(Document.id.in_(document_ids))
        .all()
    )

    document_map = {
        document.id: document
        for document in documents
    }

    results = []

    for chunk in chunks:
        document = document_map.get(chunk.document_id)

        if document is None:
            continue

        results.append(
            RetrievalResult(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                filename=document.filename,
            )
        )

    return results