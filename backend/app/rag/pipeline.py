from pathlib import Path
from uuid import UUID

from sqlalchemy.orm import Session

from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.kb_version import KBVersion, KBVersionStatus
from app.rag.chunking import chunk_text
from app.rag.embeddings import generate_embeddings
from app.rag.ingestion import calculate_checksum, extract_text
from app.rag.vector_store import delete_chunks, upsert_chunks


def _ingest_document(
    db: Session,
    version: KBVersion,
    file_path: str,
    persona: str,
) -> tuple[Document, list[UUID]]:
    path = Path(file_path)
    text = extract_text(file_path)

    document = Document(
        kb_version_id=version.id,
        filename=path.name,
        content_type=(
            "application/pdf"
            if path.suffix.lower() == ".pdf"
            else "text/plain"
        ),
        storage_path=str(path),
        checksum=calculate_checksum(file_path),
    )

    db.add(document)
    db.flush()

    chunks = chunk_text(text)
    embeddings = generate_embeddings(chunks)

    uploaded_chunk_ids = []
    vectors = []

    for index, (content, embedding) in enumerate(
        zip(chunks, embeddings)
    ):
        chunk = DocumentChunk(
            document_id=document.id,
            chunk_index=index,
            content=content,
        )

        db.add(chunk)
        db.flush()

        uploaded_chunk_ids.append(chunk.id)

        vectors.append(
            {
                "id": str(chunk.id),
                "values": embedding,
                "metadata": {
                    "knowledge_base_id": str(version.knowledge_base_id),
                    "version_id": str(version.id),
                    "document_id": str(document.id),
                    "chunk_id": str(chunk.id),
                    "persona": persona,
                },
            }
        )

    upsert_chunks(vectors)

    return document, uploaded_chunk_ids


def ingest_document(
    db: Session,
    version_id: UUID,
    file_path: str,
    persona: str = "general",
) -> Document:
    version = db.get(KBVersion, version_id)

    if version is None:
        raise ValueError("Knowledge base version not found")

    if version.status != KBVersionStatus.DRAFT:
        raise ValueError(
            "Document ingestion requires a DRAFT version"
        )

    version.status = KBVersionStatus.PROCESSING

    uploaded_chunk_ids = []

    try:
        document, uploaded_chunk_ids = _ingest_document(
            db=db,
            version=version,
            file_path=file_path,
            persona=persona,
        )

        version.status = KBVersionStatus.READY

        db.commit()
        db.refresh(document)

        return document

    except Exception:
        if uploaded_chunk_ids:
            delete_chunks(uploaded_chunk_ids)

        db.rollback()

        version = db.get(KBVersion, version_id)

        if version is not None:
            version.status = KBVersionStatus.DRAFT
            db.commit()

        raise


def ingest_documents(
    db: Session,
    version_id: UUID,
    file_paths: list[str],
    persona: str = "general",
) -> list[Document]:
    version = db.get(KBVersion, version_id)

    if version is None:
        raise ValueError("Knowledge base version not found")

    if version.status != KBVersionStatus.DRAFT:
        raise ValueError(
            "Document ingestion requires a DRAFT version"
        )

    if not file_paths:
        raise ValueError("At least one document is required")

    version.status = KBVersionStatus.PROCESSING

    uploaded_chunk_ids = []
    documents = []

    try:
        for file_path in file_paths:
            document, chunk_ids = _ingest_document(
                db=db,
                version=version,
                file_path=file_path,
                persona=persona,
            )

            documents.append(document)
            uploaded_chunk_ids.extend(chunk_ids)

        version.status = KBVersionStatus.READY

        db.commit()

        for document in documents:
            db.refresh(document)

        return documents

    except Exception:
        if uploaded_chunk_ids:
            delete_chunks(uploaded_chunk_ids)

        db.rollback()

        version = db.get(KBVersion, version_id)

        if version is not None:
            version.status = KBVersionStatus.DRAFT
            db.commit()

        raise