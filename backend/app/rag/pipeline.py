from pathlib import Path
from uuid import UUID
from app.rag.vector_store import delete_chunks,upsert_chunk
from sqlalchemy.orm import Session

from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.kb_version import KBVersion, KBVersionStatus
from app.rag.chunking import chunk_text
from app.rag.embeddings import generate_embeddings
from app.rag.ingestion import extract_text
from app.rag.vector_store import upsert_chunk


def ingest_document(
    db: Session,
    version_id: UUID,
    file_path: str,
) -> Document:
    version = db.get(KBVersion, version_id)

    if version is None:
        raise ValueError("Knowledge base version not found")

    if version.status != KBVersionStatus.DRAFT:
        raise ValueError("Document ingestion requires a DRAFT version")

    version.status = KBVersionStatus.PROCESSING

    uploaded_chunk_ids =[]

    try:
        path = Path(file_path)
        text = extract_text(file_path)

        document = Document(
            kb_version_id=version.id,
            filename=path.name,
            content_type=None,
            storage_path=str(path),
            checksum=__import__("hashlib")
            .sha256(text.encode("utf-8"))
            .hexdigest(),
        )

        db.add(document)
        db.flush()

        chunks = chunk_text(text)

        embeddings = generate_embeddings(chunks)

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

            upsert_chunk(
                chunk_id=chunk.id,
                content=content,
                knowledge_base_id=version.knowledge_base_id,
                version_id=version.id,
                document_id=document.id,
                embedding = embedding,
            )
            uploaded_chunk_ids.append(chunk.id)

            # raise RuntimeError("TEST: simulated pipeline failure")

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