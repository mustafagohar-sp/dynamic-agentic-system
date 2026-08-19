from uuid import UUID
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.models.kb_version import KBVersion, KBVersionStatus
from app.models.knowledge_base import KnowledgeBase
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.rag.vector_store import upsert_chunk


def activate_version(
    db: Session,
    knowledge_base_id: UUID,
    version_id: UUID,
) -> KBVersion:
    knowledge_base = db.get(
        KnowledgeBase,
        knowledge_base_id,
    )

    if knowledge_base is None:
        raise ValueError("Knowledge base not found")

    version = db.get(
        KBVersion,
        version_id,
    )

    if version is None:
        raise ValueError("Knowledge base version not found")

    if version.knowledge_base_id != knowledge_base_id:
        raise ValueError(
            "Version does not belong to the knowledge base"
        )

    if version.status != KBVersionStatus.READY:
        raise ValueError(
            "Only READY versions can be activated"
        )

    current_version = None

    if knowledge_base.active_version_id is not None:
        current_version = db.get(
            KBVersion,
            knowledge_base.active_version_id,
        )

    if current_version is not None:
        current_version.status = KBVersionStatus.SUPERSEDED

    version.status = KBVersionStatus.ACTIVE
    version.activated_at = datetime.now(timezone.utc)

    knowledge_base.active_version_id = version.id

    db.commit()
    db.refresh(version)

    return version


def create_version(
    db: Session,
    knowledge_base_id: UUID,
) -> KBVersion:

    knowledge_base = db.get(
        KnowledgeBase,
        knowledge_base_id,
    )

    if knowledge_base is None:
        raise ValueError("Knowledge base not found")

    latest_version = (
        db.query(KBVersion)
        .filter(
            KBVersion.knowledge_base_id == knowledge_base_id
        )
        .order_by(
            KBVersion.version_number.desc()
        )
        .first()
    )

    next_version_number = 1

    if latest_version:
        next_version_number = (
            latest_version.version_number + 1
        )

    new_version = KBVersion(
        knowledge_base_id=knowledge_base_id,
        version_number=next_version_number,
        status=KBVersionStatus.DRAFT,
    )

    db.add(new_version)
    db.commit()
    db.refresh(new_version)


    #
    # Clone active version documents + chunks
    #

    active_version_id = knowledge_base.active_version_id

    if active_version_id is None:
        return new_version


    old_documents = (
        db.query(Document)
        .filter(
            Document.kb_version_id == active_version_id
        )
        .all()
    )


    for old_document in old_documents:

        new_document = Document(
            kb_version_id=new_version.id,
            filename=old_document.filename,
            content_type=old_document.content_type,
            storage_path=old_document.storage_path,
            checksum=old_document.checksum,
        )

        db.add(new_document)
        db.commit()
        db.refresh(new_document)


        old_chunks = (
            db.query(DocumentChunk)
            .filter(
                DocumentChunk.document_id == old_document.id
            )
            .all()
        )


        for old_chunk in old_chunks:

            new_chunk = DocumentChunk(
                document_id=new_document.id,
                chunk_index=old_chunk.chunk_index,
                content=old_chunk.content,
            )

            db.add(new_chunk)
            db.commit()
            db.refresh(new_chunk)


            #
            # Clone vector into Pinecone
            #

            upsert_chunk(
                chunk_id=new_chunk.id,
                content=new_chunk.content,
                knowledge_base_id=knowledge_base_id,
                version_id=new_version.id,
                document_id=new_document.id,
            )


    return new_version