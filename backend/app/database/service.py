from uuid import UUID

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.kb_version import KBVersion
from app.models.knowledge_base import KnowledgeBase


class DatabaseService:
    def get_document_count(
        self,
        db: Session,
        knowledge_base_id: UUID,
    ) -> int:
        knowledge_base = db.get(
            KnowledgeBase,
            knowledge_base_id,
        )

        if knowledge_base is None:
            raise ValueError("Knowledge base not found")

        return (
            db.query(func.count(Document.id))
            .join(
                KBVersion,
                KBVersion.id == Document.kb_version_id,
            )
            .filter(
                KBVersion.knowledge_base_id == knowledge_base_id,
            )
            .scalar()
            or 0
        )

    def get_chunk_count(
        self,
        db: Session,
        knowledge_base_id: UUID,
    ) -> int:
        knowledge_base = db.get(
            KnowledgeBase,
            knowledge_base_id,
        )

        if knowledge_base is None:
            raise ValueError("Knowledge base not found")

        return (
            db.query(func.count(DocumentChunk.id))
            .join(
                Document,
                Document.id == DocumentChunk.document_id,
            )
            .join(
                KBVersion,
                KBVersion.id == Document.kb_version_id,
            )
            .filter(
                KBVersion.knowledge_base_id == knowledge_base_id,
            )
            .scalar()
            or 0
        )

    def get_active_version(
        self,
        db: Session,
        knowledge_base_id: UUID,
    ) -> KBVersion | None:
        knowledge_base = db.get(
            KnowledgeBase,
            knowledge_base_id,
        )

        if knowledge_base is None:
            raise ValueError("Knowledge base not found")

        if knowledge_base.active_version_id is None:
            return None

        return db.get(
            KBVersion,
            knowledge_base.active_version_id,
        )

    def get_version_status(
        self,
        db: Session,
        knowledge_base_id: UUID,
        version_id: UUID,
    ) -> str:
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

        return version.status.value