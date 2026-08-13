from uuid import UUID
from datetime import datetime, timezone 
from sqlalchemy.orm import Session

from app.models.kb_version import KBVersion, KBVersionStatus
from app.models.knowledge_base import KnowledgeBase


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