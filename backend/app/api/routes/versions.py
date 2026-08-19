from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.models.kb_version import KBVersion
from app.rag.versioning import (
    create_version,
    activate_version,
)
from app.schemas.version import VersionResponse


router = APIRouter(
    prefix="/versions",
    tags=["Versions"],
)


@router.post(
    "/knowledge-bases/{kb_id}/versions",
    response_model=VersionResponse,
)
def create_kb_version(
    kb_id: UUID,
    db: Session = Depends(get_db),
):
    return create_version(
        db=db,
        knowledge_base_id=kb_id,
    )


@router.get(
    "/knowledge-bases/{kb_id}/versions",
    response_model=list[VersionResponse],
)
def list_versions(
    kb_id: UUID,
    db: Session = Depends(get_db),
):
    return (
        db.query(KBVersion)
        .filter(
            KBVersion.knowledge_base_id == kb_id
        )
        .order_by(
            KBVersion.version_number
        )
        .all()
    )


@router.post(
    "/{version_id}/activate",
    response_model=VersionResponse,
)
def activate_kb_version(
    version_id: UUID,
    db: Session = Depends(get_db),
):
    version = db.get(
        KBVersion,
        version_id,
    )

    if version is None:
        raise ValueError("Version not found")

    return activate_version(
        db=db,
        knowledge_base_id=version.knowledge_base_id,
        version_id=version_id,
    )