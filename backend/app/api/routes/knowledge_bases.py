from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.models.knowledge_base import KnowledgeBase
from app.schemas.knowledge_base import (
    KnowledgeBaseCreate,
    KnowledgeBaseResponse,
)


router = APIRouter(
    prefix="/knowledge-bases",
    tags=["Knowledge Bases"],
)


@router.post(
    "",
    response_model=KnowledgeBaseResponse,
)
def create_knowledge_base(
    data: KnowledgeBaseCreate,
    db: Session = Depends(get_db),
):
    kb = KnowledgeBase(
        name=data.name,
        description=data.description,
    )

    db.add(kb)
    db.commit()
    db.refresh(kb)

    return kb


@router.get(
    "/{kb_id}",
    response_model=KnowledgeBaseResponse,
)
def get_knowledge_base(
    kb_id: UUID,
    db: Session = Depends(get_db),
):
    return db.get(
        KnowledgeBase,
        kb_id,
    )