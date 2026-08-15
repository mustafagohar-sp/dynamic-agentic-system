from uuid import uuid4

import pytest

from app.database.service import DatabaseService
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.kb_version import KBVersion, KBVersionStatus
from app.models.knowledge_base import KnowledgeBase


class FakeQuery:
    def __init__(self, value):
        self.value = value

    def join(self, *args, **kwargs):
        return self

    def filter(self, *args, **kwargs):
        return self

    def scalar(self):
        return self.value


class FakeDB:
    def __init__(self, knowledge_base=None, versions=None, query_values=None):
        self.knowledge_base = knowledge_base
        self.versions = versions or {}
        self.query_values = query_values or []

    def get(self, model, record_id):
        if model == KnowledgeBase:
            if (
                self.knowledge_base is not None
                and record_id == self.knowledge_base.id
            ):
                return self.knowledge_base

            return None

        if model == KBVersion:
            return self.versions.get(record_id)

        return None

    def query(self, *args):
        return FakeQuery(self.query_values.pop(0))


def create_knowledge_base():
    return KnowledgeBase(
        id=uuid4(),
        name="Northbridge FC",
        description="Northbridge FC knowledge base",
    )


def test_get_document_count():
    knowledge_base = create_knowledge_base()

    db = FakeDB(
        knowledge_base=knowledge_base,
        query_values=[12],
    )

    service = DatabaseService()

    result = service.get_document_count(
        db=db,
        knowledge_base_id=knowledge_base.id,
    )

    assert result == 12


def test_get_chunk_count():
    knowledge_base = create_knowledge_base()

    db = FakeDB(
        knowledge_base=knowledge_base,
        query_values=[419],
    )

    service = DatabaseService()

    result = service.get_chunk_count(
        db=db,
        knowledge_base_id=knowledge_base.id,
    )

    assert result == 419


def test_get_active_version():
    knowledge_base = create_knowledge_base()

    version = KBVersion(
        id=uuid4(),
        knowledge_base_id=knowledge_base.id,
        version_number=1,
        status=KBVersionStatus.ACTIVE,
    )

    knowledge_base.active_version_id = version.id

    db = FakeDB(
        knowledge_base=knowledge_base,
        versions={
            version.id: version,
        },
    )

    service = DatabaseService()

    result = service.get_active_version(
        db=db,
        knowledge_base_id=knowledge_base.id,
    )

    assert result == version


def test_get_active_version_returns_none_when_no_active_version():
    knowledge_base = create_knowledge_base()

    db = FakeDB(
        knowledge_base=knowledge_base,
    )

    service = DatabaseService()

    result = service.get_active_version(
        db=db,
        knowledge_base_id=knowledge_base.id,
    )

    assert result is None


def test_get_version_status():
    knowledge_base = create_knowledge_base()

    version = KBVersion(
        id=uuid4(),
        knowledge_base_id=knowledge_base.id,
        version_number=1,
        status=KBVersionStatus.READY,
    )

    db = FakeDB(
        knowledge_base=knowledge_base,
        versions={
            version.id: version,
        },
    )

    service = DatabaseService()

    result = service.get_version_status(
        db=db,
        knowledge_base_id=knowledge_base.id,
        version_id=version.id,
    )

    assert result == "ready"


def test_database_service_rejects_unknown_knowledge_base():
    db = FakeDB()

    service = DatabaseService()

    with pytest.raises(
        ValueError,
        match="Knowledge base not found",
    ):
        service.get_document_count(
            db=db,
            knowledge_base_id=uuid4(),
        )


def test_database_service_rejects_version_from_another_knowledge_base():
    knowledge_base = create_knowledge_base()

    other_knowledge_base = create_knowledge_base()

    version = KBVersion(
        id=uuid4(),
        knowledge_base_id=other_knowledge_base.id,
        version_number=1,
        status=KBVersionStatus.READY,
    )

    db = FakeDB(
        knowledge_base=knowledge_base,
        versions={
            version.id: version,
        },
    )

    service = DatabaseService()

    with pytest.raises(
        ValueError,
        match="Version does not belong to the knowledge base",
    ):
        service.get_version_status(
            db=db,
            knowledge_base_id=knowledge_base.id,
            version_id=version.id,
        )