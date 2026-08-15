from uuid import uuid4

import pytest

from app.models.document import Document
from app.models.kb_version import KBVersion, KBVersionStatus
from app.rag.pipeline import ingest_documents


class FakeDB:
    def __init__(self, version):
        self.version = version
        self.commits = 0
        self.rollbacks = 0
        self.refreshed = []

    def get(self, model, record_id):
        if record_id == self.version.id:
            return self.version
        return None

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def refresh(self, document):
        self.refreshed.append(document)


def test_ingest_documents_processes_multiple_files(monkeypatch):
    version = KBVersion(
        id=uuid4(),
        knowledge_base_id=uuid4(),
        version_number=1,
        status=KBVersionStatus.DRAFT,
    )

    db = FakeDB(version)

    document_1 = Document(
        id=uuid4(),
        kb_version_id=version.id,
        filename="financial.pdf",
        content_type="application/pdf",
        storage_path="financial.pdf",
        checksum="a" * 64,
    )

    document_2 = Document(
        id=uuid4(),
        kb_version_id=version.id,
        filename="commercial.pdf",
        content_type="application/pdf",
        storage_path="commercial.pdf",
        checksum="b" * 64,
    )

    calls = []

    def fake_ingest_document(db, version, file_path):
        calls.append(file_path)

        if file_path == "financial.pdf":
            return document_1, [uuid4(), uuid4()]

        return document_2, [uuid4()]

    monkeypatch.setattr(
        "app.rag.pipeline._ingest_document",
        fake_ingest_document,
    )

    documents = ingest_documents(
        db=db,
        version_id=version.id,
        file_paths=[
            "financial.pdf",
            "commercial.pdf",
        ],
    )

    assert documents == [document_1, document_2]
    assert calls == [
        "financial.pdf",
        "commercial.pdf",
    ]

    assert version.status == KBVersionStatus.READY
    assert db.commits == 1
    assert db.rollbacks == 0


def test_ingest_documents_rolls_back_on_failure(monkeypatch):
    version = KBVersion(
        id=uuid4(),
        knowledge_base_id=uuid4(),
        version_number=1,
        status=KBVersionStatus.DRAFT,
    )

    db = FakeDB(version)

    uploaded_ids = [uuid4(), uuid4()]

    document = Document(
        id=uuid4(),
        kb_version_id=version.id,
        filename="financial.pdf",
        content_type="application/pdf",
        storage_path="financial.pdf",
        checksum="a" * 64,
    )

    def fake_ingest_document(db, version, file_path):
        if file_path == "financial.pdf":
            return document, uploaded_ids

        raise RuntimeError("Simulated ingestion failure")

    deleted_ids = []

    def fake_delete_chunks(chunk_ids):
        deleted_ids.extend(chunk_ids)

    monkeypatch.setattr(
        "app.rag.pipeline._ingest_document",
        fake_ingest_document,
    )

    monkeypatch.setattr(
        "app.rag.pipeline.delete_chunks",
        fake_delete_chunks,
    )

    with pytest.raises(
        RuntimeError,
        match="Simulated ingestion failure",
    ):
        ingest_documents(
            db=db,
            version_id=version.id,
            file_paths=[
                "financial.pdf",
                "commercial.pdf",
            ],
        )

    assert deleted_ids == uploaded_ids
    assert version.status == KBVersionStatus.DRAFT
    assert db.rollbacks == 1
    assert db.commits == 1


def test_ingest_documents_rejects_empty_file_list():
    version = KBVersion(
        id=uuid4(),
        knowledge_base_id=uuid4(),
        version_number=1,
        status=KBVersionStatus.DRAFT,
    )

    db = FakeDB(version)

    with pytest.raises(
        ValueError,
        match="At least one document is required",
    ):
        ingest_documents(
            db=db,
            version_id=version.id,
            file_paths=[],
        )

    assert version.status == KBVersionStatus.DRAFT
    assert db.commits == 0
    assert db.rollbacks == 0