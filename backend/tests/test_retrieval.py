from uuid import uuid4

from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.rag.retrieval import retrieve


class FakeQuery:
    def __init__(self, records):
        self.records = records

    def filter(self, *args):
        return self

    def all(self):
        return self.records


class FakeDB:
    def __init__(self, documents):
        self.documents = documents

    def query(self, model):
        return FakeQuery(self.documents)


def test_retrieve_returns_structured_results(monkeypatch):
    document_id = uuid4()
    chunk_id = uuid4()

    document = Document(
        id=document_id,
        kb_version_id=uuid4(),
        filename="northbridge_financial_report.pdf",
        content_type="application/pdf",
        storage_path="test.pdf",
        checksum="a" * 64,
    )

    chunk = DocumentChunk(
        id=chunk_id,
        document_id=document_id,
        chunk_index=2,
        content="Northbridge FC generated £204.0 million in revenue.",
    )

    def fake_retrieve_chunks(
        db,
        knowledge_base_id,
        query,
        top_k,
    ):
        return [chunk]

    monkeypatch.setattr(
        "app.rag.retrieval.retrieve_chunks",
        fake_retrieve_chunks,
    )

    db = FakeDB([document])

    results = retrieve(
        db=db,
        knowledge_base_id=uuid4(),
        query="What was Northbridge FC's revenue?",
        top_k=5,
    )

    assert len(results) == 1

    result = results[0]

    assert result.chunk_id == chunk_id
    assert result.document_id == document_id
    assert result.chunk_index == 2
    assert result.content == (
        "Northbridge FC generated £204.0 million in revenue."
    )
    assert result.filename == "northbridge_financial_report.pdf"


def test_retrieve_returns_empty_list_when_no_chunks(monkeypatch):
    def fake_retrieve_chunks(
        db,
        knowledge_base_id,
        query,
        top_k,
    ):
        return []

    monkeypatch.setattr(
        "app.rag.retrieval.retrieve_chunks",
        fake_retrieve_chunks,
    )

    db = FakeDB([])

    results = retrieve(
        db=db,
        knowledge_base_id=uuid4(),
        query="Unknown question",
    )

    assert results == []