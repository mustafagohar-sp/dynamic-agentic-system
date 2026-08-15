from uuid import uuid4

from app.rag.context import AssembledContext
from app.router.classifier import (
    QueryClassification,
    QueryIntent,
    QueryRoute,
)
from app.router.executor import QueryExecutor
from app.router.router import RouteTarget, RoutingDecision


class FakeGroundedResponseService:
    def __init__(self):
        self.calls = []

    def answer(self, question, context):
        self.calls.append(
            {
                "question": question,
                "context": context,
            }
        )

        return "RAG response"


class FakeDatabaseService:
    def __init__(self):
        self.calls = []

    def get_document_count(self, db, knowledge_base_id):
        self.calls.append(("document_count", knowledge_base_id))
        return 12

    def get_chunk_count(self, db, knowledge_base_id):
        self.calls.append(("chunk_count", knowledge_base_id))
        return 419

    def get_active_version(self, db, knowledge_base_id):
        self.calls.append(("active_version", knowledge_base_id))
        return None

    def get_version_status(
        self,
        db,
        knowledge_base_id,
        version_id,
    ):
        self.calls.append(
            (
                "version_status",
                knowledge_base_id,
                version_id,
            )
        )
        return "ready"


def test_executor_routes_rag_to_grounded_response(monkeypatch):
    grounded_response = FakeGroundedResponseService()
    database_service = FakeDatabaseService()

    executor = QueryExecutor(
        grounded_response_service=grounded_response,
        database_service=database_service,
    )

    knowledge_base_id = uuid4()

    classification = QueryClassification(
        route=QueryRoute.RAG,
        intent=QueryIntent.DOCUMENT_KNOWLEDGE,
    )

    decision = RoutingDecision(
        target=RouteTarget.RAG,
        classification=classification,
    )

    def fake_retrieve(**kwargs):
        return []

    def fake_assemble_context(results):
        return AssembledContext(
            text="test context",
            sources=[],
        )

    monkeypatch.setattr(
        "app.router.executor.retrieve",
        fake_retrieve,
    )

    monkeypatch.setattr(
        "app.router.executor.assemble_context",
        fake_assemble_context,
    )

    result = executor.execute(
        db=None,
        knowledge_base_id=knowledge_base_id,
        query="What was the revenue?",
        decision=decision,
    )

    assert result == "RAG response"
    assert grounded_response.calls[0]["question"] == (
        "What was the revenue?"
    )


def test_executor_routes_document_query_to_database():
    grounded_response = FakeGroundedResponseService()
    database_service = FakeDatabaseService()

    executor = QueryExecutor(
        grounded_response_service=grounded_response,
        database_service=database_service,
    )

    knowledge_base_id = uuid4()

    classification = QueryClassification(
        route=QueryRoute.DATABASE,
        intent=QueryIntent.SYSTEM_METADATA,
    )

    decision = RoutingDecision(
        target=RouteTarget.DATABASE,
        classification=classification,
    )

    result = executor.execute(
        db=None,
        knowledge_base_id=knowledge_base_id,
        query="How many documents are in this knowledge base?",
        decision=decision,
    )

    assert result == 12
    assert database_service.calls == [
        ("document_count", knowledge_base_id)
    ]


def test_executor_routes_chunk_query_to_database():
    grounded_response = FakeGroundedResponseService()
    database_service = FakeDatabaseService()

    executor = QueryExecutor(
        grounded_response_service=grounded_response,
        database_service=database_service,
    )

    knowledge_base_id = uuid4()

    classification = QueryClassification(
        route=QueryRoute.DATABASE,
        intent=QueryIntent.SYSTEM_METADATA,
    )

    decision = RoutingDecision(
        target=RouteTarget.DATABASE,
        classification=classification,
    )

    result = executor.execute(
        db=None,
        knowledge_base_id=knowledge_base_id,
        query="How many chunks are stored?",
        decision=decision,
    )

    assert result == 419