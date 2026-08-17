from uuid import uuid4

from app.math.result import MathResponse
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


class FakeLLMClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate(self, messages, temperature=0.0):
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
            }
        )

        return self.response


class FakeMathEngine:
    def __init__(self):
        self.calls = []

    def calculate(self, expression):
        self.calls.append(expression)

        from app.math.engine import MathResult

        return MathResult(
            expression=expression,
            value=408,
        )


class FakeMathResultHandler:
    def __init__(self):
        self.calls = []

    def format(self, math_result):
        self.calls.append(math_result)

        return MathResponse(
            expression=math_result.expression,
            result=math_result.value,
            text=f"The result is {math_result.value}.",
        )


def test_executor_routes_rag_to_grounded_response(monkeypatch):
    grounded_response = FakeGroundedResponseService()
    database_service = FakeDatabaseService()
    llm_client = FakeLLMClient("17 * 24")
    math_engine = FakeMathEngine()
    math_result_handler = FakeMathResultHandler()

    executor = QueryExecutor(
        grounded_response_service=grounded_response,
        database_service=database_service,
        llm_client=llm_client,
        math_engine=math_engine,
        math_result_handler=math_result_handler,
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
    llm_client = FakeLLMClient("17 * 24")
    math_engine = FakeMathEngine()
    math_result_handler = FakeMathResultHandler()

    executor = QueryExecutor(
        grounded_response_service=grounded_response,
        database_service=database_service,
        llm_client=llm_client,
        math_engine=math_engine,
        math_result_handler=math_result_handler,
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
    llm_client = FakeLLMClient("17 * 24")
    math_engine = FakeMathEngine()
    math_result_handler = FakeMathResultHandler()

    executor = QueryExecutor(
        grounded_response_service=grounded_response,
        database_service=database_service,
        llm_client=llm_client,
        math_engine=math_engine,
        math_result_handler=math_result_handler,
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


def test_executor_routes_math_query_to_math_engine():
    grounded_response = FakeGroundedResponseService()
    database_service = FakeDatabaseService()
    llm_client = FakeLLMClient("17 * 24")
    math_engine = FakeMathEngine()
    math_result_handler = FakeMathResultHandler()

    executor = QueryExecutor(
        grounded_response_service=grounded_response,
        database_service=database_service,
        llm_client=llm_client,
        math_engine=math_engine,
        math_result_handler=math_result_handler,
    )

    knowledge_base_id = uuid4()

    classification = QueryClassification(
        route=QueryRoute.MATH,
        intent=QueryIntent.MATHEMATICAL_CALCULATION,
    )

    decision = RoutingDecision(
        target=RouteTarget.MATH,
        classification=classification,
    )

    result = executor.execute(
        db=None,
        knowledge_base_id=knowledge_base_id,
        query="What is 17 multiplied by 24?",
        decision=decision,
    )

    assert result == MathResponse(
        expression="17 * 24",
        result=408,
        text="The result is 408.",
    )

    assert math_engine.calls == ["17 * 24"]

    assert len(math_result_handler.calls) == 1
    assert math_result_handler.calls[0].expression == "17 * 24"
    assert math_result_handler.calls[0].value == 408

    assert len(llm_client.calls) == 1
    assert llm_client.calls[0]["temperature"] == 0.0

    messages = llm_client.calls[0]["messages"]

    assert messages[0]["role"] == "system"
    assert "mathematical expressions" in messages[0]["content"]

    assert messages[1]["role"] == "user"
    assert "17 multiplied by 24" in messages[1]["content"]