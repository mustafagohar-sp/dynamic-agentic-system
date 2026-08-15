import pytest

from app.router.classifier import (
    QueryClassification,
    QueryClassifier,
    QueryIntent,
    QueryRoute,
)


class FakeLLMClient:
    def __init__(self, response):
        self.response = response
        self.messages = None
        self.temperature = None

    def generate(self, messages, temperature=0.0):
        self.messages = messages
        self.temperature = temperature
        return self.response


def test_classifier_returns_rag_document_knowledge():
    llm = FakeLLMClient(
        '{"route": "rag", "intent": "document_knowledge"}'
    )

    classifier = QueryClassifier(llm)

    result = classifier.classify(
        "What was Northbridge FC's revenue?"
    )

    assert result == QueryClassification(
        route=QueryRoute.RAG,
        intent=QueryIntent.DOCUMENT_KNOWLEDGE,
    )

    assert llm.temperature == 0.0
    assert len(llm.messages) == 2
    assert llm.messages[0]["role"] == "system"
    assert "Return ONLY valid JSON" in llm.messages[0]["content"]
    assert llm.messages[1]["role"] == "user"


def test_classifier_returns_database_system_metadata():
    llm = FakeLLMClient(
        '{"route": "database", "intent": "system_metadata"}'
    )

    classifier = QueryClassifier(llm)

    result = classifier.classify(
        "How many documents are in this knowledge base?"
    )

    assert result.route == QueryRoute.DATABASE
    assert result.intent == QueryIntent.SYSTEM_METADATA


def test_classifier_rejects_empty_query():
    llm = FakeLLMClient(
        '{"route": "rag", "intent": "document_knowledge"}'
    )

    classifier = QueryClassifier(llm)

    with pytest.raises(ValueError, match="Query cannot be empty"):
        classifier.classify("   ")


def test_classifier_rejects_invalid_json():
    llm = FakeLLMClient("not valid json")

    classifier = QueryClassifier(llm)

    with pytest.raises(
        ValueError,
        match="LLM returned invalid classification JSON",
    ):
        classifier.classify("What was the revenue?")


def test_classifier_rejects_invalid_classification():
    llm = FakeLLMClient(
        '{"route": "unknown", "intent": "document_knowledge"}'
    )

    classifier = QueryClassifier(llm)

    with pytest.raises(
        ValueError,
        match="LLM returned an invalid query classification",
    ):
        classifier.classify("What was the revenue?")