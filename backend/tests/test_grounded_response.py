from app.rag.context import AssembledContext, ContextSource
from app.rag.grounded_response import GroundedResponseService


class FakeLLMClient:
    def __init__(self, response):
        self.messages = None
        self.temperature = None
        self.response = response

    def generate(self, messages, temperature=0.0):
        self.messages = messages
        self.temperature = temperature

        return self.response


def test_grounded_response_uses_context():
    source = ContextSource(
        source_number=1,
        filename="financial_report.pdf",
        chunk_index=3,
        content="Northbridge FC generated £204.0 million in revenue.",
    )

    context = AssembledContext(
        text=(
            "[Source 1]\n"
            "Document: financial_report.pdf\n"
            "Chunk: 3\n\n"
            "Northbridge FC generated £204.0 million in revenue.\n"
        ),
        sources=[source],
    )

    llm = FakeLLMClient(
        "Northbridge FC generated £204.0 million in revenue. [Source 1]"
    )
    service = GroundedResponseService(llm)

    response = service.answer(
        question="What was Northbridge FC's revenue?",
        context=context,
    )

    assert response.answer == (
        "Northbridge FC generated £204.0 million in revenue. [Source 1]"
    )

    assert response.sources == [source]

    assert llm.temperature == 0.0
    assert len(llm.messages) == 2

    assert llm.messages[0]["role"] == "system"
    assert "source of truth" in llm.messages[0]["content"]
    assert "[Source N]" in llm.messages[0]["content"]

    assert llm.messages[1]["role"] == "user"
    assert "£204.0 million" in llm.messages[1]["content"]
    assert "What was Northbridge FC's revenue?" in (
        llm.messages[1]["content"]
    )


def test_grounded_response_handles_empty_context():
    context = AssembledContext(
        text="",
        sources=[],
    )

    llm = FakeLLMClient(
        "The provided context does not contain enough information "
        "to answer the question."
    )
    service = GroundedResponseService(llm)

    response = service.answer(
        question="What was Northbridge FC's revenue?",
        context=context,
    )

    assert response.answer == (
        "The provided context does not contain enough information "
        "to answer the question."
    )

    assert response.sources == []