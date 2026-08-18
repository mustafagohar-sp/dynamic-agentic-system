from app.personas.config import PersonaConfig
from app.rag.context import AssembledContext, ContextSource
from app.rag.grounded_response import GroundedResponseService
from app.llm.service import LLMService
from app.personas.registry import get_persona

class FakeLLMService:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate(
        self,
        persona,
        user_message,
        system_prompt=None,
    ):
        self.calls.append(
            {
                "persona": persona,
                "user_message": user_message,
                "system_prompt": system_prompt,
            }
        )

        return self.response


def test_grounded_response_uses_context():
    source = ContextSource(
        source_number=1,
        filename="financial_report.pdf",
        chunk_index=3,
        content="Northbridge FC generated Â£204.0 million in revenue.",
    )

    context = AssembledContext(
        text=(
            "[Source 1]\n"
            "Document: financial_report.pdf\n"
            "Chunk: 3\n\n"
            "Northbridge FC generated Â£204.0 millionin revenue.\n"
        ),
        sources=[source],
    )

    llm_service = FakeLLMService(
        "Northbridge FC generated Â£204.0 million in revenue. [Source 1]"
    )

    persona = PersonaConfig(
        name="grounded_analyst",
        system_prompt="Use only the provided context.",
        temperature=0.0,
        preferred_model="test-model",
    )

    service = GroundedResponseService(
        llm_service=llm_service,
        persona=persona,
    )

    response = service.answer(
        question="What was Northbridge FC's revenue?",
        context=context,
    )

    assert response.answer == (
        "Northbridge FC generated Â£204.0 million in revenue. [Source 1]"
    )

    assert response.sources == [source]

    assert len(llm_service.calls) == 1

    call = llm_service.calls[0]

    assert call["persona"] == persona

    assert "Â£204.0 million" in call["user_message"]
    assert "What was Northbridge FC's revenue?" in call["user_message"]

    assert call["system_prompt"] is not None
    assert "source of truth" in call["system_prompt"]
    assert "[Source N]" in call["system_prompt"]


def test_grounded_response_handles_empty_context():
    context = AssembledContext(
        text="",
        sources=[],
    )

    llm_service = FakeLLMService(
        "The provided context does not contain enoughinformation "
        "to answer the question."
    )

    persona = PersonaConfig(
        name="grounded_analyst",
        system_prompt="Use only the provided context.",
        temperature=0.0,
        preferred_model="test-model",
    )

    service = GroundedResponseService(
        llm_service=llm_service,
        persona=persona,
    )

    response = service.answer(
        question="What was Northbridge FC's revenue?",
        context=context,
    )

    assert response.answer == (
        "The provided context does not contain enoughinformation "
        "to answer the question."
    )

    assert response.sources == []

    assert len(llm_service.calls) == 1


def test_grounded_response_uses_llm_service_with_grounded_persona(
    monkeypatch,
):
    calls = []

    class FakeLLMService:
        def generate(
            self,
            persona,
            user_message,
            system_prompt=None,
        ):
            calls.append(
                {
                    "persona": persona,
                    "user_message": user_message,
                    "system_prompt": system_prompt,
                }
            )

            return "Northbridge FC generated Â£204.0 million in revenue. [Source 1]"

    source = ContextSource(
        source_number=1,
        filename="financial_report.pdf",
        chunk_index=3,
        content="Northbridge FC generated Â£204.0 million in revenue.",
    )

    context = AssembledContext(
        text=(
            "[Source 1]\n"
            "Document: financial_report.pdf\n"
            "Chunk: 3\n\n"
            "Northbridge FC generated Â£204.0 million in revenue.\n"
        ),
        sources=[source],
    )

    persona = get_persona("grounded_analyst")

    service = GroundedResponseService(
        llm_service=FakeLLMService(),
        persona=persona,
    )

    response = service.answer(
        question="What was Northbridge FC's revenue?",
        context=context,
    )

    assert response.answer == (
        "Northbridge FC generated Â£204.0 million in revenue. [Source 1]"
    )

    assert response.sources == [source]

    assert len(calls) == 1

    assert calls[0]["persona"] == persona
    assert calls[0]["persona"].name == "grounded_analyst"
    assert calls[0]["persona"].temperature == 0.0

    assert calls[0]["system_prompt"] is not None
    assert "source of truth" in calls[0]["system_prompt"]
    assert "[Source N]" in calls[0]["system_prompt"]

    assert "Â£204.0 million" in calls[0]["user_message"]
    assert "What was Northbridge FC's revenue?" in calls[0]["user_message"]