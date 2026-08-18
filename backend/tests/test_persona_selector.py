import pytest

from app.personas.selector import PersonaSelector
from app.router.classifier import QueryIntent


def test_selector_uses_grounded_analyst_for_document_knowledge():
    selector = PersonaSelector()

    persona = selector.select(
        QueryIntent.DOCUMENT_KNOWLEDGE
    )

    assert persona.name == "grounded_analyst"


def test_selector_uses_general_assistant_for_system_metadata():
    selector = PersonaSelector()

    persona = selector.select(
        QueryIntent.SYSTEM_METADATA
    )

    assert persona.name == "general_assistant"


def test_selector_uses_general_assistant_for_math():
    selector = PersonaSelector()

    persona = selector.select(
        QueryIntent.MATHEMATICAL_CALCULATION
    )

    assert persona.name == "general_assistant"