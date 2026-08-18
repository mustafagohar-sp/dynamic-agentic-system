import pytest

from app.llm.model_selector import ModelSelector
from app.personas.config import PersonaConfig


def test_model_selector_returns_persona_model():
    persona = PersonaConfig(
        name="grounded_analyst",
        system_prompt="Answer using provided context.",
        temperature=0.0,
        preferred_model="openai/gpt-4o-mini",
    )

    model = ModelSelector().select(persona)

    assert model == "openai/gpt-4o-mini"


def test_model_selector_rejects_missing_model():
    persona = PersonaConfig(
        name="grounded_analyst",
        system_prompt="Answer using provided context.",
        temperature=0.0,
        preferred_model="",
    )

    with pytest.raises(
        ValueError,
        match="has no preferred model",
    ):
        ModelSelector().select(persona)