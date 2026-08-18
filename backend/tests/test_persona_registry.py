import pytest

from app.personas.registry import get_persona


def test_get_grounded_analyst_persona():
    persona = get_persona("grounded_analyst")

    assert persona.name == "grounded_analyst"
    assert persona.temperature == 0.0
    assert persona.preferred_model


def test_get_general_assistant_persona():
    persona = get_persona("general_assistant")

    assert persona.name == "general_assistant"
    assert persona.temperature == 0.2
    assert persona.preferred_model


def test_get_persona_rejects_unknown_name():
    with pytest.raises(
        ValueError,
        match="Unknown persona: unknown",
    ):
        get_persona("unknown")