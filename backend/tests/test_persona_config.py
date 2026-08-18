from app.personas.config import PersonaConfig


def test_persona_config_stores_configuration():
    persona = PersonaConfig(
        name="grounded_analyst",
        system_prompt="Answer using only the provided context.",
        temperature=0.0,
        preferred_model="openai/gpt-4o-mini",
    )

    assert persona.name == "grounded_analyst"
    assert persona.system_prompt == (
        "Answer using only the provided context."
    )
    assert persona.temperature == 0.0
    assert persona.preferred_model == "openai/gpt-4o-mini"