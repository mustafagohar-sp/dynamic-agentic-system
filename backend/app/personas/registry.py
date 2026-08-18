from app.config import settings
from app.personas.config import PersonaConfig


PERSONAS = {
    "grounded_analyst": PersonaConfig(
        name="grounded_analyst",
        system_prompt=(
            "Answer accurately and concisely using only the "
            "information provided in the context."
        ),
        temperature=0.0,
        preferred_model=settings.openrouter_model,
    ),
    "general_assistant": PersonaConfig(
        name="general_assistant",
        system_prompt=(
            "Provide clear, helpful, and concise answers "
            "to the user's question."
        ),
        temperature=0.2,
        preferred_model=settings.openrouter_model,
    ),
}


def get_persona(name: str) -> PersonaConfig:
    try:
        return PERSONAS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown persona: {name}"
        ) from exc