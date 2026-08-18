from app.personas.config import PersonaConfig
from app.personas.registry import get_persona
from app.router.classifier import QueryIntent


class PersonaSelector:
    def select(
        self,
        intent: QueryIntent,
    ) -> PersonaConfig:
        if intent == QueryIntent.DOCUMENT_KNOWLEDGE:
            return get_persona("grounded_analyst")

        if intent == QueryIntent.SYSTEM_METADATA:
            return get_persona("general_assistant")

        if intent == QueryIntent.MATHEMATICAL_CALCULATION:
            return get_persona("general_assistant")

        raise ValueError(
            f"Unsupported query intent: {intent}"
        )