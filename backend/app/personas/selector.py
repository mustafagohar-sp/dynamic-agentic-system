from app.personas.config import PersonaConfig
from app.personas.registry import get_persona


class PersonaSelector:
    def select(
        self,
        persona_name: str,
    ) -> PersonaConfig:
        return get_persona(persona_name)