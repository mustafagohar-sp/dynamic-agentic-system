from app.personas.config import PersonaConfig


class ModelSelector:
    def select(self, persona: PersonaConfig) -> str:
        if not persona.preferred_model:
            raise ValueError(
                f"Persona '{persona.name}' has no preferred model"
            )

        return persona.preferred_model