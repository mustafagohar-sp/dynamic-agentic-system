from app.llm.client import LLMClient
from app.llm.fallback import ModelFallback
from app.llm.model_selector import ModelSelector
from app.personas.config import PersonaConfig


class LLMService:
    def __init__(
        self,
        model_selector: ModelSelector,
        fallback_model: str,
    ):
        self.model_selector = model_selector
        self.fallback_model = fallback_model

    def generate(
        self,
        persona: PersonaConfig,
        user_message: str,
        system_prompt : str | None = None,
    ) -> str:
        primary_model = self.model_selector.select(persona)

        primary_client = LLMClient(
            model=primary_model,
        )

        fallback_client = LLMClient(
            model=self.fallback_model,
        )

        fallback = ModelFallback(
            primary_client=primary_client,
            fallback_client=fallback_client,
        )

        messages = [
            {
                "role": "system",
                "content": system_prompt or persona.system_prompt,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ]

        return fallback.generate(
            messages=messages,
            temperature=persona.temperature,
        )