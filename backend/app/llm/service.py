from uuid import UUID

from app.llm.client import LLMClient
from app.llm.fallback import ModelFallback
from app.llm.model_selector import ModelSelector
from app.llm.semantic_cache import SemanticCache
from app.personas.config import PersonaConfig


class LLMService:
    def __init__(
        self,
        model_selector: ModelSelector,
        fallback_model: str,
        semantic_cache: SemanticCache | None = None,
    ):
        self.model_selector = model_selector
        self.fallback_model = fallback_model
        self.semantic_cache = semantic_cache

    def generate(
        self,
        persona: PersonaConfig,
        user_message: str,
        system_prompt: str | None = None,
        knowledge_base_id: UUID | None = None,
        version_id: UUID | None = None,
    ) -> str:

        if (
            self.semantic_cache is not None
            and knowledge_base_id is not None
            and version_id is not None
        ):
            cached_response = self.semantic_cache.get(
                query=user_message,
                knowledge_base_id=knowledge_base_id,
                version_id=version_id,
            )

            if cached_response is not None:
                return cached_response

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

        response = fallback.generate(
            messages=messages,
            temperature=persona.temperature,
        )

        if (
            self.semantic_cache is not None
            and knowledge_base_id is not None
            and version_id is not None
        ):
            self.semantic_cache.set(
                query=user_message,
                response=response,
                knowledge_base_id=knowledge_base_id,
                version_id=version_id,
            )

        return response