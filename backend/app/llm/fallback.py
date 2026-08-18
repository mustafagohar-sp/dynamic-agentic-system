from app.llm.client import LLMClient


class ModelFallback:
    def __init__(
        self,
        primary_client: LLMClient,
        fallback_client: LLMClient,
    ):
        self.primary_client = primary_client
        self.fallback_client = fallback_client

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
    ) -> str:
        try:
            return self.primary_client.generate(
                messages=messages,
                temperature=temperature,
            )
        except Exception:
            return self.fallback_client.generate(
                messages=messages,
                temperature=temperature,
            )