from openai import OpenAI

from app.config import settings


class LLMClient:
    def __init__(self, model : str | None = None):
        self.client = OpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
        )
        self.model = model or settings.openrouter_model

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )

        content = response.choices[0].message.content

        if not content:
            raise ValueError("LLM returned an empty response")

        return content