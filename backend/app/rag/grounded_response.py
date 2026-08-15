from dataclasses import dataclass

from app.llm.client import LLMClient
from app.rag.context import AssembledContext , ContextSource


@dataclass(frozen=True)
class GroundedResponse:
    answer: str
    sources: list[ContextSource]


GROUNDING_SYSTEM_PROMPT = """You are a grounded question-answering assistant.

Answer the user's question using only the information provided in the
context.

Rules:
- Treat the provided context as the source of truth.
- Do not invent facts that are not supported by the context.
- Do not use outside knowledge to fill missing information.
- If the context does not contain enough information to answer the question,
  clearly say that the provided context does not contain enough information.
- Give a concise, direct answer.
- When a claim is supported by a provided source, cite it using its source
  number in the format [Source N].
"""


class GroundedResponseService:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def answer(
        self,
        question: str,
        context: AssembledContext,
    ) -> GroundedResponse:

        messages = [
            {
                "role": "system",
                "content": GROUNDING_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n\n"
                    f"{context.text}\n\n"
                    f"Question:\n\n"
                    f"{question}"
                ),
            },
        ]

        answer = self.llm_client.generate(
            messages=messages,
            temperature=0.0,
        )

        return GroundedResponse(
            answer=answer,
            sources=context.sources,
        )