from dataclasses import dataclass

from app.llm.service import LLMService
from app.personas.config import PersonaConfig
from app.rag.context import AssembledContext, ContextSource


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
    def __init__(
        self,
        llm_service: LLMService,
        persona: PersonaConfig,
    ):
        self.llm_service = llm_service
        self.persona = persona

    def answer(
        self,
        question: str,
        context: AssembledContext,
    ) -> GroundedResponse:

        user_message = (
            f"Context:\n\n"
            f"{context.text}\n\n"
            f"Question:\n\n"
            f"{question}"
        )

        answer = self.llm_service.generate(
            persona=self.persona,
            user_message=user_message,
            system_prompt=GROUNDING_SYSTEM_PROMPT,
        )

        return GroundedResponse(
            answer=answer,
            sources=context.sources,
        )