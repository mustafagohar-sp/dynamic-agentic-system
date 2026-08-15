import json
from dataclasses import dataclass
from enum import Enum

from app.llm.client import LLMClient


class QueryRoute(str, Enum):
    RAG = "rag"
    DATABASE = "database"


class QueryIntent(str, Enum):
    DOCUMENT_KNOWLEDGE = "document_knowledge"
    SYSTEM_METADATA = "system_metadata"


@dataclass(frozen=True)
class QueryClassification:
    route: QueryRoute
    intent: QueryIntent


CLASSIFICATION_SYSTEM_PROMPT = """You are a query classification system.

Classify the user's query into exactly one route and one intent.

Routes:
- rag: questions requiring information from the content of knowledge-base documents.
- database: questions about knowledge-base structure, metadata, versions, documents,
  chunks, or system state.

Intents:
- document_knowledge: questions asking about information contained in documents.
- system_metadata: questions asking about stored knowledge-base or document metadata.

Return ONLY valid JSON in exactly this format:

{
  "route": "rag" or "database",
  "intent": "document_knowledge" or "system_metadata"
}

Do not include explanations, markdown, or additional fields.
"""


class QueryClassifier:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def classify(self, query: str) -> QueryClassification:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        messages = [
            {
                "role": "system",
                "content": CLASSIFICATION_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": query,
            },
        ]

        response = self.llm_client.generate(
            messages=messages,
            temperature=0.0,
        )

        try:
            data = json.loads(response)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "LLM returned invalid classification JSON"
            ) from exc

        if not isinstance(data, dict):
            raise ValueError(
                "LLM classification must be a JSON object"
            )

        try:
            route = QueryRoute(data["route"])
            intent = QueryIntent(data["intent"])
        except (KeyError, ValueError) as exc:
            raise ValueError(
                "LLM returned an invalid query classification"
            ) from exc

        return QueryClassification(
            route=route,
            intent=intent,
        )