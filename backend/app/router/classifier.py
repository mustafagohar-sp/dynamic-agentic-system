import json
from dataclasses import dataclass
from enum import Enum

from app.llm.client import LLMClient


class QueryRoute(str, Enum):
    RAG = "rag"
    DATABASE = "database"
    MATH = "math"


class QueryIntent(str, Enum):
    DOCUMENT_KNOWLEDGE = "document_knowledge"
    SYSTEM_METADATA = "system_metadata"
    MATHEMATICAL_CALCULATION = "mathematical_calculation"


@dataclass(frozen=True)
class QueryClassification:
    route: QueryRoute
    intent: QueryIntent


CLASSIFICATION_SYSTEM_PROMPT = """You are a query classification system.

Your job is to decide where the answer should come from.

Classify the user's query into exactly one route and one intent.

Routes:

- rag:
  Use this when the user is asking about information contained inside uploaded knowledge-base documents.
  This includes:
  - company information
  - financial reports
  - revenue
  - expenses
  - players
  - teams
  - performance
  - events
  - facts mentioned in documents
  - any question where the answer must be retrieved from documents.

- database:
  Use this ONLY when the user is asking about the internal structure or state of the knowledge base system.
  Examples:
  - What is the active version?
  - List all versions
  - How many documents are stored?
  - How many chunks exist?
  - When was this knowledge base created?
  - What documents are uploaded?

- math:
  Use this ONLY when the user is asking for a calculation that can be solved directly from numbers in the query.
  Examples:
  - What is 15% of 240?
  - Calculate 50 + 25

Important rules:

1. If the question asks about information from documents, ALWAYS choose rag.
2. Words like revenue, profit, cost, player, captain, striker, winger, financial, report, performance DO NOT mean database.
3. Database is only for questions about the knowledge-base system itself.
4. Do not use database for business/company information.

Intents:

- document_knowledge:
  Questions requiring information from knowledge-base documents.

- system_metadata:
  Questions about knowledge-base structure, versions, documents, chunks, or system state.

- mathematical_calculation:
  Questions requiring arithmetic.

Return ONLY valid JSON:

{
  "route": "rag" or "database" or "math",
  "intent": "document_knowledge" or "system_metadata" or "mathematical_calculation"
}

No explanations.
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