from uuid import UUID
from typing import Any, TypedDict
from app.router.classifier import QueryClassification
from app.router.router import RoutingDecision


class AgentState(TypedDict, total=False):
    user_message: str
    knowledge_base_id: str | None
    persona : str

    classification: QueryClassification
    routing_decision: RoutingDecision

    result: Any
    error: str | None