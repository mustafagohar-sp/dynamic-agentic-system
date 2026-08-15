from dataclasses import dataclass
from enum import Enum

from app.router.classifier import QueryClassification, QueryRoute


class RouteTarget(str, Enum):
    RAG = "rag"
    DATABASE = "database"


@dataclass(frozen=True)
class RoutingDecision:
    target: RouteTarget
    classification: QueryClassification


class QueryRouter:
    def route(
        self,
        classification: QueryClassification,
    ) -> RoutingDecision:
        if classification.route == QueryRoute.RAG:
            return RoutingDecision(
                target=RouteTarget.RAG,
                classification=classification,
            )

        if classification.route == QueryRoute.DATABASE:
            return RoutingDecision(
                target=RouteTarget.DATABASE,
                classification=classification,
            )

        raise ValueError(
            f"Unsupported query route: {classification.route}"
        )