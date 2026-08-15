from app.router.classifier import (
    QueryClassification,
    QueryIntent,
    QueryRoute,
)
from app.router.router import (
    QueryRouter,
    RouteTarget,
)


def test_router_routes_rag_query():
    classification = QueryClassification(
        route=QueryRoute.RAG,
        intent=QueryIntent.DOCUMENT_KNOWLEDGE,
    )

    router = QueryRouter()

    result = router.route(classification)

    assert result.target == RouteTarget.RAG
    assert result.classification == classification


def test_router_routes_database_query():
    classification = QueryClassification(
        route=QueryRoute.DATABASE,
        intent=QueryIntent.SYSTEM_METADATA,
    )

    router = QueryRouter()

    result = router.route(classification)

    assert result.target == RouteTarget.DATABASE
    assert result.classification == classification