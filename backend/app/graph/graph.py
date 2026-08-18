from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session

from app.database.service import DatabaseService
from app.llm.client import LLMClient
from app.math.engine import MathEngine
from app.math.result import MathResultHandler
from app.rag.grounded_response import GroundedResponseService
from app.router.classifier import QueryClassifier
from app.router.executor import QueryExecutor
from app.router.router import QueryRouter

from app.graph.nodes.agent_nodes import (
    classify_node,
    execute_node,
    route_node,
)
from app.graph.state import AgentState


def build_graph(
    db: Session,
    grounded_response_service: GroundedResponseService,
    database_service: DatabaseService,
    llm_client: LLMClient,
    math_engine: MathEngine,
    math_result_handler: MathResultHandler,
):
    classifier = QueryClassifier(llm_client)
    router = QueryRouter()

    executor = QueryExecutor(
        grounded_response_service=grounded_response_service,
        database_service=database_service,
        llm_client=llm_client,
        math_engine=math_engine,
        math_result_handler=math_result_handler,
    )

    graph = StateGraph(AgentState)

    graph.add_node(
        "classify",
        lambda state: classify_node(
            state,
            classifier,
        ),
    )

    graph.add_node(
        "route",
        lambda state: route_node(
            state,
            router,
        ),
    )

    graph.add_node(
        "execute",
        lambda state: execute_node(
            state,
            db,
            executor,
        ),
    )

    graph.add_edge(START, "classify")
    graph.add_edge("classify", "route")
    graph.add_edge("route", "execute")
    graph.add_edge("execute", END)

    return graph.compile()