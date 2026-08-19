from sqlalchemy.orm import Session

from app.llm.client import LLMClient
from app.math.engine import MathEngine
from app.math.result import MathResultHandler
from app.database.service import DatabaseService
from app.rag.grounded_response import GroundedResponseService
from app.router.classifier import QueryClassifier
from app.router.executor import QueryExecutor
from app.router.router import QueryRouter
from app.graph.state import AgentState


def classify_node(
    state: AgentState,
    classifier: QueryClassifier,
) -> AgentState:

    state["classification"] = classifier.classify(
        state["user_message"]
    )

    return state


def route_node(
    state: AgentState,
    router: QueryRouter,
) -> AgentState:

    state["routing_decision"] = router.route(
        state["classification"]
    )

    return state


def execute_node(
    state: AgentState,
    db: Session,
    executor: QueryExecutor,
) -> AgentState:

    result = executor.execute(
        db=db,
        knowledge_base_id=state["knowledge_base_id"],
        query=state["user_message"],
        decision=state["routing_decision"],
        persona=state["persona"],
    )

    state["result"] = result

    return state