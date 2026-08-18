from app.router.router import QueryRouter
from app.graph.state import AgentState


def router_node(state: AgentState) -> AgentState:
    classification = state["classification"]

    decision = QueryRouter().route(classification)

    return {
        **state,
        "routing_decision": decision,
    }