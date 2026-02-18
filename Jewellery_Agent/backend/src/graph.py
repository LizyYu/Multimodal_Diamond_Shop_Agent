from langgraph.graph import StateGraph, START, END
from src.state import AgentState
from src.nodes.memory import summarize_conversation, santize_previous_ai
from src.nodes.intent import run_agent_logic
from langgraph.checkpoint.memory import MemorySaver
from src.nodes.guardrails import check_relevance, refusal_node, greeting_node
from src.nodes.knowledge_router import route_knowledge_retrieval
from src.nodes.retrieve import retrieve_documents
from functools import partial
from src.nodes.generic_inference import run_attribute_inference, run_price_inference
from src.config_nodes import AttributeConfig
from src.configs import style_config, material_config
from src.nodes.response_generator import generate_no_preference_response, generate_conflict_response
from src.nodes.final_response import generate_final_response

infer_style_node = partial(run_attribute_inference, node_config=style_config)
infer_material_node = partial(
    run_attribute_inference, node_config=material_config)


workflow = StateGraph(AgentState)

workflow.add_node("santizer", santize_previous_ai)
workflow.add_node("guardrail", check_relevance)
workflow.add_node("greeting", greeting_node)
workflow.add_node("refusal", refusal_node)
workflow.add_node("knowledge_router", route_knowledge_retrieval)
workflow.add_node("retrieve_documents", retrieve_documents)
workflow.add_node("infer_style", infer_style_node)
workflow.add_node("infer_material", infer_material_node)
workflow.add_node("infer_price", run_price_inference)
workflow.add_node("generate_conflict_response", generate_conflict_response)
workflow.add_node("generate_no_preference", generate_no_preference_response)
workflow.add_node("generate_final_response", generate_final_response)
workflow.add_node("summarizer", summarize_conversation)


def route_intent(state):
    category = state.get("is_relevant", "not_related").lower()

    if category == "greeting":
        return "greeting"
    elif category == "related":
        return "agent"
    else:
        return "refusal"


def knowledge_condition(state: AgentState):
    if state.get("needs_retrieval"):
        return "retrieve_documents"
    else:
        return "agent_logic"


def route_generic(state: AgentState, key, next_node):
    val = state.get(key)
    if not val:
        return END
    return next_node


def route_inference(state: AgentState):
    status = state.get("inference_status")

    if status == "success":
        return "next_node"
    elif status == "no_preference":
        return "generate_no_preference"

    return "generate_conflict_response"


workflow.add_edge(START, "santizer")
workflow.add_edge("santizer", "guardrail")

workflow.add_conditional_edges(
    "guardrail",
    route_intent,
    {
        "greeting": "greeting",
        "agent": "knowledge_router",
        "refusal": "refusal"
    }
)

workflow.add_conditional_edges(
    "knowledge_router",
    knowledge_condition,
    {
        "retrieve_documents": "retrieve_documents",
        "agent_logic": "infer_style"
    }
)
workflow.add_edge("retrieve_documents", "infer_style")
workflow.add_conditional_edges(
    "infer_style",
    route_inference,
    {
        "next_node": "infer_material",
        "generate_no_preference": "generate_no_preference",
        "generate_conflict_response": "generate_conflict_response"
    }
)
workflow.add_conditional_edges(
    "infer_material",
    route_inference,
    {
        "next_node": "infer_price",
        "generate_no_preference": "generate_no_preference",
        "generate_conflict_response": "generate_conflict_response"
    }
)
workflow.add_conditional_edges(
    "infer_price",
    route_inference,
    {
        "next_node": "generate_final_response",
        "generate_no_preference": "generate_no_preference",
        "generate_conflict_response": "generate_conflict_response"
    }

)

workflow.add_edge("greeting", "summarizer")
workflow.add_edge("refusal", "summarizer")
workflow.add_edge("generate_no_preference", "summarizer")
workflow.add_edge("generate_conflict_response", "summarizer")
workflow.add_edge("generate_final_response", "summarizer")
workflow.add_edge("summarizer", END)

memory = MemorySaver()

agent_graph = workflow.compile(checkpointer=memory)

graph_image = agent_graph.get_graph().draw_mermaid_png()
with open("agent_graph.png", "wb") as f:
    f.write(graph_image)
