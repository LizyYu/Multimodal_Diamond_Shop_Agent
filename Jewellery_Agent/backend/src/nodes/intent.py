from src.state import AgentState
from langchain_core.messages import SystemMessage, AIMessage


def run_agent_logic(state: AgentState):
    return {
        "messages": [
            AIMessage(
                content="I am an agent?")
        ]
    }
