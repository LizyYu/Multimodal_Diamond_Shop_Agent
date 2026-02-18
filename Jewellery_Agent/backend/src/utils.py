from langchain_core.messages import HumanMessage, AIMessage


def get_conversation_string(messages):
    recent_msgs = messages[:-1]

    history_str = ""

    for msg in recent_msgs:
        if isinstance(msg, HumanMessage):
            role = "User"
        elif isinstance(msg, AIMessage):
            role = "Agent"
        else:
            role = "System"

        history_str += f"{role}: {msg.content}\n"

    return history_str
