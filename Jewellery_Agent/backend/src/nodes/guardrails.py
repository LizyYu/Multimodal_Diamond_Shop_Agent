from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, AIMessage
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import cast

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


class RelevanceScore(BaseModel):
    category: str = Field(
        description="Must be one of: 'related', 'not_related', or 'greeting'"
    )


structured_llm = llm.with_structured_output(RelevanceScore)


def check_relevance(state):
    messages = state["messages"]
    summary = state.get("summary", "")
    last_user_msg = messages[-1]

    system_prompt = f"""
    You are the Guardrail for a Jewelry Assistant. Classify the user's latest message.

    1. 'greeting': Simple salutations like "Hi", "Hello", "Good morning", "Are you real?", "What can you help me?".
    2. 'related': Questions about jewelry, buying, selling, prices, gemstones, fashion, or style.
    3. 'not_related': Anything else (e.g., coding, weather, cooking, math).

    Current Summary Context: {summary}
    """

    raw_response = structured_llm.invoke(
        [SystemMessage(content=system_prompt), last_user_msg])
    response = cast(RelevanceScore, raw_response)

    return {"is_relevant": response.category}


def greeting_node(state):
    return {
        "messages": [
            AIMessage(content="Hello! I am your Jewelry Assistant. I can help you find the perfect ring, necklace, or answer questions about gemstones. How can I help you today?")
        ]
    }


def refusal_node(state):
    return {
        "messages": [
            AIMessage(
                content="I apologize, but I am specialized in jewelry. I cannot help with that topic. Would you like to see some rings instead?")
        ]
    }
