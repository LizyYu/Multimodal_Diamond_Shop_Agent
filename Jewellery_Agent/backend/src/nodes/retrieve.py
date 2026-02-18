from src.state import AgentState
from io import BytesIO
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from src.vector_store import VisualRetriever
from src.utils import get_conversation_string

import base64

retriever = VisualRetriever()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


def retrieve_documents(state: AgentState):
    summary = state.get("summary", "")
    messages = state["messages"]

    last_user_msg = messages[-1].content
    conversation_history = get_conversation_string(messages)

    query_prompt = f"""
    CONTEXT: {summary}
    RECENT CONVERSATION: {conversation_history}
    USER QUERY: {last_user_msg}
    
    Task: Write a concise search query to find relevant pages in a Jewelry Technical Manual.
    Example: "Diamond cut grading chart" or "Pricing strategy for 0.90 carat"
    """
    search_query = llm.invoke(query_prompt).content.strip()

    pil_images = retriever.retrieve_context_pages(search_query, k=1)

    encoded_pages = []

    if pil_images:
        for img in pil_images:
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            encoded_pages.append(img_b64)

    return {
        "retrieved_images": encoded_pages,
        "needs_retrieval": False
    }
