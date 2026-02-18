import chromadb
from typing import List, Dict
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from src.state import AgentState
from src.utils import get_conversation_string

db_client = chromadb.PersistentClient(path="./blue_nile_agentic_db")
product_collection = db_client.get_collection(name="product_knowledge")


def get_unique_styles_from_db():
    results = product_collection.get(include=["metadatas"])

    unique_styles = set()
    for meta in results["metadatas"]:
        if meta and "style" in meta:
            s = meta["style"]
            if s and s != "Unknown":
                unique_styles.add(s)

    return list(unique_styles)


VALID_STYLES = get_unique_styles_from_db()


def check_product_availability(filters):
    active_filters = {k: v for k,
                      v in filters.items() if v and str(v).lower() != "none"}

    if len(active_filters) > 1:
        where_clause = {"$and": [{k: v} for k, v in active_filters.items()]}
    else:
        where_clause = active_filters

    results = product_collection.get(where=where_clause, limit=1)
    count = len(results["ids"])

    return {"exists": count > 0, "count": count}


class StyleExtraction(BaseModel):
    identified_styles: List[str] = Field(
        description="The list of styles the user wants. Return ['None'] if undecided."
    )
    reasoning: str = Field(description="Reasoning.")


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


def infer_style_preference(state: AgentState):
    summary = state.get("summary", "")
    messages = state["messages"]
    external_knowledge = state.get("retrieved_images", "no external knowledge")
    conversation_history = get_conversation_string(messages)
    last_user_msg = messages[-1].content

    system_prompt = f"""
    You are a Jewellery Inventory Matcher.
    Analyze the conversation and the provided expert knowledge for the user's STYLE preferences.
    
    ### VALID STYLES IN OUR INVENTORY:
    {VALID_STYLES}
    
    INSTRUCTIONS:
    - Only select styles that appear EXACTLY in the list above.
    - If the user describes a style, map it to the closest match in the list.
    - If the user's request doesn't match ANY style in our list, return ["None"].
    
    EXTERNAL KNOWLEDGE:
    {external_knowledge}
    
    CONTEXT:
    {summary}
    {conversation_history}
    
    CURRENT QUERY:
    {last_user_msg}
    """

    extractor = llm.with_structured_output(StyleExtraction)
    result = extractor.invoke(system_prompt)
    detected_styles = results.idetified_styles
    style_reason = results.reasoning
