import json
import base64
import requests
import chromadb
from typing import List, Dict, Any, Optional, Union
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from src.state import AgentState
from src.utils_db import (
    product_collection,
    visual_collection,
    get_unique_values,
    get_smart_gallery,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from src.utils import get_conversation_string
from sentence_transformers import SentenceTransformer

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

client = chromadb.PersistentClient(path="./blue_nile_agentic_db")
product_collection = client.get_collection(name="product_knowledge")
visual_collection = client.get_collection(name="visual_index")


def generate_vector_search_query(state: AgentState):
    """
    Asks the LLM to rewrite the conversation context into a 
    clean, visual search query for the vector database.
    """
    summary = state.get("summary", "")
    context_str = get_conversation_string(state["messages"])
    last_msg = state["messages"][-1].content

    # Collect known attributes to refine the query
    attributes = []
    if state.get("style"):
        attributes.append(f"Style: {state['style']}")
    if state.get("material"):
        attributes.append(f"Material: {state['material']}")
    if state.get("price"):
        attributes.append(f"Price Range: {state['price']}")
    attr_str = ", ".join(attributes)

    prompt = f"""
    You are an expert Search Query Optimizer for a Jewelry Database.
    
    CONTEXT:
    User Summary: {summary}
    Conversation history: {context_str}
    Current Request: {last_msg}
    Inferred Attributes: {attr_str}
    
    TASK:
    Write a concise, descriptive search query to find the best matching jewelry images.
    Focus on visual keywords (e.g., "Vintage Halo Ring Rose Gold", "Solitaire Diamond Platinum").
    Do not include explanations, just the query string.
    """

    return llm.invoke(prompt).content.strip()


def generate_final_response(state: AgentState):
    """
    Final node: 
    1. Diagnoses filters (Availability Check).
    2. Generates visual search query.
    3. Fetches items, visualizes them, and returns final payload.
    """

    active_filters = {}

    if state.get("style"):
        val = state["style"]
        if isinstance(val, list):
            active_filters["style"] = {"$in": val} if len(val) > 1 else val[0]
        else:
            active_filters["style"] = val

    if state.get("material"):
        val = state["material"]
        if isinstance(val, list):
            active_filters["material"] = {
                "$in": val} if len(val) > 1 else val[0]
        else:
            active_filters["material"] = val

    raw_price = state.get("price")
    if raw_price and isinstance(raw_price, str):
        price_query = {}
        try:
            if "+" in raw_price:
                min_val = float(raw_price.replace("+", "").strip())
                price_query["$gte"] = min_val
            elif "-" in raw_price:
                parts = raw_price.split("-")
                price_query["$gte"] = float(parts[0].strip())
                price_query["$lte"] = float(parts[1].strip())
            else:
                pass

            if price_query:
                active_filters["price"] = price_query
        except ValueError:
            print(f"Error parsing price string: {raw_price}")

    all_rules = []
    for k, v in active_filters.items():
        if k == "price" and isinstance(v, dict) and len(v) > 1:
            all_rules.append({"price": {"$gte": v.get("$gte")}})
            all_rules.append({"price": {"$lte": v.get("$lte")}})
        else:
            all_rules.append({k: v})

    strict_where = None
    if len(all_rules) > 1:
        strict_where = {"$and": all_rules}
    elif len(all_rules) == 1:
        strict_where = all_rules[0]

    try:
        if strict_where:
            results = product_collection.get(
                where=strict_where, include=["metadatas"])
        else:
            results = product_collection.get(include=["metadatas"])
        valid_ids = results["ids"]
    except Exception as e:
        print(f"Filter Error: {e}")
        valid_ids = []

    vector_search_query = generate_vector_search_query(state)
    print(f"Generated Vector Query: {vector_search_query}")

    embedding_model = SentenceTransformer("clip-ViT-B-32")
    query_vector = embedding_model.encode(vector_search_query).tolist()

    search_args = {
        "query_embeddings": [query_vector],
        "n_results": 20,
        "include": ["metadatas", "distances"],
        "where": {"parent_id": {"$in": valid_ids}}
    }

    visual_results = visual_collection.query(**search_args)

    final_items = []
    seen_products = set()

    if visual_results["metadatas"] and visual_results["metadatas"][0]:
        metas = visual_results["metadatas"][0]
        for meta in metas:
            p_id = meta["parent_id"]
            if p_id in seen_products:
                continue

            seen_products.add(p_id)
            final_items.append(meta)
            if len(final_items) >= 5:
                break  # Top 5 results

    image_gallery = []
    lean_context = []

    for index, item in enumerate(final_items):
        try:
            image_url = item["image_url"]
            resp = requests.get(image_url, stream=True, timeout=5)

            if resp.status_code == 200:
                ctype = resp.headers.get("Content-Type", "image/jpg")
                b64_img = base64.b64encode(resp.content).decode("utf-8")

                # A. Frontend List
                image_gallery.append(f"data:{ctype};base64,{b64_img}")

                # B. LLM Context
                lean_context.append({
                    "index": index,
                    "name": item.get("name"),
                    "price": item.get("price"),
                    "description": f"{item.get('style', 'Ring')} in {item.get('material', 'Metal')}"
                })
        except Exception as e:
            print(f"Image Error: {e}")
            continue

    # --- STEP 6: FINAL PROMPT ---
    context_str = json.dumps(lean_context, indent=2)

    system_prompt = f"""
    You are a helpful Jewellery Shopping Assistant.
    
    USER QUERY: "{state['messages'][-1].content}"
    SEARCH CONTEXT: The user wanted {vector_search_query}.
    
    I have found the following items in stock:
    {context_str}
    
    INSTRUCTIONS:
    1. Write a natural, engaging response recommending these items.
    2. Compare them (e.g., "The first one is more classic, while the second...")
    3. You **MUST** display images using: ![Item Name](image_INDEX).
    4. The 'INDEX' corresponds to the 'index' field in the data above.
    
    Example:
    "I found this stunning option for you:
    ![Ring](image_0)
    
    And if you want to save budget..."
    """

    llm_output = llm.invoke(system_prompt).content

    final_payload = json.dumps({
        "response": llm_output,
        "images": image_gallery
    })

    return {
        "messages": [AIMessage(content=final_payload)],
    }
