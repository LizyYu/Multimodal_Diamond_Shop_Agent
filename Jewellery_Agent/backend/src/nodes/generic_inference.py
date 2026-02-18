from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from src.state import AgentState
from src.utils import get_conversation_string
from src.utils_db import check_product_availability
from src.config_nodes import AttributeConfig
from pydantic import BaseModel, Field
from typing import List, Optional


class GenericExtraction(BaseModel):
    identified_values: List[str] = Field(
        description="The identified preferences. Returns ['None'] if undecided.")
    reasoning: str = Field(
        description="Explain the logic of how you infer user's preference given the conversation with/without the external knowledge"
    )


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


def run_attribute_inference(state: AgentState, node_config: AttributeConfig):
    """
    Generic logic: Extraction -> Avalilabity Check
    """
    summary = state.get("summary", "")
    messages = state["messages"]

    last_user_msg = messages[-1].content
    context_str = get_conversation_string(messages)
    context_pages = state.get("retrieved_images", "no external knowledge")

    system_prompt = f"""
    You are a Jewellery Inventory Matcher.
    Analyze the conversation for the user's preference regarding: **{node_config.name}**.

    ### VALID OPTIONS:
    {node_config.valid_options}

    {node_config.prompt_template}

    EXTERNAL KNOWLEDGE:
    {context_pages}

    CONTEXT:
    {summary}
    {context_str}

    CURRENT QUERY
    {last_user_msg}

    Return the EXACT values from the list. If undecided, return ["None"].
    """

    extractor = llm.with_structured_output(GenericExtraction)
    result = extractor.invoke(system_prompt)
    detected_values = result.identified_values

    print(f"prompt: {system_prompt}")
    print(f"intent reasoning: {result.reasoning}")
    print(f"detected values: {detected_values}")

    existing_constraints = {}
    for key in node_config.dependency_keys:
        val = state.get(key)
        existing_constraints[key] = val

    # CASE 1: no valid attributed can be infered

    clean_values = [
        v for v in detected_values if v in node_config.valid_options]

    if not clean_values:
        return {
            "inference_status": "no_preference",
            "inference_reasoning": result.reasoning,
            "node_name": node_config.name,
            node_config.state_key: None
        }

    messages_to_user = []

    current_db_field = node_config.name.lower()

    check_filter = existing_constraints.copy()
    check_filter[current_db_field] = clean_values

    availability = check_product_availability(check_filter)

    # CASE 2: no product exists for the current combination
    if not availability["exists"]:
        return {
            "inference_status": "invalid_inference",
            "inference_reasoning": result.reasoning,
            "node_name": node_config.name,
            node_config.state_key: None
        }

    # CASE 3: infer the attribute successfully
    return {
        "inference_status": "success",
        "inference_reasoning": result.reasoning,
        node_config.name: clean_values,
    }


class PriceExtraction(BaseModel):
    """
    Extracts budget constraints from user text.
    """
    min_price: Optional[float] = Field(
        description="The minimum price budget, 0 if not specified")
    max_price: Optional[float] = Field(
        description="The maximum price budget, None if no limit")
    is_mentioned: bool = Field(
        description="True if the user explicitly mentioned price/budget")
    reasoning: str = Field(
        description="Explain the logic of how you infer user's preference given the conversation with/without the external knowledge"
    )


def run_price_inference(state: AgentState):
    """
    Dedicated logic for extracting and validating Price/Budget.
    """
    messages = state["messages"]
    last_user_msg = messages[-1].content
    summary = state.get("summary", "")
    context_str = get_conversation_string(messages)
    context_pages = state.get("retrieved_images", "no external knowledge")

    # 1. SPECIALIZED PROMPT FOR NUMBERS
    system_prompt = f"""
    You are a Jewelry Sales Expert.
    Analyze the conversation to detect the user's **Budget/Price Range**.
    
    RULES:
    - "Under X" -> min: 0, max: X
    - "Over X" -> min: X, max: None
    - "Between X and Y" -> min: X, max: Y
    - "Around X" -> +/- 20% of X
    - "Cheap / Affordable" -> Set max to 1000 (Soft limit)
    - "Luxury / Expensive" -> Set min to 5000 (Soft limit)
    - If no budget is mentioned, set is_mentioned = False.
    
    EXTERNAL KNOWLEDGE:
    {context_pages}
    
    CONTEXT:
    {summary}
    {context_str}
    
    CURRENT QUERY:
    {last_user_msg}
    """

    extractor = llm.with_structured_output(PriceExtraction)
    result = extractor.invoke(system_prompt)

    print(f"price reasoning: {result.reasoning}")

    # 2. HANDLE NO PREFERENCE
    if not result.is_mentioned:
        return {
            "inference_status": "no_preference",
            "inference_reasoning": result.reasoning,
            "node_name": "price",
            "price": None
        }

    price_filter = {}
    if result.min_price is not None:
        price_filter["min"] = result.min_price
    if result.max_price is not None:
        price_filter["max"] = result.max_price

    check_filter = {}
    if state.get("style"):
        check_filter["style"] = state["style"]
    if state.get("material"):
        check_filter["material"] = state["material"]

    check_filter["price"] = price_filter

    availability = check_product_availability(check_filter)

    if not availability["exists"]:
        return {
            "inference_status": "invalid_inference",
            "inference_reasoning": result.reasoning,
            "node_name": "price",
            "price": None
        }

    if result.max_price:
        price_str = f"{result.min_price or 0}-{result.max_price}"
    else:
        price_str = f"{result.min_price}+"

    return {
        "inference_status": "success",
        "inference_reasoning": result.reasoning,
        "price": price_str
    }
