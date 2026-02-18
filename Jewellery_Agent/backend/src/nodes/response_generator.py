import json
import requests
import base64

from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from src.state import AgentState
from dotenv import load_dotenv
from src.utils_db import get_unique_values, get_smart_gallery

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

DEPENDENCY_CHAIN = {
    "style": [],
    "material": ["style"],
    "price": ["style", "material"]
}


def generate_no_preference_response(state: AgentState):
    attr_name = state.get("node_name")
    reasoning = state.get("inference_reasoning")

    print("HERE?")
    allowed_dependencies = DEPENDENCY_CHAIN.get(attr_name, [])
    current_filters = {}
    for dep_key in allowed_dependencies:
        if state.get(dep_key):
            current_filters[dep_key] = state[dep_key]

    output_payload = {
        "response": "",
        "images": []
    }

    if attr_name == "price":
        current_options = []
    else:
        current_options = get_unique_values(attr_name)

    print(
        f"current_filters: {current_filters} current_options: {current_options}")
    raw_gallery_items = get_smart_gallery(
        attribute_name=attr_name,
        available_options=current_options,
        current_filters=current_filters,
        limit=3
    )

    final_image_payload = []
    llm_context_list = []

    for index, item in enumerate(raw_gallery_items):
        try:
            image_url = item["image_url"]
            image_response = requests.get(image_url, stream=True, timeout=25)

            if image_response.status_code == 200:
                content_type = image_response.headers.get(
                    "Content-Type", "image/jpg")
                encoded_image = base64.b64encode(
                    image_response.content).decode("utf-8")
                base64_string = f"data:{content_type};base64,{encoded_image}"

                final_image_payload.append(base64_string)

                llm_context_list.append({
                    "index": index,
                    "value": item["value"],
                    "name": item["name"],
                    "price": item.get("actual_price", "N/A")
                })
        except Exception as e:
            print(f"Failed to download image for {item['name']}: {e}")
            continue

    context_str = json.dumps(llm_context_list, indent=2)

    if attr_name == "price":
        goal_instruction = "The user is undecided on budget. Ask for a range."
    else:
        goal_instruction = f"The user is undecided on {attr_name}. Ask them to pick a preference."

    prompt = f"""
    You are a helpful Jewelry Assistant.
    The user is looking for a ring but is undecided about **{attr_name}**.
    
    Here is the data found in the database (Visual Examples):
    {context_str}
    
    INSTRUCTIONS:
    1. Write a natural, engaging response guiding them to choose a {attr_name}. However, the response must be a text + image style like the example output format followed.
    2. {goal_instruction}
    3. When you mention an item, you **MUST** display its image using Markdown format: ![Item Name](image_index).
    4. The 'image_index' comes strictly from the JSON data above.
    
    Example Output Format:
    "We have some lovely options! For a modern feel, take a look at this Platinum Solitaire:
    ![Platinum Ring](image_0)
    
    However, if you prefer something warmer and vintage, this Gold Halo might be better:
    ![Gold Ring](image_1)
    
    Which style speaks to you?"
    """

    response_text = llm.invoke(prompt).content

    return {
        "messages": [AIMessage(content=json.dumps({"response": response_text, "images": final_image_payload}))],
    }


def generate_conflict_response(state: AgentState):
    status = state.get("inference_status")
    reasoning = state.get("inference_reasoning")
    attr_name = state.get("node_name")
    value = state.get(attr_name)

    # This is the "Explanation" message you wanted
    prompt = f"""
    You are a helpful Jewelry Assistant.
    You inferred the user wanted **{attr_name}: {value}** based on this reasoning: "{reasoning}".
    
    However, we DO NOT have any satisfiable items in stock.
    
    Task: Write a message that:
    1. Acknowledges their context (e.g. "Since your friend is a designer...")
    2. Explains you thought {attr_name}: {value} would be perfect.
    3. Apologizes that it is out of stock.
    4. Asks if I am wrong or they would like to see alternatives.
    """
    response_text = llm.invoke(prompt).content
    final_msg = AIMessage(content=response_text)

    return {
        "messages": [final_msg]
    }
