from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from src.state import AgentState
from dotenv import load_dotenv

import json

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


def _caption_and_clean_message(message, llm):
    """
    Helper: check if a message (Human or AI) has images.
    If yes, uses LLM to caption them and returns a text-only version.
    """
    text_part = ""
    images = []

    if isinstance(message, HumanMessage):
        if isinstance(message.content, list):
            for block in message.content:
                if block.get("type") == "text":
                    text_part += block.get("text", "")
                elif block.get("type") == "image_url":
                    url = block["image_url"]["url"]
                    images.append(url)
        else:
            return None
    elif isinstance(message, AIMessage):
        try:
            data = json.loads(message.content)
            text_part = data.get("response", "")

            raw_imgs = data.get("images", [])
            for img in raw_imgs:
                # images.append(f"data:image/jpeg;base64,{img}")
                images.append(img)
        except (json.JSONDecodeError, TypeError):
            return None

    if not images:
        return None

    vlm_content = [
        {"type": "text", "text": "Describe the jewelry in these images in 1 short sentence or the image is irrevelant."}
    ]
    for img_url in images:
        vlm_content.append({
            "type": "image_url",
            "image_url": {"url": img_url}
        })

    caption_response = llm.invoke([HumanMessage(content=vlm_content)])
    caption = caption_response.content

    if isinstance(message, HumanMessage):
        return f"{text_part} [User showed images of: {caption}]"
    else:
        return f"{text_part} [Agent showed images of: {caption}]"


def santize_previous_ai(state: AgentState):
    """
    Check if PREVIOUS AI message contains JSON with images.
    If yes, ask VLM to describe them, then overwrite with a text summary. 
    """
    messages = state["messages"]
    updates = []

    if len(messages) >= 2:
        last_ai_msg = messages[-2]
        if isinstance(last_ai_msg, AIMessage):
            new_content = _caption_and_clean_message(last_ai_msg, llm)

            if new_content:
                updates.append(
                    AIMessage(id=last_ai_msg.id, content=new_content))

    return {"messages": updates}


def summarize_conversation(state: AgentState):
    """
    Check is history is too long. If so, summarized the first chat turn
    and deletes them from the active list.
    """
    stored_messages = state["messages"]
    # print(f"stored_messages: {stored_messages}")
    current_summary = state.get("summary", "")
    print(f"summary: {current_summary}")

    updates = []
    if len(stored_messages) >= 2:
        current_user_msg = stored_messages[-2]
        if isinstance(current_user_msg, HumanMessage):
            new_content = _caption_and_clean_message(current_user_msg, llm)

            if new_content:
                updates.append(HumanMessage(
                    id=current_user_msg.id, content=new_content))

    if len(stored_messages) <= 6:
        return {"messages": updates}

    to_summarize = stored_messages[:2]

    prompt = f"""
    Current Summary: {current_summary}.\n
    New Lines: {to_summarize}.\n
    Update the summary with the new lines. Keep it concise.
    """

    response = llm.invoke(prompt)
    new_summary = response.content

    for m in to_summarize:
        if m.id:
            updates.append(RemoveMessage(id=m.id))

    return {
        "summary": new_summary,
        "messages": updates
    }
