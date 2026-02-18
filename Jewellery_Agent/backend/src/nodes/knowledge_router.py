from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from src.state import AgentState
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from src.utils import get_conversation_string

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


class KnowledgeCheck(BaseModel):
    """
    Binary decision on whether to query the expert knowledge database.
    """
    need_external_knowledge: bool = Field(
        description="True if answering user's query needs expert advice, comparisons, or technical details (diamonds, materials). False if it's just a simple preference."
    )
    reasoning: str = Field(
        description="Brief explanation of why knowledge is or isn't need.")


def route_knowledge_retrieval(state: AgentState):
    """
    Analyzes the user's query to decide if we need to fetch documents
    about Diamond 4Cs, Material properties, or Selling guides.
    """
    summary = state.get("summary", "No summary yet")
    messages = state["messages"]

    last_user_msg = messages[-1]
    user_query = last_user_msg.content

    conversation_history = get_conversation_string(messages)

    system_prompt = """
    You are a Senior Jewelry Expert. Your task is to decide if you need to consult the **Technical Knowledge Base** (Qdrant) to reason about the user's requirements.

    ### THE KNOWLEDGE BASE (Reference Only):
    1. **Diamond Economics:** Strategies like "Buying Shy" (0.90ct vs 1.00ct) and price jumps.
    2. **Visual Performance:** How Cut dictates "sparkle" and Color affects "warmth".
    3. **Selling/Business:** (Likely irrelevant for buyer requirements).

    ---

    ### CRITICAL STEP: MEMORY CHECK 
    Before deciding `True`, you MUST check the **CONVERSATION SUMMARY**.
    * **IF** the summary shows that we have *already* retrieved and discussed the relevant strategy (e.g., "Agent already explained the 'Buying Shy' strategy to the user"), **RETURN FALSE**. The Agent already has this knowledge in its short-term memory.
    * **IF** the user is asking a *follow-up* question on the *same* topic (e.g., "Okay, show me examples of that"), **RETURN FALSE**.
    * **ONLY RETURN TRUE** if the user asks about a **NEW** concept (e.g., switching from "Carat" to "Color") or if the previous explanation was insufficient.

    ---

    ### CRITERIA TO RETURN TRUE (Need New Information):
    * **New Reasoning Path:** User switches focus (e.g., was talking about Cut, now asks "Does Color matter for price?"). We need to fetch the *Color* strategy docs.
    * **Specific Trade-off Logic:** User asks for specific value optimization (e.g., "How do I get the biggest stone for $5k?") and we haven't yet retrieved the "Carat vs. Price" guide.
    * **Undefined Visual Terms:** User uses new terms like "Ice" or "Fire" and we haven't mapped these to Cut/Color yet.

    ### CRITERIA TO RETURN FALSE (No Retrieval Needed):
    * **Redundant Info:** The concept (e.g., Buying Shy) is already in the Summary/History.
    * **Pure Material Decisions:** Metals (Gold vs Platinum) - Not in DB.
    * **Simple Preferences:** "I want a 1ct G VS1" (Decision made).
    * **Small Talk/Greetings.**

    Analyze the SUMMARY, CONTEXT, and QUERY to decide.
    """

    user_input = f"""
    --- CONVERSATION SUMMARY ---
    {summary}
    
    --- RECENT CONVERSATION ---
    {conversation_history}
    
    --- CURRENT QUERY ---
    {user_query}
    """

    structed_llm = llm.with_structured_output(KnowledgeCheck)
    decision = structed_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ])

    print(
        f"Knowledge Check: {decision.need_external_knowledge} ({decision.reasoning})")

    return {"needs_retrieval": decision.need_external_knowledge}
