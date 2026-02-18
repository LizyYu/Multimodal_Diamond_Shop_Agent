from typing import TypedDict, Annotated, List, Optional, Union
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from typing_extensions import NotRequired


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    summary: NotRequired[str]
    needs_retrieval: bool
    retrieved_images: List[str]

    inference_status: Optional[str]
    inference_reasoning: Optional[str]
    node_name: Optional[str]

    style: Optional[Union[str, List[str]]]
    material: Optional[Union[str, List[str]]]
    price: Optional[Union[str, List[str]]]
