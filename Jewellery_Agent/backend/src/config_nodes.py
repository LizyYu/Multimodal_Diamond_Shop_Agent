from typing import List, Callable, Dict, Any, Optional
from pydantic import BaseModel


class AttributeConfig(BaseModel):
    name: str
    state_key: str
    dependency_keys: List[str]
    valid_options: List[str]
    prompt_template: str

    is_categorical: bool = True
