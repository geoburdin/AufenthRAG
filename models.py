from pydantic import BaseModel
from enum import Enum

class Query(BaseModel):
    question: str
    context: list
    history: list

class Models(str, Enum):
    smart_llm = 'gpt-4o'
    simple_llm = 'gpt-3.5-turbo'

class SearchRequired(BaseModel):
    required: bool
    useModel: Models


class Hallucination(BaseModel):
    reasoning: str
    isHallucination: bool
