from pydantic import BaseModel
from enum import Enum
import os
import dotenv
dotenv.load_dotenv()
class Query(BaseModel):
    question: str
    context: list
    history: list

class Models(str, Enum):
    smart_llm = os.getenv("SMART_LLM", "gpt-4o-mini")
    simple_llm = os.getenv("SIMPLE_LLM", "gpt-3.5-turbo")

class SearchRequired(BaseModel):
    required: bool
    useModel: Models


class Hallucination(BaseModel):
    reasoning: str
    isHallucination: bool
