from pydantic import BaseModel, HttpUrl
from typing import List

class RunRequest(BaseModel):
    documents: str
    questions: List[str]

class RunResponse(BaseModel):
    answers: List[str]
