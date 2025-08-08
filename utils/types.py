from typing import Literal, Annotated
from langchain_core.documents import Document
from typing import List, TypedDict

class Search(TypedDict):
    """Search query."""

    query: Annotated[str, ..., "Search query to run."]
    section: Annotated[
        Literal["beginning", "middle", "end"],
        ...,
        "Section to query.",
    ]

class State(TypedDict):
    question: str
    id: int
    vector_store: object
    pc: object
    query: Search
    context: List[Document]
    final_context: List[Document]
    answer: str