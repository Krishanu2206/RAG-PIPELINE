from utils.ragpipeline import rag_pipeline

def run_rag(documents_url: str, questions: list[str]) -> list[str]:
    """
    Call your RAG pipeline with the documents and questions.
    Returns list of answers in the same order.
    """
    answers = rag_pipeline(documents_url, questions)
    return answers
