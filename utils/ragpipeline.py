from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.fetchdocs import fetch_docs
from utils.llm import llm, llm2, llm3
from utils.prompt import prompt
from utils.vectorstore import initialisevectorstore, store_documents
from utils.types import State, Search
from langgraph.graph import START, StateGraph
# from IPython.display import Image, display
import time, hashlib, os

def analyze_query(state: State):
    llm_choice = [llm, llm2, llm3][state["id"] % 3]
    structured_llm = llm_choice.with_structured_output(Search)
    query = structured_llm.invoke(state["question"])
    return {"query": query}

def retrieve(state: State):
    vector_store = state["vector_store"]
    query = state["query"]
    pinecone_filter = {"section": query["section"]}
    retrieved_docs = vector_store.similarity_search(
        query["query"],
        k=8,
        filter=pinecone_filter,
    )
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(item["document"]["chunk_text"] for item in state["final_context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    llm_choice = [llm, llm2, llm3][state["id"] % 3]
    response = llm_choice.invoke(messages)
    return {"answer": response.content}

def reranker(state : State):

    pc=state["pc"]

    documents = []
    for doc in state["context"]:
        documents.append({"id": doc.id, "chunk_text": doc.page_content})

    ranked_results = pc.inference.rerank(
        model="bge-reranker-v2-m3",
        query=state["question"],
        documents=documents,
        top_n=min(4, len(documents)),
        rank_fields=["chunk_text"],
        return_documents=True,
        parameters={
            "truncate": "END"
        }
    )

    print(ranked_results)

    return {"final_context" : ranked_results.data}

def rag_pipeline(doc_url:str, questions:list[str]) -> list[str]:

    docs = fetch_docs(doc_url)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # chunk size (characters)
        chunk_overlap=200,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )
    all_splits = text_splitter.split_documents(docs)

    print(f"Split blog post into {len(all_splits)} sub-documents.")

    total_documents = len(all_splits)
    third = total_documents // 3

    for i, document in enumerate(all_splits):
        if i < third:
            document.metadata["section"] = "beginning"
        elif i < 2 * third:
            document.metadata["section"] = "middle"
        else:
            document.metadata["section"] = "end"


    print(all_splits[0].metadata)

    namespace = hashlib.sha1(doc_url.encode("utf-8")).hexdigest()[:12]
    vector_store, pc = initialisevectorstore(namespace=namespace)

    store_documents(all_splits, vector_store)

    graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, reranker, generate])
    graph_builder.add_edge(START, "analyze_query")
    graph = graph_builder.compile()

    # display(Image(graph.get_graph().draw_mermaid_png()))

    answers=[]

    timer_start = time.time()

    for i, question in enumerate(questions):

        result = graph.invoke({"id":i, "question": question, "vector_store": vector_store, "pc": pc})

        print(f"Context of {i}: {result['context']}\n\n")
        print(f"Answer of {i}: {result['answer']}\n")
        answers.append(result["answer"])

    timer_end = time.time()

    print(f"Inference time: {timer_end - timer_start:.3f} seconds")

    return answers