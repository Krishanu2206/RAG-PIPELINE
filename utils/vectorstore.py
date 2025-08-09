from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
# from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import Pinecone, ServerlessSpec
from utils.llm import get_embeddings
import os 

load_dotenv()

def _ensure_index(pc: Pinecone, name: str):
    existing = [i["name"] for i in pc.list_indexes().get("indexes", [])]
    if name not in existing:
        pc.create_index(
            name=name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    
def initialisevectorstore(namespace:str):
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(PINECONE_API_KEY)
    index_name=os.getenv("PINECONE_INDEX_NAME", "useless")
    _ensure_index(pc, index_name)
    index = pc.Index(index_name)

    vector_store = PineconeVectorStore(embedding=get_embeddings(), index=index, namespace=namespace)

    print(index.describe_index_stats())
    print(vector_store)

    return vector_store, pc

# vector_store, pc = initialisevectorstore()

# print(index.describe_index_stats())
# print(vector_store)

async def store_documents(all_splits, vector_store):
    document_ids = await vector_store.add_documents(documents=all_splits)

    print(document_ids[:3])