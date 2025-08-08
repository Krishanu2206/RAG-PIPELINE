from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY1 = os.getenv("GOOGLE_API_KEY1")
GOOGLE_API_KEY2 = os.getenv("GOOGLE_API_KEY2")
GOOGLE_API_KEY3 = os.getenv("GOOGLE_API_KEY3")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY1,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

llm2 = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY2,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

llm3 = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY3,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# from langchain_ollama import ChatOllama

# llm = ChatOllama(
#     model="qwen3:4b",
#     temperature=0,
#     # other params...
# )

## embedding model
def get_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return embeddings

    # from langchain_ollama import OllamaEmbeddings
    # embeddings = OllamaEmbeddings(model="nomic-embed-text")
