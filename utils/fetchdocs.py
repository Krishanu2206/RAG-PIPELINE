import requests
from langchain_community.document_loaders import PyMuPDFLoader

def fetch_docs(doc_url: str, save_path: str="doc.pdf"):
    doc = requests.get(doc_url)

    with open(save_path, "wb") as f:
        f.write(doc.content)

    file_path = save_path
    loader = PyMuPDFLoader(file_path)

    docs = loader.load()
    print(docs[0].page_content)
    return docs