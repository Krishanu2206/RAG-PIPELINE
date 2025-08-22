import requests
from langchain_community.document_loaders import PyMuPDFLoader
import bs4
from langchain_community.document_loaders import WebBaseLoader
from weasyprint import HTML
import tempfile
import os

def fetch_docs(doc_url: str, save_path: str="doc.pdf"):
    doc = requests.get(doc_url)

    with open(save_path, "wb") as f:
        f.write(doc.content)

    file_path = save_path
    loader = PyMuPDFLoader(file_path)

    docs = loader.load()
    print(docs[0].page_content)
    return docs

def scrap_docs(doc_url: str, save_path: str="scraped_doc.pdf"):
    """
    Scrapes a document from a URL and converts it to PDF format.
    
    Args:
        doc_url (str): The URL of the document to scrape
        save_path (str): Path where to save the generated PDF
        
    Returns:
        list: List of loaded documents
    """
    # Create a strainer to only keep relevant content
    bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
    
    # Load the web content
    loader = WebBaseLoader(
        web_paths=(doc_url,),
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()
    
    if not docs:
        print("No content found to scrape")
        return []
    
    # Get the scraped content
    content = docs[0].page_content
    
    # Create HTML content with basic styling
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Scraped Document</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            h1, h2, h3 {{ color: #333; }}
            p {{ margin-bottom: 16px; }}
        </style>
    </head>
    <body>
        {content}
    </body>
    </html>
    """
    
    # Convert HTML to PDF using WeasyPrint
    try:
        HTML(string=html_content).write_pdf(save_path)
        print(f"PDF saved successfully to: {save_path}")
        
        # Load the generated PDF using PyMuPDFLoader
        loader = PyMuPDFLoader(save_path)
        pdf_docs = loader.load()
        
        print(f"Total characters in scraped content: {len(content)}")
        print(f"PDF loaded successfully with {len(pdf_docs)} pages")
        
        return pdf_docs
        
    except Exception as e:
        print(f"Error converting to PDF: {e}")
        return docs
