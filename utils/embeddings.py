from langchain_community.embeddings.google_palm import GooglePalmEmbeddings
from langchain.docstore.document import Document
from dotenv import load_dotenv
from typing import List, Union
import os


# Make sure to set your Google Palm API key in your environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_PALM_API_KEY"] = GOOGLE_API_KEY

embeddings_model = GooglePalmEmbeddings()

def get_embeddings(texts: Union[List[str], List[Document]]) -> List[List[float]]:
    if isinstance(texts[0], Document):
        # If input is a list of Documents, extract the page_content
        texts = [doc.page_content for doc in texts]
    
    # Generate embeddings
    embeddings = embeddings_model.embed_documents(texts)
    return embeddings

def get_query_embedding(query: str) -> List[float]:
    
    return embeddings_model.embed_query(query)