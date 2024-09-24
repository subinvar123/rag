import chromadb
import uuid
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

client = chromadb.PersistentClient(path="./chroma_db")
collection_name = "doc_collection"
collection = client.get_or_create_collection(name=collection_name)

embeddings = GoogleGenerativeAIEmbeddings(model= "models/embedding-001")


vector_store = Chroma(
    collection_name=collection_name,
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)


def store_vectors(documents):
    
    uuids = [str(uuid.uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=uuids)
 

def search_vectors(query, n_results):
    results = vector_store.similarity_search_by_vector(
        embedding=embeddings.embed_query(query), k=n_results
    )
    return results


def get_retriever():
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 5}
    )
    return retriever