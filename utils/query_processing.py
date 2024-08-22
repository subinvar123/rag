from utils.embeddings import get_query_embedding
from utils.vector_store import search_vectors
from utils.database import fetch_chunks
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import os
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
llm = GoogleGenerativeAI(model="gemini-pro")

def process_query(query: str):
    
    query_vector = get_query_embedding(query)
    similar_chunk_ids = search_vectors(query_vector, n_results=5)
    relevant_chunks = fetch_chunks(similar_chunk_ids)
    
    # Prepare the context by joining the relevant chunks
    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
    #st.write(context)
    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "thanks for asking!" at the end of the answer.

    {context}

    Question: {question}

    Helpful Answer:"""
    custom_rag_prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {"context": lambda q: context, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )
    
    response = rag_chain.invoke(query)
    return response