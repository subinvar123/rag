from utils.embeddings import get_query_embedding
from utils.vector_store import search_vectors ,get_retriever
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
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from utils.database import save_message ,fetch_messages


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
llm = GoogleGenerativeAI(model="gemini-pro", temperature="0.1")
embeddings_model = GoogleGenerativeAIEmbeddings(model= "models/embedding-001")
### Statefully manage chat history ###
store = {}

def get_chat_history(session_id):
    messages = fetch_messages(session_id)
    chat_history = []
    for role, content in messages:
        chat_history.append({"role": role, "content": content})
   
    return chat_history
    st.write(chat_history)

def process_query(query: str, session_id: str):
    # query_vector = get_query_embedding(query)
    
    retriever = get_retriever()
    chat_history = get_chat_history(session_id)
    
    ### Contextualize question ###
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )


    ### Answer question ###
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    session_id=1
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]


    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    # Save user query to chat history
    save_message(session_id, "human", query)

    response = conversational_rag_chain.invoke(
        {"input": query},
        config={"configurable": {"session_id": "session1"}},
    )["answer"]
   
    return response