
from utils.embeddings import get_query_embedding
from utils.vector_store import search_vectors
from utils.database import fetch_chunks
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_community.embeddings.google_palm import GooglePalmEmbeddings
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from utils.database import store_chunks, fetch_chunks, save_message, load_session_history, fetch_messages
from langchain_chroma import Chroma
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory



load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
llm = GoogleGenerativeAI(model="gemini-pro", temperature="0.1")
embeddings_model = GoogleGenerativeAIEmbeddings(model= "models/embedding-001")
# embeddings_model = GooglePalmEmbeddings()

def get_chat_history(session_id):
    messages = fetch_messages(session_id)
    chat_history = []
    for role, content in messages:
        chat_history.append({"role": role, "content": content})
   
    return chat_history

def process_query(query: str):
    # Get the query embedding and search for similar chunks
    query_vector = get_query_embedding(query)
    similar_chunk_ids = search_vectors(query_vector, n_results=10)
    relevant_chunks = fetch_chunks(similar_chunk_ids)
    
    # Retrieve chat history from the database
    session_id = "1"  # Replace with actual session ID
    chat_history = get_chat_history(session_id)
    
    # Format chat history for context
    formatted_chat_history = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history]
    )
    
    # Combine chat history with relevant chunks
    context = (
        f"Chat History:\n{formatted_chat_history}\n\nRelevant Context:\n" +
        "\n\n".join([chunk.page_content for chunk in relevant_chunks])
    )
    
    # Define the prompt template
    template = """You are an advanced assistant for question-answering tasks. Your goal is to provide accurate, comprehensive, and helpful responses based on the given context.

    {context}

    Question: {question}

    Helpful Answer:"""
    custom_rag_prompt = PromptTemplate.from_template(template)
    
    # Create the Retrieval-Augmented Generation (RAG) chain
    rag_chain = (
        {"context": lambda q: context, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )
    
    # Process the query and get the response
    response = rag_chain.invoke(query)
    
    # Store the current message and response
    save_message(session_id, "user", query)
    save_message(session_id, "assistant", response)
    
    return response










def process_chat_history(query: str, session_id: str):
    # Retrieve chat history from the database
    chat_history = get_chat_history(session_id)

    # Log or print the structure of chat_history to check keys
    # print(chat_history)  # Uncomment this to inspect the structure during debugging

    # Adjust the key based on the actual structure of the messages
    try:
        combined_context = "\n".join([msg['text'] for msg in chat_history]) + "\n" + query
    except KeyError:
        # Fallback if 'text' key doesn't exist, inspect other possible keys (e.g., 'message', 'content', etc.)
        combined_context = "\n".join([msg.get('message', '') for msg in chat_history]) + "\n" + query
    
    # Continue with the rest of the processing
    query_vector = get_query_embedding(combined_context)
    
    # Search for similar chunks in the uploaded PDF or document
    similar_chunk_ids = search_vectors(query_vector, n_results=5)
    relevant_chunks = fetch_chunks(similar_chunk_ids)
    
    # Vector store from documents (PDF chunks or other relevant text)
    vectorstore = Chroma.from_documents(documents=relevant_chunks, embedding=embeddings_model)
    retriever = vectorstore.as_retriever()
    
    # Define prompts to contextualize the follow-up question
    contextualize_q_system_prompt = """Given a chat history and the latest user question 
    which might reference context in the chat history, formulate a standalone question 
    which can be understood without the chat history. Do NOT answer the question, 
    just reformulate it if needed and otherwise return it as is."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    # Create history-aware retriever and chain for follow-up questions
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    qa_system_prompt = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.
    
    {context}"""
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    # Chain to answer based on retrieved documents
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    # Store session chat history for updating context across follow-up questions
    store = {}
    
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = load_session_history(session_id)
        return store[session_id]
    
    # Manage conversational RAG chain with history and update chunks
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    # Save user query to chat history
    save_message(session_id, "human", query)
    
    # Process the query using the conversational RAG chain
    result = conversational_rag_chain.invoke(
        {"input": query},
        config={"configurable": {"session_id": session_id}}
    )["answer"]
    
    # Save AI's answer to the chat history
    save_message(session_id, "ai", result)
    
    return result
