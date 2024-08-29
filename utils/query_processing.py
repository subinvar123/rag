# from utils.embeddings import get_query_embedding
# from utils.vector_store import search_vectors
# from utils.database import fetch_chunks
# from langchain.chains import RetrievalQA
# from langchain.docstore.document import Document
# from langchain_google_genai import GoogleGenerativeAI
# from dotenv import load_dotenv
# import os
# import streamlit as st
# from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# import streamlit as st

# load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
# llm = GoogleGenerativeAI(model="gemini-pro" , temperature="0.1")

# def process_query(query: str):
    
#     query_vector = get_query_embedding(query)
#     similar_chunk_ids = search_vectors(query_vector, n_results=5)
#     relevant_chunks = fetch_chunks(similar_chunk_ids)
    
#     # Prepare the context by joining the relevant chunks
#     context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
#     #st.write(context)
#     # template = """Use the following pieces of context to answer the question at the end.
#     # If you don't know the answer, just say that you don't know, don't try to make up an answer.
#     # Use three sentences maximum and keep the answer as concise as possible.
#     # Always say "thanks for asking!" at the end of the answer.

#     # {context}

#     # Question: {question}

#     # Helpful Answer:"""

#     template = """You are an advanced assistant for question-answering tasks. Your goal is to provide accurate, comprehensive, and helpful responses based on the given context.

#     Instructions:
#     1. Carefully analyze all pieces of retrieved context provided below.
#     2. Pay special attention to company names, abbreviations, and their full forms.
#     3. Extract and synthesize relevant information to form a coherent and relevant answer to the question.
#     4. If you find any information related to the question, even if it's not a complete answer, include it in your response.
#     5. If you're unsure about any part of your answer, express your level of confidence.
#     6. If you don't find any relevant information, clearly state that you don't have enough information to answer accurately.
#     7. Provide a thorough answer without unnecessary length. Adjust the response length based on the complexity of the question and the available information.
#     8. If appropriate, suggest follow-up questions or additional information that might be helpful.

#     {context}

#     Question: {question}

#     Helpful Answer:"""
#     custom_rag_prompt = PromptTemplate.from_template(template)

#     rag_chain = (
#         {"context": lambda q: context, "question": RunnablePassthrough()}
#         | custom_rag_prompt
#         | llm
#         | StrOutputParser()
#     )
    
#     response = rag_chain.invoke(query)
#     return response


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
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
llm = GoogleGenerativeAI(model="gemini-pro", temperature="0.1")

embeddings_model = GooglePalmEmbeddings()

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


def process_chat_history(query: str):
    query_vector = get_query_embedding(query)
    similar_chunk_ids = search_vectors(query_vector, n_results=5)
    relevant_chunks = fetch_chunks(similar_chunk_ids)
    #st.write(relevant_chunks)
    # Retrieve chat history from the database
    session_id = "1"  # Replace with actual session ID
    chat_history = get_chat_history(session_id)

    vectorstore = Chroma.from_documents(documents=relevant_chunks, embedding=embeddings_model)
    retriever = vectorstore.as_retriever()

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

    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = load_session_history(session_id)
        return store[session_id]

    # Ensure you save the chat history to the database when needed
    def save_all_sessions():
        for session_id, chat_history in store.items():
            for message in chat_history.messages:
                save_message(session_id, message["role"], message["content"])
    

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    save_message(session_id, "human", query)
        
    result = conversational_rag_chain.invoke(
        {"input": query},
        config={"configurable": {"session_id": session_id}}
    )["answer"]

    # Save the AI answer with role "ai"
    save_message(session_id, "ai", result)
    return result
