
from utils.embeddings import get_query_embedding
from utils.vector_store import search_vectors
from utils.database import fetch_chunks, save_message, load_session_history, fetch_messages
from langchain.chains import RetrievalQA, create_history_aware_retriever, create_retrieval_chain
from langchain.docstore.document import Document
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_community.embeddings.google_palm import GooglePalmEmbeddings
import os
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain.memory import ChatMessageHistory
from langchain.chains import LLMChain
from langchain.chains import LLMChain, RetrievalQA
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory
from langchain.schema import HumanMessage, AIMessage
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.chains import DocumentChain


# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Initialize models
llm = GoogleGenerativeAI(model="gemini-pro", temperature="0.1")
embeddings_model = GooglePalmEmbeddings()

# Initialize an in-memory chat history
demo_ephemeral_chat_history = ChatMessageHistory()

def get_chat_history(session_id):
    messages = fetch_messages(session_id)
    chat_history = ChatMessageHistory()
    for role, content in messages:
        if role == "user":
            chat_history.add_user_message(content)
        elif role == "assistant" or role == "ai":
            chat_history.add_ai_message(content)
    return chat_history

def summarize_messages():
    stored_messages = demo_ephemeral_chat_history.messages
    if len(stored_messages) == 0:
        return False
    
    summarization_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "Distill the above chat messages into a single summary message. Include as many specific details as you can."),
        ]
    )
    summarization_chain = summarization_prompt | llm
    summary_message = summarization_chain.invoke({"chat_history": stored_messages})
    
    demo_ephemeral_chat_history.clear()
    demo_ephemeral_chat_history.add_message(summary_message)
    
    return True

def process_query(query: str):
    # Get the query embedding and search for similar chunks
    query_vector = get_query_embedding(query)
    similar_chunk_ids = search_vectors(query_vector, n_results=10)
    relevant_chunks = fetch_chunks(similar_chunk_ids)
    
    # Retrieve chat history and update in-memory history
    session_id = "1"  # Replace with actual session ID
    chat_history = get_chat_history(session_id)
    demo_ephemeral_chat_history = chat_history

    # Define the prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Answer all questions to the best of your ability. The provided chat history includes facts about the user you are speaking with."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ]
    )
    
    chain = prompt | llm

    chain_with_message_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: demo_ephemeral_chat_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    # Save the user query
    save_message(session_id, "user", query)
    
    # Process the query and get the response
    result = chain_with_message_history.invoke(
        {"input": query},
        {"configurable": {"session_id": "unused"}}
    )["answer"]

    # Save the AI answer
    save_message(session_id, "ai", result)
    return result

# Fetch chat history from database and convert to in-memory history
def get_chat_history(session_id):
    messages = fetch_messages(session_id)
    chat_history = []
    for role, content in messages:
        if role == "user":
            chat_history.append(HumanMessage(content=content))
        elif role == "ai":
            chat_history.append(AIMessage(content=content))
    return chat_history

# Process query using RetrievalQA and LLMChain
def process_chat_history(query: str):
    # Get query embedding and search for similar chunks
    query_vector = get_query_embedding(query)
    similar_chunk_ids = search_vectors(query_vector, n_results=5)
    relevant_chunks = fetch_chunks(similar_chunk_ids)

    # Create in-memory chat history
    session_id = "1"  # Replace with actual session ID
    chat_history_messages = get_chat_history(session_id)
    demo_ephemeral_chat_history = ChatMessageHistory(messages=chat_history_messages)

    # Define the QA prompt template
    qa_system_prompt = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."""

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Create LLM chain with the QA prompt
    llm_chain = LLMChain(llm=llm, prompt=qa_prompt)

    # Create a vectorstore and retriever
    vectorstore = Chroma.from_documents(documents=relevant_chunks, embedding=embeddings_model)
    retriever = vectorstore.as_retriever()

    # Create the RetrievalQA chain
    retrieval_chain = RetrievalQA(
        combine_documents_chain=llm_chain,  # Use LLMChain directly for document combination
        retriever=retriever
    )

    # Create RunnableWithMessageHistory for managing chat history
    chain_with_message_history = RunnableWithMessageHistory(
        retrieval_chain,
        lambda session_id: demo_ephemeral_chat_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    # Save the user query
    save_message(session_id, "user", query)
    
    # Process the query and get the response
    result = chain_with_message_history.invoke(
        {"input": query},
        {"configurable": {"session_id": "unused"}}
    )["answer"]

    # Save the AI answer
    save_message(session_id, "ai", result)
    return result