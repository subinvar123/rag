import streamlit as st
from utils.document_loader import load_document_from_uploadedfile
from utils.text_splitter import split_text
from utils.embeddings import get_embeddings
from utils.vector_store import store_vectors
from utils.query_processing import process_query
from utils.database import store_chunks, fetch_chunks
from langchain.chains import RetrievalQA
import os
 
 
def main():
    st.set_page_config(layout="wide")
   
    # Sidebar for document upload
    with st.sidebar:
        st.title("Document Upload")
        uploaded_file = st.file_uploader("Choose a document", type=["txt", "pdf"])
        
        if uploaded_file is not None:
            if st.button("Process Document"):
                with st.spinner("Processing document..."):
                    # Document processing pipeline
                    doc = load_document_from_uploadedfile(uploaded_file)
                    chunks = split_text(doc)
                    vectors = get_embeddings(chunks)
                    chunk_ids = store_vectors(chunks, vectors)
                    store_chunks(chunks, chunk_ids)
                st.success("Document processed successfully!")
    # Main chat interface
    st.title("RAG - IR System")
 
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
 
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
 
    # React to user input
    if prompt := st.chat_input("What is your question?"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
 
        # Process the query and generate response
        with st.spinner("Generating response..."):
            response = process_query(prompt)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
 
if __name__ == "__main__":
    main()