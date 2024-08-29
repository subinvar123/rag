import os
from typing import Union
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
import pandas as pd
import tempfile
 
def load_document(file_path: Union[str, os.PathLike]) -> list[Document]:
    """
    Load a document from a file path.
   
    Args:
        file_path (Union[str, os.PathLike]): The path to the document file.
   
    Returns:
        list[Document]: A list of Document objects containing the loaded content.
   
    Raises:
        ValueError: If the file type is not supported.
    """
    file_extension = os.path.splitext(file_path)[1].lower()
   
    if file_extension == '.pdf':
        loader = PyPDFLoader(file_path)
        return loader.load()
    elif file_extension == '.txt':
        loader = TextLoader(file_path)
        return loader.load()
    elif file_extension in ['.xls', '.xlsx']:
        # Load Excel file content
        df = pd.read_excel(file_path)
        # Convert the DataFrame to a string representation
        text = df.to_string()
        # Return as a list of Document objects
        return [Document(page_content=text)]
    elif file_extension == '.docx':
        # Load DOCX file content
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        # Return as a list of Document objects
        return [Document(page_content=text)]
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")
    # file_extension = os.path.splitext(file_path)[1].lower()
   
    # if file_extension == '.pdf':
    #     loader = PyPDFLoader(file_path)
    #     return loader.load()
    # elif file_extension == '.txt':
    #     loader = TextLoader(file_path)
    #     return loader.load()
    # else:
    #     raise ValueError(f"Unsupported file type: {file_extension}")
 
def load_document_from_uploadedfile(uploaded_file) -> list[Document]:
    """
    Load a document from a Streamlit UploadedFile object.
   
    Args:
        uploaded_file (streamlit.UploadedFile): The uploaded file object from Streamlit.
   
    Returns:
        list[Document]: A list of Document objects containing the loaded content.
   
    Raises:
        ValueError: If the file type is not supported.
    """
     # Create a temporary file to save the uploaded content
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
 
    try:
        # Load the document using the temporary file path
        documents = load_document(tmp_file_path)
    finally:
        # Clean up the temporary file
        os.unlink(tmp_file_path)
 
    return documents