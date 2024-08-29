# import psycopg2
# from psycopg2.extras import execute_batch
# import json
# from dotenv import load_dotenv
# import os
# from langchain_core.documents import Document


# load_dotenv()

# DB_PARAMS = {
#     "dbname": os.getenv("POSTGRES_DB"),
#     "user": os.getenv("POSTGRES_USER"),
#     "password": os.getenv("POSTGRES_PASSWORD"),
#     "host": os.getenv("POSTGRES_HOST"),
#     "port": os.getenv("POSTGRES_PORT")
# }

# def get_db_connection():
#     return psycopg2.connect(**DB_PARAMS)

# def store_chunks(chunks, chunk_ids):
#     conn = get_db_connection()

#     with conn.cursor() as cur:
#         # Create table if not exists
#         cur.execute("""
#             CREATE TABLE IF NOT EXISTS document_chunks (
#                 id TEXT PRIMARY KEY,
#                 content TEXT,
#                 metadata JSONB
#             )
#         """)
        
#         # Prepare data for insertion
#         #2 arrays are got chunk_ids, chunks these arrays are made one using zip
#         #For example, if chunk_ids = [1, 2, 3] and chunks = [chunk1, chunk2, chunk3], then zip(chunk_ids, chunks) will produce [(1, chunk1), (2, chunk2), (3, chunk3)].
#         data = [(id, chunk.page_content, json.dumps(chunk.metadata)) for id, chunk in zip(chunk_ids, chunks)]
        
#         # Insert data
#         execute_batch(cur, """
#             INSERT INTO document_chunks (id, content, metadata)
#             VALUES (%s, %s, %s)
#             ON CONFLICT (id) DO UPDATE
#             SET content = EXCLUDED.content, metadata = EXCLUDED.metadata
#         """, data)
    
#     conn.commit()
#     conn.close()

# def fetch_chunks(chunk_ids):
#     conn = get_db_connection()
#     with conn.cursor() as cur:
#         placeholders = ','.join(['%s'] * len(chunk_ids))
#         cur.execute(f"SELECT id, content, metadata FROM document_chunks WHERE id IN ({placeholders})", chunk_ids)
#         results = cur.fetchall()
    
#     chunks = [Document(page_content=content, metadata=metadata) for _, content, metadata in results]
#     conn.close()
#     return chunks


import psycopg2
from psycopg2.extras import execute_batch
import json
from dotenv import load_dotenv
import os
from langchain_core.documents import Document
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy.exc import SQLAlchemyError
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

load_dotenv()

DB_PARAMS = {
    "dbname": os.getenv("POSTGRES_DB"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "host": os.getenv("POSTGRES_HOST"),
    "port": os.getenv("POSTGRES_PORT")
}

# SQLAlchemy setup for chat history
DATABASE_URL = f"postgresql://{DB_PARAMS['user']}:{DB_PARAMS['password']}@{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['dbname']}"
Base = declarative_base()

class Session(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True)
    session_id = Column(String, unique=True, nullable=False)
    messages = relationship("Message", back_populates="session")

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    session = relationship("Session", back_populates="messages")

engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def save_message(session_id: str, role: str, content: str):
    db = next(get_db())
    try:
        session = db.query(Session).filter(Session.session_id == session_id).first()
        if not session:
            session = Session(session_id=session_id)
            db.add(session)
            db.commit()
            db.refresh(session)

        db.add(Message(session_id=session.id, role=role, content=content))
        db.commit()
    except SQLAlchemyError:
        db.rollback()
    finally:
        db.close()

def load_session_history(session_id: str) -> BaseChatMessageHistory:
    db = next(get_db())
    chat_history = ChatMessageHistory()
    try:
        # Retrieve the session
        session = db.query(Session).filter(Session.session_id == session_id).first()
        if session:
            # Add each message to the chat history
            for message in session.messages:
                chat_history.add_message({"role": message.role, "content": message.content})
    except SQLAlchemyError:
        pass
    finally:
        db.close()

    return chat_history

def get_db_connection():
    return psycopg2.connect(**DB_PARAMS)

def store_chunks(chunks, chunk_ids):
    conn = get_db_connection()

    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id TEXT PRIMARY KEY,
                content TEXT,
                metadata JSONB
            )
        """)
        
        data = [(id, chunk.page_content, json.dumps(chunk.metadata)) for id, chunk in zip(chunk_ids, chunks)]
        
        execute_batch(cur, """
            INSERT INTO document_chunks (id, content, metadata)
            VALUES (%s, %s, %s)
            ON CONFLICT (id) DO UPDATE
            SET content = EXCLUDED.content, metadata = EXCLUDED.metadata
        """, data)
    
    conn.commit()
    conn.close()

def fetch_chunks(chunk_ids):
    conn = get_db_connection()
    with conn.cursor() as cur:
        placeholders = ','.join(['%s'] * len(chunk_ids))
        cur.execute(f"SELECT id, content, metadata FROM document_chunks WHERE id IN ({placeholders})", chunk_ids)
        results = cur.fetchall()
    
    chunks = [Document(page_content=content, metadata=metadata) for _, content, metadata in results]
    conn.close()
    return chunks

def fetch_messages(session_id):
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT role, content FROM messages
            WHERE session_id = %s
            ORDER BY id ASC
        """, (session_id,))
        results = cur.fetchall()
    
    conn.close()
    return results
    # Modify the get_session_history function to use the database
