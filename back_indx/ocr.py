#File header
"""
OCR and Ingestion Module
This module handles the OCR processing of PDF documents, ingestion into a vector database,
and optional loading into a SQLite database.
"""

#Import LangChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import OllamaEmbeddings


from pathlib import Path
import os
import bs4
from pathlib import Path
import logging
from .ingestion import ingest_pdf_folder
from .load_env import load_env, test_langsmith_connection
import sqlite3
from typing import List
from langchain_core.documents import Document


# Suppress verbose logging from libraries
logging.getLogger("pypdf").setLevel(logging.ERROR)
logging.getLogger("pdfminer").setLevel(logging.ERROR)   # if used under the hood
logging.getLogger("fitz").setLevel(logging.ERROR)       # PyMuPDF, if used




def perform_ocr(pdf_path: Path):
    env_vars = load_env()
    test_langsmith_connection(project_name="my-project")

    uploads = pdf_path
    clean_chunks, all_pages = ingest_pdf_folder(uploads, tenant="acme_inc", acl="internal")
    return clean_chunks, all_pages


def save_vector_db(clean_chunks: List[Document]) -> None:
    """
    Saves the clean chunks into a Chroma vector database.
    
    Each chunk is stored with its metadata, allowing for efficient retrieval based on content and metadata filters.
    The database is persisted to disk for future use.
    
    Args:
        clean_chunks (List[Document]): List of LangChain Document objects with metadata.
    """
    persist_dir = "chroma_store"
    embed = OllamaEmbeddings(model="nomic-embed-text")  # or HuggingFaceEmbeddings(...)
    vectordb = Chroma.from_documents(clean_chunks, embedding=embed, collection_name="company_rag_vectordb", persist_directory=persist_dir)
    vectordb.persist()
    print(f"✅ Vector DB saved at {persist_dir}")

def load_to_sqlite(clean_chunks: List[Document], db_path: str = 'chunks.db') -> None:
    """
    Loads the clean chunks into a SQLite database.
    
    Creates a table named 'chunks' if it doesn't exist, with columns based on the metadata fields.
    Each chunk is inserted as a row, with 'content' being the page_content.
    The 'doc_id' serves as the main ID to group chunks from the same document/company data.
    The 'source_path' includes the full path to the file.
    
    Args:
        clean_chunks (List[Document]): List of LangChain Document objects with metadata.
        db_path (str): Path to the SQLite database file. Defaults to 'chunks.db'.
    """
    # Connect to SQLite database (creates it if it doesn't exist)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table if not exists
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id TEXT NOT NULL,
        chunk_id TEXT NOT NULL,
        chunk_index INTEGER,
        page INTEGER,
        content TEXT NOT NULL,
        source_filename TEXT,
        source_path TEXT,
        tenant TEXT,
        acl TEXT,
        effective_date TEXT,
        version TEXT,
        dept TEXT,
        section_title TEXT,
        chunk_type TEXT,
        source TEXT
    )
    ''')
    
    # Insert each chunk
    for chunk in clean_chunks:
        metadata = chunk.metadata
        cursor.execute('''
        INSERT INTO chunks (
            doc_id, chunk_id, chunk_index, page, content,
            source_filename, source_path, tenant, acl,
            effective_date, version, dept, section_title,
            chunk_type, source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metadata.get('doc_id'),
            metadata.get('chunk_id'),
            metadata.get('chunk_index'),
            metadata.get('page'),
            chunk.page_content,
            metadata.get('source_filename'),
            metadata.get('source_path'),
            metadata.get('tenant'),
            metadata.get('acl'),
            metadata.get('effective_date'),
            metadata.get('version'),
            metadata.get('dept'),
            metadata.get('section_title'),
            metadata.get('chunk_type'),
            metadata.get('source')
        ))
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    print(f"Loaded {len(clean_chunks)} chunks into {db_path}")

# load_to_sqlite(clean_chunks, db_path='company_chunks.db')