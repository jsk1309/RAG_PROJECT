import os

from langchain_community.document_loaders import (
    PyPDFLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_chroma import Chroma


def create_vector_store(pdf_path):

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(
        documents
    )

    # -------------------------
    # PDF Statistics
    # -------------------------

    total_pages = len(documents)

    total_words = sum(
        len(doc.page_content.split())
        for doc in documents
    )

    total_characters = sum(
        len(doc.page_content)
        for doc in documents
    )

    chunk_sizes = [
        len(chunk.page_content.split())
        for chunk in chunks
    ]

    # -------------------------
    # Embeddings
    # -------------------------

    embeddings = HuggingFaceEmbeddings(
        model_name="Alibaba-NLP/gte-modernbert-base"
    )

    # -------------------------
    # Chroma DB
    # -------------------------

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="pdf-rag",
        persist_directory="./chroma_langchain_db"
    )

    stats = {
        "pages": total_pages,
        "words": total_words,
        "characters": total_characters,
        "total_chunks": len(chunks),
        "avg_chunk": round(
            sum(chunk_sizes) / len(chunk_sizes),
            2
        ),
        "largest_chunk": max(chunk_sizes),
        "smallest_chunk": min(chunk_sizes)
    }

    return stats