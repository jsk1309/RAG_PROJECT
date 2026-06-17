import os
import shutil
from collections import Counter

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_chroma import Chroma


print("STEP 1 - STARTING PDF RAG")

# documents = []

# --------------------------
# LOAD PDF FILES
# --------------------------
# print(f"Loading PDF: {}")

loader = PyPDFLoader(
    os.path.join("data", "ethical_hacking.pdf")
)
documents = loader.load()

print("STEP 2 - PDF LOADED")


# --------------------------
# CHUNKING
# --------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=200
)

chunks = splitter.split_documents(
    documents
)

print(
    "\nChunks Created:",
    len(chunks)
)

print("STEP 4 - CHUNKING COMPLETE")

# --------------------------
# EMBEDDINGS
# --------------------------

print("\nLoading Embedding Model...")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

print("STEP 5 - EMBEDDINGS LOADED")

# --------------------------
# CHROMA DATABASE
# --------------------------

print("\nCreating Chroma Database...")

if os.path.exists("chroma_langchain_db"):
    shutil.rmtree("chroma_langchain_db")
    print("Deleted existing Chroma database")

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="pdf-rag",
    persist_directory="./chroma_langchain_db"
)

print("STEP 6 - CHROMA COMPLETE")

print(
    "\nVector Database Created Successfully"
)