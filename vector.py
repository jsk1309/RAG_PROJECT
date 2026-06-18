import re

from langchain_community.document_loaders import (
    PyPDFLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_chroma import (
    Chroma
)


def clean_text(text):

    # Replace all line breaks with spaces
    text = text.replace("\n", " ")

    # Remove multiple spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def create_vector_db(pdf_path):

    # -----------------------
    # LOAD PDF
    # -----------------------

    loader = PyPDFLoader(
        pdf_path
    )

    documents = loader.load()

    # -----------------------
    # CLEAN PDF TEXT
    # -----------------------

    for doc in documents:

        doc.page_content = clean_text(
            doc.page_content
        )

    # -----------------------
    # COMBINE TEXT FOR EDA
    # -----------------------

    text = ""

    for doc in documents:

        text += doc.page_content + " "

    # -----------------------
    # CHUNKING
    # -----------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(
        documents
    )

    # -----------------------
    # EMBEDDINGS
    # -----------------------

    embeddings = HuggingFaceEmbeddings(
        model_name="google/embeddinggemma-300m"
    )

    # -----------------------
    # VECTOR STORE
    # -----------------------

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return (
        documents,
        text,
        chunks,
        vector_store
    )