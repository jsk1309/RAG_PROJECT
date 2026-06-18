import os
import streamlit as st

from dotenv import load_dotenv

from langchain_groq import ChatGroq

from vector import create_vector_db

from vector_search import (
    retrieve_context
)

# -----------------------
# PAGE CONFIG
# -----------------------

st.set_page_config(
    page_title="PDF RAG",
    layout="wide"
)

st.title(
    "📄 PDF RAG using EmbeddingGemma"
)

# -----------------------
# ENV
# -----------------------

load_dotenv()

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)

if not GROQ_API_KEY:

    st.error(
        "GROQ_API_KEY not found"
    )

    st.stop()

# -----------------------
# SIDEBAR
# -----------------------

st.sidebar.title(
    "📂 Upload PDF"
)

uploaded_file = st.sidebar.file_uploader(
    "Drag & Drop PDF",
    type=["pdf"]
)

process_pdf = st.sidebar.button(
    "Process PDF"
)

# -----------------------
# PROCESS PDF
# -----------------------

if uploaded_file and process_pdf:

    with open(
        uploaded_file.name,
        "wb"
    ) as f:

        f.write(
            uploaded_file.getbuffer()
        )

    with st.spinner(
        "Processing PDF..."
    ):

        (
            documents,
            text,
            chunks,
            vector_store
        ) = create_vector_db(
            uploaded_file.name
        )

    st.session_state.vector_store = (
        vector_store
    )

    st.session_state.documents = (
        documents
    )

    st.session_state.text = (
        text
    )

    st.session_state.chunks = (
        chunks
    )

# -----------------------
# SHOW EDA
# -----------------------

if "documents" in st.session_state:

    st.subheader(
        "📊 PDF EDA"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Pages",
            len(
                st.session_state.documents
            )
        )

    with col2:

        st.metric(
            "Words",
            len(
                st.session_state.text.split()
            )
        )

    with col3:

        st.metric(
            "Characters",
            len(
                st.session_state.text
            )
        )

    chunk_lengths = [

        len(
            chunk.page_content.split()
        )

        for chunk in
        st.session_state.chunks
    ]

    st.subheader(
        "📦 Chunk Statistics"
    )

    st.write(
        "Total Chunks:",
        len(
            st.session_state.chunks
        )
    )

    st.write(
        "Average Chunk Size:",
        round(
            sum(chunk_lengths)
            /
            len(chunk_lengths),
            2
        )
    )

    st.write(
        "Largest Chunk:",
        max(chunk_lengths)
    )

    st.write(
        "Smallest Chunk:",
        min(chunk_lengths)
    )

# -----------------------
# QUESTION
# -----------------------

question = st.text_input(
    "Ask a Question"
)

# -----------------------
# ANSWER
# -----------------------

if (
    question
    and
    "vector_store"
    in st.session_state
):

    context = retrieve_context(

        st.session_state.vector_store,

        question
    )

    llm = ChatGroq(

        model_name=
        "llama-3.1-8b-instant",

        groq_api_key=
        GROQ_API_KEY,

        temperature=0
    )

    prompt = f"""
You are a helpful PDF assistant.

Answer ONLY using the context.

If the answer is not present
in the context, say:

I could not find the answer
in the PDF.

Context:
{context}

Question:
{question}

Answer:
"""

    with st.spinner(
        "Generating Answer..."
    ):

        response = llm.invoke(
            prompt
        )

    st.subheader(
        "Answer"
    )

    st.write(
        response.content
    )

    with st.expander(
    "Retrieved Context"
):

        st.text_area(
        "Context",
        context,
        height=300
        )