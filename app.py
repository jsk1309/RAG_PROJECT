import streamlit as st
import os

from vector import create_vector_store
from main import generate_response

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📄",
    layout="wide"
)

st.title("📄 PDF Question Answering Chatbot")


uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file:

    os.makedirs(
        "uploads",
        exist_ok=True
    )

    pdf_path = os.path.join(
        "uploads",
        uploaded_file.name
    )

    with open(pdf_path, "wb") as f:
        f.write(
            uploaded_file.getbuffer()
        )

    st.success(
        f"{uploaded_file.name} uploaded successfully!"
    )

    if st.button(
        "Process PDF"
    ):

        with st.spinner(
            "Creating embeddings..."
        ):

            stats = create_vector_store(
                pdf_path
            )

            st.session_state["stats"] = stats

            # clear previous chat
            st.session_state["messages"] = []

        st.success(
            "PDF processed successfully!"
        )

# -------------------------
# PDF EDA
# -------------------------

if "stats" in st.session_state:

    stats = st.session_state["stats"]

    st.subheader("📊 PDF EDA")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Pages",
            stats["pages"]
        )

    with col2:
        st.metric(
            "Words",
            stats["words"]
        )

    with col3:
        st.metric(
            "Characters",
            stats["characters"]
        )

    st.subheader("📦 Chunk Statistics")

    col1, col2 = st.columns(2)

    with col1:

        st.write(
            f"**Total Chunks:** {stats['total_chunks']}"
        )

        st.write(
            f"**Average Chunk Size:** {stats['avg_chunk']}"
        )

    with col2:

        st.write(
            f"**Largest Chunk:** {stats['largest_chunk']}"
        )

        st.write(
            f"**Smallest Chunk:** {stats['smallest_chunk']}"
        )

# -------------------------
# CHAT SECTION
# -------------------------

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:

    with st.chat_message(
        message["role"]
    ):
        st.write(
            message["content"]
        )

question = st.chat_input(
    "Ask something about your PDF..."
)

if question:

    st.session_state["messages"].append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.write(question)

    with st.spinner(
        "Searching PDF..."
    ):

        answer = generate_response(
            question
        )

    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    with st.chat_message(
        "assistant"
    ):
        st.write(answer)