from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_chroma import (
    Chroma
)

embeddings = HuggingFaceEmbeddings(
    model_name="Alibaba-NLP/gte-modernbert-base"
)

def retrieve_context(question):

    vector_store = Chroma(
        collection_name="pdf-rag",
        persist_directory="./chroma_langchain_db",
        embedding_function=embeddings
    )

    docs = vector_store.similarity_search(
        question,
        k=3
    )

    context = "\n\n".join(
        [
            doc.page_content
            for doc in docs
        ]
    )

    return context