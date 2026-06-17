from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_chroma import (
    Chroma
)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

vector_store = Chroma(
    collection_name="pdf-rag",
    persist_directory="./chroma_langchain_db",
    embedding_function=embeddings
)


def retrieve_context(question):

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

while True:
    user_input = input("Prompt: ('q' to quit) ")
    if user_input.lower() == "q":
        print("Goodbye!")
        break

    context = retrieve_context(user_input)
    print(type(context))
    print(f"Context: \n{context}\n")
    print(" ")