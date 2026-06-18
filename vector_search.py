def retrieve_context(
    vector_store,
    question
):

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