import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from vector_search import retrieve_context

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    groq_api_key=API_KEY
)

def generate_response(question):

    context = retrieve_context(question)

    prompt = f"""
You are a helpful assistant.

Answer ONLY using the provided context.

If the answer is not found in the context,
reply:

I could not find this information in the PDF.

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content


if __name__ == "__main__":

    print("=" * 50)
    print("PDF RAG CHATBOT")
    print("=" * 50)

    while True:

        question = input("\nAsk Question: ")

        if question.lower() == "exit":

            print("\nGoodbye!")
            break

        answer = generate_response(question)

        print("\nAnswer:\n")
        print(answer)