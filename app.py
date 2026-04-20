import streamlit as st
import os
from dotenv import load_dotenv
from google import genai
from vectorstore import VectorStore

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

store = VectorStore()


def generate_answer(context, query):

    prompt = f"""
Context:
{context}

Question:
{query}

Answer in simple words only.
"""

    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt
    )

    return response.text


st.title("Enterprise Knowledge Base Q&A System (Gemini + RAG)")

query = st.text_input("Ask your question:")

if query:

    results = store.search(query)

    st.subheader(" Context:")
    for r in results:
        st.write(r)

    context = " ".join(results)

    answer = generate_answer(context, query)

    st.subheader("Answer:")
    st.write(answer)