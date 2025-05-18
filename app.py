import streamlit as st
import fitz  # PyMuPDF
import faiss
import openai
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import numpy as np
import tempfile
import os

# Set your OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Load SentenceTransformer model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Chunk text into smaller sections
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Create FAISS index from document chunks
def create_faiss_index(chunks):
    embeddings = embed_model.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, embeddings

# Retrieve top-k relevant chunks
def retrieve_chunks(query, chunks, index, embeddings, k=5):
    query_embedding = embed_model.encode([query])
    D, I = index.search(np.array(query_embedding), k)
    return [chunks[i] for i in I[0]]

# Generate answer using OpenAI
def generate_answer(context, query):
    prompt = f"""You are an expert assistant. Use the following context to answer the question.

Context:
{context}

Question:
{query}

If you can extract or infer structured data from the context, return it in a Python dictionary format suitable for plotting.
Otherwise, just return the answer."""
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You answer questions using scientific context and return data in dictionary format if charting is possible."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
    )
    return response['choices'][0]['message']['content']

# Try to parse chartable data from response
def try_extract_chart_data(response):
    try:
        code_part = response.split("```python")[1].split("```")[0]
        local_vars = {}
        exec(code_part, {}, local_vars)
        return local_vars
    except Exception:
        return None

# Streamlit App
st.title("📊 RAG Chart Bot for Research Papers")
uploaded_file = st.file_uploader("Upload an arXiv research paper (PDF)", type="pdf")

if uploaded_file:
    with st.spinner("Extracting text..."):
        text = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(text)
        index, embeddings = create_faiss_index(chunks)

    st.success("Document loaded and indexed!")

    query = st.text_input("Ask a question about the paper")

    if query:
        with st.spinner("Retrieving relevant chunks..."):
            top_chunks = retrieve_chunks(query, chunks, index, embeddings)
            context = "\n\n".join(top_chunks)

        with st.spinner("Generating answer..."):
            answer = generate_answer(context, query)

        st.markdown("### 🤖 Answer")
        st.write(answer)

        chart_data = try_extract_chart_data(answer)
        if chart_data:
            st.markdown("### 📈 Auto-generated Chart")
            df = pd.DataFrame(chart_data)
            st.line_chart(df)
