import os   
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
import numpy as np
import faiss
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv(".env")

groq_api_key = os.getenv('CHATGROQ_API_KEY')
if not groq_api_key:
    st.error("CHATGROQ_API_KEY environment variable not found.")
    st.stop()

# Initialize SentenceTransformer model
embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FAISS index (128D for all-MiniLM-L6-v2)
dimension = 384  # Dimension of 'all-MiniLM-L6-v2' embeddings
index = faiss.IndexFlatL2(dimension)

# Store text chunks separately for retrieval
text_chunks = []

# Initialize ChatGroq
llm = ChatGroq(api_key=groq_api_key, model="llama3-8b-8192")

def extract_text_from_pdf(pdf_path):
    """Extract text from the PDF file."""
    doc =  PyPDFLoader(pdf_path)
    text = " ".join([page.get_text() for page in doc])
    return text

def chunk_text(text, chunk_size=500):
    """Split text into manageable chunks."""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def summarize_text(text):
    """Summarize the extracted text using ChatGroq."""
    prompt = f"Summarize the following text:\n\n{text}"
    response = llm.invoke(prompt)
    return response.content if response else "Summarization failed."

# Streamlit UI
st.title("📄 RAG NotesBot")

# File Upload
uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    for uploaded_file in uploaded_files:
        pdf_path = f"temp_{uploaded_file.name}"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())
        
        pdf_text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(pdf_text)
        chunk_embeddings = [embeddings_model.encode(chunk) for chunk in chunks]
        embeddings_np = np.array(chunk_embeddings)
        index.add(embeddings_np)
        text_chunks.extend(chunks)
        os.remove(pdf_path)
    st.success("PDFs uploaded and processed successfully!")

# User Query
query = st.text_input("Ask a question based on uploaded PDFs:")
if query and text_chunks:
    query_embedding = embeddings_model.encode(query).reshape(1, -1)
    _, indices = index.search(query_embedding, k=3)
    relevant_text = " ".join([text_chunks[i] for i in indices[0]])
    prompt = f"Context: {relevant_text}\n\nQuestion: {query}"
    response = llm.invoke(prompt)
    st.subheader("Answer:")
    st.write(response.content)

# Summarization Button
if st.button("Summarize Documents"):
    combined_text = " ".join(text_chunks)
    summary = summarize_text(combined_text)
    st.subheader("Summary:")
    st.write(summary)
