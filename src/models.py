import os
import fitz  # PyMuPDF for PDF extraction
import faiss  # Vector database
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv(r"C:\Users\dell\Desktop\RAG NotesBot\.env\.env")

# Retrieve Groq API key
groq_api_key = os.getenv('CHATGROQ_API_KEY')
if not groq_api_key:
    raise ValueError("CHATGROQ_API_KEY environment variable not found.")

# Initialize SentenceTransformer model
embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FAISS index (128D for all-MiniLM-L6-v2)
dimension = 384  # Dimension of 'all-MiniLM-L6-v2' embeddings
index = faiss.IndexFlatL2(dimension)

# Store text chunks separately for retrieval
text_chunks = []

# Initialize FastAPI app
app = FastAPI()

# Initialize ChatGroq
llm = ChatGroq(api_key=groq_api_key, model="llama3-8b-8192")

def extract_text_from_pdf(pdf_path):
    """Extract text from the PDF file."""
    doc = fitz.open(pdf_path)
    text = " ".join([page.get_text() for page in doc])
    return text

def chunk_text(text, chunk_size=500):
    """Split text into manageable chunks."""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF and store its text embeddings in FAISS."""
    global text_chunks  # Modify global chunks list
    
    # Save the uploaded file
    pdf_path = f"temp_{file.filename}"
    with open(pdf_path, "wb") as f:
        f.write(await file.read())

    # Extract text from PDF
    pdf_text = extract_text_from_pdf(pdf_path)

    # Split text into chunks
    chunks = chunk_text(pdf_text)

    # Generate embeddings for chunks
    chunk_embeddings = [embeddings_model.encode(chunk) for chunk in chunks]

    # Convert embeddings to NumPy array
    embeddings_np = np.array(chunk_embeddings)

    # Add embeddings to FAISS index
    index.add(embeddings_np)

    # Store chunks for retrieval
    text_chunks.extend(chunks)

    # Cleanup file
    os.remove(pdf_path)

    return {"message": f"PDF '{file.filename}' uploaded and processed successfully!"}

@app.get("/query/")
async def query_pdf(question: str = Form(...)):
    """Retrieve relevant text from FAISS and use ChatGroq to answer queries."""
    if len(text_chunks) == 0:
        return {"error": "No PDFs uploaded yet."}

    # Encode the question
    query_embedding = embeddings_model.encode(question).reshape(1, -1)

    # Retrieve closest match from FAISS
    _, indices = index.search(query_embedding, k=3)  # Retrieve top-3 matches

    # Get the most relevant chunks
    relevant_text = " ".join([text_chunks[i] for i in indices[0]])

    # Send context and query to ChatGroq
    prompt = f"Context: {relevant_text}\n\nQuestion: {question}"
    response = llm.invoke(prompt)

    return {"answer": response.content}
