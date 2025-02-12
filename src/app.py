import os
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv(r".env\.env")

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Define BeautifulSoup strainer to extract relevant HTML content
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))

# Load webpage
loader = WebBaseLoader(
    web_paths=["https://lilianweng.github.io/posts/2023-06-23-agent/"],
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

# Ensure documents are loaded correctly
assert len(docs) == 1
print(f"Total characters: {len(docs[0].page_content)}")

# Create a FAISS vector store
vector_store = FAISS.from_documents(docs, embeddings)

# Initialize ChatGroq LLM
llm = ChatGroq(model_name="llama3-8b-8192", groq_api_key=os.getenv("GROQ_API_KEY"))
