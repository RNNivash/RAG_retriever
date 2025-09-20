import os
from dotenv import load_dotenv
from huggingface_hub import login

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, WebBaseLoader, PyPDFLoader
import bs4

# Load env
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(hf_token)

# Initialize embeddings (small + CPU-friendly)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Build / Load documents ---
def load_documents():
    loaders = [
        TextLoader("data/speech.txt"),
        WebBaseLoader(
            web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
            bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                class_=("post-title","post-content","post-header")
            ))
        ),
        PyPDFLoader("data/attention.pdf")
    ]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    return docs

def prepare_vectorstore(persist_dir="./chroma_db"):
    if os.path.exists(persist_dir):
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    documents = load_documents()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    documents = text_splitter.split_documents(documents)
    return Chroma.from_documents(documents, embeddings, persist_directory=persist_dir)

# --- Core RAG Search ---
db = prepare_vectorstore()

def rag_query(question: str, k: int = 3):
    results = db.similarity_search(question, k=k)
    return [doc.page_content for doc in results]