# app.py
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Optional
import os

# LangChain / Hugging Face imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(hf_token)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"token": os.getenv("HF_TOKEN")}
)

# -------------------------
# 1. FastAPI setup
# -------------------------
app = FastAPI(title="RAG Retriever API")

# -------------------------
# 2. Health check route
# -------------------------
@app.get("/")
def root():
    return {"status": "running", "message": "Hello from Render RAG API 🚀"}

# -------------------------
# 3. Lazy load RAG pipeline
# -------------------------
retrieval_chain = None

def load_rag():
    global retrieval_chain
    if retrieval_chain is not None:
        return retrieval_chain

    # Optional: load PDF (make sure attention.pdf exists in repo)
    if os.path.exists("attention.pdf"):
        loader = PyPDFLoader("attention.pdf")
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
    else:
        # fallback to dummy docs
        from langchain.schema import Document
        chunks = [Document(page_content="Transformers are neural networks with self-attention.")]

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks[:15], embeddings)

    pipe = pipeline("text2text-generation", model="google/flan-t5-base")
    llm = HuggingFacePipeline(pipeline=pipe)

    prompt = ChatPromptTemplate.from_template("""
    Answer the following question using only the provided context.
    <context>
    {context}
    </context>
    Question: {input}
    """)

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = db.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain

# -------------------------
# 4. Request/Response models
# -------------------------
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

# -------------------------
# 5. Query endpoint
# -------------------------
@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    try:
        chain = load_rag()
        result = chain.invoke({"input": request.question})
        return QueryResponse(answer=result.get("answer", "No answer generated"))
    except Exception as e:
        import traceback
        traceback.print_exc()
        # return fallback that still fits schema
        return QueryResponse(answer=f"Error: {str(e)}")























# from fastapi import FastAPI, Query
# from pydantic import BaseModel
# import uvicorn

# # === Imports from retriever pipeline ===
# import os
# from dotenv import load_dotenv
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from transformers import pipeline
# from langchain_community.llms import HuggingFacePipeline
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain

# # ======================
# # Setup
# # ======================
# load_dotenv()
# hf_token = os.getenv("HF_TOKEN")

# # Load documents
# loader = PyPDFLoader("attention.pdf")
# docs = loader.load()

# # Split into chunks
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# chunks = text_splitter.split_documents(docs)

# # Embeddings + Vector DB
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# db = FAISS.from_documents(chunks, embeddings)
# retriever = db.as_retriever()

# # LLM (small, local model)
# pipe = pipeline(
#     "text2text-generation",
#     model="google/flan-t5-base",
#     tokenizer="google/flan-t5-base",
#     max_length=512
# )
# llm = HuggingFacePipeline(pipeline=pipe)

# # Prompt
# prompt = ChatPromptTemplate.from_template("""
# Answer the following question based only on the provided context. 
# If the context does not contain the answer, say "I don’t know based on the provided context."
# <context>
# {context}
# </context>
# Question: {input}
# """)

# # Retrieval chain
# document_chain = create_stuff_documents_chain(llm, prompt)
# retrieval_chain = create_retrieval_chain(retriever, document_chain)

# # ======================
# # FastAPI App
# # ======================
# app = FastAPI(title="RAG with FastAPI", version="1.0")

# class QueryRequest(BaseModel):
#     query: str

# @app.get("/")
# def home():
#     return {"message": "RAG FastAPI is running!"}

# @app.post("/ask")
# def ask(request: QueryRequest):
#     try:
#         result = retrieval_chain.invoke({"input": request.query})
#         return {"question": request.query, "answer": result["answer"]}
#     except Exception as e:
#         return {"error": str(e)}

# # Run with: uvicorn app:app --reload
# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 8000))
#     uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)