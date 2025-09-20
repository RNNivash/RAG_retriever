# app.py
from fastapi import FastAPI, Query
from pydantic import BaseModel
import uvicorn

# === Imports from retriever pipeline ===
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# ======================
# Setup
# ======================
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# Load documents
loader = PyPDFLoader("attention.pdf")
docs = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

# Embeddings + Vector DB
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(chunks, embeddings)
retriever = db.as_retriever()

# LLM (small, local model)
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    tokenizer="google/flan-t5-base",
    max_length=512
)
llm = HuggingFacePipeline(pipeline=pipe)

# Prompt
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context. 
If the context does not contain the answer, say "I don’t know based on the provided context."
<context>
{context}
</context>
Question: {input}
""")

# Retrieval chain
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# ======================
# FastAPI App
# ======================
app = FastAPI(title="RAG with FastAPI", version="1.0")

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def home():
    return {"message": "RAG FastAPI is running!"}

@app.post("/ask")
def ask(request: QueryRequest):
    try:
        result = retrieval_chain.invoke({"input": request.query})
        return {"question": request.query, "answer": result["answer"]}
    except Exception as e:
        return {"error": str(e)}

# Run with: uvicorn app:app --reload
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)