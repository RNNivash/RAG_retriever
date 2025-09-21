# 🧠 RAG Retriever & FastAPI Deployment

This project demonstrates different approaches to **Retrieval-Augmented Generation (RAG)** using Hugging Face embeddings, LangChain, and FAISS/Chroma vector stores.
It includes **three Jupyter notebooks** exploring simple to advanced RAG pipelines and a **FastAPI app** for deploying the RAG system as an API.

<img width="1115" height="687" alt="Image" src="https://github.com/user-attachments/assets/531b9339-79a2-405c-b914-d80441a0b85f" />

---

## 🚀 Tech Stack

* **Python 3.12+**
* **LangChain** (chains, retrievers, tools, agents)
* **Hugging Face Transformers & Sentence Transformers**
* **FAISS / Chroma** (vector stores)
* **FastAPI** (API framework)
* **Uvicorn** (server)
* **Jupyter Notebooks** (experiments & testing)

---

## 📂 Project Files

### 🔹 1. `simple_rag.ipynb`

* Loads text, web, and PDF documents.
* Splits into chunks with `RecursiveCharacterTextSplitter`.
* Creates vector store with **FAISS**.
* Runs a basic RAG pipeline (query → retrieve → answer).
* ✅ Great for **beginners** exploring RAG.

### 🔹 2. `retriever.ipynb`

* Focused on **retriever + chain** with LangChain.
* Uses **ChatPromptTemplate** and **create\_retrieval\_chain**.
* Answers queries based only on context.
* ✅ Ideal for **controlled question answering**.

### 🔹 3. `rag_agent.ipynb`

* Demonstrates an **agent-based RAG**.
* Integrates multiple tools:

  * Wikipedia API
  * Arxiv API
  * Custom Retriever
* Uses **ReAct prompting** for tool selection.
* ✅ Useful for **multi-source retrieval**.

### 🔹 4. `app.py` (FastAPI App)

* Exposes `/query` endpoint.
* Accepts a question as JSON input.
* Runs the RAG pipeline and returns the answer.
* Can be deployed with **Uvicorn** or **Render**.

---

## ⚙️ How to Run

### 1️⃣ Clone Repository

```bash
git clone https://github.com/<your-username>/RAG_retriever.git
cd RAG_retriever
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Set Environment Variables

Create a `.env` file with your Hugging Face token:

```
HF_TOKEN=your_huggingface_token
```

### 4️⃣ Run Jupyter Notebooks

```bash
jupyter notebook
```

Open:

* `simple_rag.ipynb`
* `retriever.ipynb`
* `rag_agent.ipynb`

### 5️⃣ Run FastAPI

```bash
uvicorn app:app --reload
```

API will be live at:
👉 `http://127.0.0.1:8000`

---

## 📬 Example API Request

### Request

```bash
curl -X POST "http://127.0.0.1:8000/query" \
-H "Content-Type: application/json" \
-d '{"question": "What is transformer architecture?"}'
```

### Response

```json
{
  "answer": "The Transformer architecture is a deep learning model introduced in the paper 'Attention Is All You Need'..."
}
```

---

## 🌟 Features

* 📖 Load documents from text, PDF, and web sources.
* 🔎 Efficient semantic search using **FAISS/Chroma**.
* 🤖 Use **Hugging Face LLMs** for answering queries.
* 🔧 Advanced agent setup with external tools (Wikipedia, Arxiv).
* ⚡ Serve as an API with **FastAPI**.

---

## 📌 Notes

* For local use, `FAISS` works fine. For lightweight deployments, use `Chroma`.
* Make sure your **HF token** is active; expired tokens will cause embedding errors.
* The notebooks are designed for experimentation; the FastAPI app is production-ready.

---

## 📜 License
This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.

## 📬 Connect with Me

- LinkedIn: [Nivash](https://www.linkedin.com/in/nivash-r-n/)
- Portfolio: [Nivash](https://rnnivash.github.io/My_Port/)
- Email: [hello.nivashinsights@gmail.com](mailto:hello.nivashinsights@gmail.com)

🔍 **Let's leverage data science to make healthcare better!** 🚀
