# 🏥 RAG-Based Medical Chatbot Using Streamlit

An AI-powered medical chatbot built using Retrieval-Augmented Generation (RAG) that provides context-aware answers from medical documents and reports.

The chatbot uses LangChain, Pinecone, HuggingFace embeddings, and Groq LLMs to retrieve relevant medical information from PDFs and generate intelligent responses.

---

# 🚀 Features

* 📄 PDF-based medical knowledge retrieval
* 🧠 Retrieval-Augmented Generation (RAG)
* 🔍 Semantic search using Pinecone vector database
* 🤖 Groq LLM integration for fast AI responses
* 🧾 Medical report interpretation support
* 🖼️ OCR and image understanding support
* 💬 Session-based conversational memory
* ⚡ Streamlit web interface
* ☁️ Render deployment support

---

# 🛠️ Tech Stack

## Frontend

* Streamlit

## Backend / AI

* Python
* LangChain
* Groq API
* HuggingFace Embeddings
* Pinecone Vector Database

## Document Processing

* PyMuPDF
* OCR Processing
* Recursive Text Splitting

## Deployment

* Render

---

# 🧠 System Architecture

```text
User Query
    ↓
Streamlit Interface
    ↓
LangChain Retrieval Pipeline
    ↓
Pinecone Vector Search
    ↓
Relevant Medical Chunks Retrieved
    ↓
Groq LLM Generates Response
    ↓
Answer Returned To User
```

---

# 📂 Project Structure

```text
RAG-Based-Medical-Chatbot-Using-Streamlit/
│
├── chain/                 # RAG chain and prompt setup
├── embedding/             # Embedding model initialization
├── ingestion/             # PDF ingestion pipeline
├── retrieval/             # Retrieval logic
├── vectorstore/           # Pinecone vector database setup
├── app.py                 # Main Streamlit application
├── requirements.txt       # Project dependencies
└── README.md
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/Kartiksonani2005/RAG-Based-Medical-Chatbot-Using-Streamlit.git
cd RAG-Based-Medical-Chatbot-Using-Streamlit
```

---

## 2️⃣ Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file in the root directory.

```env
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

---

# 📊 How It Works

1. Medical PDFs are processed and split into chunks.
2. Text chunks are converted into embeddings using HuggingFace models.
3. Embeddings are stored inside Pinecone vector database.
4. User queries are converted into embeddings.
5. Pinecone retrieves semantically similar chunks.
6. LangChain combines retrieved context with user query.
7. Groq LLM generates final response.

---

# 🧪 Example Use Cases

* Medical question answering
* Lab report explanation
* Context-aware healthcare chatbot
* AI-assisted document understanding
* Medical PDF semantic search

---

# 📸 Future Improvements

* Multi-user authentication
* Chat history database storage
* Advanced OCR support
* Fine-tuned medical models
* Voice input support
* Docker deployment
* Multi-agent AI workflows

---

# 📌 Learning Outcomes

This project demonstrates practical implementation of:

* Retrieval-Augmented Generation (RAG)
* Vector databases
* Semantic search
* Prompt engineering
* Conversational memory
* AI application deployment
* LangChain pipelines
* LLM orchestration

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a new branch
3. Commit changes
4. Push to your branch
5. Open a Pull Request

---

# 📧 Contact

## Kartik Sonani

* LinkedIn: [https://www.linkedin.com/in/kartik-sonani-ab7a32283?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app/]

* GitHub: [https://github.com/Kartiksonani2005]

---

# ⭐ If you found this project useful

Give this repository a star ⭐ on GitHub.
