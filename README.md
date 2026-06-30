The orchestration of these steps is handled using LangChain Expression Language (LCEL), which allows the retrieval, prompting, and generation steps to be composed as a single declarative chain.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend / UI | Streamlit |
| Orchestration | LangChain (LCEL) |
| Document Loading | langchain_community PyPDFLoader |
| Text Splitting | langchain_text_splitters |
| Embeddings | HuggingFace Embeddings (BAAI/bge-small-en-v1.5) |
| Vector Store | Chroma |
| Language Model | Meta-Llama-3-8B-Instruct via HuggingFaceEndpoint |
| Environment Management | python-dotenv |

---

## Prerequisites

Before installing this project, ensure the following are available on your system:

- Python 3.10 or higher
- pip (Python package manager)
- Git
- A Hugging Face account with an API access token
- At least 4 GB of free RAM for local embedding and vector store operations
- (Optional but recommended) A virtual environment tool such as `venv` or `conda`

---

## Installation Guide

### Step 1: Clone the Repository

```bash
git clone https://github.com/<your-username>/rag-based-research-assistant.git
cd rag-based-research-assistant
```

### Step 2: Create and Activate a Virtual Environment

On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

If a `requirements.txt` file is not yet present in your project, create one with at least the following packages: