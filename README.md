# Runbook Assistant

RAG-powered chat interface for querying runbooks and playbooks during incidents.

## Features

- Upload runbooks in Markdown, plain text, or PDF
- FAISS vector index persisted to disk
- Streaming responses grounded in your runbook content
- Source citations shown per response

## Setup

```bash
pip install anthropic streamlit python-dotenv langchain langchain-community faiss-cpu sentence-transformers PyMuPDF
```

Copy `.env.example` to `.env` and add your Anthropic API key:

```
ANTHROPIC_API_KEY=your_key_here
```

## Usage

### Local

```bash
streamlit run app.py
```

### Docker

```bash
docker build -t runbook-assistant .
docker run -p 8501:8501 -e ANTHROPIC_API_KEY=your_key_here runbook-assistant
```

Open [http://localhost:8501](http://localhost:8501), upload your runbooks via the sidebar, click **Embed**, then start asking questions.
