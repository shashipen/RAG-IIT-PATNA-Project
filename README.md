# RAG From Scratch — IIT Patna AIML Certification (Project 2)

## Overview
A complete Retrieval-Augmented Generation (RAG) pipeline built from scratch.
Chat with your own documents (PDF/TXT) with cited answers.

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run with your own documents
python rag.py document1.pdf document2.txt

# 4. Run the demo (no files needed)
python rag.py
```

## How It Works

| Step | What Happens |
|------|-------------|
| 1. Ingest | Reads PDF/TXT files and extracts raw text |
| 2. Chunk | Splits text into 300-word overlapping chunks |
| 3. Embed | Converts each chunk to a TF-IDF meaning vector |
| 4. Store | Saves vectors + metadata to vector_store.json |
| 5. Retrieve | Finds top-3 most relevant chunks via cosine similarity |
| 6. Generate | Extracts best sentences matching the query |
| 7. Cite | Shows which file and chunk was used |

## File Structure
```
rag_project/
├── rag.py              # Main pipeline (all 7 steps)
├── requirements.txt    # Dependencies
├── README.md           # This file
├── vector_store.json   # Auto-generated after indexing
└── sample_docs/        # Auto-generated demo documents
```

## Example Usage
```python
from rag import RAGPipeline

rag = RAGPipeline()
rag.build_index(["my_doc.pdf", "notes.txt"])

result = rag.query("What is supervised learning?")
print(result["answer"])
print(result["citations"])
```
