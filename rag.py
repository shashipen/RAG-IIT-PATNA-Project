"""
RAG From Scratch - IIT Patna AIML Certification Project 2
Author: Student Submission
Description: A complete RAG (Retrieval-Augmented Generation) pipeline
             that lets users chat with their documents with citations.
"""

import os
import json
import re
import math
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────
# STEP 1: DOCUMENT INGESTION
# ─────────────────────────────────────────────

def ingest_txt(filepath: str) -> str:
    """Read plain text file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def ingest_pdf(filepath: str) -> str:
    """Extract text from PDF using pypdf."""
    from pypdf import PdfReader
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def ingest_document(filepath: str) -> str:
    """Auto-detect file type and extract text."""
    ext = Path(filepath).suffix.lower()
    if ext == ".txt":
        return ingest_txt(filepath)
    elif ext == ".pdf":
        return ingest_pdf(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: .txt, .pdf")


# ─────────────────────────────────────────────
# STEP 2: CHUNKING
# ─────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks by word count.
    - chunk_size: max words per chunk
    - overlap: words shared between consecutive chunks
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ─────────────────────────────────────────────
# STEP 3: EMBEDDINGS (TF-IDF style, no API needed)
# ─────────────────────────────────────────────

class TFIDFEmbedder:
    """
    Simple TF-IDF based embedder.
    Creates fixed-size meaning vectors without needing any API key.
    """

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray = None

    def _tokenize(self, text: str) -> list[str]:
        """Lowercase, remove punctuation, split into words."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return text.split()

    def fit(self, corpus: list[str]):
        """Build vocabulary and IDF from the corpus."""
        # Count document frequency
        doc_freq: dict[str, int] = {}
        all_tokens = []
        for doc in corpus:
            tokens = set(self._tokenize(doc))
            all_tokens.extend(tokens)
            for t in tokens:
                doc_freq[t] = doc_freq.get(t, 0) + 1

        # Pick top vocab_size words by doc frequency
        sorted_words = sorted(doc_freq.items(), key=lambda x: -x[1])
        vocab_words = [w for w, _ in sorted_words[:self.vocab_size]]
        self.vocab = {w: i for i, w in enumerate(vocab_words)}

        # Compute IDF
        N = len(corpus)
        self.idf = np.zeros(len(self.vocab))
        for word, idx in self.vocab.items():
            df = doc_freq.get(word, 0)
            self.idf[idx] = math.log((N + 1) / (df + 1)) + 1  # smoothed IDF

    def embed(self, text: str) -> np.ndarray:
        """Convert text to a TF-IDF vector."""
        tokens = self._tokenize(text)
        tf: dict[str, float] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        # Normalize TF
        total = len(tokens) or 1
        vec = np.zeros(len(self.vocab))
        for word, count in tf.items():
            if word in self.vocab:
                idx = self.vocab[word]
                vec[idx] = (count / total) * self.idf[idx]
        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


# ─────────────────────────────────────────────
# STEP 4: VECTOR STORE
# ─────────────────────────────────────────────

class VectorStore:
    """
    In-memory vector store.
    Stores chunk text, metadata, and embedding vectors.
    Supports save/load to JSON for persistence.
    """

    def __init__(self):
        self.chunks: list[str] = []
        self.metadata: list[dict] = []
        self.vectors: list[np.ndarray] = []

    def add(self, chunk: str, meta: dict, vector: np.ndarray):
        self.chunks.append(chunk)
        self.metadata.append(meta)
        self.vectors.append(vector)

    def save(self, path: str):
        data = {
            "chunks": self.chunks,
            "metadata": self.metadata,
            "vectors": [v.tolist() for v in self.vectors],
        }
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"[VectorStore] Saved {len(self.chunks)} chunks to {path}")

    def load(self, path: str):
        with open(path, "r") as f:
            data = json.load(f)
        self.chunks = data["chunks"]
        self.metadata = data["metadata"]
        self.vectors = [np.array(v) for v in data["vectors"]]
        print(f"[VectorStore] Loaded {len(self.chunks)} chunks from {path}")


# ─────────────────────────────────────────────
# STEP 5: RETRIEVAL (Cosine Similarity)
# ─────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieve(query_vec: np.ndarray, store: VectorStore, top_k: int = 3) -> list[dict]:
    """Find top-k most relevant chunks for a query vector."""
    scored = []
    for i, vec in enumerate(store.vectors):
        score = cosine_similarity(query_vec, vec)
        scored.append((score, i))
    scored.sort(reverse=True)
    results = []
    for score, idx in scored[:top_k]:
        results.append({
            "chunk": store.chunks[idx],
            "metadata": store.metadata[idx],
            "score": round(score, 4),
        })
    return results


# ─────────────────────────────────────────────
# STEP 6: ANSWER GENERATION (Rule-based, no API)
# ─────────────────────────────────────────────

def generate_answer(query: str, retrieved_chunks: list[dict]) -> str:
    """
    Generate an answer from retrieved chunks.
    Uses extractive summarization — no LLM API required.
    """
    if not retrieved_chunks:
        return "I could not find relevant information in the documents."

    # Find sentences in top chunks that contain query keywords
    query_words = set(query.lower().split())
    stop_words = {"what", "is", "are", "the", "a", "an", "how", "why",
                  "when", "where", "who", "does", "do", "in", "of", "to", "?"}
    keywords = query_words - stop_words

    best_sentences = []
    for item in retrieved_chunks:
        sentences = re.split(r'(?<=[.!?])\s+', item["chunk"])
        for sent in sentences:
            sent_lower = sent.lower()
            match_count = sum(1 for kw in keywords if kw in sent_lower)
            if match_count > 0:
                best_sentences.append((match_count, sent.strip()))

    best_sentences.sort(reverse=True)
    top_sentences = [s for _, s in best_sentences[:3]]

    if top_sentences:
        answer = " ".join(top_sentences)
    else:
        # Fallback: return beginning of top chunk
        answer = retrieved_chunks[0]["chunk"][:500] + "..."

    return answer


# ─────────────────────────────────────────────
# STEP 7: CITATIONS
# ─────────────────────────────────────────────

def format_citations(retrieved_chunks: list[dict]) -> str:
    """Format citation info for display."""
    lines = ["\n📚 Sources used:"]
    for i, item in enumerate(retrieved_chunks, 1):
        meta = item["metadata"]
        lines.append(
            f"  [{i}] File: {meta['filename']} | "
            f"Chunk #{meta['chunk_id']} | "
            f"Relevance score: {item['score']}"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────
# FULL RAG PIPELINE CLASS
# ─────────────────────────────────────────────

class RAGPipeline:
    """
    End-to-end RAG pipeline:
      ingest → chunk → embed → store → retrieve → generate → cite
    """

    def __init__(self, store_path: str = "vector_store.json"):
        self.store_path = store_path
        self.embedder = TFIDFEmbedder(vocab_size=2000)
        self.store = VectorStore()
        self.fitted = False

    def build_index(self, filepaths: list[str]):
        """Process documents and build the vector index."""
        all_chunks = []
        all_meta = []

        print("\n[Step 1] Ingesting documents...")
        for fp in filepaths:
            print(f"  Reading: {fp}")
            text = ingest_document(fp)
            chunks = chunk_text(text, chunk_size=300, overlap=50)
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_meta.append({
                    "filename": Path(fp).name,
                    "chunk_id": i,
                    "filepath": fp,
                })
            print(f"  → {len(chunks)} chunks from {Path(fp).name}")

        print(f"\n[Step 2] Total chunks: {len(all_chunks)}")
        print("[Step 3] Building TF-IDF vocabulary and embeddings...")
        self.embedder.fit(all_chunks)
        self.fitted = True

        print("[Step 4] Storing vectors...")
        for chunk, meta in zip(all_chunks, all_meta):
            vec = self.embedder.embed(chunk)
            self.store.add(chunk, meta, vec)

        self.store.save(self.store_path)
        print(f"[Step 4] Index saved → {self.store_path}")
        print(f"\n✅ Index built successfully! {len(all_chunks)} chunks indexed.\n")

    def load_index(self):
        """Load a previously saved index."""
        self.store.load(self.store_path)
        # Refit embedder from loaded chunks
        self.embedder.fit(self.store.chunks)
        self.fitted = True

    def query(self, question: str, top_k: int = 3) -> dict:
        """Answer a question using the RAG pipeline."""
        if not self.fitted:
            raise RuntimeError("Index not built. Call build_index() or load_index() first.")

        # Embed the query
        query_vec = self.embedder.embed(question)

        # Retrieve relevant chunks
        retrieved = retrieve(query_vec, self.store, top_k=top_k)

        # Generate answer
        answer = generate_answer(question, retrieved)

        # Format citations
        citations = format_citations(retrieved)

        return {
            "question": question,
            "answer": answer,
            "citations": citations,
            "retrieved_chunks": retrieved,
        }

    def chat(self):
        """Interactive chat loop."""
        print("=" * 60)
        print("  RAG Chatbot — Chat with Your Documents")
        print("  Type 'quit' or 'exit' to stop")
        print("=" * 60)

        history = []
        while True:
            question = input("\n🙋 You: ").strip()
            if question.lower() in ("quit", "exit", "q"):
                print("Goodbye! 👋")
                break
            if not question:
                continue

            result = self.query(question)
            print(f"\n🤖 Answer:\n{result['answer']}")
            print(result["citations"])
            history.append(result)

        return history


# ─────────────────────────────────────────────
# DEMO / MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    rag = RAGPipeline(store_path="vector_store.json")

    # Demo with sample text files if no args provided
    if len(sys.argv) < 2:
        print("Usage: python rag.py <file1.pdf> <file2.txt> ...")
        print("\nRunning demo with sample documents...\n")

        # Create sample docs for demo
        os.makedirs("sample_docs", exist_ok=True)
        with open("sample_docs/ml_basics.txt", "w") as f:
            f.write("""
Machine learning is a subset of artificial intelligence that enables computers
to learn from data without being explicitly programmed. There are three main
types of machine learning: supervised learning, unsupervised learning, and
reinforcement learning.

Supervised learning uses labeled training data to learn a mapping from inputs
to outputs. Common algorithms include linear regression, decision trees,
random forests, and neural networks.

Unsupervised learning finds hidden patterns in unlabeled data. Clustering
algorithms like K-means and hierarchical clustering are widely used.

Reinforcement learning trains agents to make decisions by rewarding good
actions and penalizing bad ones. It is used in game playing, robotics, and
autonomous systems.

Deep learning is a subfield of machine learning that uses neural networks
with many layers. Convolutional Neural Networks (CNNs) excel at image tasks.
Recurrent Neural Networks (RNNs) handle sequential data like text.
""")

        with open("sample_docs/rag_overview.txt", "w") as f:
            f.write("""
Retrieval-Augmented Generation (RAG) is a technique that combines information
retrieval with language generation. It was introduced to improve the accuracy
of language model responses by grounding them in retrieved documents.

The RAG pipeline works as follows: first, documents are ingested and split
into chunks. Each chunk is converted into an embedding vector using a language
model. These vectors are stored in a vector database for fast similarity search.

When a user asks a question, the query is also converted into a vector. The
system retrieves the most similar document chunks using cosine similarity.
The retrieved chunks are passed along with the question to the language model
which generates a grounded answer with citations.

RAG reduces hallucinations because the model is constrained to use retrieved
context rather than relying purely on its training data. It also allows
knowledge to be updated without retraining the model.

Vector databases like FAISS, Pinecone, Chroma, and Weaviate are commonly
used to store and search embeddings efficiently at scale.
""")

        rag.build_index(["sample_docs/ml_basics.txt", "sample_docs/rag_overview.txt"])

        # Demo queries
        questions = [
            "What is supervised learning?",
            "How does RAG reduce hallucinations?",
            "What are vector databases used for?",
        ]

        print("\n--- Demo Queries ---")
        for q in questions:
            result = rag.query(q)
            print(f"\n❓ Question: {result['question']}")
            print(f"✅ Answer: {result['answer'][:300]}...")
            print(result["citations"])

    else:
        # Real usage with provided files
        files = sys.argv[1:]
        rag.build_index(files)
        rag.chat()
