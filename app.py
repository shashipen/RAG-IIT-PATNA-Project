"""
RAG From Scratch — Streamlit Web Demo
IIT Patna AIML Certification — Project 2

Run with:  streamlit run app.py
Opens at:  http://localhost:8501
"""

import os
import re
import json
import math
import tempfile
import numpy as np
from pathlib import Path
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG From Scratch — IIT Patna",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main background */
.stApp { background-color: #f0f4f8; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a2b5e 0%, #0d7377 100%);
}
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] .stButton > button {
    background: white; color: #1a2b5e !important;
    font-weight: bold; border-radius: 8px; width: 100%;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #f0a500; color: white !important;
}

/* Chat messages */
.user-msg {
    background: #1a2b5e; color: white;
    padding: 12px 16px; border-radius: 12px 12px 2px 12px;
    margin: 6px 0; max-width: 80%; margin-left: auto;
    font-size: 15px;
}
.bot-msg {
    background: white; color: #222;
    padding: 12px 16px; border-radius: 2px 12px 12px 12px;
    margin: 6px 0; max-width: 85%;
    border-left: 4px solid #0d7377;
    font-size: 15px; line-height: 1.6;
}
.citation-box {
    background: #e8f4f8; border: 1px solid #0d7377;
    border-radius: 8px; padding: 10px 14px;
    font-size: 13px; color: #444; margin-top: 8px;
}
.score-badge {
    display: inline-block;
    background: #0d7377; color: white;
    padding: 2px 8px; border-radius: 12px;
    font-size: 11px; font-weight: bold;
}
.step-card {
    background: white; border-radius: 10px;
    padding: 12px; margin: 4px 0;
    border-left: 4px solid #f0a500;
    font-size: 13px;
}
.header-banner {
    background: linear-gradient(90deg, #1a2b5e, #0d7377);
    color: white; padding: 20px 28px; border-radius: 12px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  RAG PIPELINE (same as rag.py — self-contained for Streamlit)
# ══════════════════════════════════════════════════════════════════

def ingest_document(filepath: str) -> str:
    ext = Path(filepath).suffix.lower()
    if ext == ".txt":
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    elif ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(filepath)
        return "".join(page.extract_text() or "" for page in reader.pages)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def chunk_text(text: str, chunk_size=80, overlap=15) -> list:
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        chunks.append(" ".join(words[start: start + chunk_size]))
        start += chunk_size - overlap
    return chunks


class TFIDFEmbedder:
    def __init__(self, vocab_size=2000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.idf = None

    def _tok(self, text):
        return re.sub(r"[^a-z0-9\s]", "", text.lower()).split()

    def fit(self, corpus):
        df = {}
        for doc in corpus:
            for t in set(self._tok(doc)):
                df[t] = df.get(t, 0) + 1
        top = sorted(df.items(), key=lambda x: -x[1])[:self.vocab_size]
        self.vocab = {w: i for i, (w, _) in enumerate(top)}
        N = len(corpus)
        self.idf = np.array([
            math.log((N + 1) / (df.get(w, 0) + 1)) + 1
            for w in self.vocab
        ])

    def embed(self, text):
        tokens = self._tok(text)
        total = len(tokens) or 1
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        vec = np.zeros(len(self.vocab))
        for word, count in tf.items():
            if word in self.vocab:
                vec[self.vocab[word]] = (count / total) * self.idf[self.vocab[word]]
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


class VectorStore:
    def __init__(self):
        self.chunks, self.metadata, self.vectors = [], [], []

    def add(self, chunk, meta, vec):
        self.chunks.append(chunk)
        self.metadata.append(meta)
        self.vectors.append(vec)

    def size(self):
        return len(self.chunks)


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def retrieve(query_vec, store, top_k=3):
    scored = sorted(
        [(cosine_sim(query_vec, v), i) for i, v in enumerate(store.vectors)],
        reverse=True
    )
    return [{"chunk": store.chunks[i], "metadata": store.metadata[i],
             "score": round(s, 4)} for s, i in scored[:top_k]]


def generate_answer(query, retrieved):
    if not retrieved:
        return "I could not find relevant information in the documents."
    stop = {"what","is","are","the","a","an","how","why","when","where",
            "who","does","do","in","of","to","?","i","me","my","tell","about"}
    keywords = set(query.lower().split()) - stop
    if not keywords:
        return retrieved[0]["chunk"][:500] + "..."

    best = []
    for item in retrieved:
        sentences = re.split(r'(?<=[.!?])\s+', item["chunk"])
        for sent in sentences:
            sl = sent.lower()
            # Count exact keyword matches using word boundaries
            hits = sum(1 for kw in keywords
                       if re.search(r'\b' + re.escape(kw) + r'\b', sl))
            if hits == 0:
                continue
            # Big bonus if keyword appears in first 50 chars (definition sentence)
            bonus = sum(3 for kw in keywords
                        if re.search(r'\b' + re.escape(kw) + r'\b', sl[:50]))
            # Penalty: sentence contains "un" + keyword (e.g. unsupervised when asking supervised)
            penalty = sum(2 for kw in keywords
                          if re.search(r'\bun' + re.escape(kw) + r'\b', sl)
                          and not re.search(r'\bun' + re.escape(kw) + r'\b',
                                            query.lower()))
            total = hits + bonus - penalty
            if total > 0:
                best.append((total, sent.strip()))

    seen, unique = set(), []
    for score, sent in sorted(best, reverse=True):
        if sent not in seen:
            seen.add(sent)
            unique.append((score, sent))
    top = [s for _, s in unique[:3]]
    return " ".join(top) if top else retrieved[0]["chunk"][:500] + "..."


# ══════════════════════════════════════════════════════════════════
#  SAMPLE DOCUMENTS
# ══════════════════════════════════════════════════════════════════

SAMPLE_DOCS = {
    "ml_basics.txt": """
Machine learning is a subset of artificial intelligence that enables computers
to learn from data without being explicitly programmed. There are three main
types of machine learning: supervised learning, unsupervised learning, and
reinforcement learning.

Supervised learning uses labeled training data to learn a mapping from inputs
to outputs. Common algorithms include linear regression, decision trees,
random forests, support vector machines, and neural networks. Evaluation
metrics include accuracy, precision, recall, and F1-score.

Unsupervised learning finds hidden patterns in unlabeled data. Clustering
algorithms like K-means and hierarchical clustering group similar data points.
Dimensionality reduction techniques like PCA and t-SNE visualise high-dimensional data.

Reinforcement learning trains agents to make decisions by rewarding good
actions and penalizing bad ones. It is used in game playing, robotics, and
autonomous driving systems. Key algorithms include Q-learning and PPO.

Deep learning is a subfield using neural networks with many layers.
Convolutional Neural Networks (CNNs) excel at image tasks.
Recurrent Neural Networks (RNNs) and Transformers handle sequential data like text.
Transfer learning allows pre-trained models to be fine-tuned on new tasks.
""",
    "rag_overview.txt": """
Retrieval-Augmented Generation (RAG) is a technique that combines information
retrieval with language generation to improve accuracy and reduce hallucinations.

The RAG pipeline works in seven steps: First, documents are ingested and text
is extracted. Second, text is split into overlapping chunks of fixed word count.
Third, each chunk is converted into an embedding vector using a language model
or TF-IDF. Fourth, these vectors are stored in a vector database along with metadata.

Fifth, when a user asks a question the query is embedded and cosine similarity
is used to find the most relevant chunks. Sixth, the retrieved chunks are used
to generate a grounded answer. Seventh, citations show which document and chunk
was used to produce the answer.

RAG reduces hallucinations because the model is constrained to use retrieved
context rather than relying purely on its training data. It also allows
knowledge to be updated without retraining the model.

Vector databases like FAISS, Pinecone, Chroma, and Weaviate store embeddings
efficiently. TF-IDF is a simpler alternative that requires no GPU or API key.
Cosine similarity measures the angle between two vectors to find the closest match.
""",
    "cloud_computing.txt": """
Cloud computing delivers computing services including servers, storage, databases,
networking, software, analytics, and intelligence over the internet.

Amazon Web Services (AWS) is the world's largest cloud platform offering over
200 services including EC2 for virtual machines, S3 for object storage,
Lambda for serverless functions, and SageMaker for machine learning.

Microsoft Azure is the second largest cloud provider. Key services include
Azure Virtual Machines, Azure Blob Storage, Azure Functions, and Azure ML.
Azure integrates tightly with Microsoft enterprise products like Office 365.

Google Cloud Platform (GCP) offers Compute Engine, Cloud Storage, BigQuery
for data analytics, and Vertex AI for machine learning. GCP is known for
its data and AI capabilities and competitive pricing.

The three main cloud service models are IaaS (Infrastructure as a Service),
PaaS (Platform as a Service), and SaaS (Software as a Service).
Cloud security involves identity management, encryption, and compliance frameworks
like ISO 27001, SOC 2, and GDPR. Zero-trust architecture is a modern approach
to cloud security that verifies every request regardless of origin.
""",
}


# ══════════════════════════════════════════════════════════════════
#  SESSION STATE — persist pipeline across reruns
# ══════════════════════════════════════════════════════════════════

if "store" not in st.session_state:
    st.session_state.store = VectorStore()
if "embedder" not in st.session_state:
    st.session_state.embedder = TFIDFEmbedder()
if "fitted" not in st.session_state:
    st.session_state.fitted = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
if "pipeline_log" not in st.session_state:
    st.session_state.pipeline_log = []


def build_index(docs: dict):
    """docs = {filename: text_content}"""
    store = VectorStore()
    log = []
    all_chunks, all_meta = [], []

    for fname, text in docs.items():
        chunks = chunk_text(text, chunk_size=80, overlap=15)
        log.append(f"✅ **{fname}** → {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_meta.append({"filename": fname, "chunk_id": i})

    embedder = TFIDFEmbedder(vocab_size=2000)
    embedder.fit(all_chunks)

    for chunk, meta in zip(all_chunks, all_meta):
        vec = embedder.embed(chunk)
        store.add(chunk, meta, vec)

    st.session_state.store = store
    st.session_state.embedder = embedder
    st.session_state.fitted = True
    st.session_state.indexed_files = list(docs.keys())
    st.session_state.pipeline_log = log
    st.session_state.chat_history = []


# ══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🧠 RAG Pipeline")
    st.markdown("**IIT Patna — AIML Project 2**")
    st.markdown("---")

    # ── Load sample docs ──
    st.markdown("### 📚 Sample Documents")
    st.markdown("3 pre-loaded demo documents:")
    for name in SAMPLE_DOCS:
        st.markdown(f"- `{name}`")

    if st.button("🚀 Load Sample Docs & Build Index"):
        with st.spinner("Building index..."):
            build_index(SAMPLE_DOCS)
        st.success(f"✅ Index ready! {st.session_state.store.size()} chunks indexed.")

    st.markdown("---")

    # ── Upload your own docs ──
    st.markdown("### 📂 Upload Your Documents")
    uploaded = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if uploaded and st.button("📥 Index Uploaded Files"):
        docs = dict(SAMPLE_DOCS)  # keep sample docs too
        tmp_dir = tempfile.mkdtemp()
        for uf in uploaded:
            tmp_path = os.path.join(tmp_dir, uf.name)
            with open(tmp_path, "wb") as f:
                f.write(uf.read())
            try:
                text = ingest_document(tmp_path)
                docs[uf.name] = text
            except Exception as e:
                st.error(f"Error reading {uf.name}: {e}")
        with st.spinner("Building index..."):
            build_index(docs)
        st.success(f"✅ Index ready! {st.session_state.store.size()} chunks indexed.")

    st.markdown("---")

    # ── Status ──
    st.markdown("### 📊 Index Status")
    if st.session_state.fitted:
        st.success(f"✅ **Ready** — {st.session_state.store.size()} chunks")
        for f in st.session_state.indexed_files:
            st.markdown(f"  • `{f}`")
    else:
        st.warning("⚠️ No index built yet.\nClick 'Load Sample Docs' above.")

    if st.session_state.pipeline_log:
        st.markdown("**Build log:**")
        for line in st.session_state.pipeline_log:
            st.markdown(line)

    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()


# ══════════════════════════════════════════════════════════════════
#  MAIN AREA
# ══════════════════════════════════════════════════════════════════

st.markdown("""
<div class="header-banner">
  <h2 style="margin:0;color:white">🧠 RAG From Scratch — Chat with Your Documents</h2>
  <p style="margin:4px 0 0 0;opacity:0.85">IIT Patna AIML Certification · Project 2 · Powered by TF-IDF + Cosine Similarity</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["💬 Chat", "🔬 Pipeline Explorer", "ℹ️ About"])

# ─────────────────────────────────────────────────────────────────
# TAB 1 — CHAT
# ─────────────────────────────────────────────────────────────────
with tab1:
    if not st.session_state.fitted:
        st.info("👈 Click **'Load Sample Docs & Build Index'** in the sidebar to get started!")
    else:
        # Suggested questions
        st.markdown("**💡 Try these questions:**")
        cols = st.columns(3)
        suggestions = [
            "What is supervised learning?",
            "How does RAG reduce hallucinations?",
            "What services does AWS offer?",
            "What is cosine similarity?",
            "Explain reinforcement learning",
            "What is zero-trust security?",
        ]
        for i, sug in enumerate(suggestions):
            if cols[i % 3].button(sug, key=f"sug_{i}"):
                st.session_state._pending_question = sug

        st.markdown("---")

        # Chat history display
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f'<div class="user-msg">🙋 {msg["content"]}</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="bot-msg">🤖 {msg["content"]}</div>',
                                unsafe_allow_html=True)
                    if "citations" in msg:
                        cit_html = '<div class="citation-box"><b>📚 Sources used:</b><br>'
                        for c in msg["citations"]:
                            score_pct = int(c["score"] * 100)
                            cit_html += (
                                f'&nbsp;&nbsp;• <b>{c["metadata"]["filename"]}</b> — '
                                f'Chunk #{c["metadata"]["chunk_id"]} &nbsp;'
                                f'<span class="score-badge">score: {c["score"]}</span><br>'
                                f'&nbsp;&nbsp;&nbsp;&nbsp;<i style="font-size:12px">'
                                f'{c["chunk"][:120]}...</i><br>'
                            )
                        cit_html += "</div>"
                        st.markdown(cit_html, unsafe_allow_html=True)

        # Input
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            question = st.text_input(
                "Ask a question about your documents:",
                value=st.session_state.pop("_pending_question", ""),
                placeholder="e.g. What is supervised learning?",
                label_visibility="collapsed",
            )
        with col_btn:
            ask = st.button("Send 📨", use_container_width=True)

        if ask and question.strip():
            q = question.strip()
            st.session_state.chat_history.append({"role": "user", "content": q})

            query_vec = st.session_state.embedder.embed(q)
            retrieved = retrieve(query_vec, st.session_state.store, top_k=3)
            answer = generate_answer(q, retrieved)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "citations": retrieved,
            })
            st.rerun()


# ─────────────────────────────────────────────────────────────────
# TAB 2 — PIPELINE EXPLORER
# ─────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 🔬 Live Pipeline Explorer")
    st.markdown("Enter a question to see each RAG step executed in real-time:")

    explore_q = st.text_input("Enter question to trace:", placeholder="What is deep learning?",
                              key="explore_q")

    if st.button("▶ Run Pipeline Step-by-Step") and explore_q and st.session_state.fitted:
        st.markdown("---")

        with st.expander("**Step 1 — Document Ingestion** ✅", expanded=False):
            st.markdown(f"**Indexed files:** {', '.join(st.session_state.indexed_files)}")
            st.markdown(f"**Total chunks in store:** {st.session_state.store.size()}")

        with st.expander("**Step 2 — Chunking** ✅", expanded=False):
            st.markdown("Each document was split into **300-word overlapping windows** (50-word overlap).")
            sample = st.session_state.store.chunks[0][:300] + "..."
            st.code(f"Sample chunk:\n{sample}")

        with st.expander("**Step 3 — Embedding** ✅", expanded=False):
            st.markdown(f"Vocabulary size: **{len(st.session_state.embedder.vocab)} words**")
            query_vec = st.session_state.embedder.embed(explore_q)
            st.markdown(f"Query vector shape: **({len(query_vec)},)**")
            st.markdown(f"Query vector (first 10 values): `{np.round(query_vec[:10], 4).tolist()}`")

        with st.expander("**Step 4 — Vector Store** ✅", expanded=False):
            st.markdown(f"**{st.session_state.store.size()} vectors** stored in memory")
            st.json(st.session_state.store.metadata[:3])

        with st.expander("**Step 5 — Retrieval (Cosine Similarity)** ✅", expanded=True):
            query_vec = st.session_state.embedder.embed(explore_q)
            retrieved = retrieve(query_vec, st.session_state.store, top_k=3)
            for i, r in enumerate(retrieved, 1):
                score_color = "green" if r["score"] > 0.3 else "orange" if r["score"] > 0.1 else "red"
                st.markdown(f"**Rank {i}** — `{r['metadata']['filename']}` "
                            f"(Chunk #{r['metadata']['chunk_id']}) — "
                            f"Score: :{score_color}[**{r['score']}**]")
                st.caption(r["chunk"][:200] + "...")

        with st.expander("**Step 6 — Answer Generation** ✅", expanded=True):
            answer = generate_answer(explore_q, retrieved)
            st.success(f"**Generated Answer:**\n\n{answer}")

        with st.expander("**Step 7 — Citations** ✅", expanded=True):
            for i, r in enumerate(retrieved, 1):
                st.markdown(f"[{i}] **{r['metadata']['filename']}** — "
                            f"Chunk #{r['metadata']['chunk_id']} — "
                            f"Relevance: `{r['score']}`")

    elif explore_q and not st.session_state.fitted:
        st.warning("Please build the index first (sidebar → Load Sample Docs).")


# ─────────────────────────────────────────────────────────────────
# TAB 3 — ABOUT
# ─────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### ℹ️ About This Project")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Project:** RAG From Scratch  
**Course:** IIT Patna AIML Certification  
**Tech Stack:** Python 3.10+, NumPy, pypdf, Streamlit  
**No API key required** — fully local pipeline
        """)
    with col2:
        st.markdown("""
**Pipeline Steps:**
1. 📄 Ingest (PDF/TXT)
2. ✂️ Chunk (300-word windows)
3. 🔢 Embed (TF-IDF vectors)
4. 💾 Store (in-memory + JSON)
5. 🔍 Retrieve (cosine similarity)
6. 💬 Generate (extractive)
7. 📚 Cite (file + chunk + score)
        """)

    st.markdown("---")
    st.markdown("### 🗂️ File Structure")
    st.code("""
rag_project/
├── app.py              ← This Streamlit web app
├── rag.py              ← Core pipeline (CLI version)
├── requirements.txt    ← Dependencies
└── README.md           ← Setup guide
    """)

    st.markdown("### ▶️ How to Run")
    st.code("""
# Install dependencies
pip install -r requirements.txt

# Launch the web app
streamlit run app.py

# Opens automatically at:
# http://localhost:8501
    """, language="bash")
