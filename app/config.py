"""
Streaming Live RAG — Configuration
Central configuration loaded from environment variables with sensible defaults.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = PROJECT_ROOT / os.getenv("CORPUS_PATH", "streaming-rag-dev-demo-corpus/corpus/documents")
INDEX_CACHE_DIR = PROJECT_ROOT / ".index_cache"

# ─── Server ──────────────────────────────────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# ─── LLM ─────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

# ─── Embeddings ──────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ─── Controller Thresholds ───────────────────────────────────────────────────
CONTROLLER_DRIFT_THRESHOLD = float(os.getenv("CONTROLLER_DRIFT_THRESHOLD", "0.15"))
CONTROLLER_DEBOUNCE_MS = int(os.getenv("CONTROLLER_DEBOUNCE_MS", "300"))
CONTROLLER_MAX_WAIT_CHUNKS = int(os.getenv("CONTROLLER_MAX_WAIT_CHUNKS", "10"))

# ─── Retrieval ───────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "20"))
RETRIEVAL_FUSION_K = int(os.getenv("RETRIEVAL_FUSION_K", "60"))
RETRIEVAL_MIN_RELEVANCE = float(os.getenv("RETRIEVAL_MIN_RELEVANCE", "0.25"))
MAX_SUB_QUERIES = int(os.getenv("MAX_SUB_QUERIES", "5"))

# ─── Synthesis ───────────────────────────────────────────────────────────────
CITATION_ENTAILMENT_THRESHOLD = float(os.getenv("CITATION_ENTAILMENT_THRESHOLD", "0.3"))
