"""
Streaming Live RAG — Corpus Ingestion & Indexing
Reads markdown documents from the corpus directory, splits them into section-level
chunks with Doc_ID §Section metadata, builds both dense (FAISS) and sparse (BM25)
indexes. Fully corpus-agnostic — no document names or content hardcoded.
"""
from __future__ import annotations
import re
import hashlib
import pickle
from pathlib import Path

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.config import CORPUS_PATH, EMBEDDING_MODEL, INDEX_CACHE_DIR
from app.models import RetrievedChunk


class CorpusChunk:
    """A single corpus chunk with full provenance metadata."""
    def __init__(self, chunk_id: str, doc_id: str, section: str, text: str):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.section = section
        self.text = text
        self.content_hash = hashlib.md5(text.encode()).hexdigest()

    def __repr__(self):
        return f"CorpusChunk({self.doc_id} {self.section}, {len(self.text)} chars)"


class CorpusIndex:
    """
    Hybrid dense+sparse index over the corpus.
    Dense: FAISS inner-product index with normalized embeddings.
    Sparse: BM25Okapi over tokenized chunk text.
    """

    def __init__(self):
        self.chunks: list[CorpusChunk] = []
        self.embedder: SentenceTransformer | None = None
        self.faiss_index: faiss.IndexFlatIP | None = None
        self.bm25: BM25Okapi | None = None
        self._embedding_dim: int = 0
        self._chunk_embeddings: np.ndarray | None = None

    def build(self, corpus_path: Path | None = None) -> None:
        """Ingest corpus, chunk by section, build both indexes."""
        corpus_path = corpus_path or CORPUS_PATH
        print(f"[CorpusIndex] Loading corpus from {corpus_path}")

        # 1. Load and parse all documents into section-level chunks
        self.chunks = self._parse_corpus(corpus_path)
        print(f"[CorpusIndex] Parsed {len(self.chunks)} chunks from corpus")

        # 2. Check for cached index
        cache_file = INDEX_CACHE_DIR / "corpus_index.pkl"
        corpus_hash = self._corpus_hash(corpus_path)
        if cache_file.exists():
            try:
                cached = pickle.loads(cache_file.read_bytes())
                if cached.get("corpus_hash") == corpus_hash:
                    print("[CorpusIndex] Loading cached index")
                    self._load_from_cache(cached)
                    return
            except Exception:
                pass  # Rebuild on any cache error

        # 3. Build embeddings (dense index)
        print(f"[CorpusIndex] Loading embedding model: {EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self._embedding_dim = self.embedder.get_sentence_embedding_dimension()

        chunk_texts = [c.text for c in self.chunks]
        print(f"[CorpusIndex] Computing embeddings for {len(chunk_texts)} chunks...")
        self._chunk_embeddings = self.embedder.encode(
            chunk_texts, normalize_embeddings=True, show_progress_bar=True
        )

        # Build FAISS index (inner product on normalized vectors = cosine similarity)
        self.faiss_index = faiss.IndexFlatIP(self._embedding_dim)
        self.faiss_index.add(self._chunk_embeddings.astype(np.float32))
        print(f"[CorpusIndex] FAISS index built: {self.faiss_index.ntotal} vectors")

        # 4. Build BM25 (sparse index)
        tokenized = [self._tokenize(t) for t in chunk_texts]
        self.bm25 = BM25Okapi(tokenized)
        print(f"[CorpusIndex] BM25 index built")

        # 5. Cache
        self._save_to_cache(cache_file, corpus_hash)
        print("[CorpusIndex] Index cached for future use")

    def _parse_corpus(self, corpus_path: Path) -> list[CorpusChunk]:
        """Parse markdown documents into section-level chunks."""
        chunks = []
        for doc_file in sorted(corpus_path.glob("*.md")):
            content = doc_file.read_text(encoding="utf-8")
            doc_id = self._extract_doc_id(content, doc_file.stem)
            sections = self._split_sections(content, doc_id)
            chunks.extend(sections)
        return chunks

    def _extract_doc_id(self, content: str, fallback: str) -> str:
        """Extract Doc_ID from the first heading line."""
        match = re.search(r"^#\s+(Doc_\d+)", content, re.MULTILINE)
        if match:
            return match.group(1)
        # Fallback: extract from filename pattern doc_XX_...
        match = re.search(r"doc_(\d+)", fallback)
        if match:
            return f"Doc_{match.group(1).lstrip('0') or '0'}"
        return fallback

    def _split_sections(self, content: str, doc_id: str) -> list[CorpusChunk]:
        """Split a document into section-level chunks based on ## §N headers."""
        chunks = []
        # Find all section headers: ## §N Title
        section_pattern = re.compile(r"^##\s+§(\d+)\s+(.+)$", re.MULTILINE)
        matches = list(section_pattern.finditer(content))

        if not matches:
            # No sections found — treat entire document as one chunk
            chunk_id = f"{doc_id}_full"
            chunks.append(CorpusChunk(chunk_id, doc_id, "§full", content.strip()))
            return chunks

        for i, match in enumerate(matches):
            section_num = match.group(1)
            section_name = match.group(2).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            section_text = content[start:end].strip()

            # Include the section header in the text for better retrieval context
            full_text = f"{doc_id} §{section_num} {section_name}\n{section_text}"
            chunk_id = f"{doc_id}_s{section_num}"
            section_ref = f"§{section_num}"

            chunks.append(CorpusChunk(chunk_id, doc_id, section_ref, full_text))

        return chunks

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace + lowercase tokenization for BM25."""
        text = re.sub(r"[^\w\s\-]", " ", text.lower())
        return text.split()

    def _corpus_hash(self, corpus_path: Path) -> str:
        """Hash of all document contents for cache invalidation."""
        h = hashlib.md5()
        for f in sorted(corpus_path.glob("*.md")):
            h.update(f.read_bytes())
        return h.hexdigest()

    def _save_to_cache(self, cache_file: Path, corpus_hash: str):
        """Serialize index to disk."""
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "corpus_hash": corpus_hash,
            "chunks": [(c.chunk_id, c.doc_id, c.section, c.text) for c in self.chunks],
            "embeddings": self._chunk_embeddings,
            "embedding_dim": self._embedding_dim,
        }
        cache_file.write_bytes(pickle.dumps(data))

    def _load_from_cache(self, cached: dict):
        """Restore index from cache."""
        self.chunks = [
            CorpusChunk(cid, did, sec, txt)
            for cid, did, sec, txt in cached["chunks"]
        ]
        self._chunk_embeddings = cached["embeddings"]
        self._embedding_dim = cached["embedding_dim"]

        # Rebuild FAISS from cached embeddings
        self.faiss_index = faiss.IndexFlatIP(self._embedding_dim)
        self.faiss_index.add(self._chunk_embeddings.astype(np.float32))

        # Rebuild BM25
        tokenized = [self._tokenize(c.text) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)

        # Load embedder lazily (needed for query encoding)
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

    def embed_query(self, query: str) -> np.ndarray:
        """Encode a query string into a normalized embedding vector."""
        if self.embedder is None:
            self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        emb = self.embedder.encode([query], normalize_embeddings=True)
        return emb[0]

    def dense_search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """Dense (embedding cosine similarity) retrieval. Returns (idx, score) pairs."""
        q_emb = self.embed_query(query).reshape(1, -1).astype(np.float32)
        scores, indices = self.faiss_index.search(q_emb, top_k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0:
                results.append((int(idx), float(score)))
        return results

    def sparse_search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """Sparse (BM25) retrieval. Returns (idx, score) pairs."""
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((int(idx), float(scores[idx])))
        return results

    def get_chunk(self, idx: int) -> CorpusChunk:
        """Get a chunk by index."""
        return self.chunks[idx]

    def get_chunk_by_id(self, chunk_id: str) -> CorpusChunk | None:
        """Get a chunk by its ID."""
        for c in self.chunks:
            if c.chunk_id == chunk_id:
                return c
        return None

    def find_chunks_by_doc_section(self, doc_id: str, section: str) -> list[CorpusChunk]:
        """Find chunks matching a Doc_ID §Section citation."""
        return [c for c in self.chunks if c.doc_id == doc_id and c.section == section]


# Singleton
corpus_index = CorpusIndex()
