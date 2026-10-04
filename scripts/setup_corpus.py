"""
setup_corpus.py — Swap in a custom document corpus

Usage:
    python scripts/setup_corpus.py path/to/your/documents

This script:
  1. Validates the directory contains .txt or .md files
  2. Updates CORPUS_PATH in your .env file
  3. Clears the index cache so the server rebuilds on next start
  4. Parses and previews the chunks that will be indexed
  5. Prints the command to start the server

After running this, just start the server as normal:
    uvicorn app.api.server:app --host 0.0.0.0 --port 8000
"""
import sys
import os
import re
import shutil
from pathlib import Path


def parse_chunks(doc_path: Path) -> list[str]:
    """Split a document into chunks the same way the corpus indexer does."""
    text = doc_path.read_text(encoding="utf-8", errors="ignore")
    # Split on section markers or double blank lines
    chunks = re.split(r"\n§|\n{2,}", text)
    chunks = [c.strip() for c in chunks if len(c.strip()) > 40]
    return chunks


def update_env(corpus_path: str):
    env_file = Path(".env")
    if not env_file.exists():
        example = Path(".env.example")
        if example.exists():
            shutil.copy(example, env_file)
            print("  Created .env from .env.example")
        else:
            env_file.write_text(f"CORPUS_PATH={corpus_path}\n")
            print("  Created new .env")
            return

    content = env_file.read_text()
    if "CORPUS_PATH=" in content:
        content = re.sub(r"^CORPUS_PATH=.*$", lambda m: f"CORPUS_PATH={corpus_path}", content, flags=re.MULTILINE)
    else:
        content += f"\nCORPUS_PATH={corpus_path}\n"
    env_file.write_text(content)


def clear_index_cache():
    cache = Path(".index_cache")
    if cache.exists():
        shutil.rmtree(cache)
        print("  Cleared .index_cache/ (will rebuild on next server start)")
    else:
        print("  No existing cache to clear")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/setup_corpus.py path/to/your/documents")
        sys.exit(1)

    corpus_dir = Path(sys.argv[1]).resolve()

    if not corpus_dir.exists():
        print(f"Error: Directory not found: {corpus_dir}")
        sys.exit(1)

    if not corpus_dir.is_dir():
        print(f"Error: Not a directory: {corpus_dir}")
        sys.exit(1)

    # Find all documents
    docs = sorted([
        f for f in corpus_dir.iterdir()
        if f.suffix in (".txt", ".md") and f.is_file()
    ])

    if not docs:
        print(f"Error: No .txt or .md files found in {corpus_dir}")
        print("  Place your documents as plain text files (.txt or .md) in that directory.")
        sys.exit(1)

    print(f"\nStreaming Live RAG — Corpus Setup")
    print("=" * 50)
    print(f"  Document directory: {corpus_dir}")
    print(f"  Documents found:    {len(docs)}")

    # Parse and count chunks
    total_chunks = 0
    print("\n  Document breakdown:")
    for doc in docs:
        chunks = parse_chunks(doc)
        total_chunks += len(chunks)
        print(f"    {doc.name:<50} {len(chunks):>3} chunks")

    print(f"\n  Total chunks to index: {total_chunks}")

    if total_chunks < 5:
        print("\n  Warning: Very few chunks detected.")
        print("  Make sure your documents have meaningful content (>40 characters per section).")

    # Apply changes
    print("\n  Applying configuration...")
    update_env(str(corpus_dir))
    print(f"  Updated .env: CORPUS_PATH={corpus_dir}")
    clear_index_cache()

    print("\n  Done. Start the server with:")
    print("    uvicorn app.api.server:app --host 0.0.0.0 --port 8000")
    print("  or:")
    print("    docker compose up --build")
    print()
    print("  The FAISS and BM25 indexes will rebuild automatically on first startup.")
    print("  This takes roughly 10-30 seconds depending on corpus size.")
    print()
    print("  To run the benchmark evaluator against your corpus, create a benchmark")
    print("  JSON file following the schema in:")
    print("    streaming-rag-dev-demo-corpus/corpus/benchmarks/benchmark_suite.json")
    print("  and run:")
    print("    python scripts/run_benchmark_eval.py")
    print()


if __name__ == "__main__":
    main()
