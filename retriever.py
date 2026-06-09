"""
retriever.py — Embedding, Vector Store & Retrieval for Professor Reviews RAG
Spec: planning.md (Retrieval Approach section)
  - Embedding model : all-MiniLM-L6-v2 (384-dim, local)
  - Vector store    : ChromaDB (persistent, saved to ./chroma_db)
  - Top-k           : 5 chunks per query
Run: python retriever.py
"""

import json
import sys
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

CHUNKS_FILE    = "chunks.json"
CHROMA_PATH    = "./chroma_db"
COLLECTION     = "professor_reviews"
EMBED_MODEL    = "all-MiniLM-L6-v2"
DEFAULT_TOP_K  = 5

# Queries from evaluation plan (planning.md §Evaluation Plan)
EVAL_QUERIES = [
    {
        "query": "What do students say about Stefano Lonardi's exam grading?",
        "expected_source": "Stefano_Lonardi.txt",
    },
    {
        "query": "Does Derek Mkhaiel give good feedback on assignments?",
        "expected_source": "Derek_Mkhaiel.txt",
    },
    {
        "query": "Is Thomas Kuhlman's physics class curved?",
        "expected_source": "Thomas_Kuhlman.txt",
    },
]


# ──────────────────────────────────────────────
# Step 1 — Load chunks from chunks.json
# ──────────────────────────────────────────────

def load_chunks(path: str) -> list[dict]:
    """
    Load chunk records produced by pipeline.py.
    Each record: {chunk_id, source, chunk_index, text}
    """
    chunks_path = Path(path)
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"'{path}' not found. Run pipeline.py first to generate chunks."
        )
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)
    if not chunks:
        raise ValueError(f"'{path}' is empty — re-run pipeline.py.")
    return chunks


# ──────────────────────────────────────────────
# Step 2 — Initialize embedding model
# ──────────────────────────────────────────────

def load_embedding_model(model_name: str = EMBED_MODEL) -> SentenceTransformer:
    """
    Load the sentence-transformers model.
    First run downloads ~90MB; subsequent runs use the local cache.
    """
    print(f"  Loading embedding model '{model_name}'...")
    t0 = time.time()
    model = SentenceTransformer(model_name)
    print(f"  Model ready ({time.time() - t0:.1f}s)")
    return model


# ──────────────────────────────────────────────
# Step 3 — Initialize ChromaDB (persistent)
# ──────────────────────────────────────────────

def get_collection(chroma_path: str = CHROMA_PATH, collection_name: str = COLLECTION):
    """
    Create or open a persistent ChromaDB collection.
    Saves to disk so embeddings survive between runs.
    """
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(
        name=collection_name,
        # cosine distance suits sentence-transformer embeddings
        metadata={"hnsw:space": "cosine"},
    )
    return collection


# ──────────────────────────────────────────────
# Step 4 — Embed chunks and store in ChromaDB
#           (skipped if collection already has data)
# ──────────────────────────────────────────────

def embed_and_store(
    chunks: list[dict],
    model: SentenceTransformer,
    collection,
    batch_size: int = 64,
) -> None:
    """
    Embed all chunks and upsert into ChromaDB in batches.
    Batching keeps memory usage manageable for large corpora.
    """
    total = len(chunks)
    print(f"  Embedding {total} chunks in batches of {batch_size}...")
    t0 = time.time()

    for start in range(0, total, batch_size):
        batch = chunks[start : start + batch_size]
        texts      = [c["text"]     for c in batch]
        ids        = [c["chunk_id"] for c in batch]
        metadatas  = [{"source": c["source"], "chunk_id": c["chunk_id"]} for c in batch]

        # Generate 384-dim vectors for the batch
        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        # Upsert: safe to re-run; won't create duplicates
        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        done = min(start + batch_size, total)
        print(f"    Stored {done}/{total} chunks...", end="\r")

    elapsed = time.time() - t0
    print(f"\n  Done. {total} chunks embedded and stored in {elapsed:.1f}s "
          f"({total / elapsed:.0f} chunks/sec)")


# ──────────────────────────────────────────────
# Step 5 — Retrieval function
# ──────────────────────────────────────────────

def retrieve(
    query: str,
    model: SentenceTransformer,
    collection,
    k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """
    Embed the query and return the top-k most similar chunks.

    Returns a list of dicts:
      {rank, text, source, chunk_id, distance}

    Distance is cosine distance (0 = identical, 1 = orthogonal).
    Lower distance → more relevant.
    """
    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for rank, (doc, meta, dist) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ),
        start=1,
    ):
        hits.append({
            "rank":     rank,
            "text":     doc,
            "source":   meta["source"],
            "chunk_id": meta["chunk_id"],
            "distance": round(dist, 4),
        })

    return hits


# ──────────────────────────────────────────────
# Step 6 — Evaluation helper
# ──────────────────────────────────────────────

def assess_relevance(hit: dict, expected_source: str, distance_threshold: float = 0.6) -> str:
    """
    Simple heuristic: a result is RELEVANT if the distance is below the
    threshold AND it comes from the expected source file.
    """
    from_right_source = hit["source"] == expected_source
    close_enough      = hit["distance"] < distance_threshold
    if from_right_source and close_enough:
        return "✅ RELEVANT"
    elif from_right_source:
        return "⚠️  RIGHT SOURCE, HIGH DISTANCE"
    elif close_enough:
        return "⚠️  CLOSE MATCH, WRONG SOURCE"
    else:
        return "❌ NOT RELEVANT"


# ──────────────────────────────────────────────
# Step 7 — Pretty-print results
# ──────────────────────────────────────────────

def print_results(query: str, hits: list[dict], expected_source: str, top_n: int = 3) -> None:
    bar = "─" * 60
    print(f"\n{'═'*60}")
    print(f"QUERY: {query}")
    print(f"Expected source: {expected_source}")
    print(bar)

    for hit in hits[:top_n]:
        preview = hit["text"][:200].replace("\n", " ")
        ellipsis = "..." if len(hit["text"]) > 200 else ""
        relevance = assess_relevance(hit, expected_source)

        print(f"  Rank {hit['rank']}  |  dist={hit['distance']:.4f}  |  {relevance}")
        print(f"  Source   : {hit['source']}")
        print(f"  Chunk ID : {hit['chunk_id']}")
        print(f"  Preview  : {preview}{ellipsis}")
        print(bar)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Professor Reviews RAG — Embedding & Retrieval (Milestone 4)")
    print("=" * 60)

    # ── Load chunks ───────────────────────────
    print(f"\n[1/4] Loading chunks from '{CHUNKS_FILE}'...")
    try:
        chunks = load_chunks(CHUNKS_FILE)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    print(f"       {len(chunks)} chunks loaded.")

    # ── Embedding model ───────────────────────
    print("\n[2/4] Initializing embedding model...")
    model = load_embedding_model(EMBED_MODEL)

    # ── ChromaDB ──────────────────────────────
    print(f"\n[3/4] Connecting to ChromaDB at '{CHROMA_PATH}'...")
    collection = get_collection(CHROMA_PATH, COLLECTION)
    existing   = collection.count()
    print(f"       Collection '{COLLECTION}' has {existing} existing records.")

    if existing >= len(chunks):
        # Already fully embedded — skip to avoid paying embedding cost every run
        print("       Collection is up to date — skipping embedding step.")
    else:
        if existing > 0:
            print(f"       Partial data found ({existing}/{len(chunks)}). Re-embedding all...")
        else:
            print("       Empty collection — embedding all chunks now.")
        t_embed_start = time.time()
        embed_and_store(chunks, model, collection)
        print(f"       Total embedding wall time: {time.time() - t_embed_start:.1f}s")

    # ── Retrieval evaluation ──────────────────
    print("\n[4/4] Running evaluation queries...")
    for item in EVAL_QUERIES:
        t0 = time.time()
        hits = retrieve(item["query"], model, collection, k=DEFAULT_TOP_K)
        elapsed_ms = (time.time() - t0) * 1000
        print_results(item["query"], hits, item["expected_source"])
        print(f"  ⏱  Query time: {elapsed_ms:.1f}ms")

    print(f"\n{'═'*60}")
    print("Evaluation complete.")
    print("Next step → Milestone 5: generation with Groq llama-3.3-70b-versatile.")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()