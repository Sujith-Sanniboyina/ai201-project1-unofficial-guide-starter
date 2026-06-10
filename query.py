"""
query.py — Generation component for Professor Reviews RAG (Milestone 5)
Connects retriever.py (ChromaDB + embeddings) to Groq LLM for grounded answers.

Run directly:  python query.py          (runs 3 test queries)
Import:        from query import ask     (use in app.py)
"""

import os
import sys
from dotenv import load_dotenv
from groq import Groq

# Load .env before anything else so GROQ_API_KEY is available
load_dotenv()

# ── Import retrieval components from Milestone 4 ──────────────────────────────
try:
    from retriever import retrieve, load_embedding_model, get_collection
except ImportError as e:
    print(f"[ERROR] Could not import from retriever.py: {e}")
    print("Make sure retriever.py is in the same directory.")
    sys.exit(1)

# ──────────────────────────────────────────────
# Initialization  (runs once at import time)
# ──────────────────────────────────────────────

print("[query.py] Loading embedding model and ChromaDB...", end=" ", flush=True)
_model      = load_embedding_model()          # all-MiniLM-L6-v2
_collection = get_collection()                # loads existing ./chroma_db, no re-embedding
print(f"done. ({_collection.count()} chunks ready)")

# Groq client
_groq_api_key = os.getenv("GROQ_API_KEY")
if not _groq_api_key:
    print("[ERROR] GROQ_API_KEY not found. Add it to your .env file.")
    sys.exit(1)
_client = Groq(api_key=_groq_api_key)

GROQ_MODEL = "llama-3.3-70b-versatile"

# ──────────────────────────────────────────────
# Prompt template  (strict grounding)
# ──────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a helpful assistant that answers questions about college professors based ONLY on the provided context.

INSTRUCTIONS:
1. Answer ONLY using information from the context above.
2. If the context doesn't contain the answer, say exactly: "I don't have enough information to answer that question."
3. Do not use any external knowledge or training data.
4. Keep your answer concise and specific.
5. Never say "likely" or "probably" — only state what is explicitly in the context.
6. After your answer, list which sources (filenames) the information came from."""

_USER_TEMPLATE = """\
CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


# ──────────────────────────────────────────────
# Core function
# ──────────────────────────────────────────────

def ask(question: str, k: int = 5) -> dict:
    """
    Full RAG pipeline: retrieve → build prompt → generate → return structured result.

    Returns:
        {
            "answer":  str,           # LLM response
            "sources": list[str],     # unique source filenames cited
            "chunks":  list[dict],    # raw retrieved chunks (rank, text, source, distance)
        }
    """
    # Step 1: Retrieve top-k chunks from ChromaDB
    chunks = retrieve(question, _model, _collection, k=k)

    # Step 2: Build context string — number each chunk and label its source
    context_parts = []
    for c in chunks:
        context_parts.append(
            f"[{c['rank']}] (Source: {c['source']})\n{c['text']}"
        )
    context = "\n\n".join(context_parts)

    # Step 3: Collect unique sources (preserve rank order)
    seen = set()
    sources = []
    for c in chunks:
        if c["source"] not in seen:
            seen.add(c["source"])
            sources.append(c["source"])

    # Step 4: Call Groq LLM with the grounded prompt
    user_message = _USER_TEMPLATE.format(context=context, question=question)

    response = _client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.0,      # deterministic — no hallucination drift
        max_tokens=512,
    )

    answer = response.choices[0].message.content.strip()

    return {
        "answer":  answer,
        "sources": sources,
        "chunks":  chunks,
    }


# ──────────────────────────────────────────────
# Test harness  (only runs with: python query.py)
# ──────────────────────────────────────────────

def _run_tests():
    test_queries = [
        {
            "question": "Does Derek Mkhaiel give good feedback on assignments?",
            "expect_answer": True,
        },
        {
            "question": "Is Thomas Kuhlman's physics class curved?",
            "expect_answer": True,
        },
        {
            "question": "What is the university's policy on UFOs?",
            "expect_answer": False,   # out-of-scope — should refuse
        },
    ]

    bar = "─" * 60
    print(f"\n{'='*60}")
    print("MILESTONE 5 — Generation Test Queries")
    print(f"{'='*60}")

    for i, item in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {item['question']}")
        print(bar)

        result = ask(item["question"])

        print(f"ANSWER:\n{result['answer']}")
        print(f"\nSOURCES:")
        for s in result["sources"]:
            print(f"  • {s}")

        # Grounding check
        refused = "don't have enough information" in result["answer"].lower()
        if item["expect_answer"] and refused:
            verdict = "⚠️  Expected an answer but got a refusal"
        elif not item["expect_answer"] and not refused:
            verdict = "⚠️  Expected a refusal but got an answer — possible hallucination"
        elif item["expect_answer"]:
            verdict = "✅ Answered in-scope query"
        else:
            verdict = "✅ Correctly refused out-of-scope query"

        print(f"\nGrounding check: {verdict}")
        print(bar)

    print("\nAll test queries complete.")
    print("Next step → python app.py  (Gradio web interface)\n")


if __name__ == "__main__":
    _run_tests()