"""
pipeline.py — Document Ingestion & Chunking for Professor Reviews RAG
Spec: planning.md (Chunking Strategy section)
  - Chunk size: 300 characters
  - Overlap:    50 characters
  - Sentence-boundary awareness: prefer splitting at sentence endings
Run: python pipeline.py
"""

import os
import re
import json
import sys
from pathlib import Path


# ──────────────────────────────────────────────
# Stage 1 — Load documents
# ──────────────────────────────────────────────

def load_documents(data_dir: str) -> list[dict]:
    """
    Read all .txt files in data_dir.
    Returns a list of dicts: {filename, raw_text}
    Raises FileNotFoundError if the folder doesn't exist.
    Skips empty files with a warning.
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data folder not found: '{data_dir}'\n"
            "Create the folder and add your .txt review files before running."
        )

    txt_files = sorted(data_path.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(
            f"No .txt files found in '{data_dir}'. "
            "Add at least one professor review file and try again."
        )

    documents = []
    for path in txt_files:
        raw = path.read_text(encoding="utf-8", errors="replace").strip()
        if not raw:
            print(f"  [WARN] Skipping empty file: {path.name}")
            continue
        documents.append({"filename": path.name, "raw_text": raw})
        print(f"  [OK]   Loaded {path.name} ({len(raw):,} chars)")

    if not documents:
        raise ValueError(
            f"All .txt files in '{data_dir}' were empty. Add review content and retry."
        )

    return documents


# ──────────────────────────────────────────────
# Stage 2 — Clean text
# ──────────────────────────────────────────────

# Patterns that appear in copy-pasted Rate My Professors content
_NOISE_PATTERNS = [
    r"\d+ people found this helpful",          # "12 people found this helpful"
    r"Helpful\??",                              # "Helpful?" / "Helpful"
    r"(👍|👎)\s*\d*",                          # thumbs up/down emoji + count
    r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?\s+\d{1,2},?\s+\d{4}\b",  # dates
    r"\d{1,2}/\d{1,2}/\d{2,4}",               # MM/DD/YYYY
    r"We use cookies.*?learn more\.",           # cookie notice (single line)
    r"Rate My Professors is a website.*?\.",    # boilerplate sentences
    r"For the best experience.*?browser\.",
    r"^\s*\d+\s*$",                             # bare page numbers on their own line
]

_NOISE_RE = re.compile(
    "|".join(_NOISE_PATTERNS),
    flags=re.IGNORECASE | re.MULTILINE,
)


def clean_text(raw_text: str) -> str:
    """
    Remove RMP boilerplate, timestamps, thumbs, ratings labels.
    """
    text = raw_text
    
    # Remove lines with these exact patterns
    lines = text.split('\n')
    cleaned_lines = []
    
    skip_patterns = [
        r'^Quality\s*$',           # standalone "Quality"
        r'^Difficulty\s*$',        # standalone "Difficulty"  
        r'^Thumbs up\s*$',
        r'^Thumbs down\s*$',
        r'^Helpful\s*$',
        r'^\d+ people found this helpful$',
        r'^Reviewed:.*$',          # "Reviewed: Mar 5th, 2025"
        r'^\[\d+\]$',              # "[12]" vote counts
        r'^\d+$',                  # bare numbers
        r'^For Credit:.*$',        # These are metadata, not review content
        r'^Attendance:.*$',
        r'^Grade:.*$',
        r'^Textbook:.*$',
        r'^Would Take Again:.*$',
    ]
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Skip if matches any noise pattern
        skip = False
        for pattern in skip_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                skip = True
                break
        
        if skip:
            continue
        
        # Remove inline emoji thumbs
        line = re.sub(r'[👍👎]\s*\d*', '', line)
        line = line.strip()
        
        if line:
            cleaned_lines.append(line)
    
    # Join and normalize
    result = '\n'.join(cleaned_lines)
    result = re.sub(r'\n{3,}', '\n\n', result)
    
    return result


# ──────────────────────────────────────────────
# Stage 3 — Chunk text
# ──────────────────────────────────────────────

# Sentence-ending boundary: period / ! / ? followed by whitespace or end
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


def chunk_text(
    cleaned_text: str,
    chunk_size: int = 300,
    overlap: int = 50,
) -> list[str]:
    """
    Split cleaned_text into chunks of ~chunk_size characters with overlap.

    Strategy (from planning.md):
      1. Split on sentence boundaries ([.!?]) so chunks don't cut mid-sentence.
      2. Any sentence that alone exceeds chunk_size is hard-split by character
         BEFORE the packing loop — so the main loop never stalls on an
         oversized sentence.
      3. Greedily pack sentences into a chunk; on flush, carry ~overlap chars
         of context into the next chunk.

    Termination guarantee: every sentence is consumed exactly once in step 1,
    and the for-loop in step 3 iterates that fixed list — no while+index logic.
    """
    if not cleaned_text:
        return []

    # Step 1: collect sentences, immediately hard-splitting oversized ones.
    sentences: list[str] = []
    for paragraph in cleaned_text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for sent in _SENTENCE_END_RE.split(paragraph):
            sent = sent.strip()
            if not sent:
                continue
            while len(sent) > chunk_size:
                sentences.append(sent[:chunk_size])
                sent = sent[chunk_size - overlap:]
            if sent:
                sentences.append(sent)

    if not sentences:
        return []

    # Step 2: greedily pack into chunks with overlap carry.
    chunks: list[str] = []
    current: list[str] = []
    current_len: int = 0

    for sent in sentences:
        space = 1 if current else 0
        if current_len + space + len(sent) <= chunk_size:
            current.append(sent)
            current_len += space + len(sent)
        else:
            if current:
                chunks.append(" ".join(current))
            # Build overlap window from the tail of the flushed chunk.
            overlap_buf: list[str] = []
            overlap_len = 0
            for s in reversed(current):
                needed = len(s) + (1 if overlap_buf else 0)
                if overlap_len + needed > overlap:
                    break
                overlap_buf.insert(0, s)
                overlap_len += needed
            current = overlap_buf + [sent]
            current_len = overlap_len + (1 if overlap_buf else 0) + len(sent)

    if current:
        tail = " ".join(current)
        if not chunks or tail != chunks[-1]:
            chunks.append(tail)

    return [c for c in chunks if c]


# ──────────────────────────────────────────────
# Stage 4 — Build chunk records (metadata)
# ──────────────────────────────────────────────

def build_chunk_records(documents: list[dict]) -> list[dict]:
    """
    For each document, clean → chunk → attach metadata.
    Returns a flat list of chunk records ready for embedding.
    Each record: {chunk_id, source, chunk_index, text}
    """
    all_chunks = []
    global_id = 0

    for doc in documents:
        cleaned = clean_text(doc["raw_text"])
        chunks = chunk_text(cleaned, chunk_size=300, overlap=50)

        print(f"  {doc['filename']}: {len(chunks)} chunks")

        for idx, text in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"chunk_{global_id:04d}",
                "source": doc["filename"],
                "chunk_index": idx,
                "text": text,
            })
            global_id += 1

    return all_chunks

# ──────────────────────────────────────────────
# Sample output
# ──────────────────────────────────────────────

def inspect_sample_chunks(chunk_records: list[dict], num_samples: int = 5):
    """
    Print random chunks for quality inspection.
    """
    import random
    
    print("\n" + "=" * 55)
    print("SAMPLE CHUNK INSPECTION (Milestone 3 Checkpoint)")
    print("=" * 55)
    
    # Take random samples
    samples = random.sample(chunk_records, min(num_samples, len(chunk_records)))
    
    for i, chunk in enumerate(samples):
        print(f"\n--- SAMPLE CHUNK {i+1} ---")
        print(f"Source: {chunk['source']}")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Length: {len(chunk['text'])} chars")
        print(f"Content:\n{chunk['text'][:500]}")
        print("-" * 40)
        
        # Quality checks
        issues = []
        if len(chunk['text']) < 30:
            issues.append("TOO SHORT (<30 chars)")
        if '<' in chunk['text'] and '>' in chunk['text']:
            issues.append("HAS HTML TAGS")
        if 'Helpful' in chunk['text'] or 'Thumbs' in chunk['text']:
            issues.append("HAS NOISE (Helpful/Thumbs)")
        if '&amp;' in chunk['text'] or '&nbsp;' in chunk['text']:
            issues.append("HAS HTML ENTITIES")
        
        if issues:
            print(f"Issues: {', '.join(issues)}")
        else:
            print("PASS: Readable, substantive, self-contained")


def print_one_cleaned_document(documents: list[dict]):
    """
    Print one cleaned document to verify cleaning worked.
    """
    print("\n" + "=" * 55)
    print("VERIFY CLEANING: First 1000 chars of first document")
    print("=" * 55)
    doc = documents[0]
    cleaned = clean_text(doc["raw_text"])
    print(f"Source: {doc['filename']}")
    print(f"Cleaned length: {len(cleaned)} chars")
    print(f"\nContent preview:\n{cleaned[:1000]}")
    print("-" * 55)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    DATA_DIR = "data"
    OUTPUT_FILE = "chunks.json"

    print("=" * 55)
    print("Professor Reviews RAG — Ingestion & Chunking Pipeline")
    print("=" * 55)

    print(f"\n[1/3] Loading documents from '{DATA_DIR}/'...")
    try:
        documents = load_documents(DATA_DIR)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    print(f"       {len(documents)} document(s) loaded.")

    print("\n[2/3] Cleaning and chunking...")
    chunk_records = build_chunk_records(documents)
    print(f"       {len(chunk_records)} total chunks produced.")

    print(f"\n[3/3] Saving chunks to '{OUTPUT_FILE}'...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunk_records, f, indent=2, ensure_ascii=False)
    print(f"       Saved {len(chunk_records)} chunk records.")

    # ─── MILESTONE 3 CHECKPOINT: Inspect chunks ───
    print_one_cleaned_document(documents)
    inspect_sample_chunks(chunk_records, num_samples=5)

    # Summary
    lengths = [len(r["text"]) for r in chunk_records]
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    print(f"\n── Summary ──────────────────────────────────────")
    print(f"  Documents:     {len(documents)}")
    print(f"  Total chunks:  {len(chunk_records)}")
    print(f"  Avg chunk len: {avg_len:.0f} chars")
    print(f"  Min / Max:     {min(lengths)} / {max(lengths)} chars")
    print(f"  Output:        {OUTPUT_FILE}")
    print("─" * 50)
    print("\n Milestone 3 Complete!")



if __name__ == "__main__":
    main()