# The Unofficial Guide — Project 1

> **How to use this template:**
> Complete each section *after* you've built and tested the corresponding part of your system.
> Do not write placeholder text — if a section isn't done yet, leave it blank and come back.
> Every section below is required for submission. One-liners will not receive full credit.

---

## Domain

<!-- What topic or category of knowledge does your system cover?
     Why is this knowledge valuable, and why is it hard to find through official channels?
     Example: "Student reviews of CS professors at [university] — useful because official
     course descriptions don't reflect teaching style, exam difficulty, or workload." -->
     This domain provides student reviews of college professors, and provide information on their teaching quality, grading practices, exam difficulty, and helpfulness. This knowledge is hard to find because university course catalogs don't tell you which professors are easy graders, who gives good feedback, or who makes difficult test. That information is available in student-generated sources like Rate My Professors(RMP) or Reddit discussion threads.

---

## Document Sources

<!-- List every source you collected documents from.
     Be specific: include URLs, subreddit names, forum thread titles, or file names.
     Aim for variety — sources that together cover different subtopics or perspectives. -->

| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1 |Rate My Professors|Stefano Lonardi|https://www.ratemyprofessors.com/professor/169726|
| 2 |Rate My Professors|Derek Mkhaiel|https://www.ratemyprofessors.com/professor/2863312|
| 3 |Rate My Professors|Thomas Kuhlman|https://www.ratemyprofessors.com/professor/2471296|
| 4 |Rate My Professors|Marko Spasojevic|https://www.ratemyprofessors.com/professor/2237902|
| 5 |Rate My Professors|Matthew Lang|https://www.ratemyprofessors.com/professor/2202003|
| 6 |Rate My Professors|Mariam Salloum|https://www.ratemyprofessors.com/professor/2463697|
| 7 |Rate My Professors|Patrick Miller|https://www.ratemyprofessors.com/professor/2893094|
| 8 |Rate My Professors|Amey Bhangale|https://www.ratemyprofessors.com/professor/2781584|
| 9 |Rate My Professors|Eamonn Keogh|https://www.ratemyprofessors.com/professor/238914|
| 10 |Rate My Professors|Allan Knight|https://www.ratemyprofessors.com/professor/2955403|

---

## Chunking Strategy

<!-- Describe your chunking approach with enough specificity that someone else could reproduce it.
     Include:
     - Chunk size (characters or tokens) and why that size fits your documents
     - Overlap size and why (or why not) you used overlap
     - Any preprocessing you did before chunking (e.g., stripping HTML, removing headers)
     - What your final chunk count was across all documents -->

**Chunk size:** 300 characters

**Overlap:** 50 characters

**Why these choices fit your documents:**
Most of the reviews in RMP are around that 300 character range so when chunking I can capture each individual review. Additionally, the previous class activity was also 300 character chunkings which worked well for semantic search. This amount should help match the queries with proper answers.

**Final chunk count:**
337 total chunks from 10 documents

---

## Embedding Model

<!-- Name the embedding model you used and explain your choice.
     Then answer: if you were deploying this system for real users and cost wasn't a constraint,
     what tradeoffs would you weigh in choosing a different model?
     Consider: context length limits, multilingual support, accuracy on domain-specific text,
     latency, and local vs. API-hosted. -->

**Model used:** `all-MiniLM-L6-v2` from sentence-transformers (384-dimensional vectors, runs locally)

**Production tradeoff reflection:** 
Some tradeoffs I considered:

| Model | Pros | Cons | Verdict for production |
|-------|------|------|------------------------|
| `all-MiniLM-L6-v2` (current) | Fast inference (~0.01s/query), runs locally free, small memory footprint (90MB) | Lower accuracy than larger models, English-only, no nuance for sarcasm | Good for MVP, not for production |
| `all-mpnet-base-v2` | Higher accuracy (better at sentiment and nuance), 768-dim vectors capture more meaning | 2x slower, 4x larger memory (~420MB) | Better for production if budget allows |
| `text-embedding-3-small` (OpenAI) | Best accuracy, handles implicit sentiment like "shows movies instead of teaching" = bad, 512-dim efficient | ~$0.02 per 1M tokens, API latency, privacy concerns for student data | Only if accuracy is critical and data is anonymized |
| `multilingual-e5-large` | Supports non-English reviews (important for diverse campuses), 1024-dim very high accuracy | 1.2GB model, slow inference (1-2s/query), requires GPU for real-time | Only for multilingual use cases |

---

## Grounded Generation

<!-- Explain how your system enforces grounding — how does it prevent the LLM from answering
     beyond the retrieved documents?
     Describe both your system prompt (what instruction you gave the model) and any structural
     choices (e.g., how you formatted the context, whether you filtered low-relevance chunks).
     Do not just say "I told it to use the documents" — show the actual instruction or explain
     the mechanism. -->

**System prompt grounding instruction:**
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

**How source attribution is surfaced in the response:**
The response shows sources in two complementary ways:

**1. LLM-generated attribution (in the answer text):** The prompt encourages the model to mention which professor's reviews it used. The instruction doesn't explicitly require this, but the model naturally includes it when answering:

`"Yes, Derek Mkhaiel gives good feedback on assignments. Source: Derek_Mkhaiel.txt"`

**2. Programmatically appended attribution (guaranteed by code):** After the LLM generates the answer, the system extracts unique source filenames from the retrieved chunks and appends them to the response. This is NOT left to the LLM's memory. The code in `query.py` does:

`sources = list(set([chunk['source'] for chunk in retrieved_chunks]))`

---

## Evaluation Report

<!-- Run your 5 test questions from planning.md through your system and record the results.
     Be honest — a partially accurate or inaccurate result that you explain well is more
     valuable than a suspiciously perfect result. -->

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | What do students say about Stefano Lonardi's exam grading? | Exams have too much weight (50% final, 30% midterm), doesn't curve, tough grader | "I don't have enough information to answer that question" | Off-target (returned Matthew_Lang and Thomas_Kuhlman) | Inaccurate |
| 2 | Does Derek Mkhaiel give good feedback on assignments? | Yes, gives good feedback, amazing professor | "Yes, Derek Mkhaiel gives good feedback on assignments" | Relevant (Derek_Mkhaiel.txt as top result) | Accurate |
| 3 | Is Thomas Kuhlman's physics class curved? | Yes, curve is great, exam average curved to B-/B range | "There is no information about Thomas Kuhlman's physics class being curved" | Partially relevant (returned correct source but missed explicit curve mention) | Partially accurate |
| 4 | Does Marko Spasojevic offer extra credit in BIOL003? | Yes, many extra credit opportunities | "Yes, Marko Spasojevic offers extra credit in BIOL003" | Relevant (Marko_Spasojevic.txt in results) | Accurate |
| 5 | What do students say about Matthew Lang's teaching style? | Amazing lectures, caring, fast grader, remembers students by name | "Clear, passionate, entertaining, fun, good at explaining concepts, exams fair" | Relevant (Matthew_Lang.txt as top result) | Accurate |

**Retrieval quality:** Relevant / Partially relevant / Off-target  
**Response accuracy:** Accurate / Partially accurate / Inaccurate

---

## Failure Case Analysis

<!-- Identify at least one question where retrieval or generation did not work as expected.
     Write a specific explanation of *why* it failed, tied to a part of the pipeline.

     "The answer was wrong" is not an explanation.

     "The relevant information was split across a chunk boundary, so retrieval returned
     only half the context — the model didn't have enough to answer correctly" is an explanation.

     "The embedding model treated the professor's nickname as out-of-vocabulary and returned
     results from an unrelated review" is an explanation. -->

**Question that failed:** "What do students say about Stefano Lonardi's exam grading?"

**What the system returned:** "I don't have enough information to answer that question" with no sources, while the retrieved chunks came from Matthew_Lang.txt and Thomas_Kuhlman.txt (completely different professors).

**Root cause (tied to a specific pipeline stage):** This is a **retrieval failure** at the embedding/similarity search stage. The query contains the phrase "exam grading," which appears frequently in Matthew Lang's reviews (e.g., "clear grading criteria," "easy grader," "generous when it comes to grading"). The all-MiniLM-L6-v2 embedding model prioritized semantic similarity of "grading" over the named entity "Stefano Lonardi." Additionally, Matthew Lang has 88 ratings (164 chunks) while Stefano Lonardi has only 27 ratings (29 chunks), making Lang's chunks statistically more likely to be retrieved when query terms are ambiguous.

**What you would change to fix it:**
1. **Metadata filtering:** When a query contains a professor name (e.g., "Stefano Lonardi"), first filter the ChromaDB search to only chunks from that professor's source file before semantic search.
2. **Hybrid search:** Combine semantic similarity with keyword (BM25) matching to boost exact name matches.
3. **Query rewriting:** Preprocess queries to extract professor names and restructure as "Stefano Lonardi grading policy" instead of "What do students say about Stefano Lonardi's exam grading?"

### Additional Note on Question 3 (Thomas Kuhlman)

While retrieval succeeded for this query (returned Thomas_Kuhlman.txt), the system returned "no information about being curved" even though the context contained "exam average was curved to B-/B range." This is a **generation failure** caused by an overly strict prompt that required explicit phrasing like "this class is curved" rather than allowing inference from exam average descriptions. The prompt was updated to allow reasonable inferences, but the conservative behavior persisted for this specific query due to the way the curve information was presented in the retrieved chunk.

---

## Spec Reflection

<!-- Reflect on how planning.md shaped your implementation.
     Answer both questions with at least 2–3 sentences each. -->

**One way the spec helped you during implementation:**
My chunking strategy specification (300 characters, 50 overlap, sentence-boundary awareness) gave me a concrete target to communicate to Claude. Instead of vague instructions like "chunk the text reasonably," I could say "implement chunk_text() with chunk_size=300, overlap=50, and prefer cutting at sentence boundaries." This clarity meant the generated code worked on the first run without major structural changes. The spec also forced me to think about why 300 characters made sense for Rate My Professors reviews (typical review length is 150-400 characters), which helped me debug later when I saw average chunk length was 260 characters.

**One way your implementation diverged from the spec, and why:**
My original spec planned for simple character-based chunking with fixed overlap. However, when Claude generated the implementation, it used a more sophisticated sentence-boundary detection algorithm that splits on periods and line breaks while still respecting the 300-character limit. I kept this divergence because it produced higher-quality chunks that preserved complete thoughts. A hard character cut could split "Exams are hard... but the curve is generous" into two chunks, losing the connection between the statements. The sentence-aware approach prevents this. I updated my planning.md to reflect this improvement.

---

## AI Usage

<!-- Describe at least 2 specific instances where you used an AI tool during this project.
     For each: what did you give the AI as input, what did it produce, and what did you
     change, override, or direct differently?

     "I used Claude to help me code" is not sufficient.
     "I gave Claude my Chunking Strategy section from planning.md and asked it to implement
     chunk_text(). It returned a function using a fixed character split. I overrode the
     chunk size from 500 to 200 because my documents are short reviews, not long guides." -->

**Instance 1**

- *What I gave the AI:* I gave Claude my planning.md Chunking Strategy section (300 characters, 50 overlap, sentence-boundary awareness) and asked it to implement pipeline.py that loads 10 Rate My Professors .txt files, cleans them, and produces chunks with metadata.
- *What it produced:* Claude generated a complete pipeline.py with load_documents(), clean_text(), and chunk_text() functions. The chunking implementation included sentence-boundary detection using regex patterns and overlap handling.
- *What I changed or overrode:* The initial cleaning function didn't catch all Rate My Professors noise. It left "Helpful" lines, "Thumbs up/down," and "Reviewed: [date]" timestamps in the chunks. I added additional regex patterns to filter these out and also removed standalone "Quality" and "Difficulty" labels that appeared on their own lines. After these changes, the sample chunks were clean and readable.


**Instance 2**

- *What I gave the AI:* I gave Claude my Retrieval Approach section (all-MiniLM-L6-v2 embedding model, top-k=5 chunks, ChromaDB for vector store) and asked it to implement retriever.py that loads chunks.json, embeds all chunks, stores them in a persistent ChromaDB collection, and provides a retrieve() function.

- *What it produced:* Claude generated retriever.py with persistent ChromaDB storage, batch embedding (64 chunks at a time), and a retrieve() function that returns top-k chunks with distance scores. The code included proper error handling and automatic skipping of embedding if the collection already existed.
- *What I changed or overrode:* The initial retrieval test queries didn't match my evaluation plan. Claude generated generic queries like "What do students say about exams?" I replaced these with my specific evaluation questions: Stefano Lonardi's exam grading, Derek Mkhaiel's feedback quality, and Thomas Kuhlman's curve policy. I also added the assess_relevance() function to automatically judge whether retrieved chunks were relevant based on source filename and distance threshold.
