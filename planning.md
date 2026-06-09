# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

<!-- What domain did you choose? Why is this knowledge valuable and hard to find through official channels? -->
This domain provides student reviews of college professors, and provide information on their teaching quality, grading practices, exam difficulty, and helpfulness. This knowledge is hard to find because university course catalogs don't tell you which professors are easy graders, who gives good feedback, or who makes difficult test. That information is available in student-generated sources like Rate My Professors(RMP) or Reddit discussion threads.


---

## Documents

<!-- List your specific sources: URLs, subreddit names, forum threads, or file descriptions.
     Aim for at least 10 sources that together cover different subtopics or perspectives within your domain. -->

| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1 |Rate My Professors|Stefano Lonardi|https://www.ratemyprofessors.com/professor/169726|
| 2 |Rate My Professors|Derek Mkhaiel|https://www.ratemyprofessors.com/professor/2863312|
| 3 |Rate My Professors|Thomas Kuhlman|https://www.ratemyprofessors.com/professor/2237902|
| 4 |Rate My Professors|Marko Spasojevic|https://www.ratemyprofessors.com/professor/2202003|
| 5 |Rate My Professors|Matthew Lang|https://www.ratemyprofessors.com/professor/2463697|
| 6 |Rate My Professors|Mariam Salloum|https://www.ratemyprofessors.com/professor/2893094|
| 7 |Rate My Professors|Patrick Miller|https://www.ratemyprofessors.com/professor/2781584|
| 8 |Rate My Professors|Amey Bhangale|https://www.ratemyprofessors.com/professor/238914|
| 9 |Rate My Professors|Eamonn Keogh|https://www.ratemyprofessors.com/professor/238914|
| 10 |Rate My Professors|Allan Knight|https://www.ratemyprofessors.com/professor/2955403|

---

## Chunking Strategy

<!-- How will you split documents into chunks?
     State your chunk size (in tokens or characters), overlap size, and explain why those
     numbers fit the structure of your documents.
     A review-heavy corpus warrants different chunking than a long FAQ. -->

**Chunk size:** 300 characters

**Overlap:** 50 characters

**Reasoning:**
Most of the reviews in RMP are around that 300 character range so when chunking I can capture each individual review. Additionally, the previous class activity was also 300 character chunkings which worked well for semantic search. This amount should help match the queries with proper answers.
---

## Retrieval Approach

<!-- Which embedding model are you using (e.g., all-MiniLM-L6-v2 via sentence-transformers)?
     How many chunks will you retrieve per query (top-k)?
     If you were deploying this for real users and cost wasn't a constraint, what tradeoffs
     would you weigh in choosing a different embedding model — context length, multilingual
     support, accuracy on domain-specific text, latency? -->

**Embedding model:** all-MiniLM-L6-v2 via sentence-transformers (384-dimensional vectors, runs locally)

**Top-k:** 5 chunks per query

**Production tradeoff reflection:**
Some tradeoffs I would consider:

| Model | Pros | Cons |
all-MiniLM-L6-v2 (current)|Fast, local, free|Lower accuracy, English-only
all-mpnet-base-v2|Higher accuracy (better at sentiment)|768-dim vectors (slower, more memory)
text-embedding-3-small (OpenAI)|Best accuracy, handles nuance like sarcasm|Cost per query, API latency, privacy concerns
multilingual-e5-large|Supports non-English reviews|Very large (1.2GB), slow inference

---

## Evaluation Plan

<!-- List your 5 test questions with their expected correct answers.
     Questions should be specific enough that you can judge whether the system's response
     is right or wrong. "What are good dining halls?" is too vague.
     "What do students say about wait times at [dining hall name] during lunch?" is testable. -->

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 |What do students say about Stefano Lonardi's exam grading?|Exams have too much weight, doesn't curve, tough grader|
| 2 |Does Derek Mkhaiel give good feedback on assignments?|Yes, gives good feedback,amazing professor|
| 3 |Is Thomas Kuhlman's physics class curved?|Yes, curve is great, exam average curved to B-/B range|
| 4 |Does Marko Spasojevic offer extra credit in BIOL003?|Yes, many extra credit opportunities|
| 5 |What do students say about Matthew Lang's teaching style?|amazing lectures, caring, fast grader, remembers students by name|

---

## Anticipated Challenges

<!-- What could go wrong? Name at least two specific risks with reasoning.
     Consider: noisy or inconsistent documents, missing source attribution, off-topic
     retrieval, chunks that split key information across boundaries. -->

1. Opposite Opinions. People can say give contradicting reviews based on their experience which can have human biases and subjective viewpoints. Someone who got a good grade is more inclined to give a professor a good rating, while someone who didn't perform as well in class might give a lower rating to a professor.

2. Vague reviews that don't provide any valuable insight to answer the questions stated above.

---

## Architecture

<!-- Draw a diagram of your pipeline showing the five stages:
     Document Ingestion → Chunking → Embedding + Vector Store → Retrieval → Generation
     Label each stage with the tool or library you're using.
     You can use ASCII art, a Mermaid diagram, or embed a sketch as an image.
     You'll use this diagram as context when prompting AI tools to implement each stage. -->

---

## AI Tool Plan

<!-- For each part of the pipeline below, describe:
     - Which AI tool you plan to use (Claude, Copilot, ChatGPT, etc.)
     - What you'll give it as input (which sections of this planning.md, which requirements)
     - What you expect it to produce
     - How you'll verify the output matches your spec

     "I'll use AI to help me code" is not a plan.
     "I'll give Claude my Chunking Strategy section and ask it to implement chunk_text()
     with my specified chunk size and overlap" is a plan. -->

**Milestone 3 — Ingestion and chunking:**
I will give Claude my planning.md Chunking Strategy section (300 chars, 50 overlap, sentence-boundary awareness) and ask it to implement:
- load_documents(data_dir) — reads all .txt files
- clean_text(raw_text) — removes "Helpful", "Thumbs up/down", timestamps, cookie notice
- chunk_text(cleaned_text, chunk_size=300, overlap=50) — splits with sentence-boundary preference

**Milestone 4 — Embedding and retrieval:**
I will give ChatGPT my Retrieval Approach section (all-MiniLM-L6-v2, top-k=5, ChromaDB) and my pipeline diagram, asking it to implement:
- embed_chunks(chunks) — uses SentenceTransformer to generate 384-dim vectors
- store_in_chromadb(chunks, embeddings, metadata) — stores with source filename and chunk_id
- retrieve(query, k=5) — semantic search returning top chunks with distance scores

**Milestone 5 — Generation and interface:**
I will give Claude my grounded prompt requirements and ask it to implement:
- generate_answer(query, retrieved_chunks) — uses Groq's llama-3.3-70b-versatile with strict "only from context" prompt
- format_response(answer, sources) — ensures source attribution is visible
- gradio_interface() — simple web UI with input box, answer display, source list
