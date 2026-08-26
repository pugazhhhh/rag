# AskAboutMe

A retrieval-augmented chatbot that answers questions about my professional background, grounded in my resume instead of guessing.

**Built by:** Pugazhvendhan M
**Stack:** Python · ChromaDB · Gemini · Streamlit
**Type:** RAG (retrieval-augmented generation)

## 1. What it is

A chat assistant that sits on top of my resume. Instead of a static PDF someone has to read through, they can ask "what's your work experience?" or "what tracking algorithms have you used?" and get a direct, natural answer — pulled only from what's actually in the resume, not invented.

## 2. Why I built it

To go beyond listing "RAG" and "vector databases" as skills on a resume, and actually demonstrate the architecture — end to end, including the failure modes that only show up once you build one for real.

## 3. How it works

1. **Extraction** — Raw text is pulled from the resume PDF with `pypdf`.
2. **Chunking** — Text is split into ~500-character, paragraph-aware chunks with word-boundary overlap, so context isn't cut off mid-thought.
3. **Embedding & storage** — Each chunk is embedded locally (ChromaDB's built-in MiniLM model — no external API needed) and stored in a persistent `ChromaDB` vector store.
4. **Retrieval** — A question is embedded the same way, and ChromaDB runs a similarity search to pull the most relevant chunks.
5. **Generation** — Those chunks go to `Gemini` as context, which answers naturally and conversationally — grounded only in what was retrieved, with automatic fallback across model tiers if one hits a rate limit.
6. **Interface** — A `Streamlit` chat UI on top, so it reads as a conversation, not a script output.

## 4. A real bug worth mentioning

**Incident — retrieval accuracy**

Asking "what college did he study at?" returned "not mentioned" — even though it's in the resume. The cause: my chunk-overlap logic sliced text mid-word, turning `Kingston Engineering College` into a corrupted fragment (`llege 2021 – 2025`) that scored *closer* to the query than the real, complete chunk — pushing the correct answer out of the top-k results entirely.

Fixed by snapping chunk overlaps to word boundaries instead of raw character counts. It's a small bug with an outsized effect: in RAG, retrieval quality is only as good as chunk quality — a lesson that's hard to internalize without hitting it yourself.

## 5. Tech stack

`Python` · `ChromaDB` · `Gemini API` · `Streamlit` · `pypdf` · `python-dotenv`

## 6. What it demonstrates

| Skill | Description |
|---|---|
| **Architecture** | Designing a full RAG pipeline from scratch, not assembling a pre-built template. |
| **Debugging** | Diagnosing a retrieval-quality bug down to its root cause in the chunking layer. |
| **Reliability** | Handling real API failure modes — rate limits, server errors — with graceful fallback instead of crashes. |
| **Prompt design** | Tuning system prompts for concise, natural, on-topic answers rather than generic LLM verbosity. |

---
*AskAboutMe — a resume, made conversational*
