# Threadneedle

UK macro policy RAG over Bank of England, ONS and HM Treasury sources.

Documents are parsed with **Docling**, chunked with header-aware splitting, and stored in **Chroma**. A LangGraph tool-calling agent answers questions over SSE, fetching live ONS figures instead of quoting stale PDF numbers.

## Layout

```
src/threadneedle/
  indexing/             ingest → parse → chunk → Chroma
  agent/                retrieval + ONS tools
  api/                  FastAPI + SSE chatbot
sources.yaml            document manifest
static/index.html       chat UI
data/                   downloaded files, parsed cache, Chroma (gitignored)
```

## Setup

```bash
uv sync
cp .env.example .env      # add OPENAI_API_KEY for the chatbot
```

No API key is required to **index** with the default local BGE embeddings.

## Index

```bash
uv run threadneedle-index --dry-run    # parse + chunk, inspect data/parsed/parsed.jsonl
uv run threadneedle-index              # embed into Chroma
uv run threadneedle-index --stats
uv run threadneedle-index --cleanup scoped_full   # after changing parse/chunk settings
```

First run downloads the embedding model (~130MB) and the source PDFs/HTML. Docling is slower than pymupdf4llm but keeps tables and HTML structure; use `PDF_PARSER=pymupdf4llm` for a faster PDF-only pass. OCR is **off** by default (`DOCLING_DO_OCR=0`) because these PDFs already have a text layer — RapidOCR empty-result warnings are chart images, not missing prose.

Corpus (see `sources.yaml`):

- Bank of England Monetary Policy Reports (2025 quarterly)
- Matching MPC minutes
- ONS CPI, labour market and GDP bulletins (narrative)
- Budget 2025 and Spring Statement 2025 (HTML)

Add more entries to `sources.yaml` and re-run the indexer. Incremental upserts go through LangChain's `SQLRecordManager`, so unchanged chunks are skipped.

## Chatbot

```bash
uv run uvicorn threadneedle.api.app:app --reload --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The agent decides when to:

| Tool | Use |
|---|---|
| `search_policy_docs` | Semantic search with optional `edition` / `publisher` / `doc_type` filters |
| `compare_editions` | Same query per MPR/minutes edition |
| `ons_observation` | Live CPIH from the ONS open API |
| `list_corpus` | What is actually indexed |

`POST /chat/stream` emits SSE events: `token`, `tool`, `citation`, `done`, `error`. `POST /chat` returns the final reply. `GET /index/stats` inspects the Chroma collection.

## Metadata the agent filters on

Every chunk carries `source`, `publisher`, `doc_type`, `title`, `publication_date`, `edition`, `page`, `section_path`, `chunk_id`. `edition` (`2025-11`) is the field to use for temporal comparison — blending all four MPRs in one search is how an agent gives confidently outdated answers.

## Design notes

**Markdown, not plain text.** Docling (and pymupdf4llm as fallback) preserves heading structure, which the chunker splits on.

**Two-pass splitting.** Headers first, then size-bounding. Each chunk prepends a `section_path` breadcrumb so short passages still embed with context.

**Noise filter.** Chunks under 200 characters or more than 40% digits are dropped (chart axes and page furniture).

**Live ONS numbers.** Bulletin HTML is indexed for narrative. CPIH prints come from `https://api.beta.ons.gov.uk/v1` at query time.
