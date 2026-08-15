# Repo RAG

<p align="center">
  <img src="assets/logo.svg" alt="Repo RAG logo" width="720">
</p>

Repo RAG is a GitHub repository retrieval-augmented generation MVP. It indexes a codebase into structured chunks, retrieves the most relevant code and documentation for a question, and builds an LLM-ready prompt with source locations.

The project focuses on the backend RAG pipeline: parsing, chunking, embeddings, vector search, reranking, incremental indexing, and prompt construction.

## What It Does

- Clone a GitHub repository or load a local repository.
- Discover supported files: `.py`, `.md`, and `.txt`.
- Parse Python and Markdown with Tree-sitter.
- Create structured chunks for files, classes, functions, methods, Markdown sections, text files, imports, and parse errors.
- Split oversized chunks with the embedding model's tokenizer.
- Generate semantic embeddings with `sentence-transformers`.
- Store vectors in Chroma or a local JSON vector store.
- Store full chunk content and metadata separately in a JSON document store.
- Retrieve with hybrid vector + keyword search.
- Rerank candidates with a MiniLM cross-encoder.
- Expand split chunk matches with neighboring parts.
- Enforce a prompt context token budget.
- Re-index incrementally using file hashes.
- Check index health across manifest, document store, and vector store.

## Architecture

```text
Repository
  -> file discovery
  -> Tree-sitter parsing
  -> chunk creation
  -> oversized chunk splitting
  -> searchable chunk filtering
  -> embeddings
  -> vector store + document store
  -> hybrid retrieval
  -> reranking
  -> neighbor context expansion
  -> prompt token budgeting
  -> LLM-ready prompt
```

The vector store only keeps embeddings and minimal filter metadata. The document store keeps full chunk content and full metadata. Retrieval first finds chunk IDs, then hydrates the full chunks before building the prompt.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

For a local repository, the shortest path is:

```bash
python3 -m app.cli --store simple --index-name sample index \
  --repo /path/to/local/repo

python3 -m app.cli --store simple --index-name sample query \
  --question "Where is authentication implemented?" \
  --show-content
```

Use `--verbose` when you want indexing and retrieval logs:

```bash
python3 -m app.cli --verbose --store simple --index-name sample query \
  --question "How does login work?"
```

## Index A Repository

Using Chroma:

```bash
python3 -m app.cli --store chroma --index-name flask index \
  --repo https://github.com/pallets/flask
```

Using the local JSON vector store:

```bash
python3 -m app.cli --store simple --index-name flask index \
  --repo /path/to/local/repo
```

Indexing is incremental. Re-running `index` skips unchanged files, replaces changed files, and removes deleted files from the index.

## Query

Retrieve matching chunks:

```bash
python3 -m app.cli --store simple --index-name flask query \
  --question "How is routing implemented?" \
  --top-k 5
```

Show chunk content previews:

```bash
python3 -m app.cli --store simple --index-name flask query \
  --question "Where is login implemented?" \
  --show-content
```

Show retrieval score details:

```bash
python3 -m app.cli --store simple --index-name flask query \
  --question "Where is login implemented?" \
  --debug-scores
```

Print JSON results:

```bash
python3 -m app.cli --store simple --index-name flask query \
  --question "How is routing implemented?" \
  --json
```

Filter by metadata:

```bash
python3 -m app.cli --store simple --index-name flask query \
  --question "Where is login implemented?" \
  --language python \
  --chunk-type method \
  --path app
```

Print an LLM-ready prompt:

```bash
python3 -m app.cli --store simple --index-name flask query \
  --question "How is routing implemented?" \
  --prompt
```

## Ask

`ask` ensures the index exists, retrieves relevant chunks, and prints the prompt to send to an LLM.

```bash
python3 -m app.cli --store simple --index-name sample ask \
  --repo /path/to/local/repo \
  --question "How does login work?" \
  --top-k 3
```

## Index Health

Summarize index settings and counts:

```bash
python3 -m app.cli --store simple --index-name flask info
```

Check consistency between the manifest, document store, and vector store:

```bash
python3 -m app.cli --store simple --index-name flask health
```

Include `--repo` to compare manifest hashes against current files:

```bash
python3 -m app.cli --store simple --index-name flask health \
  --repo /path/to/local/repo
```

The health check reports missing document chunks, doc chunks missing from the manifest, missing vectors, orphan vectors, embedding dimension mismatches for the JSON vector store, config mismatches, and stale file hashes.

## Models

Default embedding model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Default reranker:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

Override the embedding model:

```bash
python3 -m app.cli --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --store simple \
  --index-name sample \
  query --question "How does login work?"
```

Disable reranking:

```bash
python3 -m app.cli --no-reranker --store simple --index-name sample query \
  --question "How does login work?"
```

Changing embedding models requires rebuilding the vector index because each model produces vectors in its own vector space.

## Chunking

Python chunks are syntax-aware:

- `file_metadata`
- `imports`
- `class`
- `method`
- `function`
- `parse_error`

Markdown is split by headings into `markdown_section` chunks. Plain text files become `text_file` chunks.

Python class chunks are compact: class signature, docstring, and class attributes are stored separately from method bodies. This avoids duplicating method content across class and method chunks.

Oversized functions, methods, Markdown sections, and text files are split after parsing. By default, chunks over `700` tokens are split with `100` tokens of overlap. Token counts come from the embedding model tokenizer, so the split budget reflects what the embedding model actually sees.

Split parts keep the original `chunk_type` and add:

- `is_chunk_part`
- `part_index`
- `part_count`
- `parent_chunk_id`

## Retrieval

Retrieval combines:

- vector similarity,
- BM25 keyword search,
- metadata filtering,
- MiniLM reranking,
- neighbor expansion for split chunks.

Searchable chunks are indexed into the vector store. File metadata and parse details remain available in the document store without polluting normal semantic retrieval.

When retrieval returns a split chunk part, Repo RAG includes one neighboring part on each side by default. Prompt construction then enforces a `3000` token context budget and prioritizes direct matches over neighbor context.

## Optional API

```bash
uvicorn app.api:app --reload
```

Endpoints:

- `POST /index`
- `POST /query`

## Current Scope

This is an MVP backend, not a production service. It does not yet include:

- a full UI,
- authentication,
- background workers,
- multi-user project management,
- large-scale evaluation metrics,
- broad language support beyond the current parser set.

## Credits

<p align="center">
  <img src="assets/codex-color.svg" alt="Codex logo" width="42">
  &nbsp;&nbsp;
  <img src="assets/deepseek-color.svg" alt="DeepSeek logo" width="42">
</p>

Built with help from Codex and DeepSeek.
