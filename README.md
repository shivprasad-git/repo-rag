# Repo RAG

<p align="center">
  <img src="assets/logo.svg" alt="Repo RAG logo" width="720">
</p>

Repo RAG indexes a GitHub or local repository, retrieves the most relevant code/docs for a question, and builds an LLM-ready prompt with source locations.

It currently focuses on the backend RAG pipeline: Tree-sitter parsing, chunking, embeddings, hybrid retrieval, reranking, incremental indexing, and prompt construction.

## Features

- Loads local repositories or clones GitHub repositories.
- Indexes `.py`, `.md`, and `.txt` files.
- Parses Python and Markdown with Tree-sitter.
- Creates structured chunks for classes, methods, functions, Markdown sections, text files, imports, file metadata, and parse errors.
- Splits oversized searchable chunks with tokenizer-aware overlap.
- Stores full chunk content in a JSON document store.
- Stores embeddings in Chroma or a simple JSON vector store.
- Retrieves with vector similarity + BM25 keyword search.
- Reranks candidates with a MiniLM cross-encoder.
- Expands neighboring parts for split chunks.
- Enforces a prompt context token budget.
- Supports incremental re-indexing using file hashes.
- Includes index info, health checks, evaluation, and test coverage tooling.

## Architecture

```text
repository
  -> discover files
  -> parse with Tree-sitter
  -> create chunks
  -> split oversized chunks
  -> embed searchable chunks
  -> store vectors + full chunks
  -> retrieve with vector + keyword search
  -> rerank
  -> build token-budgeted prompt
```

The vector store keeps only embeddings and minimal filter metadata. The document store keeps full chunk content and metadata. Retrieval finds chunk IDs first, then hydrates full chunks before building the prompt.

## Setup

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first indexing/querying run may download the embedding model and reranker model from Hugging Face.

## Quick Start

Index a local repo:

```bash
python3 -m app.cli index --repo /path/to/local/repo
```

Query the index:

```bash
python3 -m app.cli query \
  --index-name <repo-folder-name> \
  --question "Where is authentication implemented?" \
  --show-content
```

Build an LLM-ready prompt:

```bash
python3 -m app.cli query \
  --index-name <repo-folder-name> \
  --question "How does login work?" \
  --prompt
```

Use `ask` when you want the command to index first if needed:

```bash
python3 -m app.cli ask \
  --repo /path/to/local/repo \
  --question "How does login work?"
```

## CLI Commands

```bash
# GitHub URL or local path. Index name defaults to the repo folder/name.
python3 -m app.cli index --repo https://github.com/pallets/flask

# Use the simple JSON vector store instead of Chroma.
python3 -m app.cli --store simple index --repo /path/to/local/repo

# Retrieve matches.
python3 -m app.cli --index-name flask query --question "How is routing implemented?"

# Show chunk previews, JSON, debug scores, or prompt output.
python3 -m app.cli --index-name flask query --question "Where is login?" --show-content
python3 -m app.cli --index-name flask query --question "Where is login?" --json
python3 -m app.cli --index-name flask query --question "Where is login?" --debug-scores
python3 -m app.cli --index-name flask query --question "Where is login?" --prompt

# Filter retrieval.
python3 -m app.cli --index-name flask query \
  --question "Where is login?" \
  --language python \
  --chunk-type method \
  --path app

# Inspect or validate an index.
python3 -m app.cli --index-name flask info
python3 -m app.cli --index-name flask health --repo /path/to/local/repo

# Run retrieval evaluation against a golden set.
python3 -m app.cli --index-name flask eval --golden eval/golden/sample.json
```

## Configuration

Defaults are resolved in this order:

```text
command-line flag > RAG_* environment variable / .env > built-in default
```

Useful `.env` values:

```bash
RAG_STORE=simple
RAG_INDEX_NAME=flask
RAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RAG_EMBEDDING_DIMENSIONS=384
RAG_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RAG_RERANKER_ENABLED=false
RAG_MAX_CHUNK_TOKENS=700
RAG_CHUNK_OVERLAP_TOKENS=100
RAG_CONTEXT_WINDOW_PARTS=1
RAG_MAX_PROMPT_CONTEXT_TOKENS=3000
RAG_EVAL_RESULTS_DIR=eval/results
```

Changing the embedding model requires rebuilding the vector index because each model produces a different vector space.

## Project Layout

```text
app/
  api.py              FastAPI endpoints
  cli.py              CLI entrypoint
  pipeline.py         Index/query orchestration
  config/             Settings
  ingest/             Discovery, parsing, chunking, splitting
  embeddings/         Embedding providers
  vectorstore/        Chroma/simple vector stores
  docstore/           Full chunk JSON store
  retrieval/          Hybrid search, filters, reranking
  indexing/           Manifest, info, health checks
  eval/               Golden-set retrieval evaluation
  llm/                Prompt construction
  models/             Shared models
```

## Tests

```bash
python3 -m pytest
```

With coverage:

```bash
python3 -m coverage run -m pytest
python3 -m coverage report -m
```

Coverage is reported from the local test run.

## API

The optional FastAPI app exposes `/index` and `/query`.

```bash
uvicorn app.api:app --reload
```

## Scope

This is still a backend MVP. It does not yet include direct LLM answer generation, a full UI, authentication, background workers, broad language support, or multi-user project management.

## AI Credits

<p align="center">
  <img src="assets/codex-color.svg" alt="Codex logo" width="42">
  &nbsp;&nbsp;
  <img src="assets/deepseek-color.svg" alt="DeepSeek logo" width="42">
</p>
