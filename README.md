# Repo RAG

<p align="center">
  <img src="assets/logo.svg" alt="Repo RAG logo" width="720">
</p>

Repo RAG is a Phase 1 retrieval-augmented generation MVP for GitHub repositories.

Current capabilities:

- Clone a GitHub repository or load a local repo.
- Discover `.py`, `.md`, and `.txt` files.
- Parse Python and Markdown with Tree-sitter.
- Split Markdown by headings.
- Create chunk objects with file, symbol, qualified symbol, signature, docstring, decorators, calls, test flags, parse errors, heading hierarchy, line, language, repo, and commit metadata.
- Keep Python class chunks compact by storing class headers, docstrings, and class attributes separately from method chunks.
- Normalize file-level metadata into `file_metadata` chunks so imports and parse error details are stored once per file instead of repeated on every chunk.
- Generate embeddings through a pluggable interface.
- Store embeddings with minimal filter metadata in Chroma or a local JSON vector store.
- Store full chunk content and full metadata separately in a JSON chunk document store.
- Retrieve top-k chunks for a user question with hybrid vector + keyword search.
- Rerank retrieved candidates with a MiniLM cross-encoder.
- Build an LLM-ready prompt from retrieved repository context.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Index A Repository

Using Chroma:

```bash
python3 -m app.cli --store chroma index --repo https://github.com/pallets/flask
```

Using the offline JSON store:

```bash
python3 -m app.cli --store simple --index-name flask index --repo /path/to/local/repo
```

## Query Indexed Chunks

```bash
python3 -m app.cli --store simple --index-name flask query \
  --question "How is routing implemented?" \
  --top-k 5
```

Narrow retrieval with metadata filters:

```bash
python3 -m app.cli --store simple --index-name flask query \
  --question "Where is login implemented?" \
  --language python \
  --chunk-type method \
  --path app
```

Print an LLM-ready prompt instead of raw matches:

```bash
python3 -m app.cli --store simple --index-name flask query \
  --question "How is routing implemented?" \
  --prompt
```

## Embeddings

The default embedding provider is `sentence-transformers` with:

```text
sentence-transformers/all-MiniLM-L6-v2
```

This gives real semantic embeddings, so related phrases like `login`, `sign in`, and `authenticate` can land closer together.

You can also override the embedding model:

```bash
python3 -m app.cli --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --store simple \
  --index-name sample \
  query --question "How does login work?"
```

## Reranking

Retrieval uses hybrid vector + keyword search to gather candidates, then reranks them with:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The reranker reads the question and each candidate chunk together, then returns a stronger relevance ordering for the final top-k results.

Disable reranking when you want a faster local smoke test:

```bash
python3 -m app.cli --no-reranker --store simple --index-name sample query \
  --question "How does login work?"
```

## End-To-End Ask

`ask` indexes the repo, retrieves relevant chunks, reranks them, and prints the prompt to send to an LLM.

```bash
python3 -m app.cli --store simple --index-name sample ask \
  --repo work/sample_repo \
  --question "How does login work?" \
  --top-k 3
```

## Optional API

```bash
uvicorn app.api:app --reload
```

Endpoints:

- `POST /index`
- `POST /query`

## Notes

Changing embedding models requires rebuilding the vector index, because each model produces vectors in its own vector space.

## Credits

<p align="center">
  <img src="assets/codex-color.svg" alt="Codex logo" width="42">
  &nbsp;&nbsp;
  <img src="assets/deepseek-color.svg" alt="DeepSeek logo" width="42">
</p>

Built with help from Codex and DeepSeek.
