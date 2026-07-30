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
- Split oversized functions, methods, Markdown sections, and text files into overlapping parts using the embedding model's actual tokenizer.
- Generate embeddings through a pluggable interface.
- Store embeddings with minimal filter metadata in Chroma or a local JSON vector store.
- Store full chunk content and full metadata separately in a JSON chunk document store.
- Embed only searchable code/documentation chunks by default; keep file metadata chunks in the document store without adding them to normal semantic retrieval.
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

`ask` retrieves relevant chunks for a question and prints the prompt to send to an LLM. The first time you run `ask` for a given `--index-name`, it indexes the repository. Subsequent runs skip re-indexing and go straight to retrieval, making repeated queries fast.

```bash
python3 -m app.cli --store simple --index-name smoke ask \
  --repo work/sample_repo \
  --question "How does login work?" \
  --top-k 3
```

To force a fresh index, delete the existing index files or run the `index` command explicitly.

## Optional API

```bash
uvicorn app.api:app --reload
```

Endpoints:

- `POST /index`
- `POST /query`

## Notes

Changing embedding models requires rebuilding the vector index, because each model produces vectors in its own vector space.

Chunk storage and chunk retrieval are intentionally separate. Repo RAG stores metadata chunks, parse details, and searchable code/documentation chunks in the chunk document store, but only indexes configured searchable chunk types into the vector store. This keeps metadata available without adding noise to normal semantic search.

Oversized chunks are split after parsing. By default, chunks over `700` tokens (including special tokens) are split with `100` tokens of overlap. Token counts come from the embedding model's actual tokenizer (``sentence-transformers/all-MiniLM-L6-v2`` WordPiece tokenizer), so they match what the model will see during embedding. Split parts keep the original ``chunk_type`` and add ``is_chunk_part``, ``part_index``, ``part_count``, and ``parent_chunk_id`` metadata.

When retrieval returns a split chunk part, Repo RAG includes one neighboring part on each side by default. This keeps retrieval precise while giving the final prompt enough nearby context.

Chunk type names are centralized in `ChunkType`. Current values are `class`, `file_metadata`, `function`, `imports`, `markdown_section`, `method`, `parse_error`, and `text_file`.

## Credits

<p align="center">
  <img src="assets/codex-color.svg" alt="Codex logo" width="42">
  &nbsp;&nbsp;
  <img src="assets/deepseek-color.svg" alt="DeepSeek logo" width="42">
</p>
