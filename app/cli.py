from __future__ import annotations

import argparse
import json

from app.config import Settings
from app.indexing import check_index_health, format_health_report, format_index_info, get_index_info
from app.llm import build_prompt
from app.logging_config import setup_logging
from app.pipeline import ask_repository, index_repository, query_repository
from app.retrieval.filters import MetadataFilters


def main() -> None:
    parser = argparse.ArgumentParser(description="Repo RAG Phase 1 MVP")
    parser.add_argument("--store", choices=["chroma", "simple"], default="chroma")
    parser.add_argument("--index-name", default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--reranker-model", default=None)
    parser.add_argument("--no-reranker", action="store_true", help="Disable MiniLM cross-encoder reranking")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug-level logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Index a GitHub URL or local repository")
    index_parser.add_argument("--repo", required=True)

    query_parser = subparsers.add_parser("query", help="Retrieve relevant chunks for a question")
    query_parser.add_argument("--question", required=True)
    query_parser.add_argument("--top-k", type=int, default=5)
    query_parser.add_argument("--prompt", action="store_true", help="Print an LLM prompt instead of raw matches")
    query_parser.add_argument("--json", action="store_true", help="Print retrieved chunks as JSON")
    query_parser.add_argument("--show-content", action="store_true", help="Print a content preview for each match")
    query_parser.add_argument("--content-chars", type=int, default=700, help="Maximum preview characters per match")
    _add_filter_args(query_parser)

    ask_parser = subparsers.add_parser("ask", help="Index a repository and print an LLM-ready prompt")
    ask_parser.add_argument("--repo", required=True)
    ask_parser.add_argument("--question", required=True)
    ask_parser.add_argument("--top-k", type=int, default=5)
    _add_filter_args(ask_parser)

    health_parser = subparsers.add_parser("health", help="Check index/document/vector store consistency")
    health_parser.add_argument("--repo", help="Optional local or GitHub repo to check manifest file hashes against")

    subparsers.add_parser("info", help="Show index settings and store counts")

    args = parser.parse_args()
    setup_logging(level="DEBUG" if args.verbose else None)
    settings = _settings_from_args(args)

    if args.command == "index":
        repo_path, chunk_count = index_repository(args.repo, args.store, settings, args.index_name)
        print(f"Indexed {chunk_count} chunks from {repo_path}")
    elif args.command == "query":
        matches = query_repository(
            args.question,
            args.store,
            settings,
            args.top_k,
            args.index_name,
            filters=_metadata_filters_from_args(args),
        )
        if args.prompt:
            print(build_prompt(args.question, matches, settings=settings))
        elif args.json:
            print(_matches_as_json(matches))
        else:
            _print_matches(matches, show_content=args.show_content, content_chars=args.content_chars)
    elif args.command == "ask":
        print(
            ask_repository(
                args.repo,
                args.question,
                args.store,
                settings,
                args.top_k,
                args.index_name,
                filters=_metadata_filters_from_args(args),
            )
        )
    elif args.command == "health":
        report = check_index_health(args.store, settings, args.index_name, repo=args.repo)
        print(format_health_report(report))
    elif args.command == "info":
        info = get_index_info(args.store, settings, args.index_name)
        print(format_index_info(info))


def _add_filter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--language", help="Only retrieve chunks for this language, e.g. python or markdown")
    parser.add_argument("--chunk-type", help="Only retrieve this chunk type, e.g. function, method, class")
    parser.add_argument("--path", dest="path_prefix", help="Only retrieve chunks whose file path starts with this value")


def _settings_from_args(args: argparse.Namespace) -> Settings:
    settings = Settings()
    if args.embedding_model is None and args.reranker_model is None and not args.no_reranker:
        return settings
    return Settings(
        embedding_model=args.embedding_model or settings.embedding_model,
        reranker_model=args.reranker_model or settings.reranker_model,
        reranker_enabled=False if args.no_reranker else settings.reranker_enabled,
    )


def _metadata_filters_from_args(args: argparse.Namespace) -> MetadataFilters | None:
    filters = MetadataFilters(
        language=args.language,
        chunk_type=args.chunk_type,
        path_prefix=args.path_prefix,
    )
    return filters if filters.has_filters else None


def _print_matches(matches, show_content: bool = False, content_chars: int = 700) -> None:
    for index, (chunk, score) in enumerate(matches, start=1):
        metadata = chunk.metadata
        print(f"{index}. score={score:.4f} {metadata.get('file_path')}:{metadata.get('start_line')}-{metadata.get('end_line')}")
        print(f"   {metadata.get('chunk_type')} {metadata.get('symbol')}")
        if show_content:
            print(_preview_content(chunk.content, content_chars))


def _preview_content(content: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    preview = content.strip()
    if len(preview) > max_chars:
        preview = f"{preview[:max_chars].rstrip()}..."
    indented = "\n".join(f"   | {line}" for line in preview.splitlines())
    return indented or "   |"


def _matches_as_json(matches) -> str:
    records = []
    for chunk, score in matches:
        records.append(
            {
                "id": chunk.id,
                "score": score,
                "content": chunk.content,
                "metadata": chunk.metadata,
            }
        )
    return json.dumps(records, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
