from __future__ import annotations

import argparse

from app.config import Settings
from app.llm import build_prompt
from app.pipeline import ask_repository, index_repository, query_repository


def main() -> None:
    parser = argparse.ArgumentParser(description="Repo RAG Phase 1 MVP")
    parser.add_argument("--store", choices=["chroma", "simple"], default="chroma")
    parser.add_argument("--index-name", default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Index a GitHub URL or local repository")
    index_parser.add_argument("--repo", required=True)

    query_parser = subparsers.add_parser("query", help="Retrieve relevant chunks for a question")
    query_parser.add_argument("--question", required=True)
    query_parser.add_argument("--top-k", type=int, default=5)
    query_parser.add_argument("--prompt", action="store_true", help="Print an LLM prompt instead of raw matches")

    ask_parser = subparsers.add_parser("ask", help="Index a repository and print an LLM-ready prompt")
    ask_parser.add_argument("--repo", required=True)
    ask_parser.add_argument("--question", required=True)
    ask_parser.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()
    settings = Settings()

    if args.command == "index":
        repo_path, chunk_count = index_repository(args.repo, args.store, settings, args.index_name)
        print(f"Indexed {chunk_count} chunks from {repo_path}")
    elif args.command == "query":
        matches = query_repository(args.question, args.store, settings, args.top_k, args.index_name)
        if args.prompt:
            print(build_prompt(args.question, matches))
        else:
            _print_matches(matches)
    elif args.command == "ask":
        print(ask_repository(args.repo, args.question, args.store, settings, args.top_k, args.index_name))


def _print_matches(matches) -> None:
    for index, (chunk, score) in enumerate(matches, start=1):
        metadata = chunk.metadata
        print(f"{index}. score={score:.4f} {metadata.get('file_path')}:{metadata.get('start_line')}-{metadata.get('end_line')}")
        print(f"   {metadata.get('chunk_type')} {metadata.get('symbol')}")


if __name__ == "__main__":
    main()

