from __future__ import annotations

import math
import re
from collections import Counter

from app.models import Chunk


class BM25KeywordIndex:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.documents = [_tokenize(_search_text(chunk)) for chunk in chunks]
        self.doc_lengths = [len(document) for document in self.documents]
        self.average_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        self.term_frequencies = [Counter(document) for document in self.documents]
        self.document_frequencies = self._document_frequencies()

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        query_terms = _tokenize(query)
        if not query_terms or not self.chunks:
            return []

        scored: list[tuple[Chunk, float]] = []
        for index, chunk in enumerate(self.chunks):
            score = self._score(query_terms, index)
            score += _metadata_bonus(chunk, query_terms)
            if score > 0:
                scored.append((chunk, score))

        return sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]

    def _score(self, query_terms: list[str], document_index: int) -> float:
        score = 0.0
        document_length = self.doc_lengths[document_index]
        frequencies = self.term_frequencies[document_index]

        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if frequency == 0:
                continue
            idf = self._idf(term)
            denominator = frequency + self.k1 * (
                1 - self.b + self.b * document_length / max(self.average_doc_length, 1.0)
            )
            score += idf * (frequency * (self.k1 + 1)) / denominator

        return score

    def _idf(self, term: str) -> float:
        total_documents = len(self.documents)
        containing_documents = self.document_frequencies.get(term, 0)
        return math.log(1 + (total_documents - containing_documents + 0.5) / (containing_documents + 0.5))

    def _document_frequencies(self) -> dict[str, int]:
        frequencies: dict[str, int] = {}
        for document in self.documents:
            for term in set(document):
                frequencies[term] = frequencies.get(term, 0) + 1
        return frequencies


def _search_text(chunk: Chunk) -> str:
    metadata = chunk.metadata
    symbol = str(metadata.get("symbol", ""))
    file_path = str(metadata.get("file_path", ""))
    return "\n".join(
        [
            file_path,
            file_path,
            symbol,
            symbol,
            symbol,
            str(metadata.get("chunk_type", "")),
            chunk.content,
        ]
    )


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text.lower())


def _metadata_bonus(chunk: Chunk, query_terms: list[str]) -> float:
    metadata = chunk.metadata
    symbol_terms = set(_tokenize(str(metadata.get("symbol", ""))))
    path_terms = set(_tokenize(str(metadata.get("file_path", ""))))
    query_set = set(query_terms)
    return 2.0 * len(query_set & symbol_terms) + 0.75 * len(query_set & path_terms)
