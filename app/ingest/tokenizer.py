from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


class Tokenizer:
    """Token counter backed by the embedding model's own tokenizer (weights not loaded)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._tokenizer: object | None = None

    @property
    def tokenizer(self) -> object:
        if self._tokenizer is None:
            self._tokenizer = self._load_tokenizer()
        return self._tokenizer

    def _load_tokenizer(self) -> object:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Install transformers to use the actual tokenizer."
            ) from exc
        return AutoTokenizer.from_pretrained(self.model_name)

    def count(self, text: str) -> int:
        """Return the number of tokens the embedding model will see for *text*, special tokens included."""
        return len(self.tokenizer.encode(text, add_special_tokens=True))

    def count_many(self, texts: Sequence[str]) -> list[int]:
        """Return token counts for each text in *texts*, in the same order.

        Special tokens are excluded because they are added once per chunk, not per line.
        """
        return [len(ids) for ids in self._encode_many(list(texts))]

    def _encode_many(self, texts: list[str]) -> list[list[int]]:
        encoded = self.tokenizer(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )
        return encoded["input_ids"]


@lru_cache(maxsize=1)
def get_tokenizer(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Tokenizer:
    """Return a cached singleton Tokenizer instance."""
    return Tokenizer(model_name)