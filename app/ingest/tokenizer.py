from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


class Tokenizer:
    """Token counter backed by the same tokenizer used by the embedding model.

    Loads only the tokenizer (not the full model) via HuggingFace's
    ``AutoTokenizer``, giving accurate token counts that match what the
    embedding model actually sees, without the overhead of loading the
    entire model weights.
    """

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
        return len(self._encode(text))

    def count_many(self, texts: Sequence[str]) -> list[int]:
        """Return token counts for each text in *texts*.

        Uses batched encoding for efficiency.  The result list is in the
        same order as *texts*.
        """
        return [len(ids) for ids in self._encode_many(list(texts))]

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

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