from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

from app.embeddings.base import EmbeddingProvider


class HashEmbeddingProvider(EmbeddingProvider):
    """Deterministic local embedding provider for development and tests."""

    def __init__(self, dimensions: int = 384) -> None:
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text.lower())
        counts = Counter(tokens)

        for token, count in counts.items():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self._dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign * (1.0 + math.log(count))

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

