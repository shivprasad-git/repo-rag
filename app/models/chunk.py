from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class Chunk:
    content: str
    metadata: dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid4()))

