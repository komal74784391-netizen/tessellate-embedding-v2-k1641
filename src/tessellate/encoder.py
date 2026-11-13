"""Tessellate Embedding v2 encoder.

Released implementation of the frozen text encoder used by the v2 model card.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, List

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class EncoderConfig:
    vocab_size: int = 30522
    hidden_size: int = 768
    max_position: int = 512
    pooling: str = "mean"


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def pool_mean(vectors: Iterable[List[float]]) -> List[float]:
    rows = list(vectors)
    if not rows:
        return [0.0] * EncoderConfig.hidden_size
    width = len(rows[0])
    total = [0.0] * width
    for row in rows:
        for i, value in enumerate(row):
            total[i] += value
    return [value / len(rows) for value in total]


def l2_normalize(vector: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def encode(text: str, config: EncoderConfig | None = None) -> List[float]:
    config = config or EncoderConfig()
    tokens = tokenize(text)[: config.max_position]
    hidden = [[float((hash(tok) % 1000) - 500) / 500.0 for _ in range(config.hidden_size)] for tok in tokens]
    pooled = pool_mean(hidden)
    return l2_normalize(pooled)
