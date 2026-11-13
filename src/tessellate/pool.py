"""Pooling helpers shared by the Tessellate encoders."""

from typing import List


def cls_pool(token_vectors: List[List[float]]) -> List[float]:
    return list(token_vectors[0]) if token_vectors else []


def max_pool(token_vectors: List[List[float]]) -> List[float]:
    if not token_vectors:
        return []
    width = len(token_vectors[0])
    return [max(row[i] for row in token_vectors) for i in range(width)]


def attention_pool(token_vectors: List[List[float]], weights: List[float]) -> List[float]:
    if not token_vectors:
        return []
    width = len(token_vectors[0])
    out = [0.0] * width
    for row, weight in zip(token_vectors, weights):
        for i, value in enumerate(row):
            out[i] += value * weight
    return out
