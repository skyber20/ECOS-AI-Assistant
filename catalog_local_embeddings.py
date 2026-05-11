import hashlib
import math
import re
from typing import Iterable


EMBEDDING_DIM = 384
TOKEN_RE = re.compile(r"[\w%.-]+", re.UNICODE)


def embed_text(text: str, dimension: int = EMBEDDING_DIM) -> list[float]:
    vector = [0.0] * dimension
    for token, weight in _features(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dimension
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign * weight

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def embed_texts(texts: Iterable[str], dimension: int = EMBEDDING_DIM) -> list[list[float]]:
    return [embed_text(text, dimension=dimension) for text in texts]


def _features(text: str) -> Iterable[tuple[str, float]]:
    normalized = " ".join(text.lower().split())
    tokens = TOKEN_RE.findall(normalized)
    for token in tokens:
        yield f"w:{token}", 1.0
        if len(token) >= 5:
            for size in (3, 4):
                for index in range(0, len(token) - size + 1):
                    yield f"c{size}:{token[index:index + size]}", 0.35

    for left, right in zip(tokens, tokens[1:]):
        yield f"b:{left}_{right}", 0.75
