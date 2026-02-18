from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


OPENAI_EMBED_MODEL = "text-embedding-3-small"
STUB_EMBED_DIM = 256


def embed_texts(
    texts: list[str],
    *,
    provider: str,
    openai_api_key: str | None,
) -> list[list[float]]:
    if not texts:
        return []

    use_openai = provider == "openai" and bool(openai_api_key)
    if use_openai:
        return _embed_openai(texts=texts, api_key=openai_api_key or "", model=OPENAI_EMBED_MODEL)

    return [_embed_stub(text) for text in texts]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    size = min(len(left), len(right))
    if size == 0:
        return 0.0

    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for idx in range(size):
        lval = left[idx]
        rval = right[idx]
        dot += lval * rval
        left_norm += lval * lval
        right_norm += rval * rval

    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    return dot / (math.sqrt(left_norm) * math.sqrt(right_norm))


def _embed_stub(text: str) -> list[float]:
    vector = [0.0] * STUB_EMBED_DIM
    tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    if not tokens:
        vector[0] = 1.0
        return vector

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:2], "big") % STUB_EMBED_DIM
        sign = -1.0 if digest[2] & 1 else 1.0
        weight = 1.0 + (digest[3] / 255.0)
        vector[index] += sign * weight

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector

    return [value / norm for value in vector]


def _embed_openai(*, texts: list[str], api_key: str, model: str) -> list[list[float]]:
    payload = {"model": model, "input": texts}
    request = Request(
        "https://api.openai.com/v1/embeddings",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"OpenAI embeddings HTTP error: {exc.code} {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"OpenAI embeddings network error: {exc.reason}") from exc

    raw_items = body.get("data", [])
    if not isinstance(raw_items, list):
        raise RuntimeError("OpenAI embeddings response missing data list")

    by_index: dict[int, list[float]] = {}
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        embedding = item.get("embedding")
        if isinstance(index, int) and isinstance(embedding, list):
            values = _coerce_embedding(embedding)
            by_index[index] = values

    return [by_index.get(idx, _embed_stub(texts[idx])) for idx in range(len(texts))]


def _coerce_embedding(raw_embedding: list[Any]) -> list[float]:
    values: list[float] = []
    for value in raw_embedding:
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values
