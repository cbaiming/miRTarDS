"""Frozen off-diagonal disease-profile statistics and disease-vector encoders."""
import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np


def statistics(values):
    flat = np.asarray(values, dtype=np.float64).ravel()
    if not len(flat):
        return np.zeros(4, dtype=np.float64)
    if not np.isfinite(flat).all():
        raise ValueError("Nonfinite similarity values")
    mean = float(flat.mean())
    centered = flat - mean
    std = float(np.sqrt(np.mean(centered ** 2)))
    skew = float(np.mean(centered ** 3) / std ** 3) if std > 1e-7 * max(abs(mean), 1e-30) else 0.0
    return np.array([mean, flat.max(), std, skew])


def profile_texts(entries):
    concepts = {}
    for concept, text in entries:
        if concept in concepts and concepts[concept] != text:
            raise ValueError("Conflicting texts for the same disease concept")
        concepts[concept] = text
    if not concepts:
        raise ValueError("Empty disease profile")
    return [concepts[key] for key in sorted(concepts)]


def offdiag12(rows, profiles, texts, vectors):
    """Keep concept multiplicity and exclude position diagonals only in self blocks."""
    lookup = {text: index for index, text in enumerate(texts)}
    if len(lookup) != len(texts):
        raise ValueError("Disease vocabulary must be unique")
    units = np.asarray(vectors, dtype=np.float64)
    if units.ndim != 2 or units.shape[0] != len(texts) or not np.isfinite(units).all():
        raise ValueError("Invalid vector matrix")
    norms = np.linalg.norm(units, axis=1, keepdims=True)
    if (norms <= 0).any():
        raise ValueError("Zero disease vector")
    units = units / norms
    bags = {side: {} for side in ("a", "b")}
    for side in bags:
        for entity in rows[side + "_id"].unique():
            bags[side][entity] = np.array(
                [lookup[text] for text in profile_texts(profiles[side][entity])], dtype=int
            )
    def block(left, right):
        kernel = np.clip(units[left] @ units[right].T, 0.0, 1.0)
        kernel[left[:, None] == right] = 1.0
        return kernel
    self_cache = {}
    for entities in bags.values():
        for indices in entities.values():
            key = tuple(indices)
            if key not in self_cache:
                kernel = block(indices, indices)
                self_cache[key] = statistics(kernel[~np.eye(len(indices), dtype=bool)])
    result = []
    for left, right in rows[["a_id", "b_id"]].itertuples(index=False, name=None):
        a, b = bags["a"][left], bags["b"][right]
        result.append(np.r_[statistics(block(a, b)), self_cache[tuple(a)], self_cache[tuple(b)]])
    return np.round(np.asarray(result), 6)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args):
        raise RuntimeError("Unexpected redirect from embedding endpoint")


def encode_openai_large(texts, batch_size=128):
    """Read OPENAI_API_KEY from the environment. Never write credentials to artifacts."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("Set OPENAI_API_KEY before requesting new embeddings")
    client = urllib.request.build_opener(_NoRedirect)
    arrays = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        request = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=json.dumps(dict(model="text-embedding-3-large", input=batch, encoding_format="base64")).encode(),
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        )
        for attempt in range(3):
            try:
                with client.open(request, timeout=60) as response:
                    result = json.load(response)
                break
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                    raise RuntimeError("Embedding API HTTP " + str(exc.code)) from None
                time.sleep(2 ** attempt)
        records = sorted(result["data"], key=lambda item: item["index"])
        if result["model"] != "text-embedding-3-large" or [v["index"] for v in records] != list(range(len(batch))):
            raise ValueError("Embedding response identity mismatch")
        values = np.stack([
            np.frombuffer(base64.b64decode(item["embedding"], validate=True), dtype="<f4")
            for item in records
        ])
        if values.shape != (len(batch), 3072) or not np.isfinite(values).all():
            raise ValueError("Unexpected embedding shape")
        arrays.append(values)
    return np.vstack(arrays)

