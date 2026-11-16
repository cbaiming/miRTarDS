"""Traditional disease-profile baselines for the released test candidates.

MISIM follows Wang et al. (2010), DOI 10.1093/bioinformatics/btq241,
equations 1--5. Applying its disease-profile score to a miRNA--gene pair is
an adaptation of the original miRNA functional-similarity method.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import unicodedata
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def normalize_name(value):
    """NFKC, collapsed whitespace, and lowercase, with no synonym expansion."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).strip()).lower()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(8 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


class MISIM2010:
    """Unique MeSH-descriptor DAG and fixed maximum ancestor contributions."""

    def __init__(self, rows, delta=0.5):
        if not 0 < delta < 1:
            raise ValueError("delta must be between zero and one")
        self.delta = delta
        self.codes = defaultdict(set)
        self.names = {}
        by_code = {}
        for name, code in rows:
            name, code = name.strip(), code.strip()
            if not re.fullmatch(r"C\d+(?:\.\d+)*", code):
                raise ValueError("Expected a Category C MeSH address: " + code)
            key = normalize_name(name)
            if key in self.names and self.names[key] != name:
                raise ValueError("Ambiguous normalized descriptor: " + key)
            if code in by_code and by_code[code] != name:
                raise ValueError("MeSH address assigned to multiple descriptors")
            self.names[key] = name
            self.codes[name].add(code)
            by_code[code] = name

        self.parents = {name: set() for name in self.codes}
        for code, name in by_code.items():
            if "." not in code:
                continue
            parent_code = code.rsplit(".", 1)[0]
            if parent_code not in by_code:
                raise ValueError("Missing ancestor address: " + parent_code)
            parent = by_code[parent_code]
            if parent == name:
                raise ValueError("Self-parent descriptor")
            self.parents[name].add(parent)

        active, visited = set(), set()

        def visit(node):
            if node in active:
                raise ValueError("MeSH descriptor graph contains a cycle")
            if node in visited:
                return
            active.add(node)
            for parent in self.parents[node]:
                visit(parent)
            active.remove(node)
            visited.add(node)

        for node in self.parents:
            visit(node)
        self.weights = {name: self._contributions(name) for name in self.codes}
        self.values = {name: sum(v.values()) for name, v in self.weights.items()}

    @classmethod
    def from_csv(cls, path):
        with Path(path).open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            if header[:2] != ["Disease Name", "TreeNumber"]:
                raise ValueError("Expected Disease Name,TreeNumber CSV columns")
            return cls(list(reader))

    def _contributions(self, disease):
        distances = {disease: 0}
        queue = deque([disease])
        while queue:
            node = queue.popleft()
            for parent in self.parents[node]:
                if parent not in distances:
                    distances[parent] = distances[node] + 1
                    queue.append(parent)
        return {node: self.delta**distance for node, distance in distances.items()}

    def disease_similarity(self, left, right):
        a, b = self.weights[left], self.weights[right]
        shared = a.keys() & b.keys()
        numerator = sum(a[node] + b[node] for node in shared)
        return numerator / (self.values[left] + self.values[right])

    def score(self, left, right):
        """Direct reference formula, requiring both mapped profiles."""
        a, b = sorted(set(left)), sorted(set(right))
        if not a or not b:
            raise ValueError("MISIM requires two nonempty mapped profiles")
        forward = sum(max(self.disease_similarity(x, y) for y in b) for x in a)
        backward = sum(max(self.disease_similarity(x, y) for x in a) for y in b)
        return (forward + backward) / (len(a) + len(b))


def _semantic_matrix(mesh, descriptors):
    """Vectorized evaluation of the same descriptor-level MISIM formula."""
    ancestor_codes = {name: i for i, name in enumerate(mesh.codes)}
    row, column, value = [], [], []
    for i, disease in enumerate(descriptors):
        for ancestor, weight in mesh.weights[disease].items():
            row.append(i)
            column.append(ancestor_codes[ancestor])
            value.append(weight)
    weights = csr_matrix(
        (value, (row, column)), shape=(len(descriptors), len(ancestor_codes))
    )
    binary = weights.copy()
    binary.data[:] = 1
    direct = (weights @ binary.T).toarray()
    disease_values = np.array([mesh.values[disease] for disease in descriptors])
    similarities = (direct + direct.T) / (
        disease_values[:, None] + disease_values[None, :]
    )
    np.testing.assert_allclose(similarities, similarities.T, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.diag(similarities), 1, rtol=0, atol=1e-12)
    if similarities.min() < 0 or similarities.max() > 1 + 1e-12:
        raise ValueError("MISIM similarity outside [0, 1]")
    return similarities


def _load_profiles(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        data = json.load(handle)
    if set(data) != {"a", "b"}:
        raise ValueError("Expected a/b disease-profile dictionaries")
    return data


def compute_baselines(profile_path, rows_path, mesh_path, partition="test"):
    """Return fixed scores for one partition, without reading outcome labels.

    The returned columns align to rows.tsv.gz by row_id. Unmapped MISIM
    endpoints produce NaN with misim_covered=False. Jaccard is available when
    both original disease-name sets are nonempty.
    """
    profiles = _load_profiles(profile_path)
    frame = pd.read_csv(rows_path, sep="\t", usecols=["row_id", "a_id", "b_id", "split"], dtype={"row_id": str, "a_id": str, "b_id": str})
    if not frame.row_id.is_unique:
        raise ValueError("Duplicate row_id in candidate table")
    if partition != "all":
        frame = frame.loc[frame.split.eq(partition)]
    frame = frame[["row_id", "a_id", "b_id"]].copy()
    if frame.empty:
        raise ValueError("Requested candidate partition is empty")

    mesh = MISIM2010.from_csv(mesh_path)
    raw = {}
    mapped = {}
    for side in ("a", "b"):
        raw[side], mapped[side] = {}, {}
        for entity in sorted(frame[side + "_id"].unique()):
            bag = profiles[side][entity]
            names = {normalize_name(entry[1]) for entry in bag}
            if not names or "" in names:
                raise ValueError("Empty original disease profile")
            raw[side][entity] = names
            mapped[side][entity] = sorted(
                {mesh.names[name] for name in names if name in mesh.names}
            )
    descriptors = sorted(
        {disease for side in mapped.values() for bag in side.values() for disease in bag}
    )
    if not descriptors:
        raise ValueError("No disease names map to the supplied MeSH snapshot")
    lookup = {name: i for i, name in enumerate(descriptors)}
    similarities = _semantic_matrix(mesh, descriptors)
    mapped_indices = {
        side: {entity: np.array([lookup[d] for d in bag], dtype=int)
               for entity, bag in entities.items()}
        for side, entities in mapped.items()
    }
    records = []
    for pair in frame.itertuples(index=False):
        left, right = raw["a"][pair.a_id], raw["b"][pair.b_id]
        a = mapped_indices["a"][pair.a_id]
        b = mapped_indices["b"][pair.b_id]
        covered = bool(len(a) and len(b))
        misim = float("nan")
        if covered:
            sub = similarities[np.ix_(a, b)]
            misim = float((sub.max(1).sum() + sub.max(0).sum()) / (len(a) + len(b)))
        jaccard = len(left & right) / len(left | right)
        records.append(
            dict(row_id=pair.row_id, a_id=pair.a_id, b_id=pair.b_id,
                 misim_score=misim, misim_covered=covered,
                 mesh_count_a=len(a), mesh_count_b=len(b),
                 raw_name_count_a=len(left), raw_name_count_b=len(right),
                 jaccard_similarity=jaccard, jaccard_distance=1 - jaccard)
        )
    result = pd.DataFrame(records)
    result.attrs["provenance"] = dict(
        misim="Wang2010 equations1-5, delta0.5, adapted to miRNA-gene disease profiles",
        misim_reference="https://doi.org/10.1093/bioinformatics/btq241",
        mapping="NFKC + whitespace collapse + lowercase exact MeSH Category C name matching",
        unmapped="Omit unmapped names; abstain if either mapped endpoint is empty",
        aliases_or_fuzzy_matching=False,
        score_direction="Higher MISIM/Jaccard similarity predicts P; lower Jaccard distance predicts P",
        labels_used_for_scoring=False,
        profiles_sha256=sha256_file(profile_path),
        rows_sha256=sha256_file(rows_path),
        mesh_sha256=sha256_file(mesh_path),
        partition=partition,
        rows=len(result),
        misim_covered_rows=int(result.misim_covered.sum()),
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--partition", default="test")
    args = parser.parse_args()
    result = compute_baselines(args.profiles, args.rows, args.mesh, args.partition)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, sep="\t", index=False, float_format="%.17g")
    manifest = output.with_suffix(output.suffix + ".json")
    manifest.write_text(json.dumps(result.attrs["provenance"], indent=2) + "\n")
    print(json.dumps(result.attrs["provenance"], indent=2))


if __name__ == "__main__":
    main()

