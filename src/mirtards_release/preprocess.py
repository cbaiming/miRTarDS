"""Original sampling, disease-profile isolation and two inner folds."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from . import sampling


def profile_groups(profiles, component=None):
    parent = {str(entity): str(entity) for entity in profiles}
    def find(entity):
        while entity != parent[entity]:
            parent[entity] = parent[parent[entity]]
            entity = parent[entity]
        return entity
    def join(left, right):
        a, b = find(left), find(right)
        parent[max(a, b)] = min(a, b)
    hashes = {}
    first_profile = {}
    for entity in sorted(profiles):
        values = tuple(sorted(str(item[1]) for item in profiles[entity]))
        hashes[entity] = hashlib.sha256(json.dumps(values, ensure_ascii=False).encode()).hexdigest()
        if values in first_profile:
            join(first_profile[values], entity)
        else:
            first_profile[values] = entity
    if component is not None:
        first_component = {}
        for entity in sorted(profiles):
            key = component[entity]
            if key in first_component:
                join(first_component[key], entity)
            else:
                first_component[key] = entity
    return {entity: find(entity) for entity in profiles}, hashes


def generate_dataset(master_dir, output_dir, seed, catalog):
    """Regenerate one split from a locally acquired, licensed original master."""
    folder = Path(output_dir)
    if folder.exists():
        raise FileExistsError("Choose a new dataset output directory")
    folder.mkdir(parents=True)
    sampling.MASTER = Path(master_dir).resolve()
    sampling.OUT = folder / "legacy"
    sampling.SEED = int(seed)
    sampling.main()
    pairs = pd.read_csv(folder / "legacy/pairs_audit.tsv", sep="\t", dtype=str, keep_default_na=False)
    profiles = json.loads((folder / "legacy/disease_profiles.json").read_text(encoding="utf-8"))
    component = pairs.set_index("mature_accession").precursor_component.to_dict()
    ga, ha = profile_groups(profiles["mirna"], component)
    gb, hb = profile_groups(profiles["gene"])
    rows = pd.DataFrame(dict(
        row_id="mti:" + pairs.pair_id, split=pairs.split, a_id=pairs.mature_accession, b_id=pairs.gene_id,
        a_group=pairs.mature_accession.map(ga), b_group=pairs.gene_id.map(gb), label=pairs.observed_y.astype(int),
        disease_count_a=pairs["miRNA disease count"].astype(int), disease_count_b=pairs["gene disease count"].astype(int),
        precursor_component=pairs.precursor_component, disease_profile_hash_a=pairs.mature_accession.map(ha),
        disease_profile_hash_b=pairs.gene_id.map(hb), legacy_split=pairs.split, exclusion_reason=""
    ))
    for side in ("a", "b"):
        partitions = rows.groupby(side + "_group").legacy_split.nunique()
        bad = set(partitions[partitions > 1].index)
        hit = rows[side + "_group"].isin(bad)
        rows.loc[hit, "split"] = "quarantine"
        rows.loc[hit, "exclusion_reason"] += side + "_cross_split_profile;"
    valid = rows.loc[rows.split != "quarantine"]
    for column in ("a_id", "b_id", "a_group", "b_group", "precursor_component",
                   "disease_profile_hash_a", "disease_profile_hash_b"):
        if valid.groupby(column).split.nunique().max() != 1:
            raise ValueError("Unisolated input identity: " + column)
    rng = np.random.default_rng(int(seed) + 10000)
    for side in ("a", "b"):
        groups = np.asarray(sorted(set(rows.loc[rows.split == "train", side + "_group"])))
        assignment = {str(group): i % 5 for i, group in enumerate(groups[rng.permutation(len(groups))])}
        rows[side + "_dev_fold"] = rows[side + "_group"].map(assignment).fillna(-1).astype(int)
    indices = {part: np.flatnonzero(rows.split.eq(part)) for part in ("train", "valid", "test")}
    for fold in (0, 1):
        roles = (("fit", set(range(5)) - {fold, (fold + 1) % 5}), ("tune", {(fold + 1) % 5}),
                 ("train", set(range(5)) - {fold}), ("assess", {fold}))
        for role, allowed in roles:
            indices[f"{fold}_{role}"] = np.flatnonzero(
                rows.split.eq("train") & rows.a_dev_fold.isin(allowed) & rows.b_dev_fold.isin(allowed)
            )
    sequences = {}
    with Path(catalog).open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            sequences.setdefault(row["mature_accession"], set()).add(row["mature_sequence"].upper().replace("T", "U"))
    train_sequences = {sequence for entity in rows.loc[rows.split == "train", "a_id"]
                       for sequence in sequences.get(entity, ()) if sequence}
    rows["sequence_seen_in_train"] = [bool(train_sequences.intersection(sequences.get(entity, ()))) for entity in rows.a_id]
    rows.to_csv(folder / "rows.tsv.gz", sep="\t", index=False, compression={"method": "gzip", "mtime": 0})
    np.savez_compressed(folder / "indices.npz", **indices)
    with __import__("gzip").open(folder / "profiles.json.gz", "wt", encoding="utf-8") as handle:
        json.dump({"a": profiles["mirna"], "b": profiles["gene"]}, handle, ensure_ascii=False)
    audit = dict(data_seed=int(seed), training_seeds=[41, 42, 43], inner_seed=int(seed) + 10000,
                 counts={name: dict(rows=len(index), P=int(rows.iloc[index].label.sum()),
                                    U=int(len(index) - rows.iloc[index].label.sum())) for name, index in indices.items()},
                 infeasible_partitions=[name for name, index in indices.items() if rows.iloc[index].label.nunique() != 2],
                 sequence_seen_test_rows=int(rows.loc[rows.split == "test", "sequence_seen_in_train"].sum()))
    (folder / "construction.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit

