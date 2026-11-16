"""Regenerate data seeds 0, 1 and 2 from the licensed original master inputs."""
import argparse
import json
from pathlib import Path

from mirtards_release.preprocess import generate_dataset

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--master-dir", required=True)
parser.add_argument("--catalog", default="benchmark_data/source/mirbase_r23_mature_catalog.tsv")
parser.add_argument("--output", required=True)
parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
args = parser.parse_args()
for seed in args.seeds:
    audit = generate_dataset(args.master_dir, Path(args.output) / f"seed{seed}", seed, args.catalog)
    print(json.dumps(audit), flush=True)

