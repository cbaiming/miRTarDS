"""Encode disease names with the complete local released checkpoint."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mirtards_release.biolord import encode_texts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("texts", nargs="+")
    parser.add_argument("--model", type=Path, default=ROOT / "models/biolord-mondo-dora")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("disease_embeddings.npy"))
    args = parser.parse_args()
    import numpy as np

    vectors = encode_texts(args.texts, args.model, device=args.device)
    np.save(args.output, vectors)
    print(f"Saved {len(vectors)} normalized 768-dimensional vectors to {args.output}")


if __name__ == "__main__":
    main()
