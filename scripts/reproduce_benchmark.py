"""Reproduce both released encoder conditions using fixed data and training seeds."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


FEATURE_NAMES = {
    "openai-large": "features_openai_large.npz",
    "biolord": "features_biolord.npz",
    "biolord-original": "features_biolord_original.npz",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def input_receipt(data, feature_path, encoder, training_seed):
    import mirtards_release
    package = Path(mirtards_release.__file__).parent
    code_names = ("cli.py", "training.py", "predictor.py", "metrics.py")
    return {
        "receipt_version": 2,
        "encoder": encoder,
        "training_seed": training_seed,
        "inputs": {
            "rows.tsv.gz": sha(data / "rows.tsv.gz"),
            "indices.npz": sha(data / "indices.npz"),
            "features": sha(feature_path),
        },
        "code": {name: sha(package / name) for name in code_names},
        "driver": sha(__file__),
    }


def verify_or_record_outputs(output, data, feature_path):
    receipt = output / "reproduction_outputs.json"
    if receipt.exists():
        hashes = json.loads(receipt.read_text(encoding="utf-8"))
        for name, expected in hashes.items():
            if sha(output / name) != expected:
                raise ValueError("A saved reproduction output changed: " + str(output / name))
        return
    # A run interrupted after predictions were saved can recover by checking the
    # predictions against every reloaded component before recording output hashes.
    from mirtards_release.cli import load_data
    from mirtards_release.predictor import recipe, transform
    from mirtards_release.training import predict_ensemble
    rows, indices, features = load_data(data, feature_path)
    scores = predict_ensemble(features[indices["test"]], output / "models", transform, recipe)
    with np.load(output / "predictions.npz", allow_pickle=False) as bank:
        np.testing.assert_array_equal(bank["row_id"], rows.row_id.to_numpy(str)[indices["test"]])
        np.testing.assert_allclose(bank["scores"], scores, rtol=0, atol=1e-12)
    files = [output / "predictions.npz", output / "training.json"]
    files.extend(path for path in sorted((output / "models").rglob("*")) if path.is_file())
    save_json(receipt, {path.relative_to(output).as_posix(): sha(path) for path in files})


def run_one(args, encoder, data_seed, training_seed):
    data = Path(args.data_root) / f"seed{data_seed}"
    feature_path = data / FEATURE_NAMES[encoder]
    if args.features:
        feature_path = Path(args.features.format(
            encoder=encoder, data_seed=data_seed, data_dir=data.as_posix()
        ))
    inputs = input_receipt(data, feature_path, encoder, training_seed)
    output = Path(args.output) / encoder / f"data{data_seed}" / f"seed{training_seed}"
    marker = output / "reproduction_inputs.json"
    output.mkdir(parents=True, exist_ok=True)
    if marker.exists():
        if json.loads(marker.read_text(encoding="utf-8")) != inputs:
            raise ValueError("Existing output has different frozen inputs or code: " + str(output))
    elif any(output.iterdir()):
        raise ValueError("Existing output has no frozen-input receipt: " + str(output))
    else:
        save_json(marker, inputs)
    if not (output / "predictions.npz").exists():
        subprocess.run([
            sys.executable, "-m", "mirtards_release", "train", "--data-dir", str(data),
            "--encoder", encoder, "--features", str(feature_path),
            "--seed", str(training_seed), "--output", str(output),
        ], check=True)
    verify_or_record_outputs(output, data, feature_path)
    subprocess.run([
        sys.executable, "-m", "mirtards_release", "evaluate", "--data-dir", str(data),
        "--predictions", str(output / "predictions.npz"),
        "--output", str(output / "metrics.json"),
    ], check=True)
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    return dict(encoder=encoder, data_seed=data_seed, training_seed=training_seed, **metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="benchmark_data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--encoder", choices=("openai-large", "biolord", "biolord-original", "all"), default="all")
    parser.add_argument("--data-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--training-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--features", help=(
        "Custom feature path, optionally containing {data_seed}, {encoder}, or {data_dir}. "
        "With --encoder all, the template must contain {encoder}."
    ))
    args = parser.parse_args()
    for values in (args.data_seeds, args.training_seeds):
        if len(values) != len(set(values)) or any(seed < 0 or seed >= 2**32 for seed in values):
            parser.error("Seeds must be distinct integers in [0, 2**32).")
    if args.encoder == "all" and args.features and "{encoder}" not in args.features:
        parser.error("A custom feature template for both encoders must contain {encoder}.")
    encoders = ("openai-large", "biolord") if args.encoder == "all" else (args.encoder,)
    records = [
        run_one(args, encoder, data_seed, training_seed)
        for encoder in encoders
        for data_seed in args.data_seeds
        for training_seed in args.training_seeds
    ]
    table = pd.DataFrame(records)
    output = Path(args.output)
    table.to_csv(output / "metrics.csv", index=False)
    columns = ["balanced_aps", "raw_aps", "auroc", "f1", "accuracy", "precision", "recall"]
    means = table.groupby(["encoder", "data_seed"])[columns].mean()
    table.groupby(["encoder", "data_seed"])[columns].agg(["mean", "std"]).to_csv(output / "summary.csv")
    means.groupby("encoder").mean().to_csv(output / "encoder_summary.csv")


if __name__ == "__main__":
    main()
