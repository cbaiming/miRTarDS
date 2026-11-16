"""Command-line entry points for the released benchmark."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .predictor import recipe, transform


FEATURE_NAMES = {
    "openai-large": "features_openai_large.npz",
    "biolord": "features_biolord.npz",
    "biolord-original": "features_biolord_original.npz",
}


def load_json(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as handle:
        return json.load(handle)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_data(directory, feature_path=None, encoder="openai-large"):
    directory = Path(directory)
    rows = pd.read_csv(directory / "rows.tsv.gz", sep="\t", dtype={"row_id": str, "a_id": str, "b_id": str})
    if not rows.row_id.is_unique:
        raise ValueError("Duplicate candidate row identifiers")
    with np.load(directory / "indices.npz", allow_pickle=False) as saved:
        indices = dict(saved)
    feature_name = FEATURE_NAMES[encoder]
    path = Path(feature_path) if feature_path else directory / feature_name
    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["row_id"], rows.row_id.to_numpy(str))
        features = saved["Doff12"]
    if features.shape != (len(rows), 12):
        raise ValueError("Expected exactly twelve disease features")
    for name, index in indices.items():
        if index.dtype.kind not in "iu" or len(np.unique(index)) != len(index) or not len(index):
            raise ValueError("Invalid partition indices: " + name)
        if index.min() < 0 or index.max() >= len(rows):
            raise ValueError("Partition indices out of range: " + name)
    pairs = [("train", "test"), ("0_fit", "0_tune"), ("1_fit", "1_tune")]
    for left, right in pairs:
        if np.intersect1d(indices[left], indices[right]).size:
            raise ValueError("Overlapping row partitions")
        for column in ("a_id", "b_id", "a_group", "b_group"):
            if set(rows.iloc[indices[left]][column]) & set(rows.iloc[indices[right]][column]):
                raise ValueError("Overlapping endpoint groups: " + column)
    if not np.isfinite(features[np.r_[indices["train"], indices["test"]]]).all():
        raise ValueError("Missing train/test features")
    return rows, indices, features


def train_command(args):
    from .training import train_ensemble
    rows, indices, features = load_data(args.data_dir, args.features, args.encoder)
    batch = {}
    for role, name in (("fit", "0_fit"), ("tune", "0_tune"), ("fit2", "1_fit"),
                       ("tune2", "1_tune"), ("train", "train"), ("eval", "test")):
        batch["X_" + role] = features[indices[name]]
        if role != "eval":
            batch["y_" + role] = rows.label.to_numpy(int)[indices[name]]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "predictions.npz").exists():
        raise FileExistsError("Choose a new output directory")
    scores, info = train_ensemble(batch, args.seed, output / "models", transform, recipe)
    np.savez_compressed(output / "predictions.npz",
                        row_id=rows.row_id.to_numpy(str)[indices["test"]], scores=scores)
    info["encoder"] = args.encoder
    feature_path = Path(args.features) if args.features else Path(args.data_dir) / FEATURE_NAMES[args.encoder]
    info["feature_source"] = str(feature_path.resolve())
    save_json(output / "training.json", info)
    print(output / "predictions.npz")


def evaluate_command(args):
    from .metrics import evaluate_scores
    rows = pd.read_csv(Path(args.data_dir) / "rows.tsv.gz", sep="\t", dtype={"row_id": str}).set_index("row_id")
    with np.load(args.predictions, allow_pickle=False) as saved:
        ids, scores = saved["row_id"], saved["scores"]
    if len(set(ids)) != len(ids):
        raise ValueError("Repeated predicted rows")
    selected = rows.loc[ids]
    if not selected.split.eq("test").all():
        raise ValueError("Evaluation expects outer-test predictions")
    with np.load(Path(args.data_dir) / "indices.npz", allow_pickle=False) as saved:
        expected = pd.read_csv(Path(args.data_dir) / "rows.tsv.gz", sep="\t").row_id.to_numpy()[saved["test"]]
    np.testing.assert_array_equal(ids, expected)
    result = evaluate_scores(selected.label.to_numpy(int), scores, threshold=args.threshold)
    save_json(args.output, result)
    print(json.dumps(result, indent=2))


def feature_command(args):
    from .features import encode_openai_large, offdiag12, profile_texts
    folder = Path(args.data_dir)
    rows = pd.read_csv(folder / "rows.tsv.gz", sep="\t", dtype={"row_id": str, "a_id": str, "b_id": str})
    with np.load(folder / "indices.npz", allow_pickle=False) as saved:
        used = np.sort(np.r_[saved["train"], saved["test"]])
    profiles = load_json(folder / "profiles.json.gz")
    selected = rows.iloc[used]
    texts = sorted({text for side in ("a", "b") for entity in selected[side + "_id"].unique()
                    for text in profile_texts(profiles[side][entity])})
    if args.encoder == "biolord":
        from .biolord import encode_texts
        vectors = encode_texts(texts, args.model_dir, device=args.device, batch_size=args.batch_size)
    else:
        vectors = encode_openai_large(texts, batch_size=args.batch_size)
    features = np.full((len(rows), 12), np.nan)
    features[used] = offdiag12(selected, profiles, texts, vectors)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, row_id=rows.row_id.to_numpy(str), Doff12=features)
    save_json(path.with_suffix(".json"), dict(encoder=args.encoder, rows=len(used),
              dimension=int(vectors.shape[1]), names=len(texts), device=args.device))
    print(path)


def predict_command(args):
    from .training import predict_ensemble
    with np.load(args.features, allow_pickle=False) as saved:
        ids, features = saved["row_id"], saved["Doff12"]
    usable = np.isfinite(features).all(axis=1)
    scores = predict_ensemble(features[usable], args.model_dir, transform, recipe)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, row_id=ids[usable], scores=scores)
    print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Fit the frozen miRTarDS recipe")
    train.add_argument("--data-dir", required=True)
    train.add_argument("--features")
    train.add_argument("--encoder", choices=("openai-large", "biolord", "biolord-original"), default="openai-large")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--output", required=True)
    train.set_defaults(function=train_command)
    evaluate = commands.add_parser("evaluate", help="Evaluate frozen outer-test scores")
    evaluate.add_argument("--data-dir", required=True)
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--threshold", type=float, default=0.5)
    evaluate.add_argument("--output", required=True)
    evaluate.set_defaults(function=evaluate_command)
    features = commands.add_parser("features", help="Compute the original off-diagonal twelve statistics")
    features.add_argument("--data-dir", required=True)
    features.add_argument("--encoder", choices=("biolord", "openai-large"), required=True)
    features.add_argument("--model-dir", default="models/biolord-mondo-dora")
    features.add_argument("--device", default="cpu")
    features.add_argument("--batch-size", type=int, default=64)
    features.add_argument("--output", required=True)
    features.set_defaults(function=feature_command)
    predict = commands.add_parser("predict", help="Score a row-aligned Doff12 npz using a saved ensemble")
    predict.add_argument("--features", required=True)
    predict.add_argument("--model-dir", required=True)
    predict.add_argument("--output", required=True)
    predict.set_defaults(function=predict_command)
    args = parser.parse_args()
    args.function(args)

