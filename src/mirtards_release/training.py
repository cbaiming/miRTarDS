"""Deterministic CPU fitting and local prediction for the frozen ensemble.

This module preserves the published experiment's training arithmetic, RNG
conventions, train-only scaling, epoch selection, and component serialization.
It accepts the release's explicit transform and recipe functions.
"""

import copy
import json
import os
from pathlib import Path
import random
import time

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"

import joblib
import numpy as np
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
import torch

from .metrics import ranking_metrics

torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)


def balanced_weights(y):
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y, minlength=2)
    if counts.min() <= 0:
        raise ValueError("Both P and U classes are required")
    return len(y) / (2.0 * counts[y])


def fit_scaler(x):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    frozen = std < 1e-6
    std[frozen] = 1
    return mean, std, frozen


def scale(x, scaler):
    mean, std, frozen = scaler
    result = (x - mean) / std
    result[:, frozen] = 0
    return result.astype(np.float32)


def batches(y, size, rng):
    count = int(np.ceil(len(y) / size))
    parts = [
        np.array_split(rng.permutation(np.flatnonzero(y == label)), count)
        for label in (0, 1)
    ]
    for index in rng.permutation(count):
        yield rng.permutation(np.concatenate([parts[0][index], parts[1][index]]))


class Network(torch.nn.Module):
    def __init__(self, width, params):
        super().__init__()
        dimensions = params["hidden_dims"]
        activation = {
            "relu": torch.nn.ReLU,
            "gelu": torch.nn.GELU,
            "silu": torch.nn.SiLU,
            "tanh": torch.nn.Tanh,
        }[params["activation"]]
        initial_seed = torch.initial_seed()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(initial_seed + 173)
            first = torch.nn.Linear(width, dimensions[0])
        layers = [first, activation(), torch.nn.Dropout(params["dropout"])]
        for left, right in zip(dimensions[:-1], dimensions[1:]):
            layers.extend([
                torch.nn.Linear(left, right),
                activation(),
                torch.nn.Dropout(params["dropout"]),
            ])
        layers.append(torch.nn.Linear(dimensions[-1], 1))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x).squeeze(1)


def neural_predict(model, x):
    unique, inverse = np.unique(x, axis=0, return_inverse=True)
    model.eval()
    with torch.inference_mode():
        chunks = np.array_split(unique, max(1, int(np.ceil(len(unique) / 8192))))
        scores = np.concatenate([model(torch.from_numpy(chunk)).numpy() for chunk in chunks])
    return scores[inverse].astype(float)


def train_network(x, y, params, seed, tune=None, epochs=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    scaler = fit_scaler(x)
    fit_x = scale(x, scaler)
    tune_x = scale(tune[0], scaler) if tune is not None else None
    model = Network(x.shape[1], params)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
    )
    tx = torch.from_numpy(fit_x)
    ty = torch.tensor(y, dtype=torch.float32)
    weights = torch.tensor(balanced_weights(y), dtype=torch.float32)
    rng = np.random.default_rng(seed)
    best = None
    best_epoch = 0
    state = None
    history = []
    last_epoch = params["max_epochs"] if epochs is None else epochs
    for epoch in range(last_epoch + 1):
        if epoch:
            model.train()
            denominator = len(y) / int(np.ceil(len(y) / params["batch_size"]))
            for batch in batches(y, params["batch_size"], rng):
                optimizer.zero_grad(set_to_none=True)
                element_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    model(tx[batch]), ty[batch], reduction="none"
                )
                loss = (element_loss * weights[batch]).sum() / denominator
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 5.0, error_if_nonfinite=True
                )
                optimizer.step()
        if tune is None:
            best_epoch = epoch
        else:
            scores = neural_predict(model, tune_x)
            metrics = ranking_metrics(tune[1], scores)
            bce = float(np.mean(
                balanced_weights(tune[1]) * (np.logaddexp(0, scores) - tune[1] * scores)
            ))
            key = (metrics["balanced_aps"], -bce, -epoch)
            history.append({
                "epoch": epoch,
                "balanced_aps": metrics["balanced_aps"],
                "balanced_bce": bce,
            })
            if best is None or key > best:
                best = key
                best_epoch = epoch
                state = copy.deepcopy(model.state_dict())
            if epoch - best_epoch >= params["patience"]:
                break
    if state is not None:
        model.load_state_dict(state)
    return model, scaler, {"selected_epoch": best_epoch, "tune_history": history}


def _transform_features(transform, x):
    x = np.asarray(x, dtype=float)
    result = np.asarray(transform(x.copy()), dtype=np.float64)
    if result.ndim != 2 or result.shape[0] != len(x):
        raise ValueError("The transform must return an aligned two-dimensional array")
    if not 1 <= result.shape[1] <= 96 or not np.isfinite(result).all():
        raise ValueError("Transformed features must be finite with 1 to 96 columns")
    if len(x) > 1:
        chosen = np.unique(np.linspace(0, len(x) - 1, min(13, len(x)), dtype=int))
        single = np.concatenate([
            np.asarray(transform(x[index:index + 1].copy()), dtype=float) for index in chosen
        ])
        np.testing.assert_allclose(result[chosen], single, rtol=1e-9, atol=1e-10)
    if result.size and np.max(np.abs(result)) > 1e10:
        raise ValueError("Unstable transformed feature magnitude")
    return result


def _infer_classifier(model, evaluation):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(evaluation)[:, 1]
    return expit(model.decision_function(evaluation))


def component_predictions(component, batch, seed, artifact):
    kind = component["kind"]
    params = dict(component.get("params", {}))
    data = dict(batch)
    if component.get("columns") is not None:
        for key in data:
            if key.startswith("X_"):
                data[key] = data[key][:, component["columns"]]
    if kind == "mlp":
        params = {
            "hidden_dims": [32, 16], "dropout": 0.1, "lr": 0.001,
            "weight_decay": 0.0001, "batch_size": 256, "max_epochs": 80,
            "patience": 12, "pad_to": 0, "activation": "relu", **params,
        }
        width = max(data["X_train"].shape[1], params["pad_to"])
        for key in data:
            if key.startswith("X_") and data[key].shape[1] < width:
                data[key] = np.pad(data[key], ((0, 0), (0, width - data[key].shape[1])))
        selected = []
        histories = []
        suffixes = ("", "2") if "X_fit2" in data else ("",)
        for suffix in suffixes:
            _, _, selection = train_network(
                data["X_fit" + suffix], data["y_fit" + suffix], params, seed,
                tune=(data["X_tune" + suffix], data["y_tune" + suffix]),
            )
            selected.append(selection["selected_epoch"])
            histories.append(selection)
        epochs = int(np.ceil(np.mean(selected)))
        model, scaler, _ = train_network(
            data["X_train"], data["y_train"], params, seed, epochs=epochs
        )
        predictions = expit(neural_predict(model, scale(data["X_eval"], scaler)))
        info = {
            "kind": kind, "selected_epochs": selected, "refit_epoch": epochs,
            "config": params, "selection": histories,
        }
        torch.save({
            "state_dict": model.state_dict(), "width": width, "params": params,
        }, artifact.with_suffix(".pt"))
        np.savez(
            artifact.with_suffix(".scaler.npz"),
            mean=scaler[0], std=scaler[1], frozen=scaler[2],
        )
        reloaded = Network(width, params)
        reloaded.load_state_dict(torch.load(artifact.with_suffix(".pt"), weights_only=True)["state_dict"])
        np.testing.assert_array_equal(
            predictions, expit(neural_predict(reloaded, scale(data["X_eval"], scaler)))
        )
    else:
        scaler = StandardScaler().fit(data["X_train"])
        fit_x = scaler.transform(data["X_train"])
        evaluation = scaler.transform(data["X_eval"])
        if kind == "histgb":
            model = HistGradientBoostingClassifier(
                **{
                    "learning_rate": 0.05, "max_iter": 180, "max_leaf_nodes": 15,
                    "min_samples_leaf": 40, "l2_regularization": 1.0, **params,
                },
                early_stopping=False, random_state=seed,
            )
        elif kind == "svm":
            model = SVC(
                **{"C": 1.0, "gamma": "scale", "kernel": "rbf", **params},
                probability=False, cache_size=512, random_state=seed,
            )
        else:
            raise ValueError("Unsupported component kind: " + kind)
        model.fit(fit_x, data["y_train"], sample_weight=balanced_weights(data["y_train"]))
        predictions = _infer_classifier(model, evaluation)
        info = {"kind": kind, "config": params}
        joblib.dump({"model": model, "scaler": scaler}, artifact.with_suffix(".joblib"), compress=3)
        reloaded = joblib.load(artifact.with_suffix(".joblib"))
        np.testing.assert_array_equal(
            predictions, _infer_classifier(reloaded["model"], evaluation)
        )
    if predictions.shape != (len(data["X_eval"]),) or not np.isfinite(predictions).all():
        raise ValueError("Invalid component predictions")
    return predictions, info


def train_ensemble(batch, seed, artifact_dir, transform, recipe):
    """Fit the explicit recipe and persist every component plus ensemble.json.

    The batch contains X_fit/y_fit, X_tune/y_tune, X_train/y_train and X_eval.
    X_fit2/y_fit2 and X_tune2/y_tune2 provide the second internal selection fold.
    The evaluation outcomes are excluded from this interface.
    """
    if any(key in batch for key in ('y_eval', 'y_test', 'y_assess')):
        raise ValueError('Evaluation outcomes must be kept outside the training batch')
    start = time.monotonic()
    config = recipe() if callable(recipe) else recipe
    destination = Path(artifact_dir)
    destination.mkdir(parents=True, exist_ok=True)
    incomplete = destination / ".incomplete"
    incomplete.write_text("Training has not completed.\n", encoding="utf-8")
    data = {
        key: _transform_features(transform, value) if key.startswith("X_")
        else np.asarray(value, dtype=int)
        for key, value in batch.items()
    }
    widths = {value.shape[1] for key, value in data.items() if key.startswith("X_")}
    if len(widths) != 1:
        raise ValueError("All feature matrices must have the same transformed width")
    probabilities = []
    details = []
    weights = []
    with threadpool_limits(2):
        for index, component in enumerate(config["components"]):
            probability, info = component_predictions(
                component, data, seed + 113 * index, destination / f"component{index}"
            )
            probabilities.append(probability)
            details.append(info)
            weights.append(component.get("weight", 1.0))
    scores = np.average(probabilities, axis=0, weights=weights)
    metadata = {
        "seed": seed, "recipe": config, "feature_count": data["X_train"].shape[1],
        "components": details, "seconds": time.monotonic() - start,
        "eval_labels_received": False,
    }
    temporary_metadata = destination / "ensemble.json.tmp"
    temporary_metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_metadata.replace(destination / "ensemble.json")
    incomplete.unlink()
    return scores, metadata


def predict_ensemble(X12, artifact_dir, transform, recipe):
    """Load the saved component checkpoints and return the weighted scores."""
    destination = Path(artifact_dir)
    if (destination / ".incomplete").exists():
        raise RuntimeError("Training is incomplete in the requested model directory")
    config = recipe() if callable(recipe) else recipe
    saved_metadata = destination / "ensemble.json"
    if saved_metadata.is_file():
        metadata = json.loads(saved_metadata.read_text(encoding="utf-8"))
        if metadata["recipe"] != config:
            raise ValueError("The supplied recipe differs from the saved training recipe")
    transformed = _transform_features(transform, X12)
    if not len(transformed):
        return np.empty(0, dtype=float)
    probabilities = []
    weights = []
    with threadpool_limits(2):
        for index, component in enumerate(config["components"]):
            evaluation = transformed
            if component.get("columns") is not None:
                evaluation = evaluation[:, component["columns"]]
            artifact = destination / f"component{index}"
            if component["kind"] == "mlp":
                checkpoint = torch.load(artifact.with_suffix(".pt"), map_location="cpu", weights_only=True)
                width = checkpoint["width"]
                if evaluation.shape[1] > width:
                    raise ValueError("Input feature width exceeds the saved MLP width")
                if evaluation.shape[1] < width:
                    evaluation = np.pad(evaluation, ((0, 0), (0, width - evaluation.shape[1])))
                with np.load(artifact.with_suffix(".scaler.npz"), allow_pickle=False) as bank:
                    scaler = (bank["mean"], bank["std"], bank["frozen"])
                model = Network(width, checkpoint["params"])
                model.load_state_dict(checkpoint["state_dict"])
                probability = expit(neural_predict(model, scale(evaluation, scaler)))
            elif component["kind"] in ("histgb", "svm"):
                saved = joblib.load(artifact.with_suffix(".joblib"))
                evaluation = saved["scaler"].transform(evaluation)
                probability = _infer_classifier(saved["model"], evaluation)
            else:
                raise ValueError("Unsupported component kind: " + component["kind"])
            probabilities.append(probability)
            weights.append(component.get("weight", 1.0))
    scores = np.average(probabilities, axis=0, weights=weights)
    if scores.shape != (len(transformed),) or not np.isfinite(scores).all():
        raise ValueError("Invalid ensemble predictions")
    return scores


