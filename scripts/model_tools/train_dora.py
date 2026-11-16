"""Recalculate the ontology-only guarded DoRA adaptation with retained MONDO pools."""
import argparse
import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer


class DoRALinear(torch.nn.Module):
    def __init__(self, base, rank=4, alpha=8):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.scale = alpha / rank
        self.A = torch.nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.B = torch.nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.magnitude = torch.nn.Parameter(torch.linalg.vector_norm(base.weight.detach(), dim=1).clone())

    def forward(self, x):
        combined = self.base.weight + self.scale * (self.B @ self.A)
        ratio = self.magnitude / torch.linalg.vector_norm(
            combined, dim=1
        ).detach().clamp_min(1e-12)
        base = self.base(x)
        bias = self.base.bias if self.base.bias is not None else 0.0
        delta = self.scale * F.linear(F.linear(x, self.A), self.B)
        return base + (ratio - 1) * (base - bias) + ratio * delta


class Encoder(torch.nn.Module):
    def __init__(self, path, device):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(
            path, local_files_only=True, torch_dtype=torch.float32
        ).to(device).eval()
        self.transformer.config.output_hidden_states = False
        self.transformer.config.output_attentions = False

    def forward(self, tokens):
        hidden = self.transformer(**tokens).last_hidden_state.float()
        mask = tokens["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        return F.normalize(pooled, p=2, dim=1)


def encode(model, tokenizer, texts, device, gradients=False):
    output = []
    with torch.set_grad_enabled(gradients):
        for start in range(0, len(texts), 64):
            tokens = tokenizer(
                texts[start:start + 64], padding=True, truncation=True,
                max_length=64, return_tensors="pt"
            ).to(device)
            output.append(model(tokens))
    return torch.cat(output)


def objective(z, reference, left, right, labels, config):
    scores = (z[left] * z[right]).sum(-1)
    valid = labels[:, None] - labels[None, :] >= config["label_gap"] - 1e-7
    differences = config["scale"] * (scores[None, :] - scores[:, None])
    ranking = torch.logsumexp(torch.cat([scores.new_zeros(1), differences[valid]]), dim=0)
    anchor = (1 - (z * reference).sum(-1)).mean().clamp_min(0)
    upper = torch.triu(torch.ones((len(z), len(z)), device=z.device, dtype=torch.bool), diagonal=1)
    geometry = ((z @ z.T - reference @ reference.T)[upper] ** 2).mean()
    return ranking + config["anchor_weight"] * anchor + config["geometry_weight"] * geometry


def read_pairs(path):
    frame = pd.read_csv(path, sep="\t", keep_default_na=False, float_precision="round_trip", dtype={
        "concept_id_1": str, "concept_id_2": str, "text_1": str, "text_2": str
    })
    required = {"text_1", "text_2", "concept_id_1", "concept_id_2", "ylin"}
    if not required <= set(frame):
        raise ValueError(f"Pair schema is incomplete in {path.name}")
    labels = frame.ylin.to_numpy(float)
    if not np.isfinite(labels).all() or not ((labels >= 0) & (labels <= 1)).all():
        raise ValueError(f"Invalid ontology labels in {path.name}")
    return frame


def evaluate(model, tokenizer, dev, texts, reference, device):
    vectors = encode(model, tokenizer, texts, device).cpu().double().numpy()
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    positions = {text: i for i, text in enumerate(texts)}
    results = {}
    for name, frame in dev.items():
        a = np.array([positions[t] for t in frame.text_1])
        b = np.array([positions[t] for t in frame.text_2])
        scores = np.sum(vectors[a] * vectors[b], axis=1).clip(-1, 1)
        original = np.sum(reference[a] * reference[b], axis=1).clip(-1, 1)
        results[name] = {
            "spearman": float(spearmanr(scores, frame.ylin.to_numpy(float)).statistic),
            "cosine_rmse": float(np.sqrt(np.mean((scores - original) ** 2))),
            "rows": len(frame),
        }
    displacement = float(np.maximum(0, 1 - np.sum(vectors * reference, axis=1)).mean())
    return {
        "mean_spearman": float(np.mean([v["spearman"] for v in results.values()])),
        "displacement": displacement,
        "mean_cosine_rmse": float(np.mean([v["cosine_rmse"] for v in results.values()])),
        "strata": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads(Path(__file__).with_name("training_config.json").read_text(encoding="utf-8-sig"))
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    base = args.base / "0_Transformer" if (args.base / "0_Transformer").is_dir() else args.base
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    model = Encoder(base, args.device).requires_grad_(False).eval()
    for index in config["target_layers"]:
        attention = model.transformer.encoder.layer[index].attention.attn
        for name in config["targets"]:
            setattr(attention, name, DoRALinear(getattr(attention, name), config["rank"], config["alpha"]))
    teacher = Encoder(base, args.device).requires_grad_(False).eval()
    parameters = [p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in parameters) != 55296:
        raise ValueError("Unexpected trainable adapter size")
    split = pd.read_csv(args.data / "concept_split.tsv", sep="\t", dtype=str, keep_default_na=False)
    roots = {"MONDO:0000001"}
    if "is_root" in split:
        roots.update(split.loc[split.is_root.str.lower().isin(["1", "true"]), "mondo_id"])
    concepts = set(split.loc[split.split == "train", "mondo_id"]) - roots
    if len(concepts) != 17422:
        raise ValueError("The frozen training split must contain 17,422 non-root concepts")
    train_ids = set(split.loc[split.split == "train", "mondo_id"])
    pools = []
    for name in ("natural", "structural", "synonyms"):
        frame = read_pairs(args.data / f"train_{name}.tsv.gz")
        if not (set(frame.concept_id_1) | set(frame.concept_id_2)) <= train_ids:
            raise ValueError("Training pair endpoints must belong to the frozen training split")
        tokens = {}
        names = sorted(set(frame.text_1) | set(frame.text_2))
        for start in range(0, len(names), 512):
            batch = names[start:start + 512]
            encoded = tokenizer(batch, truncation=False, add_special_tokens=True)["input_ids"]
            tokens.update({text: tuple(ids) for text, ids in zip(batch, encoded)})
        keep = [
            a in concepts and b in concepts and tokens[x] != tokens[y]
            for a, b, x, y in frame[["concept_id_1", "concept_id_2", "text_1", "text_2"]].itertuples(index=False, name=None)
        ]
        frame = frame.loc[keep].reset_index(drop=True)
        if len(frame) < 128:
            raise ValueError(f"Insufficient eligible {name} pairs")
        pools.append(frame)
    dev = {name: read_pairs(args.data / f"dev_{name}.tsv.gz")
           for name in ("seen_seen", "seen_unseen", "unseen_unseen")}
    dev_ids = set(split.loc[split.split == "dev", "mondo_id"])
    for name, frame in dev.items():
        left_train = frame.concept_id_1.isin(train_ids).to_numpy()
        right_train = frame.concept_id_2.isin(train_ids).to_numpy()
        left_dev = frame.concept_id_1.isin(dev_ids).to_numpy()
        right_dev = frame.concept_id_2.isin(dev_ids).to_numpy()
        allowed = {"seen_seen": left_train & right_train, "seen_unseen": (left_train & right_dev) | (left_dev & right_train), "unseen_unseen": left_dev & right_dev}[name]
        if not allowed.all():
            raise ValueError(f"Development endpoints violate the {name} split")
    dev_texts = sorted({text for frame in dev.values() for values in (frame.text_1, frame.text_2) for text in values})
    reference = encode(teacher, tokenizer, dev_texts, args.device).cpu().double().numpy()
    reference /= np.linalg.norm(reference, axis=1, keepdims=True)
    args.output.mkdir(parents=True)
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=config["weight_decay"])
    best = None
    sample_hash = hashlib.sha256()
    for step in range(config["steps"] + 1):
        if step in config["checks"]:
            result = evaluate(model, tokenizer, dev, dev_texts, reference, args.device)
            result["step"] = step
            result["eligible"] = (
                result["displacement"] <= config["max_dev_displacement"]
                and result["mean_cosine_rmse"] <= config["max_dev_cosine_rmse"]
            )
            state = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
            torch.save(state, args.output / f"adapter_step{step}.pt")
            (args.output / f"dev_step{step}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            if result["eligible"] and (best is None or result["mean_spearman"] > best["mean_spearman"] + 1e-6):
                best = result
            print(json.dumps(result), flush=True)
        if step == config["steps"]:
            break
        factor = min((step + 1) / config["warmup"], (config["steps"] - step) / (config["steps"] - config["warmup"]))
        for group in optimizer.param_groups:
            group["lr"] = config["learning_rate"] * factor
        rng = np.random.default_rng(np.random.SeedSequence([42, step, 20261014]))
        left_texts, right_texts, labels = [], [], []
        for pool, count in zip(pools, config["counts"]):
            chosen = pool.iloc[rng.choice(len(pool), size=count, replace=False)]
            left_texts.extend(chosen.text_1)
            right_texts.extend(chosen.text_2)
            labels.extend(float(v) for v in chosen.ylin)
        digest = hashlib.sha256(json.dumps([left_texts, right_texts, labels], ensure_ascii=False).encode()).hexdigest()
        sample_hash.update(digest.encode())
        texts = sorted(set(left_texts + right_texts))
        positions = {text: i for i, text in enumerate(texts)}
        left = torch.tensor([positions[t] for t in left_texts], device=args.device)
        right = torch.tensor([positions[t] for t in right_texts], device=args.device)
        targets = torch.tensor(labels, device=args.device)
        reference_packet = encode(teacher, tokenizer, texts, args.device)
        optimizer.zero_grad(set_to_none=True)
        vectors = encode(model, tokenizer, texts, args.device, gradients=True)
        loss = objective(vectors, reference_packet, left, right, targets, config)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training objective")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
    if best is None:
        raise RuntimeError("No checkpoint met the frozen development guards")
    (args.output / "selection.json").write_text(json.dumps({
        "selected": best, "config": config, "sample_sha256": sample_hash.hexdigest(),
        "miRNA_target_labels_used": False
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

