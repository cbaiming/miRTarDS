"""Merge the frozen DoRA adapter into the verified BioLORD base and shard FP32 weights."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


BASE_SHA256 = "f8c969586cc6b0fd393faa7d879de93d6cc532123956041fefd3474194322050"
ADAPTER_SHA256 = "a6745ed9ea3a710b55d0981a3aaca369d7f56cf51ec4c6ae2a90900794bcb586"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    base = args.base / "0_Transformer" if (args.base / "0_Transformer").is_dir() else args.base
    if sha(base / "model.safetensors") != BASE_SHA256:
        raise ValueError("The base checkpoint differs from the frozen BioLORD weights")
    if sha(args.adapter) != ADAPTER_SHA256:
        raise ValueError("The adapter differs from the selected step-600 checkpoint")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = AutoModel.from_pretrained(
        base, local_files_only=True, torch_dtype=torch.float32
    ).to(args.device).eval()
    state = torch.load(args.adapter, map_location=args.device, weights_only=True)
    expected = set()
    with torch.no_grad():
        for index in range(8, 12):
            attention = model.encoder.layer[index].attention.attn
            for name in ("q", "v"):
                prefix = f"transformer.encoder.layer.{index}.attention.attn.{name}."
                expected.update(prefix + suffix for suffix in ("A", "B", "magnitude"))
                layer = getattr(attention, name)
                combined = layer.weight + 2.0 * (state[prefix + "B"] @ state[prefix + "A"])
                ratio = state[prefix + "magnitude"] / torch.linalg.vector_norm(
                    combined, dim=1
                ).clamp_min(1e-12)
                layer.weight.copy_(combined * ratio[:, None])
    if set(state) != expected:
        raise ValueError("Unexpected adapter keys")
    destination = args.output / "0_Transformer"
    destination.mkdir(parents=True)
    model.cpu().save_pretrained(
        destination, safe_serialization=True, max_shard_size="95MB"
    )
    AutoTokenizer.from_pretrained(base, local_files_only=True).save_pretrained(destination)
    (destination / "sentence_bert_config.json").write_text(
        json.dumps({"max_seq_length": 64, "do_lower_case": False}), encoding="utf-8"
    )
    pooling = args.output / "1_Pooling"
    pooling.mkdir()
    (pooling / "config.json").write_text(json.dumps({
        "word_embedding_dimension": 768,
        "pooling_mode_cls_token": False,
        "pooling_mode_mean_tokens": True,
        "pooling_mode_max_tokens": False,
        "pooling_mode_mean_sqrt_len_tokens": False,
    }), encoding="utf-8")
    (args.output / "modules.json").write_text(json.dumps([
        {"idx": 0, "name": "0", "path": "0_Transformer", "type": "sentence_transformers.models.Transformer"},
        {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
    ]), encoding="utf-8")
    (args.output / "config_sentence_transformers.json").write_text(json.dumps({
        "prompts": {}, "default_prompt_name": None, "similarity_fn_name": "cosine"
    }), encoding="utf-8")
    if any(file.stat().st_size >= 100_000_000 for file in args.output.rglob("*") if file.is_file()):
        raise RuntimeError("A model artifact exceeds the repository file-size limit")
    print(f"Merged model saved to {args.output}")


if __name__ == "__main__":
    main()
