"""Local inference for the released BioLORD MONDO DoRA encoder."""
from pathlib import Path

import numpy as np


def encode_texts(texts, model_dir, device="cpu", batch_size=64):
    """Return float64, L2-normalized 768-dimensional disease embeddings.

    The frozen encoder uses attention-mask mean pooling in FP32 and a 64-token
    input limit. A local sharded Transformers or SentenceTransformer folder is
    accepted. No remote model lookup is performed.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    texts = list(texts)
    if not texts:
        return np.empty((0, 768), dtype=np.float64)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Each disease name must be a nonempty string")
    root = Path(model_dir).resolve()
    transformer_dir = root / "0_Transformer" if (root / "0_Transformer").is_dir() else root
    tokenizer = AutoTokenizer.from_pretrained(transformer_dir, local_files_only=True)
    model = AutoModel.from_pretrained(
        transformer_dir, local_files_only=True, torch_dtype=torch.float32
    ).to(device).eval()
    if model.config.hidden_size != 768:
        raise ValueError("The released BioLORD encoder has 768 output dimensions")
    model.config.output_hidden_states = False
    model.config.output_attentions = False
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            tokens = tokenizer(
                texts[start:start + batch_size], padding=True, truncation=True,
                max_length=64, return_tensors="pt",
            ).to(device)
            hidden = model(**tokens).last_hidden_state.float()
            mask = tokens["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)
            vectors.append(pooled.cpu().numpy())
    result = np.concatenate(vectors).astype(np.float64)
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    if not np.isfinite(result).all() or (norms <= 0).any():
        raise ValueError("The encoder returned invalid embeddings")
    return result / norms
