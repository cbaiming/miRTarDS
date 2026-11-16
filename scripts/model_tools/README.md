# BioLORD training and inference

The [fine-tuned encoder](../../models/biolord-mondo-dora/README.md) includes full local weights, its selected adapter and tokenizer.

## Inference

```bash
python -m pip install -e ".[biolord]"
python scripts/model_tools/infer.py "diabetes mellitus" "breast carcinoma" --output disease_embeddings.npy
```

Add `--device cuda` for GPU inference. The Python interface `mirtards_release.biolord.encode_texts` returns normalized 768-dimensional vectors using masked mean pooling and a 64-token limit.

## Recalculate adaptation

```bash
python -m pip install -e ".[training]"
python scripts/model_tools/train_dora.py --base /path/to/BioLORD-2023 --data scripts/model_tools/training_data --output runs/biolord-dora --device cuda
```

The [MONDO-derived pools](training_data/README.md) contain ontology similarity targets and concept partitions. Target-interaction labels are excluded from encoder training.

[training_config.json](training_config.json) fixes seed 42, 600 FP32 updates, AdamW learning rate 1e-5, 60 warm-up steps and weight decay 0.01. DoRA uses rank 4 and alpha 8 in query/value projections of layers 8–11. Each update contains 77 natural, 38 structural and 13 synonym pairs, with microbatches of 64 and gradient clipping at 1.0.

The ranking loss uses scale 20 and label gap 0.05. Embedding-preservation and geometry-preservation penalties each have weight 10. Development checks occur at steps 0, 25, 50, 100, 200, 400 and 600. Select the highest mean development Spearman improvement above 1e-6 among checkpoints with mean displacement ≤0.01 and cosine-RMSE ≤0.03. Retraining can vary across numerical environments; the supplied checkpoint defines the reported results.

## Reconstruct merged weights

```bash
python scripts/model_tools/merge_adapter.py --base /path/to/BioLORD-2023 --adapter models/biolord-mondo-dora/adapter_step600.pt --output models/reconstructed-biolord
```

This merges the retained adapter into the hash-matching base and writes shards below 100 MB. The complete merged model is already included.

MONDO inputs retain CC BY 4.0 attribution. BioLORD weights retain their upstream UMLS/SNOMED CT conditions. See the [model card](../../models/biolord-mondo-dora/README.md).
