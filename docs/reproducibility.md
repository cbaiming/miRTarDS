# Reproduction and local inference

Use Python 3.10–3.12 and the pinned dependencies. Model fitting uses deterministic CPU execution with two threads.

```bash
python -m pip install -e .
python scripts/reproduce_benchmark.py --output runs/reproduction
```

The default command fits the two primary encoder conditions: three data seeds × three training seeds × two encoders = 18 models. Both seed lists are 0, 1, 2. Completed runs resume when input and model-file checks match. The supplied feature matrices allow reproduction without encoder inference or an API key.

## Select a condition

```bash
python scripts/reproduce_benchmark.py --encoder openai-large --output runs/openai
python scripts/reproduce_benchmark.py --encoder biolord --output runs/biolord
```

The [original BioLORD control](../results/original_biolord/README.md) uses nine separate fits and its own frozen feature matrices:

```bash
python scripts/reproduce_benchmark.py --encoder biolord-original --output runs/original-control
```

## Compute new disease features

The released fine-tuned BioLORD runs locally:

```bash
python -m pip install -e ".[biolord]"
mirtards features --data-dir benchmark_data/seed0 --encoder biolord --model-dir models/biolord-mondo-dora --output runs/biolord_seed0.npz
```

To encode with an original BioLORD checkpoint, pass its local directory through `--model-dir`. The [model guide](../scripts/model_tools/README.md) describes encoder training and inference.

For new OpenAI-large features, configure `OPENAI_API_KEY` in your environment:

```bash
mirtards features --data-dir benchmark_data/seed0 --encoder openai-large --output runs/large_seed0.npz
```

API requests incur provider charges. Provider revisions and hardware arithmetic can affect regenerated vectors, so use the shipped matrices to check the reported results.

## Score candidates

Supply an NPZ with string `row_id` and finite `(n, 12)` `Doff12` features, using the encoder associated with the fitted model:

```bash
mirtards predict --features runs/new_candidates.npz --model-dir runs/reproduction/openai-large/data0/seed0/models --output runs/new_scores.npz
```

Evaluate a released run with:

```bash
mirtards evaluate --data-dir benchmark_data/seed0 --predictions runs/reproduction/openai-large/data0/seed0/predictions.npz --output runs/metrics.json
```

Scores lie in [0,1]; classification metrics use threshold 0.5.
