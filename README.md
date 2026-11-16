# miRTarDS

Disease association profiles for microRNA target prioritization.

miRTarDS ranks candidate miRNA–gene pairs using the diseases associated with each endpoint. Disease embeddings are summarized into 12 between-profile and within-profile similarity statistics, then scored by a supervised ensemble.

## Results

| Disease encoder | Balanced APS | ROC-AUC |
|---|---:|---:|
| Original BioLORD-2023 | 0.653604 | 0.657139 |
| Fine-tuned BioLORD | 0.656023 | 0.659103 |
| OpenAI text-embedding-3-large | 0.708838 | 0.704802 |

All conditions use complete disease profiles, the same predictor, data seeds 0–2 and training seeds 0–2. Metrics are averaged over three training seeds within each split, then equally across splits. Original BioLORD is an encoder control.

Comparison on the same 1,013 MeSH-covered test pairs:

| Method | Balanced APS |
|---|---:|
| MISIM | 0.565940 |
| Mapped Jaccard | 0.576438 |
| Raw-name Jaccard | 0.577095 |
| miRTarDS, OpenAI-large | **0.708673** |

miRTarDS and raw-name Jaccard use complete profiles; MISIM and mapped Jaccard use exact MeSH descriptor sets.

[Full results](results/README.md) include per-split variability, F1, precision, recall, accuracy and an experiment that restricts all methods to the same mapped diseases. These are retrospective database-observed versus unlabelled comparisons on overlapping resplits of one source cohort.

Earlier evaluations were affected by repeated disease-profile fingerprints across partitions. The revised benchmark groups identical profiles and miRNA precursor components and matches annotation coverage during sampling; remaining limitations are documented with the results.

## Reproduce

Use Python 3.10–3.12 from the repository root:

~~~bash
python -m pip install -e .
python scripts/reproduce_benchmark.py --output runs/reproduction
~~~

This runs the two primary encoders over all nine data-seed × training-seed combinations each. Frozen feature matrices are included, so no embedding API call is needed. The [reproduction guide](docs/reproducibility.md) covers individual runs, the original-BioLORD control and new predictions.

## Data and models

- [Benchmark data and preprocessing](benchmark_data/README.md): candidates, P/U labels, disease profiles, fixed partitions and feature matrices.
- [Method](docs/method.md): feature definitions, predictor and evaluation.
- [Fine-tuned BioLORD](models/biolord-mondo-dora/README.md): complete local weights, inference and [fine-tuning code](scripts/model_tools/README.md).

Code is [MIT-licensed](LICENSE). Model weights, ontology material and database annotations retain their [source-specific terms](THIRD_PARTY_NOTICES.md). Regenerating sampling requires the [original source inputs](benchmark_data/source/README.md).

[Original reviewed preprint](https://elifesciences.org/reviewed-preprints/109602v1).
