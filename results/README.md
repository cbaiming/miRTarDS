# Results

miRTarDS uses complete disease profiles with fixed data and training seeds 0, 1, and 2. Each model is evaluated separately, then averaged across training seeds and equally across data splits.

| Disease encoder | Mean balanced APS | Mean ROC-AUC |
|---|---:|---:|
| Original BioLORD | 0.653604 | 0.657139 |
| Fine-tuned BioLORD | 0.656023 | 0.659103 |
| OpenAI-large | 0.708838 | 0.704802 |

The [original BioLORD control](original_biolord/README.md) uses the same predictor and full test pairs. Fine-tuned minus original BioLORD differs by +0.002419 balanced APS, with paired 95% interval [−0.002861, 0.007216].

The [primary report](primary/README.md) includes split-level results, F1, accuracy, recall, precision, paired intervals and sequence-overlap sensitivity. P denotes database-observed interactions; U denotes unlabelled pairs. Three splits of the same source pool share some pairs and endpoints.

## Traditional similarities

| Method on the same MeSH-covered pairs | Balanced APS | Disease information |
|---|---:|---|
| Fine-tuned BioLORD miRTarDS | 0.655517 | Complete profiles |
| OpenAI-large miRTarDS | 0.708673 | Complete profiles |
| MISIM | 0.565940 | Exact MeSH descriptors |
| Jaccard | 0.577095 | Complete name sets |
| Mapped Jaccard | 0.576438 | Exact MeSH descriptors |

Shared pairs alone leave different disease-information scopes. In the [matched-information comparison](secondary/matched_information/README.md), all methods use identical descriptor sets; OpenAI-large miRTarDS reaches 0.643622. See [similarity definitions](baselines/README.md) and [mapping sensitivity](baselines/diagnostics/README.md).

The data0–2 release subset was selected after examining five overlapping resplits.
