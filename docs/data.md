# Data and preprocessing

Data seeds 0, 1 and 2 control candidate sampling and partitions. Training seeds 0, 1 and 2 control model initialization, batches and dropout. Each encoder uses all nine combinations. The three data seeds were selected for release after examining an exploratory five-seed experiment.

## Sources and labels

The source cohort combines identity-resolved human miRTarBase R11, TarBase v9 and the recorded prediction resources. P denotes qualifying functional, luciferase or Western-blot evidence. Other candidates remain U, including historical condition-specific nonfunctional records.

Mature MIMAT accessions identify miRNAs; stable gene IDs identify genes. Diseases are deduplicated by source concept. Different concepts sharing a normalized disease name remain separate. Normalization uses NFKC, collapsed whitespace and casefolding.

## Sampling and partitions

1. Start from 8,097 eligible P pairs and 6,141,972 U pairs.
2. Allocate precursor-connected miRNA components and genes independently to train/valid/test in proportions 70/15/15. Stratify by P involvement and floor(log2(1 + disease count)).
3. Keep pairs whose endpoints share a partition. Retain eligible P pairs and sample approximately three U per P for training, one per P for valid/test.
4. Preserve U source-stratum quotas. Match U to P without replacement using nearest nonempty cells in two-dimensional log2 disease-count space.
5. Quarantine precursor or identical-profile groups spanning partitions, without replenishing samples.
6. With data seed + 10000, allocate training endpoint groups into five groups and form the two published inner fit/tune partitions.

Shared mature sequences are audited separately. Seed 0 has five test records sharing a mature sequence with training; seeds 1 and 2 have none. Sequence-excluded results accompany the main results.

## Recreate a split

Acquire the [required source inputs](../benchmark_data/source/README.md), then run from the repository root:

```bash
python scripts/build_datasets.py --master-dir local_cache/mature_master_v1 --output runs/regenerated --seeds 0 1 2
```

The [released data](../benchmark_data/README.md) provide profiles, labels, indices and aligned feature matrices. Training and test features are finite; unused valid/quarantine rows may contain NaNs. Source annotations retain their [own use conditions](../benchmark_data/DATA_LICENSES.md).
