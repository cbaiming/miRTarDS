# Matched disease-information comparison

All methods use the same candidate pairs and unique exact-mapped MeSH descriptor sets. OpenAI-large miRTarDS is refitted on those restricted profiles. Data and training seeds are 0, 1, and 2.

| Data seed | Test pairs | miRTarDS balanced APS ± training SD | MISIM | Mapped Jaccard |
|---|---:|---:|---:|---:|
| 0 | 336 | 0.684534 ± 0.003012 | 0.578584 | 0.594118 |
| 1 | 353 | 0.623085 ± 0.004510 | 0.570290 | 0.576997 |
| 2 | 324 | 0.623247 ± 0.004722 | 0.548946 | 0.558199 |
| Mean | | **0.643622** | **0.565940** | **0.576438** |

The mean advantage is 0.077682 over MISIM, with conditional paired 95% interval [0.003075, 0.145602], and 0.067184 over mapped Jaccard, with interval [−0.002653, 0.134137]. The latter includes zero. Resampling shares endpoint weights across overlapping splits.

Original partitions are retained. Restricted gene profiles recur across outer train/test on 8, 2, and 3 test rows. No complete 12-feature vector repeats across outer train/test or inner fit/tune. Within-training mixed-label duplicate groups are recorded in the audits.

This compares a supervised embedding-and-classifier pipeline with fixed similarities; encoder-only attribution requires a matched trained-head comparison. See [metrics](metrics.csv), [split summaries](comparison.csv), [predictions](predictions.tsv.gz), and [intervals](intervals.json). The [primary evaluation](../../primary/README.md) uses complete profiles.
