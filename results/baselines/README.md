# Disease-profile similarity baselines

MISIM adapts the disease-profile similarity of [Wang et al. (2010)](https://doi.org/10.1093/bioinformatics/btq241) from miRNA–miRNA to miRNA–gene pairs. Unique MeSH descriptors form an ancestor graph. Contributions decay by 0.5 per shortest edge distance; descriptor similarity sums shared-ancestor contributions and normalizes by both descriptors' semantic values. Profile similarity averages best matches in both directions.

Names undergo Unicode NFKC, whitespace collapse and lowercase normalization, followed by exact preferred-descriptor matching. Unmapped names are omitted; empty mapped endpoints produce abstentions. The supplied Category C graph's release year is unknown.

Jaccard is intersection divided by union. **Raw-name Jaccard** uses complete normalized name sets; **mapped Jaccard** uses the exact unique descriptor sets available to MISIM. Higher similarity predicts P; distance equals one minus similarity.

| Data seed | Full test pairs | Both-end MeSH coverage |
|---|---:|---:|
| 0 | 338 | 336 |
| 1 | 354 | 353 |
| 2 | 328 | 324 |

The [primary comparison](../primary/README.md) evaluates complete-profile miRTarDS on these same covered pairs. The [matched-information experiment](../secondary/matched_information/README.md) also restricts training inputs to the shared descriptor sets.

Recompute fixed scores after installing the package:

~~~bash
python -m mirtards_release.baselines --profiles benchmark_data/seed0/profiles.json.gz --rows benchmark_data/seed0/rows.tsv.gz --mesh raw_data/MeSH_Tree.csv --output baseline_seed0.tsv
~~~

Repeat for seeds 1 and 2. Similarities use disease information only. Evaluation joins labels by row identity, computes class-balanced average precision, and averages data splits equally. The [mapping analysis](diagnostics/README.md) reports information loss, formula checks and official-alias sensitivity.
