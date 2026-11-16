# Benchmark data

The release uses data-generation seeds **0, 1, and 2**. P denotes database-observed interactions; U denotes unlabelled pairs and can contain undiscovered targets.

| Data seed | Training P / U | Test P / U | Training-seen mature-sequence test rows |
|---|---|---|---:|
| 0 | 3952 / 11968 | 164 / 174 | 5 |
| 1 | 4093 / 12298 | 177 / 177 | 0 |
| 2 | 3920 / 11801 | 165 / 163 | 0 |

Precursor components and Entrez genes are partitioned 70/15/15, stratified by positive involvement and disease-count bands. Pairs require both endpoints in one partition. Training uses approximately three source-stratified U per P; validation and test use approximately one. U is matched without replacement in two-dimensional log2 disease-count space. Cross-partition identical profiles are quarantined without replacement sampling.

Two fixed inner folds select epochs. Final models refit on complete outer training data. Mature-sequence overlap is reported separately.

Seed folders contain aligned rows, outer/inner indices, concept–name profiles, sampling details and encoder features. Features concatenate AB, AA and BB mean, maximum, population standard deviation and skew. Cosines are clipped to [0,1]; AA/BB position diagonals are excluded. Distinct concepts with identical text retain multiplicity. Statistics use six decimals.

Training/test features are finite. Unused native-validation and quarantine rows retain NaNs. See the [evaluation](../results/README.md), [regeneration inputs](source/README.md) and [data terms](DATA_LICENSES.md).
