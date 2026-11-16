# Primary complete-profile evaluation

Both encoders use original complete disease profiles, identical candidate pairs, and data/training seeds 0, 1, and 2. Test sizes are 338, 354, and 328. The prediction method is fixed miRTarDS.

| Encoder | Data seed | Balanced APS ± training SD | ROC-AUC |
|---|---:|---:|---:|
| Fine-tuned BioLORD | 0 | 0.683110 ± 0.001599 | 0.691489 |
| Fine-tuned BioLORD | 1 | 0.660137 ± 0.004151 | 0.647238 |
| Fine-tuned BioLORD | 2 | 0.624822 ± 0.001747 | 0.638582 |
| OpenAI-large | 0 | 0.725073 ± 0.003314 | 0.734698 |
| OpenAI-large | 1 | 0.708435 ± 0.000999 | 0.685254 |
| OpenAI-large | 2 | 0.693006 ± 0.000888 | 0.694454 |

| Encoder | F1 | Accuracy | Recall | Precision |
|---|---:|---:|---:|---:|
| Fine-tuned BioLORD | 0.602575 | 0.613342 | 0.590458 | 0.615225 |
| OpenAI-large | 0.639194 | 0.647310 | 0.630221 | 0.648727 |

Classification uses the fixed score threshold 0.5 and test class proportions. Balanced APS weights each label class equally. Training SD measures training randomness; paired intervals condition on fitted predictions and account for shared endpoint groups.

Excluding the five training-seen mature-sequence test rows in split 0 gives mean balanced APS 0.657076 for fine-tuned BioLORD and 0.709206 for OpenAI-large. Splits 1 and 2 are unchanged. See the [sensitivity table](sequence_overlap_sensitivity.csv) and [summary](sequence_overlap_sensitivity_summary.json).

Download [per-fit metrics](per_training_seed.csv), [split summaries](per_data_summary.csv), [predictions](all_predictions.tsv.gz), and [paired intervals](paired_intervals.json). The [results overview](../README.md) distinguishes full-profile and matched-information comparisons.

The [original BioLORD control](../original_biolord/README.md) uses the same predictor and full test pairs.
