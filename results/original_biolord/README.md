# Original BioLORD control

This control uses original BioLORD-2023 with the same miRTarDS predictor, complete disease profiles and fixed partitions as the main evaluation. Data and training seeds are 0, 1, and 2, giving nine fitted models.

| Data seed | Test pairs | Balanced APS ± training SD | ROC-AUC |
|---|---:|---:|---:|
| 0 | 338 | 0.683600 ± 0.004400 | 0.690858 |
| 1 | 354 | 0.653631 ± 0.004301 | 0.642536 |
| 2 | 328 | 0.623580 ± 0.002487 | 0.638024 |

Equal-data mean balanced APS is **0.653604**, raw APS **0.650506** and ROC-AUC **0.657139**. At the fixed threshold 0.5, F1 is 0.597925, accuracy 0.608760, recall 0.586260 and precision 0.610156.

The fine-tuned-minus-original BioLORD balanced-APS difference is +0.002419, with a conditional paired 95% interval [-0.002861, 0.007216]. The interval uses endpoint resampling conditional on the fitted models and includes zero. The current task therefore gives uncertain evidence of a fine-tuning advantage.

See [per-fit metrics](per_training_seed.csv), [split summaries](per_data_summary.csv) and [comparison intervals](SUMMARY.json). The [main results](../README.md) compare all three encoders.
