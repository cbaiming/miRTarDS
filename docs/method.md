# Method

miRTarDS ranks candidate miRNA–gene interactions using the disease profiles associated with each endpoint. It accepts OpenAI text-embedding-3-large or the released fine-tuned BioLORD encoder.

## Twelve disease features

Let A and B contain the disease vectors for a miRNA and a gene. Normalize each vector to unit length and clip cosine similarities to [0,1]. Identical disease tokens have similarity 1.

For AB, AA and BB, calculate the mean, maximum, population standard deviation and biased skewness, in that order. AA and BB exclude positional self-comparisons. Different concepts sharing the same text retain their multiplicity. Empty self-comparisons yield four zeros; constant comparisons have zero skewness. Round the twelve values to six decimals.

## Prediction model

The final score is a weighted mean of four fixed branches.

| Input | Weight | Classifier |
|---|---:|---|
| Original twelve features | 0.30 | MLP, 32 → 16 → 1, ReLU, dropout 0.1 |
| Disease-distribution contrasts | 0.20 | MLP, 24 → 12 → 1, tanh, dropout 0.2 |
| Selected contrast features | 0.20 | Histogram gradient boosting |
| Selected contrast features | 0.30 | RBF support-vector machine |

The [method implementation](../src/mirtards_release/) specifies the transformations, selected columns and hyperparameters. Disease counts enter sampling, while classifier inputs use the twelve statistics and their transformations. The first branch retains two zero padding slots for numerical compatibility.

Two inner fit/tune partitions select each MLP's epoch count, with an 80-epoch limit and patience 12. Refit a new scaler and model on full training data for the ceiling of the two selected epoch counts' mean. Tree and SVM branches fit full training data. Training uses class-balanced observed labels and deterministic two-thread CPU computation.

## Evaluation

Data seeds and training seeds are each 0, 1 and 2, using all nine combinations per encoder. Balanced APS gives P and U equal evaluation mass. Other metrics use the observed test proportions; classification uses score ≥ 0.5. Average metrics across training seeds and then across data seeds.

The score supports retrospective P/U prioritization under the published sampling design. Functional regulation in a new biological experiment requires separate validation.
