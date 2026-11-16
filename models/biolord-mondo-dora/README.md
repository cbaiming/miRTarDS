---
language: en
library_name: sentence-transformers
base_model: FremyCompany/BioLORD-2023
license: other
license_name: ihtsdo-and-nlm-licences
license_link: https://huggingface.co/FremyCompany/BioLORD-2023#license
---

# Fine-tuned BioLORD disease encoder

This encoder adapts BioLORD-2023 to MONDO disease similarity using DoRA. The released checkpoint was selected at step 600 using ontology development data. Its full FP32 weights, tokenizer and pooling configuration support offline inference. Six standard Safetensors shards are individually below 100 MB.

## Local inference

After `python -m pip install -e ".[biolord]"`, run from the repository root:

```python
from mirtards_release.biolord import encode_texts

vectors = encode_texts(
    ["diabetes mellitus", "breast carcinoma"],
    "models/biolord-mondo-dora",
    device="cpu",
)
print(vectors.shape)  # (2, 768)
```

Inference uses a 64-token limit, attention-mask mean pooling in FP32 and float64 L2 normalization. Use this API to reproduce downstream disease statistics. An optional CUDA device accelerates inference.

## Training and development results

Adaptation trains 55,296 parameters in the query/value projections of the last four MPNet layers. The objective combines ontology-similarity ranking with penalties preserving embeddings and pairwise geometry. Each update samples 77 natural, 38 structural and 13 synonym pairs. Model selection uses mean Spearman correlation across seen/seen, seen/unseen and unseen/unseen development strata, subject to displacement and cosine-RMSE limits.

| Ontology-development mean Spearman | Value |
|---|---:|
| Original BioLORD-2023 | 0.547263 |
| Fine-tuned BioLORD | 0.548393 |

The development gain is small. See the [training calculation](../../scripts/model_tools/README.md), [development history](results/ontology_development_history.csv) and [downstream results](../../results/README.md).

The retained adapter and merged weights have verified provenance in [EXPORT_AUDIT.json](EXPORT_AUDIT.json). Sharding preserves all tensors exactly.

## Attribution and use conditions

BioLORD derives from work by Remy, Demuynck and Demeester, JAMIA 2024, DOI 10.1093/jamia/ocae029. Its upstream card lists IHTSDO and NLM licensing conditions and requires appropriate UMLS/SNOMED CT licensing. Those conditions apply to these derived weights. The repository's code license is separate. [BioLORD terms](https://huggingface.co/FremyCompany/BioLORD-2023#license)

Training resources derive from MONDO, credited to the Monarch Initiative and MONDO contributors, under CC BY 4.0. [MONDO license](https://github.com/monarch-initiative/mondo/blob/master/LICENSE)
