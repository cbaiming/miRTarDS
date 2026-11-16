# MONDO training data

These ontology-only inputs derive from the MONDO snapshot dated 2026-08-04. Seed 42 partitions concepts; 17,422 non-root training concepts are eligible.

| Pair pool | Rows |
|---|---:|
| Natural training | 506,291 |
| Structural training | 232,396 |
| Synonym training | 75,195 |
| Seen/seen development | 39,842 |
| Seen/unseen development | 216,694 |
| Unseen/unseen development | 14,663 |

Pair tables include disease texts, concept identifiers and the ontology-similarity target `ylin`. The trainer removes root concepts and identical token sequences before sampling. [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) records file hashes; the [training guide](../README.md) specifies sampling and optimization.

Credit the Monarch Initiative and MONDO contributors. These derived tables retain MONDO's CC BY 4.0 terms. [MONDO project](https://github.com/monarch-initiative/mondo) · [License](https://github.com/monarch-initiative/mondo/blob/master/LICENSE)
