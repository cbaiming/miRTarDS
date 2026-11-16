# MeSH mapping sensitivity

MISIM's ancestor contributions, descriptor similarity and bidirectional best-match aggregation agree with equations 1–5 of [Wang et al. (2010)](https://doi.org/10.1093/bioinformatics/btq241). The benchmark adapts that formula to miRNA–gene pairs with automatic name mapping.

Pair coverage requires one mapped descriptor at each endpoint. Exact preferred-name matching retains 90.79% of miRNA-side and 32.30% of gene-side disease-name entries among test entities. None of the 1,013 covered pairs retains both complete original name sets.

| Disease-name mapping | Gene entries retained | MISIM balanced APS | Jaccard on identical mapped sets |
|---|---:|---:|---:|
| Exact preferred names | 32.30% | 0.565940 | 0.576438 |
| Preferred-concept aliases | 42.57% | 0.571213 | 0.588004 |
| All official entry terms | 53.89% | 0.576295 | 0.601662 |

Alias sensitivities use an official MeSH 2026 archive with the supplied graph held fixed and evaluate the same covered pairs. Both MISIM gains over exact matching have paired 95% intervals crossing zero. Original-name Jaccard reaches 0.577095 on these pairs.

Aliases can collapse narrower concepts or mix species-specific meanings. The graph's year is unknown. Treat expansion as exploratory; source concept and species checks remain necessary. The [benchmark definitions](../README.md) retain exact preferred-name mapping.

Reproduce with the [NLM archive](https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2026.gz):

~~~bash
python scripts/audit_misim_aliases.py --repo-root . --mesh-archive local_cache/desc2026.gz --output runs/misim_alias_audit
~~~
