# Regenerating random splits

Additional splits require a locally acquired complete source pool:

| Relative path | Content |
|---|---|
| work/pairs.sqlite | Unique candidate IDs and evidence flags |
| profiles/miRNA_profiles.tsv | Mature identities, disease counts and precursor groups |
| profiles/gene_profiles.tsv | Entrez identities and disease counts |
| associations/miRNA_disease_deduplicated.tsv.gz | miRNA disease profiles |
| associations/gene_disease_deduplicated.tsv.gz | Gene disease profiles |
| audit/verification.json | Source validation |

The pair key is (mature_key << 32) | Entrez_gene_id. The disease-complete pool contains 8,097 qualifying P and 6,141,972 U pairs. [Source requirements](requirements.json) record schemas, evidence rules, snapshot hashes and acquisition links.

Acquire sources under their own terms. Exact regeneration requires historical snapshots; current releases can alter identities, evidence and counts. The bundled fixed benchmark supports evaluation without the complete master. Redistribution permission for every contributor to the composite master was not established, so that master remains external. See [data terms](../DATA_LICENSES.md).
