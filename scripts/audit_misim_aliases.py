"""Official MeSH2026 alias sensitivity with the original MISIM graph held fixed."""
from pathlib import Path
import sys,json,gzip,hashlib,argparse,xml.etree.ElementTree as ET
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
REPO=None
OUT=None
from mirtards_release.baselines import MISIM2010,normalize_name,_semantic_matrix
from mirtards_release.metrics import evaluate_scores

def save(name,v):
    (OUT/name).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf8")
def sha(path):
    with Path(path).open("rb") as f:
        digest=hashlib.sha256()
        for block in iter(lambda:f.read(8 << 20),b""):
            digest.update(block)
        return digest.hexdigest()

def main():
    global REPO, OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root",default=".")
    parser.add_argument("--output",required=True)
    parser.add_argument("--mesh-archive",required=True,help="Official desc2026.gz with the audited SHA256")
    args=parser.parse_args()
    REPO=Path(args.repo_root).resolve()
    OUT=Path(args.output).resolve()
    OUT.mkdir(parents=True,exist_ok=True)
    archive=Path(args.mesh_archive).resolve()
    if sha(archive)!="ccd4d0d33bebfd4c836a59e7dd0c635b1d4e20c4f5a6abc55e2eb0ba4c8716dd":
        raise ValueError("Expected the audited official MeSH2026 descriptor archive")
    if sha(REPO/"raw_data/MeSH_Tree.csv")!="3b76b96e8020faee03235a9ecbb849352b394740aa680c51b911e14d66f3ff4b":
        raise ValueError("Expected the unchanged original MeSH graph")
    mesh=MISIM2010.from_csv(REPO/"raw_data/MeSH_Tree.csv")
    profiles={};frames={};raw={};needed=set()
    for ds in (0,1,2):
        folder=REPO/f"benchmark_data/seed{ds}"
        with gzip.open(folder/"profiles.json.gz","rt",encoding="utf8") as f:profiles[ds]=json.load(f)
        frames[ds]=pd.read_csv(folder/"rows.tsv.gz",sep="\t",dtype={"a_id":str,"b_id":str,"row_id":str,"a_group":str,"b_group":str})
        frame=frames[ds].query("split=='test'").copy()
        raw[ds]={side:{} for side in ("a","b")}
        for side in ("a","b"):
            for e in frame[side+"_id"].unique():
                bag={normalize_name(x[1]) for x in profiles[ds][side][e]}
                raw[ds][side][e]=bag;needed.update(bag)
    protocol=dict(official_source="https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2026.gz",
        production_year=2026,base_graph="Original Category-C CSV, unknown production year, never replaced",
        anchor="Official DescriptorName matches original normalized preferred name exactly",
        strict="Original exact name has priority, else globally unique alias among preferred concepts only, anchored to one existing descriptor",
        broad="Original exact name has priority, else globally unique alias among all official descriptor concepts, anchored to one existing descriptor. Includes narrower/broader/related concepts within a descriptor.",
        no_fuzzy=True,outcomes_used_for_mapping=False,primary_test_subset="Original MISIM both-end-covered test rows, fixed1013rows across data0/1/2",
        extra_coverage="Reported separately, never substituted into primary performance difference")
    save("protocol.json",protocol)
    strict_index=defaultdict(set);broad_index=defaultdict(set)
    anchored={};info={};parsed=0;preferred_match=0
    with gzip.open(archive,"rb") as f:
        for _,e in ET.iterparse(f,events=("end",)):
            if e.tag!="DescriptorRecord":continue
            parsed+=1
            uid=e.findtext("DescriptorUI");name=e.findtext("DescriptorName/String")
            key=normalize_name(name)
            if key in mesh.names:
                anchored[uid]=mesh.names[key];preferred_match+=1
            strict={key};broad={key}
            for concept in e.findall("ConceptList/Concept"):
                terms={normalize_name(v.text) for v in concept.findall("TermList/Term/String") if v.text}
                cn=concept.findtext("ConceptName/String")
                if cn:terms.add(normalize_name(cn))
                broad.update(terms)
                if concept.attrib.get("PreferredConceptYN")=="Y":strict.update(terms)
            for term in strict&needed:strict_index[term].add(uid)
            for term in broad&needed:broad_index[term].add(uid)
            if (strict|broad)&needed:
                info[uid]=dict(name=name,tree_numbers=[v.text for v in e.findall("TreeNumberList/TreeNumber")])
            e.clear()
    def mapping(index):
        result=dict(mesh.names)
        added={}
        for term,uids in index.items():
            if term in result or len(uids)!=1:continue
            uid=next(iter(uids))
            if uid in anchored:
                result[term]=anchored[uid];added[term]=dict(descriptor=anchored[uid],uid=uid)
        return result,added
    strict_map,strict_added=mapping(strict_index)
    broad_map,broad_added=mapping(broad_index)
    maps={"exact":mesh.names,"preferred_concept_aliases":strict_map,"all_descriptor_entry_terms":broad_map}
    mappings=[]
    for name in sorted(needed):
        row=dict(name=name,exact=mesh.names.get(name),preferred_concept_aliases=strict_map.get(name),all_descriptor_entry_terms=broad_map.get(name),
                 official_preferred_concept_uids=";".join(sorted(strict_index[name])),official_any_concept_uids=";".join(sorted(broad_index[name])))
        mappings.append(row)
    pd.DataFrame(mappings).to_csv(OUT/"name_mappings.csv",index=False)
    save("official_records.json",info)
    matrix_descriptors=sorted(set(v for m in maps.values() for k,v in m.items() if k in needed))
    similarities=_semantic_matrix(mesh,matrix_descriptors)
    lookup={v:i for i,v in enumerate(matrix_descriptors)}
    rows=[];coverage=[]
    for ds in (0,1,2):
        test=frames[ds].query("split=='test'").copy()
        mapped={variant:{side:{} for side in ("a","b")} for variant in maps}
        for variant,mp in maps.items():
            for side in ("a","b"):
                total=0;hit=0
                for entity,bag in raw[ds][side].items():
                    total+=len(bag);hit+=sum(v in mp for v in bag)
                    mapped[variant][side][entity]=set(mp[v] for v in bag if v in mp)
                coverage.append(dict(data_seed=ds,variant=variant,side=side,name_occurrences=total,mapped_occurrences=hit,
                                     occurrence_fraction=hit/total,unique_names=len(set().union(*raw[ds][side].values())),
                                     mapped_unique_names=len({v for bag in raw[ds][side].values() for v in bag if v in mp})))
        for r in test.itertuples():
            a,b=raw[ds]["a"][r.a_id],raw[ds]["b"][r.b_id]
            row=dict(data_seed=ds,row_id=r.row_id,a_id=r.a_id,b_id=r.b_id,label=r.label,
                     raw_jaccard=len(a&b)/len(a|b),a_group=r.a_group,b_group=r.b_group)
            for variant in maps:
                x,y=mapped[variant]["a"][r.a_id],mapped[variant]["b"][r.b_id]
                covered=bool(x and y);row[variant+"_covered"]=covered
                row[variant+"_mapped_count_a"]=len(x);row[variant+"_mapped_count_b"]=len(y)
                if covered:
                    ai=np.array([lookup[d] for d in sorted(x)]);bi=np.array([lookup[d] for d in sorted(y)])
                    k=similarities[np.ix_(ai,bi)]
                    row[variant+"_misim"]=float((k.max(1).sum()+k.max(0).sum())/(len(x)+len(y)))
                    row[variant+"_jaccard"]=len(x&y)/len(x|y)
                else:
                    row[variant+"_misim"]=np.nan;row[variant+"_jaccard"]=np.nan
            rows.append(row)
    scores=pd.DataFrame(rows)
    scores.to_csv(OUT/"scores.csv",index=False,float_format="%.17g")
    pd.DataFrame(coverage).to_csv(OUT/"coverage.csv",index=False)
    results=[]
    for subset in ("original_common","variant_covered"):
        for ds in (0,1,2):
            f=scores[scores.data_seed==ds]
            for variant in maps:
                selected=f.loc[f.exact_covered if subset=="original_common" else f[variant+"_covered"]]
                for score in ("misim","jaccard"):
                    key=variant+"_"+score
                    m=evaluate_scores(selected.label.to_numpy(int),selected[key].to_numpy(float))
                    results.append(dict(subset=subset,data_seed=ds,variant=variant,metric=score,rows=len(selected),
                                        balanced_aps=m["balanced_aps"],raw_aps=m["raw_aps"],auroc=m["auroc"]))
    summary=pd.DataFrame(results);summary.to_csv(OUT/"per_data_results.csv",index=False)
    average=summary.groupby(["subset","variant","metric"])[["balanced_aps","raw_aps","auroc"]].mean().reset_index()
    average.to_csv(OUT/"mean_results.csv",index=False)
    examples=[]
    for name in sorted(set(strict_added)|set(broad_added)):
        rec=strict_added.get(name,broad_added.get(name))
        examples.append(dict(name=name,strict=name in strict_added,**rec))
    pd.DataFrame(examples).to_csv(OUT/"new_mapping_examples.csv",index=False)
    missing_official=[]
    for name in sorted(needed-set(mesh.names)):
        uids=broad_index.get(name,set())
        if uids and not any(uid in anchored for uid in uids):
            missing_official.append(dict(name=name,official_descriptors="; ".join(info[uid]["name"] for uid in sorted(uids)),
                                         tree_numbers="; ".join(",".join(info[uid]["tree_numbers"]) for uid in sorted(uids))))
    pd.DataFrame(missing_official).to_csv(OUT/"official_terms_outside_fixed_graph.csv",index=False)
    save("audit.json",dict(official_archive_sha256=sha(archive),official_archive_bytes=(archive).stat().st_size,
        parsed_official_descriptors=parsed,anchored_preferred_descriptors=preferred_match,old_descriptors=len(mesh.codes),
        original_unique_input_names=len(needed),strict_added_unique_names=len(strict_added),broad_added_unique_names=len(broad_added),
        outside_fixed_graph_input_names=len(missing_official),original_common_rows=int(scores.exact_covered.sum()),
        strict_covered_rows=int(scores.preferred_concept_aliases_covered.sum()),broad_covered_rows=int(scores.all_descriptor_entry_terms_covered.sum()),
        rows_unchanged=True,graph_unchanged=True,mapping_frozen_before_labels=True,
        scope="Exploratory mapping sensitivity on previously evaluated data; never used outcomes to select aliases or parameters"))
    print(average.to_string(index=False),flush=True)
if __name__=="__main__":main()

