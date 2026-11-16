"""Read-only mother dataset -> sealed mature-chain double-cold P/U experiment."""
from __future__ import annotations
import array, bisect, collections, csv, gzip, hashlib, json, math, random, sqlite3
import statistics, sys, time, unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MASTER = Path('benchmark_data/source/mature_master_v1')
OUT = HERE / 'dataset'
SEED = 20260917
PARTS = ('train', 'valid', 'test')
PMASK = 49
SOURCES = [('miRTarBase_R11', 1039), ('TarBase_v9', 112), ('TargetScan', 128),
           ('miRDB_v6.0', 256), ('miRWalk_3UTR', 512)]
FIELDS = ['miRNA', 'gene', 'label', 'source', 'miRNA disease count', 'gene disease count']

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def norm(x):
    return ' '.join(unicodedata.normalize('NFKC', x).casefold().split())

def original_label(flags):
    return 1 if flags & PMASK else (0 if flags & 2 else -1)

def stratum(flags):
    if flags & PMASK: return 'P'
    if flags & 2: return 'previous_conditional_nonfunctional'
    if flags & (4 | 8 | 64 | 1024): return 'other_experimental'
    return 'prediction_only'

def source(flags):
    return ';'.join(s for s, mask in SOURCES if flags & mask)

class UnionFind:
    def __init__(self, ids): self.parent = {x: x for x in ids}
    def find(self, x):
        while x != self.parent[x]:
            self.parent[x] = self.parent[self.parent[x]]; x = self.parent[x]
        return x
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b: self.parent[max(a,b)] = min(a,b)

def allocations(n, proportions):
    den = sum(proportions.values())
    exact = {k: n*v/den for k,v in proportions.items()}
    result = {k: math.floor(v) for k,v in exact.items()}
    for k in sorted(exact, key=lambda k: (-(exact[k]-result[k]), k))[:n-sum(result.values())]: result[k] += 1
    assert sum(result.values()) == n
    return result

def split_entities(ids, counts, p_ids, seed):
    bins = collections.defaultdict(list)
    for x in sorted(ids): bins[(int(x in p_ids), int(math.log2(1+counts[x])))].append(x)
    rng = random.Random(seed); result = {}
    for _, members in sorted(bins.items()):
        rng.shuffle(members)
        sizes = allocations(len(members), {'train':70,'valid':15,'test':15})
        at = 0
        for p in PARTS:
            for x in members[at:at+sizes[p]]: result[x] = p
            at += sizes[p]
    assert len(result) == len(ids)
    return result

class NearestCells:
    """Exact nearest nonempty log-count cell, sampled without replacement."""
    def __init__(self, cells):
        self.cells = cells
        by_a = collections.defaultdict(list)
        for a,b in cells: by_a[a].append(b)
        self.by_a = {a:sorted(bb) for a,bb in by_a.items()}
        self.log_a = {a:math.log2(a) for a in self.by_a}
        self.log_b = {b:math.log2(b) for _,b in cells}
    def pop(self, a, b, rng):
        la, lb = math.log2(a), math.log2(b)
        best, choices = float('inf'), []
        for ca in sorted(self.by_a, key=lambda x:((self.log_a[x]-la)**2,x)):
            dx = (self.log_a[ca]-la)**2
            if dx > best + 1e-14: break
            bs = self.by_a[ca]; right = bisect.bisect_left(bs,b); left = right-1
            while left >= 0 or right < len(bs):
                dl = (self.log_b[bs[left]]-lb)**2 if left >= 0 else float('inf')
                dr = (self.log_b[bs[right]]-lb)**2 if right < len(bs) else float('inf')
                if min(dl,dr)+dx > best+1e-14: break
                if dl <= dr: cb=bs[left]; left-=1
                else: cb=bs[right]; right+=1
                if not self.cells[(ca,cb)]: continue
                d = dx+(self.log_b[cb]-lb)**2
                if d < best-1e-14: best,choices=d,[(ca,cb)]
                elif abs(d-best) <= 1e-14: choices.append((ca,cb))
        if not choices: raise RuntimeError('U source stratum exhausted')
        cell = rng.choice(choices); entries=self.cells[cell]
        i=rng.randrange(len(entries)); pid=entries[i]; entries[i]=entries[-1]; entries.pop()
        return pid, best, cell

def quantiles(values):
    vv=sorted(values)
    def q(p):
        x=p*(len(vv)-1); lo=int(x); hi=min(lo+1,len(vv)-1)
        return vv[lo]*(hi-x)+vv[hi]*(x-lo) if hi != lo else vv[lo]
    return {str(p):q(p) for p in (0,.25,.5,.75,.9,.99,1)} if vv else {}

def write_tsv(path, rows, fields):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fields,delimiter='\t');w.writeheader();w.writerows(rows)

def main():
    started=time.time()
    if OUT.exists(): raise RuntimeError('Refusing to overwrite existing dataset')
    assert json.loads((MASTER/'audit/verification.json').read_text(encoding='utf-8'))
    mirs={int(r['mature_key']):r for r in csv.DictReader((MASTER/'profiles/miRNA_profiles.tsv').open(encoding='utf-8'),delimiter='\t')}
    genes={int(r['gene_id']):r for r in csv.DictReader((MASTER/'profiles/gene_profiles.tsv').open(encoding='utf-8'),delimiter='\t')}
    nm={k:int(r['disease_count']) for k,r in mirs.items()}; ng={k:int(r['disease_count']) for k,r in genes.items()}
    uf=UnionFind(mirs); precursor_members=collections.defaultdict(list)
    for m,r in mirs.items():
        for p in r['precursor_accessions'].split(';'):
            if p: precursor_members[p].append(m)
    for members in precursor_members.values():
        for m in members[1:]:uf.union(members[0],m)
    component={m:uf.find(m) for m in mirs}
    conn=sqlite3.connect((MASTER/'work/pairs.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    positive={}
    for pid,flags in conn.execute('SELECT pair_id,flags FROM pairs WHERE (flags & 49) != 0'):
        m,g=pid>>32,pid&0xffffffff
        if nm[m] and ng[g]:positive[pid]=flags
    assert len(positive)==8097
    active_m={m for m in mirs if nm[m]>0};active_g={g for g in genes if ng[g]>0}
    comp_counts={}
    for m in active_m:comp_counts[component[m]]=max(comp_counts.get(component[m],0),nm[m])
    ps_m={component[p>>32] for p in positive}; ps_g={p&0xffffffff for p in positive}
    cs=split_entities(set(comp_counts),comp_counts,ps_m,SEED)
    gs=split_entities(active_g,ng,ps_g,SEED+1)
    ms={m:cs[component[m]] for m in active_m}
    p_by={s:[] for s in PARTS}; pool={s:collections.defaultdict(dict) for s in PARTS}
    counts=collections.Counter();flags_counts=collections.Counter()
    print('Scanning immutable pair index',flush=True)
    for pid,flags in conn.execute('SELECT pair_id,flags FROM pairs'):
        m,g=pid>>32,pid&0xffffffff
        if not nm[m] or not ng[g]:continue
        side_a,side_b=ms[m],gs[g]; label='P' if flags&PMASK else 'U'
        counts[(side_a,side_b,label)]+=1
        if side_a!=side_b:continue
        if label=='P':p_by[side_a].append(pid);continue
        st=stratum(flags);key=(nm[m],ng[g]); flags_counts[(side_a,st)]+=1
        pool[side_a][st].setdefault(key,array.array('Q')).append(pid)
    all_rows=[];sample_stats={};rng=random.Random(SEED+2)
    for part in PARTS:
        positives=p_by[part];assert len(positives)>=25,(part,len(positives))
        ratio=3 if part=='train' else 1
        availability={st:flags_counts[(part,st)] for st in pool[part]}
        draws=allocations(ratio*len(positives),availability)
        labels=[st for st,n in sorted(draws.items()) for _ in range(n)];rng.shuffle(labels)
        matchers={st:NearestCells(cells) for st,cells in pool[part].items()}
        targets=positives*ratio;rng.shuffle(targets)
        selected=[(pid,0,0.,'',None) for pid in positives]
        for pid,st in zip(targets,labels):
            m,g=pid>>32,pid&0xffffffff
            uid,d,cell=matchers[st].pop(nm[m],ng[g],rng)
            selected.append((uid,pid,d,st,cell))
        rng.shuffle(selected)
        for pid,matched,d,st,cell in selected:
            m,g=pid>>32,pid&0xffffffff
            flags=positive[pid] if pid in positive else conn.execute('SELECT flags FROM pairs WHERE pair_id=?',(pid,)).fetchone()[0]
            is_p=bool(flags&PMASK)
            assert is_p == (not matched)
            mm,gg=(matched>>32,matched&0xffffffff) if matched else (m,g)
            all_rows.append(dict(pair_id=str(pid),split=part,mature_key=m,mature_accession=mirs[m]['mature_accession'],
                precursor_component=component[m],gene_id=g,miRNA=mirs[m]['miRNA'],gene=genes[g]['gene'],label=1 if is_p else -1,
                observed_y=int(is_p),original_label=original_label(flags),source=source(flags),flags=flags,u_stratum=st or 'P',
                **{'miRNA disease count':nm[m],'gene disease count':ng[g]},matched_positive_pair_id=str(matched) if matched else '',
                match_distance=d,mirna_count_ratio=max(nm[m]/nm[mm],nm[mm]/nm[m]),
                gene_count_ratio=max(ng[g]/ng[gg],ng[gg]/ng[g])))
        sample_stats[part]={'P':len(positives),'U':len(targets),'available_U':availability,'sampled_U':draws}
    assert len({r['pair_id'] for r in all_rows})==len(all_rows)
    assert sum(v for (a,b,l),v in counts.items() if l=='P')==8097
    assert sum(v for (a,b,l),v in counts.items() if l=='U')==6141972
    for part in PARTS:
        selected=[r for r in all_rows if r['split']==part]
        assert {int(r['pair_id']) for r in selected if r['observed_y']}==set(p_by[part])
    for i,a in enumerate(PARTS):
        for b in PARTS[i+1:]:
            for field in ['mature_key','precursor_component','gene_id']:
                assert not {r[field] for r in all_rows if r['split']==a}&{r[field] for r in all_rows if r['split']==b}
    OUT.mkdir(parents=True)
    for part in PARTS:
        with (OUT/(part+'.csv')).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,FIELDS,extrasaction='ignore');w.writeheader();w.writerows(r for r in all_rows if r['split']==part)
    write_tsv(OUT/'pairs_audit.tsv',all_rows,list(all_rows[0]))
    write_tsv(OUT/'mirna_entity_split.tsv',[dict(mature_key=m,miRNA=mirs[m]['miRNA'],precursor_accessions=mirs[m]['precursor_accessions'],component=component[m],split=ms[m]) for m in sorted(ms)],['mature_key','miRNA','precursor_accessions','component','split'])
    write_tsv(OUT/'gene_entity_split.tsv',[dict(gene_id=g,gene=genes[g]['gene'],split=gs[g]) for g in sorted(gs)],['gene_id','gene','split'])
    used_m={r['mature_accession'] for r in all_rows}; used_g={r['gene_id'] for r in all_rows}
    prof_m=collections.defaultdict(list);prof_g=collections.defaultdict(list)
    with gzip.open(MASTER/'associations/miRNA_disease_deduplicated.tsv.gz','rt',encoding='utf-8') as f:
        for r in csv.DictReader(f,delimiter='\t'):
            if r['mature_accession'] in used_m:prof_m[r['mature_accession']].append([r['normalized_disease_name'],norm(r['disease'])])
    with gzip.open(MASTER/'associations/gene_disease_deduplicated.tsv.gz','rt',encoding='utf-8') as f:
        for r in csv.DictReader(f,delimiter='\t'):
            if int(r['gene_id']) in used_g:prof_g[r['gene_id']].append([r['disease_id'],norm(r['disease'])])
    for r in all_rows:
        assert len(prof_m[r['mature_accession']])==r['miRNA disease count']
        assert len(prof_g[str(r['gene_id'])])==r['gene disease count']
    save(OUT/'disease_profiles.json',{'mirna':dict(prof_m),'gene':dict(prof_g)})
    diagnostics={}
    for part in PARTS:
        rr=[r for r in all_rows if r['split']==part]; uu=[r for r in rr if not r['observed_y']]
        diagnostics[part]={'rows':len(rr),'miRNAs':len({r['mature_key'] for r in rr}),'precursor_components':len({r['precursor_component'] for r in rr}),
          'genes':len({r['gene_id'] for r in rr}),'P_by_arm':dict(collections.Counter(r['miRNA'][-2:] for r in rr if r['observed_y'])),
          'U_by_arm':dict(collections.Counter(r['miRNA'][-2:] for r in uu)),
          'within_2fold_both':sum(r['mirna_count_ratio']<=2 and r['gene_count_ratio']<=2 for r in uu),
          'exact_count_matches':sum(r['match_distance']<1e-14 for r in uu),'match_distance':quantiles([r['match_distance'] for r in uu]),
          'mirna_count_ratio':quantiles([r['mirna_count_ratio'] for r in uu]),'gene_count_ratio':quantiles([r['gene_count_ratio'] for r in uu]),
          'disease_counts':{cl:{k:quantiles([r[k] for r in rr if r['observed_y']==y]) for k in ['miRNA disease count','gene disease count']} for cl,y in [('P',1),('U',0)]}}
    input_paths=[MASTER/'work/pairs.sqlite',MASTER/'profiles/miRNA_profiles.tsv',MASTER/'profiles/gene_profiles.tsv',
        MASTER/'associations/miRNA_disease_deduplicated.tsv.gz',MASTER/'associations/gene_disease_deduplicated.tsv.gz',MASTER/'audit/verification.json']
    manifest={'seed':SEED,'mother_P':8097,'mother_U':6141972,'original_0_moved_to_U':207,
      'candidate_pool_matrix':{'|'.join(k):v for k,v in sorted(counts.items())},'split_sizes':sample_stats,'diagnostics':diagnostics,
      'entity_partition_counts':{'mirna':dict(collections.Counter(ms.values())),'precursor_components':dict(collections.Counter(cs.values())),'gene':dict(collections.Counter(gs.values()))},
      'no_pair_entity_or_component_overlap':True,'all_eligible_P_retained':True,'all_annotations_retained':True,
      'source_sha256':{p.relative_to(MASTER).as_posix():sha(p) for p in input_paths},
      'builder_sha256':sha(Path(__file__)),'protocol_document':'docs/data.md',
      'artifacts':{p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file()},'elapsed_seconds':time.time()-started}
    save(OUT/'manifest.json',manifest)
    print(json.dumps({'split_sizes':sample_stats,'diagnostics':diagnostics,'seconds':time.time()-started},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
