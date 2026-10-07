"""Python image deduplication and official metadata filtering, no model training."""
import csv, json, random
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from data import ROOT, normalize_text, image_hash, check_split_leakage

def dhash(path):
    with Image.open(path) as im:
        pixels=np.asarray(im.convert('L').resize((9,8),Image.Resampling.LANCZOS))
    return int.from_bytes(np.packbits(pixels[:,1:]>pixels[:,:-1]).tobytes(),'big')

def prepare(out, verified_only=False):
    # Greedy filtering over all raw rows before sampling, held-out splits first.
    # Four 16-bit buckets guarantee a candidate for 64-bit distance <= 3.
    provenance={}
    if verified_only:
        for line in (ROOT/'reports/provenance-matches.jsonl').read_text().splitlines():
            r=json.loads(line); provenance[(r['community_split'],r['row_index'])]=r
    hashes=[]; retained=[]
    buckets=defaultdict(list); seen_text=set(); seen_image=set(); splits={}; audit={}
    for split in ['test','val','train']:
        kept=[]; counts=Counter(); examples=[]
        for row_index,row in enumerate(csv.DictReader((ROOT/f'data/raw/{split}.csv').open())):
            official=provenance.get((split,row_index))
            if verified_only and (official is None or official['status']!='verified'):
                counts['unverified_provenance']+=1; continue
            path=ROOT/'data/processed/images'/Path(row['image_path']).name
            if not path.exists(): counts['missing_image']+=1; continue
            txt=normalize_text(row['text']); digest=image_hash(path)
            if not txt or txt in seen_text or digest in seen_image:
                counts['exact_duplicate']+=1; continue
            h=dhash(path); candidate_ids=set()
            for chunk in range(4): candidate_ids.update(buckets[(chunk,(h>>(16*chunk))&65535)])
            duplicate=next((i for i in sorted(candidate_ids) if (h ^ hashes[i]).bit_count()<=3),None)
            if duplicate is not None:
                counts['perceptual_candidate_removed']+=1
                if len(examples)<20: examples.append({'removed':str(path.relative_to(ROOT)),'retained':retained[duplicate]['image'],'retained_split':retained[duplicate]['split'],'distance':(h^hashes[duplicate]).bit_count()})
                continue
            record={'id':f'{split}:{path.name}','text':row['text'],'image':str(path.relative_to(ROOT)),'image_sha256':digest,'label':int(row['6_way_label']),'source_url':row['original_url']}
            if verified_only:
                record['official_id']=official['official_candidates'][0]['id']
                record['subreddit']=official['official_candidates'][0]['subreddit']
            idx=len(hashes); hashes.append(h); retained.append({**record,'split':split})
            for chunk in range(4): buckets[(chunk,(h>>(16*chunk))&65535)].append(idx)
            seen_text.add(txt); seen_image.add(digest); kept.append(record)
        selected=[]; rng=random.Random(42)
        for label in range(6):
            pool=[r for r in kept if r['label']==label]; rng.shuffle(pool)
            selected.extend(pool[:200 if split=='train' else 60])
        rng.shuffle(selected); splits[split]=selected
        audit[split]={'removed':dict(counts),'available':dict(Counter(r['label'] for r in kept)),'selected':dict(Counter(r['label'] for r in selected)),'near_duplicate_examples':examples}
    check_split_leakage(splits)
    data_dir=ROOT/'data/processed'/out.name; data_dir.mkdir(parents=True,exist_ok=True)
    for split,rows in splits.items():
        (data_dir/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    (out/'data-audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False))
    return splits

