"""Match community records to official metadata by normalized title AND exact image URL."""
import csv, hashlib, json, sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from data import ROOT, normalize_text

def main():
    raw=ROOT/'data/raw'; keys=defaultdict(list); community=[]
    for split in ['train','val','test']:
        for i,r in enumerate(csv.DictReader((raw/f'{split}.csv').open())):
            row={'community_split':split,'row_index':i,'image_name':Path(r['image_path']).name,'label':int(r['6_way_label']),'text':r['text'],'url':r['original_url']}
            community.append(row); keys[(normalize_text(row['text']),row['url'])].append(row)
    matches=defaultdict(list); files={}
    csv.field_size_limit(1024*1024)
    for split in ['train','val','test']:
        path=raw/f'official-{split}.tsv'
        with path.open() as f:
            reader=csv.DictReader(f,delimiter='\t')
            if 'clean_title' not in reader.fieldnames:raise ValueError(f'{path} is not official TSV')
            count=0
            for r in reader:
                count+=1; key=(normalize_text(r['clean_title']),r['image_url'])
                if key in keys:matches[key].append({'id':r['id'],'split':split,'label':int(r['6_way_label']),'subreddit':r['subreddit'],'binary_label':int(r['2_way_label'])})
        files[split]={'rows':count,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    stats=Counter(); by_split=defaultdict(Counter)
    out=ROOT/'reports/provenance-matches.jsonl'
    with out.open('w') as f:
        for row in community:
            candidates=matches[(normalize_text(row['text']),row['url'])]
            if not candidates:status='unmatched'
            elif len(candidates)>1:status='ambiguous'
            elif candidates[0]['label']!=row['label']:status='label_mismatch'
            elif candidates[0]['split']!=row['community_split']:status='split_mismatch'
            else:status='verified'
            stats[status]+=1;by_split[row['community_split']][status]+=1
            f.write(json.dumps({**row,'status':status,'official_candidates':candidates})+'\n')
    report={'method':'normalized clean_title AND exact image_url; unique official candidate required','community_rows':len(community),'status_counts':dict(stats),'by_community_split':{s:dict(c) for s,c in by_split.items()},'official_files':files,'source':'https://github.com/entitize/Fakeddit','limitation':'Metadata correspondence does not independently validate real-world truth, image bytes, or licensing.'}
    (ROOT/'reports/provenance-audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
