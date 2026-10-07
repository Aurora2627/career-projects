"""Versioned data audit, frozen CLIP baselines and true K-shot experiments."""
import argparse, csv, hashlib, json, os, random, sys, time
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from data import ROOT, normalize_text, image_hash, check_split_leakage
os.environ.setdefault('HF_HOME', str(ROOT / 'data/cache/huggingface'))
os.environ.setdefault('TORCH_HOME', str(ROOT / 'data/cache/torch'))
import numpy as np
from PIL import Image
import torch
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from features import choose_device, extract_clip
from metrics import classification_metrics

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

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='runs/phase2')
    parser.add_argument('--require-verified',action='store_true')
    args=parser.parse_args()
    out=ROOT/args.output; out.mkdir(parents=True,exist_ok=False)
    started=time.time(); splits=prepare(out,args.require_verified)
    print('Data prepared', {s:len(v) for s,v in splits.items()},flush=True)
    rows=sum((splits[s] for s in ['train','val','test']),[])
    key=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()[:16]
    cache=ROOT/f'data/cache/phase2-clip-{key}.npz'
    torch.set_num_threads(4); device=choose_device()
    if cache.exists():
        with np.load(cache) as f: image,text=f['image'],f['text']
    else:
        image,text=extract_clip(rows,device)
        np.savez_compressed(cache,image=image,text=text)
    print('CLIP features ready',device,flush=True)
    cuts=np.cumsum([len(splits[s]) for s in ['train','val']])
    images=dict(zip(['train','val','test'],np.split(image,cuts)))
    texts=dict(zip(['train','val','test'],np.split(text,cuts)))
    y={s:np.array([r['label'] for r in rows]) for s,rows in splits.items()}
    counts=Counter(y['train']); shots=[k for k in [2,4,8] if min(counts.values())>=k]
    results=[]; full_predictions={}
    for k in shots+[None]:
        for seed in ([11,22,33,44,55] if k else [42]):
            rng=np.random.default_rng(seed)
            idx=np.concatenate([rng.choice(np.flatnonzero(y['train']==label),k,replace=False) for label in range(6)]) if k else np.arange(len(y['train']))
            fitted=[]; scalers=[]
            # Each K-shot experiment fits preprocessing only on those K-shot training samples.
            for base in [texts,images]:
                scaler=StandardScaler().fit(base['train'][idx])
                scalers.append(scaler)
                fitted.append({s:scaler.transform(v[idx] if s=='train' else v) for s,v in base.items()})
            a,b=fitted
            modes={'text':a,'image':b,'fusion':{s:np.concatenate([a[s],b[s]],axis=1) for s in y}}
            for mode,x in modes.items():
                candidates=[]
                for c in [.01,.1,1.]:
                    model=LogisticRegression(C=c,class_weight='balanced',max_iter=2000,random_state=seed).fit(x['train'],y['train'][idx])
                    score=f1_score(y['val'],model.predict(x['val']),labels=list(range(6)),average='macro',zero_division=0)
                    candidates.append((score,c,model))
                score,c,model=max(candidates,key=lambda v:v[0]); pred=model.predict(x['test'])
                results.append({'shots':k,'seed':seed,'mode':mode,'train_size':len(idx),'C':c,'val_macro_f1':score,'test':classification_metrics(y['test'],pred,model.predict_proba(x['test']))})
                if k is None:
                    full_predictions[mode]=pred
                    joblib.dump({'model':model,'mode':mode,'backend':'clip','vectorizer':None,'svd':None,'text_scaler':scalers[0],'image_scaler':scalers[1]},out/f'{mode}-classifier.joblib')
            print('Finished shots',k,'seed',seed,flush=True)
    # Paired, class-stratified test bootstrap; this measures test-sample uncertainty only.
    rng=np.random.default_rng(2026); deltas=[]
    for _ in range(1000):
        idx=np.concatenate([rng.choice(np.flatnonzero(y['test']==l),sum(y['test']==l),replace=True) for l in range(6)])
        deltas.append(f1_score(y['test'][idx],full_predictions['fusion'][idx],labels=list(range(6)),average='macro',zero_division=0)-f1_score(y['test'][idx],full_predictions['image'][idx],labels=list(range(6)),average='macro',zero_division=0))
    report={'device':device,'elapsed_seconds':time.time()-started,'sizes':{s:len(v) for s,v in splits.items()},'train_counts':{int(k):int(v) for k,v in counts.items()},'seeds':[11,22,33,44,55],'exact_shots_available':shots,'results':results,'fusion_minus_image_test_bootstrap_95pct':np.quantile(deltas,[.025,.975]).tolist(),'provenance_verified':args.require_verified,'scope':'community subset; frozen encoders; dHash candidate filter; not official benchmark'}
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    with (out/'predictions.jsonl').open('w') as f:
        for i,row in enumerate(splits['test']):f.write(json.dumps({**row,'predictions':{m:int(p[i]) for m,p in full_predictions.items()}})+'\n')
    lines=['# 第二阶段：扩大评测与真正 K-shot 实验','','| 每类训练样本 | 文本 Macro-F1 | 图像 Macro-F1 | 融合 Macro-F1 |','|---|---:|---:|---:|']
    for k in shots+[None]:
        cells=[]
        for m in ['text','image','fusion']:
            values=[r['test']['macro_f1'] for r in results if r['shots']==k and r['mode']==m]
            cells.append(f'{np.mean(values):.4f} ± {np.std(values,ddof=1):.4f}' if k else f'{values[0]:.4f}')
        lines.append('| '+('全部（长尾）' if k is None else str(k))+' | '+' | '.join(cells)+' |')
    lines.extend(['','± 为 5 个训练采样种子的样本标准差，不是置信区间。全量只运行一个种子。',f'融合相对图像 Macro-F1 的测试集分层配对 bootstrap 95% 区间：{report["fusion_minus_image_test_bootstrap_95pct"]}。该区间不包含训练集选择、域迁移或标签噪声的不确定性。','','过滤：标准化文本与解码图像精确去重，64 位 dHash 距离 ≤3 的候选图像保留 test→val→train 优先顺序。感知哈希可能误删不同图片，也可能漏掉裁剪或语义重复，尚需人工审核。','','6 类仍保留；类别 4 极少，不能声称类别均衡或充分验证该类。类别数值保持官方标注；来源核验状态见 metrics.json 与 provenance-audit.json。测试集没有用于选择 C 或哈希阈值；阈值在此次实验前固定为 3。'])
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    print(report['fusion_minus_image_test_bootstrap_95pct'],flush=True)
if __name__=='__main__': main()
