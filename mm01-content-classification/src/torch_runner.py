"""Versioned PyTorch baseline and exact K-shot runner; all model artifacts are .pt."""
import hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
from data import ROOT,load_manifest,check_split_leakage
from experiment import snapshot_run
from features import choose_device,extract_clip,extract_images
from metrics import classification_metrics
from torch_models import fit_text,text_features,fit_standardizer,standardize,fit_softmax

SPLITS=['train','val','test']
def execute(args):
    os.environ.setdefault('TORCH_HOME',str(ROOT/'data/cache/torch'))
    os.environ.setdefault('HF_HOME',str(ROOT/'data/cache/huggingface'))
    os.environ.setdefault('HF_HUB_OFFLINE','1')
    torch.set_num_threads(4);torch.manual_seed(42)
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=False)
    paths={s:ROOT/args.data/f'{s}.jsonl' for s in SPLITS}
    config={**vars(args),'framework':'Python/PyTorch','device':choose_device(),'classifier':'torch.nn.Linear(6-way softmax)','objective':'class-balanced weighted mean cross-entropy + L2(weight)/(2*C*N); no bias penalty','optimizer':'torch.optim.LBFGS (strong_wolfe)','C_candidates':[.01,.1,1.0],'selection':'validation macro-F1 only, first C wins ties','fewshot_seeds':[11,22,33,44,55] if args.fewshot else [],'encoder_training':'frozen pretrained encoders','text_svd':'train-only exact torch.linalg.svd; differs from old randomized SVD','scope':'exploratory community subset; reused held-out data, not new blind benchmark'}
    snapshot_run(out,config,paths.values())
    log=(out/'training-log.jsonl').open('w')
    def write_log(entry):log.write(json.dumps(entry)+'\n');log.flush()
    started=time.time();splits={s:load_manifest(p) for s,p in paths.items()};check_split_leakage(splits)
    rows=sum((splits[s] for s in SPLITS),[])
    key=hashlib.sha256((json.dumps(rows,sort_keys=True)+args.backend).encode()).hexdigest()[:16]
    cache=ROOT/f'data/cache/pytorch/features-{key}.pt';cache.parent.mkdir(parents=True,exist_ok=True)
    if cache.exists():
        features=torch.load(cache,map_location='cpu',weights_only=True)
    else:
        print(args.output,'extracting frozen',args.backend,'features',len(rows),flush=True)
        if args.backend=='clip':
            image,text=extract_clip(rows,config['device']);features={'image':torch.from_numpy(image),'text':torch.from_numpy(text)}
        else:features={'image':torch.from_numpy(extract_images(rows,config['device']))}
        features['input_key']=key;torch.save(features,cache)
    ends=[len(splits[s]) for s in SPLITS]
    image=dict(zip(SPLITS,features['image'].double().split(ends)))
    text_state=None
    if args.backend=='clip':text=dict(zip(SPLITS,features['text'].double().split(ends)))
    else:
        text_state,train=fit_text([r['text'] for r in splits['train']]);text={'train':train}
        for s in ['val','test']:text[s]=text_features([r['text'] for r in splits[s]],text_state)
    y={s:torch.tensor([r['label'] for r in splits[s]],dtype=torch.long) for s in SPLITS}
    counts=torch.bincount(y['train'],minlength=6);shots=[k for k in [2,4,8] if int(counts.min())>=k] if args.fewshot else []
    results=[];full_pred={}
    majority=int(counts.argmax());majority_pred=torch.full_like(y['test'],majority);majority_prob=torch.nn.functional.one_hot(majority_pred,num_classes=6).double()
    majority_metrics=classification_metrics(y['test'],majority_pred,majority_prob)
    for k in shots+[None]:
        for seed in ([11,22,33,44,55] if k else [42]):
            # Preserve the previous NumPy selection protocol, while training is entirely PyTorch.
            rng=np.random.default_rng(seed)
            idx=torch.tensor(np.concatenate([rng.choice(np.flatnonzero(y['train'].numpy()==l),k,replace=False) for l in range(6)])) if k else torch.arange(len(y['train']))
            label='full' if k is None else f'shot{k}-seed{seed}'
            sample_ids=[splits['train'][int(i)]['id'] for i in idx]
            (out/f'{label}-training-ids.json').write_text(json.dumps(sample_ids))
            scalers=[fit_standardizer(base['train'][idx]) for base in [text,image]]
            bases=[{s:standardize(v[idx] if s=='train' else v,scaler) for s,v in base.items()} for base,scaler in zip([text,image],scalers)]
            a,b=bases;modes={'text':a,'image':b,'fusion':{s:torch.cat([a[s],b[s]],dim=1) for s in SPLITS}}
            for mode,x in modes.items():
                best=None;candidates=[]
                for c in [.01,.1,1.0]:
                    model,diagnostics=fit_softmax(x['train'],y['train'][idx],c,seed,write_log,f'{label}/{mode}')
                    with torch.inference_mode():vp=model(x['val']).softmax(1);val=classification_metrics(y['val'],vp.argmax(1),vp)
                    candidates.append({'C':c,'val_macro_f1':val['macro_f1'],'optimizer':diagnostics})
                    if best is None or val['macro_f1']>best[0]:best=(val['macro_f1'],c,model,val)
                _,c,model,val=best
                with torch.inference_mode():prob=model(x['test']).softmax(1);pred=prob.argmax(1)
                result={'shots':k,'seed':seed,'mode':mode,'train_size':len(idx),'C':c,'val':val,'val_macro_f1':val['macro_f1'],'candidates':candidates,'test':classification_metrics(y['test'],pred,prob)}
                results.append(result)
                artifact=out/f'{label}-{mode}.pt'
                torch.save({'state_dict':model.state_dict(),'input_dim':x['train'].shape[1],'mode':mode,'backend':args.backend,'text_state':text_state,'scalers':scalers,'seed':seed,'shots':k,'C':c,'feature_key':key,'dtype':'float64'},artifact)
                saved=torch.load(artifact,map_location='cpu',weights_only=True)
                restored=torch.nn.Linear(saved['input_dim'],6,dtype=torch.float64);restored.load_state_dict(saved['state_dict']);restored.eval()
                with torch.inference_mode():torch.testing.assert_close(restored(x['test']),model(x['test']),rtol=0,atol=0)
                with (out/f'{label}-{mode}-predictions.jsonl').open('w') as f:
                    for row,p,ps in zip(splits['test'],pred,prob):f.write(json.dumps({'id':row['id'],'label':row['label'],'prediction':int(p),'probabilities':ps.tolist()})+'\n')
                if k is None:full_pred[mode]=pred
            print(args.output,'finished',label,flush=True)
    interval=None
    if args.fewshot:
        rng=np.random.default_rng(2026);deltas=[]
        for _ in range(1000):
            idx=torch.tensor(np.concatenate([rng.choice(np.flatnonzero(y['test'].numpy()==l),int((y['test']==l).sum()),replace=True) for l in range(6)]))
            deltas.append(classification_metrics(y['test'][idx],full_pred['fusion'][idx])['macro_f1']-classification_metrics(y['test'][idx],full_pred['image'][idx])['macro_f1'])
        interval=torch.quantile(torch.tensor(deltas,dtype=torch.float64),torch.tensor([.025,.975],dtype=torch.float64)).tolist()
    log.close()
    report={'config':config,'backend':args.backend,'sizes':{s:len(rs) for s,rs in splits.items()},'train_counts':counts.tolist(),'exact_shots_available':shots,'results':results,'majority_test':majority_metrics,'fusion_minus_image_test_bootstrap_95pct':interval,'feature_cache':str(cache.relative_to(ROOT)),'feature_sha256':hashlib.sha256(cache.read_bytes()).hexdigest(),'elapsed_seconds':time.time()-started,'checkpoint_reload':'all saved heads exactly match in-memory test logits'}
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    lines=['# PyTorch 重跑结果','','| 每类训练样本 | 文本 Macro-F1 | 图像 Macro-F1 | 融合 Macro-F1 |','|---|---:|---:|---:|']
    for k in shots+[None]:
        cells=[]
        for mode in ['text','image','fusion']:
            values=torch.tensor([r['test']['macro_f1'] for r in results if r['shots']==k and r['mode']==mode],dtype=torch.float64)
            cells.append(f'{float(values.mean()):.4f} ± {float(values.std()):.4f}' if k else f'{float(values[0]):.4f}')
        lines.append('| '+('全部（长尾）' if k is None else str(k))+' | '+' | '.join(cells)+' |')
    lines+=['',f'数据数量：{report["sizes"]}。编码器冻结；分类头、优化器、TF-IDF/SVD（ResNet 路线）、标准化与指标均基于 Python/PyTorch。',f'多数类对照测试 Macro-F1：{majority_metrics["macro_f1"]:.4f}。','原始划分清单保持不变。前后差异不能归因于框架本身；精确 SVD、数值容差可能影响预测。','± 是 5 个训练采样种子的样本标准差。全量仅一个种子；稀缺类、来源偏差与测试集复用仍存在。',f'分层配对测试 bootstrap 95% 区间（融合减图像）：{interval}。仅反映固定模型测试样本不确定性。','源码快照、输入 SHA256、环境、配置、每次采样 ID、优化日志、.pt 模型与预测均保存在此目录。']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    print(args.output,[(r['mode'],round(r['test']['macro_f1'],4)) for r in results if r['shots'] is None],flush=True)
