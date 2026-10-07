"""Source-snapshotted experiment with MPS feature extraction and float32 AdamW heads."""
import hashlib,json,os,time
import torch
from data import ROOT,load_manifest,check_split_leakage
from features import choose_device,extract_clip,extract_images
from experiment import snapshot_run
from metrics import classification_metrics
from torch_models import fit_text,text_features,fit_standardizer,standardize
from accelerated_training import train_head,load_head

def execute_accelerated(args):
    os.environ.setdefault('TORCH_HOME',str(ROOT/'data/cache/torch'))
    os.environ.setdefault('HF_HOME',str(ROOT/'data/cache/huggingface'))
    os.environ.setdefault('HF_HUB_OFFLINE','1');torch.set_num_threads(4)
    if args.epochs<1 or args.patience<1:raise ValueError('epochs and patience must be positive')
    device=choose_device(args.device)
    if device not in ['cpu','cuda','mps']:raise ValueError('Unsupported device')
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=False)
    paths={s:ROOT/args.data/f'{s}.jsonl' for s in ['train','val','test']}
    config={**vars(args),'framework':'Python/PyTorch','encoder_device':device,'classifier_device':device,'preprocessing_device':'cpu','encoder':'frozen CLIP ViT-B/32' if args.backend=='clip' else 'frozen ResNet18 + TF-IDF/SVD','classifier_dtype':'float32','optimizer':'torch.optim.AdamW','batch_size':64,'seed':42,'validation_selection':'best epoch by validation macro-F1; no test selection','scope':'fixed community subset; already-used test set; exploratory model comparison','launch_environment':'VS Code terminal/debugger (caller must launch there)'}
    snapshot_run(out,config,paths.values())
    started=time.perf_counter();splits={s:load_manifest(p) for s,p in paths.items()};check_split_leakage(splits)
    rows=sum(splits.values(),[])
    key=hashlib.sha256((json.dumps(rows,sort_keys=True)+args.backend+torch.__version__+device).encode()).hexdigest()[:16]
    cache=ROOT/f'data/cache/pytorch/accelerated-{key}.pt';cache.parent.mkdir(parents=True,exist_ok=True)
    feature_started=time.perf_counter();used_cache=cache.exists() and not args.fresh_features
    if used_cache:features=torch.load(cache,map_location='cpu',weights_only=True)
    else:
        print('Extracting',len(rows),'frozen',args.backend,'examples on',device,flush=True)
        if args.backend=='clip':
            image,text=extract_clip(rows,device);features={'image':torch.from_numpy(image),'text':torch.from_numpy(text)}
        else:features={'image':torch.from_numpy(extract_images(rows,device))}
        torch.save(features,cache)
    if device=='mps':torch.mps.synchronize()
    feature_seconds=time.perf_counter()-feature_started
    print('Feature extraction ready; heads will train on',device,flush=True)
    sizes=[len(rs) for rs in splits.values()]
    image=dict(zip(splits,features['image'].double().split(sizes)));text_state=None
    if args.backend=='clip':text=dict(zip(splits,features['text'].double().split(sizes)))
    else:
        text_state,train=fit_text([r['text'] for r in splits['train']]);text={'train':train}
        for s in ['val','test']:text[s]=text_features([r['text'] for r in splits[s]],text_state)
    scalers=[fit_standardizer(base['train']) for base in [text,image]]
    text,image=[{s:standardize(v,scaler).float() for s,v in base.items()} for base,scaler in zip([text,image],scalers)]
    modes={'text':text,'image':image,'fusion':{s:torch.cat([text[s],image[s]],dim=1) for s in splits}}
    labels={s:torch.tensor([r['label'] for r in rs],dtype=torch.long) for s,rs in splits.items()}
    results={};training_started=time.perf_counter()
    with (out/'training-log.jsonl').open('w') as logfile:
        def log(entry):
            logfile.write(json.dumps(entry)+'\n');logfile.flush()
            if entry['epoch']==1 or entry['epoch']%10==0:print(entry['head'],'epoch',entry['epoch'],'validation Macro-F1',round(entry['val_macro_f1'],4),'device',entry['parameter_device'],flush=True)
        tasks=[('text-linear','text',0),('image-linear','image',0),('fusion-linear','fusion',0),('fusion-mlp','fusion',128)] if args.heads=='all' else [('fusion-linear','fusion',0)]
        for name,mode,hidden in tasks:
            x=modes[mode]
            model,info=train_head(x['train'],labels['train'],x['val'],labels['val'],device,hidden,args.epochs,args.lr,args.weight_decay,patience=args.patience,log=log,tag=name)
            with torch.inference_mode():prob=model(x['test'].to(device)).softmax(1).cpu()
            pred=prob.argmax(1);test=classification_metrics(labels['test'],pred,prob)
            result={**info,'test':test};results[name]=result
            bundle={'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'input_dim':x['train'].shape[1],'hidden_dim':hidden,'dtype':'float32','mode':mode,'backend':args.backend,'scalers':scalers,'text_state':text_state,'seed':42,'training_device':device,'best_epoch':info['best_epoch']}
            artifact=out/f'{name}.pt';torch.save(bundle,artifact)
            restored=load_head(torch.load(artifact,map_location='cpu',weights_only=True),device)
            with torch.inference_mode():torch.testing.assert_close(restored(x['test'].to(device)),model(x['test'].to(device)),rtol=0,atol=0)
            cpu=load_head(bundle,'cpu')
            with torch.inference_mode():cpu_prob=cpu(x['test']).softmax(1)
            max_delta=float((cpu_prob-prob).abs().max());result['checkpoint_same_device_exact']='passed';result['cpu_gpu_max_probability_difference']=max_delta
            result['cpu_gpu_prediction_agreement']=float((cpu_prob.argmax(1)==pred).float().mean())
            with (out/f'{name}-predictions.jsonl').open('w') as f:
                for row,p,ps in zip(splits['test'],pred,prob):f.write(json.dumps({'id':row['id'],'label':row['label'],'prediction':int(p),'probabilities':ps.tolist()})+'\n')
            print(name,'test Macro-F1',round(test['macro_f1'],4),'saved',artifact.name,flush=True)
    if device=='mps':torch.mps.synchronize()
    report={'config':config,'sizes':dict(zip(splits,sizes)),'results':results,'feature_cache_used':used_cache,'feature_seconds':feature_seconds,'training_seconds':time.perf_counter()-training_started,'elapsed_seconds':time.perf_counter()-started,'feature_cache':str(cache.relative_to(ROOT)),'feature_sha256':hashlib.sha256(cache.read_bytes()).hexdigest()}
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    lines=['# VS Code 中的 MPS/AdamW 图文分类实验','',f'编码器设备：{device}；分类器设备：{device}；分类器 Float32；优化器 AdamW。',f'固定划分数量：{report["sizes"]}。训练轮次仅由验证集早停选择。','','| 模型 | 最佳轮次 | 验证 Macro-F1 | 测试 Macro-F1 |','|---|---:|---:|---:|']
    for name,r in results.items():lines.append(f"| {name} | {r['best_epoch']} | {r['val']['macro_f1']:.4f} | {r['test']['macro_f1']:.4f} |")
    lines+=['','所有源代码、VS Code 配置、环境、输入清单和哈希已在启动前保存；.pt 权重重新加载后与同设备模型输出逐值一致。CPU/GPU 差异另存 metrics.json，不能要求跨设备位级一致。','本次是 AdamW/MLP 与历史 LBFGS 的不同训练协议，分数差异不能归因于 GPU 加速。没有测量相对 CPU 的加速比。编码器冻结；单个随机种子、稀缺类、来源偏差和重复使用测试集的限制仍存在。']
    (out/'summary.md').write_text('\n'.join(lines)+'\n');print('ACCELERATED_RUN_FINISHED',flush=True)
