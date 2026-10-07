"""Frozen CLIP features with PyTorch classifiers and validation-only early stopping."""
import argparse, copy, hashlib, json, os, random, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from data import ROOT, load_manifest, check_split_leakage
os.environ.setdefault('HF_HOME',str(ROOT/'data/cache/huggingface'))
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from features import choose_device, extract_clip
from experiment import snapshot_run

def metric(y,pred):
    cm=torch.bincount(y.cpu()*6+pred.cpu(),minlength=36).reshape(6,6)
    tp=cm.diag().float(); denom=cm.sum(0)+cm.sum(1)
    f1=torch.where(denom>0,2*tp/denom,0)
    return {'accuracy':float(tp.sum()/cm.sum()),'macro_f1':float(f1.mean()),'per_class_f1':f1.tolist(),'confusion_matrix':cm.tolist()}

def build_head(dim,hidden):
    return nn.Sequential(nn.Linear(dim,hidden),nn.ReLU(),nn.Dropout(.2),nn.Linear(hidden,6)) if hidden else nn.Linear(dim,6)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='runs/torch-v1')
    parser.add_argument('--epochs',type=int,default=60)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--device',default='auto')
    args=parser.parse_args(); out=ROOT/args.output
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    device=choose_device(args.device)
    paths={s:ROOT/f'data/processed/phase2-verified/{s}.jsonl' for s in ['train','val','test']}
    config={**vars(args),'framework':'PyTorch','device':device,'encoder':'frozen CLIP ViT-B/32','optimizer':'AdamW','lr':.001,'weight_decay':.01,'batch_size':64,'patience':10,'selection':'highest validation macro-F1; earliest epoch wins ties','data':'phase2-verified','heads':['text-linear','image-linear','fusion-linear','fusion-mlp-128'],'note':'Fixed exploratory protocol; same test set as earlier baselines, not an untouched new benchmark.'}
    snapshot_run(out,config,paths.values())
    log=(out/'training-log.jsonl').open('w')
    splits={s:load_manifest(p) for s,p in paths.items()};check_split_leakage(splits)
    rows=sum(splits.values(),[]);cuts=np.cumsum([len(splits['train']),len(splits['val'])])
    key=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()[:16]
    cache=ROOT/f'data/cache/phase2-clip-{key}.npz'
    if cache.exists():
        with np.load(cache) as f:image,text=f['image'],f['text']
    else:
        image,text=extract_clip(rows,device);np.savez_compressed(cache,image=image,text=text)
    image=dict(zip(splits,np.split(image,cuts)));text=dict(zip(splits,np.split(text,cuts)))
    y={s:torch.tensor([r['label'] for r in rs],dtype=torch.long) for s,rs in splits.items()}
    transformed=[];scalers=[]
    for base in [text,image]:
        train=torch.tensor(base['train'],dtype=torch.float32)
        mean=train.mean(0);std=train.std(0,correction=0);std=torch.where(std<1e-6,torch.ones_like(std),std)
        scalers.append({'mean':mean,'std':std})
        transformed.append({s:(torch.tensor(v,dtype=torch.float32)-mean)/std for s,v in base.items()})
    text,image=transformed
    modes={'text':text,'image':image,'fusion':{s:torch.cat([text[s],image[s]],dim=1) for s in splits}}
    counts=torch.bincount(y['train'],minlength=6).float();weights=len(y['train'])/(6*counts)
    results={};started=time.time()
    for name,mode,hidden in [('text-linear','text',0),('image-linear','image',0),('fusion-linear','fusion',0),('fusion-mlp-128','fusion',128)]:
        torch.manual_seed(args.seed)
        x=modes[mode];model=build_head(x['train'].shape[1],hidden).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        loss_fn=nn.CrossEntropyLoss(weight=weights.to(device))
        loader=DataLoader(TensorDataset(x['train'],y['train']),batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(args.seed))
        best_score=-1.;best_state=None;stale=0;best_epoch=0
        for epoch in range(1,args.epochs+1):
            model.train();total=0
            for bx,by in loader:
                optimizer.zero_grad(set_to_none=True);loss=loss_fn(model(bx.to(device)),by.to(device));loss.backward();optimizer.step();total+=loss.item()*len(by)
            model.eval()
            with torch.inference_mode(): val=metric(y['val'],model(x['val'].to(device)).argmax(1))
            log.write(json.dumps({'head':name,'epoch':epoch,'weighted_loss_batch_average':total/len(y['train']),'val_macro_f1':val['macro_f1']})+'\n');log.flush()
            if val['macro_f1']>best_score:
                best_score=val['macro_f1'];best_state=copy.deepcopy(model.state_dict());best_epoch=epoch;stale=0
            else:stale+=1
            if stale>=10:break
        model.load_state_dict(best_state);model.eval()
        with torch.inference_mode(): prob=model(x['test'].to(device)).softmax(1).cpu();pred=prob.argmax(1)
        test=metric(y['test'],pred);results[name]={'best_epoch':best_epoch,'val_macro_f1':best_score,'test':test}
        torch.save({'state_dict':{k:v.cpu() for k,v in best_state.items()},'input_dim':x['train'].shape[1],'hidden_dim':hidden,'mode':mode,'scalers':scalers,'seed':args.seed,'encoder':'openai/clip-vit-base-patch32'},out/f'{name}.pt')
        with (out/f'{name}-predictions.jsonl').open('w') as f:
            for row,p,ps in zip(splits['test'],pred,prob):f.write(json.dumps({'id':row['id'],'label':row['label'],'prediction':int(p),'probabilities':ps.tolist()})+'\n')
        # Check that a separately reconstructed head reloads the saved checkpoint correctly.
        saved=torch.load(out/f'{name}.pt',map_location='cpu',weights_only=True)
        restored=build_head(saved['input_dim'],saved['hidden_dim']);restored.load_state_dict(saved['state_dict']);restored.eval()
        with torch.inference_mode():assert torch.allclose(restored(x['test'][:2]),model(x['test'][:2].to(device)).cpu(),atol=1e-4)
        print(name,'best epoch',best_epoch,'test Macro-F1',round(test['macro_f1'],4),flush=True)
    log.close()
    (out/'metrics.json').write_text(json.dumps({'config':config,'results':results,'elapsed_training_seconds':time.time()-started,'checkpoint_reload':'passed'},indent=2))
    lines=['# PyTorch 分类头实验','','CLIP 编码器冻结；分类头使用 torch.nn，损失为加权交叉熵，AdamW 更新参数。按验证集 Macro-F1 早停。','','| 分类头 | 最佳 epoch | 验证 Macro-F1 | 测试 Macro-F1 |','|---|---:|---:|---:|---:|']
    for name,r in results.items():lines.append(f"| {name} | {r['best_epoch']} | {r['val_macro_f1']:.4f} | {r['test']['macro_f1']:.4f} |")
    lines+=['','本次仅一个训练种子；同一测试集已有历史结果，不能声称全新盲测或稳定提升。','每个实验目录保存 source_snapshot、配置、源代码和数据 SHA256、环境信息、逐轮日志、.pt 权重及预测结果。','检查：保存的 4 个分类头均重新加载并与内存模型输出核对通过。']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
