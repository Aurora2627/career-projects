"""Float32 PyTorch classification training on CPU, CUDA or Apple MPS."""
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from metrics import classification_metrics
from torch_models import build_head

def train_head(train_x,train_y,val_x,val_y,device,hidden=0,epochs=60,lr=.001,
               weight_decay=.01,batch_size=64,patience=10,seed=42,log=None,tag=''):
    torch.manual_seed(seed)
    if device=='mps':torch.mps.manual_seed(seed)
    model=build_head(train_x.shape[1],hidden).float().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)
    counts=torch.bincount(train_y.cpu(),minlength=6).float()
    if torch.any(counts==0):raise ValueError('Training must include all six classes')
    class_weights=(len(train_y)/(6*counts)).to(device)
    loss_fn=nn.CrossEntropyLoss(weight=class_weights)
    loader=DataLoader(TensorDataset(train_x.float().cpu(),train_y.cpu()),batch_size=batch_size,shuffle=True,generator=torch.Generator().manual_seed(seed))
    vx=val_x.float().to(device);best_score=-1.;best_state=None;stale=0;steps=0
    for epoch in range(1,epochs+1):
        model.train();loss_sum=0.;weight_sum=0.
        for bx,by in loader:
            bx=bx.to(device);by=by.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits=model(bx)
            loss=loss_fn(logits,by)
            if not torch.isfinite(loss).item():raise RuntimeError('Non-finite loss')
            loss.backward()
            optimizer.step()
            batch_weights=float(class_weights[by].sum().detach().cpu())
            loss_sum+=float(loss.detach().cpu())*batch_weights;weight_sum+=batch_weights;steps+=1
        model.eval()
        with torch.inference_mode():
            probabilities=model(vx).softmax(1).cpu()
        val=classification_metrics(val_y,probabilities.argmax(1),probabilities)
        entry={'head':tag,'epoch':epoch,'train_weighted_cross_entropy':loss_sum/weight_sum,'val_macro_f1':val['macro_f1'],'parameter_device':str(next(model.parameters()).device),'dtype':str(next(model.parameters()).dtype),'optimizer_steps':steps}
        if log:log(entry)
        if val['macro_f1']>best_score:
            best_score=val['macro_f1'];best_epoch=epoch
            best_state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
            best_val=val;stale=0
        else:stale+=1
        if stale>=patience:break
    model.load_state_dict(best_state);model.eval()
    return model,{'best_epoch':best_epoch,'completed_epochs':epoch,'val':best_val,'optimizer_steps':steps,'parameter_device':str(next(model.parameters()).device),'dtype':'float32'}

def load_head(bundle,device='cpu'):
    dtype=torch.float64 if bundle.get('dtype')=='float64' else torch.float32
    model=build_head(bundle['input_dim'],bundle.get('hidden_dim',0)).to(dtype=dtype,device=device)
    model.load_state_dict(bundle['state_dict']);model.eval()
    return model
