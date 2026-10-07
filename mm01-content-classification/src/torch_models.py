"""PyTorch TF-IDF/SVD, standardization and regularized softmax classification."""
import math,re
from collections import Counter
import torch
from torch import nn
TOKEN=re.compile(r'(?u)\b\w\w+\b')

def tokens(text):
    words=TOKEN.findall(text.lower())
    return words+[' '.join(p) for p in zip(words,words[1:])]

def fit_tfidf(texts,max_features=12000):
    counts=Counter();df=Counter()
    for text in texts:
        terms=tokens(text);counts.update(terms);df.update(set(terms))
    vocab=sorted(sorted(counts,key=lambda t:(-counts[t],t))[:max_features])
    idf=torch.tensor([math.log((1+len(texts))/(1+df[t]))+1 for t in vocab],dtype=torch.float64)
    return {'vocabulary':vocab,'idf':idf}

def transform_tfidf(texts,state):
    lookup={t:i for i,t in enumerate(state['vocabulary'])}
    x=torch.zeros((len(texts),len(lookup)),dtype=torch.float64)
    for row,text in enumerate(texts):
        for term,count in Counter(tokens(text)).items():
            if term in lookup:x[row,lookup[term]]=1+math.log(count)
    x*=state['idf'];return torch.nn.functional.normalize(x,p=2,dim=1)

def fit_text(texts,seed=42):
    state=fit_tfidf(texts);x=transform_tfidf(texts,state)
    dim=min(128,x.shape[0]-1,x.shape[1]-1)
    if dim<1:raise ValueError('Insufficient training data for SVD')
    # Exact, deterministic train-only SVD replaces the historical randomized sklearn SVD.
    _,_,vh=torch.linalg.svd(x,full_matrices=False);state['components']=vh[:dim].contiguous()
    return state,x@state['components'].T

def text_features(texts,state):return transform_tfidf(texts,state)@state['components'].T

def fit_standardizer(x):
    mean=x.mean(0);std=x.std(0,correction=0)
    return {'mean':mean,'std':torch.where(std<1e-12,torch.ones_like(std),std)}

def standardize(x,state):return (x-state['mean'])/state['std']

def build_head(dim,hidden=0):
    return nn.Sequential(nn.Linear(dim,hidden),nn.ReLU(),nn.Dropout(.2),nn.Linear(hidden,6)) if hidden else nn.Linear(dim,6)

def fit_softmax(x,y,c,seed,log=None,tag=''):
    torch.manual_seed(seed);model=nn.Linear(x.shape[1],6,dtype=torch.float64)
    nn.init.zeros_(model.weight);nn.init.zeros_(model.bias)
    counts=torch.bincount(y,minlength=6).double()
    if torch.any(counts==0):raise ValueError('Every class must have at least one training example')
    weights=len(y)/(6*counts);iterations=0;last_loss=None
    # Weighted mean CE + ||W||²/(2*C*N), bias unpenalized.
    # This specifies the multiclass regularized objective explicitly.
    optimizer=torch.optim.LBFGS(model.parameters(),lr=1,max_iter=200,tolerance_grad=1e-7,tolerance_change=1e-10,line_search_fn='strong_wolfe')
    def closure():
        nonlocal iterations,last_loss
        optimizer.zero_grad();loss=nn.functional.cross_entropy(model(x),y,weight=weights)+model.weight.square().sum()/(2*c*len(y))
        loss.backward();iterations+=1;last_loss=float(loss.detach())
        if log:log({'task':tag,'C':c,'closure_evaluation':iterations,'objective':last_loss})
        return loss
    optimizer.step(closure);model.eval()
    gradient=max(float(p.grad.abs().max()) for p in model.parameters())
    return model,{'closure_evaluations':iterations,'objective':last_loss,'max_abs_gradient':gradient,'max_iterations':200}
