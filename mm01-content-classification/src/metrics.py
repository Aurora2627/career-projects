"""Six-class classification metrics computed with PyTorch only."""
import torch

def classification_metrics(y, predictions, probabilities=None):
    y=torch.as_tensor(y,dtype=torch.long).cpu();p=torch.as_tensor(predictions,dtype=torch.long).cpu()
    cm=torch.bincount(y*6+p,minlength=36).reshape(6,6)
    tp=cm.diag().double();support=cm.sum(1);predicted=cm.sum(0)
    precision=tp/predicted.clamp(min=1);recall=tp/support.clamp(min=1)
    f1=2*tp/(support+predicted).clamp(min=1)
    result={'accuracy':float(tp.sum()/len(y)),'macro_f1':float(f1.mean()),'confusion_matrix':cm.tolist(),'per_class':{str(i):{'precision':float(precision[i]),'recall':float(recall[i]),'f1-score':float(f1[i]),'support':int(support[i])} for i in range(6)}}
    if probabilities is not None:
        scores=torch.as_tensor(probabilities).cpu();aps=[]
        for label in range(6):
            order=torch.argsort(scores[:,label],descending=True,stable=True);sorted_scores=scores[order,label]
            positives=(y[order]==label).double();total=positives.sum()
            if total==0:aps.append(0.);continue
            ends=torch.cat([torch.nonzero(sorted_scores[:-1]!=sorted_scores[1:]).flatten(),torch.tensor([len(y)-1])])
            hits=positives.cumsum(0)[ends];precision_at_threshold=hits/(ends+1)
            recall_increase=torch.diff(torch.cat([torch.zeros(1,dtype=hits.dtype),hits]))/total
            aps.append(float((recall_increase*precision_at_threshold).sum()))
        result['macro_average_precision']=sum(aps)/6
    return result
