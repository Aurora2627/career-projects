"""VS Code-friendly Python/PyTorch training with explicit device and optimizer."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='data/processed/phase2-verified')
    p.add_argument('--backend',choices=['clip','resnet'],default='clip')
    p.add_argument('--output',required=True)
    p.add_argument('--optimizer',choices=['adamw','lbfgs'],default='adamw')
    p.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto')
    p.add_argument('--heads',choices=['all','fusion'],default='all')
    p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--patience',type=int,default=10)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--weight-decay',type=float,default=.01)
    p.add_argument('--fresh-features',action='store_true')
    p.add_argument('--fewshot',action='store_true')
    args=p.parse_args()
    if args.optimizer=='adamw':
        if args.fewshot:p.error('AdamW full-data GPU protocol is implemented; strict K-shot currently uses --optimizer lbfgs')
        from accelerated_runner import execute_accelerated
        execute_accelerated(args)
    else:
        if args.device not in ['auto','cpu']:p.error('Historical float64/LBFGS classifiers run on CPU. Use AdamW for GPU training.')
        if args.fewshot and args.backend!='clip':p.error('Historical K-shot protocol uses CLIP')
        from torch_runner import execute
        execute(args)
if __name__=='__main__':main()
