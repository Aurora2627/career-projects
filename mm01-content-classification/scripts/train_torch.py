"""Unified Python/PyTorch entry point for frozen-feature classification experiments."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from torch_runner import execute

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='data/processed/phase2-verified')
    p.add_argument('--backend',choices=['clip','resnet'],default='clip')
    p.add_argument('--output',required=True)
    p.add_argument('--fewshot',action='store_true')
    args=p.parse_args()
    if args.fewshot and args.backend!='clip':p.error('Few-shot protocol currently uses CLIP features')
    execute(args)
if __name__=='__main__':main()
