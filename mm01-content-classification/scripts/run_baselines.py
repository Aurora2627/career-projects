"""Run phase-one Python/PyTorch baselines on preserved fixed manifests."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from torch_runner import execute

def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=['resnet','clip'],default='resnet')
    p.add_argument('--data',default='data/processed/phase1');p.add_argument('--output',required=True)
    args=p.parse_args();args.fewshot=False;execute(args)
if __name__=='__main__':main()
