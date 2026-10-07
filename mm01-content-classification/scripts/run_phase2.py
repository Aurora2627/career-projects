"""Run expanded full-data and five-seed strict K-shot experiments with PyTorch."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from torch_runner import execute
from prepare_phase2 import dhash,prepare

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/processed/phase2-verified');p.add_argument('--output',required=True)
    args=p.parse_args();args.backend='clip';args.fewshot=True;execute(args)
if __name__=='__main__':main()
