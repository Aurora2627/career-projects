import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from data import ROOT
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'data/cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--run',default='runs/pytorch-phase2-verified'); args=parser.parse_args()
    out=ROOT/args.run; report=json.loads((out/'metrics.json').read_text()); results=report['results']
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for m in ['text','image','fusion']:
        xs=report['exact_shots_available']; groups=[[r['test']['macro_f1'] for r in results if r['shots']==k and r['mode']==m] for k in xs]
        axes[0].errorbar(xs,[np.mean(v) for v in groups],yerr=[np.std(v,ddof=1) for v in groups],marker='o',capsize=4,label=m)
    axes[0].set(xlabel='Training examples per class',ylabel='Test Macro-F1',title='True K-shot: mean ± sample SD (5 seeds)',xticks=xs,ylim=(0,1)); axes[0].legend()
    full=[r for r in results if r['shots'] is None]
    axes[1].bar([r['mode'] for r in full],[r['test']['macro_f1'] for r in full],color=['#5687ad','#76b1a0','#dd9860'])
    axes[1].set(ylabel='Test Macro-F1',title='Full capped training set (one seed)',ylim=(0,1))
    for i,r in enumerate(full):axes[1].text(i,r['test']['macro_f1']+.02,f"{r['test']['macro_f1']:.4f}",ha='center')
    fig.savefig(out/'learning-curve.png',dpi=180);plt.close(fig)
if __name__=='__main__':main()
