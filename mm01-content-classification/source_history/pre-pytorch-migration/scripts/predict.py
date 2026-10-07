import argparse, os, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from data import ROOT
os.environ.setdefault('TORCH_HOME',str(ROOT/'data/cache/torch'))
os.environ.setdefault('HF_HOME',str(ROOT/'data/cache/huggingface'))
import joblib,numpy as np
import torch
torch.set_num_threads(4)
from features import choose_device,extract_images,extract_clip
parser=argparse.ArgumentParser()
parser.add_argument('--artifact',required=True,help='Trusted local classifier artifact relative to project')
parser.add_argument('--image',required=True)
parser.add_argument('--text',required=True)
args=parser.parse_args()
# Load only artifacts produced by this project; pickle-based artifacts are executable.
bundle=joblib.load(ROOT/args.artifact)
rows=[{'text':args.text,'image':str(Path(args.image).resolve())}]
if bundle['backend']=='clip':image,text=extract_clip(rows,choose_device())
else:
 image=extract_images(rows,choose_device())
 text=bundle['svd'].transform(bundle['vectorizer'].transform([args.text]))
image=bundle['image_scaler'].transform(image);text=bundle['text_scaler'].transform(text)
x={'image':image,'text':text,'fusion':np.concatenate([text,image],axis=1)}[bundle['mode']]
probs=bundle['model'].predict_proba(x)[0]
print({'label':int(bundle['model'].classes_[int(np.argmax(probs))]),'probabilities':probs.tolist(),'scope':'community-subset classifier; numeric labels follow the supplied manifest'})
