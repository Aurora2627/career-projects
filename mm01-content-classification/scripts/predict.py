"""Predict using a saved, trusted local PyTorch experiment checkpoint."""
import argparse,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from data import ROOT
os.environ.setdefault('TORCH_HOME',str(ROOT/'data/cache/torch'));os.environ.setdefault('HF_HOME',str(ROOT/'data/cache/huggingface'));os.environ.setdefault('HF_HUB_OFFLINE','1')
import torch
from features import choose_device,extract_clip,extract_images
from torch_models import text_features,standardize

def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact',required=True);p.add_argument('--image',required=True);p.add_argument('--text',required=True)
    args=p.parse_args();torch.set_num_threads(4)
    bundle=torch.load(ROOT/args.artifact,map_location='cpu',weights_only=True)
    rows=[{'text':args.text,'image':str(Path(args.image).resolve())}]
    if bundle['backend']=='clip':
        image,text=extract_clip(rows,choose_device());text=torch.from_numpy(text).double()
    else:image=extract_images(rows,choose_device());text=text_features([args.text],bundle['text_state'])
    image=torch.from_numpy(image).double();text=standardize(text,bundle['scalers'][0]);image=standardize(image,bundle['scalers'][1])
    x={'text':text,'image':image,'fusion':torch.cat([text,image],dim=1)}[bundle['mode']]
    model=torch.nn.Linear(bundle['input_dim'],6,dtype=torch.float64);model.load_state_dict(bundle['state_dict']);model.eval()
    with torch.inference_mode():prob=model(x).softmax(1)[0]
    print({'label':int(prob.argmax()),'probabilities':prob.tolist(),'scope':'community-subset numeric-label classifier'})
if __name__=='__main__':main()
