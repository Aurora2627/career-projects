"""VS Code-compatible prediction for both historical and GPU-trained PyTorch heads."""
import argparse,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from data import ROOT
os.environ.setdefault('TORCH_HOME',str(ROOT/'data/cache/torch'));os.environ.setdefault('HF_HOME',str(ROOT/'data/cache/huggingface'));os.environ.setdefault('HF_HUB_OFFLINE','1')
import torch
from features import choose_device,extract_clip,extract_images
from torch_models import text_features,standardize
from accelerated_training import load_head

def predict(artifact,image_path,text_input,requested_device='auto'):
    bundle=torch.load(ROOT/artifact,map_location='cpu',weights_only=True)
    device=choose_device(requested_device)
    rows=[{'text':text_input,'image':str(Path(image_path).resolve())}]
    if bundle['backend']=='clip':
        image,text=extract_clip(rows,device);text=torch.from_numpy(text).double()
    else:image=extract_images(rows,device);text=text_features([text_input],bundle['text_state'])
    image=torch.from_numpy(image).double();text=standardize(text,bundle['scalers'][0]);image=standardize(image,bundle['scalers'][1])
    x={'text':text,'image':image,'fusion':torch.cat([text,image],dim=1)}[bundle['mode']]
    classifier_device='cpu' if bundle.get('dtype')=='float64' else device
    dtype=torch.float64 if bundle.get('dtype')=='float64' else torch.float32
    model=load_head(bundle,classifier_device)
    with torch.inference_mode():prob=model(x.to(device=classifier_device,dtype=dtype)).softmax(1)[0].cpu()
    return {'label':int(prob.argmax()),'probabilities':prob.tolist(),'encoder_device':device,'classifier_device':classifier_device,'scope':'community-subset numeric-label classifier'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact',required=True);p.add_argument('--image',required=True);p.add_argument('--text',required=True);p.add_argument('--device',default='auto',choices=['auto','cpu','mps','cuda'])
    args=p.parse_args();torch.set_num_threads(4);print(predict(args.artifact,args.image,args.text,args.device))
if __name__=='__main__':main()
