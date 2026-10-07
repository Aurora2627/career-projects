"""Check the user's default Python and run a small PyTorch forward/backward step."""
import argparse,json,platform,sys
from pathlib import Path

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report');args=parser.parse_args()
    import pip,torch,torchvision,transformers,numpy,PIL,matplotlib,debugpy
    torch.manual_seed(42);torch.set_num_threads(4)
    model=torch.nn.Linear(4,2);optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    x=torch.randn(8,4);y=torch.arange(8)%2
    before=model.weight.detach().clone()
    loss=torch.nn.functional.cross_entropy(model(x),y);loss.backward();optimizer.step()
    assert not torch.equal(before,model.weight)
    report={'python_executable':sys.executable,'python_version':platform.python_version(),'architecture':platform.machine(),'torch_version':torch.__version__,'torch_path':torch.__file__,'torchvision':torchvision.__version__,'transformers':transformers.__version__,'numpy':numpy.__version__,'pillow':PIL.__version__,'matplotlib':matplotlib.__version__,'debugpy':debugpy.__version__,'pip':pip.__version__,'mps_available':torch.backends.mps.is_available(),'pytorch_forward_backward_optimizer':'passed','loss':float(loss.detach())}
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if args.report:Path(args.report).write_text(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
