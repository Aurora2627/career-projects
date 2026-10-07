"""Detect and actually exercise Apple GPU (MPS); run from VS Code's terminal."""
import argparse,json,platform,sys,time
from pathlib import Path
import torch

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report',default='reports/vscode-mps-check.json');args=parser.parse_args()
    torch.set_num_threads(4)
    report={'python_executable':sys.executable,'python_version':platform.python_version(),'platform':platform.platform(),'torch_version':torch.__version__,'torch_path':torch.__file__,'mps_built':torch.backends.mps.is_built(),'mps_available':torch.backends.mps.is_available(),'actual_gpu_training':'not attempted: MPS unavailable'}
    if report['mps_available']:
        try:
            torch.manual_seed(42);device=torch.device('mps')
            model=torch.nn.Linear(256,128).to(device)
            x=torch.randn(256,256,device=device,dtype=torch.float32);y=torch.arange(256,device=device)%128
            before=model.weight.detach().clone();optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
            torch.mps.synchronize();started=time.perf_counter()
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True);output=model(x)
                loss=torch.nn.functional.cross_entropy(output,y);loss.backward();optimizer.step()
            torch.mps.synchronize()
            assert torch.isfinite(loss).item() and not torch.equal(before,model.weight)
            report.update(actual_gpu_training='passed',tensor_device=str(output.device),parameter_device=str(model.weight.device),gpu_steps=3,elapsed_seconds=time.perf_counter()-started,loss=float(loss.detach().cpu()),note='Small correctness smoke test; not a speed benchmark.')
        except Exception as exc:report.update(actual_gpu_training='failed',error=repr(exc))
    target=Path(args.report);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    print('MPS_CHECK_FINISHED',flush=True)
if __name__=='__main__':main()
