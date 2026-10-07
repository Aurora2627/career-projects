"""Verify cached local encoders and original PyTorch checkpoints under system Python."""
import ast,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    checks={}
    for name in ['pytorch-phase1-resnet','pytorch-phase2-verified']:
        run=ROOT/'runs'/name;row=json.loads((run/'input_manifests/test.jsonl').read_text().splitlines()[0]);expected=json.loads((run/'full-fusion-predictions.jsonl').read_text().splitlines()[0])
        result=subprocess.run([sys.executable,str(ROOT/'scripts/predict.py'),'--artifact',f'runs/{name}/full-fusion.pt','--image',str(ROOT/row['image']),'--text',row['text']],capture_output=True,text=True,check=True)
        actual=ast.literal_eval(result.stdout.strip().splitlines()[-1]);assert actual['label']==expected['prediction']
        checks[name]={'prediction_matches':True,'maximum_probability_difference':max(abs(a-b) for a,b in zip(actual['probabilities'],expected['probabilities']))}
    (ROOT/'reports/system-python-prediction.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
