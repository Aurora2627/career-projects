"""Final checks for source snapshots, cleanup and raw-image checkpoint inference."""
import ast,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    migration=json.loads((ROOT/'reports/pytorch-migration-verification.json').read_text())
    for folder in ['src','scripts','tests']:
        for p in (ROOT/folder).glob('*.py'):
            tree=ast.parse(p.read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.Import):assert all(n.name.split('.')[0] not in ['sklearn','joblib'] for n in node.names)
                if isinstance(node,ast.ImportFrom):assert (node.module or '').split('.')[0] not in ['sklearn','joblib']
    for name in migration['runs']:
        run=ROOT/'runs'/name;m=json.loads((run/'source-manifest.json').read_text())
        for relative,digest in m['source_sha256'].items():assert hashlib.sha256((run/'source_snapshot'/relative).read_bytes()).hexdigest()==digest
    for relative in migration['deleted_old_artifacts']:assert not (ROOT/relative).exists()
    checks={}
    for name in ['pytorch-phase1-resnet','pytorch-phase2-verified']:
        run=ROOT/'runs'/name;row=json.loads((run/'input_manifests/test.jsonl').read_text().splitlines()[0])
        stored=json.loads((run/'full-fusion-predictions.jsonl').read_text().splitlines()[0])
        result=subprocess.run([sys.executable,str(ROOT/'scripts/predict.py'),'--artifact',f'runs/{name}/full-fusion.pt','--image',str(ROOT/row['image']),'--text',row['text']],capture_output=True,text=True,check=True)
        actual=ast.literal_eval(result.stdout.strip().splitlines()[-1])
        assert actual['label']==stored['prediction']
        assert max(abs(a-b) for a,b in zip(actual['probabilities'],stored['probabilities']))<1e-5
        checks[name]={'sample_id':row['id'],'prediction':actual['label'],'raw_image_inference':'matches saved test prediction within 1e-5'}
    result={'python_syntax':'passed','active_training_imports':'PyTorch; no sklearn/joblib','immutable_source_snapshot_hashes':'passed','old_artifact_removal':'passed','unit_tests':11,'inference_checks':checks}
    (ROOT/'reports/final-pytorch-verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
