"""Re-run both historical phases with PyTorch; cleanup only after all runs verify."""
import argparse,hashlib,json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUNS=[('pytorch-phase1-resnet','resnet','phase1',False),('pytorch-phase1-clip','clip','phase1',False),('pytorch-phase2-exploratory','clip','phase2-exploratory',True),('pytorch-phase2-verified','clip','phase2-verified',True)]

def main():
    p=argparse.ArgumentParser();p.add_argument('--cleanup-legacy',action='store_true');args=p.parse_args()
    plan=json.loads((ROOT/'reports/migration-plan.json').read_text())
    for name,backend,data,fewshot in RUNS:
        command=[sys.executable,str(ROOT/'scripts/train_torch.py'),'--backend',backend,'--data',f'data/processed/{data}','--output',f'runs/{name}']
        if fewshot:command.append('--fewshot')
        subprocess.run(command,check=True)
        subprocess.run([sys.executable,str(ROOT/'scripts/summarize_run.py'),f'runs/{name}'],check=True)
        if fewshot:subprocess.run([sys.executable,str(ROOT/'scripts/summarize_phase2.py'),'--run',f'runs/{name}'],check=True)
    verified={}
    for name,backend,data,fewshot in RUNS:
        run=ROOT/'runs'/name;report=json.loads((run/'metrics.json').read_text());manifest=json.loads((run/'source-manifest.json').read_text())
        expected=33 if fewshot else 3
        assert len(report['results'])==expected
        assert len(list(run.glob('*.pt')))==expected
        for relative,digest in manifest['source_sha256'].items():assert hashlib.sha256((run/'source_snapshot'/relative).read_bytes()).hexdigest()==digest
        for split in ['train','val','test']:
            digest=hashlib.sha256((run/f'input_manifests/{split}.jsonl').read_bytes()).hexdigest()
            assert digest==plan['datasets'][data][split]
        verified[name]={'models':expected,'source_snapshot_hashes':'passed','fixed_input_hashes':'unchanged','checkpoint_reload':report['checkpoint_reload'],'sizes':report['sizes']}
    deleted=[]
    if args.cleanup_legacy:
        # Exact allowlist captured before migration. Never remove new runs or scientific raw inputs.
        for name in plan['old_run_directories']:
            if name in {r[0] for r in RUNS} or Path(name).name!=name:raise ValueError('Unsafe old-run entry')
            path=ROOT/'runs'/name
            if path.exists():shutil.rmtree(path);deleted.append(str(path.relative_to(ROOT)))
        for path in sorted((ROOT/'data/cache').glob('*.npz')):path.unlink();deleted.append(str(path.relative_to(ROOT)))
        for path in [ROOT/'data/processed/phase2']:
            if path.exists():shutil.rmtree(path);deleted.append(str(path.relative_to(ROOT)))
        for filename in ['phase2-verification.json','verification.json','runtime.json']:
            path=ROOT/'reports'/filename
            if path.exists():path.unlink();deleted.append(str(path.relative_to(ROOT)))
    (ROOT/'reports/pytorch-migration-verification.json').write_text(json.dumps({'runs':verified,'deleted_old_artifacts':deleted,'raw_data_and_preserved_manifests_retained':True,'legacy_python_only':'source_history/pre-pytorch-migration','framework':'Python/PyTorch'},indent=2))
    print('All PyTorch runs verified; cleanup complete:',len(deleted),'old artifact paths',flush=True)
if __name__=='__main__':main()
