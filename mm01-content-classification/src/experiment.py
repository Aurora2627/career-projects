"""Capture the exact source and input hashes before an experiment starts."""
import hashlib, json, platform, shutil, sys
from importlib.metadata import distributions
from pathlib import Path
from data import ROOT

def snapshot_run(out, config, manifests=()):
    out=Path(out); destination=out/'source_snapshot'
    destination.mkdir(parents=True,exist_ok=False)
    files={}
    for folder in ['src','scripts','configs','tests']:
        for source in sorted((ROOT/folder).rglob('*')):
            if not source.is_file() or '__pycache__' in source.parts:continue
            relative=source.relative_to(ROOT); target=destination/relative
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
            files[str(relative)]=hashlib.sha256(source.read_bytes()).hexdigest()
    for filename in ['README.md','QUICKSTART.md','requirements-lock.txt']:
        source=ROOT/filename
        if source.exists():shutil.copy2(source,destination/filename)
    manifests=list(manifests)
    inputs={str(Path(p).relative_to(ROOT)):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in manifests}
    inputs_dir=out/'input_manifests';inputs_dir.mkdir(exist_ok=False)
    for source in manifests:shutil.copy2(source,inputs_dir/Path(source).name)
    (out/'experiment-config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False))
    (out/'source-manifest.json').write_text(json.dumps({'source_sha256':files,'input_sha256':inputs},indent=2))
    (out/'environment.json').write_text(json.dumps({'python':sys.version,'platform':platform.platform(),'packages':{d.metadata['Name']:d.version for d in distributions()}},indent=2))
