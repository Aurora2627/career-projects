from pathlib import Path
import shutil,json
root=Path('outputs/career-projects/mm01-content-classification')
history=root/'source_history/pre-pytorch-migration'
for folder in ['src','scripts','tests']:
 for p in (root/folder).rglob('*.py'):
  if '__pycache__' in p.parts:continue
  dst=history/p.relative_to(root);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
(history/'README.md').write_text('仅保存迁移前的 Python 源码，供追溯。旧实验数据、模型与评测输出不保留。这些文件不属于当前训练入口；新增开发只使用 PyTorch。\n')
phase2=root/'scripts/run_phase2.py'
s=phase2.read_text();start=s.index('def dhash');end=s.index('\ndef main():')
prep='''"""Python image deduplication and official metadata filtering, no model training."""
import csv, json, random
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from data import ROOT, normalize_text, image_hash, check_split_leakage

'''+s[start:end]+'\n'
(root/'src/prepare_phase2.py').write_text(prep)
for s,d in [('','phase1'),('phase2','phase2-exploratory')]:
 dest=root/'data/processed'/d;dest.mkdir(exist_ok=True)
 for split in ['train','val','test']:
  src=root/'data/processed'/s/f'{split}.jsonl';shutil.copy2(src,dest/f'{split}.jsonl')
# Capture hashes of preserved scientific inputs before replacing old outputs.
import hashlib
old_dirs=sorted(p.name for p in (root/'runs').iterdir() if p.is_dir())
(root/'reports/migration-plan.json').write_text(json.dumps({'old_run_directories':old_dirs,'datasets':{d:{s:hashlib.sha256((root/f'data/processed/{d}/{s}.jsonl').read_bytes()).hexdigest() for s in ['train','val','test']} for d in ['phase1','phase2-exploratory','phase2-verified']},'cleanup_scope':'old run artifacts, old derived npz feature caches and obsolete reports; raw data, images, fixed manifests, pretrained encoders and Python source history retained'},indent=2))
