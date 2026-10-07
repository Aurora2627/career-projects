import argparse,csv,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'data/cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument('run',help='Run directory relative to project')
args=parser.parse_args()
run=ROOT/args.run
report=json.loads((run/'metrics.json').read_text())
rows={r['id']:r for r in (json.loads(line) for line in (ROOT/report['config']['data']/'test.jsonl').read_text().splitlines())}
predictions=[json.loads(line) for line in (run/'full-fusion-predictions.jsonl').read_text().splitlines()]
with (run/'error-review.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=['id','text','image','label','prediction','confidence','review_category','notes'])
 writer.writeheader()
 for pred in predictions:
  if pred['label']!=pred['prediction']:
   row=rows[pred['id']]
   writer.writerow({k:row[k] for k in ['id','text','image','label']}|{'prediction':pred['prediction'],'confidence':max(pred['probabilities']),'review_category':'','notes':''})
results={r['mode']:r for r in report['results'] if r['shots'] is None}; names=list(results)
fig,ax=plt.subplots(figsize=(7,4))
values=[results[n]['test']['macro_f1'] for n in names]
ax.bar(names,values,color=['#a0a0a0','#4477aa','#228833','#cc6677'])
for i,v in enumerate(values):ax.text(i,v+.015,f'{v:.3f}',ha='center')
ax.set_ylim(0,1);ax.set_ylabel('Macro-F1');ax.set_title(f"PyTorch exploratory results (test n={report['sizes']['test']})")
fig.tight_layout();fig.savefig(run/'comparison.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(5,5)); matrix=results['fusion']['test']['confusion_matrix']
ax.imshow(matrix,cmap='Blues');ax.set_xticks(range(6));ax.set_yticks(range(6));ax.set_xlabel('Predicted label');ax.set_ylabel('True label');ax.set_title('Fusion confusion matrix')
for i,line in enumerate(matrix):
 for j,v in enumerate(line):ax.text(j,i,str(v),ha='center',va='center')
fig.tight_layout();fig.savefig(run/'confusion-matrix.png',dpi=160);plt.close(fig)
print('Saved comparison charts and error-review.csv')
