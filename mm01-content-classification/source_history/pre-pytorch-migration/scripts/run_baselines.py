import argparse, hashlib, json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import ROOT, load_manifest, check_split_leakage
os.environ.setdefault("TORCH_HOME", str(ROOT / "data/cache/torch"))
os.environ.setdefault("HF_HOME", str(ROOT / "data/cache/huggingface"))
import numpy as np
import joblib
import torch
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from metrics import classification_metrics
from features import choose_device, extract_images, extract_clip

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["resnet", "clip"], default="resnet")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="runs/baseline-v1")
    args = parser.parse_args()
    out = ROOT / args.output
    if out.exists(): raise ValueError("Output already exists; use a new run directory")
    out.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = choose_device(args.device)
    splits = {s:load_manifest(ROOT / f"data/processed/{s}.jsonl") for s in ["train","val","test"]}
    check_split_leakage(splits)
    rows = sum(splits.values(), [])
    sizes = [len(x) for x in splits.values()]
    cuts = np.cumsum(sizes)[:-1]
    labels = {s:np.array([r["label"] for r in rs]) for s,rs in splits.items()}
    t0 = time.perf_counter()
    cache_key = hashlib.sha256((json.dumps(rows, sort_keys=True)+args.backend).encode()).hexdigest()[:16]
    cache = ROOT / f"data/cache/features-{cache_key}.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        with np.load(cache) as saved: image = saved["image"]; clip_text = saved["text"] if "text" in saved else None
    elif args.backend == "clip":
        image, clip_text = extract_clip(rows, device)
        np.savez_compressed(cache, image=image, text=clip_text)
    else:
        image = extract_images(rows, device)
        clip_text = None
        np.savez_compressed(cache, image=image)
    image = dict(zip(splits, np.split(image,cuts)))
    vectorizer = svd = None
    if clip_text is not None:
        text = dict(zip(splits,np.split(clip_text,cuts)))
    else:
        vectorizer = TfidfVectorizer(max_features=12000, ngram_range=(1,2), min_df=1, sublinear_tf=True)
        train = vectorizer.fit_transform([r["text"] for r in splits["train"]])
        svd = TruncatedSVD(n_components=min(128,train.shape[0]-1,train.shape[1]-1),random_state=args.seed)
        text = {"train":svd.fit_transform(train)}
        for s in ["val","test"]:text[s]=svd.transform(vectorizer.transform([r["text"] for r in splits[s]]))
    # Fit transforms only on training; use validation only for regularization selection.
    scalers = []
    for features in [image,text]:
        scaler=StandardScaler().fit(features["train"])
        scalers.append(scaler)
        for s in features:features[s]=scaler.transform(features[s])
    modes={"text":text,"image":image,"fusion":{s:np.concatenate([text[s],image[s]],axis=1) for s in splits}}
    results={}
    majority=DummyClassifier(strategy="most_frequent").fit(np.zeros((sizes[0],1)),labels["train"])
    majority_prob=majority.predict_proba(np.zeros((sizes[2],1)))
    results["majority"]={"test":classification_metrics(labels["test"],majority.predict(np.zeros((sizes[2],1))),majority_prob)}
    for name, features in modes.items():
        candidates=[]
        for c in [0.01,0.1,1.0]:
            model=LogisticRegression(C=c,max_iter=2000,class_weight="balanced",random_state=args.seed)
            model.fit(features["train"],labels["train"])
            val=classification_metrics(labels["val"],model.predict(features["val"]),model.predict_proba(features["val"]))
            candidates.append((val["macro_f1"],c,model,val))
        score,c,model,val=max(candidates,key=lambda x:x[0])
        prob=model.predict_proba(features["test"]); pred=model.predict(features["test"])
        joblib.dump({"model":model,"mode":name,"backend":args.backend,"vectorizer":vectorizer,"svd":svd,"image_scaler":scalers[0],"text_scaler":scalers[1]},out/f"{name}-classifier.joblib")
        results[name]={"C":c,"val":val,"test":classification_metrics(labels["test"],pred,prob)}
        with (out/f"{name}-predictions.jsonl").open("w") as f:
            for row,y,prediction,ps in zip(splits["test"],labels["test"],pred,prob):
                f.write(json.dumps({"id":row["id"],"label":int(y),"prediction":int(prediction),"probabilities":ps.tolist()})+"\n")
        print(name, "validation Macro-F1",round(score,4),"test Macro-F1",round(results[name]["test"]["macro_f1"],4),flush=True)
    report={"backend":args.backend,"image_encoder":"frozen pretrained ResNet18" if args.backend=="resnet" else "frozen CLIP ViT-B/32","text_encoder":"TF-IDF + train-only SVD" if args.backend=="resnet" else "frozen CLIP text","device":device,"seed":args.seed,"sizes":dict(zip(splits,sizes)),"torch":torch.__version__,"feature_cache_key":cache_key,"elapsed_seconds":time.perf_counter()-t0,"results":results,"scope":"community-subset exploratory experiment, not official benchmark"}
    (out/"metrics.json").write_text(json.dumps(report,indent=2))
    lines=["# 初步基线实验", "", "社区 Fakeddit 子集；不代表官方基准或真实审核效果。", "", f"设备：{device}；编码器：{report['image_encoder']}；文本：{report['text_encoder']}。", "", "| 方法 | 测试 Accuracy | 测试 Macro-F1 |", "|---|---:|---:|"]
    for name,result in results.items():
        m=result["test"]; lines.append(f"| {name} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} |")
    lines.extend(["", "C 仅按验证集选择，测试集仅最终评估。所有单模态和融合使用相同样本。", "", "限制：小规模按类限额社区子集，稀缺类不足上限、官方划分来源未核实、尚未做近重复与领域偏差检查、编码器冻结、未做多种子置信区间。"])
    (out/"summary.md").write_text("\n".join(lines)+"\n")
if __name__=="__main__":main()
