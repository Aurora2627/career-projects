import argparse, csv, json, random, sys, zipfile
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import ROOT, normalize_text, image_hash, check_split_leakage

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-per-class", type=int, default=60)
    parser.add_argument("--eval-per-class", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    raw = ROOT / "data/raw"
    images = ROOT / "data/processed/images"
    images.mkdir(parents=True, exist_ok=True)
    all_rows, audit = {}, {"source": "ams-99/fakeddit_9k community subset", "official_split_provenance": "unverified", "seed": args.seed, "filters": {}, "counts": {}}
    seen_text, seen_image = set(), set()
    with zipfile.ZipFile(raw / "image_cache.zip") as archive:
        members = {Path(n).name: n for n in archive.namelist() if not n.endswith("/")}
        # Preserve held-out examples first; remove overlap from training data.
        for split in ["test", "val", "train"]:
            kept, counts = [], Counter()
            with (raw / f"{split}.csv").open() as handle:
                records = list(csv.DictReader(handle))
            for row in records:
                name = Path(row["image_path"]).name
                member = members.get(name)
                if not member or not normalize_text(row["text"]):
                    counts["missing_image_or_text"] += 1
                    continue
                target = images / name
                if not target.exists():
                    target.write_bytes(archive.read(member))
                try:
                    digest = image_hash(target)
                except Exception:
                    counts["invalid_image"] += 1
                    continue
                text_key = normalize_text(row["text"])
                if text_key in seen_text or digest in seen_image:
                    counts["exact_duplicate"] += 1
                    continue
                label = int(row["6_way_label"])
                if label not in range(6):
                    raise ValueError("Unexpected label")
                seen_text.add(text_key); seen_image.add(digest)
                kept.append({"id": f"{split}:{name}", "text": row["text"], "image": str(target.relative_to(ROOT)), "image_sha256": digest, "label": label, "source_url": row["original_url"]})
            rng = random.Random(args.seed)
            selected = []
            limit = args.train_per_class if split == "train" else args.eval_per_class
            for label in range(6):
                pool = [r for r in kept if r["label"] == label]
                rng.shuffle(pool)
                if len(pool) < 2:
                    raise ValueError(f"Too few examples in {split} class {label}: {len(pool)}")
                if len(pool) < limit:
                    counts[f"class_{label}_below_requested_cap"] = len(pool)
                selected.extend(pool[:limit])
            rng.shuffle(selected)
            all_rows[split] = selected
            audit["filters"][split] = dict(counts)
            audit["counts"][split] = dict(Counter(r["label"] for r in selected))
    check_split_leakage(all_rows)
    for split, rows in all_rows.items():
        (ROOT / f"data/processed/{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    audit["limitations"] = ["Community balanced subset; split derivation and official label mapping not independently verified", "Exact text and decoded-image dedup only; perceptual/semantic duplicates remain possible", "Per-class capped subset with a scarce class; not balanced or representative of platform prevalence"]
    (ROOT / "reports/data-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps(audit, ensure_ascii=False, indent=2))
if __name__ == "__main__": main()
