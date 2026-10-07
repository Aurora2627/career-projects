import hashlib, json, re
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

def normalize_text(text):
    return re.sub(r"\s+", " ", text.casefold()).strip()

def load_manifest(path):
    path = Path(path)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("Empty manifest")
    seen = set()
    for row in rows:
        if row["id"] in seen:
            raise ValueError("Duplicate ID")
        seen.add(row["id"])
        if not isinstance(row["label"], int) or row["label"] not in range(6):
            raise ValueError("Expected integer 6-way label, 0..5")
        if not normalize_text(row["text"]):
            raise ValueError("Empty text")
        if not (ROOT / row["image"]).is_file():
            raise ValueError("Missing image: " + row["image"])
    return rows

def image_hash(path):
    with Image.open(path) as image:
        image = image.convert("RGB")
        return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()

def check_split_leakage(splits):
    owners = {"text": {}, "image": {}}
    for split, rows in splits.items():
        for row in rows:
            keys = {"text": normalize_text(row["text"]), "image": row["image_sha256"]}
            for kind, key in keys.items():
                prior = owners[kind].get(key)
                if prior is not None and prior != split:
                    raise ValueError(f"Cross-split {kind} duplicate: {prior}/{split}")
                owners[kind][key] = split
