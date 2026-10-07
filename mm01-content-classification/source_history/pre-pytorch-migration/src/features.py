import numpy as np
import torch
from PIL import Image
from data import ROOT

def choose_device(requested="auto"):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable in this process")
    return requested

def extract_images(rows, device, batch_size=16):
    from torchvision.models import resnet18, ResNet18_Weights
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    preprocess = weights.transforms()
    batches = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            tensors = []
            for row in rows[start:start+batch_size]:
                with Image.open(ROOT / row["image"]) as im:
                    tensors.append(preprocess(im.convert("RGB")))
            batches.append(model(torch.stack(tensors).to(device)).cpu().numpy())
    return np.concatenate(batches)

def extract_clip(rows, device, batch_size=8, model_name="openai/clip-vit-base-patch32"):
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(model_name).eval().to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    image_features, text_features = [], []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start+batch_size]
            images = []
            for row in batch:
                with Image.open(ROOT / row["image"]) as image: images.append(image.convert("RGB").copy())
            inputs = processor(text=[r["text"] for r in batch], images=images, padding=True, truncation=True, return_tensors="pt")
            outputs = model(**{k:v.to(device) for k,v in inputs.items()})
            image_features.append(outputs.image_embeds.cpu().numpy())
            text_features.append(outputs.text_embeds.cpu().numpy())
    return np.concatenate(image_features), np.concatenate(text_features)
