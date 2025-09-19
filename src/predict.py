# src/predict.py
# src/predict.py
import os
import joblib  # make sure this is here
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F


HF_REPO_ID = "naitikganvir/sentence-transform-detector"  # Hugging Face repo

def load_model(model_dir="models/transformer_model", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"📂 Loading local model from {model_dir}")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    label_encoder_path = os.path.join(model_dir, "label_encoder.joblib")
    if not os.path.exists(label_encoder_path):
        raise FileNotFoundError(f"Label encoder not found at {label_encoder_path}")

    le = joblib.load(label_encoder_path)
    return model, tokenizer, le, device



def predict(sentence, model, tokenizer, label_encoder, device, return_attention=False):
    model.eval()
    inputs = tokenizer([sentence], return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=return_attention)
        probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        label = label_encoder.inverse_transform([pred_idx])[0]

    attn = None
    if return_attention:
        last_layer = outputs.attentions[-1][0].cpu().numpy()
        cls_to_tokens = last_layer[:, 0, :].mean(axis=0)
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        attn = list(zip(tokens, cls_to_tokens.tolist()))

    return {
        "label": label,
        "confidence": float(probs[pred_idx]),
        "probs": probs.tolist(),
        "attention": attn,
    }
