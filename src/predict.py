# src/predict.py
import os, joblib, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

def load_model(model_dir="naitikganvir/sentence-transform-detector", device=None, hf_token=None):
    """
    Load model either from local folder or Hugging Face repo.
    hf_token: Hugging Face token string for private models
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pass token when loading from HF
    use_auth = {"use_auth_token": hf_token} if hf_token else {}

    model = AutoModelForSequenceClassification.from_pretrained(model_dir, **use_auth)
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, **use_auth)

    # Load label encoder: either from local or HF repo
    label_path_local = os.path.join("models/transformer_model", "label_encoder.joblib")
    if os.path.exists(label_path_local):
        le = joblib.load(label_path_local)
    else:
        # Try to download from HF repo
        from huggingface_hub import hf_hub_download
        le_file = hf_hub_download(
            repo_id=model_dir,
            filename="label_encoder.joblib",
            token=hf_token
        )
        le = joblib.load(le_file)

    return model, tokenizer, le, device


def predict(sentence, model, tokenizer, label_encoder, device, return_attention=False):
    model.eval()
    inputs = tokenizer([sentence], return_tensors="pt", truncation=True, padding=True)
    inputs = {k:v.to(device) for k,v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=return_attention)
        probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        label = label_encoder.inverse_transform([pred_idx])[0]

    attn = None
    if return_attention:
        last_layer = outputs.attentions[-1][0].cpu().numpy()
        cls_to_tokens = last_layer[:, 0, :].mean(axis=0)
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        attn = list(zip(tokens, cls_to_tokens.tolist()))

    return {"label": label, "confidence": float(probs[pred_idx]), "probs": probs.tolist(), "attention": attn}
