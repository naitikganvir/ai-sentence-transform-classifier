# src/explain.py
import shap
import torch
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_explainer(model_dir="models/transformer_model"):
    """Load model, tokenizer, and label encoder for SHAP explainability."""
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    le = joblib.load(f"{model_dir}/label_encoder.joblib")

    def f(texts):
        # Ensure we always pass a list of strings
        if isinstance(texts, str):
            texts = [texts]
        enc = tokenizer(list(texts), return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            out = model(**enc)
            return out.logits.numpy()

    explainer = shap.Explainer(f, tokenizer)
    return explainer, tokenizer, le

if __name__ == "__main__":
    explainer, tokenizer, le = load_explainer()

    # Example sentence
    sentence = "The new product will be launched by the company."

    # Compute SHAP values
    shap_values = explainer(sentence)

    # Save explanation to HTML (works outside Jupyter/IPython)
    html = shap.plots.text(shap_values[0], display=False)
    output_path = "shap_explanation.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ SHAP explanation saved to {output_path}. Open it in your browser.")
