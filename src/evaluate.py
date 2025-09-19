# src/evaluate.py
import argparse, joblib, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns, matplotlib.pyplot as plt
from utils import load_csv, stratified_split

def evaluate(model_dir, data_path):
    df = load_csv(data_path)
    _, _, test_df = stratified_split(df)
    le = joblib.load(f"{model_dir}/label_encoder.joblib")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    texts = test_df['Transformed Sentence'].astype(str).tolist()
    enc = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        preds = np.argmax(model(**enc).logits.cpu().numpy(), axis=1)

    y_true = le.transform(test_df['Label'].values)
    report = classification_report(y_true, preds, target_names=le.classes_, zero_division=0, output_dict=True)

    print("Classification Report:")
    for cls, metrics in report.items():
        if isinstance(metrics, dict):
            print(f"{cls}: F1={metrics['f1-score']:.3f}, Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}")

    cm = confusion_matrix(y_true, preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{model_dir}/confusion_matrix.png")
    print(f"📊 Confusion matrix saved to {model_dir}/confusion_matrix.png")

    f1_scores = {cls: report[cls]['f1-score'] for cls in le.classes_}
    worst = min(f1_scores, key=f1_scores.get)
    print(f"⚠️ Weakest class: {worst} (F1={f1_scores[worst]:.3f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="models/transformer_model")
    parser.add_argument("--data_path", type=str, default="data/nlp_sentence_data.csv")
    args = parser.parse_args()
    evaluate(args.model_dir, args.data_path)
