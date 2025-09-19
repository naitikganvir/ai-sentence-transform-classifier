# src/train.py
import argparse
import os
import numpy as np
import joblib
import inspect
import transformers
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from utils import load_csv, stratified_split, encode_labels, get_tokenizer, prepare_hf_dataset

# 🔑 Hugging Face Hub imports
from huggingface_hub import HfApi, HfFolder, Repository

HF_REPO_ID = "naitikganvir/sentence-transform-detector"  # your Hugging Face repo


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    p, r, f, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f}


def build_training_args(
    output_dir, train_batch_size, eval_batch_size, epochs, max_logging_steps=50
):
    """
    Build TrainingArguments robustly across old/new transformers versions.
    """
    base_args = dict(
        output_dir=output_dir,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=epochs,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=max_logging_steps,
    )

    sig = inspect.signature(TrainingArguments.__init__)
    params = sig.parameters.keys()
    extra_args = {}

    # Check support for evaluation/save strategies
    if "evaluation_strategy" in params and "save_strategy" in params:
        extra_args.update(
            {
                "evaluation_strategy": "epoch",
                "save_strategy": "epoch",
            }
        )
        if "load_best_model_at_end" in params and "metric_for_best_model" in params:
            extra_args.update(
                {"load_best_model_at_end": True, "metric_for_best_model": "f1"}
            )
    else:
        print("⚠️ transformers version does not support evaluation/save strategies.")

    final_args = {**base_args, **extra_args}
    print(
        "TrainingArguments will be created with these keys:",
        ", ".join(final_args.keys()),
    )
    return TrainingArguments(**final_args)


def main(args):
    try:
        import accelerate  # noqa
    except ImportError:
        print("⚠️ accelerate not installed. Run: pip install accelerate")
    print("Transformers version:", transformers.__version__)

    df = load_csv(args.data_path)
    train_df, val_df, test_df = stratified_split(df)
    le, _, _, _ = encode_labels(train_df, val_df, test_df)
    num_labels = len(le.classes_)
    print("Labels:", list(le.classes_))

    tokenizer = get_tokenizer(args.model_name, max_length=args.max_length)
    train_inputs = prepare_hf_dataset(
        train_df, tokenizer, label_encoder=le, max_length=args.max_length
    )
    val_inputs = prepare_hf_dataset(
        val_df, tokenizer, label_encoder=le, max_length=args.max_length
    )

    train_ds = Dataset.from_dict(train_inputs)
    val_ds = Dataset.from_dict(val_inputs)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=num_labels
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = build_training_args(
        args.output_dir,
        args.train_batch_size,
        args.eval_batch_size,
        args.epochs,
        max_logging_steps=args.logging_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    joblib.dump(le, os.path.join(args.output_dir, "label_encoder.joblib"))
    print("✅ Training complete. Model saved locally:", args.output_dir)

    # --- Push to Hugging Face Hub ---
    print("📤 Uploading to Hugging Face Hub:", HF_REPO_ID)
    api = HfApi()
    token = HfFolder.get_token()
    repo = Repository(local_dir=args.output_dir, clone_from=HF_REPO_ID, use_auth_token=token)

    # Add label encoder to repo
    repo.lfs_track(["*.joblib"])
    repo.git_add()
    repo.git_commit("Add trained model + label encoder")
    repo.git_push()
    print("🚀 Model + label encoder uploaded to:", HF_REPO_ID)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/nlp_sentence_data.csv")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="models/transformer_model")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--logging_steps", type=int, default=50)
    args = parser.parse_args()
    main(args)
