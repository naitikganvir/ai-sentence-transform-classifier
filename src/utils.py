# src/utils.py
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from transformers import AutoTokenizer

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {'Original Sentence','Transformed Sentence','Label'}
    if not expected.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns: {expected}. Found: {df.columns}")
    df = df.dropna(subset=['Transformed Sentence','Label']).reset_index(drop=True)
    return df

def stratified_split(df: pd.DataFrame, label_col='Label', seed=42):
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=seed)
    X = df.index.values
    y = df[label_col].values
    train_idx, rest_idx = next(sss1.split(X, y))
    rest_df = df.iloc[rest_idx].reset_index(drop=True)
    train_df = df.iloc[train_idx].reset_index(drop=True)

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    rest_idx_in_rest, test_idx_in_rest = next(sss2.split(rest_df.index.values, rest_df[label_col].values))
    val_df = rest_df.iloc[rest_idx_in_rest].reset_index(drop=True)
    test_df = rest_df.iloc[test_idx_in_rest].reset_index(drop=True)
    return train_df, val_df, test_df

def encode_labels(train_df, val_df, test_df, label_col='Label'):
    le = LabelEncoder()
    le.fit(train_df[label_col].values)
    y_train = le.transform(train_df[label_col].values)
    y_val = le.transform(val_df[label_col].values)
    y_test = le.transform(test_df[label_col].values)
    return le, y_train, y_val, y_test

def get_tokenizer(model_name="bert-base-uncased", max_length=128):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.model_max_length = max_length
    return tokenizer

def prepare_hf_dataset(df, tokenizer, text_col='Transformed Sentence', label_col='Label', label_encoder=None, max_length=128):
    texts = df[text_col].astype(str).tolist()
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_length)
    if label_encoder is not None:
        labels = label_encoder.transform(df[label_col].values)
    else:
        labels = [0]*len(texts)
    inputs['labels'] = labels
    return inputs
