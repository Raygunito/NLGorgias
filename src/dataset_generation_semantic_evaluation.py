import os

import pandas as pd
import torch
from sentence_transformers import CrossEncoder
import nltk
from tqdm import tqdm
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Initialize the Cross-Encoder once
device = "cpu"
ce = CrossEncoder("roberta-large-mnli", device=device)

def max_sentence_contradiction(premise_par, hypothesis_par):
    sents_p = sent_tokenize(premise_par)
    sents_h = sent_tokenize(hypothesis_par)
    pairs   = list(zip(sents_p, sents_h))
    logits  = ce.predict(pairs, batch_size=8)
    probs   = torch.softmax(torch.tensor(logits), dim=1)
    return float(probs[:, 0].max().item())

# List of (label, CSV path)
BASE = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE, "data", "dataset_generation_evaluation")

files = [
    ("common",    os.path.join(DATA_DIR, "eval_common_dataset.csv")),
    ("conflict",  os.path.join(DATA_DIR, "eval_conflict_dataset.csv")),
    ("no_common", os.path.join(DATA_DIR, "eval_no_common.csv")),
]

# Only one progress bar over the three datasets
for label, path in tqdm(files, desc="Processing datasets"):
    df = pd.read_csv(path)

    # Compute contradiction scores for each hypothesis type (no per-row tqdm)
    df['contra_specific_vs_manual_sent'] = df.apply(
        lambda r: max_sentence_contradiction(
            r['Manual NL Translation'],
            r['Translation with specific prompt']
        ), axis=1
    )
    df['contra_general_vs_manual_sent'] = df.apply(
        lambda r: max_sentence_contradiction(
            r['Manual NL Translation'],
            r['Translation with general prompt']
        ), axis=1
    )
    df['contra_naive_vs_manual_sent'] = df.apply(
        lambda r: max_sentence_contradiction(
            r['Manual NL Translation'],
            r['Translation with naive prompt']
        ), axis=1
    )

    # Compute and print mean scores
    mean_specific = df['contra_specific_vs_manual_sent'].mean()
    mean_general  = df['contra_general_vs_manual_sent'].mean()
    mean_naive    = df['contra_naive_vs_manual_sent'].mean()

    print(f"\n=== Dataset: {label} ===")
    print(f"Mean contradiction (specific vs manual) = {mean_specific:.4f}")
    print(f"Mean contradiction (general vs manual)  = {mean_general:.4f}")
    print(f"Mean contradiction (naive vs manual)    = {mean_naive:.4f}")