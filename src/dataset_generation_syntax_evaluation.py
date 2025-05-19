import os

import pandas as pd
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize


# METEOR Function
def compute_meteor(hypothesis: str, reference: str) -> float:
    hyp_tokens = word_tokenize(hypothesis, language='english')
    ref_tokens = word_tokenize(reference, language='english')
    return meteor_score([ref_tokens], hyp_tokens)


# List of (label, file path) tuples
BASE = os.path.dirname(os.path.dirname(__file__))  # remonte d’un niveau depuis src/app
DATA_DIR = os.path.join(BASE, "data", "dataset_generation_evaluation")

files = [
    ("common",    os.path.join(DATA_DIR, "eval_common_dataset.csv")),
    ("conflict",  os.path.join(DATA_DIR, "eval_conflict_dataset.csv")),
    ("no_common", os.path.join(DATA_DIR, "eval_no_common.csv")),
]

for label, path in files:
    df = pd.read_csv(path)

    # Compute METEOR scores for each hypothesis type
    df['METEOR_specific_vs_manual'] = df.apply(
        lambda r: compute_meteor(r['Translation with specific prompt'],
                                 r['Manual NL Translation']), axis=1)
    df['METEOR_general_vs_manual'] = df.apply(
        lambda r: compute_meteor(r['Translation with general prompt'],
                                 r['Manual NL Translation']), axis=1)
    df['METEOR_naive_vs_manual'] = df.apply(
        lambda r: compute_meteor(r['Translation with naive prompt'],
                                 r['Manual NL Translation']), axis=1)

    # Compute mean scores
    mean_specific = df['METEOR_specific_vs_manual'].mean()
    mean_general = df['METEOR_general_vs_manual'].mean()
    mean_naive = df['METEOR_naive_vs_manual'].mean()

    print(f"\nDataset: {label}")
    print(f"METEOR mean (specific vs. manual) = {mean_specific:.4f}")
    print(f"METEOR mean (general vs. manual)  = {mean_general:.4f}")
    print(f"METEOR mean (naive vs. manual)    = {mean_naive:.4f}")