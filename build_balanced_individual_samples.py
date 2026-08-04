"""
Builds larger, class-balanced individual (non-combo) training samples for mtcsd and
ezstance, and pushes each to its own HF dataset repo. Does NOT touch the existing
data/samples/{mtcsd,ezstance}_sample.csv (those stay as the 2,814-row combo-matrix
inputs) or their pushed repos.

Why balanced, and why this specific size: reviewing results from the 2,814-row
individual fine-tunes showed mtcsd-qlora underperforming zero-shot on its own test
set (50.7% vs 52.2%) with clear majority-class collapse (NONE recall 77.0% vs AGAINST
recall only 18.5%) -- a data-imbalance problem (mtcsd's full train pool is 48% NONE),
not an under-training problem. More *unbalanced* data would likely make this worse.
FAVOR is mtcsd's scarcest class at 2,810 examples in the full pool -- that's the
natural ceiling for a balanced sample, so both datasets here use 2,810/class (8,430
total) for a like-for-like comparison: same size, same balance, so any performance
gap between the two is attributable to the domains themselves, not sample size/skew.

Run locally: python build_balanced_individual_samples.py
"""
from pathlib import Path

import pandas as pd
from datasets import Dataset

from prepare_stance_data import build_conversations

REPO_ROOT = Path(__file__).parent
SAMPLES_DIR = REPO_ROOT / "data" / "samples"
PRIVATE = False  # public, per project convention
PER_CLASS = 2810  # mtcsd's scarcest class (FAVOR) in the full train pool -- shared cap
SEED = 42

TOPIC_MODELING_DATA_IN = Path(
    "/Users/nityaakalra/Desktop/nyt_topic_modeling/topic_modeling_paper/data_in"
)

SOURCES = {
    "mtcsd": {
        "raw_path": TOPIC_MODELING_DATA_IN / "mtcsd" / "mtcsd_train.csv",
        "read_kwargs": {},
        "text_column": "content",
        "target_column": "query",
        "label_column": "stance_label",
    },
    "ezstance": {
        "raw_path": TOPIC_MODELING_DATA_IN / "raw" / "ezstance" / "subtaskA" / "mixed" / "raw_train_all_onecol.csv",
        "read_kwargs": {},
        "text_column": "Text",
        "target_column": "Target 1",
        "label_column": "Stance 1",
    },
}


def balanced_sample(df, label_column, per_class, seed):
    label_norm = df[label_column].astype(str).str.strip().str.upper()
    parts = []
    for label, group in df.groupby(label_norm):
        n = min(per_class, len(group))
        if n < per_class:
            print(f"  warning: class {label!r} only has {len(group)} rows, wanted {per_class}")
        parts.append(group.sample(n=n, random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def main():
    for name, src in SOURCES.items():
        print(f"=== {name} ===")
        df = pd.read_csv(src["raw_path"], **src["read_kwargs"])
        print(f"  full pool: {len(df)} rows")

        sampled = balanced_sample(df, src["label_column"], PER_CLASS, SEED)
        print(f"  balanced sample: {len(sampled)} rows ({PER_CLASS}/class)")

        sample_path = SAMPLES_DIR / f"{name}_sample_balanced_{len(sampled)}.csv"
        sampled.to_csv(sample_path, index=False)
        print(f"  wrote {sample_path}")

        conversations = build_conversations(
            sampled, src["text_column"], src["target_column"], src["label_column"]
        )
        dataset = Dataset.from_dict({"conversations": conversations})

        repo_id = f"nityaak/{name}-balanced-{len(sampled)}-stance-conversations"
        dataset.push_to_hub(repo_id, private=PRIVATE)
        print(f"  pushed to https://huggingface.co/datasets/{repo_id}")
        print()


if __name__ == "__main__":
    main()
