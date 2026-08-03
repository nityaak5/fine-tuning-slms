"""
Builds all 7 non-empty subsets of {semeval, mtcsd, ezstance} from the size-controlled
samples in data/samples/ (each exactly 2,814 rows -- semeval's full size, the binding
constraint across all three datasets), converts each combo to the `conversations`
format, and prints label-balance diagnostics per combo. Push to HF (public,
nityaak/stance-matrix-*) is commented out by default -- uncomment when ready.

Reuses prepare_stance_data.py's STANCE_PROMPT_TEMPLATE/build_conversations (this is a
one-off experiment-generation script, not a teaching notebook, so importing the
canonical copy instead of duplicating it is the right call here).

Run locally: python build_experiment_matrix.py
"""
import json
from itertools import combinations
from pathlib import Path

import pandas as pd
from datasets import Dataset, concatenate_datasets

from prepare_stance_data import build_conversations

REPO_ROOT = Path(__file__).parent
SAMPLES_DIR = REPO_ROOT / "data" / "samples"
PRIVATE = False  # public, per project convention

SOURCES = ["semeval", "mtcsd", "ezstance"]


def load_source(name):
    df = pd.read_csv(SAMPLES_DIR / f"{name}_sample.csv")
    conversations = build_conversations(df, text_column="content", target_column="query", label_column="stance_label")
    return Dataset.from_dict({"conversations": conversations})


def dedupe(dataset):
    seen = set()
    keep_indices = []
    for i, convo in enumerate(dataset["conversations"]):
        key = json.dumps(convo, sort_keys=True)
        if key not in seen:
            seen.add(key)
            keep_indices.append(i)
    if len(keep_indices) < len(dataset):
        print(f"  removed {len(dataset) - len(keep_indices)} exact duplicates")
    return dataset.select(keep_indices)


def label_balance(dataset):
    from collections import Counter
    labels = [json.loads(convo[1]["content"])["stance"] for convo in dataset["conversations"]]
    return dict(Counter(labels))


def main():
    per_source = {name: load_source(name) for name in SOURCES}
    for name, ds in per_source.items():
        print(f"{name}: {len(ds)} examples, {label_balance(ds)}")
    print()

    for r in range(1, len(SOURCES) + 1):
        for combo in combinations(SOURCES, r):
            combo_name = "-".join(combo)
            repo_id = f"nityaak/stance-matrix-{combo_name}"

            combined = concatenate_datasets([per_source[name] for name in combo])
            combined = combined.shuffle(seed=42)
            combined = dedupe(combined)

            print(f"=== {repo_id} ===")
            print(f"  {len(combined)} examples from {len(combo)} source(s): {combo}")
            print(f"  label balance: {label_balance(combined)}")

            combined.push_to_hub(repo_id, private=PRIVATE)
            print(f"  pushed to https://huggingface.co/datasets/{repo_id}")
            print()


if __name__ == "__main__":
    main()
