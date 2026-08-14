"""
Recovers MT-CSD's official train/validation split (currently merged together in our
local mtcsd_train.csv) and pushes one HF dataset repo with train/validation/test
splits for mtcsd.

Why: our local data_in/mtcsd/mtcsd_train.csv turned out to be the official
train.json + valid.json (from https://github.com/nfq729/MT-CSD/tree/main/data)
concatenated in original order, per domain -- verified by matching each domain's
stance-label sequence position-for-position against the two source JSON files (exact
match for all 5 domains: Biden, Bitcoin, SpaceX, Tesla, Trump). mtcsd_test.csv is
already exactly official test.json, untouched here.

OFFICIAL_TRAIN_COUNTS below is the recovered boundary: for each domain, the first N
rows (in the CSV's existing order) are official train, the rest are official valid.
No reshuffling -- order was already confirmed to match the source JSON order.

Reuses prepare_stance_data.py's canonical build_conversations() (one-off
dataset-generation script, not a teaching notebook -- see build_experiment_matrix.py
for the same convention).

Run locally: python build_mtcsd_dataset.py
"""
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict

from prepare_stance_data import build_conversations

MTCSD_DIR = Path(
    "/Users/nityaakalra/Desktop/nyt_topic_modeling/topic_modeling_paper/data_in/mtcsd"
)
REPO_ID = "nityaak/mtcsd-stance"
PRIVATE = False  # public, per project convention

# Recovered from https://github.com/nfq729/MT-CSD/tree/main/data -- train.json/valid.json
# lengths per domain, verified by exact stance-label-sequence match against our local CSV.
OFFICIAL_TRAIN_COUNTS = {
    "Biden": 2017,
    "Bitcoin": 2283,
    "SpaceX": 1440,
    "Tesla": 2552,
    "Trump": 2633,
}


def split_train_valid(df):
    train_parts, valid_parts = [], []
    for domain, group in df.groupby("domain", sort=False):
        n_train = OFFICIAL_TRAIN_COUNTS[domain]
        train_parts.append(group.iloc[:n_train])
        valid_parts.append(group.iloc[n_train:])
    return pd.concat(train_parts), pd.concat(valid_parts)


def label_balance(df):
    return df["stance_label"].value_counts().to_dict()


def to_conversations_dataset(df):
    conversations = build_conversations(df, text_column="content", target_column="query", label_column="stance_label")
    return Dataset.from_dict({"conversations": conversations})


def main():
    full_train_pool = pd.read_csv(MTCSD_DIR / "mtcsd_train.csv")
    test_df = pd.read_csv(MTCSD_DIR / "mtcsd_test.csv")

    train_df, valid_df = split_train_valid(full_train_pool)

    print(f"train:      {len(train_df)} rows, {label_balance(train_df)}")
    print(f"validation: {len(valid_df)} rows, {label_balance(valid_df)}")
    print(f"test:       {len(test_df)} rows, {label_balance(test_df)}")

    dataset_dict = DatasetDict(
        {
            "train": to_conversations_dataset(train_df),
            "validation": to_conversations_dataset(valid_df),
            "test": to_conversations_dataset(test_df),
        }
    )

    dataset_dict.push_to_hub(REPO_ID, private=PRIVATE)
    print(f"Pushed to https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
