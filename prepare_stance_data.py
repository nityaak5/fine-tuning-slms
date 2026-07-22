"""
Convert a raw stance-detection dataset (CSV/JSON/JSONL with text/target/label columns)
into the `conversations`-formatted dataset finetune_stance.py expects, and optionally
push it to a private HuggingFace dataset repo.

Runs locally -- no GPU/CUDA required. This is the one stance-specific piece of the
pipeline; finetune_stance.py itself never needs to know the label set or prompt wording.

Conversations are written in {"role", "content"} form (not ShareGPT {"from", "value"})
because finetune_stance.py is run with --no_from_foundation_model when continuing from
an already instruct-tuned checkpoint (e.g. unsloth/Qwen3-1.7B-unsloth-bnb-4bit), which
applies the tokenizer's own native chat template -- that template expects role/content.

Prompt and output format mirror `stance_classification_prompt_no_label_definitions` in
the topic_modeling_paper repo's genai_functions.py (STANCE_PROMPT_NAME =
"default_no_label_definitions") -- that's the prompt used to establish the zero-shot
Qwen3-1.7B baseline this fine-tune is meant to improve on. The fine-tuned model must be
tested with that exact same prompt/JSON format, so if that testing prompt ever changes,
this template needs to change with it.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from datasets import Dataset

STANCE_PROMPT_TEMPLATE = (
    "Your task is to determine the stance of the following document toward the given query.\n\n"
    "QUERY: {target}\n\n"
    "DOCUMENT: {text}\n\n"
    'Return valid JSON in exactly this format: {{"stance": "FAVOR"}}\n'
    'The "stance" value must be exactly one of: "FAVOR", "AGAINST", "NONE".\n'
)

# Mirrors StanceDetectionInterface.canonicalize_stance_label in topic_modeling_paper --
# keep in sync with that mapping so raw labels from any dataset land on the same fixed
# vocabulary the test-time prompt above expects.
STANCE_LABEL_MAP = {
    "FAVOR": "FAVOR",
    "AGAINST": "AGAINST",
    "NONE": "NONE",
    "PRO": "FAVOR",
    "NEUTRAL": "NONE",
    "UNCLEAR": "NONE",
    "UNRELATED": "NONE",
    "SUPPORTS": "FAVOR",
    "DENIES": "AGAINST",
}


def canonicalize_label(raw_label):
    normalized = str(raw_label).strip().upper()
    if normalized not in STANCE_LABEL_MAP:
        raise ValueError(
            f"Unrecognized stance label {raw_label!r} -- add it to STANCE_LABEL_MAP "
            f"(known: {sorted(STANCE_LABEL_MAP)})"
        )
    return STANCE_LABEL_MAP[normalized]


def build_conversations(df, text_column, target_column, label_column):
    conversations = []
    for _, row in df.iterrows():
        prompt = STANCE_PROMPT_TEMPLATE.format(target=row[target_column], text=row[text_column])
        answer = json.dumps({"stance": canonicalize_label(row[label_column])})
        conversations.append(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ]
        )
    return conversations


def main(args):
    input_path = Path(args.input_path)
    if input_path.suffix == ".csv":
        df = pd.read_csv(input_path)
    elif input_path.suffix == ".jsonl":
        df = pd.read_json(input_path, lines=True)
    elif input_path.suffix == ".json":
        df = pd.read_json(input_path)
    else:
        raise ValueError(f"Unsupported input format: {input_path.suffix}")

    conversations = build_conversations(df, args.text_column, args.target_column, args.label_column)
    dataset = Dataset.from_dict({"conversations": conversations})

    if args.push_to_hub_id:
        dataset.push_to_hub(args.push_to_hub_id, private=args.private)
        print(f"Pushed {len(dataset)} examples to https://huggingface.co/datasets/{args.push_to_hub_id}")
    else:
        output_path = Path(args.output_path or "prepared_stance_data.jsonl")
        dataset.to_json(output_path)
        print(f"Wrote {len(dataset)} examples to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_path",
        help="Path to the raw stance dataset (.csv, .json, or .jsonl).",
    )
    parser.add_argument("--text_column", default="content")
    parser.add_argument("--target_column", default="query")
    parser.add_argument("--label_column", default="stance_label")
    parser.add_argument(
        "--push_to_hub_id",
        default=None,
        help="e.g. your-hf-username/stance-dataset-v1. If omitted, writes a local JSONL instead.",
    )
    parser.add_argument(
        "--private",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Push as a private repo (default) or pass --no-private for a public repo.",
    )
    parser.add_argument("--output_path", default=None, help="Local output path if not pushing to Hub")
    args = parser.parse_args()
    main(args)
