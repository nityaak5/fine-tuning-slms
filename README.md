# fine-tuning-slms

Fine-tuning small language models (SLMs) for stance detection: trained on SURF's
Snellius supercomputer, tested against an existing vLLM-based evaluation pipeline
on a university GPU cluster. Structure is deliberately minimal, following
[sara-nl/LLM-finetune](https://github.com/sara-nl/LLM-finetune)'s pattern. 

## How it fits together

```
Mac (this repo)                Snellius (train)                Tilburg cluster (test)
----------------------   git   ----------------------  HF Hub   ----------------------
prepare_stance_data.py  ---->  finetune_stance.py      ----->   existing vLLM pipeline
  (local, no GPU,               (QLoRA/LoRA via                  (unchanged -- just
   pushes dataset to Hub)        Unsloth + TRL,                   point VLLM_MODEL at
                                  pushes merged model               the new HF repo)
                                  to Hub after training)
```

Code moves between machines via git (push from Mac, pull on Snellius). Data and
trained models move via private HuggingFace Hub repos -- nothing is copied by hand.

## Files

- **`prepare_stance_data.py`** -- run *locally*, no GPU needed. Converts a raw
  stance dataset (text/target/label columns) into the chat-style `conversations`
  format the training script expects, and optionally pushes it to a private HF
  dataset repo.
- **`finetune_stance.py`** -- run *on Snellius*. QLoRA/LoRA fine-tuning via
  [Unsloth](https://unsloth.ai/) + TRL's `SFTTrainer`. QLoRA vs LoRA is a single
  flag (`--load_in_4bit` / `--no_load_in_4bit`), not a separate script. After
  training, merges the LoRA adapter into the base model and pushes it to a
  private HF model repo, along with `training_summary.json` and
  `loss_curve.png` -- same `gpu_stats` field names (`gpu_name`,
  `memory_used_mb`, `memory_total_mb`, `memory_utilization_percent`,
  `runtime_seconds`) as the `summary.json` the Tilburg vLLM eval pipeline
  already writes per inference run, plus the hyperparameters used, TRL's
  training metrics, and the LoRA weight-update check -- so each pushed model
  repo is self-documenting about how (and how expensively) it was trained.
- **`finetune_stance.job`** -- SLURM script that runs `finetune_stance.py` on a
  single non-exclusive A100 (18 cores + 1 GPU, ~128 SBU/hr).
- **`build_experiment_matrix.py`** -- run *locally*, no GPU needed. Builds all
  7 non-empty subsets of `{semeval, mtcsd, ezstance}` from the size-controlled
  samples in `data/samples/` (each exactly 2,814 rows -- semeval's full size,
  the binding constraint across all three), prints label-balance diagnostics
  per combo, and pushes each to its own public repo
  (`nityaak/stance-matrix-*`) -- push commented out by default, uncomment
  when ready.
- **`finetune_stance_sweep.job`** -- runs the full experiment matrix (all 7
  non-empty subsets of `{semeval, mtcsd, ezstance}`, from
  `build_experiment_matrix.py`'s pushed datasets) as one SLURM job array
  (`sbatch` once, SLURM fans out each row of `DATA_DIRS` as an independent
  task, running in parallel where A100 capacity allows) -- instead of
  duplicating `finetune_stance.job` per variant, which drifts out of sync.
  `push_to_hub_id`/`output_dir` are derived from each row the same way as
  `finetune_stance.job`, so only `DATA_DIRS` + `--array` need editing to
  add/remove experiments.
- **`notebooks/`** -- learning/exploration notebooks (each defines its own logic
  inline rather than importing from the scripts above, on purpose):
  - `01_create_single_dataset.ipynb` / `02_combine_multiple_datasets.ipynb` --
    build the `conversations`-formatted training dataset locally, run on your Mac.
  - `04_prepare_test_set.ipynb` -- converts the held-out gold test set
    (`semeval2016_task6_testdata_gold`, 1249 examples, confirmed zero overlap
    with training data) into the same `conversations` format and pushes it to
    its own HF dataset repo, so it's loadable from anywhere (including Colab).
  - `05_prepare_additional_test_sets.ipynb` -- same idea as `04`, for the
    **test** splits of mtcsd (`mtcsd_test.csv`, 2371 examples) and EZ-STANCE
    (`ezstance_test_mixed.csv`, 7798 examples) -- pushed to their own HF repos
    so the Colab notebook can check whether the semeval-trained model
    generalizes to domains it's never seen.
  - `03_colab_hyperparameter_playground.ipynb` -- cheap, fast hyperparameter
    sandbox on a free Colab T4: trains on the full dataset with the same
    hyperparameters as `finetune_stance.job`, shows a loss curve, verifies LoRA
    weights actually changed, compares before/after predictions and full
    test-set F1 on the real held-out test set from `04`, pushes the merged
    model to a sandbox HF repo, and checks generalization F1 on the mtcsd/
    EZ-STANCE test sets from `05`. Not a substitute for the real run -- just a
    fast way to see a hyperparameter's effect before spending Snellius time on it.

Model and dataset for the current experiment: `unsloth/Qwen3-1.7B-unsloth-bnb-4bit`
(the instruct checkpoint, continuing from the one that underperformed zero-shot on
stance detection) fine-tuned on `nityaak/semeval-stance-conversations` (SemEval-2016
Task 6, converted by `prepare_stance_data.py`). Both scripts still take model/dataset
as CLI arguments rather than hardcoding them, so swapping either for a future
experiment is just a different `.job` file.

## CLI arguments

`finetune_stance.py` parses two dataclasses together via
`HfArgumentParser((ExperimentArguments, SFTConfig))` -- every field on *both*
dataclasses automatically becomes a `--flag`, so adding a new `SFTConfig`/
`TrainingArguments` flag to a `.job` file never requires a code change. Only
`ExperimentArguments` is defined in this repo; everything else comes from TRL/HF.

### Arguments defined in this repo (`ExperimentArguments`)

| Flag | Default | Meaning |
|---|---|---|
| `--pretrained_model_name_or_path` | *(required)* | HF repo id or path to load the base model from |
| `--data_dir` | *(required)* | HF repo id or local path to the `conversations`-formatted dataset |
| `--from_foundation_model` / `--no_from_foundation_model` | `True` | `True` if fine-tuning a raw foundation model (applies a fresh ShareGPT/chatml template); `--no_from_foundation_model` if continuing from an already instruct-tuned checkpoint (preserves its native chat template) |
| `--load_in_4bit` / `--no_load_in_4bit` | `True` | QLoRA (4-bit base) vs LoRA (16-bit base) -- same code path either way |
| `--push_to_hub_id` | `None` | If set, merges the LoRA adapter into the base model after training and pushes to this HF model repo |
| `--carbon_tracking` / `--no_carbon_tracking` | `True` | Logs training energy/CO2 via CarbonTracker to `<output_dir>/carbontracker/` |

### Arguments inherited from TRL's `SFTConfig` (and its parent `TrainingArguments`)

These are the ones currently set in `finetune_stance.job`, plus their real defaults for
comparison. Full field lists (too long to reproduce here): [`SFTConfig`
source](https://github.com/huggingface/trl/blob/main/trl/trainer/sft_config.py),
[`TrainingArguments` docs](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments).

| Flag | Library default | Currently set in `finetune_stance.job` |
|---|---|---|
| `--learning_rate` | `2e-5` (SFTConfig overrides `TrainingArguments`' `5e-5`) | *not set* -- silently uses `2e-5` |
| `--num_train_epochs` | `3.0` | `1` |
| `--per_device_train_batch_size` | `8` | `8` (job file abbreviates this to `--per_device_train`, see note below) |
| `--per_device_eval_batch_size` | `8` | `8` (abbreviated to `--per_device_eval`) |
| `--gradient_accumulation_steps` | `1` | `4` |
| `--max_length` | `1024` | `512` |
| `--optim` | `"adamw_torch"` | `adamw_8bit` |
| `--bf16` | `False` | `True` (needs an explicit value, `--bf16 True`, not a bare flag) |
| `--packing` | `False` | set (bare flag) |
| `--logging_steps` | `500` | `10` |

**Note on abbreviated flags:** `--per_device_train` / `--per_device_eval` in the job
file work because argparse auto-expands unambiguous prefixes of long flag names --
they are not real field names or aliases. This is fragile: if TRL/HF ever add another
field sharing that prefix, the abbreviation becomes ambiguous and errors. Consider
spelling them out in full (`--per_device_train_batch_size`) for robustness.

**Note on `dataset_text_field`:** this is a valid `SFTConfig` flag, but
`finetune_stance.py:185` hardcodes `sft_config.dataset_text_field = "text"` after
parsing -- so passing `--dataset_text_field` on the CLI has no effect; it's always
overwritten.

## Setup on Snellius (one-time)

Do the install on an actual GPU node, not the login node, so it picks up the
right GPU architecture. Give it a full hour -- the unsloth install pulls in a lot
(transformers, trl, peft, xformers, bitsandbytes, etc.) and can take a while:

```bash
tmux                                                          # survives a dropped SSH connection
srun -p gpu_a100 -n 1 --gpus=1 -c 18 -t 01:00:00 --pty bash   # or gpu_h100, -c 16
nvidia-smi                                                     # confirm you actually got a GPU

module load 2025
module load Python/3.13.5-GCCcore-14.3.0 CUDA/12.9.1
python -m venv venv
source venv/bin/activate
pip install -U pip
pip install uv   # much more reliable resolver than plain pip for this dependency stack
uv pip install unsloth --torch-backend=cu128
# ^ --torch-backend=auto sounds nicer but silently fell back to a CPU-only torch build
# for us -- pin the CUDA backend explicitly (cu128 matched CUDA/12.9.1 here) and verify:
python -c "import torch; assert torch.cuda.is_available()"
pip install carbontracker   # tracks training energy/CO2, logged to <output_dir>/carbontracker/
pip install matplotlib      # loss curve plot, saved to <output_dir>/loss_curve.png
hf auth login   # once, so training can push merged models to the Hub
```

## Running an experiment

```bash
sbatch finetune_stance.job
```

## Testing the fine-tuned model

Add the pushed HF repo id as `VLLM_MODEL` in the Tilburg pipeline's `config.json`
-- no code changes needed there, it already resolves `VLLM_MODEL` as any HF Hub
repo id or local path.
