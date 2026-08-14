"""
Optuna hyperparameter study for Qwen3-4B on the full nityaak/mtcsd-stance
train/validation pools (10,925 / 2,209 rows) -- the real Snellius run following the
Colab sanity check (notebooks/06_optuna_hyperparameter_tuning_colab.ipynb), which
confirmed the train-fresh-model -> validate -> report-to-Optuna loop works on a small
scale.

Only tunes learning_rate and num_train_epochs -- everything else (LoRA r/alpha,
weight_decay, warmup_ratio, batch size) is already validated against Unsloth's own
hyperparameter guide.

Every trial trains a fresh model and discards it (save_strategy="no", deleted after
scoring) -- this script only finds the winning hyperparameters, it does not produce a
deployable model. Once it finishes, inspect study.best_params (via the downloaded
sqlite file + optuna-dashboard, or optuna.load_study(...)) and run finetune_stance.job
once, for real, with --learning_rate/--num_train_epochs set to those values.

Run via sbatch: sbatch tune_hyperparams_mtcsd.job (do not run this .py directly on a
login node -- see that file for the GPU allocation).
"""
import unsloth  # noqa: F401  -- must be imported before trl/transformers, see finetune_stance.py

import os

import optuna
import torch
from datasets import load_dataset
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

from finetune_stance import apply_lora, prepare_dataset

MODEL_NAME = "unsloth/Qwen3-4B-unsloth-bnb-4bit"
MAX_SEQ_LENGTH = 512
DATASET_ID = "nityaak/mtcsd-stance"

SCRATCH_DIR = f"/scratch-shared/{os.environ['USER']}"
STUDY_NAME = "qwen3-4b-mtcsd-tuning"
STORAGE_PATH = f"{SCRATCH_DIR}/optuna_studies/mtcsd_tuning_study.db"

# Kept just under the .job file's #SBATCH --time so Optuna stops starting new trials
# before SLURM kills the job for exceeding its walltime, leaving margin for the final
# sqlite write. Both this and #SBATCH --time are first estimates, not measured --
# adjust once real per-trial time on an A100 is known (Colab's T4 timing doesn't
# transfer). Override via the job file's OPTUNA_TIMEOUT_SECONDS env var.
DEFAULT_TIMEOUT_SECONDS = 6 * 3600 - 1200

# Plateau-based early stop, on top of the timeout above -- ends the study early if
# it's stopped making real progress, instead of always running until the timeout
# regardless (saves time/SBU once more trials clearly aren't helping).
PLATEAU_MIN_TRIALS = 15  # must clear TPESampler's n_startup_trials (10, its library default -- verified against installed optuna) before this can fire, otherwise it could mistake the random startup phase for a real plateau
PLATEAU_PATIENCE = 10  # how many of the most recent trials to check for improvement across
PLATEAU_MIN_RELATIVE_IMPROVEMENT = 0.01  # 1% -- smaller than this doesn't count as real progress, just noise


def build_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # A100/H100 -> bf16, auto-detected
        load_in_4bit=True,
    )
    model = apply_lora(model, MAX_SEQ_LENGTH)
    return model, tokenizer


class OptunaPruningCallback(TrainerCallback):
    """Reports each epoch's validation loss to the trial and raises TrialPruned if
    this trial is clearly losing against how other trials looked at the same point --
    deferred from the Colab sanity check, which had too few trials for a pruning
    baseline to mean anything."""

    def __init__(self, trial):
        self.trial = trial

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        val_loss = metrics.get("eval_loss")
        if val_loss is None:
            return
        step = int(state.epoch) if state.epoch is not None else state.global_step
        self.trial.report(val_loss, step=step)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


def objective(trial, train_dataset, val_dataset):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    num_train_epochs = trial.suggest_int("num_train_epochs", 1, 3)

    model, tokenizer = build_model()

    sft_config = SFTConfig(
        output_dir=f"{SCRATCH_DIR}/optuna_trials/trial_{trial.number}",
        save_strategy="no",  # every trial's weights are discarded, see module docstring
        per_device_train_batch_size=8,  # same as finetune_stance.job
        gradient_accumulation_steps=2,  # same as finetune_stance.job
        num_train_epochs=num_train_epochs,  # tuned
        learning_rate=learning_rate,  # tuned
        weight_decay=0.01,  # fixed, matches Unsloth's recommended range
        warmup_ratio=0.1,  # fixed, matches Unsloth's recommended range
        optim="adamw_8bit",  # same as finetune_stance.job
        bf16=True,  # A100/H100, unlike the Colab notebook's T4 fp16 fallback
        packing=True,  # restored -- full-size pools here, unlike the Colab sample
        eval_strategy="epoch",
        logging_steps=10,
        max_length=MAX_SEQ_LENGTH,
        completion_only_loss=True,
        report_to="none",
        seed=42,  # explicit, not just relying on the installed transformers default -- matches TPESampler/apply_lora
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[OptunaPruningCallback(trial)],
    )

    try:
        trainer.train()
        eval_result = trainer.evaluate()
        val_loss = eval_result["eval_loss"]
        print(f"Trial {trial.number}: lr={learning_rate:.2e}, epochs={num_train_epochs} -> val_loss={val_loss:.4f}")
        return val_loss
    finally:
        del model, tokenizer, trainer
        torch.cuda.empty_cache()


def plateau_stop_callback(study, trial):
    """Called by Optuna after every trial. Ends the study early (study.stop()) once
    the best val_loss found so far hasn't meaningfully improved in a while -- see the
    PLATEAU_* constants above for exactly what "a while" and "meaningfully" mean."""
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed_trials) < PLATEAU_MIN_TRIALS:
        return  # too early to judge a real plateau vs. just not having tried enough yet

    # "Best value as of PLATEAU_PATIENCE trials ago" vs. "best value right now" --
    # if that hasn't moved much, more trials aren't finding anything better.
    trials_before_window = completed_trials[:-PLATEAU_PATIENCE]
    if not trials_before_window:
        return  # not enough trials before the patience window yet either
    best_before_window = min(t.value for t in trials_before_window)
    best_now = study.best_value

    relative_improvement = (best_before_window - best_now) / best_before_window
    if relative_improvement <= PLATEAU_MIN_RELATIVE_IMPROVEMENT:
        print(
            f"Stopping early: best val_loss improved only {relative_improvement:.2%} "
            f"over the last {PLATEAU_PATIENCE} trials (needed > {PLATEAU_MIN_RELATIVE_IMPROVEMENT:.0%} to keep going)."
        )
        study.stop()


def main():
    os.makedirs(os.path.dirname(STORAGE_PATH), exist_ok=True)

    train_dataset = load_dataset(DATASET_ID, split="train")
    val_dataset = load_dataset(DATASET_ID, split="validation")
    print(f"train: {len(train_dataset)}, validation: {len(val_dataset)}")

    # tokenizer=None is safe here -- prepare_dataset() only touches the tokenizer
    # inside its from_foundation_model=True branch, which this run never takes (we're
    # continuing Qwen3's own instruct checkpoint, same as finetune_stance.job).
    # Doing this once here, not per-trial, since the prompt/completion split doesn't
    # depend on which trial's model is currently being trained.
    train_dataset = prepare_dataset(train_dataset, tokenizer=None, from_foundation_model=False)
    val_dataset = prepare_dataset(val_dataset, tokenizer=None, from_foundation_model=False)

    sampler = optuna.samplers.TPESampler(seed=42)  # reproducible trial sequence, see project notes
    pruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///{STORAGE_PATH}",
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,  # resumes if this job is re-submitted after an earlier partial run
    )

    timeout_seconds = int(os.environ.get("OPTUNA_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS))
    study.optimize(
        lambda trial: objective(trial, train_dataset, val_dataset),
        timeout=timeout_seconds,
        callbacks=[plateau_stop_callback],
    )

    print(f"\nBest trial: #{study.best_trial.number}")
    print(f"Best val_loss: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    print(f"Study stored at {STORAGE_PATH} -- download this file to inspect with optuna-dashboard or optuna.load_study().")

    # Every trial tried, not just the winner -- one row per trial (params, value,
    # state, duration), sorted so the best-scoring trials are easiest to scan first.
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values("value")
    print("\nAll trials tried:")
    print(trials_df.to_string(index=False))

    csv_path = f"{SCRATCH_DIR}/optuna_studies/mtcsd_tuning_results.csv"
    trials_df.to_csv(csv_path, index=False)
    print(f"\nAll trials saved to {csv_path}")


if __name__ == "__main__":
    main()
