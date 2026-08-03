import unsloth  # noqa: F401  -- must be imported before trl/transformers/peft so its patches apply
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
from trl import SFTConfig, SFTTrainer
from transformers import HfArgumentParser, TrainerCallback

from datasets import load_dataset
from huggingface_hub import HfApi


class CarbonTrackerCallback(TrainerCallback):
    """Drives a CarbonTracker instance from TRL's epoch hooks."""

    def __init__(self, tracker):
        self.tracker = tracker

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.tracker.epoch_start()

    def on_epoch_end(self, args, state, control, **kwargs):
        self.tracker.epoch_end()


@dataclass
class ExperimentArguments:
    """
    Arguments corresponding to the experiments of the user
    """

    pretrained_model_name_or_path: str = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint or HuggingFace repo id to load the model from. "
                "Not decided yet -- pass explicitly via --pretrained_model_name_or_path."
            )
        },
    )
    data_dir: str = field(
        default=None,
        metadata={
            "help": (
                "The directory or HuggingFace repo id to load the (conversations-formatted) "
                "stance dataset from. Not decided yet -- pass explicitly via --data_dir."
            )
        },
    )
    from_foundation_model: bool = field(
        default=True,
        metadata={
            "help": "Flag to specify whether the finetuning starts from a foundation model or a instruct-finetuned model"
        },
    )
    load_in_4bit: bool = field(
        default=True,
        metadata={
            "help": (
                "QLoRA (True, 4-bit base model) vs LoRA (False, 16-bit base model). "
                "Same script and code path either way -- this is the only thing that changes."
            )
        },
    )
    push_to_hub_id: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "If set, merge the trained LoRA adapter into the base model and push the "
                "result to this HuggingFace Hub repo id (e.g. 'your-username/stance-model-v1') "
                "after training. Requires being logged in (see README) or an HF_TOKEN env var."
            )
        },
    )
    carbon_tracking: bool = field(
        default=True,
        metadata={
            "help": "Track training energy/CO2 usage via CarbonTracker, logged to <output_dir>/carbontracker/."
        },
    )

    def __post_init__(self):
        if self.pretrained_model_name_or_path is None or self.data_dir is None:
            raise ValueError(
                f"Please specify the model and data! Received model: {self.pretrained_model_name_or_path} and data: {self.data_dir}"
            )


def prepare_dataset(dataset, tokenizer, from_foundation_model=False):
    # Define own template if finetuning from pre-trained model. If continue from a instruct finetune, then use the native tokenizer and chat template
    if from_foundation_model:
        tokenizer = get_chat_template(
            tokenizer,
            mapping={
                "role": "from",
                "content": "value",
                "user": "human",
                "assistant": "gpt",
            },
            chat_template="chatml",
            map_eos_token=True,
        )

    # Split each conversation into "prompt" (everything up to the assistant's turn) and
    # "completion" (just the assistant's turn), instead of flattening into one "text"
    # string. This is what lets completion_only_loss (set in main()) restrict training
    # loss to just the JSON answer -- a flattened "text" field loses the boundary
    # between the fixed instructions and the answer, so loss would be computed over
    # both (verified: Qwen3's chat template doesn't support the alternative
    # assistant_only_loss/{% generation %} masking mechanism -- it returns an all-zero
    # mask -- so prompt/completion splitting is the only approach that actually works
    # here). SFTTrainer applies the chat template to prompt+completion internally, so
    # we don't render text ourselves anymore.
    def split_prompt_completion(examples):
        convos = examples["conversations"]
        return {
            "prompt": [convo[:-1] for convo in convos],
            "completion": [convo[-1:] for convo in convos],
        }

    dataset = dataset.map(
        split_prompt_completion, batched=True, num_proc=os.cpu_count() // 2
    )

    return dataset


def apply_lora(model, max_seq_length):
    # Same call works for QLoRA and LoRA -- the 4-bit vs 16-bit choice is made when the base
    # model is loaded (load_in_4bit), not here.
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # rank of parameters. Higher R means more parameters
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,  # scaling of the weights
        lora_dropout=0,  # Dropout = 0 is currently optimized
        bias="none",  # Bias = "none" is currently optimized
        use_gradient_checkpointing="unsloth",
        max_seq_length=max_seq_length,
        random_state=47,
    )

    return model


def snapshot_trainable_params(model):
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def verify_weights_updated(model, before_snapshot):
    # Sum of absolute differences, not np.allclose/loose-tolerance equality -- LoRA A is
    # initialized with small Gaussian values, so a loose tolerance can miss real updates.
    print("Verifying LoRA weight updates (sum of abs differences per tensor):")
    diffs = {}
    unchanged = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        diff = (param.detach() - before_snapshot[name]).abs().sum().item()
        diffs[name] = diff
        print(f"  {name}: {diff:.6f}")
        if diff == 0:
            unchanged.append(name)
    if unchanged:
        print(f"WARNING: {len(unchanged)} trainable param(s) never changed: {unchanged}")
    else:
        print("All trainable parameters changed during training.")
    return {"all_changed": not unchanged, "unchanged_params": unchanged, "abs_diff_per_param": diffs}


def build_training_summary(user_config, sft_config, trainer_stats, weight_check):
    # Mirrors the summary.json structure the Tilburg vLLM eval pipeline already writes
    # per inference run (genai_functions.py) -- same gpu_stats field names -- so training
    # runs get the same at-a-glance observability inference runs already have.
    gpu_props = torch.cuda.get_device_properties(0)
    memory_used_mb = torch.cuda.max_memory_reserved() / 1024**2
    memory_total_mb = gpu_props.total_memory / 1024**2
    return {
        "experiment_metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pretrained_model_name_or_path": user_config.pretrained_model_name_or_path,
            "data_dir": user_config.data_dir,
            "push_to_hub_id": user_config.push_to_hub_id,
        },
        "configuration": {
            "load_in_4bit": user_config.load_in_4bit,
            "from_foundation_model": user_config.from_foundation_model,
            "learning_rate": sft_config.learning_rate,
            "num_train_epochs": sft_config.num_train_epochs,
            "per_device_train_batch_size": sft_config.per_device_train_batch_size,
            "gradient_accumulation_steps": sft_config.gradient_accumulation_steps,
            "effective_batch_size": (
                sft_config.per_device_train_batch_size * sft_config.gradient_accumulation_steps
            ),
            "weight_decay": sft_config.weight_decay,
            "warmup_ratio": sft_config.warmup_ratio,
            "optim": sft_config.optim,
            "max_length": sft_config.max_length,
            "packing": sft_config.packing,
            "completion_only_loss": sft_config.completion_only_loss,
        },
        "training_stats": trainer_stats.metrics,
        "gpu_stats": {
            "gpu_id": 0,
            "gpu_name": gpu_props.name,
            "memory_used_mb": round(memory_used_mb, 1),
            "memory_total_mb": round(memory_total_mb, 1),
            "memory_utilization_percent": round(memory_used_mb / memory_total_mb * 100, 2),
            "runtime_seconds": round(trainer_stats.metrics["train_runtime"], 1),
        },
        "weight_verification": {
            "all_changed": weight_check["all_changed"],
            "unchanged_params": weight_check["unchanged_params"],
        },
    }


def plot_loss_curve(trainer, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless -- no display available on a Snellius compute node
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"matplotlib not available, skipping loss curve plot: {exc}")
        return

    logged_steps = [entry for entry in trainer.state.log_history if "loss" in entry]
    if not logged_steps:
        print("No training loss recorded, skipping loss curve plot.")
        return

    steps = [entry["step"] for entry in logged_steps]
    losses = [entry["loss"] for entry in logged_steps]

    plt.figure()
    plt.plot(steps, losses, marker="o")
    plt.xlabel("Step")
    plt.ylabel("Training loss")
    plt.title("Training loss curve")
    plot_path = Path(output_dir) / "loss_curve.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Loss curve saved to {plot_path}")


def main(user_config, sft_config):
    # Load dataset
    dataset = load_dataset(user_config.data_dir, split="train")

    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=user_config.pretrained_model_name_or_path,
        max_seq_length=sft_config.max_length,
        device_map="auto",
        dtype=None,  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit=user_config.load_in_4bit,
    )

    # Continuing from an already instruct-tuned checkpoint can leave tokenizer.eos_token
    # as an unresolved placeholder (e.g. '<EOS_TOKEN>') rather than the model's real
    # end-of-turn token. Restore it from the tokenizer's own special tokens if so, before
    # SFTTrainer validates it against the vocabulary.
    if tokenizer.convert_tokens_to_ids(tokenizer.eos_token) is None:
        fixed_eos_token = tokenizer.special_tokens_map.get("eos_token")
        if fixed_eos_token is None or tokenizer.convert_tokens_to_ids(fixed_eos_token) is None:
            raise ValueError(
                f"tokenizer.eos_token ({tokenizer.eos_token!r}) is not in the vocabulary, and "
                f"tokenizer.special_tokens_map has no usable fallback ({tokenizer.special_tokens_map!r})"
            )
        print(f"Fixing broken tokenizer.eos_token ({tokenizer.eos_token!r} -> {fixed_eos_token!r})")
        tokenizer.eos_token = fixed_eos_token
        sft_config.eos_token = fixed_eos_token

    # Map the dataset to prompt/completion columns (see prepare_dataset() for why)
    dataset = prepare_dataset(dataset, tokenizer, user_config.from_foundation_model)
    # Restrict training loss to the completion (the JSON answer) -- without this,
    # loss is computed over the fixed instructions too, which are ~75% of every
    # example and never need to be learned (see prepare_dataset()'s comment).
    sft_config.completion_only_loss = True

    # Patch the model with parameter-efficient finetuning
    model = apply_lora(model, sft_config.max_length)

    before_snapshot = snapshot_trainable_params(model)

    tracker = None
    carbon_log_dir = None
    if user_config.carbon_tracking:
        try:
            from carbontracker.tracker import CarbonTracker

            carbon_log_dir = Path(sft_config.output_dir) / "carbontracker"
            carbon_log_dir.mkdir(parents=True, exist_ok=True)
            tracker = CarbonTracker(
                epochs=int(sft_config.num_train_epochs),
                components="gpu",
                log_dir=str(carbon_log_dir),
                monitor_epochs=-1,
            )
        except Exception as exc:
            print(f"CarbonTracker not available: {exc}")
            tracker = None

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        callbacks=[CarbonTrackerCallback(tracker)] if tracker is not None else None,
    )

    trainer_stats = trainer.train()
    print(trainer_stats)

    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(
        f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
    )

    plot_loss_curve(trainer, sft_config.output_dir)
    weight_check = verify_weights_updated(model, before_snapshot)

    if tracker is not None:
        try:
            tracker.stop()
        except Exception:
            pass
        print(f"CarbonTracker logs written to {carbon_log_dir}")

    training_summary = build_training_summary(user_config, sft_config, trainer_stats, weight_check)
    summary_path = Path(sft_config.output_dir) / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(training_summary, f, indent=2)
    print(f"Training summary written to {summary_path}")

    if user_config.push_to_hub_id:
        model.push_to_hub_merged(
            user_config.push_to_hub_id,
            tokenizer,
            save_method="merged_16bit",
            token=os.environ.get("HF_TOKEN"),
        )
        print(f"Merged model pushed to https://huggingface.co/{user_config.push_to_hub_id}")

        # Also attach the training summary + loss curve to the same model repo, so
        # anyone looking at the model can see how (and how expensively) it was trained --
        # push_to_hub_merged() only uploads model/tokenizer files, not arbitrary extras.
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.upload_file(
            path_or_fileobj=str(summary_path),
            path_in_repo="training_summary.json",
            repo_id=user_config.push_to_hub_id,
        )
        loss_curve_path = Path(sft_config.output_dir) / "loss_curve.png"
        if loss_curve_path.exists():
            api.upload_file(
                path_or_fileobj=str(loss_curve_path),
                path_in_repo="loss_curve.png",
                repo_id=user_config.push_to_hub_id,
            )
        print(f"training_summary.json and loss_curve.png attached to https://huggingface.co/{user_config.push_to_hub_id}")


if __name__ == "__main__":
    # Parse both SFTConfig arguments and the extended model/training arguments
    parser = HfArgumentParser((ExperimentArguments, SFTConfig))
    user_config, sft_config = parser.parse_args_into_dataclasses()
    print(user_config, sft_config)
    main(user_config, sft_config)
