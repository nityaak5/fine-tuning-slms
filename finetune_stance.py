import os
from dataclasses import dataclass, field
from typing import Optional

from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import HfArgumentParser

from datasets import load_dataset


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

    # Dataset-specific function to convert the samples (in dictionaries) to strings with corresponding template
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            for convo in convos
        ]
        return {
            "text": texts,
        }

    # Apply applying the samples dictionaries to string
    dataset = dataset.map(
        formatting_prompts_func, batched=True, num_proc=os.cpu_count() // 2
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


def main(user_config, sft_config):
    # Load dataset
    dataset = load_dataset(user_config.data_dir, split="train")

    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=user_config.pretrained_model_name_or_path,
        max_seq_length=sft_config.max_seq_length,
        device_map="auto",
        dtype=None,  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit=user_config.load_in_4bit,
    )

    # Map the dataset to the right template
    dataset = prepare_dataset(dataset, tokenizer, user_config.from_foundation_model)
    # Pass the key of the dataset object
    sft_config.dataset_text_field = "text"

    # Patch the model with parameter-efficient finetuning
    model = apply_lora(model, sft_config.max_seq_length)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
    )

    trainer_stats = trainer.train()
    print(trainer_stats)

    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(
        f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
    )

    if user_config.push_to_hub_id:
        model.push_to_hub_merged(
            user_config.push_to_hub_id,
            tokenizer,
            save_method="merged_16bit",
            token=os.environ.get("HF_TOKEN"),
        )
        print(f"Merged model pushed to https://huggingface.co/{user_config.push_to_hub_id}")


if __name__ == "__main__":
    # Parse both SFTConfig arguments and the extended model/training arguments
    parser = HfArgumentParser((ExperimentArguments, SFTConfig))
    user_config, sft_config = parser.parse_args_into_dataclasses()
    print(user_config, sft_config)
    main(user_config, sft_config)
