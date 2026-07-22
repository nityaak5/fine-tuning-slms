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
  private HF model repo.
- **`finetune_stance.job`** -- SLURM script that runs `finetune_stance.py` on a
  single non-exclusive A100 (18 cores + 1 GPU, ~128 SBU/hr).

Model and dataset aren't decided yet, so both scripts take them as required CLI
arguments rather than hardcoded defaults -- `finetune_stance.job` currently has
`TODO_...` placeholders where those go.

## Setup on Snellius (one-time)

Do the install on an actual GPU node, not the login node, so it picks up the
right GPU architecture:

```bash
tmux                                                          # survives a dropped SSH connection
srun -p gpu_a100 -n 1 --gpus=1 -c 18 -t 00:30:00 --pty bash   # or gpu_h100, -c 16

module load 2025
module load Python/3.11.3-GCCcore-12.3.0 CUDA/12.1.1
python -m venv venv
source venv/bin/activate
pip install -U pip
pip install "unsloth[cu121-ampere-torch240] @ git+https://github.com/unslothai/unsloth.git"
pip install carbontracker   # tracks training energy/CO2, logged to <output_dir>/carbontracker/
huggingface-cli login   # once, so training can push merged models to the Hub
```

## Running an experiment

```bash
sbatch finetune_stance.job
```

## Testing the fine-tuned model

Add the pushed HF repo id as `VLLM_MODEL` in the Tilburg pipeline's `config.json`
-- no code changes needed there, it already resolves `VLLM_MODEL` as any HF Hub
repo id or local path.
