# Snellius commands cheat sheet

## 1. Log in and start a session that survives disconnects
```bash
ssh <username>@snellius.surf.nl
tmux
```
Reconnect later with: `tmux attach`

## 2. Get a GPU node (one-time setup only, not for training)
```bash
srun -p gpu_a100 -n 1 --gpus=1 -c 18 -t 01:00:00 --pty bash
```
Check you're actually on a GPU node:
```bash
nvidia-smi
```

## 3. One-time environment setup (run once, on the GPU node from step 2)
```bash
git clone https://github.com/nityaak5/fine-tuning-slms.git
cd fine-tuning-slms

module load 2025
module load Python/3.13.5-GCCcore-14.3.0 CUDA/12.9.1
python -m venv venv
source venv/bin/activate
pip install -U pip
pip install uv
uv pip install unsloth --torch-backend=cu128
python -c "import torch; assert torch.cuda.is_available()"   # must print nothing (no AssertionError)
pip install carbontracker
pip install matplotlib
hf auth login
```
Then leave the GPU node:
```bash
exit
```

## 4. Run a training job
```bash
cd fine-tuning-slms
sbatch finetune_stance.job
```

## 5. Check on jobs
```bash
squeue -u $USER              # jobs currently running or queued
sacct -j <jobid>              # info on a finished job
tail -f logs/finetune_stance_<jobid>.out   # live output while it runs
```

## 6. Check budget
```bash
accinfo -u $USER
budget-overview -p gpu_a100   # most accurate remaining balance -- prefer this one
accuse -u $USER                # SBU consumption over time (monthly by default, -d for daily)
```

## 7. Pull code updates later
```bash
cd fine-tuning-slms
git pull
```
(No need to redo step 3 unless dependencies changed.)
