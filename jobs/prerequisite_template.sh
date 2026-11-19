#!/bin/bash
# Filled in by server_utils.send_to_server for each file a sweep needs that isn't on the
# cluster yet (a model's stored token categories, say): makes it, and the sweep's array
# waits for this to succeed. Submitted with a job name derived from the file and
# --dependency=singleton, so two sweeps needing the same file make it once: the second job
# runs after the first, finds it current, and does nothing. {{name}} fields are placeholders.
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --constraint=80gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-11:59:00
#SBATCH --output={sweep_dir}/logs/{log}.out

export PATH="$HOME/.local/bin:$PATH"                       # uv
export ICL_RESULTS={results}
export HF_HOME={scratch}/.cache/huggingface
export HF_TOKEN_PATH=$HOME/.cache/huggingface/token
export HF_HUB_OFFLINE=1        # weights must already be downloaded (send_to_server sees to that)
export TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
cd {repo}
echo "node: $(hostname)  start: $(date -Is)  job: $SLURM_JOB_ID  commit: $(git rev-parse HEAD)"
echo "making: {what}"
uv sync --check -q || {{ echo "environment out of date: run 'uv sync' on the login node"; exit 1; }}
uv run --no-sync python {command}