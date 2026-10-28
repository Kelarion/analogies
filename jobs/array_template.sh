#!/bin/bash
# Job-array template filled in by server_utils.send_to_server: {{name}} fields are
# placeholders, so literal braces in bash have to be doubled ({{ }}).
#
# Partition: main caps resources (2 GPUs, 48G, 8 CPUs per partitions.md). long has no cap
# but is preemptible: a preempted task is requeued and rerun from the start, which is safe
# here because a run only saves at the end.
#SBATCH --job-name=icl
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --constraint=80gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-11:59:00
#SBATCH --array=0-{n_tot}%{max_parallel}
#SBATCH --output={sweep_dir}/logs/%a.out

# set explicitly rather than relying on the submitting shell's environment
export PATH="$HOME/.local/bin:$PATH"                       # uv
export ICL_RESULTS={results}
export HF_HOME={scratch}/.cache/huggingface
export HF_TOKEN_PATH=$HOME/.cache/huggingface/token
export HF_HUB_OFFLINE=1        # weights must already be downloaded (hf download ...)
export TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1   # progress bars rewrite the log
                               # file many times a second; nobody watches them in batch
cd {repo}
echo "node: $(hostname)  start: $(date -Is)  job: ${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}"
echo "commit: $(git rev-parse HEAD)"
env | grep "^HF_" | sort            # where this job looks for weights
uv sync --check -q || {{ echo "environment out of date: run 'uv sync' on the login node"; exit 1; }}
uv run --no-sync python jobs/run_experiment.py $SLURM_ARRAY_TASK_ID {n_task} {sweep_dir}
