#!/bin/bash
# Filled in by server_utils.send_to_server when a sweep needs weights that aren't cached.
# CPU only. Jobs with the same name run one at a time (singleton), so two sweeps needing
# the same model don't download it twice at once; the second finds it already there.
#SBATCH --job-name=hf-download
#SBATCH --partition=main-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0-04:00:00
#SBATCH --dependency=singleton
#SBATCH --output={sweep_dir}/logs/download.out

export HF_HOME={scratch}/.cache/huggingface
export HF_TOKEN_PATH=$HOME/.cache/huggingface/token
cd {repo}
.venv/bin/python jobs/weights.py download {models}