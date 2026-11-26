#!/bin/bash
# Filled in by server_utils.send_to_server when a sweep needs weights that aren't cached.
# CPU only. Jobs with the same name run one at a time (singleton), so two sweeps needing
# the same model don't download it twice at once; the second finds it already there.
# Only for clusters whose compute nodes have internet; the resource lines ({{cpu_sbatch}})
# come from the cluster's cpu_sbatch in clusters.toml.
#SBATCH --job-name=hf-download
{cpu_sbatch}
#SBATCH --dependency=singleton
#SBATCH --output={sweep_dir}/logs/download.out

export HF_HOME={scratch}/.cache/huggingface
export HF_TOKEN_PATH=$HOME/.cache/huggingface/token
cd {repo}
.venv/bin/python jobs/weights.py download {models}