#!/bin/bash
# Job-array template filled in by server_utils.send_to_server: {{name}} fields are
# placeholders, so literal braces in bash have to be doubled ({{ }}).
#
# The resource lines ({{sbatch}}) come from the cluster's section of clusters.toml.
#SBATCH --job-name=icl
{sbatch}
#SBATCH --array=0-{last}%{max_parallel}
#SBATCH --output={sweep_dir}/logs/%a.out

# set explicitly rather than relying on the submitting shell's environment
export PATH="$HOME/.local/bin:$PATH"                       # uv
export ICL_RESULTS={results}
export HF_HOME={scratch}/.cache/huggingface
export HF_TOKEN_PATH=$HOME/.cache/huggingface/token
export HF_HUB_OFFLINE=1        # weights must already be downloaded (hf download ...)
export TQDM_DISABLE=1 HF_HUB_DISABLE_PROGRESS_BARS=1   # progress bars rewrite the log
                               # file many times a second; nobody watches them in batch
# grow memory segments in place instead of caching many odd-sized blocks: generation's KV
# cache grows a step at a time, which otherwise fragments the cache until allocations fail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd {repo}
echo "node: $(hostname)  start: $(date -Is)  job: ${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}"
echo "commit: $(git rev-parse HEAD)"
env | grep "^HF_" | sort            # where this job looks for weights
uv sync --check -q || {{ echo "environment out of date: run 'uv sync' on the login node"; exit 1; }}
# Python's output goes to the node's local disk, and is appended to this task's log in the
# sweep folder at the end (retrying), so a flaky connection to scratch can't kill the run
# by failing its prints. To watch a running task: ssh to its node, tail -f <tmpdir>/run.log
RUN_LOG=$SLURM_TMPDIR/run.log
echo "python output: $RUN_LOG on $(hostname), copied here when the task ends"
save_log() {{
    [ -n "$LOG_SAVED" ] && return
    LOG_SAVED=1                   # claimed first: a signal arriving mid-copy mustn't copy again
    for i in 1 2 3 4 5; do
        cat "$RUN_LOG" >> {sweep_dir}/logs/$SLURM_ARRAY_TASK_ID.out 2>/dev/null && return
        sleep 15
    done
}}
trap save_log EXIT TERM
uv run --no-sync python jobs/run_experiment.py $SLURM_ARRAY_TASK_ID {n_task} {sweep_dir} {per_job} {n_tot} > "$RUN_LOG" 2>&1
status=$?
save_log
exit $status