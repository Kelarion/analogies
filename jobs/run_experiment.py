"""
One array task of a sweep submitted with server_utils.send_to_server:

    python jobs/run_experiment.py ARRAY_INDEX N_TASK SWEEP_DIR [PER_JOB N_TOT]

runs combinations ARRAY_INDEX * PER_JOB ... (up to N_TOT), one after another in this
process, so a model is loaded once and reused (icl_model.get_lm caches it). Combination c
is task (c % N_TASK) with model (c // N_TASK), from SWEEP_DIR's pickles; consecutive
combinations share a model. Each result is saved to server_utils.SAVE_DIR and marked in
SWEEP_DIR/done/<c>; combinations already marked are skipped, so rerunning an array task
(resubmit_missing, a requeue after preemption) only redoes what's missing. A combination
that fails is reported and the rest still run; the exit code is 1 if any failed.
"""
import os
import sys
import pickle as pkl
import traceback

import super_experiments as sxp
import server_utils as su

idx, n_task, sweep = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
per_job = int(sys.argv[4]) if len(sys.argv) > 4 else 1
n_tot = int(sys.argv[5]) if len(sys.argv) > 5 else (idx + 1) * per_job
done_dir = os.path.join(sweep, 'done')
os.makedirs(done_dir, exist_ok=True)

failed = []
for c in range(idx * per_job, min((idx + 1) * per_job, n_tot)):
    if os.path.exists(os.path.join(done_dir, str(c))):
        print(f'[{c}] already done, skipping', flush=True)
        continue
    try:
        with open(os.path.join(sweep, 'task_%d.pkl' % (c % n_task)), 'rb') as f:
            task_dict = pkl.load(f)
        with open(os.path.join(sweep, 'model_%d.pkl' % (c // n_task)), 'rb') as f:
            mod_dict = pkl.load(f)
        print(f'[{c}] task: ', task_dict['args'], f'\n[{c}] model:', mod_dict['args'], flush=True)

        exp = sxp.Experiment(task_dict['task'](**task_dict['args']),
                             mod_dict['model'](**mod_dict['args']))
        exp.run()
        exp.save_experiment(su.SAVE_DIR)
        print(f'[{c}] saved', exp.file_suffix(), 'to', su.SAVE_DIR, flush=True)
        open(os.path.join(done_dir, str(c)), 'w').close()
    except Exception:
        failed.append(c)
        print(f'[{c}] FAILED', flush=True)
        traceback.print_exc()
        sys.stderr.flush()

if failed:
    print('failed combinations:', failed, flush=True)
    sys.exit(1)