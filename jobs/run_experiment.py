"""
One array task of a sweep submitted with server_utils.send_to_server:

    python jobs/run_experiment.py ARRAY_INDEX N_TASK SWEEP_DIR

runs task (ARRAY_INDEX % N_TASK) with model (ARRAY_INDEX // N_TASK) from SWEEP_DIR's
pickles, and saves the result to server_utils.SAVE_DIR.
"""
import os
import sys
import pickle as pkl

import super_experiments as sxp
import server_utils as su

idx, n_task, sweep = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]

with open(os.path.join(sweep, 'task_%d.pkl' % (idx % n_task)), 'rb') as f:
    task_dict = pkl.load(f)
with open(os.path.join(sweep, 'model_%d.pkl' % (idx // n_task)), 'rb') as f:
    mod_dict = pkl.load(f)

exp = sxp.Experiment(task_dict['task'](**task_dict['args']),
                     mod_dict['model'](**mod_dict['args']))
print('task: ', task_dict['args'], '\nmodel:', mod_dict['args'], flush=True)

exp.run()
exp.save_experiment(su.SAVE_DIR)
print('saved', exp.file_suffix(), 'to', su.SAVE_DIR, flush=True)

os.makedirs(os.path.join(sweep, 'done'), exist_ok=True)     # for jobs/sweep_status.py
open(os.path.join(sweep, 'done', str(idx)), 'w').close()