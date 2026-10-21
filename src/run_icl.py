# %% Setup
import os

import numpy as np
import matplotlib.pyplot as plt

import automata as auto
import server_utils as su
import super_experiments as sxp
from icl_tasks import AutomatonTask
from icl_model import ICLModel
from icl_utils import plot_icl

SAVE_DIR = su.SAVE_DIR          # $SCRATCH/icl_results, or $ICL_RESULTS if set

# %% What to run: lists are swept, tuples are matched 1-1 (as in send_to_server)
task_args = {'task': AutomatonTask,
             'grammar': auto.FSA(auto.cycle(5), length=1),
             'num_trials': [[200]],       # a list-valued field needs wrapping: [[...]]
             'samps': 20,
             'holdout': True,
             'max_tok': np.inf,
             'seed': 0}

mod_args = {'model': ICLModel,
            'model_name': os.environ.get('MODEL', 'Qwen/Qwen3.8-27B'),
            'mode': 'dialogue',
            'think': [True, False],
            'k_hide': 1,
            'effort': 'low',
            'seed': 0}

# %% Either: submit as a Slurm job array, one job per combination (from the cluster)
# su.send_to_server(task_args, mod_args, verbose=True)            # or name='my-sweep'

# %% Or: run here, e.g. in an interactive session (each model is loaded once)
exps, _ = su.get_all_experiments(task_args, mod_args)
exps.sort(key=lambda e: e['model_args']['args'].get('model_name', ''))
for e in exps:
    exp = sxp.Experiment(e['task_args']['task'](**e['task_args']['args']),
                         e['model_args']['model'](**e['model_args']['args']))
    print(exp.task.args | exp.model.args, flush=True)
    exp.run()
    exp.save_experiment(SAVE_DIR)

# %% Load and plot (no model needed; leave keys out of the dicts to match any value)
records, prm = su.load_experiments(task_args, mod_args, SAVE_DIR=SAVE_DIR)
fig, axs = plt.subplots(1, len(records), figsize=(5 * len(records), 4), squeeze=False)
for ax, rec in zip(axs[0], records):
    m, a = rec['metrics'], rec['model_args']
    plot_icl(ax, m['logp_ctx'], m['data'], n_trl=50, k_hide=a['k_hide'],
             title=f"{a['mode']}, think={a['think']}", show_assistant=(a['mode'] != 'raw'))
plt.tight_layout()
plt.show()