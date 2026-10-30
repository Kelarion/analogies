# %% Setup
import os

import numpy as np
import matplotlib.pyplot as plt

import automata as auto
import server_utils as su
from server_utils import Build
import super_experiments as sxp
from icl_tasks import RelationTask
from icl_model import ICLModel
from icl_utils import plot_icl

SAVE_DIR = su.SAVE_DIR          # $SCRATCH/icl_results, or $ICL_RESULTS if set

# %% What to run: lists are swept, tuples are matched 1-1 (as in send_to_server)
# The task's fields describe the grammar (see the end of icl_tasks.py), which is made
# inside the job. Fields can be swept, linked and loaded by one at a time, e.g.
#   'k': list(range(5, 50, 5))                 sweep k
#   'k': (5, 10, 20), 'degree': (2, 2, 3)      (k, degree) pairs, not all 9 combinations
#   'graph_seed': [0, 1, 2]                    three different graphs of each size

# k = su.Set([8, 16])
k = su.Set([8])
task_args = {'task': RelationTask,
             'k': k,
             'd': 2,
             'degree': 0.9,
            #  'strong': True,
             'strong': False,
            #  'backtrack': False,
             'graph_seed': 0,                  # same graph for every prompt condition below
             # 'num_trials': [[None, None], [k**2, k**2], [2*k**2, 2*k**2]],       # a list-valued field needs wrapping: [[...]]
            #  'num_trials': [2*[4*k**2], 4*[2*k**2], 8*[k**2]],
            #  'num_trials': [[1, 1, 1, 1, 1, 1]],
            #  'num_trials': 4*[2*k**2],
             'num_trials': [[None, None]],
             'length': 1,
             'samps': 20,
             'holdout': True, 
             'max_tok': np.inf, 
             # the prompt, tailored to the task (see GrammarTask); left out, there is none. E.g.
            #  'explicitness': [0, 1, 2, 3],        # 0 none, 1 vague, 2 moderate, 3 explicit
             'explicitness': 3,
             # 'explicitness': 0,
             'describe_format': True,
            #  'describe_format': (True, False), 
            #  'describe_task': (True, False),
             'describe_task': True,
             'describe_vocab': True,
            #  'hint': [False, True],
            #  'answer_only': False,
             'seed': 0,
             }

# %% Check what a prompt says for a task, without a model (symbols shown as w1, w2, ...)
# import prompts
# prompts.preview(RelationTask(k=10, degree=2, num_trials=[None, None], samps=1, holdout=True,
#                              max_tok=np.inf, explicitness=3, describe_format=True,
#                              describe_task=True, hint=True, answer_only=True), mode='dialogue')

mod_args = {'model': ICLModel,
            'model_name': 'Qwen/Qwen3.8-27B',
            'mode': 'dialogue',
            # 'mode': 'raw',
            # 'mode': 'single',
            # 'think': [True, False],
            'think': False,
            'k_hide': 1,
            'effort': 'low',
            'thinking_budget': 10_000,
            'seed': 0,
            'equiv': True,
            'delim': ['\n', ','],
            'tokens': ['any+space+ascii+content', 'words+space+content', 'chars+space'],
            }

# %% Either: submit as a Slurm job array, one job per combination. Works from the laptop
# (pickles copied to the cluster over ssh) or on the cluster; push/pull first either way.
# su.send_to_server(task_args, mod_args, verbose=True)            # or name='my-sweep'

# %% Or: run here, e.g. in an interactive session (each model is loaded once)
exps, _ = su.get_all_experiments(task_args, mod_args)
exps.sort(key=lambda e: e['model_args']['args'].get('model_name', ''))

# %%
for e in exps:
    exp = sxp.Experiment(e['task_args']['task'](**e['task_args']['args']),
                         e['model_args']['model'](**e['model_args']['args']))
    print(exp.task.args | exp.model.args, flush=True)
    exp.run()
    exp.save_experiment(SAVE_DIR)

# %% Load and plot (no model needed; leave keys out of the dicts to match any value)
# su.fetch_results()      # on the laptop: copy new results over from the cluster first
records, prm = su.load_experiments(task_args, mod_args, SAVE_DIR=SAVE_DIR)
fig, axs = plt.subplots(1, len(records), figsize=(5 * len(records), 4), squeeze=False)
for ax, rec in zip(axs[0], records):
    m, a = rec['metrics'], rec['model_args']
    plot_icl(ax, m['logp_ctx'], m['data'], n_trl=50, k_hide=a['k_hide'],
             title=f"{a['mode']}, think={a['think']}", show_assistant=(a['mode'] != 'raw'))
plt.tight_layout()
plt.show()

#%%
from functools import partial
from icl_utils import true_dist, uniform_block, ngram

refs = {
        'uniform': uniform_block,
        # 'true': true_dist,
        'bigram': partial(ngram, order=2, alpha=1),
        }

records, prm = su.load_experiments(task_args, mod_args, SAVE_DIR=SAVE_DIR)
fig, axs = plt.subplots(1, len(records), figsize=(5 * len(records), 4), squeeze=False)
for ax, rec in zip(axs[0], records):
    m, a = rec['metrics'], rec['model_args']
    plot_icl(ax, m['logp_ctx'], m['data'], n_trl=15, k_hide=1, kl=True,
             title=f"{a['mode']}, think={a['think']}", show_assistant=(a['mode'] != 'raw'))
plt.tight_layout()
plt.show()

#%%
refs = {
        'uniform': uniform_block,
        # 'true': true_dist,
        'bigram': partial(ngram, order=1, alpha=1),
        }

isw.plot_runs(records, prm, rows='tokens', cols='delim', n_trl=15, refs=refs,
              kl=True)
plt.show()
