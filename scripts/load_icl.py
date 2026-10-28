# %% Setup
# CODE_DIR = 'C:/Users/mmall/OneDrive/Documents/github/analogies/src'

import os, sys
from functools import partial

# sys.path.append(CODE_DIR)

import numpy as np
import matplotlib.pyplot as plt

import server_utils as su
from icl_tasks import RelationTask
from icl_model import ICLModel
from icl_utils import true_dist, uniform_block, ngram
import icl_sweeps as isw

SAVE_DIR = su.SAVE_DIR          # $SCRATCH/icl_results, or $ICL_RESULTS if set

# %% Load: leave keys out to match any value

task_args = {'task': RelationTask,
             # 'k': su.Set([8, 16]),
             'd': 2,
             }

mod_args = {'model': ICLModel,
            'model_name': 'Qwen/Qwen3.8-27B',
            # 'mode': 'raw',
            }

# su.fetch_results()      # on the laptop: copy new results over from the cluster first
records, prm = su.load_experiments(task_args, mod_args, SAVE_DIR=SAVE_DIR)
isw.varying(prm);         # what differs between the runs, and over what values

# %% Which runs: fix everything except the grid (and line) dimensions

these = {}
these |= {'explicitness': [0,1,2,3]}
# these |= {'describe_task': True}      # describe_format and answer_only go with it
these |= {'hint': False}
these |= {'k': 8}
# these |= {'num_trials': (None, None)}                 # a list-valued field as a tuple
# these |= {'num_trials': [(64, 64), (128, 128)]}       # a list: any of these
# these |= {'k': lambda k: k >= 8}

# rows = 'k'
rows = 'explicitness'
# rows = None

# cols = 'hint'
cols = 'num_trials'
# cols = None

# lines = 'num_trials'
# lines = 'explicitness'
lines = 'describe_task'
# lines = None

refs = {
        'uniform': uniform_block,
        # 'true': true_dist,
        'bigram': partial(ngram, order=2, alpha=0.5),
        }

kl = True                # -KL(Y || model) instead of -cross-entropy (0 = the true distribution)
# kl = False

isw.varying(prm, isw.select(prm, these));     # what's left varying after `these`

# %% Held-out log-likelihood of each block

isw.plot_heldout(records, prm, hue=lines, rows=rows, cols=cols, where=these, refs=refs, kl=kl)
plt.show()

# %% The trial-by-trial curves (plot_icl), one run per panel
# Only rows and cols can vary here, so if `lines` is set, fix it to one value:
these_icl = these
# these_icl = these | {lines: (None, None)}

isw.plot_runs(records, prm, rows=rows, cols=cols, where=these_icl, n_trl=4000, refs=refs,
              kl=kl)
plt.show()
