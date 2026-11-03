# %% Setup

from functools import partial

import numpy as np
import matplotlib.pyplot as plt

import server_utils as su
from icl_tasks import RelationTask
from icl_model import ICLModel
from icl_utils import true_dist, uniform_block, ngram
import icl_sweeps as isw
import icl_utils

SAVE_DIR = su.SAVE_DIR          # $SCRATCH/icl_results, or $ICL_RESULTS if set

# %% Load: leave keys out to match any value

k = su.Set([8])
task_args = {'task': RelationTask,
             'k': k,
             'd': 2,
             'degree': 0.9,
        }

mod_args = {'model': ICLModel,
            'model_name': ['Qwen/Qwen3.8-27B', 'google/gemma-4-31B'],
        
        }

# su.fetch_results()      # on the laptop: copy new results over from the cluster first
records, prm = su.load_experiments(task_args, mod_args, SAVE_DIR=SAVE_DIR)
isw.varying(prm);         # what differs between the runs, and over what values

# %% Which runs: fix everything except the grid (and line) dimensions

these = {}
these |= {'model_name': 'Qwen/Qwen3.8-27B'}
# these |= {'model_name': 'google/gemma-4-31B'}
these |= {'mode': 'raw'}
these |= {'commutes': [0, np.inf]}
# these |= {'explicitness': [0,3]}
these |= {'describe_task': [True, False]}      # describe_format and answer_only go with it
these |= {'hint': False}
these |= {'share_toks': [False, {'action'}, {'node'}, True]}
these |= {}
these |= {'k': 8}
# these |= {'tokens': [icl_utils.Vocabulary('nouns'), icl_utils.Vocabulary('chars')]}
these |= {'tokens': icl_utils.Vocabulary('nouns')}
# these |= {'num_trials': (None, None)}                 # a list-valued field as a tuple
# these |= {'num_trials': [(64, 64), (128, 128)]}       # a list: any of these
# these |= {'k': lambda k: k >= 8}

# rows = 'k'
# rows = 'explicitness'
# rows = 'share_toks'
# rows = 'tokens'
rows = 'describe_task'
# rows = None

# cols = 'explicitness'
# cols = 'hint'
# cols = 'commutes'
cols = 'share_toks'
# cols = 'num_trials'
# cols = None
# cols = 'describe_task'

lines = 'commutes'
# lines = 'num_trials'
# lines = 'explicitness'
# lines = 'describe_task'
# lines = 'tokens'
# lines = None

refs = {
        # 'uniform': uniform_block,
        # 'true': true_dist,
        'bigram': partial(ngram, order=1, alpha=0.5),
        }

kl = True                # -KL(Y || model) instead of -cross-entropy (0 = the true distribution)
# kl = False

# isw.varying(prm, isw.select(prm, these));     # what's left varying after `these`

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
