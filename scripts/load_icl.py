# %% Setup

from functools import partial

import numpy as np
import matplotlib.pyplot as plt

import server_utils as su
import icl_tasks
from icl_model import ICLModel
from icl_utils import true_dist, uniform_block, ngram
import icl_sweeps as isw
import icl_utils
import plotting as tpl
import automata as auto
import networkx as nx

SAVE_DIR = su.SAVE_DIR          # $SCRATCH/icl_results, or $ICL_RESULTS if set

# %% Load: leave keys out to match any value

# k = su.Set([8])
# k = su.Set([3,4,5,6,7])
k = [4]

# task_args = {'task': RelationTask,
#              'k': k,
#              'd': [1,2],
#              'edges': k,
#         #      'd': 2,
#         #      'degree': 0.9,
#               'connected': None,
#         }

# mod_args = {'model': ICLModel,
#             'model_name': ['Qwen/Qwen3.8-27B', 'google/gemma-4-31B'],
        
#         }

task_args = {
             # 'task': icl_tasks.RelationTask,
             'task': icl_tasks.ExhaustiveRelationTask,
             'k': 4,
             'd': 1,
            #  'degree': 0.5,
             'edges': 4,
            #  'index': [0,1,2,3,4],
            'index': 0,
             'connected': None, 
             'keep_connected': False,
            #  'backtrack': False,
             # 'graph_seed': 0,                  # same graph for every prompt condition below
             # 'num_trials': [[None, None], [k**2, k**2], [2*k**2, 2*k**2]],       # a list-valued field needs wrapping: [[...]]
             # 'num_trials': [2*[8*k**2], 4*[4*k**2], 8*[2*k**2], 16*[k**2]],
            #  'num_trials': [i*[None] for i in [1,2,4,8]],
            #  'num_trials': [[100, 100, 100, 100]],
            #  'num_trials': 4*[2*k**2],
             'num_trials': [1*[None], 2*[None]],
             # 'length': 1,
             'samps': 1,
             'holdout': True, 
             'max_tok': np.inf, 
             # the prompt, tailored to the task (see GrammarTask); left out, there is none. E.g.
            #  'explicitness': [0, 1, 2, 3],        # 0 none, 1 vague, 2 moderate, 3 explicit
             'explicitness': 2,
            #  'explicitness': (2, 0),
             'describe_format': True,
            #  'describe_format': (True, False), 
            #  'describe_task': (True, False),
             'describe_task': True,
            #  'describe_vocab': (True,False),
            #  'hint': [0, 1, 2],
            'hint': 0,
            #  'answer_only': (True, False),
            'answer_only': True,
            #  'share_toks': [False, {'action'}, {'node'}],
             'share_toks': False,
             'seed': 0,
             }

mod_args = {'model': ICLModel,
            # 'model_name': ['Qwen/Qwen3.8-27B', 'google/gemma-4-31B'],
            # 'model_name': ['Qwen/Qwen3.8-27B', 'google/gemma-4-31B-it'],
            'model_name': 'google/gemma-4-31B-it',
            # 'mode': ['dialogue', 'raw'],
            # 'mode': ('raw', 'dialogue'),
            'mode': 'dialogue',
            'think': True,
            # 'think': (False, True),
            'k_hide': 1,
            # 'effort': 'low',
            'effort': None,
            'thinking_budget': 10000,
            'seed': 0,
            # 'equiv': (True, False),
            'equiv': True,
            'delim': ',',
            # 'delim': ',',
            'tokens': icl_utils.Vocabulary('nouns'),
            }

# %% Check what a prompt says for a task, without a model (symbols shown as w1, w2, ...)

# su.fetch_results()      # on the laptop: copy new results over from the cluster first
records, prm = su.load_experiments(task_args, mod_args, SAVE_DIR=SAVE_DIR)
isw.varying(prm);         # what differs between the runs, and over what values

# %% Which runs: fix everything except the grid (and line) dimensions

these = {}
# these |= {'model_name': 'Qwen/Qwen3.8-27B'}
these |= {'model_name': ['google/gemma-4-31B','Qwen/Qwen3.8-27B']}
these |= {'mode': 'raw'}
# these |= {'commutes': 0}
# these |= {'describe_task': False}
# these |= {'explicitness': [0,3]}
these |= {'describe_task': [True, False]}      # describe_format and answer_only go with it
these |= {'hint': 0}
# these |= {'share_toks': [False, {'action'}, {'node'}, True]}
these |= {'share_toks': False}
these |= {'k': 6}
# these |= {'tokens': [icl_utils.Vocabulary('nouns'), icl_utils.Vocabulary('chars')]}
# these |= {'tokens': icl_utils.Vocabulary('nouns')}
# these |= {'num_trials': (None, None)}                 # a list-valued field as a tuple
# these |= {'num_trials': [[72,72]]}       # a list: any of these
# these |= {'k': lambda k: k >= 8}
# these |= {'delim': ','}
these |= {'d':1}

# rows = 'k'
# rows = 'explicitness'
# rows = 'share_toks'
# rows = 'tokens'
rows = 'describe_task'
# rows = 'hint'
# rows = None

# cols = 'explicitness'
# cols = 'hint'
# cols = 'commutes'
# cols = 'share_toks'
cols = 'num_trials'
# cols = None
# cols = 'describe_task'

lines = 'model_name'
# lines = 'd'
# lines = 'num_trials'
# lines = 'commutes'
# lines = 'num_trials'
# lines = 'explicitness'
# lines = 'describe_task'
# lines = 'tokens'
# lines = None

refs = {
        'uniform': uniform_block,
        # 'true': true_dist,
        # 'bigram': partial(ngram, order=1, alpha=0.5),
        }

kl = True                # -KL(Y || model) instead of -cross-entropy (0 = the true distribution)
# kl = False

# isw.varying(prm, isw.select(prm, these));     # what's left varying after `these`

# %% Held-out log-likelihood of each block

isw.plot_heldout(records, prm, hue=lines, rows=rows, cols=cols, 
                 where=these, refs=refs, kl=kl,
                 err='iqr', traces=True, renorm=False)
plt.ylim([-5,0.1])
plt.show()

# %% The trial-by-trial curves (plot_icl), one run per panel
# Only rows and cols can vary here, so if `lines` is set, fix it to one value:
these_icl = these
# these_icl = these | {lines: (None, None)}

isw.plot_runs(records, prm, rows=rows, cols=cols, 
              where=these_icl | {'model_name': 'Qwen/Qwen3.8-27B'},
              n_trl=4000, refs=refs,
              kl=kl, renorm=False)
plt.show()

#%%

deez = {'hint': 0, 'model_name': 'Qwen/Qwen3.8-27B', 'share_toks': False, 'num_trials': [16*[16]]}

idx = isw.select(prm, these | deez)

foo = isw.trace(records[idx[0]])

isw.plot_heldout(records, prm, hue=lines, rows=rows, cols=cols, 
                 where=these | deez, refs=refs, kl=kl,
                 err='iqr', traces=0.5, renorm=False)
plt.ylim([-5,0.1])
plt.show()

#%%

heldout = foo['final']*foo['hidden']
plt.scatter(foo['block'][heldout], foo['ll'][heldout])

#%%

blk = 0
didit = np.where(foo['ll'][heldout][foo['block'][heldout]==blk] > -1)[0]

print(np.array(foo['ll'][heldout][foo['block'][heldout]==blk])[didit])

#%%
dis = 5

G = auto.as_graph(records[idx[0]]['metrics']['data']['automaton'][didit[dis]])
lbl = records[idx[0]]['metrics']['data']['code'][didit[dis]][blk]
G = nx.relabel_nodes(G, {i:n for i,n in enumerate(lbl[-4:])})
tpl.draw_multigraph(G, 'action', with_labels=True)

T = np.array(records[idx[0]]['metrics']['data']['blocks'][didit[dis]][blk])
print(T[-1])
