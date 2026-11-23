import os

import numpy as np

import icl_tasks
from icl_model import ICLModel
import icl_utils
import server_utils as su

SAVE_DIR = su.SAVE_DIR          # $SCRATCH/icl_results, or $ICL_RESULTS if set

# %% What to run: lists are swept, tuples are matched 1-1 (as in send_to_server)
# The task's fields describe the grammar (see the end of icl_tasks.py), which is made
# inside the job. Fields can be swept, linked and loaded by one at a time, e.g.
#   'k': list(range(5, 50, 5))                 sweep k
#   'k': (5, 10, 20), 'degree': (2, 2, 3)      (k, degree) pairs, not all 9 combinations
#   'graph_seed': [0, 1, 2]                    three different graphs of each size

# k = su.Set([8, 16])

# k = su.Set([4,5,6,7])

task_args = {
             # 'task': icl_tasks.RelationTask,
             'task': icl_tasks.ExhaustiveRelationTask,
             'k': 4,
             'd': 1,
            #  'degree': 0.5,
             'edges': 4,
             'index': [0,1,2],
             'all_edges': True, 
             # 'index': [2,4],
             # 'target': [0,1,2],
            # 'index': 0,
             'connected': None, 
             'keep_connected': False,
            #  'backtrack': False,
             # 'graph_seed': 0,                  # same graph for every prompt condition below
             'num_trials': [i*[None] for i in [1,2,4]],
             # 'length': 1,
             'samps': 8,
             'holdout': True,
             'max_tok': np.inf, 
             # the prompt, tailored to the task (see GrammarTask); left out, there is none. E.g.
             # 'explicitness': [0, 2],        # 0 none, 1 vague, 2 moderate, 3 explicit
            #  'explicitness': 2, 
             'explicitness': (2, 1),
             'describe_format': True,
            #  'describe_format': (True, False), 
            #  'describe_task': (True, False),
             'describe_task': True,
             'describe_vocab': True,
             # 'hint': [0, 2],
             'hint': 0,
             'answer_only': True,
             # 'answer_only': True,
             'share_toks': False,
             'show': [None, {'node'}],
            #  'share_toks': False,
             'seed': 0,
             }

# %% Check what a prompt says for a task, without a model (symbols shown as w1, w2, ...)
# import prompts
# prompts.preview(RelationTask(k=10, degree=2, num_trials=[None, None], samps=1, holdout=True,
#                              max_tok=np.inf, explicitness=3, describe_format=True,
#                              describe_task=True, hint=True, answer_only=True), mode='dialogue')

mod_args = {'model': ICLModel,
            # 'model_name': ['Qwen/Qwen3.5-35B-A3B-Base', 'google/gemma-4-31B'],
            'model_name': ['Qwen/Qwen3.8-27B','google/gemma-4-31B-it','thinkingmachines/Inkling-2-Mini'],
            # 'model_name': 'thinkingmachines/Inkling-2-Mini',
            # 'mode': ['dialogue', 'raw'],
            # 'mode': ('raw', 'dialogue'),
            'mode': 'dialogue',
            'think': [True, False],
            # 'think': (False, True),
            'k_hide': 1,
            'effort': 'low',
            # 'effort': None,
            'thinking_budget': 10000,
            'seed': 0,
            # 'equiv': (True, False),
            'equiv': True,
            # 'delim': [',', '\n'],
            'delim': ',',
            'group': [None, {'action','node'}, {'block'}],
            'categories': {'kmeans100'},
            # 'geometry': [None,
            #              Geometry(group={'action', 'node'}, categories={'hypernyms'}, within=2, between=0.5, beta=2),
            #              Geometry(group={'block'}, categories={'hypernyms'}, within=2, between=0.5, beta=2)],
            # 'tokens': [icl_utils.Vocabulary('nouns'), icl_utils.Vocabulary('words', content=True)],
            'tokens': icl_utils.Vocabulary('words', content=True),
            }
