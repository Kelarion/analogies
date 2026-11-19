"""
Choosing the shared hyperparameters of stored token categories (categories.METHODS) by
held-out likelihood, as an ordinary sweep through server_utils.send_to_server:

    import server_utils as su, bmf_sweep as bs
    su.send_to_server({'task': bs.HeldOut, 'seed': [0, 1]},
                      {'model': bs.BMFFit,
                       'model_name': ['Qwen/Qwen3.8-27B', 'google/gemma-3-27b-it'],
                       'method': ['semibmf', 'bipca'], 'dim': [100, 300],
                       'lr': [0.05, 0.1, 0.25], 'J_prior': ['none', 'boltzmann']},
                      name='bmf-hyperparameters', per_job=8)

Each combination fits one configuration to a model's embeddings of the fit vocabulary
(categories.FIT_TOKENS; a random `rows` of them, the same ones for every configuration given
the task's seed), rescaled as every stored fit is, and scores it with sbmf's
bi-cross-validation (bae_util.gabriel_bicv): the mean log-likelihood per held-out entry, over
a k_row x k_col grid of row and column blocks, each held out once, under common random
numbers. metrics['test_ll'] is the score (higher is better); metrics['train_ll'] the fit's.

Hyperparameters left as None take categories.METHODS' value, so a sweep starts from the
current table and varies what it lists. The annealing settings used are those a stored fit
would use (fit()'s own defaults filled in), so the winner can be copied straight into
METHODS. `reg` is each method's main regularizer (REG). gabriel_bicv assumes tree_reg = 0,
sbmf's default.

Fitting is CPU work (numba); the array template asks for a GPU it won't use, so for a large
sweep pass send_to_server a CPU template. The embeddings are cached once per model before the
array starts (BMFFit.prerequisites; that job does load the weights), so fits never load them.

Afterwards, rank() summarises (args, metrics) pairs, e.g. from server_utils.load_experiments.
"""
import time
from dataclasses import dataclass

import numpy as np

import super_experiments as sxp
import categories

REG = {'semibmf': 'weight_l2_reg', 'bipca': 'sparse_reg'}     # each method's main regularizer
FIT_DEFAULTS = {'initial_temp': 0, 'decay_rate': 1.0, 'period': 10, 'min_temp': 1, 'max_iter': 600}


@dataclass
class HeldOut(sxp.Task):
    """How configurations are scored: which rows, which held-out grid (see the module docstring)."""
    rows: int = 4000          # tokens fitted (a random subset of the fit vocabulary's)
    k_row: int = 2            # held-out grid: k_row x k_col blocks, each held out once
    k_col: int = 2
    fold_iter: int = 50       # M-steps for the held-out columns' weights
    n_samp: int = 10          # E-step draws averaged for the held-out rows' codes
    seed: int = 0

    def sample(self):
        return {'rows': self.rows, 'k_row': self.k_row, 'k_col': self.k_col,
                'fold_iter': self.fold_iter, 'n_samp': self.n_samp, 'seed': self.seed}


@dataclass
class BMFFit(sxp.Model):
    """One configuration of a method, fitted to a model's embeddings and scored on held-out data.
    None takes categories.METHODS' value."""
    model_name: str = 'Qwen/Qwen3.8-27B'
    embedding: str = 'canonical'
    method: str = 'semibmf'
    dim: int = 300
    lr: float = None
    max_iter: int = None
    initial_temp: float = None
    decay_rate: float = None
    period: int = None
    J_prior: str = None
    reg: float = None

    @classmethod
    def prerequisites(cls, args):
        """The model's cached embeddings (see server_utils.send_to_server)."""
        default = {k: getattr(cls, k) for k in ('model_name', 'embedding')}
        return [categories.embeddings_prerequisite(args.get('model_name', default['model_name']),
                                                   args.get('embedding', default['embedding']))]

    def settings(self):
        """(constructor fields, fit settings) this configuration uses."""
        model = {} if self.J_prior is None else {'J_prior': self.J_prior}
        if self.reg is not None:
            model[REG[self.method]] = self.reg
        given = {k: getattr(self, k) for k in ('lr', 'max_iter', 'initial_temp', 'decay_rate', 'period')
                 if getattr(self, k) is not None}
        fit = {**FIT_DEFAULTS, **categories.METHODS[self.method]['fit'], **given}
        return {**categories.METHODS[self.method]['model'], **model}, fit

    def fit(self, rows, k_row, k_col, fold_iter, n_samp, seed):
        from sbmf import bae_util
        t = time.time()
        _, X = categories.embeddings(self.model_name, self.embedding)
        idx = np.sort(np.random.default_rng(seed).choice(len(X), min(rows, len(X)), replace=False))
        X = categories.preprocess(X[idx])
        model_kw, fit_kw = self.settings()
        settings = categories.fit_settings(self.method, **fit_kw)     # a schedule name -> object
        model = categories.build(self.method, self.dim, **model_kw)
        anneal = {k: settings.pop(k) for k in FIT_DEFAULTS}
        train_ll, test_ll = bae_util.gabriel_bicv(model, X, k_row=k_row, k_col=k_col, seed=seed,
                                                  fold_iter=fold_iter, n_samp=n_samp, **anneal, **settings)
        self.metrics = {'train_ll': float(train_ll), 'test_ll': float(test_ll), 'rows': int(len(X)),
                        'model': model_kw, 'fit': fit_kw, 'seconds': round(time.time() - t, 1)}


def rank(results, by=('model_name', 'method', 'dim')):
    """Print configurations' held-out log-likelihood, best first within each group of `by`.
    results: (model args, metrics) pairs, as dicts."""
    groups = {}
    for args, m in results:
        groups.setdefault(tuple(args.get(k) for k in by), []).append((m['test_ll'], m['train_ll'], args, m))
    for key in sorted(groups, key=str):
        print(', '.join(f'{k}={v}' for k, v in zip(by, key)))
        for te, tr, args, m in sorted(groups[key], key=lambda r: -r[0]):
            print(f'    test {te:+.4f}  train {tr:+.4f}   model {m["model"]}   fit {m["fit"]}')