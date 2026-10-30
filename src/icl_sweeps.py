"""
Choosing runs from a batch by their parameters, and plotting them on a grid.

    records, prm = su.load_experiments(task_args, mod_args, SAVE_DIR=SAVE_DIR)
    varying(prm)                                          # what differs between the runs
    plot_runs(records, prm, rows='k', cols='hint', where={'explicitness': 3})
    plot_heldout(records, prm, hue='num_trials', rows='k', cols='hint',
                 where={'explicitness': 3})

`where` fixes parameters, as {name: condition}, the condition being
    a value          runs with that value (a list-valued field, e.g. num_trials, as a tuple)
    a list or set    runs with any of these values
    a function       runs passing the test, e.g. lambda k: k >= 8
Every grid cell (and every line within it) must then come down to a single run; if it
doesn't, the error names the parameters still varying, which is the cue to fix them in `where`.
"""
import numpy as np
import pandas as pd
import scipy.special as spc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from icl_utils import xent, plot_icl, split_block, ylabel


# -- choosing runs ---------------------------------------------------------------------

def _hashable(v):
    """A parameter value that can be compared and grouped: lists and arrays as tuples,
    numpy scalars as plain numbers."""
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (list, tuple, np.ndarray)):
        return tuple(_hashable(x) for x in v)
    return v


def _key(v):
    """Sort key putting None first, then numbers, then strings, then anything else."""
    if isinstance(v, tuple):
        return (0, tuple(_key(x) for x in v))
    if v is None:
        return (1,)
    if isinstance(v, (bool, int, float)):
        return (2, v)
    if isinstance(v, str):
        return (3, v)
    return (4, repr(v))


def _order(vals):
    return sorted(set(vals), key=_key)


def _column(prm, name):
    if name not in prm:
        raise KeyError(f"no parameter {name!r}; there are {sorted(prm)}")
    return [_hashable(v) for v in prm[name]]       # a 2-d array (e.g. num_trials) by rows


def select(prm, where=None):
    """Indices of the runs (into records) matching `where` (see the module docstring)."""
    n = len(next(iter(prm.values())))
    keep = np.ones(n, bool)
    for name, cond in (where or {}).items():
        col = _column(prm, name)
        if callable(cond):
            keep &= np.array([bool(cond(v)) for v in col])
        elif isinstance(cond, (list, set)):
            ok = {_hashable(c) for c in cond}
            keep &= np.array([v in ok for v in col])
        else:
            keep &= np.array([v == _hashable(cond) for v in col])
    return np.where(keep)[0]


def varying(prm, runs=None, show=True):
    """{parameter: values} of the parameters that differ between `runs` (default all)."""
    runs = np.arange(len(next(iter(prm.values())))) if runs is None else runs
    out = {}
    for name in sorted(prm):
        col = _column(prm, name)
        vals = _order(col[r] for r in runs)
        if len(vals) > 1:
            out[name] = vals
    if show:
        for k, v in out.items():
            print(f'{k:>24}: {v}')
    return out


def _grid(prm, where, dims):
    """
    The runs matching `where`, arranged by the parameters in `dims` (None for a dimension
    not used): returns ({(value per dim): run}, [values of each dim]). Raises if two runs
    land in the same place, naming what still varies between them.
    """
    runs = select(prm, where)
    if len(runs) == 0:
        raise ValueError(f"no runs match {where}")
    cols = {d: _column(prm, d) for d in dims if d is not None}
    levels = [_order(cols[d][r] for r in runs) if d else [None] for d in dims]

    cells = {}
    for r in runs:
        cells.setdefault(tuple(cols[d][r] if d else None for d in dims), []).append(r)
    clash = {k: v for k, v in cells.items() if len(v) > 1}
    if clash:
        k, rs = next(iter(clash.items()))
        where_ = ', '.join(f'{d}={v}' for d, v in zip(dims, k) if d)
        rest = {n: v for n, v in varying(prm, rs, show=False).items() if n not in dims}
        raise ValueError(
            f"{len(rs)} runs at {where_ or 'the one cell'} (and {len(clash) - 1} other "
            f"places), differing in {rest or 'nothing (duplicate runs?)'}; "
            f"fix these in `where`")
    return {k: v[0] for k, v in cells.items()}, levels


def _label(name, val):
    return f'{name}={val}'


def _figure(levels, size, sharey):
    nr, nc = len(levels[0]), len(levels[1])
    fig, axs = plt.subplots(nr, nc, figsize=(size[0] * nc, size[1] * nr),
                            sharey=sharey, squeeze=False)
    return fig, axs


def _title(rows, cols, rv, cv):
    return ', '.join(_label(n, v) for n, v in [(rows, rv), (cols, cv)] if n)


# -- plot_icl on a grid ------------------------------------------------------------------

def plot_runs(records, prm, rows=None, cols=None, where=None, n_trl=50, refs=(),
              perplexity=False, kl=False, size=(5, 4), title=None, k_hide=None):
    """
    plot_icl for each run matching `where`, on a grid of `rows` x `cols` (parameter
    names; leave either out for a single row or column). Empty cells are hidden.
    perplexity, kl: as for plot_icl.
    title: f(record) for each panel's title, instead of the cell's parameter values.
    Returns the axes.
    """
    cells, levels = _grid(prm, where, (rows, cols))
    fig, axs = _figure(levels, size, sharey=False)
    for i, rv in enumerate(levels[0]):
        for j, cv in enumerate(levels[1]):
            ax = axs[i, j]
            if (rv, cv) not in cells:
                ax.set_visible(False)
                continue
            rec = records[cells[rv, cv]]
            m, a = rec['metrics'], rec['model_args']
            plot_icl(ax, m['logp_ctx'], m['data'], n_trl=n_trl, 
                     k_hide=a.get('k_hide') if k_hide is None else k_hide,
                     title=title(rec) if title else _title(rows, cols, rv, cv),
                     show_assistant=(a.get('mode') != 'raw'), refs=refs,
                     perplexity=perplexity, kl=kl)
    fig.tight_layout()
    return axs


# -- the held-out test, block by block ---------------------------------------------------

def heldout(record, refs=(), perplexity=False, kl=False):
    """
    Score of each block's held-out token (the first hidden token of its final trial, the
    star in plot_icl), for each sample: {'model': (samps, n_blocks), <ref>: (same)}.
    refs: {name: predictor}, as for plot_icl, scored at the same tokens.
    perplexity, kl: as for plot_icl (kl: -KL(Y || model) instead of -cross-entropy).
    """
    m = record['metrics']
    data, refs = m['data'], dict(refs)
    out = {k: [] for k in ['model', *refs]}
    for i, lpc in enumerate(m['logp_ctx']):
        Y, B, nblk = data['Y'][i], data['B'][i], len(data['blocks'][i])
        P = {k: pred(data, i) for k, pred in refs.items()}
        t = [np.argmax(B == b) + len(lpc[('user', b)]) - 1 for b in range(nblk)]
        best = spc.xlogy(Y[t], Y[t]).sum(-1) if kl else np.zeros(nblk)    # -H(Y)
        out['model'].append([xent(lpc[('assistant', b)][0], Y[t[b]]) - best[b] for b in range(nblk)])
        for k, Pk in P.items():
            Q = Pk[t]
            with np.errstate(invalid='ignore', divide='ignore'):
                out[k].append(spc.xlogy(Y[t], Q / Q.sum(-1, keepdims=True)).sum(-1) - best)
    f = (lambda v: np.exp(-v)) if perplexity else (lambda v: v)
    return {k: f(np.array(v, float)) for k, v in out.items()}


REF_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X']


def plot_heldout(records, prm, hue=None, rows=None, cols=None, where=None, refs=(),
                 perplexity=False, kl=False, cmap='viridis', size=(4, 3.5), sharey=True):
    """
    Held-out log-likelihood (or perplexity, or -KL; see plot_icl) of each block's test
    token against the block,
    mean and std over samples (as plot_icl): one line per value of `hue`, on a grid of `rows` x `cols`.
    refs are drawn dashed with hollow markers in the colour of their line.
    Returns the axes.
    """
    cells, levels = _grid(prm, where, (rows, cols, hue))
    hues = levels[2]
    colors = plt.get_cmap(cmap)(np.linspace(0, 0.9, len(hues))) if hue else ['k']
    fig, axs = _figure(levels, size, sharey)
    style = dict(ls='--', lw=0.8, ms=5, mfc='none')
    nmax = 0
    for i, rv in enumerate(levels[0]):
        for j, cv in enumerate(levels[1]):
            ax = axs[i, j]
            here = [(h, cells[rv, cv, hv]) for h, hv in enumerate(hues) if (rv, cv, hv) in cells]
            if not here:
                ax.set_visible(False)
                continue
            for h, r in here:
                s = heldout(records[r], refs, perplexity, kl)
                x = np.arange(1, s['model'].shape[1] + 1)
                nmax = max(nmax, len(x))
                ax.errorbar(x, np.nanmean(s['model'], 0), np.nanstd(s['model'], 0),
                            color=colors[h], marker='o', ms=6, lw=2, capsize=2)
                for k, name in enumerate(refs):
                    ax.plot(x, np.nanmean(s[name], 0), color=colors[h], alpha=0.8,
                            marker=REF_MARKERS[k % len(REF_MARKERS)], **style)
            ax.set_title(_title(rows, cols, rv, cv))
            ax.set_xlabel('block')
            if j == 0:
                ax.set_ylabel('held-out ' + ylabel(perplexity, kl))
    for ax in axs.flat:
        ax.set_xticks(np.arange(1, nmax + 1))
    # one legend for the whole grid, in the first panel: every hue, then every reference
    handles = [Line2D([], [], color=colors[h], marker='o', lw=2, label=_label(hue, hv))
               for h, hv in enumerate(hues)] if hue else []
    handles += [Line2D([], [], color='0.4', marker=REF_MARKERS[k % len(REF_MARKERS)],
                       label=name, **style) for k, name in enumerate(refs)]
    if handles:
        next(ax for ax in axs.flat if ax.get_visible()).legend(handles=handles, fontsize='small')
    fig.tight_layout()
    return axs


# -- one run, token by token -----------------------------------------------------------

def trace(record, refs=()):
    """
    Everything plot_icl plots for one run, plus what's needed to see where its numbers
    come from: a DataFrame with one row per token of every sample.

        sample, block, trial   where the token is (trial within its block)
        pos                    position within its trial (the delimiter is the last)
        t                      index into data['X'][sample], data['Y'][sample]
        word                   the token, as the model saw it (that sample's word list)
        hidden                 whether it's among the tokens plot_icl shows (the last
                               k_hide of the trial, or those after the first k_show)
        final                  whether it's in the block's final trial, scored from the
                               completion (the assistant span) rather than the context
        ll                     the plotted score, -cross-entropy against Y, renormalised
                               over the task alphabet (as xent does)
        n_valid                number of tokens Y allows
        p_valid                (renormalised) probability on the tokens Y allows
        top, p_top             the model's most likely task token and its probability
        task_mass              probability on the task alphabet at all, before
                               renormalising: when this is small, ll is a ratio of scraps
        kl                     KL(Y || model) over the task alphabet: -ll less the entropy
                               of Y, so 0 for the true distribution (plot_icl(kl=True)
                               plots -kl); the same as -ll wherever Y is deterministic
        ll_<ref>, kl_<ref>     each reference predictor's score, as in plot_icl

    plot_icl's curve for block b is the mean (and std) over samples of ll on the hidden
    rows, by trial: tr[tr.hidden].groupby(['block', 'trial']).ll.agg(['mean', 'std']).
    One difference: plot_icl scores the final trial's first hidden token from the last
    context row, this from the completion's first row; they're the same row in 'raw'
    and 'single' mode, but not in 'dialogue', where the context row predicts the template.
    """
    m, a = record['metrics'], record['model_args']
    data, refs = m['data'], dict(refs)
    k_show, k_hide = a.get('k_show'), a.get('k_hide')
    rows = []
    for i, lpc in enumerate(m['logp_ctx']):
        X, Y, B, blocks = data['X'][i], data['Y'][i], data['B'][i], data['blocks'][i]
        words = [w.replace('\u0120', ' ') for w in m['task_tokens'][i]] if 'task_tokens' in m else None
        P = {k: pred(data, i) for k, pred in refs.items()}
        for b, trials in enumerate(blocks):
            t0 = int(np.argmax(B == b))
            ctx, asst = lpc[('user', b)], lpc[('assistant', b)]
            n_ctx, comp = len(ctx) - 1, split_block(trials, k_show, k_hide)[1]
            n = sum(len(tr) for tr in trials)
            lp = np.full((n, ctx.shape[1]), np.nan)
            lp[:n_ctx] = ctx[:n_ctx]                        # row r of ctx predicts token r
            lp[n_ctx:n_ctx + len(comp)] = asst[:len(comp)]  # the completion; its delimiter unscored
            trial = np.repeat(np.arange(len(trials)), [len(tr) for tr in trials])
            pos = np.concatenate([np.arange(len(tr)) for tr in trials])
            real = np.repeat([len(tr) - 1 for tr in trials], [len(tr) for tr in trials])
            first = np.array([split_block([tr], k_show, k_hide)[0].size for tr in trials])
            ts = t0 + np.arange(n)
            Yb = Y[ts]
            with np.errstate(invalid='ignore', divide='ignore'):
                mass = np.exp(spc.logsumexp(lp, -1))
                p = np.exp(lp - spc.logsumexp(lp, -1, keepdims=True))
                top = np.nanargmax(np.nan_to_num(lp, nan=-np.inf), -1)
                row = {'sample': i, 'block': b, 'trial': trial, 'pos': pos, 't': ts,
                       'word': [words[x] for x in X[ts]] if words else X[ts],
                       'hidden': (pos >= first[trial]) & (pos < real),
                       'final': trial == len(trials) - 1,
                       'll': xent(lp, Yb), 'best': spc.xlogy(Yb, Yb).sum(-1),
                       'n_valid': (Yb > 0).sum(-1),
                       'p_valid': (p * (Yb > 0)).sum(-1),
                       'top': [words[x] for x in top] if words else top,
                       'p_top': p[np.arange(n), top], 'task_mass': mass}
                for k, Pk in P.items():
                    Q = Pk[ts]
                    row['ll_' + k] = spc.xlogy(Yb, Q / Q.sum(-1, keepdims=True)).sum(-1)
                    row['kl_' + k] = row['best'] - row['ll_' + k]
            row['kl'] = row['best'] - row['ll']
            df = pd.DataFrame(row).drop(columns='best')
            df.loc[np.isnan(lp).all(-1), ['ll', 'kl', 'p_valid', 'top', 'p_top', 'task_mass']] = np.nan
            rows.append(df)
    return pd.concat(rows, ignore_index=True)
    