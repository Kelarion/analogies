"""
Choosing runs from a batch by their parameters, and plotting them on a grid.

    records, prm = su.load_experiments(task_args, mod_args, SAVE_DIR=SAVE_DIR)
    varying(prm)                                          # what differs between the runs
    plot_runs(records, prm, rows='k', cols='hint', where={'explicitness': 3})
    plot_heldout(records, prm, hue='num_trials', rows='k', cols='hint',
                 where={'explicitness': 3})

`where` fixes parameters, as {name: condition}, the condition being
    a value          runs with that value (a list-valued field, e.g. num_trials, as a tuple;
                     a set-valued one as a set, e.g. share_toks={'action'})
    a list           runs with any of these values, e.g. k=[4, 8] or share_toks=[False, {'action'}]
    a function       runs passing the test, e.g. lambda k: k >= 8
As when sending runs, a list means alternatives and anything else is a value.
Every grid cell (and every line within it) must then come down to a single run; if it
doesn't, the error names the parameters still varying, which is the cue to fix them in `where`.
"""
import json

import numpy as np
import pandas as pd
import scipy.special as spc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from icl_utils import xent, plot_icl, split_block, ylabel, spread, draw_traces


# -- choosing runs ---------------------------------------------------------------------

def _hashable(v):
    """A parameter value that can be compared and grouped: lists and arrays as tuples,
    sets as frozensets, numpy scalars as plain numbers."""
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (list, tuple, np.ndarray)):
        return tuple(_hashable(x) for x in v)
    if isinstance(v, (set, frozenset)):
        return frozenset(_hashable(x) for x in v)
    return v


def _key(v):
    """Sort key putting None first, then numbers, then strings, then sets (by their sorted
    elements, so the order doesn't depend on the hash seed), then anything else."""
    if isinstance(v, tuple):
        return (0, tuple(_key(x) for x in v))
    if v is None:
        return (1,)
    if isinstance(v, (bool, int, float)):
        return (2, v)
    if isinstance(v, str):
        return (3, v)
    if isinstance(v, frozenset):
        return (4, tuple(sorted(_key(x) for x in v)))
    return (5, repr(v))


def _show(v):
    """A value as printed in labels and listings: sets in sorted order, as {a, b}."""
    if isinstance(v, frozenset):
        return '{' + ', '.join(_show(x) for x in sorted(v, key=_key)) + '}'
    return str(v)


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
        elif isinstance(cond, list):
            ok = {_hashable(c) for c in cond}
            keep &= np.array([v in ok for v in col])
        else:
            if isinstance(cond, (set, frozenset)) and not any(isinstance(v, frozenset) for v in col):
                raise ValueError(f"where={{{name!r}: {cond!r}}}: a set is a value, and no {name} is a "
                                 f"set; for runs with any of these values, use a list: {sorted(cond, key=_key)}")
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
            print(f"{k:>24}: [{', '.join(_show(x) for x in v)}]")
    return out


def _digest_form(v):
    """A value as the save names see it (su.canon), so values that would be saved under
    different names come out different."""
    try:
        import server_utils as su
    except ImportError:                                     # a rough stand-in
        return repr(v)
    try:
        return su._key(v)
    except (TypeError, ValueError) as e:                    # can't be named: say why
        return f'<{e}>'


def _type(v):
    t = type(v)
    return t.__name__ if t.__module__ == 'builtins' else f'{t.__module__}.{t.__name__}'


def compare_runs(records, runs, show=True):
    """
    The arguments that differ between `runs` (indices into records) as the saved file
    names see them, i.e. by how arg_digest serialises each value rather than by ==, side
    by side for the task and the model (so a name on both sides, like seed, is compared
    on each). Runs that look identical in prm but were saved as separate files differ
    here, e.g. in type: hint=False on one, hint=0 or np.False_ on the other.
    Returns {(side, name): [(value, type name) per run]}.
    """
    runs = [int(r) for r in runs]
    out = {}
    for side in ['task_args', 'model_args']:
        args = [records[r][side] for r in runs]
        for name in sorted(set().union(*args)):
            missing = object()
            vals = [a.get(name, missing) for a in args]
            forms = {'<absent>' if v is missing else _digest_form(v) for v in vals}
            if len(forms) > 1:
                out[side[:-5], name] = [('<absent>', '') if v is missing else
                                        (v, _type(v)) for v in vals]
    if show:
        try:
            from super_experiments import arg_digest
            print('runs', runs, 'in', [records[r].get('folder') for r in runs])
            print('  file suffix now:', [arg_digest(records[r]['task_args']) + '_' +
                                         arg_digest(records[r]['model_args']) for r in runs])
        except ImportError:
            pass
        for (side, name), vals in out.items():
            print(f'  {side} {name}: ' + ', '.join(f'{v!r} ({t})' for v, t in vals))
        if not out:
            print('  no difference in the arguments: the same run saved twice?')
    return out


def _grid(prm, where, dims, records=None):
    """
    The runs matching `where`, arranged by the parameters in `dims` (None for a dimension
    not used): returns ({(value per dim): run}, [values of each dim]). Raises if two runs
    land in the same place, naming what still varies between them (and, given records,
    what differs in how they were saved, when nothing differs in prm).
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
        where_ = ', '.join(_label(d, v) for d, v in zip(dims, k) if d)
        rest = {n: v for n, v in varying(prm, rs, show=False).items() if n not in dims}
        if rest:
            why = f"differing in {rest}; fix these in `where`"
        else:
            saved = compare_runs(records, rs, show=False) if records is not None else {}
            why = ("identical in prm but saved separately, differing in " +
                   '; '.join(f"{side} {n}: " + ' vs '.join(f'{v!r} ({t})' for v, t in vals)
                             for (side, n), vals in saved.items())
                   if saved else "identical in prm (duplicate runs?)")
            why += f"; see compare_runs(records, {[int(r) for r in rs]})"
        raise ValueError(f"{len(rs)} runs at {where_ or 'the one cell'} (and "
                         f"{len(clash) - 1} other places), {why}")
    return {k: v[0] for k, v in cells.items()}, levels


def _label(name, val):
    return f'{name}={_show(val)}'


def _figure(levels, size, sharey):
    nr, nc = len(levels[0]), len(levels[1])
    fig, axs = plt.subplots(nr, nc, figsize=(size[0] * nc, size[1] * nr),
                            sharey=sharey, squeeze=False)
    return fig, axs


def _title(rows, cols, rv, cv):
    return ', '.join(_label(n, v) for n, v in [(rows, rv), (cols, cv)] if n)


# -- plot_icl on a grid ------------------------------------------------------------------

def plot_runs(records, prm, rows=None, cols=None, where=None, n_trl=50, refs=(),
              perplexity=False, kl=False, err='std', traces=False, renorm=True, size=(5, 4),
              title=None):
    """
    plot_icl for each run matching `where`, on a grid of `rows` x `cols` (parameter
    names; leave either out for a single row or column). Empty cells are hidden.
    perplexity, kl, err, traces, renorm: as for plot_icl.
    title: f(record) for each panel's title, instead of the cell's parameter values.
    Returns the axes.
    """
    cells, levels = _grid(prm, where, (rows, cols), records)
    fig, axs = _figure(levels, size, sharey=False)
    for i, rv in enumerate(levels[0]):
        for j, cv in enumerate(levels[1]):
            ax = axs[i, j]
            if (rv, cv) not in cells:
                ax.set_visible(False)
                continue
            rec = records[cells[rv, cv]]
            m, a = rec['metrics'], rec['model_args']
            plot_icl(ax, m['logp_ctx'], m['data'], n_trl=n_trl, k_hide=a.get('k_hide'),
                     title=title(rec) if title else _title(rows, cols, rv, cv),
                     show_assistant=(a.get('mode') != 'raw'), refs=refs,
                     perplexity=perplexity, kl=kl, err=err, traces=traces,
                     renorm=renorm)
    fig.tight_layout()
    return axs


# -- the held-out test, block by block ---------------------------------------------------

def heldout(record, refs=(), perplexity=False, kl=False, renorm=True):
    """
    Score of each block's held-out token (the first hidden token of its final trial, the
    star in plot_icl), for each sample: {'model': (samps, n_blocks), <ref>: (same)}.
    refs: {name: predictor}, as for plot_icl, scored at the same tokens.
    perplexity, kl, renorm: as for plot_icl (kl: -KL(Y || model) instead of
    -cross-entropy; renorm=False: the model's probabilities over the whole vocabulary).
    """
    m = record['metrics']
    data, refs = m['data'], dict(refs)
    out = {k: [] for k in ['model', *refs]}
    for i, lpc in enumerate(m['logp_ctx']):
        Y, B, nblk = data['Y'][i], data['B'][i], len(data['blocks'][i])
        P = {k: pred(data, i) for k, pred in refs.items()}
        t = [np.argmax(B == b) + len(lpc[('user', b)]) - 1 for b in range(nblk)]
        best = spc.xlogy(Y[t], Y[t]).sum(-1) if kl else np.zeros(nblk)    # -H(Y)
        out['model'].append([xent(lpc[('assistant', b)][0], Y[t[b]], renorm) - best[b]
                             for b in range(nblk)])
        for k, Pk in P.items():
            Q = Pk[t]
            with np.errstate(invalid='ignore', divide='ignore'):
                out[k].append(spc.xlogy(Y[t], Q / Q.sum(-1, keepdims=True)).sum(-1) - best)
    f = (lambda v: np.exp(-v)) if perplexity else (lambda v: v)
    return {k: f(np.array(v, float)) for k, v in out.items()}


REF_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X']


def plot_heldout(records, prm, hue=None, rows=None, cols=None, where=None, refs=(),
                 perplexity=False, kl=False, err='std', traces=False, renorm=True, cmap='viridis',
                 size=(4, 3.5), sharey=True):
    """
    Held-out log-likelihood (or perplexity, or -KL; see plot_icl) of each block's test
    token against the block, summarised over samples as `err` says (see plot_icl: mean
    and std by default), one line per value of `hue`, on a grid of `rows` x `cols`.
    traces: also draw each sample's line, thin and translucent (True, or the opacity).
    renorm: as for plot_icl (False: the model's probabilities over the whole vocabulary).
    refs are drawn dashed with hollow markers in the colour of their line.
    Returns the axes.
    """
    cells, levels = _grid(prm, where, (rows, cols, hue), records)
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
                s = heldout(records[r], refs, perplexity, kl, renorm)
                x = np.arange(1, s['model'].shape[1] + 1)
                nmax = max(nmax, len(x))
                if traces:
                    draw_traces(ax, x, s['model'], colors[h], traces)
                mu, e = spread(s['model'], err)
                ax.errorbar(x, mu, e, color=colors[h], marker='o', ms=6, lw=2, capsize=2)
                for k, name in enumerate(refs):
                    ax.plot(x, spread(s[name], err)[0], color=colors[h], alpha=0.8,
                            marker=REF_MARKERS[k % len(REF_MARKERS)], **style)
            ax.set_title(_title(rows, cols, rv, cv))
            ax.set_xlabel('block')
            if j == 0:
                ax.set_ylabel('held-out ' + ylabel(perplexity, kl, renorm))
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
        ll                     -cross-entropy against Y, with the model's probabilities
                               normalised over the whole vocabulary (not renormalised to
                               the task alphabet: plot_icl's default is ll - log(task_mass))
        n_valid                number of tokens Y allows
        p_valid                probability on the tokens Y allows
        top, p_top             the model's most likely task token and its probability
        task_mass              probability on the task alphabet at all; dividing p_valid
                               or p_top by it renormalises them
        kl                     KL(Y || model): -ll less the entropy of Y, so 0 for the true
                               distribution; the same as -ll wherever Y is deterministic.
                               kl + log(task_mass) is plot_icl(kl=True)'s default
        ll_<ref>, kl_<ref>     each reference predictor's score, as in plot_icl
        top_any, p_top_any     the most likely token over the whole vocabulary, and its
                               probability (not renormalised), e.g. '<|im_end|>'
        top_text, p_top_text   the same over text only, without special tokens (runs
                               saved before these were recorded leave all four out)

    All probabilities are over the whole vocabulary, as plot_icl(renorm=False) scores
    them. plot_icl's curve for block b is the mean (and std) over samples of ll on the
    hidden rows, by trial, renormalised or not:
        tr['ll_task'] = tr.ll - np.log(tr.task_mass)        # plot_icl's default
        tr[tr.hidden].groupby(['block', 'trial']).ll_task.agg(['mean', 'std'])
    One difference: plot_icl scores the final trial's first hidden token from the last
    context row, this from the completion's first row; they're the same row in 'raw'
    and 'single' mode, but not in 'dialogue', where the context row predicts the template.
    """
    m, a = record['metrics'], record['model_args']
    data, refs = m['data'], dict(refs)
    k_show, k_hide = a.get('k_show'), a.get('k_hide')
    has_top = 'top_ids' in m
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
            if has_top:                                     # the same rows of the top tokens
                tid, tlp = np.full((n, 2), -1), np.full((n, 2), np.nan)
                for arr, src in [(tid, m['top_ids'][i]), (tlp, m['top_lp'][i])]:
                    arr[:n_ctx] = src[('user', b)][:n_ctx]
                    arr[n_ctx:n_ctx + len(comp)] = src[('assistant', b)][:len(comp)]
            trial = np.repeat(np.arange(len(trials)), [len(tr) for tr in trials])
            pos = np.concatenate([np.arange(len(tr)) for tr in trials])
            real = np.repeat([len(tr) - 1 for tr in trials], [len(tr) for tr in trials])
            first = np.array([split_block([tr], k_show, k_hide)[0].size for tr in trials])
            ts = t0 + np.arange(n)
            Yb = Y[ts]
            with np.errstate(invalid='ignore', divide='ignore'):
                mass = np.exp(spc.logsumexp(lp, -1))
                p = np.exp(lp)                              # over the whole vocabulary
                top = np.nanargmax(np.nan_to_num(lp, nan=-np.inf), -1)
                row = {'sample': i, 'block': b, 'trial': trial, 'pos': pos, 't': ts,
                       'word': [words[x] for x in X[ts]] if words else X[ts],
                       'hidden': (pos >= first[trial]) & (pos < real),
                       'final': trial == len(trials) - 1,
                       'll': xent(lp, Yb, renorm=False), 'best': spc.xlogy(Yb, Yb).sum(-1),
                       'n_valid': (Yb > 0).sum(-1),
                       'p_valid': (p * (Yb > 0)).sum(-1),
                       'top': [words[x] for x in top] if words else top,
                       'p_top': p[np.arange(n), top], 'task_mass': mass}
                if has_top:
                    word_of = lambda j: m['top_vocab'].get(int(j)) if j >= 0 else None
                    for c, name in enumerate(['any', 'text']):
                        row['top_' + name] = [word_of(j) for j in tid[:, c]]
                        row['p_top_' + name] = np.exp(tlp[:, c])
                for k, Pk in P.items():
                    Q = Pk[ts]
                    row['ll_' + k] = spc.xlogy(Yb, Q / Q.sum(-1, keepdims=True)).sum(-1)
                    row['kl_' + k] = row['best'] - row['ll_' + k]
            row['kl'] = row['best'] - row['ll']
            df = pd.DataFrame(row).drop(columns='best')
            df.loc[np.isnan(lp).all(-1), ['ll', 'kl', 'p_valid', 'top', 'p_top', 'task_mass']] = np.nan
            rows.append(df)
    return pd.concat(rows, ignore_index=True)
    