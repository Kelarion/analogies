"""
Choosing runs from a batch by their parameters, and plotting them on a grid.

    records, prm = su.load_experiments(task_args, mod_args, SAVE_DIR=SAVE_DIR)
    varying(prm)                                          # what differs between the runs
    runs(records).value_counts(['k', 'hint'])             # which combinations exist
    table(records, rows='test')                           # every run's scores, one DataFrame
    plot_runs(records, prm, rows='k', cols='hint', where={'explicitness': 3})
    plot_heldout(records, prm, hue='num_trials', rows='k', cols='hint',
                 where={'explicitness': 3})

Restricting to well-defined tests: keep='fixed' (plot_runs, plot_heldout, heldout,
trace) leaves out each block whose test answer some symmetry of its training graph
moves (see fixed_answers), so only answers that structure alone identifies count.

The kernel-model references are in icl_utils, beside ngram (kernel_ref, cue_ref,
copy_or_sink, and KERNEL_REFS for the ones discussed):
    plot_heldout(records, prm, hue='k', refs=icl_utils.KERNEL_REFS, keep='fixed')

`where` fixes parameters, as {name: condition}, the condition being
    a value          runs with that value (a list-valued field, e.g. num_trials, as a tuple;
                     a set-valued one as a set, e.g. share_toks={'action'})
    a list           runs with any of these values, e.g. k=[4, 8] or share_toks=[False, {'action'}]
    a function       runs passing the test, e.g. lambda k: k >= 8
As when sending runs, a list means alternatives and anything else is a value.
Every grid cell (and every line within it) must then come down to a single run; if it
doesn't, the error names the parameters still varying, which is the cue to fix them in `where`.
Naming an argument, in `where` or as a grid dimension, also leaves out the runs saved
without it (from before it existed), as load_experiments does: prm shows such runs as None.
"""
import json
import warnings
from functools import partial

import numpy as np
import pandas as pd
import scipy.special as spc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import itertools

import networkx as nx

from icl_utils import xent, kl_rows, plot_icl, split_block, ylabel, spread, draw_traces, block_code


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


ABSENT = '<absent>'     # in runs() and table(): an argument the run was saved without


def _absent(v):
    return isinstance(v, str) and v == ABSENT


def _n_runs(prm):
    return len(prm) if isinstance(prm, pd.DataFrame) else len(next(iter(prm.values())))


def _column(prm, name):
    """A parameter's value for each run, from prm (load_experiments) or runs(records)."""
    if name not in prm:
        raise KeyError(f"no parameter {name!r}; there are {sorted(prm)}")
    if isinstance(prm, pd.DataFrame):
        return list(prm[name])                     # already as _hashable makes them
    return [_hashable(v) for v in prm[name]]       # a 2-d array (e.g. num_trials) by rows


def _has(record, name):
    """Whether a run was saved with argument `name` (task or model)."""
    return name in record['task_args'] or name in record['model_args']


def select(prm, where=None, records=None):
    """
    Indices of the runs (into records) matching `where` (see the module docstring), from
    prm (load_experiments) or the table runs(records).

    An argument a run was saved without (from before it existed) matches nothing `where`
    gives for it. runs(records) shows it as ABSENT, so that's automatic; prm shows it as
    None, like an explicit None, so with prm pass records too, or such runs match
    where={name: None} (and lists containing None).
    """
    n = _n_runs(prm)
    keep = np.ones(n, bool)
    for name, cond in (where or {}).items():
        col = _column(prm, name)
        if callable(cond):
            keep &= np.array([not _absent(v) and bool(cond(v)) for v in col])
        elif isinstance(cond, list):
            ok = {_hashable(c) for c in cond}
            keep &= np.array([v in ok for v in col])
        else:
            if isinstance(cond, (set, frozenset)) and not any(isinstance(v, frozenset) for v in col):
                raise ValueError(f"where={{{name!r}: {cond!r}}}: a set is a value, and no {name} is a "
                                 f"set; for runs with any of these values, use a list: {sorted(cond, key=_key)}")
            keep &= np.array([v == _hashable(cond) for v in col])
    if records is not None and where:
        keep &= np.array([all(_has(rec, name) for name in where) for rec in records])
    return np.where(keep)[0]


def varying(prm, runs=None, show=True):
    """{parameter: values} of the parameters that differ between `runs` (default all)."""
    runs = np.arange(_n_runs(prm)) if runs is None else runs
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

    Given records, an argument named in `where` or `dims` leaves out the runs saved
    without it (from before it existed), as load_experiments does for the arguments it's
    given; a note says how many.
    """
    runs = select(prm, where)
    if records is not None:
        names = list(where or {}) + [d for d in dims if d is not None]
        kept = [r for r in runs if all(_has(records[r], n) for n in names)]
        if len(kept) < len(runs):
            gone = sorted({n for r in runs for n in names if not _has(records[r], n)})
            print(f"left out {len(runs) - len(kept)} of {len(runs)} runs saved without "
                  f"{', '.join(gone)} (from before {'it' if len(gone) == 1 else 'they'} existed)")
        runs = np.array(kept, int)
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
            absent = sorted({n for (_, n), vals in saved.items()
                             if any(isinstance(v, str) and v == '<absent>' for v, _ in vals)})
            if absent:
                why += (f"; to keep only the runs saved with {', '.join(absent)}, name "
                        f"{'it' if len(absent) == 1 else 'them'} in `where` (e.g. "
                        f"{absent[0]}=None)")
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
              title=None, keep=None):
    """
    plot_icl for each run matching `where`, on a grid of `rows` x `cols` (parameter
    names; leave either out for a single row or column). Empty cells are hidden.
    perplexity, kl, err, traces, renorm: as for plot_icl.
    title: f(record) for each panel's title, instead of the cell's parameter values.
    keep: which blocks of which samples to show (see block_mask), e.g. 'fixed'; the
    title then says how many blocks were kept.
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
            M = block_mask(rec, keep)
            name = title(rec) if title else _title(rows, cols, rv, cv)
            if M is not None:
                name += f' [{int(M.sum())}/{M.size} blocks]'
            plot_icl(ax, _masked_logp(m['logp_ctx'], M), m['data'], n_trl=n_trl,
                     k_hide=a.get('k_hide'), title=name,
                     show_assistant=(a.get('mode') != 'raw'), refs=refs,
                     perplexity=perplexity, kl=kl, err=err, traces=traces,
                     renorm=renorm)
    fig.tight_layout()
    return axs


# -- the held-out test, block by block ---------------------------------------------------

def heldout(record, refs=(), perplexity=False, kl=False, renorm=True, keep=None):
    """
    Score of each block's held-out token (the first hidden token of its final trial, the
    star in plot_icl), for each sample: {'model': (samps, n_blocks), <ref>: (same)}.
    refs: {name: predictor}, as for plot_icl, scored at the same tokens.
    perplexity, kl, renorm: as for plot_icl (kl: -KL(Y || model) instead of
    -cross-entropy; renorm=False: the model's probabilities over the whole vocabulary).
    keep: which blocks of which samples to score (see block_mask); the rest are nan,
    which the summaries skip.
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
    out = {k: f(np.array(v, float)) for k, v in out.items()}
    M = block_mask(record, keep)
    if M is not None:
        for v in out.values():
            v[~M] = np.nan
    return out


REF_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X']


def _as_curve(v):
    """(x, Y), Y (samples, len(x)), from what a plot_grid function returns: a Series by
    (sample, x) (as curve() makes), a DataFrame with a row per sample and a column per x,
    a Series by x alone (one row, so no spread), or a tuple (x, Y)."""
    if isinstance(v, tuple) and len(v) == 2:
        x = np.asarray(v[0])
        return x, np.asarray(v[1], float).reshape(-1, len(x))
    if isinstance(v, pd.DataFrame):
        return v.columns.to_numpy(), v.to_numpy(float)
    if isinstance(v, pd.Series):
        if v.index.nlevels == 2:
            W = v.groupby(level=[0, 1]).mean().unstack(level=1)
            return W.columns.to_numpy(), W.to_numpy(float)
        return v.index.to_numpy(), v.to_numpy(float)[None]
    raise TypeError("a plot function returns a Series by (sample, x), a DataFrame (a row "
                    f"per sample, a column per x), or (x, Y); not a {type(v).__name__}")


def _draw_grid(records, prm, curves_of, rows, cols, hue, where, err, traces, cmap, size,
               sharey, xlabel, ylabel, note=None):
    """
    The grid behind plot_heldout and plot_grid: one panel per (rows, cols) value, one
    colour per hue value, each cell a single run (see _grid). curves_of(record) gives
    {name: (x, Y)}: the first is the run's line (centre and error bars over samples by
    `err`, each sample's line too with `traces`), the others (e.g. references) dashed
    centres with hollow markers in the same colour. x that isn't numeric is plotted as
    categories, in order. note(record): extra text for a panel's title when it has one
    line. Returns the axes.
    """
    cells, levels = _grid(prm, where, (rows, cols, hue), records)
    hues = levels[2]
    colors = plt.get_cmap(cmap)(np.linspace(0, 0.9, len(hues))) if hue else ['k']
    curves = {key: curves_of(records[r]) for key, r in cells.items()}
    extras = list(dict.fromkeys(n for c in curves.values() for n in list(c)[1:]))
    xs = [x for c in curves.values() for x, _ in c.values()]
    numeric = all(np.issubdtype(np.asarray(x).dtype, np.number)
                  and not np.issubdtype(np.asarray(x).dtype, np.bool_) for x in xs)
    if numeric:
        where_x = lambda x: np.asarray(x, float)
        ticks = np.unique(np.concatenate([np.asarray(x, float) for x in xs])) if xs else []
        integer = len(ticks) and len(ticks) <= 20 and np.all(ticks == np.round(ticks))
    else:
        cats = _order(v for x in xs for v in x.tolist())
        pos = {v: p for p, v in enumerate(cats)}
        where_x = lambda x: np.array([pos[v] for v in x.tolist()], float)

    fig, axs = _figure(levels, size, sharey)
    style = dict(ls='--', lw=0.8, ms=5, mfc='none')
    for i, rv in enumerate(levels[0]):
        for j, cv in enumerate(levels[1]):
            ax = axs[i, j]
            here = [(h, (rv, cv, hv)) for h, hv in enumerate(hues) if (rv, cv, hv) in curves]
            if not here:
                ax.set_visible(False)
                continue
            for h, key in here:
                (x, Y), *rest = curves[key].values()
                if traces:
                    draw_traces(ax, where_x(x), Y, colors[h], traces)
                mu, e = spread(Y, err)
                ax.errorbar(where_x(x), mu, e, color=colors[h], marker='o', ms=6, lw=2, capsize=2)
                for name, (x2, Y2) in list(curves[key].items())[1:]:
                    k = extras.index(name)
                    ax.plot(where_x(x2), spread(Y2, err)[0], color=colors[h], alpha=0.8,
                            marker=REF_MARKERS[k % len(REF_MARKERS)], **style)
            title = _title(rows, cols, rv, cv)
            if note is not None and len(here) == 1:
                extra = note(records[cells[here[0][1]]])
                title += f' {extra}' if extra else ''
            ax.set_title(title)
            ax.set_xlabel(xlabel or '')
            if j == 0 and ylabel:
                ax.set_ylabel(ylabel)
    for ax in axs.flat:
        if not numeric:
            ax.set_xticks(range(len(cats)), [_show(v) for v in cats])
        elif integer:
            ax.set_xticks(ticks)
    # one legend for the whole grid, in the first panel: every hue, then every extra line
    handles = [Line2D([], [], color=colors[h], marker='o', lw=2, label=_label(hue, hv))
               for h, hv in enumerate(hues)] if hue else []
    handles += [Line2D([], [], color='0.4', marker=REF_MARKERS[k % len(REF_MARKERS)],
                       label=name, **style) for k, name in enumerate(extras)]
    if handles:
        next(ax for ax in axs.flat if ax.get_visible()).legend(handles=handles, fontsize='small')
    fig.tight_layout()
    return axs


def plot_heldout(records, prm, hue=None, rows=None, cols=None, where=None, refs=(),
                 perplexity=False, kl=False, err='std', traces=False, renorm=True, cmap='viridis',
                 size=(4, 3.5), sharey=True, keep=None):
    """
    Held-out log-likelihood (or perplexity, or -KL; see plot_icl) of each block's test
    token against the block, summarised over samples as `err` says (see plot_icl: mean
    and std by default), one line per value of `hue`, on a grid of `rows` x `cols`.
    traces: also draw each sample's line, thin and translucent (True, or the opacity).
    renorm: as for plot_icl (False: the model's probabilities over the whole vocabulary).
    refs are drawn dashed with hollow markers in the colour of their line.
    keep: which blocks of which samples to score (see block_mask), e.g. 'fixed'; with
    one line per panel, the title says how many blocks were kept.
    For anything else against block, or any other x, see plot_grid.
    Returns the axes.
    """
    def curves_of(rec):
        s = heldout(rec, refs, perplexity, kl, renorm, keep)
        x = np.arange(1, s['model'].shape[1] + 1)
        return {k: (x, v) for k, v in s.items()}

    def note(rec):
        if keep is None:
            return None
        M = block_mask(rec, keep)
        return f'[{int(M.sum())}/{M.size} blocks]'

    return _draw_grid(records, prm, curves_of, rows, cols, hue, where, err, traces, cmap,
                      size, sharey, 'block', 'held-out ' + ylabel(perplexity, kl, renorm), note)


# -- any function of a run's trace, on the grid ------------------------------------------

def curve(df, y, x='block'):
    """
    Values of df's rows (trace rows, usually a selection) as a Series by (sample, x),
    averaged over rows sharing one: what a plot_grid function returns. y is a column
    name, a function of df, or an array; x a column ('block' is numbered from 1, as
    plot_heldout does).
    """
    vals = df[y] if isinstance(y, str) else (y(df) if callable(y) else y)
    xs = df[x].to_numpy() + 1 if x == 'block' else df[x].to_numpy()
    idx = pd.MultiIndex.from_arrays([df['sample'].to_numpy(), xs], names=['sample', x])
    return pd.Series(np.asarray(vals, float), index=idx).groupby(level=[0, 1]).mean()


def at_target(df, col='ll'):
    """Each row's vector column `col` (ll, ll_assistant, ref_<name>, ...) at the token Y
    puts most weight on (the row's target)."""
    return np.array([np.asarray(v)[j] for v, j in zip(df[col], df['target'])], float)


def score(df, col='ll', renorm=True):
    """Each row's -cross-entropy against Y from vector column `col`: sum(y * col), as
    plot_icl and plot_heldout score (ll[target] when one token is right). renorm: less
    log(task mass) (task_mass_assistant for the assistant span's columns), i.e. over the
    task alphabet; refs (ref_<name>) are distributions over it already."""
    out = np.array([np.nansum(np.where(np.asarray(y) > 0, np.asarray(y) * np.asarray(v), 0))
                    if not np.isnan(np.asarray(v)[np.asarray(y) > 0]).any() else np.nan
                    for y, v in zip(df['y'], df[col])], float)
    if renorm and not col.startswith('ref_'):
        mass = df['task_mass_assistant' if col.endswith('_assistant') else 'task_mass']
        out = out - np.log(mass.to_numpy(float))
    return out


def plot_grid(records, prm, f, rows=None, cols=None, hue=None, where=None, refs=(),
              keep=None, err='std', traces=False, xlabel=None, ylabel=None, cmap='viridis',
              size=(4, 3.5), sharey=True):
    """
    Any quantity of each run, on plot_heldout's grid: f(tr) takes a run's trace() and
    returns what to plot against what, for each sample. The summary over samples is by
    `err` (see plot_icl), each sample's line too with `traces`; one line per value of
    `hue`, panels by `rows` x `cols`, each a single run (see the module docstring).

    f returns a Series by (sample, x) -- curve(rows, y, x) makes one --, a DataFrame
    with a row per sample and a column per x, or (x, Y); or a dict of such, {name: ...},
    whose first is the run's line and the others extra dashed lines (e.g. a reference),
    named in the legend. x that isn't numeric is plotted as categories.

    tr.attrs['args'] holds the run's arguments (task and model), e.g. to normalise by k.
    refs: reference predictors, as trace columns ref_<name>. keep: only the trace rows
    of the blocks block_mask keeps (e.g. 'fixed').

        test = lambda tr: tr[tr.final & tr.hidden]           # each block's test token
        # plot_heldout, from the user span (equal in 'raw'), renormalised
        plot_grid(records, prm, lambda tr: curve(test(tr), score), ...)
        # how often the model's top token over the vocabulary is the right word
        plot_grid(records, prm, lambda tr: curve(test(tr), lambda t: t.top_any == t.word),
                  ylabel='top token right', ...)
        # task mass along the block, every hidden token, by trial
        plot_grid(records, prm, lambda tr: curve(tr[tr.hidden], 'task_mass', x='trial'), ...)
        # the model and a reference
        plot_grid(records, prm, lambda tr: {'model': curve(test(tr), score),
                  'bigram': curve(test(tr), lambda t: score(t, 'ref_bigram'))},
                  refs={'bigram': partial(ngram, order=1, alpha=0.5)}, ...)
    Returns the axes.
    """
    def curves_of(rec):
        tr = trace(rec, refs=refs, keep=keep)
        if keep is not None:
            tr = tr[tr['kept']]
        tr.attrs['args'] = {**rec['task_args'], **rec['model_args']}
        out = f(tr)
        if not isinstance(out, dict):
            out = {ylabel or 'value': out}
        return {k: _as_curve(v) for k, v in out.items()}

    return _draw_grid(records, prm, curves_of, rows, cols, hue, where, err, traces, cmap,
                      size, sharey, xlabel, ylabel)


# -- one run, token by token -----------------------------------------------------------

def _grammar_names(data, i):
    """A name for each grammar token of sample i, the same in every block: 'delim' for the
    delimiter, kind_label for automata that saved their vocabulary (e.g. 'node_2',
    'action_0'), else 'tok_<r>'."""
    nt = block_code(data, i).shape[1]
    kinds, desc = data.get('kinds'), (data.get('automaton') or [None] * (i + 1))[i]
    if kinds is None or desc is None or 'vocab' not in desc:
        names = ['tok_%d' % r for r in range(nt)]
    else:
        names = [f'{kinds[r]}_{desc["vocab"][r]}' for r in range(nt)]
    names[0] = 'delim'
    if len(set(names)) < len(names):                  # labels that clash: fall back on r
        names = ['delim'] + ['tok_%d' % r for r in range(1, nt)]
    return names


def trace(record, refs=(), keep=None):
    """
    The run's scores as a DataFrame, one row per token of every sample, close to how
    they're stored (logp_ctx) but easy to slice: each score is a vector over the block's
    alphabet, in grammar order, so index j is the same token role (tr.attrs['alphabet'][j],
    e.g. 'node_2') in every block and sample, though the words change.

        sample, block, trial   where the token is (trial within its block)
        pos                    position within its trial (the delimiter is the last)
        t                      index into data['X'][sample], data['Y'][sample]
        word                   the token, as the model saw it (that sample's word list)
        name                   its grammar name: 'delim', and for automata kind_label,
                               e.g. 'node_2', 'action_0' (else 'tok_<r>')
        hidden                 whether it's among the tokens plot_icl shows (the last
                               k_hide of the trial, or those after the first k_show)
        final                  whether it's in the block's final trial, whose hidden part
                               is the completion (the assistant span)
        target                 index into the vectors of the token Y puts most weight on
        y                      Y over the alphabet (for tasks where more than one
                               token is right; n_valid says how many)
        n_valid                number of tokens Y allows
        ll                     log-probabilities, over the whole vocabulary, from the
                               user span: the context rows, and for the final trial's
                               first hidden token the user span's last row (what follows
                               the context, before any template). nan where there is no
                               user-span row (later completion tokens). Comparable
                               across modes: in 'raw' and 'single' it's the stream itself.
        task_mass              probability on the whole task alphabet (every block's)
                               at that row; ll - log(task_mass) renormalises to it, as
                               plot_icl's default does
        top_any, p_top_any     the most likely token over the whole vocabulary, and
        top_text, p_top_text   over text only, and their probabilities (runs saved
                               before these were recorded leave them out)
        ll_assistant, task_mass_assistant, top_any_assistant, ...
                               the same from the assistant span: the completion's rows,
                               so only the final trial's hidden tokens. In 'dialogue' this
                               is the assistant turn's prediction, after the template's
                               header; in 'raw' and 'single' its first row is the user
                               span's last, so ll and ll_assistant agree on that token.
        ll_iso, task_mass_iso, top_any_iso, p_top_any_iso, top_text_iso, p_top_text_iso,
        ll_iso_assistant, ...  the same from the comparison run, which scores each block on
                               its own, without the blocks before it or any reasoning
                               (ICLModel's logp_iso, top_ids_iso, top_lp_iso)
        kl_iso, kl_iso_task    KL(with ‖ without earlier blocks) at that row, from ll's
                               prediction to ll_iso's, over the whole vocabulary (stored) and
                               over the task alphabet (from the two), as in plot_icl(...,
                               iso=...). From the user span, like ll
        kl_iso_assistant, kl_iso_task_assistant
                               the same from the assistant span, like ll_assistant; when
                               thinking, this also measures the reasoning trace
                               (runs saved before these were recorded leave out what they
                               lack; runs that stored kl_iso_task have no ll_iso)
        ref_<name>             each reference predictor's log-probabilities, as a vector
                               the same way
        kept                   (given keep) whether block_mask keeps the token's block
                               of this sample, e.g. tr[tr.kept] with keep='fixed'

    For example, the test token of each block, its likelihood from each span, and every
    node's probability from the assistant turn:
        test = tr[tr.final & tr.hidden]
        at = lambda col: np.array([v[j] for v, j in zip(test[col], test.target)])
        user_ll, asst_ll = at('ll'), at('ll_assistant')
        L = np.stack(test.ll_assistant)                    # (tests, alphabet)
        nodes = [j for j, nm in enumerate(tr.attrs['alphabet']) if nm.startswith('node')]
        P = np.exp(L[:, nodes])
    Where more than one token is right (n_valid > 1, e.g. a trial's first node), the
    score plot_icl uses is -cross-entropy, sum(y * ll), rather than ll[target]:
        xe = [np.nansum(np.where(y > 0, y * v, 0)) for y, v in zip(tr.y, tr.ll)]
    plot_icl's curve is its mean over samples, less log(task_mass), on the hidden rows,
    by block and trial (it uses the user span for the completion's first token too).
    """
    m, a = record['metrics'], record['model_args']
    data, refs = m['data'], dict(refs)
    k_show, k_hide = a.get('k_show'), a.get('k_hide')
    # the runs whose scores are stored, as (column suffix, log-probs, top ids, top log-probs)
    runs = [('', 'logp_ctx', 'top_ids', 'top_lp'), ('_iso', 'logp_iso', 'top_ids_iso', 'top_lp_iso')]
    runs = [r for r in runs if r[1] in m]
    has_top = {sfx: tid in m for sfx, _, tid, _ in runs}
    M = block_mask(record, keep)
    word_of = lambda j: m['top_vocab'].get(int(j)) if j >= 0 else None

    rows, alphabets = [], []
    for i, lpc in enumerate(m['logp_ctx']):
        X, Y, B, blocks = data['X'][i], data['Y'][i], data['B'][i], data['blocks'][i]
        words = [w.replace('\u0120', ' ') for w in m['task_tokens'][i]] if 'task_tokens' in m else None
        P = {k: pred(data, i) for k, pred in refs.items()}
        code, names = block_code(data, i), _grammar_names(data, i)
        alphabets.append(tuple(names))
        for b, trials in enumerate(blocks):
            t0 = int(np.argmax(B == b))
            n_ctx, comp = len(lpc[('user', b)]) - 1, split_block(trials, k_show, k_hide)[1]
            n = sum(len(tr) for tr in trials)

            def place(src, side, fill):
                """A span's stored rows (src[(side, b)]) on the block's tokens: the user
                span's row r predicts token r; the assistant span's rows the completion."""
                a = src[(side, b)]
                out = np.full((n,) + a.shape[1:], fill, np.result_type(a, type(fill)))
                lo, k = (0, n_ctx + 1) if side == 'user' else (n_ctx, len(comp))
                k = min(k, len(a))
                out[lo:lo + k] = a[:k]
                return out

            trial = np.repeat(np.arange(len(trials)), [len(tr) for tr in trials])
            pos = np.concatenate([np.arange(len(tr)) for tr in trials])
            real = np.repeat([len(tr) - 1 for tr in trials], [len(tr) for tr in trials])
            first = np.array([split_block([tr], k_show, k_hide)[0].size for tr in trials])
            ts = t0 + np.arange(n)
            Yb = Y[ts]
            sym = code[b]                                   # grammar token -> symbol, this block
            gram = {int(x): r for r, x in enumerate(sym)}
            row = {'sample': i, 'block': b, 'trial': trial, 'pos': pos, 't': ts,
                   'word': [words[x] for x in X[ts]] if words else X[ts],
                   'name': [names[gram[int(x)]] if int(x) in gram else None for x in X[ts]],
                   'hidden': (pos >= first[trial]) & (pos < real),
                   'final': trial == len(trials) - 1,
                   'target': [gram.get(int(y), -1) for y in Yb.argmax(-1)],
                   'y': list(Yb[:, sym]),
                   'n_valid': (Yb > 0).sum(-1)}
            for side, suffix in [('user', ''), ('assistant', '_assistant')]:
                span = {}
                for run, lps, tids, tlps in runs:
                    lp = span[run] = place(m[lps][i], side, np.nan)
                    with np.errstate(invalid='ignore', divide='ignore'):
                        row[f'll{run}{suffix}'] = list(lp[:, sym])
                        mass = np.exp(spc.logsumexp(lp, -1))
                    row[f'task_mass{run}{suffix}'] = np.where(np.isnan(lp).all(-1), np.nan, mass)
                    if has_top[run]:
                        tid, tlp = place(m[tids][i], side, -1), place(m[tlps][i], side, np.nan)
                        for c, nm in enumerate(['any', 'text']):
                            row[f'top_{nm}{run}{suffix}'] = [word_of(j) for j in tid[:, c]]
                            row[f'p_top_{nm}{run}{suffix}'] = np.exp(tlp[:, c])
                if 'kl_iso' in m:                           # one value per row, placed like ll
                    row['kl_iso' + suffix] = place(m['kl_iso'][i], side, np.nan)
                if 'kl_iso_task' in m:                      # stored by runs before logp_iso
                    row['kl_iso_task' + suffix] = place(m['kl_iso_task'][i], side, np.nan)
                elif '_iso' in span:
                    with np.errstate(invalid='ignore'):
                        row['kl_iso_task' + suffix] = kl_rows(span[''], span['_iso'])
            with np.errstate(divide='ignore'):
                for k, Pk in P.items():
                    Q = Pk[ts]
                    row['ref_' + k] = list(np.log(Q[:, sym] / Q.sum(-1, keepdims=True)))
            if M is not None:
                row['kept'] = bool(M[i, b])
            rows.append(pd.DataFrame(row))
    tr = pd.concat(rows, ignore_index=True)
    tr.attrs['alphabet'] = list(alphabets[0]) if len(set(alphabets)) == 1 else {
        i: list(al) for i, al in enumerate(alphabets)}      # per sample, if they differ
    return tr


# -- tables across runs -----------------------------------------------------------------

def runs(records, varying=False):
    """
    One row per run (indexed by its position in records), its saved arguments as columns:
    for checking which combinations exist, or choosing runs, with pandas.

        r = runs(records)
        r.value_counts(['k', 'd'], dropna=False)       # how many runs of each combination
        r.groupby(['k', 'd'], dropna=False).size().unstack()     # ... as a grid
        r[(r.k == 5) & (r.hint == 0)].index            # run indices, e.g. for table()

    None is kept as None, so group with dropna=False (pandas leaves None out otherwise).

    Values are as select compares them (lists as tuples, sets as frozensets), so they
    group and compare. An argument a run was saved without (from before it existed) is
    ABSENT ('<absent>'), unlike prm, where it's None like an explicit None. A name in
    both the task's and the model's arguments (seed) becomes task_seed and model_seed.
    varying: only the arguments that differ between the runs.
    """
    tasks = [r['task_args'] for r in records]
    mods = [r['model_args'] for r in records]
    both = set().union(*map(set, tasks)) & set().union(*map(set, mods))
    names = list(dict.fromkeys(k for a in tasks + mods for k in a))
    val = lambda args, k: _hashable(args[k]) if k in args else ABSENT
    rows = []
    for t, m in zip(tasks, mods):
        row = {}
        for k in names:
            if k in both:
                row['task_' + k], row['model_' + k] = val(t, k), val(m, k)
            else:
                row[k] = val(t if k in t else m, k)
        rows.append(row)
    cols = {}
    for c in dict.fromkeys(c for row in rows for c in row):
        vals = [row[c] for row in rows]                  # None stays None, not NaN
        cols[c] = pd.Series(vals, dtype=object if any(v is None for v in vals) else None)
    out = pd.DataFrame(cols).set_axis(pd.RangeIndex(len(records), name='run'))
    if varying:
        out = out[[c for c in out if out[c].map(_key).nunique() > 1]]
    return out


def table(records, which=None, rows='hidden', params=None, refs=(), keep=None):
    """
    trace() of many runs as one DataFrame, with a `run` column (the index into records)
    and the runs' arguments as columns, for slicing and plotting by hand:

        tb = table(records, rows='test')
        tb.groupby(['k', 'block']).task_mass.mean().unstack()
        tb[tb.k == 5].groupby('trial').task_mass.mean().plot()

    which:  run indices to include (e.g. from select, or runs(records)[...].index);
            default all
    rows:   'hidden' the tokens plot_icl shows (default); 'test' each block's held-out
            token (final and hidden); 'all' every token
    params: argument columns to add (from runs()): default those that differ between
            the runs included; 'all', or a list of names
    refs, keep: as for trace.
    The alphabet of each run (trace's attrs['alphabet']) is in attrs['alphabet'][run].
    """
    which = list(range(len(records))) if which is None else [int(r) for r in which]
    if rows not in ('hidden', 'test', 'all'):
        raise ValueError(f"rows is 'hidden', 'test' or 'all', not {rows!r}")
    info = runs([records[r] for r in which]).set_axis(pd.Index(which, name='run'))
    if params is None:
        params = [c for c in info if info[c].map(_key).nunique() > 1]
    elif params == 'all':
        params = list(info)
    parts, alphabet = [], {}
    for r in which:
        tr = trace(records[r], refs=refs, keep=keep)
        alphabet[r] = tr.attrs.get('alphabet')
        if rows == 'hidden':
            tr = tr[tr.hidden]
        elif rows == 'test':
            tr = tr[tr.hidden & tr.final]
        tr.insert(0, 'run', r)
        parts.append(tr)
    out = pd.concat(parts, ignore_index=True)
    out = out.join(info[params], on='run')
    out.attrs['alphabet'] = alphabet
    return out


# -- which tests are well defined -----------------------------------------------------

_FIXED = {}                                    # id(metrics) -> (metrics, result): a cache


def _task_name(record):
    t = record.get('task')
    return t if isinstance(t, str) else getattr(t, '__name__', str(t))


def _sample_graph(data, i, actions_shown=True):
    """Sample i's graph, as a MultiDiGraph with an `action` on each edge, and the label
    of each grammar token. From data['automaton'] (an FSA description) when the run
    saved it; otherwise rebuilt from every transition its blocks show, with grammar
    tokens as labels (an edge no block shows is missed), and with action None on every
    edge if the trials don't show actions."""
    if 'automaton' in data:
        desc = data['automaton'][i]
        if desc['type'] != 'FSA':
            raise TypeError(f"fixed answers need an FSA, not a {desc['type']}")
        G = nx.MultiDiGraph()
        G.add_nodes_from(n for n, _ in desc['nodes'])
        for u, v, _, d in desc['edges']:
            G.add_edge(u, v, action=d.get('action', v))     # FSA's default action: the target
            if not desc['directed']:
                G.add_edge(v, u, action=d.get('action', u))
        return G, desc['vocab']
    code = block_code(data, i)
    G = nx.MultiDiGraph()
    kinds = data.get('kinds')
    for b, trials in enumerate(data['blocks'][i]):
        gram = {int(x): r for r, x in enumerate(code[b])}          # symbol -> grammar token
        for tr in trials:
            toks = [gram[int(x)] for x in tr[:-1]]
            if kinds is not None:
                is_act = [kinds[t] == 'action' for t in toks]
            else:                                   # nodes and actions alternate, if shown
                is_act = [actions_shown and p % 2 == 1 for p in range(len(toks))]
            u = a = None
            for t, act in zip(toks, is_act):
                if act:
                    a = t
                    continue
                if u is not None:
                    G.add_edge(u, t, action=a)
                u, a = t, None
    return G, list(range(code.shape[1]))


def _test_transition(toks, kinds, label, actions_shown=True):
    """The last transition (u, a, v) of a trial's grammar tokens `toks`, as labels: v its
    last node, u the node before, a the action between them, or None if the trials don't
    show actions. By kind when the run saved data['kinds'], otherwise by position."""
    if kinds is not None:
        nodes = [p for p, t in enumerate(toks) if kinds[t] == 'node']
        if len(nodes) < 2:
            raise ValueError("the test trial shows fewer than two nodes, so it has no "
                             "transition to check")
        p, q = nodes[-2], nodes[-1]
        acts = [toks[x] for x in range(p + 1, q) if kinds[toks[x]] == 'action']
        return label[toks[p]], (label[acts[-1]] if acts else None), label[toks[q]]
    if actions_shown:
        u, a, v = toks[-3:]
        return label[u], label[a], label[v]
    return label[toks[-2]], None, label[toks[-1]]


def _action_blind(G):
    """G with every action the same (None) and parallel edges merged: the graph as trials
    that show only nodes present it."""
    H = nx.MultiDiGraph()
    H.add_nodes_from(G)
    H.add_edges_from((x, y, {'action': None}) for x, y in dict.fromkeys(G.edges()))
    return H


def _training_graph(G, u, a, v, unit, symmetric):
    """G without the block's held-out unit (and, if symmetric, the units reversing it),
    as AutomatonTask holds them out: unit 'edge' every edge u -> v whatever its action;
    'pair' every edge u -a-> x; 'transition' just u -a-> v (ExhaustiveRelationTask).
    The reverse of an edge is every edge v -> u; of a pair, the pairs (x, b) for each x
    that a takes u to and each b with an edge x -> u."""
    gone = set()                                     # (x, y, action) to drop
    edges = [(x, y, d['action']) for x, y, d in G.edges(data=True)]
    if unit == 'edge':
        gone |= {e for e in edges if e[:2] == (u, v)}
        if symmetric:
            gone |= {e for e in edges if e[:2] == (v, u)}
    elif unit == 'pair':
        gone |= {e for e in edges if (e[0], e[2]) == (u, a)}
        if symmetric:
            dests = {y for x, y, c in edges if (x, c) == (u, a)}
            back = {(x, c) for x, y, c in edges if x in dests and y == u}
            gone |= {e for e in edges if (e[0], e[2]) in back}
    elif unit == 'transition':
        gone.add((u, v, a))
        if symmetric:
            gone |= {e for e in edges if e[:2] == (v, u)}
    else:
        raise ValueError(f"unit is 'edge', 'pair' or 'transition', not {unit!r}")
    H = nx.MultiDiGraph()
    H.add_nodes_from(G)
    H.add_edges_from((x, y, {'action': c}) for x, y, c in edges if (x, y, c) not in gone)
    return H


def _refine(G, colours):
    """Colour refinement (1-dimensional Weisfeiler-Leman) of a MultiDiGraph with an
    `action` on each edge, from initial `colours`: nodes are split by the multiset of
    (action, direction, colour) of their edges until no class splits. The colours are
    canonical, so refining two graphs together compares them: an isomorphism only maps a
    node to one of the same colour."""
    col = dict(colours)
    n = len(set(col.values()))
    while True:
        sig = {x: (col[x],
                   tuple(sorted((repr(d['action']), 1, col[y]) for _, y, d in G.out_edges(x, data=True))),
                   tuple(sorted((repr(d['action']), 0, col[y]) for y, _, d in G.in_edges(x, data=True))))
               for x in G}
        names = {s: c for c, s in enumerate(sorted(set(sig.values()), key=repr))}
        col = {x: names[sig[x]] for x in G}
        if len(names) == n:
            return col
        n = len(names)


def fixed_target(G, u, a, v):
    """
    Whether every symmetry of G that fixes node u and action a also fixes node v. A
    symmetry relabels the nodes and permutes the actions so that G (a MultiDiGraph with
    an `action` on each edge) maps onto itself. If one moves v, then knowing v answers
    (u, a) under one labelling of G doesn't say which node answers it under another.

    No polynomial-time algorithm is known for graph symmetry in general, but this is
    fast in practice (milliseconds at tens of nodes): for each permutation of the other
    actions ((d-1)! of them), colour refinement of G against the relabelled G, u marked
    in both, leaves the nodes v could map to -- for a random graph almost always just v
    -- and a VF2 search (networkx) settles each other candidate.
    """
    from networkx.algorithms.isomorphism import MultiDiGraphMatcher
    acts = sorted({d['action'] for _, _, d in G.edges(data=True)} | {a}, key=repr)
    others = [b for b in acts if b != a]
    nodes = list(G)
    n, idx = len(nodes), {x: t for t, x in enumerate(nodes)}
    for perm in itertools.permutations(others):
        q = dict(zip(others, perm))
        q[a] = a
        H = nx.MultiDiGraph()
        H.add_nodes_from(nodes)
        H.add_edges_from((x, y, {'action': q[d['action']]}) for x, y, d in G.edges(data=True))
        U = nx.disjoint_union(G, H)                     # G's nodes 0..n-1, H's n..2n-1
        start = {t: int(t % n == idx[u]) for t in U}
        col = _refine(U, start)
        if sorted(col[t] for t in range(n)) != sorted(col[t] for t in range(n, 2 * n)):
            continue                                   # no isomorphism fixing u
        for w in nodes:
            if w == v or col[n + idx[w]] != col[idx[v]]:
                continue
            mark = dict(start)
            mark[idx[v]] = mark[n + idx[w]] = 2         # v in G, w in H
            c2 = _refine(U, mark)
            if sorted(c2[t] for t in range(n)) != sorted(c2[t] for t in range(n, 2 * n)):
                continue
            G2, H2 = G.copy(), H.copy()
            nx.set_node_attributes(G2, {x: c2[idx[x]] for x in nodes}, '_c')
            nx.set_node_attributes(H2, {x: c2[n + idx[x]] for x in nodes}, '_c')
            acts_of = lambda e: sorted((repr(d['action']) for d in e.values()))
            gm = MultiDiGraphMatcher(G2, H2, node_match=lambda p, r: p['_c'] == r['_c'],
                                     edge_match=lambda p, r: acts_of(p) == acts_of(r))
            if gm.is_isomorphic():
                return False
    return True


def fixed_answers(record):
    """
    (samps, n_blocks) bool: whether each block's test answer is well defined, i.e. no
    symmetry of the block's training graph moves it (see fixed_target). The training
    graph is the sample's graph without the block's held-out unit (and its reverse,
    with symmetric=True), the unit being the run's: every edge u -> v for
    holdout_unit='edge', every u -a-> x for 'pair', u -a-> v alone for
    ExhaustiveRelationTask. The test is the final trial's last transition u -a-> v, and
    a symmetry must fix u and a. If one moves v, the answer seen under one block's
    words doesn't say which node it is under another's. Runs saved without
    data['automaton'] rebuild each sample's graph from the transitions its blocks show,
    with a warning. Cached per record. Needs nothing from automata.

    If the trials don't show actions (show without 'action'), the test is u -> v, and the
    symmetries are those of the graph with the actions erased: a node answers the test
    only if no symmetry of that graph fixing u moves it.
    """
    m = record['metrics']
    hit = _FIXED.get(id(m))
    if hit is not None and hit[0] is m:
        return hit[1].copy()
    data, targs, task = m['data'], record['task_args'], _task_name(record)
    exhaustive = task == 'ExhaustiveRelationTask'
    if not (exhaustive or targs.get('holdout')):
        raise ValueError(f"fixed answers need a hold-out run ({task} with holdout=False)")
    if 'automaton' not in data:
        warnings.warn(f"{record.get('folder', 'run')}: no saved automaton; rebuilding each "
                      "sample's graph from the transitions its blocks show", stacklevel=2)
    unit = 'transition' if exhaustive else targs.get('holdout_unit', 'edge')
    symmetric = bool(targs.get('symmetric', False))
    show = targs.get('show')
    actions_shown = show is None or 'action' in show
    kinds = data.get('kinds')
    out = np.zeros((len(m['logp_ctx']), len(data['blocks'][0])), bool)
    for i in range(len(out)):
        G, label = _sample_graph(data, i, actions_shown)
        code = block_code(data, i)
        done = {}                                    # blocks testing the same transition
        for b, trials in enumerate(data['blocks'][i]):
            gram = {int(x): r for r, x in enumerate(code[b])}       # symbol -> grammar token
            toks = [gram[int(x)] for x in trials[-1][:-1]]
            u, a, v = _test_transition(toks, kinds, label, actions_shown)
            if (u, a, v) not in done:
                if a is not None:
                    H = _training_graph(G, u, a, v, unit, symmetric)
                    done[u, a, v] = fixed_target(H, u, a, v)
                else:                                 # actions hidden: the action-blind graph
                    acts = {d['action'] for _, y, d in G.out_edges(u, data=True) if y == v}
                    if unit != 'edge' and len(acts) != 1:
                        raise ValueError(f"the test {u} -> {v} doesn't show its action, and "
                                         f"{len(acts)} actions take {u} to {v}: can't tell "
                                         f"which {unit!r} was held out")
                    H = _training_graph(G, u, next(iter(acts), None), v, unit, symmetric)
                    done[u, a, v] = fixed_target(_action_blind(H), u, None, v)
            out[i, b] = done[u, a, v]
    _FIXED[id(m)] = (m, out)
    return out.copy()


def block_mask(record, keep):
    """
    Which blocks of which samples of `record` to use, as a (samps, n_blocks) bool
    array (None for all), from `keep`:
        None        all of them
        'fixed'     the blocks whose test answer is well defined (fixed_answers)
        'fixed samples'  whole samples whose every block's answer is
        a function  f(record) -> bool array, (samps, n_blocks) or (samps,) for whole
                    samples, e.g. lambda r: fixed_answers(r)[:, -1] (samples whose
                    final block is well defined)
    """
    if keep is None:
        return None
    if keep == 'fixed':
        M = fixed_answers(record)
    elif keep == 'fixed samples':
        M = fixed_answers(record).all(1)
    elif callable(keep):
        M = np.asarray(keep(record), bool)
    else:
        raise ValueError(f"keep is None, 'fixed', 'fixed samples' or a function, not {keep!r}")
    n, nb = len(record['metrics']['logp_ctx']), len(record['metrics']['data']['blocks'][0])
    if M.shape == (n,):
        M = np.repeat(M[:, None], nb, 1)
    if M.shape != (n, nb):
        raise ValueError(f"keep gave shape {M.shape}; it should be ({n}, {nb}) or ({n},)")
    return M


def _masked_logp(logp_ctx, M):
    """logp_ctx with the left-out blocks' scores as nan, which plot_icl's summaries skip."""
    if M is None:
        return logp_ctx
    out = []
    for i, lpc in enumerate(logp_ctx):
        d = dict(lpc)
        for b in np.flatnonzero(~M[i]):
            for side in ('user', 'assistant'):
                if (side, int(b)) in d:
                    d[side, int(b)] = np.full(np.shape(lpc[side, int(b)]), np.nan)
        out.append(d)
    return out


# -- ExhaustiveRelationTask: by target, and the probability on every node ---------------

def _exhaustive_samples(record):
    """For each sample of an ExhaustiveRelationTask run: its target, its held-out edge
    (u, a, v) as graph labels, whether the answer is structurally unique (fixed_answers),
    the training set's other edges, and the label of each node token: a list of dicts."""
    if _task_name(record) != 'ExhaustiveRelationTask':
        raise ValueError(f"needs an ExhaustiveRelationTask run, not {_task_name(record)}")
    m = record['metrics']
    data = m['data']
    fixed = fixed_answers(record)[:, 0]                 # every block holds out the same edge
    nodes_r = [r for r, k in enumerate(data['kinds']) if k == 'node']
    show = record['task_args'].get('show')
    actions_shown = show is None or 'action' in show
    out = []
    for i in range(len(m['logp_ctx'])):
        G, label = _sample_graph(data, i, actions_shown)
        h = data['holdout'][i] if 'holdout' in data else {}
        code = block_code(data, i)
        gram = {int(x): r for r, x in enumerate(code[0])}
        toks = [gram[int(x)] for x in data['blocks'][i][0][-1][:-1]]
        u, a, v = _test_transition(toks, data['kinds'], label, actions_shown)
        rest = sorted((x, y, d['action']) for x, y, d in G.edges(data=True) if (x, y, d['action']) != (u, v, a))
        out.append({'target': h.get('target', v), 'edge': (u, a, v), 'unique': bool(fixed[i]),
                    'rest': rest, 'node_tokens': nodes_r, 'node_labels': [label[r] for r in nodes_r]})
    return out


def _training_set_title(s, d):
    """'1→0, 2→0, 3→1 | 0?' (with the action as 1-a0→0 when there are several)."""
    arrow = (lambda x, y, c: f'{x}→{y}') if d == 1 else (lambda x, y, c: f'{x}-{c}→{y}')
    u, a, _ = s['edge']
    es = [arrow(*e) for e in s['rest']]
    lines = [', '.join(es[j:j + 5]) for j in range(0, len(es), 5)]          # 5 edges a line
    return ',\n'.join(lines) + ' | ' + (f'{u}?' if d == 1 else f'{u}-{a}?')


def _panels(n, ncols, size, sharey):
    nc = max(1, min(ncols, n))
    nr = -(-n // nc)
    fig, axs = plt.subplots(nr, nc, figsize=(size[0] * nc, size[1] * nr), sharey=sharey, squeeze=False)
    axs = axs.ravel()
    for ax in axs[n:]:
        ax.set_visible(False)
    return fig, axs


def plot_heldout_targets(records, prm, by='index', where=None, refs=(), unique=True,
                         separate=False, perplexity=False, kl=False, err='std', traces=False,
                         renorm=True, cmap='tab10', ncols=4, size=(4, 3.5), sharey=True):
    """
    plot_heldout for ExhaustiveRelationTask runs made with `index` (and target=None, so
    each run's samples went through its training set's answers): one line per target,
    from the samples that got it, in one panel per value of `by` (one run each, the
    rest fixed by `where`), or with separate=True in a panel of its own, one per
    (value of `by`, target). unique: only targets whose answer no symmetry of the
    training set moves (fixed_answers). Each panel's title is the training set (its
    edges, and the query); each line's label gives the target, its answer node and how
    many samples it has. refs, perplexity, kl, err, traces, renorm: as for
    plot_heldout. Returns the axes.
    """
    cells, levels = _grid(prm, where, (by,), records)
    panels = []                                        # (value, record, samples, targets, left out)
    for val in levels[0]:
        rec = records[cells[(val,)]]
        S = _exhaustive_samples(rec)
        targets = sorted({x['target'] for x in S if x['unique'] or not unique})
        left = len({x['target'] for x in S}) - len(targets)
        if separate:
            panels += [(val, rec, S, [t], 0) for t in targets]
        else:
            panels.append((val, rec, S, targets, left))
    if not panels:
        raise ValueError("no targets to show" + (" (unique=False shows moved ones)" if unique else ""))
    fig, axs = _panels(len(panels), ncols, size, sharey)
    style = dict(ls='--', lw=0.8, ms=5, mfc='none')
    colors = plt.get_cmap(cmap)
    scores = {}
    for ax, (val, rec, S, targets, left) in zip(axs, panels):
        if id(rec) not in scores:
            scores[id(rec)] = heldout(rec, refs, perplexity, kl, renorm)
        s = scores[id(rec)]
        d = len({c for *_, c in S[0]['rest']} | {S[0]['edge'][1]})
        all_t = sorted({x['target'] for x in S})
        x = np.arange(1, s['model'].shape[1] + 1)
        if not targets:
            ax.text(0.5, 0.5, 'no structurally unique target', ha='center', va='center',
                    transform=ax.transAxes, color='0.4')
        for t in targets:
            rows = [i for i, x_ in enumerate(S) if x_['target'] == t]
            c = colors(all_t.index(t) % 10)           # a target's colour is the same either way
            if traces:
                draw_traces(ax, x, s['model'][rows], c, traces)
            mu, e = spread(s['model'][rows], err)
            ax.errorbar(x, mu, e, color=c, marker='o', ms=5, lw=2, capsize=2,
                        label=f"target {t}: node {S[rows[0]]['edge'][2]} (n={len(rows)})"
                              + ('' if S[rows[0]]['unique'] else ', moved'))
            for k, name in enumerate(refs):
                ax.plot(x, spread(s[name][rows], err)[0], color=c, alpha=0.8,
                        marker=REF_MARKERS[k % len(REF_MARKERS)], **style)
        head = _label(by, val) + (f', target {targets[0]}' if separate else '')
        rows0 = [i for i, x_ in enumerate(S) if not targets or x_['target'] == targets[0]]
        ax.set_title(f"{head}\n{_training_set_title(S[rows0[0]], d)}"
                     + (f"\n({left} moved target{'s' * (left != 1)} left out)" if left else ''),
                     fontsize='small')
        ax.set_xlabel('block')
        ax.set_ylabel('held-out ' + ylabel(perplexity, kl, renorm), fontsize='small')
        ax.set_xticks(x)
        handles, _ = ax.get_legend_handles_labels()
        handles += [Line2D([], [], color='0.4', marker=REF_MARKERS[k % len(REF_MARKERS)],
                           label=name, **style) for k, name in enumerate(refs)] if targets else []
        if handles:
            ax.legend(handles=handles, fontsize='x-small')
    fig.tight_layout()
    return axs


def node_logprobs(record, over='task'):
    """
    The model's log-probability on every node at each block's test token, for an
    ExhaustiveRelationTask run: an array (samps, n_blocks, n_nodes), the nodes in the
    order of _exhaustive_samples(record)[i]['node_labels'] (the graph's labels, the same
    in every sample of a run). over: what it's normalised over -- 'task' the block's
    whole task alphabet (as renorm=True elsewhere), 'nodes' just the nodes, 'vocab' the
    model's whole vocabulary (as renorm=False).
    """
    m = record['metrics']
    data = m['data']
    S = _exhaustive_samples(record)
    nb = len(data['blocks'][0])
    out = np.full((len(S), nb, len(S[0]['node_tokens'])), np.nan)
    for i, lpc in enumerate(m['logp_ctx']):
        code = block_code(data, i)
        for b in range(nb):
            lp = np.asarray(lpc[('assistant', b)][0], float)
            nodes = code[b][S[i]['node_tokens']]
            if over == 'task':
                lp = lp - spc.logsumexp(lp)
            elif over == 'nodes':
                lp = lp - spc.logsumexp(lp[nodes])
            elif over != 'vocab':
                raise ValueError(f"over is 'task', 'nodes' or 'vocab', not {over!r}")
            out[i, b] = lp[nodes]
    return out


def plot_node_probs(records, prm, by='index', where=None, unique=True, over='task', prob=False,
                    err='std', traces=False, ncols=4, size=(4, 3.5), sharey=True):
    """
    For ExhaustiveRelationTask runs: one panel per (value of `by`, target), each block's
    log-probability (prob=True: probability) on every node at the test token, averaged
    over the samples with that target, one line per node. The answer is drawn thick and
    black, the query's source dotted, and with several relations the source's targets
    under the other relations (which exclusivity rules out) dashed. over: as for
    node_logprobs. unique: only targets that no symmetry of the training set moves.
    Returns the axes.
    """
    cells, levels = _grid(prm, where, (by,), records)
    panels = []                                        # (value, record, samples, target, rows)
    for val in levels[0]:
        rec = records[cells[(val,)]]
        S = _exhaustive_samples(rec)
        for t in sorted({x['target'] for x in S if x['unique'] or not unique}):
            panels.append((val, rec, S, t, [i for i, x in enumerate(S) if x['target'] == t]))
    if not panels:
        raise ValueError("no targets to show" + (" (unique=False shows moved ones)" if unique else ""))
    fig, axs = _panels(len(panels), ncols, size, sharey)
    cache = {}
    colors = plt.get_cmap('tab10')
    for ax, (val, rec, S, t, rows) in zip(axs, panels):
        if id(rec) not in cache:
            cache[id(rec)] = node_logprobs(rec, over)
        L = cache[id(rec)][rows]                       # (samples, blocks, nodes)
        if prob:
            L = np.exp(L)
        u, a, v = S[rows[0]]['edge']
        d = len({c for *_, c in S[0]['rest']} | {a})
        ruled_out = ({y for x, y, c in S[0]['rest'] if x == u and c != a}
                     if a is not None else set())          # no actions shown: none ruled out
        x = np.arange(1, L.shape[1] + 1)
        for n, lab in enumerate(S[0]['node_labels']):
            role = ('answer' if lab == v else 'source' if lab == u else
                    'other relation' if lab in ruled_out else '')
            c = 'k' if role == 'answer' else colors(n % 10)
            ls = {'answer': '-', 'source': ':', 'other relation': '--'}.get(role, '-')
            if traces:
                draw_traces(ax, x, L[:, :, n], c, traces)
            mu, e = spread(L[:, :, n], err)
            ax.errorbar(x, mu, e, color=c, ls=ls, lw=2.5 if role == 'answer' else 1.2,
                        marker='o', ms=4, capsize=2,
                        label=f'node {lab}' + (f' ({role})' if role else ''))
        ax.set_title(f"{_label(by, val)}, target {t} (n={len(rows)})\n{_training_set_title(S[rows[0]], d)}",
                     fontsize='small')
        ax.set_xlabel('block')
        ax.set_ylabel(('probability' if prob else 'log-probability') + ' at the test'
                      + {'task': '', 'nodes': ', over the nodes', 'vocab': ', whole vocabulary'}[over],
                      fontsize='small')
        ax.set_xticks(x)
        ax.legend(fontsize='x-small')
    fig.tight_layout()
    return axs