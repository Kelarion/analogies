"""
Choosing runs from a batch by their parameters, and plotting them on a grid.

    records, prm = su.load_experiments(task_args, mod_args, SAVE_DIR=SAVE_DIR)
    varying(prm)                                          # what differs between the runs
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

from icl_utils import xent, plot_icl, split_block, ylabel, spread, draw_traces, block_code


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
                s = heldout(records[r], refs, perplexity, kl, renorm, keep)
                x = np.arange(1, s['model'].shape[1] + 1)
                nmax = max(nmax, len(x))
                if traces:
                    draw_traces(ax, x, s['model'], colors[h], traces)
                mu, e = spread(s['model'], err)
                ax.errorbar(x, mu, e, color=colors[h], marker='o', ms=6, lw=2, capsize=2)
                for k, name in enumerate(refs):
                    ax.plot(x, spread(s[name], err)[0], color=colors[h], alpha=0.8,
                            marker=REF_MARKERS[k % len(REF_MARKERS)], **style)
            name = _title(rows, cols, rv, cv)
            if keep is not None and len(here) == 1:
                M = block_mask(records[here[0][1]], keep)
                name += f' [{int(M.sum())}/{M.size} blocks]'
            ax.set_title(name)
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
    has_top = 'top_ids' in m
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
            ctx, asst = lpc[('user', b)], lpc[('assistant', b)]
            n_ctx, comp = len(ctx) - 1, split_block(trials, k_show, k_hide)[1]
            n = sum(len(tr) for tr in trials)
            V = ctx.shape[1]
            span = {}                                       # 'user' / 'assistant' -> (n, V) rows
            span['user'] = np.full((n, V), np.nan)
            span['user'][:n_ctx + 1] = ctx[:n_ctx + 1]      # row r predicts token r
            span['assistant'] = np.full((n, V), np.nan)
            span['assistant'][n_ctx:n_ctx + len(comp)] = asst[:len(comp)]
            tops = {}
            if has_top:                                     # the same rows of the top tokens
                for side, lo, hi, s0 in [('user', 0, n_ctx + 1, 0), ('assistant', n_ctx, n_ctx + len(comp), 0)]:
                    tid, tlp = np.full((n, 2), -1), np.full((n, 2), np.nan)
                    src_i, src_l = m['top_ids'][i][(side, b)], m['top_lp'][i][(side, b)]
                    k = min(hi - lo, len(src_i))
                    tid[lo:lo + k], tlp[lo:lo + k] = src_i[:k], src_l[:k]
                    tops[side] = (tid, tlp)
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
                lp = span[side]
                with np.errstate(invalid='ignore', divide='ignore'):
                    row['ll' + suffix] = list(lp[:, sym])
                    mass = np.exp(spc.logsumexp(lp, -1))
                row['task_mass' + suffix] = np.where(np.isnan(lp).all(-1), np.nan, mass)
                if has_top:
                    tid, tlp = tops[side]
                    for c, nm in enumerate(['any', 'text']):
                        row[f'top_{nm}{suffix}'] = [word_of(j) for j in tid[:, c]]
                        row[f'p_top_{nm}{suffix}'] = np.exp(tlp[:, c])
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


# -- which tests are well defined -----------------------------------------------------

_FIXED = {}                                    # id(metrics) -> (metrics, result): a cache


def _task_name(record):
    t = record.get('task')
    return t if isinstance(t, str) else getattr(t, '__name__', str(t))


def _sample_graph(data, i):
    """Sample i's graph, as a MultiDiGraph with an `action` on each edge, and the label
    of each grammar token. From data['automaton'] (an FSA description) when the run
    saved it; otherwise rebuilt from every transition its blocks show, with grammar
    tokens as labels (an edge no block shows is missed)."""
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
    for b, trials in enumerate(data['blocks'][i]):
        gram = {int(x): r for r, x in enumerate(code[b])}          # symbol -> grammar token
        for tr in trials:
            toks = [gram[int(x)] for x in tr[:-1]]
            for p in range(0, len(toks) - 2, 2):
                G.add_edge(toks[p], toks[p + 2], action=toks[p + 1])
    return G, list(range(code.shape[1]))


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
    out = np.zeros((len(m['logp_ctx']), len(data['blocks'][0])), bool)
    for i in range(len(out)):
        G, label = _sample_graph(data, i)
        code = block_code(data, i)
        done = {}                                    # blocks testing the same transition
        for b, trials in enumerate(data['blocks'][i]):
            gram = {int(x): r for r, x in enumerate(code[b])}       # symbol -> grammar token
            u, a, v = (label[gram[int(x)]] for x in trials[-1][:-1][-3:])
            if (u, a, v) not in done:
                done[u, a, v] = fixed_target(_training_graph(G, u, a, v, unit, symmetric), u, a, v)
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
    out = []
    for i in range(len(m['logp_ctx'])):
        G, label = _sample_graph(data, i)
        h = data['holdout'][i] if 'holdout' in data else {}
        code = block_code(data, i)
        gram = {int(x): r for r, x in enumerate(code[0])}
        u, a, v = (label[gram[int(x)]] for x in data['blocks'][i][0][-1][:-1][-3:])
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
        ruled_out = {y for x, y, c in S[0]['rest'] if x == u and c != a}
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