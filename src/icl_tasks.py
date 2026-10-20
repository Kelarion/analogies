from dataclasses import dataclass

import numpy as np

import super_experiments as sxp
import grammars as gram
import automata as auto


@dataclass
class GrammarTask(sxp.Task):
    """Sequences of trials from any Sequential, in blocks of num_trials
    trials, each trial ending in the 0 delimiter. Subclasses change what a
    block is through _block_sizes, _new_block and _trial."""

    grammar: gram.Sequential
    num_trials: list
    samps: int
    share_toks: bool = False    # should train and test sequences share tokens?
    min_tok: int = 2
    max_tok: int = 50
    seed: int = None

    # -- what a block is ----------------------------------------------------

    def _block_sizes(self):
        if any(n is None for n in self.num_trials):
            raise ValueError("num_trials=None lists transitions: use AutomatonTask")
        return np.asarray(self.num_trials, int)

    def _new_block(self, rng, b):
        """Called before the first trial of block b; returns its context."""
        return None

    def _block_len(self, b, ctx):
        """Number of trials in block b, given its context (from _new_block)."""
        return self.num_trials[b]

    def _trial(self, rng, b, ctx, last):
        """One trial of block b, as (tokens, probs); last marks its final trial."""
        return self._draw(rng)

    def _draw(self, rng, tries=10_000, **kw):
        """A sentence with min_tok <= length <= max_tok, by rejection."""
        for _ in range(tries):
            toks, probs = self.grammar.generate(rng=rng, **kw)
            if self.min_tok <= len(toks) <= self.max_tok:
                return toks, probs
        raise RuntimeError(f"no sentence of length {self.min_tok}..{self.max_tok} "
                           f"in {tries} tries")

    # -- sampling -------------------------------------------------------------

    def sample(self):

        rng = np.random.default_rng(self.seed)
        nt = len(self.grammar.vocab)
        nblk = len(self._block_sizes())

        seqs = []
        Ys = []
        blks = []
        which_block = []
        for _ in range(self.samps):

            seq = []
            pr = []
            trial_len = []
            block = []                                     # block of each trial
            for b in range(nblk):
                ctx = self._new_block(rng, b)
                n = self._block_len(b, ctx)
                for j in range(n):
                    toks, probs = self._trial(rng, b, ctx, j == n - 1)
                    trial_len.append(len(toks) + 1)
                    block.append(b)
                    seq += toks + [0]
                    pr += probs

            X, Y = np.array(seq), np.array(pr)
            trl_blk = np.repeat(block, trial_len)                 # block of each position

            if self.share_toks:                    # one alphabet, remapped per block
                V = nt
                code = np.stack([np.arange(nt)] +
                                [np.concatenate([[0], rng.permutation(nt - 1) + 1])
                                for _ in range(nblk - 1)])
            else:                                  # disjoint alphabet per block
                V = (nt - 1) * nblk + 1
                code = np.zeros((nblk, nt), int)
                code[:, 1:] = np.arange(1, nt) + (nt - 1) * np.arange(nblk)[:, None]

            rows = np.arange(len(X))[:, None]
            Ynew = np.zeros((len(Y), V))
            Ynew[rows, code[trl_blk]] = Y          # move each row's columns to its block
            X = code[trl_blk, X]                   # 0 maps to 0, so delimiters survive
            Y = Ynew

            # split the encoded sequence into trials (each ending in the 0 delimiter),
            # then group the trials by block
            trials = np.split(X, np.where(X == 0)[0] + 1)[:-1]
            blocks = [[t for t, b in zip(trials, block) if b == k] for k in range(nblk)]

            seqs.append(X)
            Ys.append(Y)
            blks.append(blocks)
            which_block.append(trl_blk)

        return {'X': seqs, 'Y': Ys, 'B': which_block, 'blocks': blks, 'V': V}


@dataclass
class AutomatonTask(GrammarTask):
    """GrammarTask for an automaton (FSA or PDA). There can be any number of
    blocks, as for GrammarTask, and each is one of these kinds:

    - sampled (num_trials a number): num_trials walks, as in GrammarTask
    - listed (num_trials None): every transition once, in a fresh random
      order each sample (see the automaton's transitions())

    With holdout=True, each block holds out one random unit from all its
    trials but the last, which ends with it: a (node, action) pair for an
    FSA, a (state, pop, action) group for a PDA (see sample_holdout). A listed
    block leaves the unit's transitions out of the listing and ends with one
    of them, so it has a trial more than the listing it is left out of, and
    its length can vary between samples. With symmetric=True as well, the
    units that reverse it (see the automaton's reverse()) are held out of
    every trial, the last included, so u -> v is tested having seen neither
    u -> v nor v -> u."""

    holdout: bool = False
    symmetric: bool = False

    def _block_sizes(self):
        g = self.grammar
        if not isinstance(g, auto.Automaton):
            raise TypeError(f"AutomatonTask needs an automaton, not {type(g).__name__}")
        if self.symmetric and not self.holdout:
            raise ValueError("symmetric=True needs holdout=True")
        self._listings = {}                                  # forbidden units -> transitions()
        return np.array([len(self._listing(())) if n is None else n for n in self.num_trials])

    def _listing(self, forbid):
        """transitions() of the automaton with `forbid` held out, cached."""
        key = frozenset(forbid)
        if key not in self._listings:
            g = self.grammar.without(key) if key else self.grammar
            self._listings[key] = g.transitions()
        return self._listings[key]

    def _test_trial(self, rng, h, forbid):
        """One of the transitions of unit h, chosen uniformly, with its probs
        from the full listing: those listed once the rest of `forbid` is held
        out but not once h is too."""
        kept = {tuple(t) for t, _ in self._listing(forbid)}
        own = [tuple(t) for t, _ in self._listing(set(forbid) - {h}) if tuple(t) not in kept]
        if not own:
            raise ValueError(f"the transitions of {h!r} look like others under show, "
                             "so they can't be held out of a listing")
        full = {tuple(t): p for t, p in self._listing(())}
        t = own[rng.integers(len(own))]
        return list(t), full[t]

    def _new_block(self, rng, b):
        g = self.grammar
        h, forbid = None, set()
        if self.holdout:                                     # what to withhold
            h = g.sample_holdout(rng)
            forbid = ({h} | g.reverse(h)) if self.symmetric else {h}
        if self.num_trials[b] is None:                       # a shuffled listing
            trans = self._listing(forbid)
            trials = [trans[k] for k in rng.permutation(len(trans))]
            if h is not None:
                trials.append(self._test_trial(rng, h, forbid))
            return trials
        return (h, forbid) if self.holdout else None

    def _block_len(self, b, ctx):
        return len(ctx) if self.num_trials[b] is None else self.num_trials[b]

    def _trial(self, rng, b, ctx, last):
        if self.num_trials[b] is None:
            return ctx.pop(0)
        if self.holdout:
            h, forbid = ctx
            return self._draw(rng, forbid=forbid, end_with=h if last else None)
        return self._draw(rng)