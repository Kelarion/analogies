from dataclasses import dataclass

import numpy as np

import super_experiments as sxp
import grammars as gram
import automata as auto
import prompts


@dataclass(kw_only=True)
class GrammarTask(sxp.Task):
    """Sequences of trials from any Sequential, in blocks of num_trials
    trials, each trial ending in the 0 delimiter. Subclasses change what a
    block is through _block_sizes, _new_block and _trial.

    The prompt the model gets is chosen by the fields below (see prompts.spec_from),
    tailored to the task by prompt_facts, checked against the task when the task is
    made, and passed on to the model as data['prompt']. So a sweep over prompts is a
    sweep over task_args, one field at a time. All off (the default) is no prompt.

    explicitness     0 none, 1 vague, 2 moderate, 3 explicit: what the model is told
                     about where the sequences come from
    describe_format  say what a trial looks like (triplets, N words, varying lengths)
    describe_vocab   say what the words are (from the model's token pool)
    describe_task    "Your job is to complete the next word in the sequence."
    hint             open each block with the start of its final trial
    answer_only      "Output ONLY the next word." (left out in 'raw' mode)
    prompt           a spec (see prompts.py), used instead of the fields above when
                     given: e.g. 'one_word', or '' for no prompt

    The grammar is either given (grammar=..., or a Build of one), or left out and made by
    make_grammar() from the task's own fields, as the subclasses at the end of this file do.
    Either way it is made when the task is, i.e. inside the job. Fields are keyword-only,
    so subclasses can add required ones."""

    grammar: gram.Sequential = None
    num_trials: list
    samps: int
    share_toks: bool = False    # should train and test sequences share tokens?
    min_tok: int = 2
    max_tok: int = 50
    seed: int = None
    explicitness: int = 0
    describe_format: bool = False
    describe_vocab: bool = False
    describe_task: bool = False
    hint: bool = False
    answer_only: bool = False
    prompt: str = None

    def __post_init__(self):
        super().__post_init__()                           # records .args, builds any Build
        if self.grammar is None:
            self.grammar = self.make_grammar()
        spec = self.prompt_spec                           # checks the fields
        prompts.check(spec, self.prompt_facts())

    def make_grammar(self):
        """The grammar, from the task's fields, when none is given. Subclasses that
        describe their grammar by fields override this."""
        raise TypeError(f"{type(self).__name__} needs grammar=...")

    @property
    def prompt_spec(self):
        """The spec of the prompt (see prompts.py): `prompt` if given, else the fields'."""
        if self.prompt is not None:
            return self.prompt
        return prompts.spec_from(self.explicitness, self.describe_format, self.describe_vocab,
                                 self.describe_task, self.hint, self.answer_only)

    # -- what the prompt can say about the task --------------------------------

    def prompt_facts(self):
        """What a prompt may assume about the task (the `when` of a prompts.Part sees
        these, and its text can use them as fields). Plain values, so they are saved
        with the data. Subclasses add to them."""
        g = self.grammar
        fixed = isinstance(g, gram.RegularGrammar) and g.length is not None
        return {'structure': 'grammar', 'noun': 'grammar', 'shows': None,
                'trial_len': g.length if fixed else None,      # tokens per trial, if fixed
                'n_blocks': len(self.num_trials), 'share_toks': self.share_toks,
                'holdout': False, 'listed': False}

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

        facts, spec = self.prompt_facts(), self.prompt_spec
        prompts.check(spec, facts)
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

        if facts['trial_len'] is not None:                 # the prompt may say so
            lens = {len(t) - 1 for b in blks for blk in b for t in blk}
            assert lens == {facts['trial_len']}, f"trial lengths {lens}, expected {facts['trial_len']}"

        return {'X': seqs, 'Y': Ys, 'B': which_block, 'blocks': blks, 'V': V,
                'prompt': {'spec': spec, 'facts': facts}}


@dataclass(kw_only=True)
class AutomatonTask(GrammarTask):
    """GrammarTask for an automaton (FSA or PDA). There can be any number of
    blocks, as for GrammarTask, and each is one of these kinds:

    - sampled (num_trials a number): num_trials walks, as in GrammarTask
    - listed (num_trials None): every transition once, in a fresh random
      order each sample (see the automaton's transitions())

    With holdout=True, each block holds out one random unit from all its
    trials but the last, which ends with it: a (u, v) node pair -- every
    edge u -> v -- for an FSA, a (state, pop, action) group for a PDA (see
    sample_holdout). A listed
    block leaves the unit's transitions out of the listing and ends with one
    of them, so it has a trial more than the listing it is left out of, and
    its length can vary between samples. With symmetric=True as well, the
    units that reverse it (see the automaton's reverse()) are held out of
    every trial, the last included, so u -> v is tested having seen neither
    u -> v nor v -> u."""

    holdout: bool = False
    symmetric: bool = False

    def prompt_facts(self):
        g = self.grammar
        if not isinstance(g, auto.Automaton):        # _block_sizes says so when sampling
            return super().prompt_facts()
        start, step = len(g._start_kinds), len(g._step_kinds)
        lens = {start + step if n is None                  # a listing: one step each
                else (start + g.length * step if g.length is not None else None)
                for n in self.num_trials}
        pda = isinstance(g, auto.PDA)
        return dict(super().prompt_facts(),
                    structure='pda' if pda else 'fsa', noun='automaton' if pda else 'graph',
                    shows=tuple(g._step_kinds), trial_len=lens.pop() if len(lens) == 1 else None,
                    holdout=self.holdout, listed=any(n is None for n in self.num_trials))

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


# -- tasks that describe their grammar by fields -------------------------------------------
# Each field is an ordinary task argument: it can be swept (a list), linked to other fields
# (tuples, matched 1-1), and matched on its own when loading. The grammar is made inside the
# job. For random graphs, graph_seed fixes which graph: the same graph_seed gives the same
# graph in every job (across prompt conditions, models and reruns), None a fresh one per job.

@dataclass(kw_only=True)
class RelationTask(AutomatonTask):
    """AutomatonTask on an FSA over a random relation (automata.random_relation)."""
    k: int                       # nodes
    d: int = 2                   # relations
    degree: int = 0.5
    commutes: float = 0.0
    inverse: bool = False
    total: bool = False
    backtrack: bool = False
    length: int = None           # steps per walk; sampled blocks with holdout need this
    mean_length: float = None    # ... or this (see automata.FSA)
    graph_seed: int = 0

    def make_grammar(self):
        G = auto.spatial_relation(self.k, d=self.d, degree=self.degree,
                                 commutes=self.commutes, inverse=self.inverse, 
                                 total=self.total, 
                                 seed=self.graph_seed)

        return auto.FSA(G, backtrack=self.backtrack, length=self.length,
                        mean_length=self.mean_length)


@dataclass(kw_only=True)
class CycleTask(AutomatonTask):
    """AutomatonTask on an FSA walking a directed k-cycle (automata.cycle)."""
    k: int
    length: int = 1

    def make_grammar(self):
        return auto.FSA(auto.cycle(self.k), length=self.length)


@dataclass(kw_only=True)
class GridTask(AutomatonTask):
    """AutomatonTask on an FSA moving around an n x m grid (automata.grid)."""
    n: int
    m: int = None
    torus: bool = False
    slip: float = 0.0
    length: int = 1

    def make_grammar(self):
        return auto.FSA(auto.grid(self.n, self.m, torus=self.torus, slip=self.slip),
                        length=self.length)
