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
    hint             0 none; 1 open each block with the start of its final trial; 2 and,
                     after the first, with how the previous block's words map onto its own
                     ("apple -> banana, cup -> cup"); 3 and with the final trial's answer
                     too, the final block's included, so the model is told the test answer:
                     a ceiling, or a control
    answer_only      "Output ONLY the next word." (left out in 'raw' mode)
    prompt           a spec (see prompts.py), used instead of the fields above when
                     given: e.g. 'one_word', or '' for no prompt

    Which tokens the blocks share. Each block has its own alphabet by default: the grammar's
    tokens become different task symbols in every block. share_toks shares some of them:
      False            nothing shared (each block its own symbols)
      True             everything shared: one alphabet for all blocks
      {'action'}       only tokens of these kinds, a set of some of the automaton's KINDS
      {'action', 'push'}  ('action', 'node' for an FSA; 'top', 'action', 'push', 'node' for a
                       PDA), e.g. the actions shared and the nodes separate per block
    Each setting has one spelling: a set must name some kinds but not all (those are False
    and True). Sets are values, not sweeps, so {'action'} needs no brackets in a sweep dict;
    a sweep over settings is a list, [False, {'action'}, True].
    shuffle_shared: whether the shared tokens are reassigned at random in each block after the
    first (True, as share_toks always did), or keep the same symbol in every block (False), so
    a shared action means the same action everywhere. Reassignment stays within a kind: an
    action's symbol only ever stands for another action. A grammar that isn't an automaton
    has one kind, 'token', so for it only True and False mean anything.
    The task's mapping from grammar tokens to symbols, block by block, is returned as
    data['code'] (code[b, r]: grammar token r's symbol in block b), and each grammar token's
    kind as data['kinds']. The model only maps symbols to its own tokens.

    The grammar is either given (grammar=..., or a Build of one), or left out and made by
    make_grammar() from the task's own fields, as the subclasses at the end of this file do.
    Either way it is made when the task is, i.e. inside the job. A subclass can replace it
    between samples (see _new_sample): an AutomatonTask that makes its own grammar makes a
    fresh one for every sample. Fields are keyword-only, so subclasses can add required
    ones."""

    grammar: gram.Sequential = None
    num_trials: list
    samps: int
    share_toks: object = False  # False, True, or a set of kinds: {'action'} (see above)
    shuffle_shared: bool = False # reassign the shared tokens at random in each later block?
    min_tok: int = 2
    max_tok: int = 50
    seed: int = None
    explicitness: int = 0
    describe_format: bool = False
    describe_vocab: bool = False
    describe_task: bool = False
    hint: int = 0
    answer_only: bool = False
    prompt: str = None

    def __post_init__(self):
        # stored as an int however it's given (2, np.int64(2), 'remap'), before .args is
        # recorded, so the same level is always the same run
        self.hint = prompts.level(self.hint, prompts.HINT, 'hint')
        super().__post_init__()                           # records .args, builds any Build
        if self.grammar is None:
            self.grammar = self.make_grammar()
        self._shared_kinds()                              # checks share_toks
        spec = self.prompt_spec                           # checks the fields
        prompts.check(spec, self.prompt_facts())

    def make_grammar(self, sample=0):
        """The grammar, from the task's fields, when none is given: sample's, for tasks
        that make one per sample (see AutomatonTask). Subclasses that describe their
        grammar by fields override this."""
        raise TypeError(f"{type(self).__name__} needs grammar=...")

    @property
    def prompt_spec(self):
        """The spec of the prompt (see prompts.py): `prompt` if given, else the fields'."""
        if self.prompt is not None:
            return self.prompt
        return prompts.spec_from(self.explicitness, self.describe_format, self.describe_vocab,
                                 self.describe_task, self.hint, self.answer_only)

    # -- which tokens are shared ------------------------------------------------

    def _kind_names(self):
        """The grammar's kinds, in order: an automaton's KINDS, else the one kind 'token'."""
        kinds = getattr(self.grammar, 'KINDS', None)
        return tuple(kinds) if kinds and hasattr(self.grammar, '_token_of') else ('token',)

    def _kinds(self):
        """Each grammar token's kind (None for the delimiter, 0)."""
        nt = len(self.grammar.vocab)
        if self._kind_names() == ('token',):
            return [None] + ['token'] * (nt - 1)
        out = [None] * nt
        for kind, table in self.grammar._token_of.items():
            for t in table.values():
                out[t] = kind
        return out

    def _shared_kinds(self):
        """The kinds share_toks shares, in kind order. Raises unless it is False, True, or a set
        naming some of the grammar's kinds but not all (so each setting has one spelling)."""
        s, order = self.share_toks, self._kind_names()
        if s is False or s is None:
            return ()
        if s is True:
            return order
        if not isinstance(s, (set, frozenset)):
            raise TypeError(f"share_toks is False, True or a set of kinds like {{'action'}}, not {s!r}")
        if set(s) - set(order):
            raise ValueError(f"share_toks={s!r}: the kinds are {set(order)}")
        if not s:
            raise ValueError("share_toks=set() shares nothing: write share_toks=False")
        if set(s) == set(order):
            raise ValueError(f"share_toks={s!r} shares every kind: write share_toks=True")
        return tuple(k for k in order if k in s)

    def _code(self, rng, nblk):
        """(code, V): code[b, r] is the symbol grammar token r becomes in block b (0 stays 0);
        V the number of symbols. Shared tokens come first (1..), the same in every block unless
        shuffle_shared reassigns them within each kind in blocks after the first; then each
        block's own tokens. With nothing shared this is the disjoint layout as before, and with
        everything shared in a grammar of one kind the same draws as before."""
        kinds = self._kinds(); shared = set(self._shared_kinds())
        nt = len(kinds)
        sh = np.array([r for r in range(1, nt) if kinds[r] in shared], int)
        own = np.array([r for r in range(1, nt) if kinds[r] not in shared], int)
        code = np.zeros((nblk, nt), int)
        code[:, sh] = 1 + np.arange(len(sh))
        code[:, own] = 1 + len(sh) + np.arange(len(own)) + len(own) * np.arange(nblk)[:, None]
        if self.shuffle_shared and len(sh):
            groups = [sh[[kinds[r] == k for r in sh]] for k in self._kind_names() if k in shared]
            for b in range(1, nblk):
                for g in groups:
                    code[b, g] = code[0, g][rng.permutation(len(g))]
        return code, 1 + len(sh) + nblk * len(own)

    # -- what the prompt can say about the task --------------------------------

    def prompt_facts(self):
        """What a prompt may assume about the task (the `when` of a prompts.Part sees
        these, and its text can use them as fields). Plain values, so they are saved
        with the data. Subclasses add to them."""
        g = self.grammar
        fixed = isinstance(g, gram.RegularGrammar) and g.length is not None
        return {'structure': 'grammar', 'noun': 'grammar', 'shows': None,
                'trial_len': g.length if fixed else None,      # tokens per trial, if fixed
                'n_blocks': len(self.num_trials), **self._share_facts(),
                'holdout': False, 'listed': False}

    def _share_facts(self):
        """share_toks: every token shared and reassigned per block (what prompts have always
        meant by it); shared_kinds: the kinds shared at all; shuffle_shared: as the field."""
        kinds = self._shared_kinds()
        return {'share_toks': kinds == self._kind_names() and self.shuffle_shared,
                'shared_kinds': kinds, 'shuffle_shared': self.shuffle_shared}

    # -- what a block is ----------------------------------------------------

    def _new_sample(self, rng, s):
        """Called before the first block of sample s. A subclass can replace the grammar
        here, with _set_grammar."""

    def _set_grammar(self, g):
        """Use grammar g from now on, e.g. a fresh one for the next sample. Its tokens must
        match the current grammar's, kind for kind, since the prompt, the symbols and the
        data's V are set up once for every sample."""
        before, prev = self._kinds(), self.grammar
        self.grammar = g
        if self._kinds() != before:
            self.grammar = prev
            raise ValueError(f"the new grammar has {len(g.vocab)} tokens, the old one "
                             f"{len(prev.vocab)}, or other kinds of token: the samples need "
                             "the same ones (e.g. every action must have some edge)")

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
        nblk = len(self._block_sizes())

        seqs = []
        codes = []
        Ys = []
        blks = []
        which_block = []
        self._used = []                                    # the grammar of each sample
        for s in range(self.samps):
            self._new_sample(rng, s)
            self._used.append(self.grammar)

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

            code, V = self._code(rng, nblk)        # grammar token -> symbol, per block

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
            codes.append(code)
            Ys.append(Y)
            blks.append(blocks)
            which_block.append(trl_blk)

        if facts['trial_len'] is not None:                 # the prompt may say so
            lens = {len(t) - 1 for b in blks for blk in b for t in blk}
            assert lens == {facts['trial_len']}, f"trial lengths {lens}, expected {facts['trial_len']}"

        return {'X': seqs, 'Y': Ys, 'B': which_block, 'blocks': blks, 'V': V,
                'code': codes, 'kinds': self._kinds(),
                'prompt': {'spec': spec, 'facts': facts}}


@dataclass(kw_only=True)
class AutomatonTask(GrammarTask):
    """GrammarTask for an automaton (FSA or PDA). There can be any number of
    blocks, as for GrammarTask, and each is one of these kinds:

    - sampled (num_trials a number): num_trials walks, as in GrammarTask
    - listed (num_trials None): every transition once, in a fresh random
      order each sample (see the automaton's transitions())

    With holdout=True, each block holds out one random unit from all its
    trials but the last, which ends with it: for an FSA, Edge(u, v) -- every
    edge u -> v whatever its action -- for a transition drawn uniformly (or
    with holdout_unit="pair" a (node, action) pair); for a PDA, a (state, pop,
    action) group (see sample_holdout). A listed
    block leaves the unit's transitions out of the listing and ends with one
    of them, so it has a trial more than the listing it is left out of, and
    its length can vary between samples. With symmetric=True as well, the
    units that reverse it (see the automaton's reverse()) are held out of
    every trial, the last included, so u -> v is tested having seen neither
    u -> v nor v -> u. With keep_connected (FSA edges only), the hold-out
    never splits a weakly or strongly connected component of the graph (see
    holdout_edges). For an FSA (either unit), the hold-out never takes away the
    last edge of a node or of an action, so every node and action the trials
    show is still somewhere in the block's context, and only the combination is
    new in the test (see FSA.sample_holdout). A sampled block only draws
    hold-outs that some walk can end on with all that held out (see
    can_end_with), redrawing the rest.
    max_redraws: how many draws to try before giving up."""

    holdout: bool = False
    symmetric: bool = False
    holdout_unit: str = "edge"          # FSA only: "edge" or "pair"
    keep_connected: bool = True         # FSA edges only
    max_redraws: int = 1000

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

    def _new_sample(self, rng, s):
        """A fresh automaton for each sample, make_grammar(s), when the task makes its own
        (sample 0's is made with the task). For a random graph (RelationTask) every sample
        has its own; a fixed one (CycleTask, GridTask) just comes out the same again. A
        given grammar is kept for every sample."""
        if s > 0 and self.args['grammar'] is None:
            self._set_grammar(self.make_grammar(s))

    def sample(self):
        """GrammarTask.sample, plus the automaton of each sample as data['automaton'][i]
        (see Automaton.describe), which the model saves with the rest of the data."""
        data = super().sample()
        desc = {}
        for g in self._used:
            if id(g) not in desc:
                desc[id(g)] = g.describe()
        return dict(data, automaton=[desc[id(g)] for g in self._used])

    def _set_grammar(self, g):
        super()._set_grammar(g)
        self._listings = {}                                  # listings were the old grammar's

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
            for _ in range(self.max_redraws):
                h = (g.sample_holdout(rng, unit=self.holdout_unit, symmetric=self.symmetric,
                                      keep_connected=self.keep_connected
                                      and self.holdout_unit == "edge")
                     if isinstance(g, auto.FSA) else g.sample_holdout(rng))
                forbid = ({h} | g.reverse(h)) if self.symmetric else {h}
                if self.num_trials[b] is None or g.can_end_with(h, forbid):
                    break                                    # listings need no walk
            else:
                raise ValueError(f"no hold-out in {self.max_redraws} draws could be "
                                 "ended on by a walk that avoids it")
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
# job, afresh for each sample (see AutomatonTask._new_sample). For random graphs, graph_seed
# fixes which graphs: the same graph_seed gives the same graph for each sample in every job
# (across prompt conditions, models and reruns), None fresh ones.

@dataclass(kw_only=True)
class RelationTask(AutomatonTask):
    """AutomatonTask on an FSA over a random relation (automata.random_relation)."""
    k: int                       # nodes
    d: int = 2                   # relations
    degree: float = 0.5
    commutes: float = 0.0
    inverse: bool = False
    total: bool = False
    strong: bool = True
    backtrack: bool = False
    length: int = None           # steps per walk; sampled blocks with holdout need this
    mean_length: float = None    # ... or this (see automata.FSA)
    graph_seed: int = 0

    def make_grammar(self, sample=0):
        """The FSA on sample's own random graph, seeded by (graph_seed, sample): so a
        graph_seed gives the same graph for each sample in every job, and different
        graph_seeds share none. graph_seed=None: a fresh graph each time."""
        seed = self.graph_seed
        if seed is not None:
            seed = int(np.random.SeedSequence([seed, sample]).generate_state(1)[0])
        G = auto.spatial_relation(self.k, d=self.d, degree=self.degree,
                                 commutes=self.commutes, inverse=self.inverse, 
                                 total=self.total, seed=seed,
                                 connectedness='strong' if self.strong else 'weak',
                                 )

        return auto.FSA(G, backtrack=self.backtrack, length=self.length,
                        mean_length=self.mean_length)


@dataclass(kw_only=True)
class CycleTask(AutomatonTask):
    """AutomatonTask on an FSA walking a directed k-cycle (automata.cycle)."""
    k: int
    length: int = 1

    def make_grammar(self, sample=0):
        return auto.FSA(auto.cycle(self.k), length=self.length)


@dataclass(kw_only=True)
class GridTask(AutomatonTask):
    """AutomatonTask on an FSA moving around an n x m grid (automata.grid)."""
    n: int
    m: int = None
    torus: bool = False
    slip: float = 0.0
    length: int = 1

    def make_grammar(self, sample=0):
        return auto.FSA(auto.grid(self.n, self.m, torus=self.torus, slip=self.slip),
                        length=self.length)