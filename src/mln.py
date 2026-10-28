"""Random digraphs from a Markov logic network.

    model = MLN(20, ["2.0 R(x,y) & R(y,z) -> R(x,z)",    # transitivity
                     "1.0 R(x,y) -> R(y,x)"],             # symmetry
                field=-1.5, constraint="strong")
    print(model.describe())
    G = model.sample()                                    # or mln(20, [...], ...)
    FSA(G, length=10)

    # from constructors, with a block model over 3 latent groups
    model = MLN(30, [symmetric("R"), transitive("R"),
                     categorical("In"), sbm("In", "R", within=1.0, between=-3.0)],
                latent=3, beta=2.0)

Each rule is a weight and a formula. Formulas use predicates applied to
variables -- R(x,y), C(x) -- combined with ~ (or !), &, |, -> (or =>),
<-> (or <=>), parentheses, True/False, x = y, x != y, x < y, and
exists G: ... (expanded into a disjunction over G's domain). An argument may
also be a node's number, e.g. In(x,2) for latent node 2. A rule may end with
"where <condition>", a formula over the rule's variables that only involves
atoms already fixed when the rule is sampled (evidence, the latent graph, or an
earlier stage): only the groundings where it holds are kept, e.g.
"2.0 R(x,y) & R(y,z) -> R(x,z) where exists G: In(x,G) & In(y,G) & In(z,G)" is
transitivity within groups. The condition adds nothing to the rule's size, and
fewer groundings mean less memory. Variables are
universally quantified: a rule is grounded by substituting nodes for its
variables in every way (distinct nodes for distinct variables, by default),
and each grounding is one factor. The weight is "inf" (or "-inf") for a hard
rule, which is given weight `hard`.

    P(atoms) = exp(sum_rules weight * #true groundings) / Z

Binary predicates listed in `relations` become the graph's edges, with action
= their index in that list. Every other predicate is hidden: sampled along
with the edges and left out of the graph, except that unary predicates are
stored as boolean node attributes and others in G.graph["hidden"]. Atoms that
appear in no grounding are fixed false, so with distinct=True there are no
self-loops unless a rule mentions R(x,x).
"""

import itertools
import re
import types as _types
import warnings

import networkx as nx
import numpy as np


# -- parsing ------------------------------------------------------------------

_TOKEN = re.compile(r"\s*(<->|<=>|->|=>|!=|[()~!&|,=<:]|[A-Za-z_][A-Za-z0-9_]*|[0-9]+)")
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*$")
_WEIGHT = re.compile(r"\s*([+-]?(?:inf|[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?))\s*:?\s*(.+)$",
                     re.IGNORECASE)


def _tokenize(s):
    out, pos, s = [], 0, s.rstrip()
    while pos < len(s):
        m = _TOKEN.match(s, pos)
        if not m:
            raise ValueError(f"can't parse {s[pos:]!r}")
        out.append(m.group(1))
        pos = m.end()
    return out


class _Parser:
    """Recursive descent, loosest first: <->, -> (right-associative), |, &, ~."""

    def __init__(self, text):
        self.text, self.toks, self.i = text, _tokenize(text), 0

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self, want=None):
        t = self.peek()
        if t is None or (want is not None and t != want):
            raise ValueError(f"in {self.text!r}: expected {want or 'more'}, got {t!r}")
        self.i += 1
        return t

    def name(self):
        t = self.take()
        if not _NAME.match(t):
            raise ValueError(f"in {self.text!r}: expected a name, got {t!r}")
        return t

    def arg(self):                               # a variable, or a node's number
        t = self.take()
        if t.isdigit():
            return int(t)
        if not _NAME.match(t):
            raise ValueError(f"in {self.text!r}: expected a variable or a number, got {t!r}")
        return t

    def parse(self):
        e = self.iff()
        if self.peek() is not None:
            raise ValueError(f"in {self.text!r}: unexpected {self.peek()!r}")
        return e

    def iff(self):
        a = self.imp()
        while self.peek() in ("<->", "<=>"):
            self.take()
            a = ("iff", a, self.imp())
        return a

    def imp(self):
        a = self.disj()
        if self.peek() in ("->", "=>"):
            self.take()
            return ("imp", a, self.imp())
        return a

    def disj(self):
        xs = [self.conj()]
        while self.peek() == "|":
            self.take()
            xs.append(self.conj())
        return xs[0] if len(xs) == 1 else ("or", xs)

    def conj(self):
        xs = [self.unary()]
        while self.peek() == "&":
            self.take()
            xs.append(self.unary())
        return xs[0] if len(xs) == 1 else ("and", xs)

    def unary(self):
        t = self.peek()
        if t in ("~", "!"):
            self.take()
            return ("not", self.unary())
        if t == "(":
            self.take()
            e = self.iff()
            self.take(")")
            return e
        n = self.name()
        if n in ("True", "False"):
            return ("const", n == "True")
        if n == "exists":                                    # exists G, H: body
            vs = [self.name()]
            while self.peek() == ",":
                self.take()
                vs.append(self.name())
            self.take(":")
            return ("exists", tuple(vs), self.iff())
        if self.peek() == "(":                               # a predicate
            self.take()
            args = [self.arg()]
            while self.peek() == ",":
                self.take()
                args.append(self.arg())
            self.take(")")
            return ("atom", n, tuple(args))
        op = self.take()                                     # x = y, x != y, x < y
        if op not in ("=", "!=", "<"):
            raise ValueError(f"in {self.text!r}: expected '(', '=', '!=' or '<' after {n!r}")
        if op == "<":
            return ("lt", n, self.name())
        e = ("eq",) + tuple(sorted((n, self.name())))
        return e if op == "=" else ("not", e)


def _literals(e, out):
    if e[0] in ("atom", "eq", "lt"):
        if e not in out:
            out.append(e)
    elif e[0] in ("not", "exists"):
        _literals(e[-1], out)
    elif e[0] in ("and", "or"):
        for x in e[1]:
            _literals(x, out)
    elif e[0] in ("imp", "iff"):
        _literals(e[1], out)
        _literals(e[2], out)


def _eval(e, val):
    kind = e[0]
    if kind in ("atom", "eq", "lt"):
        return val[e]
    if kind == "const":
        return e[1]
    if kind == "not":
        return not _eval(e[1], val)
    if kind == "and":
        return all(_eval(x, val) for x in e[1])
    if kind == "or":
        return any(_eval(x, val) for x in e[1])
    if kind == "imp":
        return (not _eval(e[1], val)) or _eval(e[2], val)
    return _eval(e[1], val) == _eval(e[2], val)


def _bound(e, out):
    """Variables bound by exists anywhere in e."""
    if e[0] == "exists":
        out.update(e[1])
    if e[0] in ("not", "exists"):
        _bound(e[-1], out)
    elif e[0] in ("and", "or"):
        for x in e[1]:
            _bound(x, out)
    elif e[0] in ("imp", "iff"):
        _bound(e[1], out)
        _bound(e[2], out)
    return out


def _expand(e, dom, env=None):
    """Replace each exists by a disjunction over its variables' domains (sizes
    in dom, by type), substituting numbers for the bound variables; equalities
    and comparisons between two numbers become constants."""
    env = env or {}
    kind = e[0]
    if kind == "atom":
        return ("atom", e[1], tuple(env.get(a, a) if isinstance(a, str) else a for a in e[2]))
    if kind in ("eq", "lt"):
        a, b = (env.get(x, x) if isinstance(x, str) else x for x in e[1:])
        if isinstance(a, int) and isinstance(b, int):
            return ("const", a == b if kind == "eq" else a < b)
        return (kind, a, b)
    if kind == "const":
        return e
    if kind == "exists":
        vs, body = e[1], e[2]
        parts = [_expand(body, dom, {**env, **dict(zip(vs, vals))})
                 for vals in itertools.product(*[range(dom[_vtype(v)]) for v in vs])]
        return ("or", parts) if parts else ("const", False)
    if kind == "not":
        return ("not", _expand(e[1], dom, env))
    if kind in ("and", "or"):
        return (kind, [_expand(x, dom, env) for x in e[1]])
    return (kind, _expand(e[1], dom, env), _expand(e[2], dom, env))


_WHERE = re.compile(r"\bwhere\b")


def _split_where(text):
    """(the rule, its where condition or None)."""
    parts = _WHERE.split(text, maxsplit=1)
    return (parts[0].rstrip(), parts[1].strip()) if len(parts) == 2 else (text, None)


def _eval_vec(e, val):
    """Evaluate a formula on arrays: val(literal) gives a bool array per literal."""
    kind = e[0]
    if kind in ("atom", "eq", "lt"):
        return val(e)
    if kind == "const":
        return e[1]
    if kind == "not":
        return np.logical_not(_eval_vec(e[1], val))
    if kind in ("and", "or"):
        parts = [_eval_vec(x, val) for x in e[1]]
        out = parts[0]
        for p in parts[1:]:
            out = (out & p) if kind == "and" else (out | p)
        return out
    a, b = _eval_vec(e[1], val), _eval_vec(e[2], val)
    return (~a | b) if kind == "imp" else (a == b)


def _parse_rule(text, dom=None):
    """(weight, literals, free variables, truth table over the literals).

    Literals are atoms (arguments are variables or node numbers), equalities
    and comparisons. With dom ({"n": k, "l": m}) any exists is expanded over
    the domains first; without it, a rule with exists gets no truth table
    (enough for finding predicates and types)."""
    text = _split_where(text)[0]                 # the condition is handled when grounding
    m = _WEIGHT.match(text)
    if not m:
        raise ValueError(f"a rule is a weight and a formula, got {text!r}")
    w, e = float(m.group(1)), _Parser(m.group(2)).parse()
    bound = _bound(e, set())
    if dom is not None and bound:
        e = _expand(e, dom)
    lits = []
    _literals(e, lits)
    vars_ = []
    for lit in lits:
        for v in (lit[2] if lit[0] == "atom" else lit[1:]):
            if isinstance(v, str) and v not in vars_ and (dom is None and v not in bound or dom is not None):
                vars_.append(v)
    if dom is None and bound:
        return w, lits, vars_, None
    if len(lits) > 16:
        raise ValueError(f"{text!r} has {len(lits)} distinct literals (after expanding exists); "
                         f"at most 16 -- a Block can say this without grounding it")
    table = np.array([_eval(e, {lit: bool(b >> j & 1) for j, lit in enumerate(lits)})
                      for b in range(2 ** len(lits))], float)
    return w, lits, vars_, table


def _tuples(sizes, types, distinct):
    """All assignments of values to variables with these domain sizes; with
    distinct, variables of the same type take different values."""
    r = len(sizes)
    if r == 0:
        return np.zeros((1, 0), np.int32)
    tup = np.stack(np.meshgrid(*[np.arange(s, dtype=np.int32) for s in sizes], indexing="ij"), -1).reshape(-1, r)
    if distinct:
        keep = np.ones(len(tup), bool)
        for i in range(r):
            for j in range(i + 1, r):
                if types[i] == types[j]:
                    keep &= tup[:, i] != tup[:, j]
        tup = tup[keep]
    return tup


# -- connectivity -------------------------------------------------------------

def _reach(adj, s, t):
    """Is t reachable from s along the boolean adjacency matrix adj?"""
    seen = np.zeros(len(adj), bool)
    seen[s] = True
    front = seen.copy()
    while front.any() and not seen[t]:
        front = adj[front].any(0) & ~seen
        seen |= front
    return bool(seen[t])


def _connected(adj, strong):
    if not strong:
        adj = adj | adj.T
    k = len(adj)
    return all(_reach(adj, 0, t) and (not strong or _reach(adj.T, 0, t)) for t in range(1, k))


def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


# -- blocks: atoms sampled together over their allowed states -----------------

# -- structures: predicates sampled by their own moves -------------------------

class Structure:
    """Something the sampler treats as a unit: it owns some predicates and
    samples their atoms by its own moves, so constraints it builds in never
    need grounding. Block (Categorical, Boltzmann, Path, ...) and Spatial are
    the built-in kinds. A new kind says:

        predicates()    [(name, argument types)], types from "n" (node), "l" (latent)
        rules()         its constraints as rule strings, for describe(); not grounded
        mirrors()       [(mirror, base)]: mirror(y,x) is the same fact as base(x,y)
        relations()     relations it would like shown first, in order
        functional()    predicates with at most one successor per node
        runtime(ctx)    an object that samples it: owned() (atoms it moves),
                        init(i) (a start for replica i), valid(i), start() (once
                        every replica has started), sweep(beta) (one sweep),
                        swap(i, j) (replicas i and j exchanged states),
                        active(), report(); ctx holds the model's state, scores
                        and helpers (see _run)
    Structures of the same kind may share one runtime (blocks are batched
    together): runtime_group() names the group, and the group's first member's
    runtime(ctx, members) makes it.
    """

    def predicates(self):
        raise NotImplementedError

    def rules(self):
        return []

    def mirrors(self):
        return []

    def relations(self):
        return []

    def functional(self):
        return set()

    def bind_latent(self, H):
        """Called with the latent graph (or None) before sampling."""

    def bind_domain(self, sig, dom, distinct):
        """Called with the predicates' argument types and domain sizes."""

    def runtime_group(self):
        return id(self)

    def runtime(self, ctx, members):
        raise NotImplementedError


class _BlockRuntime:
    """Samples every Block, declared and detected: each block updated by
    choosing among its states given everything else, blocks of one shape that
    share no grounding batched into one vectorised step; Blocks with too many
    states are updated an atom at a time."""

    def __init__(self, ctx, declared):
        self.c, c = ctx, ctx
        self.blocks, self.atomwise, self.declared = [], {}, list(declared)
        self.own = np.zeros(c.N + 2, bool)
        masks = {id(P): self.where_mask(P) for P in declared if getattr(P, "where", None)}
        sized, state_sets = {}, {}
        for P0 in declared:
            for x in (range(c.k) if P0.cross is None else range(c.dom[P0.args[P0.per]])):
                atoms = self.atoms_of(P0, x, masks)
                if not len(atoms):
                    continue                     # a conditional block with no entries
                P = P0
                if getattr(P0, "where", None):   # conditional blocks differ in size
                    key = (id(P0), len(atoms))
                    if key not in sized:
                        import copy
                        Q = copy.copy(P0)
                        Q.m = None
                        Q.bind(len(atoms), P0.same)
                        sized[key] = Q
                    P = sized[key]
                self.own[atoms] = True
                if P.n_states() > c.block_limit:
                    if not getattr(P, "atomwise", True):
                        raise ValueError(f"{P!r} can't be updated an atom at a time; raise block_limit")
                    for j, a_ in enumerate(atoms):
                        self.atomwise[int(a_)] = (P, atoms, j)
                    continue
                if id(P) not in state_sets:      # states and weights, once per bound Block
                    St_ = np.asarray(P.states(), np.int8)
                    lw = P.logweights(St_)
                    state_sets[id(P)] = (St_[np.isfinite(lw)], lw[np.isfinite(lw)])
                St_, lw = state_sets[id(P)]
                if not len(St_):
                    raise ValueError(f"{P0!r} has no allowed states for block {x}")
                self.blocks.append(self.make_block(atoms, St_, lw, "declared"))

    def where_mask(self, P):
        """mask[own, entry]: whether the Block's condition holds."""
        c = self.c
        own_v, (ent_v, _, _) = P._names()
        it, vt = P.args[P.per], P.args[1 - P.per]
        tup = np.stack(np.meshgrid(np.arange(c.dom[it], dtype=np.int32), np.arange(c.dom[vt], dtype=np.int32),
                                   indexing="ij"), -1).reshape(-1, 2)
        return c.condition_holds(P.where, tup, {own_v: 0, ent_v: 1}, repr(P)).reshape(c.dom[it], c.dom[vt])

    def atoms_of(self, P, x, masks):             # block x's atoms under Block P
        c = self.c
        if P.cross is None:
            return np.array([c.offset[p] + x for p in P.preds])
        d1 = c.dims[P.cross][1]
        vals = np.arange(c.dom[P.args[1 - P.per]])
        if P.same:
            vals = vals[vals != x]               # not its own entry
        if id(P) in masks:
            vals = vals[masks[id(P)][x, vals]]   # only entries meeting the condition
        a0, a1 = (x, vals) if P.per == 0 else (vals, x)
        return c.offset[P.cross] + a0 * d1 + a1

    def make_block(self, atoms, pats, lw, kind_):
        from scipy.sparse import csr_matrix
        c = self.c
        atoms = np.asarray(atoms, np.int64)
        b = len(atoms)
        parts = [c.inc_f[np.searchsorted(c.free, a_)] for a_ in atoms]   # (factors, masks) per atom
        all_f = np.concatenate([p[0] for p in parts]) if parts else np.zeros(0, np.int64)
        all_m = np.concatenate([p[1] for p in parts]) if parts else np.zeros(0, np.int64)
        col = np.repeat(np.arange(b), [len(p[0]) for p in parts])
        fs, row = np.unique(all_f, return_inverse=True)
        fs = fs.astype(np.int64)
        masks = csr_matrix((all_m.astype(np.int64), (row, col)), shape=(len(fs), b))   # factor x atom -> bits
        pats = np.asarray(pats, np.int8)
        index = {key: i for i, key in enumerate(_pattern_keys(pats))}   # state -> its row
        sh = c.shown[atoms]
        fixed_cols = [(np.flatnonzero(sh[:, 0] == l)) for l in c.fixed if (sh[:, 0] == l).any()]
        clear = np.bincount(row, weights=all_m, minlength=len(fs)).astype(np.int32)   # distinct bits: sum = OR
        return dict(atoms=atoms, fs=fs.astype(np.int32), G=c.G_all[fs],
                    r=c.rule_of[fs].astype(np.int16 if len(c.W) < 2 ** 15 else np.int32),
                    clear=clear, **_state_corrections(masks, pats),
                    pats=pats, lw=np.asarray(lw, float), index=index, kind=kind_,
                    shown_cols=np.flatnonzero(sh[:, 0] >= 0), sh=sh,
                    fixed_cols=fixed_cols, fixed_cnt=[pats[:, j].sum(1) for j in fixed_cols])

    def add_detected(self, atoms, pats):
        self.blocks.append(self.make_block(atoms, pats, np.zeros(len(pats)), "detected"))

    def finalize(self):
        c = self.c
        self.kinds = [B["kind"] for B in self.blocks]
        self.sizes = [len(B["atoms"]) for B in self.blocks]
        self.batches = _batch_blocks(self.blocks, c.F)
        self.decl = [B for B in self.blocks if B["kind"] == "declared"]
        self.groups = {}                         # atom-wise Blocks: one entry per block of atoms
        for a_, (P, atoms, j) in self.atomwise.items():
            self.groups.setdefault((id(P), int(atoms[0])), (P, atoms))
        pos = np.full(c.N + 2, -1)
        pos[c.free] = np.arange(len(c.free))
        self.seq = np.array(sorted(pos[a_] for a_ in self.atomwise), np.int64)

    def owned(self):
        return self.own

    def active(self):
        return bool(self.batches or len(self.seq))

    def init(self, i):                           # replica i's declared blocks, from their weights
        c = self.c
        for B in self.decl:
            pr_ = np.exp(B["lw"] - B["lw"].max())
            c.state[i, B["atoms"]] = B["pats"][c.rng.choice(len(B["pats"]), p=pr_ / pr_.sum())]
        for P, atoms in self.groups.values():
            c.state[i, atoms] = P.initial(c.rng, 1)[0]

    def valid(self, i):                          # every declared block of replica i in its support
        c = self.c
        for B in self.decl:
            if _pattern_keys(c.state[i, B["atoms"]][None])[0] not in B["index"]:
                return False
        return all(P.allowed(c.state[i, atoms][None])[0] for P, atoms in self.groups.values())

    def start(self):                             # the blocks' log weight, and each one's state
        c = self.c
        for B in self.decl:
            c.Sp += B["lw"][[B["index"][key] for key in _pattern_keys(c.state[:, B["atoms"]])]]
        for P, atoms in self.groups.values():
            c.Sp += P.logweights(c.state[:, atoms])
        for Bt in self.batches:
            if Bt["kind"] == "declared":
                cur = c.state[:, Bt["atoms"]]    # (R, blocks, atoms)
                Bt["cur"] = np.array([[Bt["index"][key] for key in _pattern_keys(cur[:, j])]
                                      for j in range(cur.shape[1])]).T

    def swap(self, i, j):
        for Bt in self.batches:
            if "cur" in Bt:
                Bt["cur"][[i, j]] = Bt["cur"][[j, i]]

    def sweep(self, beta_t):
        c = self.c
        state, S, Sp, Edges, rng, R = c.state, c.S, c.Sp, c.Edges, c.rng, c.R
        T, W, h, free, shown, constraint = c.T, c.W, c.h, c.free, c.shown, c.constraint
        codes, delta, valid, allowed = c.codes, c.delta, c.valid, c.allowed
        for j in self.seq[rng.permutation(len(self.seq))]:   # atoms of large Blocks, one at a time
            a = free[j]
            d = delta(j)                                             # rules only
            pd = 0.0
            if a in self.atomwise:                    # the block's change, per replica
                P, atoms, c = self.atomwise[a]
                pat = state[:, atoms].copy()
                pat[:, c] = 1
                lw1 = np.where(P.allowed(pat), P.logweights(pat), -np.inf)
                pat[:, c] = 0
                lw0 = np.where(P.allowed(pat), P.logweights(pat), -np.inf)
                pd = lw1 - lw0
            new = (rng.random(R) < _sigmoid(beta_t * (d + h[a] + pd))).astype(np.int8)
            ch = new != state[:, a]
            if not ch.any():
                continue
            l, u, v = shown[a]
            if l >= 0 and constraint is not None:
                for i in np.flatnonzero(ch):
                    if not allowed(i, l, (u, v) if new[i] else None,
                                   None if new[i] else (u, v)):
                        ch[i] = False
            S[ch] += np.where(new[ch] == 1, d[ch], -d[ch])
            if a in self.atomwise:
                Sp[ch] += np.where(new[ch] == 1, pd[ch], -pd[ch])
            state[ch, a] = new[ch]
            if l >= 0:
                Edges[ch, l, u, v] = new[ch]
        for bt in rng.permutation(len(self.batches)):  # a batch at a time
            Bt = self.batches[bt]
            A_, P_ = Bt["atoms"], Bt["pats"]                           # (blocks, atoms), (states, atoms)
            nB, S_ = len(A_), len(P_)
            r_ = Bt["r"]
            w_ = W[r_]
            starts = Bt["starts"]                                      # each block's first factor
            full = codes(None, Gf=Bt["G"])                             # (R, factors)
            sc_cur = np.add.reduceat(T[r_, full] * w_, starts, axis=1) # (R, blocks): current
            base = full & ~Bt["clear"]                                 # every block atom off
            T0 = T[r_, base] * w_
            f_ = Bt["pm_f"]                                            # each state: the off score,
            corr = T[r_[f_], base[:, f_] | Bt["pm_bits"]] * w_[f_] - T0[:, f_]   # plus changes where
            key = (Bt["pm_blk"] * S_ + Bt["pm_s"])[None] + (nB * S_) * np.arange(R)[:, None]   # its atoms are on
            sc = np.add.reduceat(T0, starts, axis=1)[:, :, None] + np.bincount(
                key.ravel(), weights=corr.ravel(), minlength=R * nB * S_).reshape(R, nB, S_)
            logit = beta_t[:, None, None] * (sc + Bt["lw"][None, None] + (h[A_] @ P_.T)[None])
            for c, cnt in zip(Bt["fixed_cols"], Bt["fixed_cnt"]):      # keep edge counts
                logit = np.where(cnt[None, None] == state[:, A_[:, c]].sum(-1)[:, :, None], logit, -np.inf)
            live = np.isfinite(logit).any(2)
            if not live.any():
                continue
            logit = np.where(live[:, :, None], logit, 0.0)
            logit -= logit.max(2, keepdims=True)
            cum = np.cumsum(np.exp(logit), 2)
            new = np.minimum((cum < rng.random((R, nB, 1)) * cum[:, :, -1:]).sum(2), S_ - 1)
            cur = state[:, A_]                                         # (R, blocks, atoms)
            nxt = P_[new]
            ch = live & (nxt != cur).any(2)
            sc_ = Bt["shown_cols"]
            if constraint is not None and len(sc_):                   # checked in turn
                for i, j in zip(*np.nonzero(ch & (nxt[:, :, sc_] != cur[:, :, sc_]).any(2))):
                    E = Edges[i].copy()
                    for c in sc_:
                        E[Bt["L"][j, c], Bt["U"][j, c], Bt["V"][j, c]] = nxt[i, j, c]
                    removes = (cur[i, j, sc_] > nxt[i, j, sc_]).any()
                    if (callable(constraint) or removes) and not valid(E):
                        ch[i, j] = False         # keep the current state
                    else:
                        Edges[i] = E
            if not ch.any():
                continue
            pick = np.take_along_axis(sc, new[:, :, None], 2)[:, :, 0]
            S += np.where(ch, pick - sc_cur, 0.0).sum(1)
            if Bt["kind"] == "declared":
                Sp += np.where(ch, Bt["lw"][new] - Bt["lw"][Bt["cur"]], 0.0).sum(1)
                Bt["cur"] = np.where(ch, new, Bt["cur"])
            state[:, A_] = np.where(ch[:, :, None], nxt, cur)
            for c in sc_:
                Edges[:, Bt["L"][:, c], Bt["U"][:, c], Bt["V"][:, c]] = state[:, A_[:, c]]

    def report(self):
        sizes = {}
        for kd, b in zip(self.kinds, self.sizes):
            if kd == "detected":
                sizes[b] = sizes.get(b, 0) + 1
        return {"blocks": {"declared": self.kinds.count("declared"),
                           "detected, by size": dict(sorted(sizes.items()))}}


class _SpatialRuntime:
    """Samples one Spatial: moves that keep its rules by construction."""

    def __init__(self, ctx, P):
        self.c, self.P = ctx, P
        c = ctx
        self.own = np.zeros(c.N + 2, bool)
        for a in P.base:
            self.own[c.offset[a]:c.offset[a] + c.k * c.k] = True
        self.count = [0, 0]
        self.rel_index = {p: l for l, p in enumerate(c.relations)}
        self.mirror = {a: b for a, b in P.pairs}
        self.single = set(P.singles)

    def owned(self):
        return self.own

    def active(self):
        return True

    def mats(self, i):
        c = self.c
        return [c.state[i, c.offset[a]:c.offset[a] + c.k * c.k].reshape(c.k, c.k).astype(int) for a in self.P.base]

    def holds(self, i):                          # how many relations hold from x to y
        C = 0
        for a, A_ in zip(self.P.base, self.mats(i)):
            C = C + (A_ if a in self.single else A_ + A_.T)   # a pair's mirror holds the other way
        return C

    def valid(self, i, total=None):              # replica i obeys the rules
        total = self.P.total if total is None else total
        for a, A_ in zip(self.P.base, self.mats(i)):
            pair = a not in self.single
            if A_.trace() or (A_.sum(1) > 1).any() or (pair and (A_.sum(0) > 1).any()):
                return False
            if total and ((A_.sum(1) != 1).any() or (pair and (A_.sum(0) != 1).any())):
                return False
        return bool((self.holds(i) <= 1).all())

    def init(self, i):
        c, P = self.c, self.P
        for a in P.base:
            c.state[i, c.offset[a]:c.offset[a] + c.k * c.k] = 0
        if not P.total:
            return                               # start with no edges
        for a in P.base:                         # a random permutation for each pair, obeying the rest
            if a in self.single:
                continue
            for _ in range(5000):
                perm = c.rng.permutation(c.k)
                c.state[i, c.offset[a] + np.arange(c.k) * c.k + perm] = 1
                if self.valid(i, total=False):
                    break
                c.state[i, c.offset[a] + np.arange(c.k) * c.k + perm] = 0
            else:
                raise ValueError(f"{P!r}: found no start where every node has each direction "
                                 f"(too few nodes?)")
        for a in P.singles:                      # then each single relation: any free target
            C = self.holds(i)
            for x in c.rng.permutation(c.k):
                ys = [y for y in range(c.k) if y != x and C[x, y] == 0]
                if not ys:
                    raise ValueError(f"{P!r}: found no start where every node has each direction "
                                     f"(too few nodes?)")
                y = int(c.rng.choice(ys))
                c.state[i, c.offset[a] + x * c.k + y] = 1
                C[x, y] += 1

    def start(self):                             # each node's target and predecessor, per relation
        c = self.c
        self.tgt, self.pred = {}, {}
        for a in self.P.base:
            adj = c.state[:, c.offset[a]:c.offset[a] + c.k * c.k].reshape(c.R, c.k, c.k)
            self.tgt[a] = np.where(adj.any(2), adj.argmax(2), -1)
            if a not in self.single:             # a pair's relation is one-to-one
                self.pred[a] = np.where(adj.any(1), adj.argmax(1), -1)

    def swap(self, i, j):
        for arr in list(self.tgt.values()) + list(self.pred.values()):
            arr[[i, j]] = arr[[j, i]]

    def plan_ok(self, a, i, rem, add):
        """Does replica i still obey the rules after removing `rem` from
        relation a and adding `add`?"""
        t = self.tgt[a][i].copy()
        for u, v in rem:
            t[u] = -1
        for u, v in add:
            if u == v or t[u] >= 0:
                return False
            t[u] = v
        pair = a not in self.single
        for u, v in add:
            if pair and (t[v] == u or (t == v).sum() > 1):   # a 2-cycle, or v with two predecessors
                return False
            for b in self.P.base:                # another relation from u to v (or, with a
                if b == a:                       # mirror involved, from v to u)?
                    continue
                if self.tgt[b][i, u] == v:
                    return False
                if (pair or b not in self.single) and self.tgt[b][i, v] == u:
                    return False
        return True

    def edges(self, E, a, rem, add):             # update one replica's shown edges
        la, lb = self.rel_index.get(a), self.rel_index.get(self.mirror.get(a))
        for (u, v), val in [(e, 0) for e in rem] + [(e, 1) for e in add]:
            if la is not None:
                E[la, u, v] = val
            if lb is not None:
                E[lb, v, u] = val

    def sweep(self, beta_t):
        c, P = self.c, self.P
        state, S, Edges, rng, R, k = c.state, c.S, c.Edges, c.rng, c.R, c.k
        offset, h, delta, pos_of, constraint, valid = c.offset, c.h, c.delta, c.pos_of, c.constraint, c.valid
        for _ in range(k * len(P.base)):     # each replica proposes its own move
            ai = rng.integers(len(P.base), size=R)
            xs = rng.integers(k, size=R)
            cs = rng.integers(k, size=R)
            cs = np.where(cs == xs, -1, cs)  # "none" takes x's own slot: k equal choices
            rot = (rng.random(R) < 0.5) & (k >= 3)
            trio = np.argsort(rng.random((R, k)), axis=1)[:, :3] if k >= 3 else None
            plans = {}
            for i in range(R):
                a, x, c = P.base[int(ai[i])], int(xs[i]), int(cs[i])
                ti = int(self.tgt[a][i, x])
                single = a in self.single
                if rot[i] and not single:    # rotate three nodes' targets: x <- y's, y <- z's, z <- x's
                    x, y, z = (int(v) for v in trio[i])
                    t3 = [int(self.tgt[a][i, v]) for v in (x, y, z)]
                    rem = tuple((u, t) for u, t in zip((x, y, z), t3) if t >= 0)
                    add = tuple((u, t) for u, t in zip((x, y, z), t3[1:] + t3[:1]) if t >= 0)
                    if set(rem) == set(add):
                        continue
                elif c < 0:
                    if ti < 0 or P.total:
                        continue
                    rem, add = ((x, ti),), ()
                elif ti == c:
                    continue
                else:
                    pi = -1 if single else int(self.pred[a][i, c])   # a single relation: just point at c
                    if pi < 0:
                        if P.total and not single:
                            continue
                        rem, add = ((x, ti),) if ti >= 0 else (), ((x, c),)
                    else:
                        if ti == pi:         # p would point to itself
                            continue
                        rem = (((x, ti),) if ti >= 0 else ()) + ((pi, c),)
                        add = ((x, c),) + (((pi, ti),) if ti >= 0 else ())
                if self.plan_ok(a, i, rem, add):
                    plans.setdefault((a, rem, add), []).append(i)
            for (a, rem, add), rows in plans.items():
                rows = np.array(rows)
                seq_ = [(offset[a] + u * k + v, 0) for u, v in rem] + [(offset[a] + u * k + v, 1) for u, v in add]
                cols_ = [at for at, _ in seq_]
                old = state[np.ix_(rows, cols_)].copy()
                g_rules, g_field = np.zeros(len(rows)), 0.0
                for at, val in seq_:                         # flip in turn, scoring each given the last
                    d = delta(pos_of[at], rows)
                    g_rules += d if val else -d
                    g_field += h[at] if val else -h[at]
                    state[rows, at] = val
                ok = np.log(rng.random(len(rows))) < beta_t[rows] * (g_rules + g_field)
                if constraint is not None and rem:
                    for q in np.flatnonzero(ok):
                        E = Edges[rows[q]].copy()
                        self.edges(E, a, rem, add)
                        if not valid(E):
                            ok[q] = False
                back = rows[~ok]
                if len(back):
                    state[np.ix_(back, cols_)] = old[~ok]
                acc = rows[ok]
                self.count[0] += len(rows)
                self.count[1] += len(acc)
                if not len(acc):
                    continue
                S[acc] += g_rules[ok]
                for u, v in rem:
                    self.tgt[a][acc, u] = -1
                    if a in self.pred:
                        self.pred[a][acc, v] = -1
                for u, v in add:
                    self.tgt[a][acc, u] = v
                    if a in self.pred:
                        self.pred[a][acc, v] = u
                for i in acc:
                    self.edges(Edges[i], a, rem, add)

    def report(self):
        return {"spatial acceptance": round(self.count[1] / max(self.count[0], 1), 3)}



class Block(Structure):
    """Atoms that are sampled together, a block at a time, choosing among
    their allowed states: a declared alternative to grounding constraints as
    rules. The same distribution applies to every block, independently. It
    covers either

      - unary predicates `preds` (a list): node x's pattern is A(x), B(x), ...
      - one two-argument predicate (a string), with argument types `args`
        ("nl": node, latent -- the default; "nn": between nodes; "ll": between
        latent nodes) and one block per value of argument `per` (0 or 1) over
        the other argument. E.g. "In" (args "nl"): node x's pattern is
        In(x, 0), ..., In(x, m-1); "R" with args "nn": node x's out-edges
        R(x, y), y != x (per=1: its in-edges). The number of entries is known
        once mln binds it. Between nodes of one type a node is not its own
        entry, so weights there must be single numbers.

    where       a condition choosing each block's entries, written with the
                block's variable and the entry's (x and y, or G and H for latent
                arguments), e.g. "exists G: In(x,G) & In(y,G)": node x's block
                is then only its same-group entries, and the rest of its row is
                left to the rules. The condition must be fixed by the time the
                block is sampled -- evidence, or an earlier stage -- and since
                blocks then differ in size, weights must be single numbers.

        P(pattern) ~ exp(sum of the weights of the single-node rules it satisfies)

    restricted to the patterns in states(). The sampler updates a node's
    pattern by choosing among states() directly, so its cost grows with the
    number of states rather than 2^m, and a restriction built into states()
    (exactly one group, at most two, ...) costs nothing to respect.

    rules       single-node rules over a list of unary preds, all in one
                variable, e.g. "-1.0 A(x)", "2.0 A(x) & B(x)", "inf A(x) | B(x)".
    states      the allowed patterns, a (number of states, m) 0/1 array.
                Default: all 2^m patterns that satisfy the hard rules (m <= 16).
                States are authoritative: subclasses may restrict them further
                than their rules() say.

    Subclass it to build in structure: override states(), and logweights() if
    the weights are easier to compute in code; anything that depends on m goes
    in _build(), called once m is known. If the block has more states than
    mln's block_limit, its atoms are instead updated one at a time, which uses
    logweights() and allowed() on arbitrary patterns and initial() for a
    starting point, never states(); override n_states(), allowed() and
    initial() to make that possible without enumerating. One-at-a-time updates
    can't move between states that differ in two atoms with nothing allowed in
    between (Categorical is the extreme case), so a block like that must stay
    within the block limit.
    """

    def __init__(self, preds, rules=(), states=None, args="nl", per=0, where=None):
        self.where = where
        if isinstance(preds, str):
            self.cross, self.preds, self.m = preds, None, None
            if len(args) != 2 or set(args) - {"n", "l"} or per not in (0, 1):
                raise ValueError("args is two of 'n' (node) / 'l' (latent), e.g. 'nl'; per is 0 or 1")
            self.args, self.per, self.same = tuple(args), per, False
        else:
            self.cross, self.preds, self.m = None, list(preds), len(preds)
        self._rules = list(rules)
        self._states = None if states is None else np.asarray(states, np.int8)
        if self.m is not None:
            self._build()

    def predicates(self):
        if self.cross is not None:
            return [(self.cross, tuple(self.args))]
        return [(p, ("n",)) for p in self.preds]

    def functional(self):
        one = (isinstance(self, Categorical) or getattr(self, "hi_in", None) == 1)
        if self.cross is not None and self.args[0] == self.args[1] and self.per == 0 and not self.where and one:
            return {self.cross}
        return set()

    def runtime_group(self):
        return "blocks"                          # all Blocks share one batched runtime

    def runtime(self, ctx, members):
        return _BlockRuntime(ctx, members)

    def bind_domain(self, sig, dom, distinct):
        if self.cross is not None:               # a two-argument predicate, e.g. In(x,G)
            vt = self.args[1 - self.per]
            same = self.args[0] == self.args[1] and distinct
            if dom[vt] - same < 1:
                raise ValueError(f"{self!r} has no entries: no {'latent ' if vt == 'l' else ''}nodes")
            self.bind(dom[vt] - same, same)
        else:
            self.bind(len(self.preds))

    def bind(self, m, same=False):
        """Fix the number of entries per block (for a two-argument predicate):
        the size of the varying argument's domain, less one when both arguments
        are the same type (`same`: a node is not its own entry)."""
        if self.cross is not None and (self.m, self.same) != (m, same):
            self.m, self.same = m, same
            self.preds = [f"{self.cross}[{g}]" for g in range(m)]
            self._build()

    def _names(self):
        """Variable names for rule forms: the block's own variable, then three
        for the varying argument."""
        pool = {"n": ["x", "y", "z", "w"], "l": ["G", "H", "K", "L"]}
        it, vt = self.args[self.per], self.args[1 - self.per]
        own = pool[it][0]
        vary = [v for v in pool[vt] if v != own][:3]
        return own, vary

    def _atom(self, own, v):
        """The predicate with the block's variable and a varying one in place,
        and the where condition on that entry."""
        a, b = (own, v) if self.per == 0 else (v, own)
        return f"{self.cross}({a},{b})" + self._cond(v)

    def _uniform_only(self, *params):
        if (self.same or self.where) and any(p is not None and np.ndim(p) > 0 for p in params):
            raise ValueError(f"{self!r}: its blocks' entries differ from node to node (a relation "
                             f"within one type, or where=), so give weights as single numbers")

    def _cond(self, v):
        """The where condition for entry variable v, as a conjunct ("" if none)."""
        if not self.where:
            return ""
        _, (ent, _, _) = self._names()
        cond = re.sub(r"\b" + ent + r"\b", v, self.where)
        return " & (" + cond + ")"

    def _build(self):
        pass

    def bind_latent(self, H):
        """Called with the latent graph (or None) before sampling; a block whose
        states depend on it overrides this."""

    def rules(self):
        return list(self._rules)

    def states(self):
        if self._states is not None:
            return self._states
        m = self.m
        if m > 16:
            raise ValueError(f"{m} entries is too many to enumerate; give states")
        return ((np.arange(2 ** m)[:, None] >> np.arange(m)) & 1).astype(np.int8)

    def n_states(self):
        return len(self.states())

    def allowed(self, patterns):
        """Which of these patterns (rows) are states."""
        S = np.asarray(self.states(), np.int64)
        codes = set(((S << np.arange(S.shape[1])).sum(1)).tolist())
        pc = (np.asarray(patterns, np.int64) << np.arange(S.shape[1])).sum(1)
        return np.array([c in codes for c in pc.tolist()])

    def initial(self, rng, size):
        """`size` patterns to start from, drawn from the block's weights."""
        S = np.asarray(self.states(), np.int8)
        lw = self.logweights(S)
        p = np.exp(lw - lw[np.isfinite(lw)].max())
        return S[rng.choice(len(S), size, p=p / p.sum())]

    def logweights(self, states):
        """Log weight of each state from rules(); -inf where a hard rule fails."""
        out = np.zeros(len(states))
        for text in self.rules():
            if self.cross is not None:
                raise ValueError("a block over a two-argument predicate needs logweights() in code")
            w, lits, vars_, table = _parse_rule(text)
            if len(vars_) > 1 or any(l[0] == "atom" and (l[1] not in self.preds or len(l[2]) != 1)
                                     for l in lits):
                raise ValueError(f"{text!r} is not a single-node rule over {self.preds}")
            idx = np.zeros(len(states), np.int64)
            for j, l in enumerate(lits):
                bit = states[:, self.preds.index(l[1])] if l[0] == "atom" else 1   # x = x
                idx |= np.asarray(bit, np.int64) << j
            val = table[idx]
            out += np.where(val > 0, 0.0, -np.inf) if np.isinf(w) and w > 0 else \
                   np.where(val > 0, -np.inf, 0.0) if np.isinf(w) else w * val
        return out

    def __repr__(self):
        cond = f", where {self.where!r}" if self.where else ""
        return f"{type(self).__name__}({self.cross if self.cross is not None else self.preds!r}{cond})"


class Categorical(Block):
    """Each node has exactly one entry: one of the unary preds, or one latent
    node for a (node, latent) predicate -- the SBM's memberships. P(entry a) ~
    exp(logits[a]); logits default to uniform. m states."""

    def __init__(self, preds, logits=None, args="nl", per=0, where=None):
        self.logits_in = None if logits is None else np.asarray(logits, float)
        super().__init__(preds, args=args, per=per, where=where)

    def _build(self):
        m = self.m
        if self.cross is not None:
            self._uniform_only(self.logits_in)
        self.logits = np.zeros(m) if self.logits_in is None else self.logits_in
        if self.logits.shape != (m,):
            raise ValueError(f"{self!r} has {m} entries but {len(self.logits)} logits")
        if self.cross is None:                   # describe it as rules
            p = self.preds
            self._rules = [f"{float(w)!r} {a}(x)" for a, w in zip(p, self.logits) if w]
            self._rules.append("inf " + " | ".join(f"{a}(x)" for a in p))
            self._rules += [f"inf ~({a}(x) & {b}(x))" for i, a in enumerate(p) for b in p[i + 1:]]
        self._states = np.eye(m, dtype=np.int8)

    def logweights(self, states):                # direct: the rules only describe it
        return np.asarray(states, float) @ self.logits

    def rules(self):
        if self.cross is None:
            return list(self._rules)
        own, (u, v, _) = self._names()
        out = [f"inf exists {u}: {self._atom(own, u)}",
               f"inf ~({self._atom(own, u)} & {self._atom(own, v)})"]
        if self.logits_in is not None:
            out += [f"{float(l)!r} {self._atom(own, g)}" for g, l in enumerate(self.logits_in) if l]
        return out

    def allowed(self, patterns):
        return np.asarray(patterns).sum(1) == 1

    atomwise = False                             # single flips can't move it


class Boltzmann(Block):
    """A pairwise Boltzmann distribution over a node's entries,

        P(s) ~ exp(sum_a h[a] s_a + sum_{a<b} J[a, b] s_a s_b),

    optionally limited to min_active..max_active active entries per node,
    which keeps the number of states to sum_j C(m, j). h is a number (the same
    for every entry) or a list; J is a number (the same coupling for every
    pair), an m x m matrix (upper triangle used) or, for unary preds,
    {(a, b): weight}. A single negative J is soft mutual exclusivity with one
    weight: each extra entry a node has costs |J| per entry it already holds.
    Boltzmann(..., J=-w, min_active=1) is a soft Categorical, which it
    approaches as w grows."""

    def __init__(self, preds, h=None, J=None, max_active=None, min_active=0, args="nl", per=0,
                 where=None):
        self.h_in, self.J_in, self.hi_in, self.lo = h, J, max_active, min_active
        super().__init__(preds, args=args, per=per, where=where)

    def _build(self):
        m, h, J = self.m, self.h_in, self.J_in
        if self.cross is not None:
            self._uniform_only(h, None if isinstance(J, dict) else J)
        self.h = np.zeros(m) if h is None else np.broadcast_to(np.asarray(h, float), (m,)).copy()
        self.Jm = np.zeros((m, m))
        if J is None:
            pass
        elif np.ndim(J) == 0:
            self.Jm = np.triu(np.full((m, m), float(J)), 1)
        elif isinstance(J, dict):
            if self.cross is not None:
                raise ValueError("for a (node, latent) predicate give J as a number or matrix")
            for (a, b), w in J.items():
                self.Jm[self.preds.index(a), self.preds.index(b)] += w
        else:
            self.Jm = np.triu(np.asarray(J, float), 1)
        self.hi = m if self.hi_in is None else self.hi_in
        if self.cross is None:                   # describe it as rules
            p = self.preds
            self._rules = [f"{float(w)!r} {a}(x)" for a, w in zip(p, self.h) if w]
            self._rules += [f"{float(self.Jm[i, j])!r} {p[i]}(x) & {p[j]}(x)"
                            for i in range(m) for j in range(i + 1, m) if self.Jm[i, j]]

    def logweights(self, states):                # direct: the rules only describe it
        s = np.asarray(states, float)
        return s @ self.h + np.einsum("si,ij,sj->s", s, self.Jm, s)

    def rules(self):
        if self.cross is None:
            return list(self._rules)
        h, J, out = self.h_in, self.J_in, []
        own, (u, v, _) = self._names()
        at = lambda w: self._atom(own, w)
        if h is not None and np.ndim(h) == 0:
            out += [f"{float(h)!r} {at(u)}"] if h else []
        elif h is not None:
            out += [f"{float(c)!r} {at(g)}" for g, c in enumerate(h) if c]
        if J is not None and np.ndim(J) == 0:
            out += [f"{float(J)!r} {at(u)} & {at(v)} & {u} < {v}"] if J else []
        elif J is not None:
            out += [f"{float(J[i][j])!r} {at(i)} & {at(j)}"
                    for i in range(len(J)) for j in range(i + 1, len(J)) if J[i][j]]

        def several(r):                          # r entries, in increasing order
            vs = [f"{u}{i}" for i in range(r)]
            body = " & ".join(at(w) for w in vs)
            return vs, body + "".join(f" & {a} < {b}" for a, b in zip(vs, vs[1:]))

        if self.hi_in is not None:
            out.append(f"inf ~({several(self.hi_in + 1)[1]})")
        if self.lo >= 1:
            vs, body = several(self.lo)
            out.append(f"inf exists {', '.join(vs)}: {body}")
        return out

    def n_states(self):
        from math import comb
        return sum(comb(self.m, r) for r in range(self.lo, self.hi + 1))

    def states(self):
        if self.n_states() > 10 ** 6:
            raise ValueError(f"{self!r} has {self.n_states()} states, too many to list")
        m, out = self.m, []
        for r in range(self.lo, self.hi + 1):
            for on in itertools.combinations(range(m), r):
                s = np.zeros(m, np.int8)
                s[list(on)] = 1
                out.append(s)
        return np.array(out, np.int8).reshape(-1, m)

    def allowed(self, patterns):
        c = np.asarray(patterns).sum(1)
        return (c >= self.lo) & (c <= self.hi)

    def initial(self, rng, size):
        if self.n_states() <= 10 ** 5:
            return super().initial(rng, size)
        out = np.zeros((size, self.m), np.int8)  # lo random entries each
        for i in range(size):
            out[i, rng.choice(self.m, self.lo, replace=False)] = 1
        return out


class Path(Block):
    """Nested membership along a latent forest: each node's entries of the
    (node, latent) predicate `member` form one path from a root of the forest
    `rel` (rel(child, parent)) down to some latent node, its deepest group. So
    the node is in that group and all of its ancestors. P(deepest group g) ~
    exp(logits[g]); with allow_empty, a node may also be in no group (log
    weight empty_logit). On a forest with no edges this is Categorical.

    The forest must be known before memberships are sampled: from a latent
    graph passed to mln, or sampled in an earlier stage (the default order).
    """

    def __init__(self, member, rel="Parent", logits=None, allow_empty=False, empty_logit=0.0):
        self.rel, self.logits_in = rel, None if logits is None else np.asarray(logits, float)
        self.allow_empty, self.empty_logit, self.H = allow_empty, empty_logit, None
        super().__init__(member)

    def bind_latent(self, H):
        if H is None or not hasattr(H, "graph") or self.rel not in H.graph.get("relations", {}):
            raise ValueError(f"{self!r} needs a latent graph with relation {self.rel!r}; sample it "
                             f"in an earlier stage (order='hidden_first' or 'latent_first') or pass latent=")
        self.H = H
        self._build()

    def _build(self):
        m = self.m
        self.logits = np.zeros(m) if self.logits_in is None else self.logits_in
        if self.logits.shape != (m,):
            raise ValueError(f"{self!r} has {m} groups but {len(self.logits)} logits")
        if self.H is None:
            self._states = np.eye(m, dtype=np.int8)   # until the forest is known
            return
        l = self.H.graph["relations"][self.rel]
        parent = {}
        for u, v, d in self.H.edges(data=True):
            if d.get("action") == l:
                if u in parent and parent[u] != v:
                    raise ValueError(f"latent node {u} has two parents in {self.rel!r}")
                parent[u] = v
        S = np.zeros((m, m), np.int8)
        for g in range(m):                       # g and its ancestors
            x, seen = g, set()
            while x is not None and x not in seen:
                S[g, x] = 1
                seen.add(x)
                x = parent.get(x)
            if x is not None:
                raise ValueError(f"{self.rel!r} has a cycle through latent node {x}")
        self._end = np.arange(m)
        if self.allow_empty:
            S = np.vstack([np.zeros((1, m), np.int8), S])
        self._states = S

    def rules(self):
        p, r = self.cross, self.rel
        out = [f"inf {p}(x,G) & {r}(G,H) -> {p}(x,H)",                     # closed upward
               f"inf ~({p}(x,G) & {p}(x,H) & {r}(G,P) & {r}(H,P))",       # one child per level
               f"inf ~({p}(x,G) & {p}(x,H) & ~(exists P: {r}(G,P)) & ~(exists Q: {r}(H,Q)))"]  # one root
        if not self.allow_empty:
            out.append(f"inf exists G: {p}(x,G)")
        elif self.empty_logit:
            out.append(f"{float(self.empty_logit)!r} ~(exists G: {p}(x,G))")
        if self.logits_in is not None:                                   # weight on the deepest group
            out += [f"{float(l)!r} {p}(x,{g}) & ~(exists H: {p}(x,H) & {r}(H,{g}))"
                    for g, l in enumerate(self.logits_in) if l]
        return out

    def logweights(self, states):
        s = np.asarray(states)
        out = np.full(len(s), -np.inf)
        S = self._states
        for i, row in enumerate(s):              # match each pattern to its path
            hit = np.flatnonzero((S == row).all(1))
            if len(hit):
                j = hit[0]
                if self.allow_empty:
                    out[i] = self.empty_logit if j == 0 else self.logits[j - 1]
                else:
                    out[i] = self.logits[j]
        return out

    def allowed(self, patterns):
        return np.isfinite(self.logweights(patterns))

    atomwise = False


class Spatial(Structure):
    """Relations as spatial directions, sampled as one structure. Give pairs
    (a, b) of inverse directions -- by default ("N", "S") and ("E", "W") --
    and/or single relations with no inverse, e.g. Spatial("A", "B", "C") or
    Spatial(("N", "S"), "A"). Then:

      functional   every node has at most one edge in each relation
      inverse      for a pair, b(y,x) exactly when a(x,y): b is stored as a's
                   mirror image, so only a is sampled and the inverse always
                   holds; so a pair's relations are one-to-one, and have no
                   2-cycles
      exclusive    at most one relation from x to y, counting mirrors (so no
                   collapsed directions like N = W)
      total        (option) every node has exactly one edge in each relation:
                   a pair's relations are then permutations, a single one a
                   function (several nodes may point to the same node)

    These hold by construction. Each move picks a node x, one of the sampled
    relations and a new target c (or none), and adds, removes or moves x's
    edge there. For a pair, if c already has a predecessor p, it swaps: x
    takes c and p takes x's old target; or it rotates three nodes' targets.
    Moves keep every rule above and are accepted according to the change in
    the other rules' score. With total, nothing is removed.

    Commutation and other geometry are ordinary rules on top, e.g.
    commute("N", "E") (with total, one implies the rest; without, add every
    orientation). Rules may mention all the relations; those about a mirrored
    relation read its partner's atoms transposed. No field, density or
    evidence on mirrored relations: put them on the sampled ones.

        MLN(16, [Spatial(total=True), commute("N", "E", "inf")])
    """

    def __init__(self, *items, total=False):
        self.items, self.pairs, self.singles = [], [], []
        for it in (items or (("N", "S"), ("E", "W"))):
            if isinstance(it, str):
                self.singles.append(it)
                self.items.append(it)
            elif len(it) == 2 and all(isinstance(p, str) for p in it):
                self.pairs.append(tuple(it))
                self.items.append(tuple(it))
            else:
                raise ValueError(f"Spatial takes relation names or pairs of them, not {it!r}")
        self.total = total
        self.preds = [r for it in self.items for r in ((it,) if isinstance(it, str) else it)]
        if len(set(self.preds)) != len(self.preds):
            raise ValueError("each relation can appear only once")

    @property
    def base(self):                              # the sampled relations, in order
        return [it if isinstance(it, str) else it[0] for it in self.items]

    def predicates(self):
        return [(p, ("n", "n")) for p in self.preds]

    def mirrors(self):
        return [(b, a) for a, b in self.pairs]

    def relations(self):
        return list(self.preds)

    def functional(self):
        return set(self.preds)

    def rules(self):
        out = [f"inf {r}(x,y) & {r}(x,z) -> y = z" for r in self.preds]
        if self.total:
            out += [f"inf exists y: {r}(x,y)" for r in self.preds]
        out += [f"inf {a}(x,y) <-> {b}(y,x)" for a, b in self.pairs]
        out += [f"inf ~({a}(x,y) & {b}(x,y))" for i, a in enumerate(self.preds) for b in self.preds[i + 1:]]
        return out

    def runtime(self, ctx, members):
        return _SpatialRuntime(ctx, self)

    def __repr__(self):
        return ("Spatial(" + ", ".join(it if isinstance(it, str) else "/".join(it) for it in self.items) +
                (", total" if self.total else "") + ")")


# -- components: constructors for common patterns ------------------------------

class Component:
    """A bundle of model parts that mln unpacks wherever it accepts rules:

        rules       rule strings scaled by beta (the structure you tune)
        fixed       rule strings not scaled by beta
        blocks      Block objects
        joint_only  fixed rule strings used only with order="joint" (such as
                    normalisers that undo the tilt joint sampling causes)
        field       {predicate: weight}, used unless mln's field sets it
        evidence    {predicate: true tuples}, fixed atoms it brings
        relations   relations it would like shown first, in this order
        nodes       the number of nodes it needs, if fixed
        node_attrs  {attribute: {node: value}} set on every sample's nodes
    """

    def __init__(self, rules=(), fixed=(), blocks=(), joint_only=(), field=None, evidence=None,
                 relations=(), nodes=None, node_attrs=None, name=""):
        self.rules, self.fixed, self.blocks = list(rules), list(fixed), list(blocks)
        self.joint_only, self.field, self.name = list(joint_only), dict(field or {}), name
        self.evidence, self.relations = dict(evidence or {}), list(relations)
        self.nodes, self.node_attrs = nodes, dict(node_attrs or {})

    def __repr__(self):
        return f"Component({self.name})"


def _w(w):
    return "inf" if w == np.inf else ("-inf" if w == -np.inf else repr(float(w)) if not isinstance(w, str) else w)


def _where(where):
    return f" where {where}" if where else ""


def _vars(latent, n=3):
    return ["G", "H", "K", "L"][:n] if latent else ["x", "y", "z", "u"][:n]


def symmetric(rel, w=1.0, latent=False, where=None):
    """rel(x,y) -> rel(y,x), weight w (scaled by beta; "inf" for hard). `where`
    restricts it to groundings meeting a fixed condition on x, y (see MLN)."""
    x, y = _vars(latent, 2)
    return Component(rules=[f"{_w(w)} {rel}({x},{y}) -> {rel}({y},{x}){_where(where)}"],
                     name=f"symmetric {rel}")


def antisymmetric(rel, w=1.0, latent=False, where=None):
    """Never both rel(x,y) and rel(y,x), weight w; `where` as for symmetric."""
    x, y = _vars(latent, 2)
    return Component(rules=[f"{_w(w)} ~({rel}({x},{y}) & {rel}({y},{x})){_where(where)}"],
                     name=f"antisymmetric {rel}")


def transitive(rel, w=1.0, latent=False, where=None):
    """rel(x,y) & rel(y,z) -> rel(x,z), weight w; `where` restricts it to
    groundings meeting a fixed condition on x, y, z, e.g. all in one group."""
    x, y, z = _vars(latent, 3)
    return Component(rules=[f"{_w(w)} {rel}({x},{y}) & {rel}({y},{z}) -> {rel}({x},{z}){_where(where)}"],
                     name=f"transitive {rel}")


def functional(rel, w=np.inf, latent=False, inverse=False, where=None, total=False):
    """At most one rel-successor per node (with inverse, at most one
    predecessor; with total, exactly one). Hard by default, as a Block over
    each node's row (column) of rel, so nothing is grounded for it; with a
    finite w, the soft rule rel(x,y) & rel(x,z) -> y = z, weight w (scaled by
    beta). `where` limits it to entries meeting a condition (see Block), e.g.
    same-group successors."""
    t = "ll" if latent else "nn"
    if total and w != np.inf:
        raise ValueError("total= needs the hard (block) form")
    if w == np.inf:
        B = (Categorical(rel, args=t, per=int(inverse), where=where) if total else
             Boltzmann(rel, max_active=1, args=t, per=int(inverse), where=where))
        return Component(blocks=[B], name=f"functional {rel}" + (" (total)" if total else ""))
    x, y, z = _vars(latent, 3)
    body = f"{rel}({x},{y}) & {rel}({x},{z}) -> {y} = {z}" if not inverse else \
        f"{rel}({y},{x}) & {rel}({z},{x}) -> {y} = {z}"
    cond = None
    if where:                                    # the condition on (x, y), for both entries
        other = re.sub(r"\b" + y + r"\b", z, where)
        cond = "(" + where + ") & (" + other + ")"
    return Component(rules=[f"{_w(w)} {body}{_where(cond)}"], name=f"functional {rel}")


def inverse(a, b, w=1.0, latent=False, where=None):
    """b is a's inverse: a(x,y) <-> b(y,x), weight w (scaled by beta; "inf" for
    hard). k^2 groundings."""
    x, y = _vars(latent, 2)
    return Component(rules=[f"{_w(w)} {a}({x},{y}) <-> {b}({y},{x}){_where(where)}"],
                     name=f"inverse {a} {b}")


def commute(a, b, w=1.0, latent=False, where=None):
    """Going a then b ends where going b then a does:
    a(x,y) & b(y,z) & b(x,u) -> a(u,z), weight w (scaled by beta). Meant for
    functional relations -- each node with exactly one a- and one b-successor,
    e.g. functional(a, total=True) -- where it says exactly that the two
    compositions agree. It has four variables, so about k^4 groundings: fine
    for tens of nodes, heavy beyond (1.3 million at k = 36, 94 million at 100).

    Any commuting pair satisfies it, including a = b, so for grid-like
    structure add e.g. "inf N(x,y) -> ~E(x,y)". Even then the result is a pair
    of commuting directions -- a torus of some shape, possibly twisted -- not
    a particular n x m grid: fixing the size takes rules about cycle lengths,
    or use torus() for a fixed grid."""
    x, y, z, u = _vars(latent, 4)
    C = Component(rules=[f"{_w(w)} {a}({x},{y}) & {b}({y},{z}) & {b}({x},{u}) -> {a}({u},{z}){_where(where)}"],
                  name=f"commute {a} {b}")
    C.needs_total = [a, b]                       # checked by MLN: otherwise this isn't commutation
    return C


def directions(*pairs, latent=False):
    """Relations as directions: each pair (a, b) are inverses (a(x,y) <-> b(y,x)),
    and between two nodes at most one direction holds (no a(x,y) & c(x,y)). Hard
    rules on a single pair of nodes, which automatic detection samples as one
    block per pair of nodes, choosing among its few allowed states (5 for two
    pairs: nothing, or one direction one way and its inverse back). It also
    rules out collapsed directions such as N = W or N = S.

        directions(("N", "S"), ("E", "W"))

    It doesn't say how many edges a node has; for that use rules, not
    functional() blocks: a declared block on N would take N's atoms away from
    the per-pair blocks."""
    rels = [r for p in pairs for r in p]
    x, y = _vars(latent, 2)
    rules = [f"inf {a}({x},{y}) <-> {b}({y},{x})" for a, b in pairs]
    rules += [f"inf ~({a}({x},{y}) & {b}({x},{y}))" for i, a in enumerate(rels) for b in rels[i + 1:]]
    return Component(fixed=rules, relations=rels, name="directions " + " ".join("/".join(p) for p in pairs))


def categorical(member, logits=None):
    """Each node is in exactly one latent node via the (node, latent)
    predicate `member`, e.g. categorical("In"); P(group) ~ exp(logits)."""
    return Component(blocks=[Categorical(member, logits)], name=f"categorical {member}")


def sbm(member, rel, within, between):
    """A stochastic block model on relation rel, for nodes in exactly one
    latent group each (pair it with categorical(member)): an edge x -> y has
    log-odds `within` if x and y share a group, `between` otherwise. With
    order="joint" it also adds the normalisers that keep group membership
    following its own block; staged orders need none.

        mln(30, [categorical("In"), sbm("In", "R", 1.0, -3.0)], latent=3)
    """
    m_, r_ = member, rel
    # base log-odds `between` on every pair, plus (within - between) per shared
    # group: m groundings per pair of nodes instead of m^2
    nb = lambda t: float(np.log1p(np.exp(t)))
    return Component(
        fixed=[f"{_w(between)} {r_}(x,y)",
               f"{_w(within - between)} {m_}(x,G) & {m_}(y,G) & {r_}(x,y)"],
        joint_only=[f"{_w(nb(between) - nb(within))} {m_}(x,G) & {m_}(y,G)"],
        name=f"sbm {member} {rel}")


def forest(rel="Parent", latent=True, attach=None, where=None):
    """rel is a forest: rel(child, parent), at most one parent each (a Block
    over each node's parents), and acyclic because a parent always has a
    smaller index. `attach` is a field on rel (how readily nodes take a
    parent; few roots when large). With `where` (see Block) only the entries
    meeting the condition form the forest, e.g. a forest within each group."""
    x, y, _ = _vars(latent, 3)
    return Component(fixed=[f"inf {rel}({x},{y}) -> {y} < {x}{_where(where)}"],
                     blocks=[Boltzmann(rel, max_active=1, args="ll" if latent else "nn", where=where)],
                     field={rel: attach} if attach is not None else None, name=f"forest {rel}")


def nested(member, rel="Parent"):
    """Membership is closed upward along the latent forest rel: being in a
    group means being in its parent too."""
    return Component(fixed=[f"inf {member}(x,G) & {rel}(G,H) -> {member}(x,H)"],
                     name=f"nested {member} {rel}")


def path(member, rel="Parent", logits=None, allow_empty=False):
    """Each node is in one root-to-group path of the latent forest rel: a
    categorical choice of its deepest group, with all ancestors implied. Use it
    with forest(rel); on a forest with no edges it is categorical(member)."""
    return Component(blocks=[Path(member, rel, logits, allow_empty)], name=f"path {member} {rel}")


def _vtype(v):
    """A lowercase variable ranges over the k nodes ("n"), an uppercase one over
    the latent nodes ("l")."""
    return "l" if v[0].isupper() else "n"


def torus(n, m=None, rels=("N", "S", "E", "W"), wrap=True):
    """An n x m grid as four fixed relations, north, south, east and west, for
    other rules to build on (the model then needs k = n * m nodes). Node
    r * m + c is row r (0 at the top), column c, with a `pos` attribute
    (r, c); the relations are shown first, in the order given, so actions are
    0 = N, 1 = S, 2 = E, 3 = W as in automata.grid. With wrap (the default) the
    edges wrap around; without it, moves off the grid are absent.

    The grid is given as evidence rather than sampled from rules: rules can
    describe a torus locally, but its hard, interlocking constraints would
    freeze a sampler long before it found one, and the distribution they
    define is just this grid with its nodes relabelled.

        MLN(36, [torus(6), "1.0 N(x,y) -> (C(x) <-> C(y))"])  # labels agreeing north-south
    """
    m = n if m is None else m
    moves = dict(zip(rels, [(-1, 0), (1, 0), (0, 1), (0, -1)]))
    ev = {}
    for p, (dr, dc) in moves.items():
        ev[p] = []
        for r in range(n):
            for c in range(m):
                rr, cc = r + dr, c + dc
                if wrap:
                    rr, cc = rr % n, cc % m
                elif not (0 <= rr < n and 0 <= cc < m):
                    continue
                ev[p].append((r * m + c, rr * m + cc))
    return Component(fixed=[f"0 {p}(x,y)" for p in rels], evidence=ev, relations=list(rels),
                     nodes=n * m, node_attrs={"pos": {r * m + c: (r, c) for r in range(n) for c in range(m)}},
                     name=f"torus {n}x{m}" if wrap else f"grid {n}x{m}")


def _signatures(texts, declared=()):
    """Argument types of every predicate in these rules and blocks, checked to
    be the same everywhere."""
    sig, names = {}, {"n": "node", "l": "latent", None: "?"}

    def note(p, s_, where):                      # None: a number, type from elsewhere
        old = sig.setdefault(p, s_)
        if len(old) != len(s_) or any(a and b and a != b for a, b in zip(old, s_)):
            raise ValueError(f"{p} is used with arguments ({', '.join(names[t] for t in old)}) "
                             f"and ({', '.join(names[t] for t in s_)}), in {where!r}")
        sig[p] = tuple(a or b for a, b in zip(old, s_))

    for t in texts:
        cond = _split_where(t)[1]
        lits = _parse_rule(t)[1] + (_parse_rule("0 " + cond)[1] if cond else [])
        for lit in lits:
            if lit[0] == "atom":
                note(lit[1], tuple(None if isinstance(v, int) else _vtype(v) for v in lit[2]), t)
            elif all(isinstance(v, str) for v in lit[1:]) and _vtype(lit[1]) != _vtype(lit[2]):
                raise ValueError(f"{lit[1]} and {lit[2]} are of different types, in {t!r}")
    for P in declared:
        for p, types in P.predicates():
            note(p, tuple(types), f"structure {P!r}")
        for t in P.rules():                      # e.g. Path's rules mention its forest
            for lit in _parse_rule(t)[1]:
                if lit[0] == "atom":
                    note(lit[1], tuple(None if isinstance(v, int) else _vtype(v) for v in lit[2]),
                         f"block {P!r}")
    for p, s_ in sig.items():
        if None in s_:
            raise ValueError(f"{p} is only used with numbers in some argument; use it with a "
                             f"variable there somewhere, so its type is known")
    return sig


def _latent_graph(m, G0, sig):
    """A graph over latent nodes 0..m-1 from the latent predicates sampled in G0."""
    H = nx.MultiDiGraph()
    H.add_nodes_from(range(m))
    hidden = G0.graph.get("hidden", {})
    rel = sorted(p for p, s in sig.items() if s == ("l", "l") and p in hidden)
    for l, p in enumerate(rel):
        for u, v in np.argwhere(hidden[p]):
            H.add_edge(int(u), int(v), action=l)
    H.graph["relations"] = {p: l for l, p in enumerate(rel)}
    for p, s in sig.items():
        if s == ("l",) and p in hidden:
            for g in range(m):
                H.nodes[g][p] = bool(hidden[p][g])
    H.graph["hidden"] = {p: hidden[p] for p, s in sig.items()
                         if p in hidden and all(t == "l" for t in s) and len(s) > 2}
    return H


STAGES = {-1: "fixed", 0: "latent", 1: "hidden", 2: "relations"}


def _stage_plan(sig, relations, evidence, have_latent_graph, order):
    """stage_of(predicate): -1 fixed (evidence), 0 latent graph, 1 hidden,
    2 relations; under "joint" everything sampled is one stage (2)."""
    def stage_of(p):
        if p in evidence or (have_latent_graph and all(t == "l" for t in sig[p])):
            return -1
        if order == "joint":
            return 2
        if all(t == "l" for t in sig[p]):
            return 0
        if p in relations:
            return 2
        return 1 if order == "hidden_first" else 2
    return stage_of


def _rule_preds(text):
    return [lit[1] for lit in _parse_rule(text)[1] if lit[0] == "atom"]


def _staged(model, rules, fixed, sampler):
    """Sample model in stages ("latent_first" / "hidden_first"): each stage is
    run given the ones before, and the last is returned. None when the model
    has only one stage, so it can be run directly."""
    declared = [p for p in fixed if isinstance(p, Structure)]
    fixed_rules = [p for p in fixed if not isinstance(p, Structure)]
    sig, relations, stage_of = model.sig, model.relations, model._stage_of

    def rule_stage(t):
        return max((stage_of(p) for p in _rule_preds(t)), default=-1)

    def block_stage(P):
        return max(stage_of(p) for p, _ in P.predicates())

    def per_pred(d, s_):                         # field / density entries for stage s_
        if d is None:
            return None
        if not isinstance(d, dict):
            return d if s_ == 2 else None        # a number means "every relation"
        return {p: v for p, v in d.items() if p in sig and stage_of(p) == s_} or None

    targeted = set()                             # predicates with their own field or density
    for d in (model.field, model.density):
        if isinstance(d, dict):
            targeted |= {p for p in d if p in sig}
    present = ({rule_stage(t) for t in rules + fixed_rules} | {block_stage(P) for P in declared} |
               {stage_of(p) for p in targeted})
    if len(present - {-1}) <= 1:
        return None                              # one stage: no need to split

    def declare(p):                              # a weightless rule, so the stage has p
        names = {"n": iter("xyzuvw"), "l": iter("GHKLPQ")}
        return f"0 {p}({','.join(next(names[t]) for t in sig[p])})"
    n, rng = sampler.pop("n"), sampler.pop("rng")
    out = []
    for _ in range(n):
        lat, ev, info = model.latent, dict(model.evidence or {}), []
        for s_ in (0, 1, 2):
            rs = [t for t in rules if rule_stage(t) == s_]
            ps = [t for t in fixed_rules if rule_stage(t) == s_]
            pr = [P for P in declared if block_stage(P) == s_]
            mentioned = {p for t in rs + ps for p in _rule_preds(t)} | {p for P in pr for p, _ in P.predicates()}
            ps = ps + [declare(p) for p in sorted(targeted) if stage_of(p) == s_ and p not in mentioned]
            if not (rs or ps or pr):
                continue
            last = s_ == 2
            G = _run(model.k, rs, pr + ps, beta=model.beta, latent=lat, evidence=ev,
                     field=per_pred(model.field, s_), density=per_pred(model.density, s_),
                     edges=model.edges if last else None,
                     constraint=model.constraint if last else None,
                     relations=relations if last else [], normalize=model.normalize,
                     hard=model.hard, distinct=model.distinct, n=1, rng=rng, **sampler)
            info.append(G.graph["mln"])
            if s_ == 0:                          # the latent graph, for later stages
                lat = _latent_graph(int(model.latent), G, sig)
                continue
            for p in sig:                        # this stage's results become evidence
                if stage_of(p) != s_ or p in relations:
                    continue
                if sig[p] == ("n",):
                    ev[p] = [x for x in G if G.nodes[x].get(p)]
                elif sig[p] == ("n", "l"):
                    ev[p] = [(x, g) for x in G for g in G.nodes[x][p]]
                elif p in G.graph["hidden"]:
                    ev[p] = [tuple(int(i) for i in t) for t in np.argwhere(G.graph["hidden"][p])]
        G.graph["mln"] = dict(G.graph["mln"], stages=info)
        if lat is not None and not isinstance(lat, (int, np.integer)):
            G.graph["latent"] = lat
        out.append(G)
    return out[0] if n == 1 else out


def _pattern_keys(rows):
    """A hashable key for each 0/1 row."""
    return [r.tobytes() for r in np.packbits(np.asarray(rows, np.uint8), axis=1)]


def _state_corrections(masks, pats):
    """For a block: which (factor, state) pairs differ from the all-off state,
    and with what bits -- usually one per factor, as a state turns on few
    atoms. Factors as positions in the block's factor list, and states."""
    from scipy.sparse import csr_matrix
    pm = (masks @ csr_matrix(pats.T.astype(np.int64))).tocoo()     # masks: sparse, factors x atoms
    return dict(pm_f=pm.row.astype(np.int32), pm_bits=pm.data.astype(np.uint16),   # bits < 2^16
                pm_s=pm.col.astype(np.int16 if len(pats) < 2 ** 15 else np.int32))


def _batch_blocks(blocks, F, chunk=200_000):
    """Group blocks into batches updated in one vectorised step: blocks of the
    same shape (atom count, states, weights, which atoms are shown or have
    fixed counts) that share no grounding, so their updates are independent.
    Greedy colouring within each shape, cut into chunks of at most `chunk`
    factors. The blocks keep only what initialisation and reports use."""
    from scipy.sparse import csr_matrix
    shapes = {}
    for i, B in enumerate(blocks):
        key = (B["kind"], len(B["atoms"]), B["pats"].tobytes(), B["lw"].tobytes(),
               tuple(B["sh"][:, 0]), tuple(map(tuple, B["fixed_cols"])))
        shapes.setdefault(key, []).append(i)
    batches = []
    for members in shapes.values():
        if len(members) > 1:                     # blocks sharing a grounding: different colours
            indptr = np.concatenate([[0], np.cumsum([len(blocks[i]["fs"]) for i in members])])
            cols = np.concatenate([blocks[i]["fs"] for i in members])          # int32, built directly
            A = csr_matrix((np.ones(len(cols), np.int8), cols, indptr), shape=(len(members), F))
            del cols
            C = (A @ A.T).tocsr()
            colour = np.full(len(members), -1)
            for q in np.argsort(-np.diff(C.indptr), kind="stable"):
                nb = colour[C.indices[C.indptr[q]:C.indptr[q + 1]]]
                used = np.zeros(len(nb) + 2, bool)
                used[nb[(nb >= 0) & (nb < len(used))]] = True
                colour[q] = int(np.argmin(used))
        else:
            colour = np.zeros(1, int)
        for c in range(colour.max() + 1):
            group, cur, size = [members[q] for q in np.flatnonzero(colour == c)], [], 0
            for i in group + [None]:
                if i is None or (cur and size + len(blocks[i]["fs"]) > chunk):
                    batches.append(_make_batch([blocks[j] for j in cur]))
                    for j in cur:                # the batch holds these now
                        for key in ("G", "r", "clear", "pm_f", "pm_bits", "pm_s"):
                            blocks[j].pop(key, None)
                    cur, size = [], 0
                if i is not None:
                    cur.append(i)
                    size += len(blocks[i]["fs"])
    for B in blocks:
        B.pop("fs", None)
    return batches


def _make_batch(bl):
    """Concatenate blocks of one shape into a batch."""
    nf = np.array([len(B["G"]) for B in bl])
    starts = np.concatenate([[0], np.cumsum(nf)[:-1]])
    first = bl[0]
    sh = np.stack([B["sh"] for B in bl])         # (blocks, atoms, 3)
    return dict(
        atoms=np.stack([B["atoms"] for B in bl]), pats=first["pats"], lw=first["lw"],
        kind=first["kind"], index=first["index"], starts=starts.astype(np.int64),
        G=np.concatenate([B["G"] for B in bl]), r=np.concatenate([B["r"] for B in bl]),
        clear=np.concatenate([B["clear"] for B in bl]),
        pm_f=np.concatenate([B["pm_f"].astype(np.int64) + o for B, o in zip(bl, starts)]),
        pm_bits=np.concatenate([B["pm_bits"] for B in bl]),
        pm_s=np.concatenate([B["pm_s"].astype(np.int64) for B in bl]),
        pm_blk=np.concatenate([np.full(len(B["pm_f"]), j, np.int64) for j, B in enumerate(bl)]),
        shown_cols=first["shown_cols"], L=sh[:, :, 0], U=sh[:, :, 1], V=sh[:, :, 2],
        fixed_cols=first["fixed_cols"], fixed_cnt=first["fixed_cnt"])


def _colour_classes(js, inc_f, F, chunk=200_000):
    """Split atoms (positions js into inc_f) into colour classes: atoms in one
    class share no grounding, so their conditionals are independent and they
    can be updated at once. Greedy colouring, largest degree first; big classes
    are cut into chunks of at most `chunk` factors. Each class is (positions,
    concatenated factor ids, their masks, start of each atom's segment)."""
    if not len(js):
        return [], 0
    from scipy.sparse import csr_matrix
    rows = np.concatenate([np.full(len(inc_f[j][0]), i) for i, j in enumerate(js)])
    cols = np.concatenate([inc_f[j][0] for j in js])
    A = csr_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(len(js), F))
    C = (A @ A.T).tocsr()                        # atoms sharing a grounding
    colour = np.full(len(js), -1)
    for i in np.argsort(-np.diff(C.indptr), kind="stable"):
        nb = colour[C.indices[C.indptr[i]:C.indptr[i + 1]]]
        used = np.zeros(len(nb) + 2, bool)
        used[nb[(nb >= 0) & (nb < len(used))]] = True
        colour[i] = int(np.argmin(used))
    classes = []
    for c in range(colour.max() + 1):
        members, cur, size = js[colour == c], [], 0
        for j in list(members) + [None]:
            if j is None or (cur and size + len(inc_f[j][0]) > chunk):
                Fc = np.concatenate([inc_f[q][0] for q in cur])
                mc = np.concatenate([inc_f[q][1] for q in cur])
                starts = np.cumsum([0] + [len(inc_f[q][0]) for q in cur[:-1]])
                classes.append((np.array(cur), Fc, mc, starts))
                cur, size = [], 0
            if j is not None:
                cur.append(j)
                size += len(inc_f[j][0])
    return classes, int(colour.max()) + 1


def _block_states(atoms, groundings, N, ONE, limit):
    """All 0/1 assignments to `atoms` satisfying every grounding in
    `groundings` (each (row of atom indices, truth table), with all its
    non-constant atoms among `atoms`), by depth-first search with pruning.
    None if there are more than `limit`."""
    pos = {int(a): i for i, a in enumerate(atoms)}
    b = len(atoms)
    checks = [[] for _ in range(b)]              # groundings completed at each depth
    for row, table in groundings:
        cols, last = [], -1
        for j, a in enumerate(row):
            a = int(a)
            if a < N:
                cols.append((j, pos[a], 0))
                last = max(last, pos[a])
            else:
                cols.append((j, -1, 1 if a == ONE else 0))
        if last >= 0:
            checks[last].append((cols, table))
    out, val = [], np.zeros(b, np.int8)

    def ok(d):
        for cols, table in checks[d]:
            idx = 0
            for j, p, const in cols:
                if (val[p] if p >= 0 else const):
                    idx |= 1 << j
            if table[idx] <= 0:
                return False
        return True

    def rec(d):
        if d == b:
            out.append(val.copy())
            return len(out) <= limit
        for v in (0, 1):
            val[d] = v
            if ok(d) and not rec(d + 1):
                return False
        return True

    if not rec(0):
        return None
    return np.array(out, np.int8)


# -- the sampler --------------------------------------------------------------

class MLN:
    """A Markov logic network over k nodes, and optionally latent nodes: build
    it once, inspect it, sample from it.

        model = MLN(30, [symmetric("R"), transitive("R"),
                         categorical("In"), sbm("In", "R", 1.0, -3.0)],
                    latent=3, beta=2.0)
        print(model.describe())          # every rule, its kind and stage
        G = model.sample()

    model.rules lists every rule after components are expanded, with its kind
    and stage; model.predicates each predicate's argument types and stage.

    rules       rule strings and Components (symmetric, antisymmetric,
                transitive, functional, inverse, commute, directions,
                categorical, sbm, forest, nested, path, torus).
                Plain rule strings are scaled by beta.
    beta        multiplies the weights of all soft rules in `rules`: one inverse
                temperature for the structure. Hard rules, fixed rules, blocks
                and fields are not scaled.
    fixed       Block objects (Categorical, Boltzmann, Path, or a subclass) and
                rule strings not scaled by beta. A Block covers a list of
                unary predicates, or one two-argument predicate -- In(x,G), a
                relation R(x,y), a latent relation Parent(G,H) -- with a block
                per node (or latent node) over its allowed states, so its
                constraints are sampled without being grounded.
    latent      latent nodes: a number m, or a graph over latent nodes 0..m-1
                (its relations and unary node attributes become evidence).
                Uppercase variables (G, H, ...) range over latent nodes and
                lowercase ones over the k nodes: In(x,G) relates a node to a
                latent node, Near(G,H) two latent nodes. A (node, latent)
                predicate is reported per node as a list, G.nodes[x]["In"]; the
                latent graph is in G.graph["latent"].
    evidence    fixed atoms: {predicate: list of argument tuples that are true};
                every other atom of a listed predicate is false. Fixed atoms are
                constants inside the rules, so rules through them link only the
                atoms that are sampled.
    field       a weight on every atom of a predicate, like an Ising field: a
                number (for every relation) or {predicate: weight}. Not scaled by
                beta. -2.0 alone gives each edge probability sigmoid(-2) = 0.12.
    degree      a target mean degree per node instead of a density: a number
                (for every relation) or {predicate: degree}, for two-argument
                predicates. degree=2 means each node has on average 2 outgoing
                edges in that relation -- and so 2 incoming, 4 in all -- i.e.
                2k edges; for In(x,G), 2 memberships per node. It is converted
                to a density: degree / (number of possible targets: k - 1
                between nodes, m for latent ones, m - 1 between latent ones).
    density     a target probability for each atom of a predicate (a number for
                every relation, or {predicate: probability}): its field is tuned
                during burn-in to hit it on average, then frozen (reported in
                G.graph["mln"]["field"]). Check the spread across graphs: near a
                sharp sparse/dense transition the right field can give a mix of
                sparse and dense graphs.
    edges       a fixed number of edges: an int (for every relation) or
                {relation: count}. Those relations then change only by moving an
                edge (swap moves), so rules rearrange edges but can't delete
                them. A field on such a relation has no effect.
    normalize   divide each soft rule's weight by the mean number of its
                groundings an atom appears in. Tends to overcorrect, leaving
                many-variable rules (transitivity) weakly enforced.
    relations   the (node, node) predicates to show as edges; action = position
                in this list. Default: all of them, sorted. Others are hidden.
    constraint  None, "strong", "weak", or a function of the shown edges (a bool
                array [relation, u, v]) saying whether they are allowed.
                Connectivity is of the union of the relations. Moves that would
                break it are refused, which conditions on it exactly.
    hard        the weight given to "inf" rules: a violation costs e^-hard.
    distinct    distinct variables of the same type bind distinct nodes.
    order       "hidden_first" (default) samples in stages, each given the ones
                before: predicates over latent nodes only (the latent graph),
                then hidden ones (memberships like In(x,G), and anything not in
                relations), then relations -- latent structure first, then data
                given it. "joint" samples everything together, so relations also
                shape the hidden structure. "latent_first" samples the latent
                graph, then everything else jointly. A rule belongs to the stage
                of the latest predicate it mentions; field, density, edges and
                constraint apply in the stage of their predicates.
                G.graph["mln"]["stages"] has each stage's report.

    Formulas may also compare variables of the same type by index, x < y; e.g.
    Parent(G,H) -> H < G keeps a latent forest acyclic.
    """

    def __init__(self, k, rules, beta=1.0, fixed=(), latent=None, evidence=None, field=None,
                 density=None, degree=None, edges=None, normalize=False, relations=None, constraint=None,
                 hard=20.0, distinct=True, order="hidden_first"):
        if order not in ("joint", "latent_first", "hidden_first"):
            raise ValueError("order must be 'joint', 'latent_first' or 'hidden_first'")
        self.k, self.beta, self.latent = k, beta, latent
        self.density, self.edges, self.normalize, self.constraint = density, edges, normalize, constraint
        self.hard, self.distinct, self.order = hard, distinct, order
        listify = lambda x: [x] if isinstance(x, (str, Structure, Component)) else list(x)
        entries, fields, comp_ev, comp_rel, self._node_attrs = [], {}, {}, [], {}
        for item, default in [(r, "scaled") for r in listify(rules)] + [(p, "fixed") for p in listify(fixed)]:
            if isinstance(item, Component):
                if item.nodes is not None and item.nodes != k:
                    raise ValueError(f"{item!r} needs {item.nodes} nodes, not {k}")
                for p, tuples in item.evidence.items():
                    if p in comp_ev or p in (evidence or {}):
                        raise ValueError(f"evidence for {p!r} is given twice")
                    comp_ev[p] = tuples
                comp_rel += [p for p in item.relations if p not in comp_rel]
                for a, vals in item.node_attrs.items():
                    self._node_attrs.setdefault(a, {}).update(vals)
                entries += [(r, "scaled", item.name) for r in item.rules]
                entries += [(r, "fixed", item.name) for r in item.fixed]
                entries += [(P, "structure", item.name) for P in item.blocks]
                entries += [(r, "joint-only", item.name) for r in item.joint_only]
                fields.update(item.field)
            elif isinstance(item, Structure):
                entries.append((item, "structure", ""))
            else:
                entries.append((item, default, ""))
        for e, kd, src in entries:               # structures' preferred relation order
            if kd == "structure":
                comp_rel += [p for p in e.relations() if p not in comp_rel]
        self._entries = entries
        total = set()                            # predicates with at most one successor per node
        for e, kd, _ in entries:
            if kd == "structure":
                total |= e.functional()
        for item in listify(rules) + listify(fixed):
            for p in getattr(item, "needs_total", []):
                if p not in total and p not in (evidence or {}) and p not in comp_ev:
                    warnings.warn(f"{item.name}: {p} has no at-most-one-successor constraint (e.g. "
                                  f"functional({p!r}, total=True)), so the rule doesn't mean commutation, "
                                  f"and it is satisfied by adding edges -- expect dense graphs")
        self.evidence = {**comp_ev, **(evidence or {})} or None
        if field is None or isinstance(field, dict):
            self.field = {**fields, **(field or {})} or None
        else:
            self.field = field
        self.sig = _signatures([e for e, kd, _ in entries if kd != "structure"],
                               [e for e, kd, _ in entries if kd == "structure"])
        if relations is None:                    # components' preferred order first
            rest = sorted(p for p, s_ in self.sig.items() if s_ == ("n", "n") and p not in comp_rel)
            relations = [p for p in comp_rel if p in self.sig] + rest
        self.relations = relations
        have_graph = latent is not None and not isinstance(latent, (int, np.integer))
        fixed = dict(self.evidence or {})
        if have_graph:
            fixed.update(dict.fromkeys(latent.graph.get("relations", {})))
        self._stage_of = _stage_plan(self.sig, self.relations, fixed, have_graph, order)
        if degree is not None:                   # a target degree is a density in other units
            mirrored = {b for e, kd, _ in entries if kd == "structure" for b, _ in e.mirrors()}
            if not isinstance(degree, dict):
                degree = {p: degree for p in self.relations if p not in mirrored}
            m_lat = 0 if latent is None else (int(latent) if isinstance(latent, (int, np.integer))
                                              else latent.number_of_nodes())
            dens = (dict(self.density) if isinstance(self.density, dict) else
                    {p: self.density for p in self.relations if p not in mirrored}
                    if self.density is not None else {})
            for p, d in degree.items():
                types = self.sig.get(p)
                if types is None or len(types) != 2:
                    raise ValueError(f"degree for {p!r}: it must be a two-argument predicate of the model")
                if p in dens:
                    raise ValueError(f"{p!r} has both a density and a degree")
                size = (k if types[1] == "n" else m_lat) - (types[0] == types[1] and distinct)
                if not 0 < d < size:
                    raise ValueError(f"degree for {p!r} must be between 0 and {size}, not {d}")
                if p in total and d > 1:
                    raise ValueError(f"degree for {p!r} is {d}, but each node has at most one {p} edge")
                dens[p] = d / size
            self.density = dens
        if isinstance(self.density, dict) or self.density is not None:   # targets a structure can't reach
            for p, q in (self.density.items() if isinstance(self.density, dict) else
                         ((p, self.density) for p in self.relations)):
                types = self.sig.get(p)
                if p in total and types and len(types) == 2:
                    size = (k if types[1] == "n" else (0 if latent is None else
                            int(latent) if isinstance(latent, (int, np.integer)) else latent.number_of_nodes()))
                    size -= types[0] == types[1] and distinct
                    if size and q * size > 1:
                        warnings.warn(f"density {q} for {p!r} means {q * size:.2g} edges per node, but each "
                                      f"node has at most one: it will saturate (use degree=, at most 1)")
        self.degree = degree

    def _stage(self, item):
        ps = [p for p, _ in item.predicates()] if isinstance(item, Structure) else _rule_preds(item)
        return STAGES[max((self._stage_of(p) for p in ps if p in self.sig), default=-1)]

    @property
    def rules(self):
        """Every rule after expanding components: dicts with the rule string,
        its kind (scaled by beta, fixed, structure, joint-only), its stage, and the
        component it came from. A Structure is listed as its equivalent rules, with
        "sampled by" naming it: those rules describe it, while the sampler
        samples them by its own moves instead of grounding them."""
        out = []
        for e, kd, src in self._entries:
            if kd == "structure":
                for r in (e.rules() or [e]):
                    out.append({"rule": r, "kind": "structure", "stage": self._stage(e),
                                "from": src, "sampled by": e})
                continue
            unused = kd == "joint-only" and self.order != "joint"
            out.append({"rule": e, "kind": kd + (" (unused: staged order)" if unused else ""),
                        "stage": self._stage(e), "from": src})
        return out

    @property
    def structures(self):
        return [e for e, kd, _ in self._entries if kd == "structure"]

    @property
    def predicates(self):
        names = {"n": "node", "l": "latent"}
        return {p: {"args": tuple(names[t] for t in s_), "stage": STAGES[self._stage_of(p)],
                    "shown": p in self.relations}
                for p, s_ in sorted(self.sig.items())}

    def describe(self):
        lat = self.latent if isinstance(self.latent, (int, np.integer)) or self.latent is None \
            else f"a graph over {self.latent.number_of_nodes()}"
        lines = [f"MLN over {self.k} nodes" + (f", latent nodes: {lat}" if lat else "") +
                 f"; order {self.order}; beta {self.beta}", "predicates:"]
        for p, d in self.predicates.items():
            shown = f"shown, action {self.relations.index(p)}" if d["shown"] else "hidden"
            lines.append(f"  {p:10s} ({', '.join(d['args'])})".ljust(36) + f"stage {d['stage']:9s} {shown}")
        lines.append("rules (scaled rules at their base weight, before beta):")
        for r in self.rules:
            src = r["from"] or ""
            if "sampled by" in r:
                src = (src + ", " if src else "") + f"sampled by {r['sampled by']!r}"
            lines.append(f"  [{r['stage']:9s}] {r['kind']:10s} {r['rule']!s}".ljust(70) +
                         (f"  <- {src}" if src else ""))
        if self.field:
            lines.append(f"field: {self.field}")
        if self.degree:
            lines.append(f"degree: {self.degree}  (as density: "
                         f"{ {p: round(v, 4) for p, v in self.density.items()} })")
        elif self.density:
            lines.append(f"density: {self.density}")
        return "\n".join(lines)

    def __repr__(self):
        return f"MLN(k={self.k}, {len(self._entries)} rules and blocks, order={self.order!r})"

    def sample(self, n=1, sweeps=1000, method="anneal", anneal=0.5, temps=None, beta_min=0.1,
               adapt=0.5, block_limit=4096, auto_blocks=True, couple_min=3.0, block_atoms=64,
               rng=None):
        """Sample n graphs (a list if n > 1, else one graph).

        sweeps      Gibbs sweeps per graph (each updates every sampled atom, every
                    block, and the swap moves for fixed edge counts).
        method      "anneal": raise the inverse temperature from 0 to 1 over the
                    first `anneal` fraction of the sweeps, then hold it at 1.
                    Finds low-energy graphs well, but at low temperature settles
                    in whichever basin it reaches first. "tempering": parallel
                    tempering, `temps` replicas per graph (default about 0.8 *
                    sqrt(number of atoms)) between inverse temperatures beta_min
                    and 1, swapping neighbours each sweep; their spacing is tuned
                    over the first `adapt` fraction of the sweeps and then frozen.
                    Slower per sweep, but correct where single chains stick; a
                    warning says if some neighbours rarely swap.
        block_limit Blocks with at most this many states are updated a node at a
                    time choosing among their states; larger ones an atom at a
                    time, using the block's weights.
        auto_blocks also update together atoms tied by strong rules: atoms in a
                    grounding of a hard rule, or of a rule with |weight| >=
                    couple_min (after beta), are linked; each cluster of at most
                    block_atoms atoms with at most block_limit states becomes a
                    block, its states those satisfying the hard groundings inside
                    it. Blocks may overlap; every atom also gets its own update.
        rng         a numpy Generator.

        Each graph's G.graph["mln"] reports the score, the false groundings and
        number of groundings of each rule, the sampled atoms per predicate, the
        blocks made, the tuned fields, and for tempering the swap rates.
        """
        joint = self.order == "joint"
        rules = [e for e, kd, _ in self._entries if kd == "scaled"]
        fixed = [e for e, kd, _ in self._entries
                 if kd in ("fixed", "structure") or (kd == "joint-only" and joint)]
        sampler = dict(n=n, sweeps=sweeps, method=method, anneal=anneal, temps=temps,
                       beta_min=beta_min, adapt=adapt, block_limit=block_limit,
                       auto_blocks=auto_blocks, couple_min=couple_min, block_atoms=block_atoms,
                       rng=np.random.default_rng() if rng is None else rng)
        out = _staged(self, rules, fixed, dict(sampler)) if not joint else None
        if out is None:
            out = _run(self.k, rules, fixed, beta=self.beta, latent=self.latent,
                       evidence=self.evidence, field=self.field, density=self.density,
                       edges=self.edges, normalize=self.normalize, relations=self.relations,
                       constraint=self.constraint, hard=self.hard, distinct=self.distinct, **sampler)
        for G in ([out] if n == 1 else out):     # attributes components give the nodes
            for a, vals in self._node_attrs.items():
                nx.set_node_attributes(G, vals, a)
        return out


_MODEL_ARGS = ("beta", "fixed", "latent", "evidence", "field", "density", "degree", "edges", "normalize",
               "relations", "constraint", "hard", "distinct", "order")


def mln(k, rules, **kwargs):
    """Shortcut: MLN(k, rules, ...).sample(...), with each keyword going to the
    constructor (the model) or to sample() (how to sample it)."""
    model = {key: v for key, v in kwargs.items() if key in _MODEL_ARGS}
    return MLN(k, rules, **model).sample(**{key: v for key, v in kwargs.items() if key not in model})


def full_graph(G):
    """One graph with everything a sample contains: the visible nodes 0..k-1
    and latent nodes "L0", "L1", ..., each with kind "visible" / "latent" and
    its true unary predicates as attributes, and an edge for every true atom of
    every two-argument predicate -- shown relations, memberships like In(x,G),
    relations on latent nodes, hidden relations -- with the predicate's name
    as the edge's "predicate". Predicates with more arguments stay in
    G.graph["hidden"]."""
    types = G.graph.get("predicates", {})
    lat = G.graph.get("latent")
    m = lat if isinstance(lat, (int, np.integer)) else (lat.number_of_nodes() if lat is not None else 0)
    name = lambda t, i: i if t == "node" else f"L{i}"
    F = nx.MultiDiGraph()
    F.add_nodes_from(G.nodes(data=True), kind="visible")
    F.add_nodes_from((f"L{g}" for g in range(m)), kind="latent")
    for u, v, d in G.edges(data=True):           # shown relations
        rel = {l: p for p, l in G.graph.get("relations", {}).items()}
        F.add_edge(u, v, predicate=rel.get(d.get("action"), d.get("action")))
    for p, t in types.items():
        if t == ("node", "latent"):              # memberships, stored per node
            for x in G:
                for g in G.nodes[x].get(p, []):
                    F.add_edge(x, f"L{g}", predicate=p)
                F.nodes[x].pop(p, None)
    if lat is not None and not isinstance(lat, (int, np.integer)):
        lrel = {l: p for p, l in lat.graph.get("relations", {}).items()}
        for u, v, d in lat.edges(data=True):     # the latent graph's relations
            F.add_edge(f"L{u}", f"L{v}", predicate=lrel.get(d.get("action")))
        for g, d in lat.nodes(data=True):
            F.nodes[f"L{g}"].update({p: val for p, val in d.items() if isinstance(val, bool)})
    for p, arr in G.graph.get("hidden", {}).items():
        t = types.get(p)
        if t is None:
            continue
        if len(t) == 2:                          # hidden two-argument relations
            for a, b in np.argwhere(arr):
                F.add_edge(name(t[0], int(a)), name(t[1], int(b)), predicate=p)
        elif t == ("latent",):
            for g in np.flatnonzero(arr):
                F.nodes[f"L{g}"][p] = True
    return F


_DIRECTIONS = {"N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0)}


def gluing_diagram(G, directions=None, root=None, ax=None, max_patches=None):
    """Draw G the way topologists draw a space: flat patches of grid (by
    default N up, E right), every edge that fits its patch drawn in full, and
    every other edge as two labelled half-edges -- one leaving its start, one
    arriving at its end -- that are glued together.

    Each leaving half-edge is labelled with the node it lands on (->9), each
    arriving one with the node it comes from (13->). Patches are grown
    breadth-first from a root; a node whose grid point is already taken (a
    twist, a dislocation) starts a new patch, side by side, like the charts
    of an atlas. Gluings within a patch that shift by the same translation
    (either way) share a colour and a letter in the legend, like matched
    sides of a polygon: a flat torus is one patch with two or three of them.
    Gluings between patches are grey. A tree-like,
    non-Euclidean structure would fall into many tiny patches; past
    max_patches (default: a quarter of the nodes) the spanning tree is drawn
    in layers instead, with every other edge a labelled pair of half-edges.

    directions  {relation: (dx, dy)}. By default N/S/E/W if the graph has
                those relations, else the mirrored pairs the sample records
                (G.graph["mirrors"], from Spatial): the first sampled relation
                points right, its mirror left, the second up, its mirror down.
                Pairs that mirror each other are drawn once.
    Returns {letter: (dx, dy) translation} for the gluings within patches.
    """
    import matplotlib.pyplot as plt
    from collections import deque
    rels = G.graph.get("relations", {})
    rel = {l: p for p, l in rels.items()}
    if directions is None and any(p in rels for p in _DIRECTIONS):
        directions = _DIRECTIONS
    elif directions is None and G.graph.get("mirrors"):
        mirrors = G.graph["mirrors"]             # mirror -> sampled relation
        bases = [p for p in rels if p in set(mirrors.values())]
        if len(bases) > 2:
            warnings.warn(f"{len(bases)} axes ({bases}); drawing the first two")
        unit = [(1, 0), (0, 1)]
        directions = {}
        for a, (dx, dy) in zip(bases, unit):
            directions[a] = (dx, dy)
            directions.update({b: (-dx, -dy) for b, base in mirrors.items() if base == a})
    elif directions is None:
        raise ValueError(f"gluing_diagram: which relations are directions? The graph has {sorted(rels)}; "
                         f"pass directions={{relation: (dx, dy)}}")
    dirs = dict(directions)
    if not any(p in dirs for p in rels):
        raise ValueError(f"gluing_diagram: none of the directions {sorted(dirs)} is a relation of the graph "
                         f"({sorted(rels)})")
    links = {}                                   # one per geometric link: (u, v, (dx, dy))
    for u, v, d in G.edges(data=True):
        p = rel.get(d.get("action"))
        if p not in dirs:
            continue
        dx, dy = dirs[p]
        a, b, step = (u, v, (dx, dy)) if (dx, dy) > (0, 0) else (v, u, (-dx, -dy))
        links[(a, b, step)] = True
    links = list(links)
    nbrs = {x: [] for x in G}
    for a, b, (dx, dy) in links:
        nbrs[a].append((b, (dx, dy)))
        nbrs[b].append((a, (-dx, -dy)))
    order = ([root] if root is not None else []) + sorted(G, key=lambda x: -len(nbrs[x]))
    patch, pos, tree, shift = {}, {}, set(), 0  # patch of each node; positions; tree links
    for r in order:
        if r in patch:
            continue
        pid = len(set(patch.values())) if patch else 0
        local, taken, q = {r: (0, 0)}, {(0, 0)}, deque([r])
        while q:
            x = q.popleft()
            for y, (dx, dy) in sorted(nbrs[x], key=lambda t: t[1]):
                c = (local[x][0] + dx, local[x][1] + dy)
                if y in patch or y in local or c in taken:
                    continue
                local[y] = c
                taken.add(c)
                tree.add(frozenset((x, y)))
                q.append(y)
        lo = min(p[0] for p in local.values())
        for x, p in local.items():
            patch[x], pos[x] = pid, (p[0] - lo + shift, p[1])
        shift += max(p[0] for p in local.values()) - lo + 3
    n_patches = len(set(patch.values()))
    limit = max(1, len(G) // 4) if max_patches is None else max_patches
    layered = n_patches > limit
    if layered:                                  # not grid-like: the spanning tree, in layers by depth
        pos, shift, seen = {}, 0, set()
        for r in order:
            if r in seen:
                continue
            depth, q = {r: 0}, deque([r])
            while q:
                x = q.popleft()
                for y, _ in nbrs[x]:
                    if y not in depth:
                        depth[y] = depth[x] + 1
                        q.append(y)
            tree = tree | {frozenset((x, y)) for x in depth for y, _ in nbrs[x]
                           if y in depth and depth[y] == depth[x] + 1 and
                           min((z for z, _ in nbrs[y] if depth.get(z) == depth[y] - 1)) == x}
            seen |= set(depth)
            layers = {}
            for x, dd in depth.items():
                layers.setdefault(dd, []).append(x)
            width = max(len(v) for v in layers.values())
            for dd, xs in layers.items():
                for i, x in enumerate(sorted(xs)):
                    pos[x] = (shift + (i - (len(xs) - 1) / 2) * 1.6, -dd * 1.6)
            shift += width * 1.6 + 3
    if ax is None:
        ax = plt.gca()
    cmap = plt.get_cmap("tab10")
    letters, out = {}, {}

    def letter_for(key):
        if key not in letters:
            n = len(letters)
            letters[key] = chr(97 + n % 26) + ("" if n < 26 else str(n // 26))
        return letters[key], cmap(list(letters).index(key) % 10)

    for a, b, (dx, dy) in links:
        pa, pb = pos[a], pos[b]
        if layered:
            if frozenset((a, b)) in tree:
                ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color="0.35", lw=1.6, zorder=1)
                continue
            col = "0.5"
            sx, sy = 0.25, 0.35
        else:
            if patch[a] == patch[b] and (pb[0] - pa[0], pb[1] - pa[1]) == (dx, dy):
                ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color="0.35", lw=1.6, zorder=1)
                continue
            if patch[a] == patch[b]:             # a translation; t and -t are one gluing
                t = (pb[0] - pa[0] - dx, pb[1] - pa[1] - dy)
                lab, col = letter_for(max(t, (-t[0], -t[1])))
                out[lab] = max(t, (-t[0], -t[1]))
            else:                                # between patches
                col = "0.5"
            sx, sy = dx, dy
        for (px, py), sgn, txt in [(pa, 1, f"\u2192{b}"), (pb, -1, f"{a}\u2192")]:
            ex, ey = px + sgn * 0.42 * sx, py + sgn * 0.42 * sy
            ax.annotate("", xy=(ex, ey) if sgn > 0 else (px, py), xytext=(px, py) if sgn > 0 else (ex, ey),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6, shrinkA=7, shrinkB=0), zorder=2)
            ax.text(px + sgn * 0.62 * sx, py + sgn * 0.62 * sy, txt, color=col, fontsize=8,
                    fontweight="bold", ha="center", va="center", zorder=3)
    for x, (px, py) in pos.items():
        ax.scatter([px], [py], s=330, color="#cfdcec", edgecolor="0.4", zorder=4)
        ax.text(px, py, str(x), ha="center", va="center", fontsize=9, zorder=5)
    if out:                                      # legend: colour -> translation
        from matplotlib.lines import Line2D
        ax.legend([Line2D([0], [0], color=letter_for(t)[1], lw=2) for t in out.values()],
                  [f"{l}: shift {t}" for l, t in out.items()], loc="upper left",
                  bbox_to_anchor=(1.0, 1.0), fontsize=8, frameon=False)
    ax.set_aspect("equal")
    ax.axis("off")
    return out


def _run(k, rules, fixed, beta, latent, evidence, field, density, edges, normalize,
         relations, constraint, n, sweeps, method, anneal, temps, beta_min, adapt,
         block_limit, auto_blocks, couple_min, block_atoms, hard, distinct, rng):
    """Ground and sample one stage: rules (scaled by beta) and fixed (Block
    objects and rule strings not scaled by beta) as plain lists. See MLN for
    the arguments."""
    rng = np.random.default_rng() if rng is None else rng
    if method not in ("anneal", "tempering"):
        raise ValueError("method must be 'anneal' or 'tempering'")
    declared = [p for p in fixed if isinstance(p, Structure)]
    fixed_rules = [p for p in fixed if not isinstance(p, Structure)]
    rules = list(rules) + fixed_rules
    parsed = [_parse_rule(r) for r in rules]
    formula = list(rules)                        # the text of each entry in rules
    kind = ["rule"] * (len(rules) - len(fixed_rules)) + ["fixed"] * len(fixed_rules)

    # the latent graph: m latent nodes, and its relations become evidence
    if latent is None:
        m_lat, Hlat = 0, None
    elif isinstance(latent, (int, np.integer)):
        m_lat, Hlat = int(latent), None
    else:
        Hlat = latent
        m_lat = Hlat.number_of_nodes()
        if sorted(Hlat.nodes) != list(range(m_lat)):
            raise ValueError("the latent graph's nodes must be 0..m-1")
    dom = {"n": k, "l": m_lat}

    sig = _signatures(rules, declared)
    vtype = _vtype
    if any("l" in s for s in sig.values()) and m_lat == 0:
        raise ValueError("rules use latent variables (uppercase) but latent= gives no latent nodes")

    def var_list(s):                             # variables for an anchor or field rule
        return ",".join(f"x{j}" if t == "n" else f"X{j}" for j, t in enumerate(s))

    covered = {}
    for P in declared:
        P.bind_domain(sig, dom, distinct)
        for p, _ in P.predicates():
            if p in covered:
                raise ValueError(f"{p!r} is in two structures")
            covered[p] = P
            rules.append(f"anchor {p}")              # zero weight: makes its atoms exist
            formula.append(f"0 {p}({var_list(sig[p])})")
            parsed.append(_parse_rule(formula[-1]))
            kind.append("anchor")

    # evidence: atoms fixed to the listed tuples (true) and false otherwise
    if evidence is None:
        evidence = {}
    evidence = {p: list(v) for p, v in evidence.items()}
    if Hlat is not None:
        for p, l in Hlat.graph.get("relations", {}).items():
            if p in sig:
                evidence.setdefault(p, [(u, v) for u, v, d in Hlat.edges(data=True)
                                        if d.get("action") == l])
        for p in sig:
            if sig[p] == ("l",) and p not in evidence and any(p in Hlat.nodes[g] for g in Hlat):
                evidence[p] = [g for g in Hlat if Hlat.nodes[g].get(p)]
        for p, arr in Hlat.graph.get("hidden", {}).items():
            if p in sig and p not in evidence and all(t == "l" for t in sig[p]):
                evidence[p] = [tuple(int(i) for i in t) for t in np.argwhere(arr)]
    for p in evidence:
        if p not in sig:
            raise ValueError(f"evidence for {p!r}, which no rule uses")
        if p in covered:
            raise ValueError(f"{p!r} has evidence and a Block")

    if relations is None:
        relations = sorted(p for p, s in sig.items() if s == ("n", "n"))
    # fields live outside the factors (in h, below) so density can tune them;
    # a zero-weight rule per field just makes its atoms part of the model
    mirrored = {b for P in declared for b, _ in P.mirrors()}   # these share their partner's atoms
    if field is None:
        field = {}
    elif not isinstance(field, dict):
        field = {p: field for p in relations if p not in mirrored}
    if density is None:
        density = {}
    elif not isinstance(density, dict):
        density = {p: density for p in relations if p not in mirrored}
    mu = {p: float(h) for p, h in field.items()}
    for p, q in density.items():
        if not 0 < q < 1:
            raise ValueError(f"density for {p!r} must be strictly between 0 and 1")
        mu.setdefault(p, float(np.log(q / (1 - q))))
    field_rule = {}
    for p in mu:
        if p not in sig:
            raise ValueError(f"field or density on {p!r}, which no rule uses")
        if p in evidence and p in density:
            raise ValueError(f"{p!r} is evidence, so it can't have a density")
        args = var_list(sig[p])
        field_rule[p] = len(parsed)
        rules.append(f"field {p}")
        formula.append(f"0 {p}({args})")
        parsed.append(_parse_rule(formula[-1]))
        kind.append("field")
    for p in relations:
        if sig.get(p) != ("n", "n"):
            raise ValueError(f"relation {p!r} is not a binary predicate between nodes")
    offset, N, dims = {}, 0, {}
    for p in sorted(sig):
        offset[p] = N
        dims[p] = tuple(dom[t] for t in sig[p])
        N += int(np.prod(dims[p], dtype=np.int64))
    ZERO, ONE = N, N + 1
    # evidence atoms are replaced by the constants ONE / ZERO wherever they occur
    if N + 2 >= 2 ** 31:
        raise ValueError("too many atoms")
    cmap = np.arange(N + 2, dtype=np.int32)    # atom indices fit in 32 bits
    ev_true = []
    for p, tuples in evidence.items():
        cmap[offset[p]:offset[p] + int(np.prod(dims[p]))] = ZERO
        for t in tuples:
            t = (t,) if np.isscalar(t) else tuple(t)
            if len(t) != len(dims[p]) or any(not 0 <= int(v) < d for v, d in zip(t, dims[p])):
                raise ValueError(f"evidence {p}{t} is out of range")
            ix = offset[p] + int(np.ravel_multi_index(tuple(int(v) for v in t), dims[p]))
            cmap[ix] = ONE
            ev_true.append(ix)

    mirror_of = {}                               # mirrored relation -> the one sampled
    for P in declared:
        for b, a in P.mirrors():
            if sig.get(a) != sig.get(b) or len(sig.get(a, ())) != 2 or sig[a][0] != sig[a][1]:
                raise ValueError(f"{P!r}: {b} and {a} must be relations within one type")
            mirror_of[b] = a
            xs, ys = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
            cmap[offset[b] + (ys * k + xs).ravel()] = offset[a] + (xs * k + ys).ravel()   # b(y,x) is a(x,y)

    def literal_atoms(lit, tup, vpos, where):
        """The atom index (or ONE / ZERO) of a literal at each row of tup."""
        col = lambda a: a if isinstance(a, int) else tup[:, vpos[a]]   # a number, or a variable
        if lit[0] == "atom":
            _, p, args = lit
            ix = np.full(len(tup), offset[p], np.int32)
            stride = 1
            for j in range(len(args) - 1, -1, -1):
                if isinstance(args[j], int) and not 0 <= args[j] < dims[p][j]:
                    raise ValueError(f"{p}: argument {args[j]} is out of range, in {where!r}")
                ix += col(args[j]) * stride
                stride *= dims[p][j]
            return cmap[ix]
        if lit[0] == "eq":
            return np.where(np.broadcast_to(col(lit[1]) == col(lit[2]), len(tup)), ONE, ZERO).astype(np.int32)
        return np.where(np.broadcast_to(col(lit[1]) < col(lit[2]), len(tup)), ONE, ZERO).astype(np.int32)   # lt

    def condition_holds(cond, tup, vpos, where):
        """Evaluate a where condition at each row of tup. It may use only the
        variables in vpos (plus its own exists), and only atoms fixed in this
        stage: evidence, the latent graph, or earlier stages."""
        e = _expand(_Parser(cond).parse(), dom)
        lits = []
        _literals(e, lits)
        used = {v for lit in lits for v in (lit[2] if lit[0] == "atom" else lit[1:]) if isinstance(v, str)}
        if used - set(vpos):
            raise ValueError(f"in {where!r}: the condition uses {sorted(used - set(vpos))}, "
                             f"which the rule doesn't bind")
        vals = {}
        for lit in lits:
            ix = literal_atoms(lit, tup, vpos, where)
            if ((ix != ONE) & (ix != ZERO)).any():
                raise ValueError(f"in {where!r}: the where condition involves atoms sampled in "
                                 f"the same stage; it must be fixed by then (use a staged order)")
            vals[lit] = ix == ONE
        return np.broadcast_to(_eval_vec(e, vals.__getitem__), (len(tup),)).copy()

    # ground every rule: rows of atom indices, padded with ZERO to M columns
    parsed = [_parse_rule(t, dom) for t in formula]   # exists expanded over the domains
    M = max(len(lits) for _, lits, _, _ in parsed)
    POW = 1 << np.arange(M, dtype=np.int64)
    grounds, rule_of, tables, weights = [], [], [], []
    for r, (w, lits, vars_, table) in enumerate(parsed):
        types = [vtype(v) for v in vars_]
        tup = _tuples([dom[t] for t in types], types, distinct)
        vpos = {v: j for j, v in enumerate(vars_)}
        cond = _split_where(formula[r])[1]
        if cond:                                 # keep only the groundings where it holds
            tup = tup[condition_holds(cond, tup, vpos, formula[r])]
        cols = []
        for lit in lits:
            cols.append(literal_atoms(lit, tup, vpos, rules[r]))
        if w == 0 and cols:                      # weightless rows with nothing sampled: drop
            live = (np.stack(cols, 1) < N).any(1)
            cols = [c[live] for c in cols]
            tup = tup[live]
        cols += [np.full(len(tup), ZERO, np.int32)] * (M - len(lits))
        grounds.append(np.stack(cols, 1) if cols else np.zeros((len(tup), 0), np.int32))
        rule_of.append(np.full(len(tup), r))
        m = len(lits)
        tables.append(table[np.arange(2 ** M) & ((1 << m) - 1)])
        if np.isinf(w):
            w = np.sign(w) * hard
        elif kind[r] == "rule":
            w = w * beta
            used = grounds[-1][grounds[-1] < N]
            if normalize and len(used):
                w = w * len(np.unique(used)) / len(used)
        weights.append(w)
    for p in mu:
        if p in mirror_of:
            raise ValueError(f"field or density on {p!r}, which mirrors {mirror_of[p]!r}: put it there")
    h = np.zeros(N + 2)                          # per-atom field
    field_atoms = {p: np.unique(grounds[r][:, 0][grounds[r][:, 0] < N]) for p, r in field_rule.items()}
    for p, fa in field_atoms.items():
        h[fa] = mu[p]
    G_all = np.concatenate(grounds)
    del grounds
    rule_of = np.concatenate(rule_of).astype(np.int32)
    T = np.array(tables)
    W = np.array(weights)
    F = len(G_all)

    # which factors each atom is in, and at which positions (a bit mask)
    # (32-bit arrays and one stable sort by atom, to keep the memory peak low)
    flat = G_all.ravel()
    keep = flat < N
    atom_of = flat[keep]
    fac_of = np.repeat(np.arange(F, dtype=np.int32), M)[keep]
    mask = np.tile(POW.astype(np.int32), F)[keep]
    del flat, keep
    order_ = np.argsort(atom_of, kind="stable")  # by atom; factors stay in order within one
    atom_of, fac_of, mask = atom_of[order_], fac_of[order_], mask[order_]
    del order_
    if len(atom_of) > 1:                         # an atom twice in one factor: merge its bits
        new = np.ones(len(atom_of), bool)
        new[1:] = (atom_of[1:] != atom_of[:-1]) | (fac_of[1:] != fac_of[:-1])
        if not new.all():
            starts = np.flatnonzero(new)
            mask = np.bitwise_or.reduceat(mask, starts)
            atom_of, fac_of = atom_of[starts], fac_of[starts]
    first = np.ones(len(atom_of), bool)
    first[1:] = atom_of[1:] != atom_of[:-1]
    free = atom_of[first].astype(np.int64)       # atoms in no factor stay false
    lo = np.flatnonzero(first)
    hi = np.append(lo[1:], len(atom_of))
    inc_f = [(fac_of[s:e], mask[s:e]) for s, e in zip(lo, hi)]   # per free atom
    is_free = np.zeros(N + 2, bool)
    is_free[free] = True

    # shown edges: atom -> (relation, u, v)
    L = len(relations)
    shown = np.full((N + 2, 3), -1)
    for l, p in enumerate(relations):
        u, v = np.divmod(np.arange(k * k), k)
        shown[offset[p] + u * k + v] = np.stack([np.full(k * k, l), u, v], 1)

    # relations with a fixed edge count: their atoms move only by swaps
    if edges is None:
        edges = {}
    elif not isinstance(edges, dict):
        edges = {p: edges for p in relations}
    fixed = {}                                   # relation index -> its free atoms
    for p, m in edges.items():
        if p not in relations:
            raise ValueError(f"edges given for {p!r}, which is not a relation")
        if p in density:
            raise ValueError(f"{p!r} can't have both a fixed edge count and a density")
        if p in evidence:
            raise ValueError(f"{p!r} is evidence, so its edge count is already fixed")
        if p in covered:
            raise ValueError(f"{p!r} has a Block (or Spatial), which already shapes its edges; "
                             f"a fixed edge count on it isn't supported")
        l = relations.index(p)
        fa = offset[p] + np.arange(k * k)
        fa = fa[is_free[fa]]
        if not 0 <= m <= len(fa):
            raise ValueError(f"{p} can have 0..{len(fa)} edges, not {m}")
        fixed[l] = fa
    swap_set = np.zeros(N + 2, bool)
    for fa in fixed.values():
        swap_set[fa] = True
    # -- structures: each samples its own predicates by its own moves; Blocks,
    # declared and detected, share one batched runtime
    for P in declared:
        P.bind_latent(Hlat)
    ctx = _types.SimpleNamespace(
        k=k, dom=dom, offset=offset, dims=dims, sig=sig, N=N, ONE=ONE, ZERO=ZERO, free=free,
        is_free=is_free, inc_f=inc_f, F=F, M=M, POW=POW, G_all=G_all, rule_of=rule_of, T=T, W=W,
        h=h, rng=rng, relations=relations, mirror_of=mirror_of, shown=shown, fixed=fixed,
        block_limit=block_limit, constraint=constraint, condition_holds=condition_holds)
    groups = {}
    for P in declared:
        groups.setdefault(P.runtime_group(), []).append(P)
    runtimes = [members[0].runtime(ctx, members) for members in groups.values()]
    block_rt = next((rt for rt in runtimes if isinstance(rt, _BlockRuntime)), None)
    if block_rt is None:                         # detected blocks need somewhere to live
        block_rt = _BlockRuntime(ctx, [])
        runtimes.append(block_rt)
    in_declared = np.zeros(N + 2, bool)          # atoms that structures move themselves
    for rt in runtimes:
        in_declared |= rt.owned()

    # automatic blocks: atoms linked by strong rules (hard, or |weight| >=
    # couple_min) form clusters; each small cluster becomes a block whose
    # states are those satisfying the hard groundings inside it
    auto_skipped = 0
    if auto_blocks:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        is_hard_rule = np.abs(W) >= hard

        def clusters(rule_ids):
            """Small clusters of atoms linked by groundings of these rules."""
            sel = linkable & np.isin(rule_of, rule_ids)  # groundings that can link two atoms
            if not sel.any():
                return [], 0
            rows = G_all[sel]
            rows = np.where(rows < N, rows, rows[:, :1])       # constants link to self
            a1, a2 = rows[:, :-1].ravel(), rows[:, 1:].ravel()
            ok_ = (a1 < N) & (a2 < N) & ~in_declared[a1] & ~in_declared[a2]
            gph = coo_matrix((np.ones(ok_.sum()), (a1[ok_], a2[ok_])), shape=(N, N))
            _, comp = connected_components(gph, directed=False)
            linked = np.zeros(N, bool)
            linked[a1[ok_]] = linked[a2[ok_]] = True
            ids, sizes = np.unique(comp[linked], return_counts=True)
            out, big = [], 0
            order_ = np.argsort(comp[linked], kind="stable")
            members = np.flatnonzero(linked)[order_]
            starts = np.searchsorted(comp[linked][order_], ids)
            for c0, sz, st in zip(ids, sizes, starts):
                if 2 <= sz <= block_atoms:
                    out.append(members[st:st + sz])
                elif sz > block_atoms:
                    big += 1
            return out, big

        strong_r = np.flatnonzero((np.abs(W) >= couple_min) & np.isin(np.array(kind), ["rule", "fixed"]))
        usable_atom = np.zeros(N + 2, bool)      # atoms a detected block could include
        usable_atom[free] = True
        usable_atom[in_declared] = False
        linkable = np.zeros(F, bool)
        for j in range(M):                       # a column at a time: no copy of the table
            linkable_j = usable_atom[G_all[:, j]]
            if j == 0:
                count = linkable_j.astype(np.int8)
            else:
                count += linkable_j
        linkable = count >= 2
        del count
        candidates = {}
        for group in [[r] for r in strong_r] + ([list(strong_r)] if len(strong_r) > 1 else []):
            cl, big = clusters(group)
            auto_skipped += big
            for atoms in cl:
                candidates.setdefault(tuple(sorted(atoms.tolist())), None)
        for key_ in candidates:
            atoms = np.array(key_)
            aset = set(key_)
            fs = set()
            for a_ in key_:                      # hard groundings inside the block
                f, _ = inc_f[np.searchsorted(free, a_)]
                fs.update(int(ff) for ff in f if is_hard_rule[rule_of[ff]])
            internal = [(G_all[ff], T[rule_of[ff]]) for ff in sorted(fs)
                        if all(int(x) in aset for x in G_all[ff] if x < N)]
            pats = _block_states(atoms, internal, N, ONE, block_limit)
            if pats is None:
                auto_skipped += 1
            elif len(pats) >= 2:
                block_rt.add_detected(atoms, pats)
    block_rt.finalize()                          # batches
    gibbs = np.flatnonzero(~swap_set[free] & ~in_declared[free])   # positions in `free`
    classes, n_colours = _colour_classes(gibbs, inc_f, F)
    pos_of = np.full(N + 2, -1)
    pos_of[free] = np.arange(len(free))
    ctx.pos_of = pos_of

    # replicas: n chains, times `temps` temperatures for tempering
    if method == "tempering":
        nT = temps or max(4, int(np.ceil(0.8 * np.sqrt(len(free)))))
    else:
        nT = 1
    R = n * nT
    ladder = np.geomspace(beta_min, 1.0, nT) if nT > 1 else np.ones(1)
    gaps = np.diff(np.log(ladder))                # log-spacings, tuned while adapting
    n_adapt = int(adapt * sweeps) if nT > 1 else 0
    state = np.zeros((R, N + 2), np.int8)
    state[:, ONE] = 1
    state[:, ev_true] = 1                        # evidence
    ctx.state, ctx.R = state, R

    def edges_of(rows):
        out = np.zeros((len(rows), L, k, k), np.int8)
        for l, p in enumerate(relations):
            src = mirror_of.get(p)
            if src:                              # b(y,x) is a(x,y)
                out[:, l] = state[rows, offset[src]:offset[src] + k * k].reshape(-1, k, k).transpose(0, 2, 1)
            else:
                out[:, l] = state[rows, offset[p]:offset[p] + k * k].reshape(-1, k, k)
        return out

    def valid(E):
        if constraint is None:
            return True
        if callable(constraint):
            return bool(constraint(E.astype(bool)))
        return _connected(E.any(0), constraint == "strong")

    if constraint not in (None, "strong", "weak") and not callable(constraint):
        raise ValueError("constraint must be None, 'strong', 'weak' or a function")
    if constraint is not None and L == 0:
        raise ValueError("a constraint needs at least one relation")
    p0 = relations[0] if L else None
    for i in range(R):                           # a random start that satisfies it all
        for _ in range(100):
            state[i, free] = rng.random(len(free)) < 0.5
            for rt in runtimes:
                rt.init(i)
            cyc = None
            if constraint in ("strong", "weak"):  # a cycle (or path) in relation 0
                c = rng.permutation(k)
                c2 = np.roll(c, -1) if constraint == "strong" else c[1:]
                cyc = offset[p0] + c[:len(c2)] * k + c2
                if not is_free[cyc].all():
                    raise ValueError(f"{p0} can't hold the edges of a cycle")
            for l, fa in fixed.items():          # exactly edges[p] edges, at random
                m = edges[relations[l]]
                state[i, fa] = 0
                if cyc is not None and l == 0:
                    if m < len(cyc):
                        raise ValueError(f"{p0} needs at least {len(cyc)} edges to be "
                                         f"{constraint}ly connected, not {m}")
                    state[i, cyc] = 1
                    rest = np.setdiff1d(fa, cyc)
                    state[i, rng.choice(rest, m - len(cyc), replace=False)] = 1
                else:
                    state[i, rng.choice(fa, m, replace=False)] = 1
            E = edges_of([i])[0]
            if cyc is not None and 0 not in fixed and not valid(E):
                rel0 = offset[p0] + np.arange(k * k)
                if in_declared[rel0].any():
                    state[i, rel0[is_free[rel0]]] = 0    # under a Block: exactly the cycle
                state[i, cyc] = 1                # repair
                E = edges_of([i])[0]
            if valid(E) and all(rt.valid(i) for rt in runtimes):
                break
        else:
            raise ValueError("found no starting graph that satisfies the constraint and the structures")
    Sp = np.zeros(R)                             # structures' log weights, per replica
    ctx.Sp, ctx.valid, ctx.edges_of = Sp, valid, edges_of
    for rt in runtimes:
        rt.start()
    Edges = edges_of(np.arange(R))
    ctx.Edges = Edges

    def codes(fs, rows=None, Gf=None):
        """Each factor's bit pattern -- bit j is its j-th atom -- per replica:
        built a column at a time, avoiding a (replicas x factors x M) array.
        Gf: the factors' rows, if already at hand."""
        st = state if rows is None else state[rows]
        if Gf is None:
            Gf = G_all[fs]
        out = st[:, Gf[:, 0]].astype(np.int64)
        for j in range(1, M):
            out |= st[:, Gf[:, j]].astype(np.int64) << j
        return out

    def scores(rows):
        idx = codes(np.arange(F), rows)
        return T[rule_of, idx], (T[rule_of, idx] * W[rule_of]).sum(-1)

    S = scores(np.arange(R))[1]

    def delta(j, rows=None):
        """score(atom free[j] = 1) - score(= 0), per replica (or for some rows)."""
        f, ma = inc_f[j]
        ra = rule_of[f]
        idx = codes(f, rows)                                         # (R, factors)
        return (T[ra, idx | ma] - T[ra, idx & ~ma]) @ W[ra]

    def allowed(i, l, add, rem):
        """May replica i, in relation l, add edge `add` and remove edge `rem`
        (each (u, v) or None)?"""
        E = Edges[i].astype(bool)
        if add is not None:
            E[l][add] = True
        if rem is not None:
            E[l][rem] = False
        if callable(constraint):
            return bool(constraint(E))
        if rem is None:                          # connectivity only fears removals
            return True
        adj = E.any(0)
        u, v = rem
        if adj[u, v]:                            # another relation keeps u -> v
            return True
        if constraint == "weak":
            if adj[v, u]:
                return True
            adj = adj | adj.T
        return _reach(adj, u, v)

    ctx.codes, ctx.S, ctx.delta, ctx.allowed = codes, S, delta, allowed

    # no weighted grounding with two sampled atoms: every update is already exact
    coupled = ((G_all < N).sum(1) > 1) & (W[rule_of] != 0)
    independent = (not coupled.any() and not any(rt.active() for rt in runtimes) and not fixed
                   and constraint is None and not density)
    if independent:                              # one exact sweep, at beta = 1
        sweeps, anneal = 1, 0.0
    ramp = max(1, round(anneal * sweeps))
    if method == "anneal":                       # when density tunes the fields
        cal = (ramp, ramp + int(adapt * (sweeps - ramp)))
    else:
        cal = (0, int(adapt * sweeps))
    if density and cal[1] <= cal[0]:
        raise ValueError("density needs some adaptation sweeps: raise adapt or sweeps")
    top = np.arange(n) * nT + nT - 1             # the beta = 1 replicas
    tried, accepted = np.zeros(max(nT - 1, 1)), np.zeros(max(nT - 1, 1))
    for t in range(sweeps):
        if method == "anneal":
            beta_t = np.full(R, min(1.0, (t + 1) / ramp) if anneal > 0 else 1.0)
        else:
            beta_t = np.tile(ladder, n)
        for ci in rng.permutation(len(classes)):  # Gibbs moves, a colour at a time
            js, Fc, mc, starts = classes[ci]
            atoms_c = free[js]
            rc = rule_of[Fc]
            idx = codes(Fc)                                            # (R, factors)
            d = np.add.reduceat((T[rc, idx | mc] - T[rc, idx & ~mc]) * W[rc], starts, axis=1)
            new = (rng.random(d.shape) < _sigmoid(beta_t[:, None] * (d + h[atoms_c]))).astype(np.int8)
            old = state[:, atoms_c]
            ch = new != old
            if not ch.any():
                continue
            sh = shown[atoms_c]
            if constraint is not None and (sh[:, 0] >= 0).any():
                for i, c in zip(*np.nonzero(ch & (sh[:, 0] >= 0)[None])):   # checked in turn
                    l, u, v = sh[c]
                    if allowed(i, l, (u, v) if new[i, c] else None, None if new[i, c] else (u, v)):
                        Edges[i, l, u, v] = new[i, c]
                    else:
                        ch[i, c] = False
            S += np.where(ch, np.where(new == 1, d, -d), 0.0).sum(1)
            state[:, atoms_c] = np.where(ch, new, old)
            for c in np.flatnonzero(sh[:, 0] >= 0):
                l, u, v = sh[c]
                Edges[:, l, u, v] = state[:, atoms_c[c]]
        for rt in runtimes:                      # each structure's own moves
            rt.sweep(beta_t)
        for l, fa in fixed.items():              # swap moves: exchange two atoms
            for x, y in rng.integers(len(fa), size=(len(fa), 2)):
                if x == y:
                    continue
                a, b = fa[x], fa[y]
                old = state[:, a].copy()
                ch = old != state[:, b]
                if not ch.any():
                    continue
                da = delta(pos_of[a])
                state[ch, a] = 1 - old[ch]       # a flipped; then b, given that
                db = delta(pos_of[b])
                gain = np.where(old == 0, da, -da) + np.where(old == 1, db, -db)
                fgain = np.where(old == 0, h[a] - h[b], h[b] - h[a])
                ok = ch & (np.log(rng.random(R)) < beta_t * (gain + fgain))
                ea, eb = tuple(shown[a][1:]), tuple(shown[b][1:])
                if constraint is not None:
                    for i in np.flatnonzero(ok):
                        add, rem = (ea, eb) if old[i] == 0 else (eb, ea)
                        if not allowed(i, l, add, rem):
                            ok[i] = False
                back = ch & ~ok
                state[back, a] = old[back]
                state[ok, b] = old[ok]
                S[ok] += gain[ok]
                Edges[ok, l, ea[0], ea[1]] = 1 - old[ok]
                Edges[ok, l, eb[0], eb[1]] = old[ok]
        if method == "tempering":                # swap neighbouring temperatures
            if t == n_adapt:                      # report post-adaptation swaps only
                tried[:], accepted[:] = 0, 0
            St = S + state.astype(float) @ h + Sp  # rules + fields + blocks
            for c in range(n):
                for s in range(t % 2, nT - 1, 2):
                    i, i2 = c * nT + s, c * nT + s + 1
                    la = (ladder[s] - ladder[s + 1]) * (St[i2] - St[i])
                    tried[s] += 1
                    if la >= 0 or rng.random() < np.exp(la):
                        accepted[s] += 1
                        for arr in (state, S, St, Sp, Edges):
                            arr[[i, i2]] = arr[[i2, i]]
                        for rt in runtimes:      # bookkeeping that follows the states
                            rt.swap(i, i2)
            if t < n_adapt and (t + 1) % 10 == 0:  # widen easy gaps, narrow hard ones
                A = accepted / np.maximum(tried, 1)
                gaps *= np.exp(A - A.mean())
                gaps *= -np.log(beta_min) / gaps.sum()
                ladder = beta_min * np.exp(np.concatenate([[0.0], np.cumsum(gaps)]))
                ladder[-1] = 1.0
                tried[:], accepted[:] = 0, 0
        if cal[0] <= t < cal[1]:                  # tune fields toward density
            step = 1.0 / (1.0 + 0.05 * (t - cal[0]))
            for p, q in density.items():
                cur = np.clip(state[np.ix_(top, field_atoms[p])].mean(), 1e-3, 1 - 1e-3)
                mu[p] += step * (np.log(q / (1 - q)) - np.log(cur / (1 - cur)))
                h[field_atoms[p]] = mu[p]

    if method == "tempering" and tried.sum() and (accepted / np.maximum(tried, 1)).min() < 0.05:
        warnings.warn(f"some neighbouring temperatures rarely swap (rates "
                      f"{(accepted / np.maximum(tried, 1)).round(2).tolist()}); try more temps")
    rows = top
    truth, final = scores(rows)
    final = final + state[rows].astype(float) @ h + Sp[rows]
    graphs = []
    for c, i in enumerate(rows):
        G = nx.MultiDiGraph()
        G.add_nodes_from(range(k))
        for u, v, l in sorted(map(tuple, np.argwhere(Edges[i].transpose(1, 2, 0)))):
            G.add_edge(int(u), int(v), action=int(l))
        hidden = {}
        for p in sorted(sig):
            if p in relations or (Hlat is not None and p in evidence and "n" not in sig[p]):
                continue                         # shown, or part of the latent graph
            vals = state[i, offset[p]:offset[p] + int(np.prod(dims[p]))].astype(bool).reshape(dims[p])
            if sig[p] == ("n",):
                for x in range(k):
                    G.nodes[x][p] = bool(vals[x])
            elif sig[p] == ("n", "l"):
                for x in range(k):
                    G.nodes[x][p] = np.flatnonzero(vals[x]).tolist()
            else:
                hidden[p] = vals
        G.graph["relations"] = {p: l for l, p in enumerate(relations)}
        G.graph["mirrors"] = {b: a for b, a in mirror_of.items() if b in relations and a in relations}
        G.graph["predicates"] = {p: tuple({"n": "node", "l": "latent"}[t] for t in sig[p])
                                 for p in sorted(sig)}
        G.graph["hidden"] = hidden
        if m_lat:
            G.graph["latent"] = Hlat if Hlat is not None else m_lat
        false = np.bincount(rule_of, weights=1 - truth[c], minlength=len(rules))
        G.graph["mln"] = {"score": float(final[c]),
                          "false_groundings": {r.strip(): int(f) for r, f, kd
                                               in zip(rules, false, kind) if kd in ("rule", "fixed")}}
        if mu:
            G.graph["mln"]["field"] = {p: round(float(v), 4) for p, v in mu.items()}
        G.graph["mln"]["colours"] = n_colours
        for rt in runtimes:
            G.graph["mln"].update(rt.report())
        G.graph["mln"]["blocks"]["clusters too big"] = auto_skipped
        if independent:
            G.graph["mln"]["independent"] = True     # sampled exactly in one sweep
        G.graph["mln"]["groundings"] = {r.strip(): int(c) for r, c, kd in
                                        zip(rules, np.bincount(rule_of, minlength=len(rules)), kind)
                                        if kd in ("rule", "fixed")}
        G.graph["mln"]["sampled atoms"] = {p: int(is_free[offset[p]:offset[p] + int(np.prod(dims[p]))].sum())
                                           for p in sorted(sig)}
        if method == "tempering":
            G.graph["mln"]["swap_rate"] = (accepted / np.maximum(tried, 1)).round(3).tolist()
            G.graph["mln"]["betas"] = ladder.round(4).tolist()
        graphs.append(G)
    return graphs[0] if n == 1 else graphs