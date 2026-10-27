"""Automata: walks on graphs that emit some of their labels.

FSA and PDA live here; grammars (PCFG, RegularGrammar) stay in grammars.py.
Both kinds are Sequentials, so anything that only needs `generate`,
`continuations` and `vocab` takes either."""

from collections import defaultdict
from dataclasses import dataclass

import networkx as nx
import numpy as np
import string

from grammars import EOS, _TOL, Sequential, _pick
import mln

@dataclass(frozen=True, order=True)
class Edge:
    """A hold-out unit for an FSA: Edge(u, v, a) is the one transition u -a-> v,
    and Edge(u, v) every edge u -> v whatever its action. Use it in forbid,
    end_with and reverse like a (node, action) pair. A class of its own rather
    than a tuple, so it can't be mistaken for one."""
    u: object
    v: object
    action: object = None


def _ordered(labels):
    """Sorted when the labels are comparable (ints), else in first-seen order."""
    labels = list(dict.fromkeys(labels))
    try:
        return sorted(labels)
    except TypeError:
        return labels


def _node_list(x, idx):
    """A single node label, or an iterable of them, as a list."""
    try:
        if x in idx:
            return [x]
    except TypeError:                 # unhashable, so a list or set of nodes
        pass
    return list(x)


def _listing(cands, nt):
    """(tokens, probs) for each distinct emission among (mass, emission)
    pairs. probs[k] is the distribution of token k given tokens < k under the
    process that draws an emission by mass, ending with EOS."""
    tot = defaultdict(float)
    for m, e in cands:
        if m > 0.0:
            tot[tuple(e)] += m
    out = []
    for e in tot:
        rows = []
        for t in range(len(e) + 1):
            v = np.zeros(nt)
            for e2, m in tot.items():
                if e2[:t] == e[:t]:
                    v[e2[t] if t < len(e2) else EOS] += m
            rows.append(v / v.sum())
        out.append((list(e), rows))
    return out


class ObservedState:
    """Exact filtering for a walk that shows only some of its labels.

    The grammar supplies start() and options(config, steps), each a list of
    (prob, emission, next_config). An emission is the tuple of tokens a step
    shows; halting is (stop, (EOS,), None). The state keeps every way the
    current step could still go, weighted by its probability given the tokens
    so far, so continuations are exact whatever is hidden.

    Every step shows at least one token, which keeps the number of steps
    observable. With everything shown there is a single candidate; with
    hidden labels there may be several, and with a hidden stack their number
    can grow quickly."""

    def __init__(self, g):
        self.g = g
        self.tokens = []
        self.steps = -1                  # the start is step -1
        self.log_scale = 0.0
        self.viable = True
        self.ended = False
        self._load(g.start())

    def _load(self, opts):
        acc = defaultdict(float)
        for p, e, c in opts:
            if p > 0.0:
                acc[(e, c)] += p
        tot = sum(acc.values())
        self.pending = [(p / tot, e, c) for (e, c), p in acc.items()] if tot > 0 else []
        self.k = 0                       # tokens of the current step seen so far
        if not self.pending:
            self.viable = False
        elif all(not e for _, e, _ in self.pending):   # a start that shows nothing
            self._commit()

    def _commit(self):
        """The current step is fully shown: its outcomes become the belief."""
        belief = defaultdict(float)
        for p, _, c in self.pending:
            belief[c] += p
        self.steps += 1
        self._load([(b * p, e, c2) for c, b in belief.items()
                    for p, e, c2 in self.g.options(c, self.steps)])

    def push(self, token):
        self.tokens.append(token)
        if not self.viable or self.ended:
            self.viable = False
            return False
        k = self.k
        keep = [(p, e, c) for p, e, c in self.pending if e[k] == token]
        tot = sum(p for p, _, _ in keep)
        if tot <= 0.0:
            self.viable, self.pending = False, []
            return False
        self.log_scale += float(np.log(tot))
        self.pending = [(p / tot, e, c) for p, e, c in keep]
        self.k = k + 1
        if token == EOS:
            self.ended = True
        elif all(len(e) == self.k for _, e, _ in self.pending):
            self._commit()
        return True

    def continuations(self):
        """(tokens, probs): the exact next-token distribution, token 0 = EOS."""
        if not self.viable or self.ended:
            return [], []
        acc = defaultdict(float)
        for p, e, _ in self.pending:
            acc[e[self.k]] += p
        toks = sorted(t for t in acc if acc[t] > 0.0)
        return toks, [acc[t] for t in toks]

    @property
    def _stop(self):
        if not self.viable or self.ended:
            return 0.0
        return sum(p for p, e, _ in self.pending if e[self.k] == EOS)

    @property
    def prefix_logprob(self):
        return self.log_scale + float(np.log(self.g.p_finite)) if self.viable else -np.inf

    @property
    def sentence_logprob(self):
        s = self._stop
        return self.log_scale + float(np.log(s * self.g.p_finite)) if s > 0 else -np.inf

    @property
    def complete(self):
        return self._stop > 0.0

    def best_tree(self):
        """The walk as labels. Only defined when every label is shown, since
        then there is exactly one parse."""
        if not self.complete:
            raise ValueError("the tokens so far are not a complete walk")
        if not self.g.fully_shown:
            raise NotImplementedError("recovering the hidden labels is not implemented")
        return self.g.decode(self.tokens)

    def tree(self, rng=None):
        return self.best_tree()


class Automaton(Sequential):
    """A walk on a graph that shows some of its labels: the first node (if
    nodes are shown), then for each step the shown kinds, in KINDS order.

    A subclass provides KINDS, vocab, the token tables in _token_of, p_finite,
    and

        start()                  [(prob, emission, config)] to begin
        options(config, steps)   [(prob, emission, next config)] for one step,
                                 halting as (prob, (EOS,), None)
        transitions()            [(tokens, probs)], one trial per transition

    and ObservedState does the rest: generation, exact continuations and
    log probabilities with any labels hidden."""

    State = ObservedState

    def _set_show(self, show):
        show = frozenset((show,) if isinstance(show, str) else show)
        if not show or not show <= set(self.KINDS):
            raise ValueError(f"show must be a non-empty subset of {self.KINDS}")
        self.show = show
        self._step_kinds = [k for k in self.KINDS if k in show]
        self._start_kinds = ["node"] if "node" in show else []

    @property
    def fully_shown(self):
        return len(self.show) == len(self.KINDS)

    def encode(self, walk, strict=True):
        """Labels to tokens. Position decides which kind each entry is.
        Unknown labels raise, or with strict=False become -1, which never
        matches."""
        out, s = [], len(self._start_kinds)
        for k, x in enumerate(walk):
            kind = (self._start_kinds[k] if k < s
                    else self._step_kinds[(k - s) % len(self._step_kinds)])
            table = self._token_of[kind]
            if x in table:
                out.append(table[x])
            elif strict:
                raise KeyError(f"unknown {kind} {x!r} at position {k}")
            else:
                out.append(-1)
        return out

    def decode(self, tokens):
        return [self.vocab[t] for t in tokens if t != EOS]

    def _emit(self, **tok):
        return tuple(int(tok[k]) for k in self._step_kinds)


class FSA(Automaton):
    """Walks on a graph whose edges carry actions (attribute `action`, default
    the target node). Emits node, action, node, action, ..., node.

        g = FSA(grid(4), length=12)             # nodes 0..15, actions 0..3
        toks, probs = g.generate(rng, forbid={(5, 2), (10, 0)})
        g.decode(toks)                          # [6, 0, 2, 2, 3, 1, 7, ...]
        g.pairs(toks)                           # [(6, 0), (2, 2), (3, 1), ...]

    show picks which labels the walk shows: ("node", "action") by default,
    ("action",) for the action sequence alone, ("node",) for the node
    sequence alone. Hidden labels are summed over exactly, so continuations
    and log probabilities stay exact. Token ids do not depend on show.

    Node and action labels are used as given -- with `grid` they are ints --
    and are never converted to strings. Nodes and actions may share values
    (node 2, action 2), since a label's position in the walk says which it is.
    Tokens are separate: 0 is EOS, 1..A the actions in sorted order, then the
    nodes in graph order. Pass tokens to `state`; use `encode` to turn a walk of
    labels into tokens.

    Termination, as for RegularGrammar: length=n takes exactly n actions,
    mean_length=k halts at rate 1/k, and with neither the sinks halt (falling
    back to mean_length=10 if the graph has none). The mode is fixed by the
    full graph, so `without` never changes it, and the vocabulary is always the
    full graph's, so token ids agree between a grammar and its restrictions.

    forbid is a set of (node, action) pairs never to be taken. With
    condition=False the other actions at that node are renormalised, as if the
    edge did not exist. With condition=True the walk is the original one
    conditioned on never taking a forbidden pair -- exactly what rejection
    sampling would give, which also shifts probability at *earlier* steps away
    from nodes where a forbidden action was likely.

    backtrack=False forbids stepping straight back: arriving at v from u, the
    edges from v to u are dropped and the rest renormalised -- unless nothing
    else is left, at a dead end, when the walk may turn back. Only edges are
    dropped, so with slip an action that sometimes lands on u stays available
    with its other outcomes (and so becomes a little less likely). The
    walk is then a Markov chain on (node, node just left), and length,
    mean_length, end_at and condition apply to that chain exactly, so the
    bridge plans around backtracks rather than being forced into them. The
    forced last step of generate(end_with=...) is exempt.
    """

    KINDS = ("action", "node")           # the order within a step

    def __init__(self, graph, init=None, length=None, mean_length=None,
                 forbid=(), condition=False, action="action", weights="weight",
                 end_at=None, show=("node", "action"), backtrack=True, _mode=None):
        if length is not None and mean_length is not None:
            raise ValueError("give length or mean_length, not both")
        self.graph, self.init, self.condition = graph, init, condition
        self.backtrack = backtrack
        self.end_at = end_at
        self._action, self._weights = action, weights

        nodes = list(graph)
        idx = {s: i for i, s in enumerate(nodes)}
        n = len(nodes)
        edges = []                                   # (i, j, action, weight)
        for s, t, d in graph.edges(data=True):
            w = float(d.get(weights, 1.0))
            edges.append((idx[s], idx[t], d.get(action, t), w))
            if not graph.is_directed():
                edges.append((idx[t], idx[s], d.get(action, s), w))

        # vocabulary from the *full* graph, so restrictions share token ids
        acts = _ordered(e[2] for e in edges)
        A = len(acts)
        self.actions, self.states, self.nt_name = acts, nodes, nodes
        self.vocab = [None] + acts + nodes
        self.action_tok = np.arange(1, A + 1)
        self.node_tok = np.arange(A + 1, A + n + 1)
        self.token_of_action = {a: k + 1 for k, a in enumerate(acts)}
        self.token_of_node = {s: A + 1 + i for i, s in enumerate(nodes)}
        self._token_of = {"action": self.token_of_action, "node": self.token_of_node}
        self._set_show(show)
        self.node_of = {int(t): k for k, t in enumerate(self.node_tok)}
        aix = {a: k for k, a in enumerate(acts)}
        self._edges = [(i, j, a) for i, j, a, _ in edges]
        self._raw = edges                            # with weights, never restricted
        self._node_index = idx
        if end_at is not None and end_at not in idx:
            raise KeyError(f"unknown node {end_at!r}")
        dest = defaultdict(set)
        for i, j, a, _ in edges:
            dest[(i, a)].add(j)
        # does (node, action) always determine the next node?
        self.deterministic = all(len(v) == 1 for v in dest.values())
        self.forbid = frozenset(self._as_pairs(forbid))

        # raw probabilities from the full graph, then drop forbidden edges
        tot = np.zeros(n)
        for i, _, _, w in edges:
            tot[i] += w
        if _mode is None:                            # decide on the full graph
            if length is not None:
                _mode = ("length", int(length))
            elif mean_length is not None:
                _mode = ("mean", float(mean_length))
            elif np.any(tot <= 0):
                _mode = ("sink", None)
            else:
                _mode = ("mean", 10.0)
        self._mode = _mode
        self.length = _mode[1] if _mode[0] == "length" else None
        self.mean_length = _mode[1] if _mode[0] == "mean" else None
        if self.length is not None and self.length < 0:
            raise ValueError("length must be non-negative")
        if self.mean_length is not None and self.mean_length < 1:
            raise ValueError("mean_length must be at least 1")

        keep = [(i, j, a, w / tot[i]) for i, j, a, w in edges
                if (nodes[i], a) not in self.forbid
                and Edge(nodes[i], nodes[j]) not in self.forbid
                and Edge(nodes[i], nodes[j], a) not in self.forbid]
        if not condition:                            # renormalise what is left
            left = np.zeros(n)
            for i, _, _, q in keep:
                left[i] += q
            keep = [(i, j, a, q / left[i]) for i, j, a, q in keep]
        out_j = [[] for _ in range(n)]
        out_a = [[] for _ in range(n)]
        out_q = [[] for _ in range(n)]
        A0 = np.zeros((n, n))
        for i, j, a, q in keep:
            out_j[i].append(j); out_a[i].append(aix[a]); out_q[i].append(q)
            A0[i, j] += q
        self.out_j = [np.array(x, int) for x in out_j]
        self.out_a = [np.array(x, int) for x in out_a]
        self.out_q = [np.array(x, float) for x in out_q]
        self.n_actions = A

        # The walk's state is a config (node, node just left), the latter -1 at
        # the start, or always when backtracking is allowed (then the configs
        # are just the nodes, in order). The bridge below runs over configs.
        cfgs = [(i, -1) for i in range(n)]
        if not backtrack:
            cfgs += sorted({(int(j), i) for i in range(n) for j in self.out_j[i]})
        sid = {c: k for k, c in enumerate(cfgs)}
        N = len(cfgs)
        self._cfg_index = sid
        self._cfg_q, self._cfg_next = [], []
        T = np.zeros((N, N))
        for k, (i, came_from) in enumerate(cfgs):
            q = self.out_q[i]
            if came_from >= 0:
                q = self._no_backtrack(i, came_from, q)
            nxt = np.array([sid[(int(j), -1 if backtrack else i)] for j in self.out_j[i]], int)
            np.add.at(T[k], nxt, q)
            self._cfg_q.append(q)
            self._cfg_next.append(nxt)
        node_of_cfg = np.array([i for i, _ in cfgs], int)

        pi = np.zeros(n)
        if init is None:
            w = np.array([float(graph.nodes[s].get("start", 0.0)) for s in nodes])
            pi = w if w.sum() > 0 else np.ones(n)
        elif isinstance(init, dict):
            for s, w in init.items():
                pi[idx[s]] = float(w)
        else:
            for s in _node_list(init, idx):
                pi[idx[s]] = 1.0
        pi /= pi.sum()

        # Which nodes halt for want of anything to do? Locally, a node whose
        # actions were all held out has nowhere left to go, so it halts. Under
        # conditioning the original walk still runs there: it halts at its usual
        # rate and otherwise takes a held-out action, and that mass is lost --
        # so only the full graph's sinks halt outright.
        sinks = tot <= 0
        dead = (sinks if condition else (A0.sum(1) <= _TOL))[node_of_cfg]
        if end_at is not None and _mode[0] == "sink":
            raise ValueError("ending at a node needs length or mean_length")
        at_end = (node_of_cfg == idx[end_at]) if end_at is not None else None
        if self.length is not None:
            Z = np.ones((self.length + 1, N))        # Z[k, c] = P(k more steps)
            if end_at is not None:                   # ... and then be at end_at
                Z[0] = at_end.astype(float)
            for k in range(1, self.length + 1):
                Z[k] = T @ Z[k - 1]
            self.Z = Z
            z = Z[-1]
        else:
            h = (np.full(N, 1.0 / self.mean_length) if self.mean_length is not None
                 else np.zeros(N))
            h[dead] = 1.0
            M = (1.0 - h)[:, None] * T
            # halting only counts at end_at, if given: walks that halt anywhere
            # else are conditioned away
            h_end = h.copy()
            if end_at is not None:
                h_end[~at_end] = 0.0
            can = h_end > 0
            while True:
                nxt = can | ((M[:, can].sum(1) > 0) if can.any() else False)
                if np.array_equal(nxt, can):
                    break
                can = nxt
            z = np.zeros(N)
            if can.any():
                sub = np.where(can)[0]
                z[sub] = np.linalg.solve(np.eye(len(sub)) - M[np.ix_(sub, sub)], h_end[sub])
            z = np.clip(z, 0.0, 1.0)
            self.h, self.h_end, self.z = h, h_end, z
        z = z[:n]                                    # the start configs (i, -1)
        self.p_finite = float(pi @ z)
        if self.p_finite <= _TOL:
            raise ValueError("no walk can finish under these restrictions")
        self.pi = pi * z / self.p_finite

    # -- what a walk shows, step by step ----------------------------------------

    def start(self):
        show = "node" in self.show
        return [(float(self.pi[i]), (int(self.node_tok[i]),) if show else (), (int(i), -1))
                for i in np.nonzero(self.pi)[0]]

    def options(self, config, steps):
        i, _ = config
        w, stop = self._edge_weights(config, steps)
        prev = -1 if self.backtrack else int(i)
        out = [(stop, (EOS,), None)] if stop > 0.0 else []
        for wk, a, j in zip(w, self.out_a[i], self.out_j[i]):
            out.append((float(wk), self._emit(action=a + 1, node=self.node_tok[j]), (int(j), prev)))
        return out

    def _no_backtrack(self, i, u, q):
        """Node i's edge probabilities q after arriving from u: the edges back
        to u dropped and the rest scaled up to the same total -- unless no
        other edge is left, a dead end, when q stands."""
        back = self.out_j[i] == u
        if not back.any() or q[~back].sum() <= 0.0:
            return q
        return np.where(back, 0.0, q) * (q.sum() / q[~back].sum())

    def transitions(self):
        """One trial per edge, showing what a one-step walk shows (the grammar
        itself when length=1), honouring forbid. Edges that look the same
        under `show` are one trial."""
        g = self if self.length == 1 else self._restricted((), mode=("length", 1))
        cands = [(p0 * p1, e0 + e1)
                 for p0, e0, i in g.start()
                 for p1, e1, _ in g.options(i, 0) if e1 != (EOS,)]
        return _listing(cands, len(self.vocab))

    # -- labels <-> tokens ----------------------------------------------------

    def pairs(self, tokens):
        """The (node, action) pairs a token sequence takes, as labels."""
        if not self.fully_shown:
            raise ValueError("pairs needs both nodes and actions shown")
        return [(self.vocab[tokens[k - 1]], self.vocab[tokens[k]])
                for k in range(1, len(tokens), 2) if tokens[k] != EOS]

    def edge_actions(self, u, v):
        """The actions labelling edges u -> v, e.g. to hold out an edge:
        forbid={(u, a) for a in g.edge_actions(u, v)}."""
        i, j = self._node_index[u], self._node_index[v]
        return _ordered(a for ii, jj, a in self._edges if ii == i and jj == j)

    def reverse(self, pair):
        """The units that undo `pair`. For a (node, action) pair (u, a): the
        pairs (v, b) for each node v that a can take u to, and each action b
        with an edge v -> u, in the full graph -- with slip (or any
        nondeterminism) this can be several pairs per destination. For
        Edge(u, v, a) or Edge(u, v): {Edge(v, u)}, every edge back whatever its
        action, if the full graph has one, else nothing. To hold out both
        directions: forbid={pair} | g.reverse(pair)."""
        unit, = self._as_pairs([pair])
        if isinstance(unit, Edge):
            i, j = self._node_index[unit.u], self._node_index[unit.v]
            return {Edge(unit.v, unit.u)} if any(ii == j and jj == i
                                                 for ii, jj, _ in self._edges) else set()
        u, a = unit
        i = self._node_index[u]
        dests = {j for ii, j, aa in self._edges if ii == i and aa == a}
        if not dests:
            raise KeyError(f"no action {a!r} at node {u!r}")
        return {(self.states[j], b) for j in dests
                for b in self.edge_actions(self.states[j], u)}

    # -- hold-outs --------------------------------------------------------------

    def generate(self, rng=None, tree=False, max_len=10_000, forbid=(),
                 end_with=None):
        """Sample a walk, as for any grammar. `forbid` holds out (node, action)
        pairs and Edges for this call only.

        end_with=(node, action) makes that pair the last one taken: the walk is
        conditioned to be at `node` with one action to go (exactly, not by
        rejection), takes `action`, lands where the full graph sends it, and
        stops. Pass a set or list of pairs to pick one uniformly. The pair is
        not added to `forbid`, so hold it out there if it must not also appear
        earlier -- leave it out to make a control sequence ending on a seen pair.
        end_with=Edge(u, v, a) takes that transition: at u, action a, landing
        on v (Edge(u, v) chooses among u's actions to v by weight). The probs
        row for v is still where the full graph sends a from u, as the truth
        to score against.

        probs[k] is always the distribution token k was drawn from: the bridge
        for the prefix, the forced last step (whatever mass the prefix had on
        stopping), the full graph's transition for the last node, then EOS.
        With actions hidden, the forced step's stopping mass goes to the
        destination distribution instead. In length mode the walk still has
        exactly `length` actions in total."""
        rng = np.random.default_rng() if rng is None else rng
        if end_with is None:
            g = self._restricted(forbid) if forbid else self
            return Sequential.generate(g, rng, tree, max_len)

        s_end, a_end, v_end = self._choose_end(end_with, rng)
        kind, val = self._mode
        if kind == "length":
            if val < 1:
                raise ValueError("a walk ending on a pair needs length >= 1")
            pre_mode = ("length", val - 1)
        elif kind == "mean":
            pre_mode = self._mode
        else:
            raise ValueError("end_with needs length or mean_length")
        pre = self._restricted(forbid, end_at=s_end, mode=pre_mode)
        toks, probs = Sequential.generate(pre, rng, False, max_len)

        last = probs[-1].copy()                  # the prefix stopping becomes
        stop, last[EOS] = last[EOS], 0.0         # the forced last step
        at = self.token_of_action[a_end]
        dest = self._raw_dest(s_end, a_end)
        nt = (self.token_of_node[v_end] if v_end is not None
              else int(_pick(rng, np.arange(len(dest)), dest)))
        eos = np.zeros(len(self.vocab))
        eos[EOS] = 1.0
        if "action" in self.show:
            last[at] += stop
            rows, extra = [last], [at]
            if "node" in self.show:
                rows.append(dest)
                extra.append(nt)
        else:
            last += stop * dest
            rows, extra = [last], [nt]
        toks = toks + extra
        probs = probs[:-1] + rows + [eos]
        return (toks, probs, self.decode(toks)) if tree else (toks, probs)

    def sample_holdout(self, rng, unit="pair"):
        """A random unit to hold out. unit="edge": Edge(u, v), every edge
        u -> v whatever its action, for a transition u -a-> v drawn uniformly
        among those left (so node pairs joined by more actions come up more
        often). unit="pair": a (node, action) pair, a node uniformly among
        those with an action left (so never a sink), then one of its actions
        uniformly."""
        if unit == "edge":
            es = list(dict.fromkeys((i, int(a), int(j)) for i in range(len(self.states))
                                    for a, j in zip(self.out_a[i], self.out_j[i])))
            if not es:
                raise ValueError("no edge to hold out")
            i, _, j = es[rng.integers(len(es))]
            return Edge(self.states[i], self.states[j])
        if unit != "pair":
            raise ValueError(f"unit must be 'pair' or 'edge', not {unit!r}")
        nodes = [i for i in range(len(self.states)) if len(self.out_a[i])]
        if not nodes:
            raise ValueError("no node has an action to hold out")
        i = nodes[rng.integers(len(nodes))]                  # node *index*
        acts = list(dict.fromkeys(self.out_a[i]))            # each action once
        a = acts[rng.integers(len(acts))]                    # action index
        return (self.states[i], self.actions[a])             # as *labels*

    def can_end_with(self, unit, forbid=()):
        """Whether generate(forbid=forbid, end_with=unit) has any walk to give,
        for one unit: a (node, action) pair or an Edge. False when the unit has
        no edge, or when length/mean_length leave no room for it."""
        kind, val = self._mode
        if kind == "sink" or (kind == "length" and val < 1):
            return False
        unit, = self._as_pairs([unit])
        if isinstance(unit, Edge):
            i, j = self._node_index[unit.u], self._node_index[unit.v]
            if not any(ii == i and jj == j and (unit.action is None or a == unit.action)
                       for ii, jj, a, _ in self._raw):
                return False
            s = unit.u
        else:
            s = unit[0]
            if not self._raw_dest(*unit).any():
                return False
        pre_mode = ("length", val - 1) if kind == "length" else self._mode
        try:
            self._restricted(forbid, end_at=s, mode=pre_mode)
        except ValueError:
            return False
        return True

    def _choose_end(self, end_with, rng):
        if isinstance(end_with, (set, frozenset, list)):
            pairs = _ordered(self._as_pairs(end_with))
            if not pairs:
                raise ValueError("end_with is empty")
            end_with = pairs[int(rng.integers(len(pairs)))]
        unit, = self._as_pairs([end_with])
        if isinstance(unit, Edge):               # an action u -> v, by weight
            i, j = self._node_index[unit.u], self._node_index[unit.v]
            acts = [(a, w) for ii, jj, a, w in self._raw if ii == i and jj == j
                    and (unit.action is None or a == unit.action)]
            if not acts:
                raise KeyError(f"no edge {unit.u!r} -> {unit.v!r}"
                               + ("" if unit.action is None else f" with action {unit.action!r}"))
            w = np.array([w for _, w in acts], float)
            k = int(_pick(rng, np.arange(len(acts)), w / w.sum()))
            return unit.u, acts[k][0], unit.v
        s_end, a_end = unit
        if not self._raw_dest(s_end, a_end).any():
            raise KeyError(f"no action {a_end!r} at node {s_end!r}")
        return s_end, a_end, None

    def _raw_dest(self, s, a):
        """Where action a from node s goes in the full graph, over the vocab."""
        i = self._node_index[s]
        d = np.zeros(len(self.vocab))
        for ii, j, aa, w in self._raw:
            if ii == i and aa == a:
                d[self.node_tok[j]] += w
        tot = d.sum()
        return d / tot if tot > 0 else d

    def _restricted(self, forbid, end_at=None, mode=None):
        mode = mode or self._mode
        key = (frozenset(self._as_pairs(forbid)), end_at, mode)
        cache = self.__dict__.setdefault("_cache", {})
        if key not in cache:
            cache[key] = FSA(self.graph, init=self.init,
                             forbid=self.forbid | key[0], condition=self.condition,
                             action=self._action, weights=self._weights,
                             end_at=end_at, show=self.show, backtrack=self.backtrack,
                             _mode=mode)
        return cache[key]

    def _as_pairs(self, forbid):
        """Validate held-out (node, action) pairs and Edges."""
        out = set()
        for item in forbid:
            if isinstance(item, Edge):
                for s in (item.u, item.v):
                    if s not in self._node_index:
                        raise KeyError(f"unknown node {s!r}")
                if item.action is not None and item.action not in self.token_of_action:
                    raise KeyError(f"unknown action {item.action!r}")
                out.add(item)
                continue
            try:
                s, a = item
            except (TypeError, ValueError):
                raise ValueError(f"expected a (node, action) pair, got {item!r}")
            if s not in self._node_index:
                raise KeyError(f"unknown node {s!r}")
            if a not in self.token_of_action:
                raise KeyError(f"unknown action {a!r}")
            out.add((s, a))
        return out

    def without(self, pairs):
        """The same grammar with more (node, action) pairs forbidden."""
        return FSA(self.graph, init=self.init, forbid=self.forbid | set(pairs),
                   condition=self.condition, action=self._action,
                   weights=self._weights, end_at=self.end_at, show=self.show,
                   backtrack=self.backtrack, _mode=self._mode)

    # -- one step ---------------------------------------------------------------

    def _edge_weights(self, config, steps):
        """Probability of each allowed edge out of the config's node, and of
        stopping. config is (node index, node just left, or -1)."""
        k = self._cfg_index[config]
        q, j = self._cfg_q[k], self._cfg_next[k]
        if self.length is not None:
            rem = self.length - steps
            if rem <= 0:
                return np.zeros_like(q), 1.0
            return q * self.Z[rem - 1][j] / self.Z[rem][k], 0.0
        zk = self.z[k]
        return (1.0 - self.h[k]) * q * self.z[j] / zk, self.h_end[k] / zk

    def action_dist(self, i, steps):
        w, stop = self._edge_weights((i, -1), steps)
        return np.bincount(self.out_a[i], weights=w, minlength=self.n_actions), stop

    def dest_dist(self, i, action_token, steps):
        w, _ = self._edge_weights((i, -1), steps)
        m = self.out_a[i] == action_token - 1
        d = np.bincount(self.out_j[i][m], weights=w[m], minlength=len(self.pi))
        tot = d.sum()
        return d / tot if tot > 0 else d


class PDA(Automaton):
    """Runs of a pushdown automaton given as a MultiDiGraph. States, actions
    and stack symbols are labels as given (ints, typically). Each edge is a
    transition, with attributes

        action   its label (default: the target state)
        pop      the stack symbol it needs on top, which it pops; or None for
                 an edge that ignores the stack top and pops nothing
        push     the symbols it then pushes, top first: a tuple (or one
                 symbol); None or () pushes nothing
        weight   relative probability among the moves open at (state, top)

    The classic 0^n 1^n machine, with states p, q, r = 0, 1, 2, actions
    0, 1, and 2 for epsilon, and stack symbols Z, A = 0, 1:

        G = nx.MultiDiGraph()
        G.add_edge(0, 0, action=0, pop=0, push=(1, 0))    # 0; Z/AZ
        G.add_edge(0, 0, action=0, pop=1, push=(1, 1))    # 0; A/AA
        G.add_edge(0, 1, action=2)                         # epsilon
        G.add_edge(1, 1, action=1, pop=1, push=())         # 1; A/epsilon
        G.add_edge(1, 2, action=2, pop=0, push=(0,))       # epsilon; Z/Z
        g = PDA(G, init=0, final={2})

    So an edge is the usual X/γ instruction, with pop=None playing the part of
    ε/γ: ε/ε leaves the stack alone and ε/γ pushes γ on top of it.

    Each distinct push is itself a labelled action: push label k means
    pushing the sequence g.pushes[k], so the empty push () has a label too.
    Labels follow `pushes` if given, and any others continue in sorted order.

    A run starts in `init` with the stack (bottom,). A run is stuck when no
    move is open: no edge from its state needs the symbol on top, and none
    ignores the stack (an empty stack leaves no moves). Termination, as for
    FSA:

      length=n        exactly n steps, ending anywhere; runs that would get
                      stuck sooner are conditioned away
      mean_length=k   halt at rate 1/k anywhere; a stuck run halts
      neither         halt in a final state (default: the states with no
                      outgoing edges); a final state with moves left halts
                      with weight `stop` against its edge weights. Runs stuck
                      elsewhere, or that never halt, are conditioned away

    Hold-outs work as for FSA, on groups (state, pop, action) rather than
    (node, action) pairs, since the same action can belong to edges with
    different pop symbols: forbid removes a group's edges (the other moves at
    that state and top are renormalised), and generate(end_with=group) ends
    on it. sample_holdout picks one at random.

    Conditioning is exact. It uses the probability that each stack symbol is
    popped, per pair of states and (with length) at each step, since only the
    states and the timing of pops matter to what can happen next.

    show picks the labels: a run shows its first state if "node" is shown,
    then for each step, in this order,

        top      the edge's pop symbol, or None for an edge that ignores the
                 stack (a top label of its own)
        action   the edge's label
        push     the edge's push label (the empty push for an edge that
                 pushes nothing)
        node     the state arrived at

    Hidden labels are summed over exactly. Labels of different kinds may share
    values; tokens never do: 0 is EOS, then actions, states, top labels (the
    stack symbols, then None if any edge ignores the stack) and push labels,
    so ids do not depend on show. Every
    step shows at least one token, so there are no silent moves.
    """

    KINDS = ("top", "action", "push", "node")

    def __init__(self, graph, init=None, length=None, mean_length=None, bottom=0,
                 final=None, stop=1.0, show=KINDS, pushes=None, action="action",
                 weights="weight", forbid=(), end_at=None, iters=100_000):
        if length is not None and mean_length is not None:
            raise ValueError("give length or mean_length, not both")
        self.length = None if length is None else int(length)
        self.mean_length = None if mean_length is None else float(mean_length)
        if self.length is not None and self.length < 0:
            raise ValueError("length must be non-negative")
        if self.mean_length is not None and self.mean_length < 1:
            raise ValueError("mean_length must be at least 1")
        nodes = list(graph)
        if not nodes:
            raise ValueError("the graph has no nodes")
        idx = {s: i for i, s in enumerate(nodes)}
        n = len(nodes)
        raw = []
        for s, t, d in graph.edges(data=True):
            push = d.get("push")
            push = (() if push is None else tuple(push)
                    if isinstance(push, (tuple, list)) else (push,))
            raw.append((idx[s], idx[t], d.get(action, t), d.get("pop"), push,
                        float(d.get(weights, 1.0))))
        syms = _ordered([bottom] + [x for e in raw for x in
                                    ((e[3],) if e[3] is not None else ()) + e[4]])
        # The dynamics need every move as "X on top: replace it with this", so
        # an edge that ignores the stack becomes one move per symbol X, putting
        # X back under its push. Labels come from the edge as written.
        edges = []        # (i, j, action, X, effective push, weight, pop, push as written)
        for i, j, a, pop, push, w in raw:
            for X in (syms if pop is None else [pop]):
                edges.append((i, j, a, X, push + (X,) if pop is None else push, w, pop, push))

        acts = _ordered(e[2] for e in edges)
        given = [tuple(p) for p in (pushes or ())]
        if len(set(given)) != len(given):
            raise ValueError("pushes lists a sequence twice")
        seqs = given + sorted({e[7] for e in edges} - set(given))
        self.graph, self.init = graph, init
        self._args = dict(bottom=bottom, final=final, stop=stop, show=show,
                          action=action, weights=weights, iters=iters)
        self._raw, self._node_index = raw, idx
        tops = syms + ([None] if any(e[3] is None for e in raw) else [])
        self.push_label = {p: k for k, p in enumerate(seqs)}
        self.actions, self.states, self.symbols, self.pushes = acts, nodes, syms, seqs
        self.nt_name = nodes
        self.vocab = [None] + acts + nodes + tops + list(range(len(seqs)))
        off = np.cumsum([1, len(acts), n, len(tops)])
        self.token_of_action = {x: int(off[0]) + k for k, x in enumerate(acts)}
        self.token_of_node = {x: int(off[1]) + k for k, x in enumerate(nodes)}
        self.token_of_top = {x: int(off[2]) + k for k, x in enumerate(tops)}
        self.token_of_push = {k: int(off[3]) + k for k in range(len(seqs))}
        self.node_tok = np.arange(off[1], off[1] + n)
        self._token_of = {"action": self.token_of_action, "node": self.token_of_node,
                          "top": self.token_of_top, "push": self.token_of_push}
        self._set_show(show)

        # Hold-outs act on groups (state, pop, action): the edges an agent in
        # that state, with that pop symbol on top (None: edges ignoring the
        # stack), would be choosing between when it takes that action. The
        # vocabulary above is the full graph's, so restrictions share tokens.
        self._all_groups = {(nodes[e[0]], e[3], e[2]) for e in raw}
        self.forbid = frozenset(self._as_groups(forbid))
        edges = [e for e in edges if (nodes[e[0]], e[6], e[2]) not in self.forbid]
        self._groups = defaultdict(list)                 # state index -> its groups
        for e in edges:
            if (e[6], e[2]) not in self._groups[e[0]]:
                self._groups[e[0]].append((e[6], e[2]))
        self.end_at = end_at
        if end_at is not None:
            s_end, X_end = end_at
            if s_end not in idx:
                raise KeyError(f"unknown state {s_end!r}")
            if X_end is not None and X_end not in syms:
                raise KeyError(f"unknown stack symbol {X_end!r}")
            if self.length is None and self.mean_length is None:
                raise ValueError("ending at a configuration needs length or mean_length")
            self._end = (idx[s_end], X_end)

        fin = np.zeros(n, bool)
        if self.length is not None or self.mean_length is not None:
            fin[:] = True                          # runs may end anywhere
        elif final is None:
            for i in range(n):
                fin[i] = not any(e[0] == i for e in raw)
        else:
            for s in _node_list(final, idx):
                fin[idx[s]] = True
        self.final = fin

        # the moves open at each (state, top), with halting as one more option
        by = defaultdict(list)
        for i, j, a, X, push, w, pop0, push0 in edges:
            by[(i, X)].append((w, j, self.token_of_action[a], push, self.token_of_top[pop0],
                               self.token_of_push[self.push_label[push0]]))
        self._opts, self._halt = {}, {}
        for i in range(n):
            for X in syms:
                moves = by.get((i, X), [])
                mw = sum(m[0] for m in moves)
                if self.length is not None:        # no halting before the end
                    hw, scale = 0.0, (1.0 / mw if moves else 0.0)
                elif self.mean_length is not None:     # halt at rate 1/k; stuck halts
                    hw = 1.0 / self.mean_length if moves else 1.0
                    scale = (1.0 - hw) / mw if moves else 0.0
                else:                              # halt only in final states
                    hw = (stop if moves else 1.0) if fin[i] else 0.0
                    scale = 1.0 / (hw + mw) if hw + mw > 0 else 0.0
                    hw = hw * scale
                if hw <= 0.0 and not moves:
                    continue
                self._halt[(i, X)] = hw * self._endv((X,))[i]   # halting that counts
                self._opts[(i, X)] = [(w * scale, j, atok, push, ttok, ptok)
                                      for w, j, atok, push, ttok, ptok in moves]

        S = len(syms)
        sx = {X: k for k, X in enumerate(syms)}
        self._sx = sx
        if self.length is not None:
            self._step_tables(n, syms)
        else:
            # M[X][i, j]: with X on top in state i, X is eventually popped and
            # the state is then j. H[X][i]: the run halts, accepted, before X
            # is popped. Least fixpoint, by iteration from zero.
            M, H = np.zeros((S, n, n)), np.zeros((S, n))
            for _ in range(iters):
                M2, H2 = np.zeros_like(M), np.zeros_like(H)
                for (i, X), moves in self._opts.items():
                    x = sx[X]
                    H2[x, i] += self._halt[(i, X)]
                    for p, j, _, push, _, _ in moves:
                        row, h = np.zeros(n), 0.0
                        row[j] = 1.0
                        for Y in push:
                            h += row @ H[sx[Y]]
                            row = row @ M[sx[Y]]
                        M2[x, i] += p * row
                        H2[x, i] += p * h
                delta = max(np.abs(M2 - M).max(), np.abs(H2 - H).max())
                M, H = M2, H2
                if delta < 1e-14:
                    break
            self.M, self.H = M, H
        self._acc_cache = {}

        pi = np.zeros(n)
        if init is None:
            w = np.array([float(graph.nodes[s].get("start", 0.0)) for s in nodes])
            pi = w if w.sum() > 0 else np.ones(n)
        elif isinstance(init, dict):
            for s, w in init.items():
                pi[idx[s]] = float(w)
        else:
            for s in _node_list(init, idx):
                pi[idx[s]] = 1.0
        pi /= pi.sum()
        self.bottom = bottom
        a0 = (self._ahead((bottom,), self.length) if self.length is not None
              else self._acc((bottom,)))
        self.p_finite = float(pi @ a0)
        if self.p_finite <= _TOL:
            raise ValueError("no run is ever accepted" if self.length is None
                             else f"no run can take {self.length} steps")
        self.pi = pi * a0 / self.p_finite

    def _endv(self, stack):
        """Over states: 1 where a run with this stack may end (everywhere,
        unless end_at restricts it to one state and pop symbol)."""
        n = len(self.states)
        if self.end_at is None:
            return np.ones(n)
        v = np.zeros(n)
        s, X = self._end
        if stack and (X is None or stack[0] == X):
            v[s] = 1.0
        return v

    def _step_tables(self, n, syms):
        """For length mode. P[X][m][i, j]: with X on top in state i, X is popped
        at exactly step m, in state j. S[X][m][i]: m steps are taken without
        popping X. Both count only runs that never get stuck, and are filled
        for m = 1..length in order, since a move pushing Y1..Yr pops X when
        the last of them is popped, which happens strictly later."""
        L = self.length
        P = {X: np.zeros((L + 1, n, n)) for X in syms}
        S = {X: np.zeros((L + 1, n)) for X in syms}
        for X in syms:
            S[X][0] = self._endv((X,))
        # for each pushed sequence g, R[g][t][j, k]: starting in j with g on
        # top, all of g is popped at exactly step t, in state k; T[g][t][j]:
        # t steps are taken with some of g still there
        sufs = sorted({push[d:] for moves in self._opts.values()
                       for _, _, _, push, _, _ in moves
                       for d in range(len(push) + 1)}, key=len)
        R = {g: [] for g in sufs}
        T = {g: [] for g in sufs}
        for m in range(1, L + 1):
            t = m - 1
            for g in sufs:
                if not g:
                    R[g].append(np.eye(n) if t == 0 else np.zeros((n, n)))
                    T[g].append(np.zeros(n))
                    continue
                Y, rest = g[0], g[1:]
                r, a = np.zeros((n, n)), S[Y][t].copy()
                for u in range(1, t + 1):
                    r += P[Y][u] @ R[rest][t - u]
                    a += P[Y][u] @ T[rest][t - u]
                R[g].append(r)
                T[g].append(a)
            for (i, X), moves in self._opts.items():
                for p, j, _, push, _, _ in moves:
                    P[X][m][i] += p * R[push][t][j]
                    S[X][m][i] += p * T[push][t][j]
        self.P, self.S = P, S

    def _ahead(self, stack, k):
        """For length mode: P(k more steps can be taken, ending where end_at
        allows | each state, this stack). Each step pops at most one symbol,
        so only the top k + 1 can matter: k for the steps, and the one that
        may be on top at the end."""
        n = len(self.states)
        if k == 0:
            return self._endv(stack)
        if not stack:
            return np.zeros(n)
        key = (k, stack[:k + 1])
        v = self._acc_cache.get(key)
        if v is None:
            X, rest = stack[0], stack[1:]
            v = self.S[X][k].copy()
            for m in range(1, k + 1):
                v += self.P[X][m] @ self._ahead(rest, k - m)
            self._acc_cache[key] = v
        return v

    def _acc(self, stack):
        """P(accepted | each state, this stack), the stack a tuple, top first."""
        v = self._acc_cache.get(stack)
        if v is not None:
            return v
        v = (self.final.astype(float) if self.end_at is None
             else np.zeros(len(self.states)))           # an empty stack ends nowhere
        for d in range(len(stack) - 1, -1, -1):
            suf = stack[d:]
            w = self._acc_cache.get(suf)
            if w is None:
                x = self._sx[stack[d]]
                w = self._acc_cache[suf] = self.H[x] + self.M[x] @ v
            v = w
        return v

    # -- what a run shows, step by step ---------------------------------------

    def start(self):
        show = "node" in self.show
        return [(float(self.pi[i]), (int(self.node_tok[i]),) if show else (),
                 (int(i), (self.bottom,)))
                for i in np.nonzero(self.pi)[0]]

    def options(self, config, steps=0):
        i, stack = config
        if self.length is not None:
            return self._options_len(i, stack, self.length - steps)
        if not stack:                              # emptied: halt, accepted or not
            return [(1.0, (EOS,), None)] if self.final[i] else []
        X, rest = stack[0], stack[1:]
        here = self._acc(stack)[i]
        if here <= 0.0:
            return []
        out = []
        h = self._halt.get((i, X), 0.0)
        if h > 0.0:
            out.append((h / here, (EOS,), None))
        for p, j, atok, push, ttok, ptok in self._opts.get((i, X), ()):
            new = push + rest
            a = self._acc(new)[j]
            if a > 0.0:
                e = self._emit(top=ttok, action=atok, push=ptok, node=self.node_tok[j])
                out.append((p * a / here, e, (j, new)))
        return out

    def _options_len(self, i, stack, rem):
        if rem <= 0:
            return [(1.0, (EOS,), None)]
        if not stack:
            return []
        here = self._ahead(stack, rem)[i]
        if here <= 0.0:
            return []
        out, rest = [], stack[1:]
        for p, j, atok, push, ttok, ptok in self._opts.get((i, stack[0]), ()):
            new = push + rest
            a = self._ahead(new, rem - 1)[j]
            if a > 0.0:
                e = self._emit(top=ttok, action=atok, push=ptok, node=self.node_tok[j])
                out.append((p * a / here, e, (j, new)))
        return out

    # -- hold-outs --------------------------------------------------------------

    def generate(self, rng=None, tree=False, max_len=10_000, forbid=(),
                 end_with=None):
        """Sample a run, as for any grammar. As for FSA.generate, `forbid`
        holds out groups for this call only, and end_with=(state, pop, action)
        makes that group the last step: the run is conditioned to be in
        `state` with `pop` on top and one step to go (exactly, not by
        rejection), takes one of the group's edges by weight in the full
        graph, and stops. Pass a set or list of groups to pick one uniformly.

        probs[k] is the distribution token k was drawn from, except that the
        first token of the forced step gets the prefix's last distribution,
        with its stopping mass moved to that token's distribution under the
        forced step. In length mode the run still has `length` steps."""
        rng = np.random.default_rng() if rng is None else rng
        if end_with is None:
            g = self._restricted(forbid) if forbid else self
            return Sequential.generate(g, rng, tree, max_len)

        s_end, pop, a_end = self._choose_end(end_with, rng)
        pre = self._restricted(forbid, end_at=(s_end, pop), **self._prefix_mode())
        toks, probs = Sequential.generate(pre, rng, False, max_len)

        steps = defaultdict(float)               # the forced step's emissions
        i = self._node_index[s_end]
        tot = sum(w for ii, _, a, p, _, w in self._raw if (ii, p, a) == (i, pop, a_end))
        for ii, j, a, p, push, w in self._raw:
            if (ii, p, a) == (i, pop, a_end):
                e = self._emit(top=self.token_of_top[pop], action=self.token_of_action[a],
                               push=self.token_of_push[self.push_label[push]],
                               node=self.node_tok[j])
                steps[e] += w / tot
        opts = list(steps)
        e = opts[int(_pick(rng, np.arange(len(opts)), [steps[o] for o in opts]))]
        rows = {tuple(t): r for t, r in _listing([(steps[o], o) for o in opts],
                                                 len(self.vocab))}[e]
        last = probs[-1].copy()
        stop, last[EOS] = last[EOS], 0.0
        last += stop * rows[0]
        toks = toks + list(e)
        probs = probs[:-1] + [last] + rows[1:]
        return (toks, probs, self.decode(toks)) if tree else (toks, probs)

    def can_end_with(self, group, forbid=()):
        """Whether generate(forbid=forbid, end_with=group) has any run to give,
        for one (state, pop, action) group."""
        (s, pop, _), = self._as_groups([group])
        try:
            self._restricted(set(forbid), end_at=(s, pop), **self._prefix_mode())
        except ValueError:
            return False
        return True

    def _prefix_mode(self):
        """The termination for the part of a run before a forced last step."""
        if self.length is not None:
            if self.length < 1:
                raise ValueError("a run ending on a group needs length >= 1")
            return dict(length=self.length - 1)
        if self.mean_length is not None:
            return dict(mean_length=self.mean_length)
        raise ValueError("end_with needs length or mean_length")

    def holdout_groups(self):
        """The groups some run can end on while holding them out before then,
        i.e. those generate(forbid=[g], end_with=g) accepts. Stack gating
        rules many out: with the stack only growing in a state, say, its
        bottom symbol is never on top there again."""
        if getattr(self, "_feasible", None) is None:
            mode, ok = self._prefix_mode(), []
            for i in sorted(self._groups):
                for pop, a in self._groups[i]:
                    g = (self.states[i], pop, a)
                    try:
                        self._restricted([g], end_at=(g[0], pop), **mode)
                    except ValueError:
                        continue
                    ok.append(g)
            self._feasible = ok
        return self._feasible

    def sample_holdout(self, rng):
        """A random group to hold out, among holdout_groups(): a state
        uniformly, then one of its groups uniformly."""
        by = defaultdict(list)
        for g in self.holdout_groups():
            by[g[0]].append(g)
        if not by:
            raise ValueError("no group can be held out and then ended on")
        states = list(by)
        s = states[int(rng.integers(len(states)))]
        return by[s][int(rng.integers(len(by[s])))]

    def _choose_end(self, end_with, rng):
        if isinstance(end_with, (set, frozenset, list)):
            groups = _ordered(self._as_groups(end_with))
            if not groups:
                raise ValueError("end_with is empty")
            end_with = groups[int(rng.integers(len(groups)))]
        (group,) = self._as_groups([end_with])
        return group

    def reverse(self, group):
        """The groups that undo `group` = (s, pop, a), as FSA.reverse does for
        pairs: every group at a state t the group's edges reach that has an
        edge back to s, whatever its pop symbol. To hold out a transition in
        both directions: forbid={group} | g.reverse(group)."""
        (s, pop, a), = self._as_groups([group])
        i = self._node_index[s]
        dests = {j for ii, j, aa, p, _, _ in self._raw if (ii, p, aa) == (i, pop, a)}
        return {(self.states[t], p, aa) for t, j, aa, p, _, _ in self._raw
                if t in dests and j == i}

    def _as_groups(self, items):
        """Validate (state, pop, action) groups against the full graph."""
        out = set()
        for item in items:
            try:
                s, pop, a = item
            except (TypeError, ValueError):
                raise ValueError(f"expected a (state, pop, action) group, got {item!r}")
            if (s, pop, a) not in self._all_groups:
                raise KeyError(f"no edge from state {s!r} with pop {pop!r} "
                               f"and action {a!r}")
            out.add((s, pop, a))
        return out

    def _restricted(self, forbid, end_at=None, **mode):
        mode = mode or dict(length=self.length, mean_length=self.mean_length)
        key = (frozenset(self._as_groups(forbid)), end_at, tuple(sorted(mode.items())))
        cache = self.__dict__.setdefault("_cache", {})
        if key not in cache:
            cache[key] = PDA(self.graph, init=self.init, pushes=self.pushes,
                             forbid=self.forbid | key[0], end_at=end_at,
                             **mode, **self._args)
        return cache[key]

    def without(self, groups):
        """The same automaton with more (state, pop, action) groups forbidden."""
        return PDA(self.graph, init=self.init, length=self.length,
                   mean_length=self.mean_length, pushes=self.pushes,
                   forbid=self.forbid | set(groups), end_at=self.end_at, **self._args)

    def transitions(self):
        """One trial per transition: the state if shown, then one step. The
        process: a state uniformly among those with moves, a top uniformly
        among the symbols it has moves for, then a move by weight. There is no
        conditioning on acceptance, so a transition is listed whenever the
        automaton has it. Transitions that look the same under `show` are one
        trial."""
        at = defaultdict(list)
        for (i, X), moves in self._opts.items():
            if moves:
                at[i].append((X, moves))
        cands = []
        for i, tops in at.items():
            e0 = (int(self.node_tok[i]),) if "node" in self.show else ()
            for X, moves in tops:
                tot = sum(m[0] for m in moves)
                for p, j, atok, push, ttok, ptok in moves:
                    e = self._emit(top=ttok, action=atok, push=ptok, node=self.node_tok[j])
                    cands.append((p / tot / len(tops) / len(at), e0 + e))
        return _listing(cands, len(self.vocab))


# ==========================================================================
# graphs
# ==========================================================================

def grid(n, m=None, torus=False, slip=0.0,
         moves=((-1, 0), (1, 0), (0, 1), (0, -1))):
    """An n x m grid as a MultiDiGraph with integer labels throughout.

    Nodes are r * m + c for row r (0 at the top) and column c, each with a
    `pos` attribute (r, c). Actions are indices into `moves`: by default
    0 = N, 1 = S, 2 = E, 3 = W. Off-grid moves are absent unless torus.

    slip > 0 makes it nondeterministic: an action reaches its intended cell with
    probability 1 - slip and each perpendicular neighbour with slip / 2. Slip
    toward the boundary lands on the intended cell instead, so every action's
    weights sum to 1 and the choice among actions stays uniform."""
    m = n if m is None else m
    G = nx.MultiDiGraph()
    for r in range(n):
        for c in range(m):
            G.add_node(r * m + c, pos=(r, c))

    def land(r, c, d):
        rr, cc = r + d[0], c + d[1]
        if torus:
            return (rr % n) * m + (cc % m)
        return rr * m + cc if 0 <= rr < n and 0 <= cc < m else None

    for r in range(n):
        for c in range(m):
            for a, d in enumerate(moves):
                target = land(r, c, d)
                if target is None:
                    continue
                w = defaultdict(float)
                w[target] += 1.0 - slip
                for e in moves:
                    if slip and e[0] * d[0] + e[1] * d[1] == 0 and tuple(e) != tuple(d):
                        side = land(r, c, e)
                        w[side if side is not None else target] += slip / 2
                for t, p in w.items():
                    if p > 0:
                        G.add_edge(r * m + c, t, action=a, weight=p)
    return G


def cycle(k):
    """A directed k-cycle with a single action, 0."""
    return nx.MultiDiGraph([(i, (i + 1) % k, {"action": 0}) for i in range(k)])


def random_relation(k, r=1, degree=2, density=None):

    rels = ['R' + str(i) for i in range(r)]
    properties = [f"inf {R}(x,y) -> {R}(x,y)" for R in rels] # tautology

    mod = mln.MLN(k, properties, degree=degree, density=density, constraint='weak')

    return mod.sample()


def spatial_relation(k, d, commutes=False, total=False, density=None):

    rels = ['R' + str(i) for i in range(d)]
    pairs = [(rel, 'L'+str(i)) for i,rel in enumerate(rels)]

    properties = [mln.Spatial(*pairs, total=total)]

    if commutes:
        for i,p1 in enumerate(pairs):
            for p2 in pairs[i+1:]:
                properties.append(mln.commute(p1[0], p2[0]))
                properties.append(mln.commute(p1[0], p2[1]))
                properties.append(mln.commute(p1[1], p2[0]))
                properties.append(mln.commute(p1[1], p2[1]))


    mod = mln.MLN(k, properties, density=density)

    return mod.sample()


def block_relation(k, n, r=1, density=0.1):

    rels = ['R' + str(i) for i in range(d)]
    
    properties = [mln.categorical('In')] # latent category
    properties += [mln.sbm('In', R) for R in rels]

    mod = mln.MLN(k, properties, latents=n)

    return mod.sample()