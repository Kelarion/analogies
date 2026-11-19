"""
Assigning tokens to a task's symbols with a geometry: groups of symbols that get similar tokens.

    geom = Geometry({'block', 'node'}, categories={'hypernyms', 'frames'}, embedding='input', within=2, between=0.5)
    ICLModel(..., tokens=Vocabulary('words', content=True), geometry=geom)

With a geometry, ICLModel draws each sample's tokens through an Assigner instead of uniformly
from the pool. A Geometry is a frozen dataclass, saved and hashed like any argument; its repr
lists only fields that differ from their defaults, so fields added later leave digests alone.

  group       which symbols should be similar, as a set, the way share_toks names what is
              shared; read off the task's data (data['code'] and data['kinds'], see
              GrammarTask), so nothing here is task-specific:
                a kind        each grammar token of that kind is grouped with its role-mates,
                ('action',    its symbols in the other blocks: where share_toks={'action'}
                 'node', ..)  makes an action the same token in every block, group={'action'}
                              makes it similar tokens
                'block'       each block's own symbols are grouped together
              e.g. {'block'}, {'node', 'action'}, {'action'}, {'block', 'node'}. Symbols shared
              across blocks are left out of these groups. With 'block' and a kind, groups
              overlap: a node's token must be similar to its block-mates' and to its
              role-mates', so it must carry both groups' categories (a learned code with many
              categories per token makes this easy; WordNet, nearly a tree, makes it hard).
              Symbols in no group are drawn uniformly, but kept out of every group's category.
  categories  where similar tokens come from: a set of sources, whose categories are pooled
              and used alike, whatever their source (only the similarity check cares how
              well they work):
                'hypernyms'    WordNet hypernyms (and own synsets) of each word's three most
                               frequent noun and verb senses
                'supersenses'  WordNet's lexicographer files: noun.artifact, verb.motion, ...
                'domains'      WordNet topic domains: law, military, medicine, ...
                'wholes'       what a word is part, member or substance of: wheel -> car
                'frames'       FrameNet frames whose lexical units include the word
                a stored fit   learned codes for the run's model, by method and dimension
                               ('semibmf300', 'bipca300'), fitted on this geometry's embedding
                               (see categories.py): the same Geometry uses each model's own
                               codes, made before the sweep runs if missing
                a path         an .npz of binary codes, with 'tokens' (strings as the tokenizer
                               decodes them, e.g. ' king') and 'codes' (one row per token, one
                               column per category; optionally 'names', one per column)
              (WordNet sources read a word's three most frequent senses, as lowercase lemmas;
              FrameNet matches a word or its base form.) Each group gets a category its tokens
              all carry (with combine=2, two), and no token outside the group carries it, so
              groups are kept apart by construction. Categories covering more than max_share
              of the pool (WordNet's 'entity', say) are never used.
  embedding   the similarity used to check draws and to report them:
                'input'       the model's input embeddings
                'canonical'   the unembeddings, centred and whitened over the vocabulary (Park
                              et al. 2024's canonical representation)
                a path        an .npz with 'tokens' and 'vectors'
              Cosine similarity of vectors centred on the pool's mean, in z units: standard
              deviations of random pool pairs' similarity above their mean.
  within      if given, each group's mean pairwise similarity must be at least this (z)
  between     if given, the mean similarity of pairs of grouped symbols that share no group
              must be within +-this (z)
  beta        how strongly tokens are drawn towards their group-mates (0: uniformly within
              their categories). After a first draw, each grouped symbol's token is resampled
              a few times from the tokens its categories allow, with probability proportional
              to exp(beta * mean similarity to its group-mates, in z); the categories still
              decide what is allowed, so it never breaks them. Helps when categories alone
              give weak groups (single learned latents, overlapping groups).

A draw failing a check is redrawn, with the same categories a few times, then with others;
without checks the categories alone decide. Each sample's groups, categories and achieved
similarities go into ICLModel's metrics['geometry'].

Which embedding: input embeddings are what the model starts from for each token, but in large
models they are nearly orthogonal, and carry spelling as much as meaning; the canonical
representation is more semantic, but tracks shared prefixes. The z units make thresholds
comparable between them (and between models).
"""
import os
from dataclasses import dataclass, fields

import numpy as np

import vocabulary as voc
import categories as stored

EMBEDDINGS = ('input', 'canonical')
SOURCES = ('hypernyms', 'supersenses', 'domains', 'wholes', 'frames')
N_SENSES = 3            # WordNet senses whose hypernyms a token carries
TRIES = 400             # categories tried per group
REDRAWS = 20            # draws tried per category
SWEEPS = 3              # resampling sweeps when beta > 0
RESTARTS = 50           # whole samples tried before giving up


@dataclass(frozen=True, repr=False)
class Geometry:
    """How a sample's tokens are assigned: see the module docstring."""
    group: frozenset = frozenset({'block'})
    categories: frozenset = frozenset({'hypernyms'})
    embedding: str = 'input'
    within: float = None
    between: float = None
    combine: int = 1
    max_share: float = 0.05
    beta: float = 0.0

    def __post_init__(self):
        put = lambda k, v: object.__setattr__(self, k, v)
        if isinstance(self.group, str):
            raise TypeError(f"group is a set, like share_toks: write group={{{self.group!r}}}")
        if not isinstance(self.group, (set, frozenset)) or not all(isinstance(k, str) for k in self.group):
            raise TypeError(f"group is a set of 'block' and kinds, e.g. {{'block'}} or {{'action'}}, not {self.group!r}")
        if not self.group:
            raise ValueError("group=set() groups nothing: for uniform draws, leave ICLModel's geometry None")
        put('group', frozenset(self.group))
        if isinstance(self.categories, str):
            raise TypeError(f"categories is a set of sources: write categories={{{self.categories!r}}}")
        if not isinstance(self.categories, (set, frozenset)) or not self.categories:
            raise TypeError(f"categories is a non-empty set of sources from {SOURCES} or .npz paths, not {self.categories!r}")
        for c in self.categories:
            if not isinstance(c, str):
                raise TypeError(f"categories: {c!r} isn't a source name or path")
            if c not in SOURCES and not c.endswith('.npz') and not stored.is_name(c):
                raise ValueError(f"categories: {c!r} is neither one of {SOURCES}, a stored fit "
                                 f"(<method><dim>, method one of {sorted(stored.METHODS)}, e.g. "
                                 f"'semibmf300'), nor a path to an .npz")
        if any(stored.is_name(c) for c in self.categories) and self.embedding not in stored.EMBEDDINGS:
            raise ValueError(f"stored fits are made on one of {stored.EMBEDDINGS}, not on {self.embedding!r}")
        put('categories', frozenset(self.categories))
        if not isinstance(self.embedding, str) or not (self.embedding in EMBEDDINGS or self.embedding.endswith('.npz')):
            raise ValueError(f"embedding={self.embedding!r}: one of {EMBEDDINGS}, or a path to an .npz of tokens and vectors")
        for k in ('within', 'between'):
            if getattr(self, k) is not None:
                put(k, float(getattr(self, k)))
        if self.between is not None and self.between < 0:
            raise ValueError("between is a bound on |similarity|, so not negative")
        if self.combine not in (1, 2):
            raise ValueError("combine is 1 or 2 (categories a group's tokens share)")
        put('beta', float(self.beta))
        if self.beta < 0:
            raise ValueError("beta >= 0 (0: uniform within categories)")
        put('max_share', float(self.max_share))
        if not 0 < self.max_share <= 1:
            raise ValueError("max_share is a share of the pool, in (0, 1]")

    def __repr__(self):
        """Only fields that differ from their defaults; the group set sorted, since a set's own
        repr changes with the hash seed (and the digest of a run is made from this)."""
        show = lambda v: '{' + ', '.join(repr(k) for k in sorted(v)) + '}' if isinstance(v, frozenset) else repr(v)
        args = [show(self.group)] + [f'{f.name}={show(getattr(self, f.name))}' for f in fields(self)[1:]
                                     if getattr(self, f.name) != f.default]
        return f"{type(self).__name__}({', '.join(args)})"


# -- which symbols form groups ------------------------------------------------------------

def symbol_groups(code, kinds, group):
    """Groups of symbols (arrays, possibly overlapping), from one sample's code (code[b, r]:
    grammar token r's symbol in block b), the grammar tokens' kinds (kinds[0] is the
    delimiter's, None) and a Geometry's group set (see the module docstring)."""
    code = np.asarray(code); nblk, nt = code.shape
    names = list(dict.fromkeys(k for k in kinds[1:]))
    unknown = set(group) - {'block'} - set(names)
    if unknown:
        raise ValueError(f"group={set(group)}: {sorted(unknown)} isn't 'block' or one of this task's kinds {names}")
    blocks_of = {}
    for b in range(nblk):
        for s in code[b, 1:]:
            blocks_of.setdefault(int(s), set()).add(b)
    own = lambda s: len(blocks_of[s]) == 1
    uniq = lambda xs: list(dict.fromkeys(int(x) for x in xs))
    groups = []
    if 'block' in group:
        groups += [[s for s in uniq(code[b, 1:]) if own(s)] for b in range(nblk)]
    for r in range(1, nt):
        if kinds[r] in group:
            groups.append([s for s in uniq(code[:, r]) if own(s)])
    out = []
    for g in groups:
        if len(g) >= 2 and sorted(g) not in [sorted(h.tolist()) for h in out]:
            out.append(np.array(g, int))
    return out


# -- categories -----------------------------------------------------------------------------

def _senses(w, pos, top=N_SENSES):
    """A word's `top` most frequent senses of the given parts of speech, as a lowercase lemma."""
    wn = voc._wordnet(); w = w.lower(); out = []
    for p in pos:
        for s in wn.synsets(w, p):
            if s.instance_hypernyms():
                continue
            q = 'a' if s.pos() == 's' else s.pos()
            ls = [l for l in s.lemmas() if l.name() in {w, wn.morphy(w, q)}]
            if ls:
                out.append((1 + sum(l.count() for l in ls), s))
    return [s for _, s in sorted(out, key=lambda x: -x[0])[:top]]


def _source_categories(source, words):
    """{category name: [indices of the words in it]} for one WordNet or FrameNet source."""
    out = {}
    def add(name, i):
        out.setdefault(name, []).append(i)
    if source == 'frames':
        from nltk.corpus import framenet as fn
        wn = voc._wordnet(); ix = {}
        for i, w in enumerate(words):
            for b in {w.lower()} | {wn.morphy(w.lower(), p) for p in 'nvar'} - {None}:
                ix.setdefault(b, set()).add(i)
        for lu in fn.lus():
            for i in ix.get(lu.name.rsplit('.', 1)[0].lower(), ()):
                if i not in out.get('frame:' + lu.frame.name, []):
                    add('frame:' + lu.frame.name, i)
        return out
    for i, w in enumerate(words):
        if source == 'hypernyms':
            names = {'hyp:' + a.name() for s in _senses(w, 'nv') for p in s.hypernym_paths() for a in p}
        else:
            ss = _senses(w, 'nvar')
            if source == 'supersenses':
                names = {'sup:' + s.lexname() for s in ss}
            elif source == 'domains':
                names = {'dom:' + d.name() for s in ss for d in s.topic_domains()}
            else:
                names = {'whole:' + h.name() for s in ss
                         for h in s.part_holonyms() + s.member_holonyms() + s.substance_holonyms()}
        for name in names:
            add(name, i)
    return out


class Categories:
    """Categories over a pool's tokens: members (token indices into the pool) of each usable
    category, the base columns it is defined by, and each token's base columns."""

    def __init__(self, geom, strs, usable, model=None):
        n = len(strs); names, members = [], []
        self.meta = []                     # what each stored fit used was made from
        for src in sorted(geom.categories):
            if src in SOURCES:
                cols = _source_categories(src, [s.strip() for s in strs])
                names += list(cols); members += [np.array(cols[k], int) for k in cols]
            else:
                if src.endswith('.npz'):
                    f = np.load(src, allow_pickle=True)
                    toks, codes = f['tokens'], np.asarray(f['codes']).astype(bool)
                    cols = list(f['names']) if 'names' in f else [f'c{k}' for k in range(codes.shape[1])]
                    tag = os.path.splitext(os.path.basename(src))[0]
                else:
                    if model is None:
                        raise ValueError(f"categories {src!r} are stored per model: say which model")
                    toks, codes, meta = stored.load(model, src, geom.embedding)
                    self.meta.append(meta)
                    cols, tag = [f'c{k}' for k in range(codes.shape[1])], src
                row = {str(t): i for i, t in enumerate(toks)}
                names += [f'{tag}:{x}' for x in cols]
                have = np.array([row.get(s, -1) for s in strs])
                members += [np.where((have >= 0) & codes[np.maximum(have, 0), k])[0] for k in range(codes.shape[1])]
        members = [m[usable[m]] for m in members]
        keep = [k for k, m in enumerate(members) if 2 <= len(m) <= geom.max_share * n]
        self._dropped = (sum(len(m) > geom.max_share * n for m in members), sum(len(m) < 2 for m in members), len(members))
        self.base_names = [names[k] for k in keep]; base = [members[k] for k in keep]
        self.token_cols = [[] for _ in range(n)]
        for c, m in enumerate(base):
            for i in m:
                self.token_cols[i].append(c)
        self.token_cols = [frozenset(c) for c in self.token_cols]
        self.base = base
        if geom.combine == 1:
            self.defs = [(c,) for c in range(len(base))]; self.members = base
        else:
            self.defs, self.members = [], []
            for a in range(len(base)):
                for b in range(a + 1, len(base)):
                    m = np.intersect1d(base[a], base[b], assume_unique=True)
                    if len(m) >= 2:
                        self.defs.append((a, b)); self.members.append(m)
        self.sizes = np.array([len(m) for m in self.members])
        if not len(self.members):
            big, small, total = self._dropped
            raise ValueError(f"no usable categories in {sorted(geom.categories)} for this pool: of {total}, {big} "
                             f"cover more than max_share={geom.max_share} of its {n} tokens and {small} fewer "
                             f"than 2 (raise max_share for broad categories)")
        self.n = n; self._mask, self._carrier, self._of = {}, {}, {}
        self._pair = {d: c for c, d in enumerate(self.defs)} if geom.combine == 2 else None

    def mask(self, c):
        """Which pool tokens are in category c."""
        if c not in self._mask:
            m = np.zeros(self.n, bool); m[self.members[c]] = True; self._mask[c] = m
        return self._mask[c]

    def carriers(self, c):
        """Which pool tokens carry any of the base categories c is defined by."""
        if c not in self._carrier:
            m = np.zeros(self.n, bool)
            for k in self.defs[c]:
                m[self.base[k]] = True
            self._carrier[c] = m
        return self._carrier[c]

    def of_token(self, t):
        """The categories token t is in."""
        if t not in self._of:
            cols = sorted(self.token_cols[t])
            if self._pair is None:
                self._of[t] = cols
            else:
                self._of[t] = [self._pair[(a, b)] for i, a in enumerate(cols) for b in cols[i + 1:]
                               if (a, b) in self._pair]
        return self._of[t]

    def name(self, c):
        return '&'.join(self.base_names[k] for k in self.defs[c])


# -- similarity -----------------------------------------------------------------------------

def _canonical(W, ids, chunk=16384):
    """Park et al.'s canonical representation of the tokens `ids`: the unembedding rows W
    (vocabulary x d, a torch tensor) centred and whitened over the whole vocabulary."""
    import torch
    with torch.no_grad():
        V, d = W.shape
        mu = torch.zeros(d, dtype=torch.float64, device=W.device)
        for k in range(0, V, chunk):
            mu += W[k:k + chunk].double().sum(0)
        mu /= V
        C = torch.zeros(d, d, dtype=torch.float64, device=W.device)
        for k in range(0, V, chunk):
            x = W[k:k + chunk].double() - mu
            C += x.T @ x
        C /= V
        ev, Q = torch.linalg.eigh(C)
        A = Q @ torch.diag(ev.clamp_min(ev.max() * 1e-12).rsqrt()) @ Q.T
        idx = torch.as_tensor(np.asarray(ids), dtype=torch.long, device=W.device)
        return ((W[idx].double() - mu) @ A).float().cpu().numpy()


def model_vectors(hf, ids, embedding, vocab_size):
    """The tokens `ids` in a Hugging Face model's 'input' embeddings or 'canonical'
    representation (its unembeddings whitened over the first vocab_size rows), as numpy."""
    import torch
    if embedding == 'input':
        W = hf.get_input_embeddings().weight
        with torch.no_grad():
            return W[torch.as_tensor(np.asarray(ids), dtype=torch.long, device=W.device)].float().cpu().numpy()
    if embedding == 'canonical':
        return _canonical(hf.get_output_embeddings().weight[:vocab_size], ids)
    raise ValueError(f"embedding {embedding!r}: one of {EMBEDDINGS}")


def _vectors(geom, pool, lm, model=None):
    """(vectors for the pool's tokens, mask of those that have one): from a file; else, for
    'input' or 'canonical', from the model's cached embeddings of the fit vocabulary if there
    are any (the vectors its stored codes were fitted on; tokens outside it go unused); else
    from its weights."""
    n = pool.size
    if geom.embedding in stored.EMBEDDINGS and model is not None and \
            stored.embeddings_status(model, geom.embedding) == 'ok':
        toks, V = stored.embeddings(model, geom.embedding)
        row = {str(t): i for i, t in enumerate(toks)}
        have = np.array([row.get(pool.strs[i], -1) for i in pool.ids])
        X = np.zeros((n, V.shape[1]), np.float32)
        X[have >= 0] = V[have[have >= 0]]
        return X, have >= 0
    if geom.embedding.endswith('.npz'):
        f = np.load(geom.embedding, allow_pickle=True)
        row = {str(t): i for i, t in enumerate(f['tokens'])}
        have = np.array([row.get(pool.strs[i], -1) for i in pool.ids])
        X = np.zeros((n, f['vectors'].shape[1]), np.float32)
        X[have >= 0] = np.asarray(f['vectors'][have[have >= 0]], np.float32)
        return X, have >= 0
    hf = getattr(lm, 'hf_model', None)
    if hf is None:
        raise ValueError(f"embedding={geom.embedding!r} needs the model's weights (lm.hf_model)")
    return model_vectors(hf, pool.ids, geom.embedding, len(pool.tokenizer)), np.ones(n, bool)


# -- assignment -------------------------------------------------------------------------------

class Assigner:
    """Draws one sample's task tokens for a pool and a geometry (see the module docstring)."""

    def __init__(self, geom, pool, lm=None, model=None, seed=0):
        self.geom, self.pool = geom, pool
        X, self.usable = _vectors(geom, pool, lm, model)
        X = X - X[self.usable].mean(0)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        self.X = np.where(norms > 0, X / np.where(norms > 0, norms, 1), 0).astype(np.float32)
        rng = np.random.default_rng(seed); u = np.where(self.usable)[0]
        a, b = rng.choice(u, 20000), rng.choice(u, 20000); ok = a != b
        q = np.einsum('ij,ij->i', self.X[a[ok]], self.X[b[ok]])
        self.mu, self.sd = float(q.mean()), float(q.std())
        self.cats = Categories(geom, [pool.strs[i] for i in pool.ids], self.usable, model)

    def z(self, A, B):
        """z-unit similarities between pool indices A and B."""
        return (self.X[A] @ self.X[B].T - self.mu) / self.sd

    def assign(self, n, rng, groups):
        """(task_toks, info): ids for symbols 0..n-1 (0 the delimiter), and what was drawn.
        groups: arrays of symbols, possibly overlapping (see symbol_groups)."""
        groups = [g[g < n] for g in groups]; groups = [g for g in groups if len(g) >= 2]
        of = {}
        for gi, g in enumerate(groups):
            for s in g:
                of.setdefault(int(s), []).append(gi)
        singles = [s for s in range(1, n) if s not in of]
        for _ in range(RESTARTS):
            cat = self._categories(groups, of, rng)
            if cat is None:
                continue
            for _ in range(REDRAWS):
                drawn = self._tokens(groups, of, singles, cat, rng)
                if drawn is None:
                    break                            # nothing fits these categories
                idx, cands = drawn
                if self.geom.beta > 0:
                    idx = self._resample(groups, of, cands, idx, rng)
                info = self._info(groups, of, cat, idx)
                g_ = self.geom
                if g_.within is not None and min(info['within_z'], default=np.inf) < g_.within:
                    continue
                if g_.between is not None and info['between_z'] is not None and abs(info['between_z']) > g_.between:
                    continue
                toks = np.empty(n, int); toks[0] = self.pool.delim_id
                for s, t in idx.items():
                    toks[s] = self.pool.ids[t]
                return toks, info
        raise RuntimeError(f"couldn't draw {len(groups)} groups (sizes {[len(g) for g in groups]}) under "
                           f"{self.geom!r} in {RESTARTS} tries: loosen within/between, raise max_share, "
                           f"use a bigger pool, or (for overlapping groups) categories with more per token")

    def _categories(self, groups, of, rng):
        """A category for each group (largest groups first), with every symbol's categories
        intersecting; None if that fails. A symbol already constrained proposes a category of
        one of its candidate tokens, so intersections are found rather than hoped for."""
        cats = self.cats; used = set(); cat = {}; inter = {s: None for s in of}
        fits = {}
        for gi in sorted(range(len(groups)), key=lambda i: -len(groups[i])):
            g = groups[gi]; found = None
            if len(g) not in fits:
                fits[len(g)] = np.where(cats.sizes >= len(g))[0]
            pool_ = fits[len(g)]
            if not len(pool_):
                return None
            for _ in range(TRIES):
                held = [s for s in g if inter[int(s)] is not None]
                if held:
                    cand = np.where(inter[int(held[rng.integers(len(held))])])[0]
                    opts = [c for c in cats.of_token(int(cand[rng.integers(len(cand))]))
                            if cats.sizes[c] >= len(g)] if len(cand) else []
                    if not opts:
                        continue
                    c = opts[rng.integers(len(opts))]
                else:
                    c = int(pool_[rng.integers(len(pool_))])
                if set(cats.defs[c]) & used:
                    continue
                M = cats.mask(c)
                new = {int(s): M if inter[int(s)] is None else inter[int(s)] & M for s in g}
                if not all(m.any() for m in new.values()):
                    continue
                if np.count_nonzero(np.logical_or.reduce(list(new.values()))) < len(g):
                    continue
                found = c; break
            if found is None:
                return None
            cat[gi] = found; used |= set(cats.defs[found]); inter.update(new)
        return cat

    def _tokens(self, groups, of, singles, cat, rng):
        """{symbol: pool index}: each grouped symbol a token in all its groups' categories and
        carrying none of the others'; each other symbol one carrying none at all. None if some
        symbol has no token left."""
        cats = self.cats
        car = {gi: cats.carriers(c) for gi, c in cat.items()}
        cands = {}
        for s, gs in of.items():
            m = self.usable.copy()
            for gi in gs:
                m &= cats.mask(cat[gi])
            for gi in cat:
                if gi not in gs:
                    m &= ~car[gi]
            cands[s] = m
        if singles:
            m = self.usable.copy()
            for gi in cat:
                m &= ~car[gi]
            for s in singles:
                cands[s] = m
        taken = np.zeros(len(self.usable), bool); idx = {}
        for s in sorted(cands, key=lambda s: np.count_nonzero(cands[s])):
            free = np.where(cands[s] & ~taken)[0]
            if not len(free):
                return None
            t = int(free[rng.integers(len(free))]); idx[s] = t; taken[t] = True
        return idx, cands

    def _resample(self, groups, of, cands, idx, rng):
        """SWEEPS sweeps resampling each grouped symbol's token among those its categories
        allow (and no other symbol holds), with probability ~ exp(beta * mean z to its mates)."""
        mates = {s: sorted({int(m) for gi in of[s] for m in groups[gi]} - {s}) for s in of}
        for _ in range(SWEEPS):
            for s in rng.permutation(sorted(of)):
                s = int(s); held = np.zeros(len(self.usable), bool)
                held[[t for k, t in idx.items() if k != s]] = True
                cand = np.where(cands[s] & ~held)[0]
                score = self.geom.beta * self.z(cand, np.array([idx[m] for m in mates[s]])).mean(1)
                p = np.exp(score - score.max()); idx[s] = int(cand[rng.choice(len(cand), p=p / p.sum())])
        return idx

    def _info(self, groups, of, cat, idx):
        """Each group's symbols, category and mean pairwise similarity, and the mean similarity
        of pairs of grouped symbols sharing no group (z units)."""
        within = []
        for g in groups:
            ts = np.array([idx[int(s)] for s in g]); Z = self.z(ts, ts)
            within.append(float(Z[np.triu_indices(len(ts), 1)].mean()))
        syms = sorted(of); ts = np.array([idx[s] for s in syms]); Z = self.z(ts, ts)
        apart = np.array([[not set(of[a]) & set(of[b]) for b in syms] for a in syms])
        np.fill_diagonal(apart, False)
        return {'groups': [g.tolist() for g in groups],
                'categories': [self.cats.name(cat[gi]) for gi in range(len(groups))],
                'within_z': within, 'between_z': float(Z[apart].mean()) if apart.any() else None}


_ASSIGNERS = {}


def get_assigner(geom, pool, lm=None, model=None):
    """The Assigner for a geometry, pool and model (whose stored codes it reads), built the
    first time it is asked for."""
    key = (geom, id(pool), model)
    if key not in _ASSIGNERS:
        _ASSIGNERS[key] = Assigner(geom, pool, lm, model)
    return _ASSIGNERS[key]