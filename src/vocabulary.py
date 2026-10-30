"""
The tokens that task symbols are written as.

A Vocabulary is a pool of token ids for one tokenizer, chosen by a spec, together with the
delimiter and the scoring equivalence (TokenEquivalence) that go with it. ICLModel makes one
from its `tokens`, `delim` and `equiv` fields and asks it for each sample's assignment of
symbols to tokens:

    vocab = get_vocabulary('words+zipf=3:4+noverb', tokenizer)
    vocab.size, vocab.table           # how many tokens; their features, one row each
    task_toks = vocab.assign(V, rng)  # ids; task_toks[0] is the delimiter

It needs only the tokenizer, so pools can be built and inspected without loading weights.

A spec is a base pool and modifiers joined by '+', as prompt specs are. Specs are put in a
canonical form (canonical()), so 'words+noverb+zipf=3:4' and 'words + zipf=3:4 + noverb' are
the same run.

  base pools
    nouns       top-1000-nouns.txt, in file order, as ' word'      [as before, less a duplicate]
    chars       ASCII letters, then digits, as bare characters       [as before]
    words       English words in wordfreq's frequency order, lowercase and alphabetic,
                as ' word': all parts of speech, function words included
    any         every ordinary token: no special or added tokens (so nothing like <think>
                or <|im_end|>), nothing that decodes to partial UTF-8, whitespace, control
                characters or the punctuation in STRUCTURAL, and nothing that fails to
                re-encode to itself. One token per equivalence class: ' cat' rather than 'cat'
                (unless nospace). In token id order.

  form
    space       with a leading space (the default for nouns and words)
    nospace     without (the default for chars)
                For nouns, words and chars these choose the form each word is written in;
                for any, they keep only tokens of that form.
    ascii       only tokens whose text is ASCII (mostly useful with any)
    content     no function words (NLTK's English stopwords: 'the', 'it', 'there', ...)

  frequency (wordfreq's Zipf scale: log10 of occurrences per billion words, so 3 is once
  per million and 6 once per thousand)
    zipf=lo:hi  Zipf frequency in [lo, hi), so bands like 3:4, 4:5 tile
    rank=a:b    of the tokens left by everything else, those ranked a..b-1 by frequency
                (0 the most frequent), e.g. rank=0:200 or rank=500:1000
                For tokens that are word pieces rather than words (in any), the Zipf
                frequency is that of the piece read as a word ('ing'), so treat it loosely.

  lexical semantics, from WordNet. Each sense of a word is weighted by its SemCor count
  plus one (so with no counts, every sense counts the same); proper names and abbreviations
  are left out, and words WordNet doesn't know fail these filters. They don't filter on part of speech unless
  asked (concrete=0.8 alone keeps ' great', whose rare noun sense is a person), so pair
  them: words+content+ndom=0.8+concrete=0.8 for concrete nouns.
    noverb          no verb senses at all
    ndom=lo:hi      noun dominance: the share of the word's weight in noun senses, in
                    [lo, hi]. ndom=1 is noun-only, ndom=0.8 mostly a noun.
    concrete=lo:hi  of the noun weight, the share in physical entities (senses under
                    physical_entity.n.01) rather than abstractions, in [lo, hi].
                    concrete=0.8 for concrete nouns, concrete=:0.2 for abstract ones.

  A range is lo:hi with either end left open ('3:', ':0.2'), or one number, meaning that
  number and up.

Modifiers filter the base pool but never reorder it, so a draw depends only on which tokens
survive. 'chars' gives exactly the ids, in the same order, that ICLModel's old TOKEN_POOLS did,
and assign() draws as before, so seeded runs reproduce. 'nouns' does too, except that the old
pool held ' no' twice (the file lists it twice), which could give two symbols the same word;
it now holds it once, so a seed draws different nouns than it used to.

Semantic structure (not yet): assign() draws uniformly from the pool, whatever the task's
structure. Controlling similarity within a block versus between blocks, or between tokens
with the same role in different blocks, means assign() needs that structure (each block's
alphabet, as GrammarTask's `code`, and each symbol's kind, as icl_utils.token_kinds) and a
similarity between pool tokens (WordNet distances, or the model's embeddings). That is a
different assign(), and the pool and features here are what it would draw from.
"""
import os
import string
import unicodedata
from functools import lru_cache

import numpy as np
import scipy.special as spc

POOLS = ('nouns', 'chars', 'words', 'any')
FLAGS = ('space', 'nospace', 'ascii', 'content', 'noverb')
RANGES = ('zipf', 'rank', 'ndom', 'concrete')

# punctuation that structures text, kept out of 'any': sentence ends, separators, brackets
# and quotes ('!' is also ChatLM's placeholder, which must not turn up in a prompt's text)
STRUCTURAL = set('.,;:!?()[]{}<>"\'`|\\/')

N_WORDS = 100_000           # how far down wordfreq's list 'words' looks


# -- specs ------------------------------------------------------------------------

def parse(spec):
    """(base pool, {modifier: True for a flag, (lo, hi) for a range}) for a spec."""
    base, *mods = [p.strip() for p in spec.split('+')]
    if base not in POOLS:
        raise ValueError(f"unknown token pool {base!r}; pools are {POOLS}")
    out = {}
    for m in mods:
        name, eq, val = (s.strip() for s in m.partition('='))
        if name in FLAGS and not eq:
            v = True
        elif name in RANGES and eq:
            v = _range(name, val)
        else:
            raise ValueError(f"bad modifier {m!r} in token spec {spec!r}: flags are {FLAGS}, "
                             f"ranges (name=lo:hi) {RANGES}")
        if name in out:
            raise ValueError(f"token spec {spec!r} gives {name!r} twice")
        out[name] = v
    if 'space' in out and 'nospace' in out:
        raise ValueError(f"token spec {spec!r} asks for both space and nospace")
    return base, out


def _range(name, val):
    typ = int if name == 'rank' else float
    lo, sep, hi = val.partition(':')
    try:
        lo = typ(lo) if lo.strip() else None
        hi = typ(hi) if sep and hi.strip() else None
    except ValueError:
        raise ValueError(f"bad range {val!r} for {name}") from None
    if lo is None and hi is None:
        raise ValueError(f"{name}={val!r} leaves both ends open")
    if lo is not None and hi is not None and hi < lo:
        raise ValueError(f"{name}={val!r} is empty")
    return lo, hi


def canonical(spec):
    """The spec written one way (modifiers sorted, numbers written alike), so equal specs
    hash alike. A spec with no modifiers comes back unchanged: canonical('nouns') == 'nouns'."""
    base, mods = parse(spec)
    num = lambda x: '' if x is None else (str(x) if isinstance(x, int) else f'{x:.10g}')
    fmt = lambda k, v: k if v is True else f'{k}={num(v[0])}:{num(v[1])}'
    return '+'.join([base] + [fmt(k, mods[k]) for k in sorted(mods)])


def base_pool(spec):
    """The spec's base pool, e.g. for prompts that say what the tokens are."""
    return parse(spec)[0]


# -- word features --------------------------------------------------------------------

@lru_cache(maxsize=1)
def _zipfs():
    """{word: Zipf frequency}, read off wordfreq's list of frequency bins (bin i holds the
    words with frequency 10^(-i/100), i.e. Zipf 9 - i/100)."""
    from wordfreq import get_frequency_list
    return {w: 9 - i / 100 for i, words in enumerate(get_frequency_list('en'))
            for w in words}


def zipf(word):
    """wordfreq's Zipf frequency of an English word (0 if it has never seen it). Looked
    up directly rather than through zipf_frequency, which re-tokenizes every word and
    takes a millisecond each."""
    return _zipfs().get(word.lower(), 0.0)


@lru_cache(maxsize=1)
def function_words():
    """NLTK's English stopwords: pronouns, determiners, prepositions, auxiliaries, ..."""
    try:
        from nltk.corpus import stopwords
        return frozenset(stopwords.words('english'))
    except LookupError:
        import nltk
        nltk.download('stopwords', quiet=True)
        from nltk.corpus import stopwords
        return frozenset(stopwords.words('english'))


@lru_cache(maxsize=1)
def _wordnet():
    try:
        from nltk.corpus import wordnet as wn
        wn.ensure_loaded()
    except LookupError:
        import nltk
        nltk.download('wordnet', quiet=True)
        from nltk.corpus import wordnet as wn
    return wn


@lru_cache(maxsize=None)
def _physical(synset_name):
    wn = _wordnet()
    phys = wn.synset('physical_entity.n.01')
    return any(phys in path for path in wn.synset(synset_name).hypernym_paths())


@lru_cache(maxsize=None)
def wordnet_profile(word):
    """WordNet's senses of a word (any inflection: 'years' is 'year') that it is a lowercase
    lemma of, so leaving out proper names ('us' as the United States, 'paris') and
    abbreviations ('kv' as kV, 'acc' as ACC), each weighted by its SemCor count + 1, as ({'n', 'v', 'a', 'r'}: weight by part of speech, weight of the
    noun senses that are physical entities). None if WordNet doesn't know the word."""
    wn = _wordnet()
    w = word.lower()
    synsets = [s for s in wn.synsets(w) if not s.instance_hypernyms()]
    if not synsets:
        return None
    weight, phys = dict.fromkeys('nvar', 0.0), 0.0
    for s in synsets:
        pos = 'a' if s.pos() == 's' else s.pos()          # satellites are adjectives
        forms = {w, wn.morphy(w, pos)}
        lemmas = [l for l in s.lemmas() if l.name() in forms]   # case-sensitive
        if not lemmas:
            continue
        c = 1 + sum(l.count() for l in lemmas)
        weight[pos] += c
        if pos == 'n' and _physical(s.name()):
            phys += c
    return (weight, phys) if any(weight.values()) else None


FEATURES = ('zipf', 'verb', 'ndom', 'concrete')


def features(words, which=FEATURES):
    """Per word, as {name: array}, those of these that are in `which`: zipf, verb (has a
    verb sense), ndom, concrete. The WordNet ones (all but zipf) are NaN for words WordNet
    doesn't know, and concrete for words with no noun senses."""
    out = {'zipf': np.array([zipf(w) for w in words], float)} if 'zipf' in which else {}
    wordnet = [k for k in FEATURES[1:] if k in which]
    if not wordnet:
        return out
    out.update({k: np.full(len(words), np.nan) for k in FEATURES[1:]})
    for j, w in enumerate(words):
        prof = wordnet_profile(w)
        if prof is None:
            continue
        weight, phys = prof
        out['verb'][j] = weight['v'] > 0
        out['ndom'][j] = weight['n'] / sum(weight.values())
        if weight['n']:
            out['concrete'][j] = phys / weight['n']
    return {k: v for k, v in out.items() if k in which}


# -- tokenizer facts ----------------------------------------------------------------------

_DECODED = {}


def decoded(tokenizer):
    """Every id's text, decoded on its own (clean_up_tokenization_spaces would turn ' ,'
    into ','). Cached per tokenizer."""
    key = (tokenizer.name_or_path, len(tokenizer))
    if key not in _DECODED:
        _DECODED[key] = tokenizer.batch_decode([[i] for i in range(len(tokenizer))],
                                               clean_up_tokenization_spaces=False)
    return _DECODED[key]


class TokenEquivalence:
    """Groups the tokenizer's vocabulary into classes of tokens that count as the same
    output, so that cross-entropy doesn't penalise the model for picking the 'wrong'
    surface form. Two tokens are equivalent when canon() maps their decoded strings to
    the same thing; the default ignores leading/trailing whitespace. Pass e.g.
    canon=lambda s: s.strip().lower() to also ignore case, or canon=lambda s: s.lstrip(' ')
    to only ignore a leading space."""

    def __init__(self, tokenizer, canon=str.strip):
        self.canon = canon
        self.strs = decoded(tokenizer)
        groups = {}
        for i, s in enumerate(self.strs):
            groups.setdefault(canon(s), []).append(i)
        self.groups = {k: np.array(v) for k, v in groups.items()}

    def members(self, task_toks):
        """The token ids equivalent to each task token (each list includes the token itself)."""
        keys = [self.canon(self.strs[t]) for t in task_toks]
        if len(set(keys)) < len(keys):
            dup = sorted({repr(self.strs[t]) for t, k in zip(task_toks, keys) if keys.count(k) > 1})
            raise ValueError(f"task tokens {', '.join(dup)} are equivalent under canon, so the "
                             f"task would be ambiguous after collapsing; choose other tokens or canon")
        return [self.groups[k] for k in keys]

    def logprobs(self, logs, task_toks):
        """(T, len(task_toks)): log p(next token is in task token j's class), normalised
        over the full vocabulary. Same layout as logs[:, task_toks] - logsumexp(logs)."""
        lse = spc.logsumexp(logs, axis=-1, keepdims=True)
        return np.stack([spc.logsumexp(logs[:, m], axis=-1) for m in self.members(task_toks)],
                        axis=-1) - lse


# -- the vocabulary ------------------------------------------------------------------------

class Vocabulary:
    """
    The pool of token ids a spec picks for a tokenizer (see the module docstring), plus
      delim_id   the delimiter's id (task symbol 0)
      equiv      a TokenEquivalence to score with, or None
    The pool never holds the delimiter or anything equivalent to it, nor two tokens
    equivalent to each other, so any draw from it can be scored with equiv.
    """

    def __init__(self, spec, tokenizer, delim=',', equiv=False):
        self.spec = canonical(spec)
        self.base, self.mods = parse(self.spec)
        self.tokenizer = tokenizer
        self.strs = decoded(tokenizer)
        self.equiv = TokenEquivalence(tokenizer) if equiv else None
        self.canon = self.equiv.canon if equiv else str.strip
        d = self._encode(delim)
        if len(d) != 1:
            raise ValueError(f"delimiter {delim!r} isn't a single token")
        self.delim_id = d[0]
        self.ids = self._build()

    def __repr__(self):
        show = ', '.join(repr(self.strs[i]) for i in self.ids[:6])
        return f"Vocabulary({self.spec!r}, {self.size} tokens: {show}, ...)"

    @property
    def size(self):
        return len(self.ids)

    @property
    def words(self):
        """The pool's tokens as text, stripped."""
        return [self.strs[i].strip() for i in self.ids]

    @property
    def table(self):
        """One row per pool token, in pool order: id, token (as decoded), and the features
        the modifiers filter on (see features())."""
        import pandas as pd
        return pd.DataFrame({'id': self.ids, 'token': [self.strs[i] for i in self.ids],
                             **features(self.words)})

    def assign(self, n, rng):
        """Ids for task symbols 0..n-1: symbol 0 the delimiter, the rest distinct tokens
        drawn uniformly from the pool. Draws exactly as ICLModel always has (a permutation
        of the whole pool, its first n kept and the first of those replaced by the
        delimiter), so a seed gives the same assignment as before."""
        if n > self.size:
            raise ValueError(f"the task needs {n} tokens but {self.spec!r} has only {self.size}")
        toks = rng.permutation(self.ids)[:n]
        toks[0] = self.delim_id
        return toks

    # -- building the pool ------------------------------------------------------------

    def _encode(self, text):
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def _build(self):
        ids = self._base()
        bad = self.canon(self.strs[self.delim_id])
        ids = [i for i in ids if self.canon(self.strs[i]) != bad]
        ids = self._dedupe(ids)
        ids = self._filter(np.array(ids, dtype=int))
        if not len(ids):
            raise ValueError(f"no tokens left in {self.spec!r}")
        return ids

    def _space(self, default):
        return 'space' in self.mods or (default and 'nospace' not in self.mods)

    def _base(self):
        if self.base == 'nouns':
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'top-1000-nouns.txt')) as f:
                return self._legacy([w.strip() for w in f], self._space(True))
        if self.base == 'chars':
            return self._legacy(string.ascii_letters + string.digits, self._space(False))
        if self.base == 'words':
            from wordfreq import top_n_list
            words = [w for w in top_n_list('en', N_WORDS) if w.isalpha() and w.islower()]
            return self._forms(words, self._space(True))
        return self._any()

    def _legacy(self, words, space):
        """The old pools' rule: the bare word must be one token, then its spaced form (if
        space) is used. (The old code raised if that wasn't one token either; this skips it.)"""
        ids = []
        for w in words:
            toks = self._encode(w)
            if len(toks) == 1:
                s = self.strs[toks[0]]
                form = self._encode(' ' + s if space else s)
                if len(form) == 1:
                    ids.append(form[0])
        return ids

    def _forms(self, words, space):
        """The words that are one token in the form used."""
        forms = [(' ' + w) if space else w for w in words]
        return [t[0] for t in self.tokenizer(forms, add_special_tokens=False).input_ids
                if len(t) == 1]

    def _any(self):
        special = set(self.tokenizer.all_special_ids) | set(self.tokenizer.get_added_vocab().values())
        cand = [i for i, s in enumerate(self.strs) if i not in special and _clean(s)]
        if 'space' in self.mods:
            cand = [i for i in cand if self.strs[i].startswith(' ')]
        elif 'nospace' in self.mods:
            cand = [i for i in cand if not self.strs[i].startswith(' ')]
        enc = self.tokenizer([self.strs[i] for i in cand], add_special_tokens=False).input_ids
        cand = [i for i, e in zip(cand, enc) if e == [i]]          # re-encodes to itself
        # one per class, preferring the spaced form, then the lower id
        best = {}
        for i in sorted(cand, key=lambda i: (not self.strs[i].startswith(' '), i)):
            best.setdefault(self.canon(self.strs[i]), i)
        return sorted(best.values())

    def _dedupe(self, ids):
        """The first token of each equivalence class, keeping order."""
        seen, out = set(), []
        for i in ids:
            k = self.canon(self.strs[i])
            if k not in seen:
                seen.add(k)
                out.append(i)
        return out

    def _filter(self, ids):
        m = self.mods
        words = [self.strs[i].strip() for i in ids]
        keep = np.ones(len(ids), bool)
        if 'ascii' in m:
            keep &= np.array([w.isascii() for w in words], bool)
        if 'content' in m:
            keep &= np.array([w.lower() not in function_words() for w in words], bool)
        need = {'zipf', 'rank', 'noverb', 'ndom', 'concrete'} & set(m)
        if need:
            f = features(words, {'verb' if k == 'noverb' else 'zipf' if k == 'rank' else k
                                 for k in need})
            if 'zipf' in m:
                keep &= _within(f['zipf'], *m['zipf'], closed=False)
            if 'noverb' in m:
                keep &= f['verb'] == 0                             # NaN (unknown) fails
            if 'ndom' in m:
                keep &= _within(f['ndom'], *m['ndom'])
            if 'concrete' in m:
                keep &= _within(f['concrete'], *m['concrete'])
            if 'rank' in m:                                        # last: rank among the rest
                lo, hi = m['rank']
                rest = np.where(keep)[0]
                order = rest[np.argsort(-f['zipf'][rest], kind='stable')]
                keep[:] = False
                keep[order[slice(lo, hi)]] = True
        return ids[keep]


def _within(x, lo, hi, closed=True):
    """lo <= x <= hi (x < hi if not closed); NaN fails."""
    ok = ~np.isnan(x)
    if lo is not None:
        ok &= x >= lo
    if hi is not None:
        ok &= (x <= hi) if closed else (x < hi)
    return ok


def _clean(s):
    """Text with no special job: some non-whitespace, no partial UTF-8, no control or
    format characters (newlines, tabs, zero-width), no STRUCTURAL punctuation, and
    whitespace only as a single leading space."""
    if not s.strip() or '\ufffd' in s or s != s.rstrip() or s.startswith('  '):
        return False
    if s.startswith(' ') and s[1].isspace():
        return False
    return not any(c in STRUCTURAL or unicodedata.category(c)[0] in 'CZ'
                   for c in s.lstrip(' '))


# -- cache ------------------------------------------------------------------------------

_VOCABS = {}


def get_vocabulary(spec, tokenizer, delim=',', equiv=False):
    """A Vocabulary, built the first time it is asked for with this tokenizer."""
    key = (tokenizer.name_or_path, len(tokenizer), canonical(spec), delim, bool(equiv))
    if key not in _VOCABS:
        _VOCABS[key] = Vocabulary(spec, tokenizer, delim, equiv)
    return _VOCABS[key]