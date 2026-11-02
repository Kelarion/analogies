"""
The tokens that task symbols are written as.

Two objects:

  Vocabulary  which tokens to draw: a condition on a token, made of a base pool (what its text
              must be), a written form (space) and filters. A plain frozen dataclass that
              knows nothing of tokenizers, so it is the argument ICLModel takes (as `tokens`),
              and is saved, hashed and compared like any other value:
                  vocab = Vocabulary('words', content=True, zipf=(3, 4), noverb=True)
              It applies the condition to any strings, read as tokens' text:
                  vocab.filter([' dog', 'dog', ' dogs', ' the'])  # [' dog']
                  vocab.mask(tokens), vocab.table(tokens)          # pass/fail; their features
                  vocab.words()                                    # its base pool, in its form
              With a tokenizer, mask and filter also check that each string is one token.
  TokenPool   the tokens of one tokenizer that pass: every ordinary token (no special or added
              tokens, clean text, re-encodes to itself) that the vocabulary accepts, one per
              equivalence class, minus the delimiter; plus the delimiter's id and the scoring
              equivalence (TokenEquivalence). ICLModel gets one from get_pool() and asks it
              for each sample's assignment of symbols to tokens:
                  pool = get_pool(vocab, tokenizer, delim=',', equiv=True)
                  pool.size, pool.table            # how many tokens; their features, one row each
                  task_toks = pool.assign(V, rng)  # ids; task_toks[0] is the delimiter
              It needs only the tokenizer, so pools can be built and inspected without weights.

A Vocabulary puts itself in one canonical form when it is made (ranges as float tuples,
excluded words sorted), so equal vocabularies are equal, hash
alike and have the same repr, which is what a run's folder digest is made from. Nothing
rewrites it afterwards, so what a run saves is what was passed. Its repr lists only the
fields that differ from their defaults, so fields added later (with defaults) leave existing
runs' digests alone.

A base pool's name alone can also be passed as a string ('nouns', the default, and 'chars'
are how runs before Vocabulary were saved). So are the '+'-joined spec strings that a few
runs were saved with ('words+content+space'), but only in the canonical form canonical()
gives them, the form they were saved in: anything else is rejected, since two spellings of
one spec would save as two different runs.

  base pools (pool): what a token's text, stripped, must be
    nouns       a word in top-1000-nouns.txt (pool order: the file's)
    chars       an ASCII letter or digit (letters, then digits)
    words       an English word in wordfreq's list, lowercase and alphabetic (its frequency
                order): all parts of speech, function words included
    any         anything (token id order)
  Every pool holds only ordinary tokens: no special or added tokens (so nothing like <think>
  or <|im_end|>), nothing that decodes to partial UTF-8, whitespace, control characters or
  the punctuation in STRUCTURAL, nothing that fails to re-encode to itself, and one token per
  equivalence class (' cat' rather than 'cat', when both pass).

  form
    space       the token's written form, for every pool: True (the default) only tokens with
                a single leading space (' king'), False only tokens without (`king`), None
                both. The right form depends on what precedes the tokens: after ', ' or a
                space, the spaced form is what the model would see; after a newline, the bare
                one.
    ascii       only tokens whose text is ASCII (mostly useful with any)
    content     no function words (NLTK's English stopwords: 'the', 'it', 'there', ...)

  frequency (wordfreq's Zipf scale: log10 of occurrences per billion words, so 3 is once
  per million and 6 once per thousand)
    zipf=(lo, hi)   Zipf frequency in [lo, hi), so bands like (3, 4), (4, 5) tile
    rank=(a, b)     of the tokens left by everything else, those ranked a..b-1 by frequency
                    (0 the most frequent), e.g. rank=(0, 200) or rank=(500, 1000)
                    For tokens that are word pieces rather than words (in any), the Zipf
                    frequency is that of the piece read as a word ('ing'), so treat it loosely.

  lexical semantics, from WordNet. Each sense of a word is weighted by its SemCor count
  plus one (so with no counts, every sense counts the same); proper names and abbreviations
  are left out, and words WordNet doesn't know fail these filters. They don't filter on part
  of speech unless asked (concrete=0.8 alone keeps ' great', whose rare noun sense is a
  person), so pair them: content=True, ndom=0.8, concrete=0.8 for concrete nouns.
    noverb             no verb senses at all
    plain=(lo, hi)     the share of the word's weight in plain senses, in [lo, hi]: not the
                       name of a letter or note (q, phi, sigma, ti, fa) and not a short
                       abbreviation of a much longer word (kg, cm, mph, rpm: up to 4 letters,
                       found in order but not contiguously in a head word at least twice as
                       long; clippings like 'photo' or 'gym' are plain). plain=0.5 drops those
                       that are mostly symbols or abbreviations, plain=1 any that ever are
                       (alpha, pi, chi, lime and nun too).
    lemma              only words in their own base form, for their main part of speech
                       ('spider', not 'spiders'): inflected forms are near-duplicates of their
                       base form in the model's representations, and would dominate any
                       similarity structure
    ndom=(lo, hi)      noun dominance: the share of the word's weight in noun senses, in
                       [lo, hi]. ndom=1 is noun-only, ndom=0.8 mostly a noun.
    concrete=(lo, hi)  of the noun weight, the share in physical entities (senses under
                       physical_entity.n.01) rather than abstractions, in [lo, hi].
                       concrete=0.8 for concrete nouns, concrete=(None, 0.2) for abstract ones.

  by hand
    exclude=(...)      words to leave out, whatever the other filters say (case and
                       surrounding space ignored): for what no rule catches cleanly, like the
                       odd unit or symbol name ('cc', 'lb', 'xi'), or taboo words

  tokens (the one filter that needs the tokenizer)
    whole=(lo, hi)     the share of the token's uses, as the start of a word in running text,
                       in which it is the whole word, in [lo, hi] (see whole_shares). Tokens
                       like ' comp', ' col' or ' ab' are WordNet lemmas (and frequent), but in
                       text mostly begin longer words; whole=0.5 drops them. Only for tokens
                       written with a leading space; others fail it.

  A range is (lo, hi) with None for an open end, or one number, meaning that number and up.
  ('lo:hi' strings work too.)

Filters never reorder the base pool, so a draw depends only on which tokens survive.
(Pools before this definition differ a little: 'nouns' and 'words' wrote each listed word in
one form and kept it if that was one token; 'chars' was bare by default and 'any' took either
form. Runs made with them won't reproduce exactly.)

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
from dataclasses import dataclass, fields
from functools import lru_cache

import numpy as np
import scipy.special as spc

POOLS = ('nouns', 'chars', 'words', 'any')
FLAGS = ('space', 'nospace', 'anyspace', 'ascii', 'content', 'noverb', 'lemma')
RANGES = ('zipf', 'rank', 'ndom', 'concrete', 'whole', 'plain')

# punctuation that structures text, kept out of 'any': sentence ends, separators, brackets
# and quotes ('!' is also ChatLM's placeholder, which must not turn up in a prompt's text)
STRUCTURAL = set('.,;:!?()[]{}<>"\'`|\\/')

N_WORDS = 100_000           # how far down wordfreq's list 'words' looks
N_WHOLE = 300_000           # how far down it whole_shares() looks


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
    if sum(k in out for k in ('space', 'nospace', 'anyspace')) > 1:
        raise ValueError(f"token spec {spec!r} asks for more than one of space, nospace, anyspace")
    return base, out


def _range(name, val):
    """(lo, hi) from 'lo:hi' (either end open), a (lo, hi) pair (None for an open end), or one
    number (that number and up); ints for rank, floats otherwise."""
    typ = int if name == 'rank' else float
    if isinstance(val, str):
        lo, sep, hi = val.partition(':')
        lo, hi = lo.strip() or None, (hi.strip() or None) if sep else None
    elif isinstance(val, (tuple, list)):
        if len(val) != 2:
            raise ValueError(f"{name}={val!r}: a range is (lo, hi)")
        lo, hi = val
    else:
        lo, hi = val, None
    try:
        lo = None if lo is None else typ(lo)
        hi = None if hi is None else typ(hi)
    except (TypeError, ValueError):
        raise ValueError(f"bad range {val!r} for {name}") from None
    if lo is None and hi is None:
        raise ValueError(f"{name}={val!r} leaves both ends open")
    if lo is not None and hi is not None and hi < lo:
        raise ValueError(f"{name}={val!r} is empty")
    return lo, hi


def canonical(spec):
    """A spec string written one way (modifiers sorted, numbers written alike): the form
    the runs saved with spec strings have. canonical('nouns') == 'nouns'."""
    base, mods = parse(spec)
    num = lambda x: '' if x is None else (str(x) if isinstance(x, int) else f'{x:.10g}')
    fmt = lambda k, v: k if v is True else f'{k}={num(v[0])}:{num(v[1])}'
    return '+'.join([base] + [fmt(k, mods[k]) for k in sorted(mods)])


def base_pool(tokens):
    """The base pool's name, e.g. for prompts that say what the tokens are."""
    return as_vocabulary(tokens).pool


@dataclass(frozen=True, repr=False)
class Vocabulary:
    """What to draw task tokens from: a base pool and filters on it (see the module
    docstring for each field). Knows nothing of tokenizers; get_pool() makes it concrete.
    Canonical once made, so equal vocabularies are equal, hash alike and have one repr."""
    pool: str = 'nouns'
    space: bool = True
    ascii: bool = False
    content: bool = False
    noverb: bool = False
    zipf: tuple = None
    rank: tuple = None
    ndom: tuple = None
    concrete: tuple = None
    lemma: bool = False
    whole: tuple = None
    plain: tuple = None
    exclude: tuple = ()

    def __post_init__(self):
        put = lambda k, v: object.__setattr__(self, k, v)
        if self.pool not in POOLS:
            raise ValueError(f"unknown token pool {self.pool!r}; pools are {POOLS}")
        if self.space is not None:
            put('space', bool(self.space))
        for k in ('ascii', 'content', 'noverb', 'lemma'):
            put(k, bool(getattr(self, k)))
        for k in RANGES:
            if getattr(self, k) is not None:
                put(k, _range(k, getattr(self, k)))
        if isinstance(self.exclude, str):
            raise ValueError("exclude is a list of words, not a string")
        put('exclude', tuple(sorted({w.strip().lower() for w in self.exclude})))

    def __repr__(self):
        args = [repr(self.pool)] + [f'{f.name}={getattr(self, f.name)!r}' for f in fields(self)[1:]
                                    if getattr(self, f.name) != f.default]
        return f"{type(self).__name__}({', '.join(args)})"

    @classmethod
    def from_spec(cls, spec):
        """The Vocabulary a '+'-joined spec string describes ('words+zipf=3:4+noverb')."""
        base, mods = parse(spec)
        kw = {k: v for k, v in mods.items() if k not in ('space', 'nospace', 'anyspace')}
        if 'nospace' in mods:
            kw['space'] = False
        elif 'anyspace' in mods:
            kw['space'] = None
        return cls(base, **kw)

    @property
    def mods(self):
        """The filters as parse() gives them: {flag: True, range: (lo, hi)}."""
        out = {k: True for k in ('ascii', 'content', 'noverb', 'lemma') if getattr(self, k)}
        out.update({k: getattr(self, k) for k in RANGES if getattr(self, k) is not None})
        return out

    # -- word-level: everything that doesn't need a tokenizer -------------------------

    def form(self, word):
        """A word written in this vocabulary's form: ' word' if space is True, 'word' if
        False, as given if None."""
        return word if self.space is None else (' ' if self.space else '') + word.strip()

    def in_form(self, token):
        """Whether a string is written in this vocabulary's form: a single leading space and no
        other surrounding whitespace if space is True, no surrounding whitespace if False,
        either of those if None."""
        bare = token.strip()
        if not bare:
            return False
        return {True: token == ' ' + bare, False: token == bare, None: token in (bare, ' ' + bare)}[self.space]

    def _premask(self, tokens):
        """The cheap part of the condition: form, clean text, base pool."""
        base = None if self.pool == 'any' else base_set(self.pool)
        return np.array([self.in_form(t) and _clean(t) and (base is None or t.strip() in base)
                         for t in tokens], bool)

    def mask(self, tokens, tokenizer=None):
        """Whether each string, read as a token's text, passes: written in this vocabulary's
        form, clean, in its base pool, and through its filters. With a tokenizer, also one
        token of it, which the token-level filter (whole) needs. rank is taken among the
        strings that pass the rest, so it depends on the list."""
        tokens = list(tokens)
        keep = self._premask(tokens); whole = None
        if tokenizer is not None:
            enc = tokenizer(tokens, add_special_tokens=False).input_ids
            keep &= np.array([len(e) == 1 for e in enc], bool)
            if 'whole' in self.mods:
                share = whole_shares(tokenizer)
                whole = np.array([share.get(e[0], np.nan) if len(e) == 1 else np.nan for e in enc])
        elif 'whole' in self.mods:
            raise ValueError(f"{self!r} filters on tokens (whole=...): pass a tokenizer")
        return self._mask(tokens, whole, keep)

    def _mask(self, words, whole=None, keep=None):
        """The filters on the words' text (stripped), after `keep` (rank among those kept)."""
        m = self.mods
        words = [w.strip() for w in words]
        keep = np.ones(len(words), bool) if keep is None else keep.copy()
        if 'whole' in m:
            keep &= _within(whole, *m['whole'])
        if 'ascii' in m:
            keep &= np.array([w.isascii() for w in words], bool)
        if 'content' in m:
            keep &= np.array([w.lower() not in function_words() for w in words], bool)
        if self.exclude:
            keep &= np.array([w.lower() not in self.exclude for w in words], bool)
        need = {'zipf', 'rank', 'noverb', 'ndom', 'concrete', 'lemma', 'plain'} & set(m)
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
            if 'lemma' in m:
                keep &= f['lemma'] == 1                            # NaN (unknown) fails
            if 'plain' in m:
                keep &= _within(f['plain'], *m['plain'])
            if 'rank' in m:                                        # last: rank among the rest
                lo, hi = m['rank']
                rest = np.where(keep)[0]
                order = rest[np.argsort(-f['zipf'][rest], kind='stable')]
                keep[:] = False
                keep[order[slice(lo, hi)]] = True
        return keep

    def filter(self, tokens, tokenizer=None):
        """The strings that pass mask(), in their order."""
        tokens = list(tokens)
        return [t for t, k in zip(tokens, self.mask(tokens, tokenizer)) if k]

    def words(self, tokenizer=None):
        """The base pool's words written in this vocabulary's form (bare if space is None),
        filtered, in base order. With a tokenizer (which whole needs), only those that are one
        token: then they are the pool's tokens, but for equivalence classes and the delimiter
        (and rank, which is taken within whichever list is filtered). 'any' is defined by a
        tokenizer's tokens, so it has no word list."""
        return self.filter([self.form(w) for w in base_words(self.pool)], tokenizer)

    def table(self, tokens=None, tokenizer=None):
        """One row per string (default: words()), with the features the filters use; with a
        tokenizer, also each one's whole share (NaN if it isn't one token)."""
        import pandas as pd
        tokens = self.words(tokenizer) if tokens is None else list(tokens)
        t = pd.DataFrame({'token': tokens, **features([w.strip() for w in tokens])})
        if tokenizer is not None:
            share = whole_shares(tokenizer)
            enc = tokenizer(tokens, add_special_tokens=False).input_ids
            t['whole'] = [share.get(e[0], np.nan) if len(e) == 1 else np.nan for e in enc]
        return t


_SHARES = {}


def whole_shares(tokenizer):
    """{token id: share}: of the times a token starts a word in running text, the share in
    which it is the whole word. Read off wordfreq's list (N_WHOLE words, by frequency): each
    word written with a leading space adds its frequency to its first token, as a whole word
    if that is its only token, as a start otherwise. Ids that start no listed word are
    absent. Cached per tokenizer."""
    key = (tokenizer.name_or_path, len(tokenizer))
    if key not in _SHARES:
        from wordfreq import top_n_list
        words = top_n_list('en', N_WHOLE)
        whole, start = {}, {}
        for w, t in zip(words, tokenizer([' ' + w for w in words], add_special_tokens=False).input_ids):
            if t:
                d = whole if len(t) == 1 else start
                d[t[0]] = d.get(t[0], 0.0) + 10 ** (zipf(w) - 9)
        _SHARES[key] = {i: whole.get(i, 0.0) / (whole.get(i, 0.0) + start.get(i, 0.0))
                        for i in set(whole) | set(start)}
    return _SHARES[key]


@lru_cache(maxsize=None)
def base_set(pool):
    """{word: position in base order} for a word-list pool."""
    return {w: k for k, w in enumerate(base_words(pool))}


def base_words(pool):
    """The words a base pool is made from, in base order (none for 'any')."""
    if pool == 'nouns':
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'top-1000-nouns.txt')) as f:
            return list(dict.fromkeys(w.strip() for w in f if w.strip()))
    if pool == 'chars':
        return list(string.ascii_letters + string.digits)
    if pool == 'words':
        from wordfreq import top_n_list
        return [w for w in top_n_list('en', N_WORDS) if w.isalpha() and w.islower()]
    raise ValueError("'any' is made of a tokenizer's tokens, so it has no word list")


def as_vocabulary(tokens):
    """A Vocabulary from a Vocabulary, a pool name or a spec string."""
    if isinstance(tokens, Vocabulary):
        return tokens
    if isinstance(tokens, str):
        return Vocabulary.from_spec(tokens)
    raise TypeError(f"tokens must be a Vocabulary or a pool name, not {type(tokens).__name__}")


def check_tokens(tokens):
    """Raise unless `tokens` is a valid argument as given: a Vocabulary, a pool name, or a
    spec string in the canonical form the runs saved with spec strings have (anything else
    would save under a spelling that loading has to repeat exactly)."""
    vocab = as_vocabulary(tokens)
    if isinstance(tokens, str) and '+' in tokens and tokens != canonical(tokens):
        raise ValueError(f"spec strings are only accepted in their saved form ({canonical(tokens)!r}); "
                         f"for new runs pass tokens={vocab!r}")
    return vocab


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


FEATURES = ('zipf', 'verb', 'ndom', 'concrete', 'lemma', 'plain')


@lru_cache(maxsize=None)
def _symbolic(synset_name):
    wn = _wordnet()
    marks = {wn.synset('letter.n.02'), wn.synset('solfa_syllable.n.01')}
    return any(marks & set(path) for path in wn.synset(synset_name).hypernym_paths())


@lru_cache(maxsize=None)
def plain_share(word):
    """Of a word's sense weight (the senses and weights wordnet_profile uses), the share in
    senses that are neither the name of a letter or note nor a short abbreviation of a much
    longer word (the word has at most 4 letters, and its letters appear in order, but not
    contiguously, in the sense's head lemma, which is at least twice as long: kg in kilogram,
    mph in miles per hour). NaN if WordNet doesn't know the word."""
    wn = _wordnet()
    w = word.lower()
    total = plain = 0.0
    for s in wn.synsets(w):
        if s.instance_hypernyms():
            continue
        pos = 'a' if s.pos() == 's' else s.pos()
        lemmas = [l for l in s.lemmas() if l.name() in {w, wn.morphy(w, pos)}]
        if not lemmas:
            continue
        c = 1 + sum(l.count() for l in lemmas)
        total += c
        head = s.lemmas()[0].name().lower().replace('_', '').replace('-', '')
        it = iter(head)
        abbrev = (head != w and len(w) <= 4 and len(head) >= 2 * len(w)
                  and all(ch in it for ch in w) and w not in head)
        if not (abbrev or _symbolic(s.name())):
            plain += c
    return plain / total if total else np.nan


def features(words, which=FEATURES):
    """Per word, as {name: array}, those of these that are in `which`: zipf, verb (has a
    verb sense), ndom, concrete, lemma (is its own base form for its main part of speech),
    plain (see plain_share).
    The WordNet ones (all but zipf) are NaN for words WordNet doesn't know, and concrete for
    words with no noun senses."""
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
        main = max(weight, key=weight.get)
        out['lemma'][j] = _wordnet().morphy(w.lower(), main) == w.lower()
        out['plain'][j] = plain_share(w)
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

class TokenPool:
    """
    The token ids a Vocabulary picks for a tokenizer (see the module docstring), plus
      delim_id   the delimiter's id (task symbol 0)
      equiv      a TokenEquivalence to score with, or None
    The pool never holds the delimiter or anything equivalent to it, nor two tokens
    equivalent to each other, so any draw from it can be scored with equiv.
    """

    def __init__(self, vocab, tokenizer, delim=',', equiv=False):
        self.vocab = as_vocabulary(vocab)
        self.spec = repr(self.vocab)
        self.base, self.mods = self.vocab.pool, self.vocab.mods
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
        return f"TokenPool({self.spec}, {self.size} tokens: {show}, ...)"

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
        the filters use (see features())."""
        t = self.vocab.table(self.words)
        t.insert(0, 'token', [self.strs[i] for i in self.ids]); t.insert(0, 'id', self.ids)
        share = whole_shares(self.tokenizer)
        t['whole'] = [share.get(i, np.nan) for i in self.ids]
        return t

    def assign(self, n, rng):
        """Ids for task symbols 0..n-1: symbol 0 the delimiter, the rest distinct tokens
        drawn uniformly from the pool. Draws exactly as ICLModel always has (a permutation
        of the whole pool, its first n kept and the first of those replaced by the
        delimiter), so a seed gives the same assignment as before."""
        if n > self.size:
            raise ValueError(f"the task needs {n} tokens but {self.spec} has only {self.size}")
        toks = rng.permutation(self.ids)[:n]
        toks[0] = self.delim_id
        return toks

    # -- building the pool ------------------------------------------------------------

    def _encode(self, text):
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def _build(self):
        """Every ordinary token the vocabulary accepts (see the module docstring), the delimiter
        and its equivalents removed, one per equivalence class (preferring the spaced form,
        then base order), in base order; then the filters, rank last."""
        special = set(self.tokenizer.all_special_ids) | set(self.tokenizer.get_added_vocab().values())
        cand = np.array([i for i in range(len(self.strs)) if i not in special], int)
        cand = cand[self.vocab._premask([self.strs[i] for i in cand])]
        enc = self.tokenizer([self.strs[i] for i in cand], add_special_tokens=False).input_ids
        cand = cand[np.array([e == [i] for i, e in zip(cand, enc)], bool)]       # re-encodes to itself
        bad = self.canon(self.strs[self.delim_id])
        cand = [i for i in cand if self.canon(self.strs[i]) != bad]
        order = (lambda i: i) if self.base == 'any' else (lambda i: base_set(self.base)[self.strs[i].strip()])
        best = {}
        for i in sorted(cand, key=lambda i: (not self.strs[i].startswith(' '), order(i), i)):
            best.setdefault(self.canon(self.strs[i]), i)
        ids = np.array(sorted(best.values(), key=lambda i: (order(i), i)), int)
        whole = None
        if 'whole' in self.mods:
            share = whole_shares(self.tokenizer)
            whole = np.array([share.get(i, np.nan) for i in ids])
        ids = ids[self.vocab._mask([self.strs[i] for i in ids], whole)]
        if not len(ids):
            raise ValueError(f"no tokens left in {self.spec}")
        return ids


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

_POOLS = {}


def get_pool(tokens, tokenizer, delim=',', equiv=False):
    """The TokenPool for a Vocabulary (or pool name) and tokenizer, built the first time
    it is asked for."""
    vocab = as_vocabulary(tokens)
    key = (tokenizer.name_or_path, len(tokenizer), vocab, delim, bool(equiv))
    if key not in _POOLS:
        _POOLS[key] = TokenPool(vocab, tokenizer, delim, equiv)
    return _POOLS[key]