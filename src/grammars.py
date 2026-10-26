import sys

import pickle as pkl

import numpy as np
import scipy.linalg as la
import networkx as nx
from networkx.algorithms.simple_paths import all_simple_paths
from networkx.linalg.graphmatrix import incidence_matrix
from networkx.algorithms.lowest_common_ancestors import all_pairs_lowest_common_ancestor
from networkx.algorithms.shortest_paths.weighted import dijkstra_path_length as dpl
import torch

import os
import re
import inspect
import functools
import random
from time import time
from itertools import combinations, product
from collections import defaultdict

#%% 
class ParsedSequence(object):
    """
    A class for interacting with a parsed sequence. Takes in strings of the form:

    (s (a1 (b1 word1 ) (b2 (c1 word2) (c2 word3) ) (a2 word4) )

    which concisely, but illegibly, represent a parse tree. Converts this to a more legible 
    list of edges and nodes, lets you compute various nice things, and also implements some 
    simple perturbations of the sequence (like swapping phrases, or pairs of words).

    __init__ inputs:
        bs: the bracketed sentence string
        dep_tree (Bool, default False): is this a dependency parse?
        op_brak (default '['): what kind of bracket starts a phrase
        cl_brak (default ']'): what kind of bracket closes a phrase
        no_words (default True): are the words in addition to phrase tags?

    Attributes (most of them, at least):
        edges: list of tuples, (parent, child), representing the edges of the tree
        nodes: list of strings, 'POS [word]', of all tokens (terminal and non-terminal)
        words: list of strings, 'word', which are the words of the sentence (i.e. terminals)
        brackets: array of {+1, -1, 0}, for {opening, closing, terminal tokens}
        bracketed_string: string, the input used to create the object
        parents: array, the parent of each token (is -1 for the root)
        node_span: array, for each token, the index of the token right after its constituent ends
        subtree_order: array, ints, the `order' of each token

    Note that attributes which return indices, they index the tokens in the order
    they appear in the bracketed sequence -- for a dependency parse, this doesn't 
    generally match the order in the actual sentence. It also doesn't match the 
    indices of word in the sentence (which go from 0, ..., num_terminal_tokens).
    To convert between these three indexing coordinates, use the `term2node`, `node2word`, 
    `word2node` etc. attributes, where a2b contains indices of a in b coordinates.

    Methods (external use):
        tree_dist(i, j, term=True):
            Distance between tokens i and j on the parse tree, returns an integer
            `term` toggles whether i and j index the words, or all nodes
        bracket_dist(i, j):
            Number of (open or closed) brackets between i and j, not including 
            brackets associated with terminal tokens. Related to tree distance.
            Only works on terminal tokens
        is_relative(i, j, order=1):
            Are tokens i and j in the same phrase? 
            `order` is the maximum depth of the subtree that is considered a phrase 
        
    """
    def __init__(self, bs, dep_tree=False, op_brak='(', cl_brak=')', no_words=False):
        """
        Take a bracketed sentence, bs, and compute various quantities defined 
        on it and/or the corresponding parse tree. Provides methods to ask
        questions about the tokens and subtrees.
        
        If the brackets represent a dependency parse, set `dep_tree` to be True 
        """
        
        super(ParsedSequence,self).__init__()
        
        self.bracketed_string = bs
        self.dep_parse = dep_tree

        op = np.array([i for i,j in enumerate(bs) if j==op_brak])
        cl = np.array([i for i,j in enumerate(bs) if j==cl_brak])
        braks = np.sort(np.append(op,cl))
        
        N_node = len(op)
        
        # First: find children
        iscl = np.isin(braks,cl) # label brackets
        
        term_op = braks[np.diff(iscl.astype(int),append=0)==1]
        term_cl = braks[np.diff(iscl.astype(int),append=0)==-1]
        
        assert np.all(np.isin(term_op,op)), "Unequal open and closed brackets!"
        
        leaves = np.where(np.isin(op,term_op))[0]
        # parse depth of each token is the total number of unclosed brackets
        depth = np.cumsum(~iscl*2-1)[np.isin(braks,op)]-1 
        
        # algorithm for finding all edges
        # takes advantage of the fact that nodes only have one parent in a tree
        nodes = np.arange(N_node)
        drawn = list(leaves)
        edges = []
        labels = ['' for _ in range(N_node)]
        i = 0
        while ~np.all(np.isin(nodes,drawn[:i])):
            # find edge
            if drawn[i]>0: # start token has no parent
                parent = np.where((op<op[drawn[i]]) & (depth==(depth[drawn[i]]-1)))[0][-1]
                edges.append((parent, drawn[i]))
            if parent not in drawn:
                drawn.append(parent)
                
            # get token string
            tok_end = braks[np.argmax(braks>op[drawn[i]])]
            tok = bs[op[drawn[i]]:tok_end]
            labels[drawn[i]] = tok.replace(op_brak,'')#.replace(' ','')
            
            i += 1
            
        self.edges = edges
        self.nodes = labels
        self.term2word = leaves # indices of terminal tokens in the sentence
        
        parents = [edges[np.where(np.array(edges)[:,1]==i)[0][0]][0] \
                   for i in range(1,len(nodes))]
        self.parents = np.append(-1,parents) # seq-indexed
        
        self.depth = depth
        
        # Find the order of each subtree 
        depf = np.append(self.depth,0)
        # first, for each node find the node right after its constituent ends
        span = [(depf[i+1:]<=depf[i]).argmax()+(i+1) for i in range(len(self.nodes))]
        # then find the maxmimum distance between each node and its descendants
        max_depth = [self.depth[i:span[i]].max()-self.depth[i] for i in range(len(self.nodes))]
        self.node_span = np.array(span) # each token's corresponding closing
        self.subtree_order = np.array(max_depth) # each node's order (=0 for terminals)
        
        # Represent the bracketed sentence as +/- 1 string
        brah = np.array([1 if b in op[~np.isin(op,term_op)] \
                         else -1 if b in cl[~np.isin(cl,term_cl)] \
                             else 0 if b in term_op \
                                 else np.nan for b in braks])
        brah = brah[~np.isnan(brah)]
        self.brackets = brah.astype(int)
        self.term2brak = np.where(brah==0)[0]

        ## Heuristic conversion to original sentence
        nolspace = ["n't", ".", ",", "'s","''", "%", ":", ";"] # special words
        norspace = ["``"]

        if dep_tree:
            words = labels
        else:
            words = np.array(labels)[leaves]

        line = []
        self.word_tags = []
        self.string = ''
        prev = False
        for i,w in enumerate(words):
            tag, wrd = w.split(' ')
            self.word_tags.append(tag)
            if (wrd in nolspace) or (i == 0) or prev:
                line.append(wrd)
                self.string += wrd
            else:
                line.append(' ' + wrd)
                self.string += ' ' + wrd

            if wrd in norspace:
                prev = True
            else:
                prev = False
    
        if no_words:
            self.words = leaves
            self.ntok = len(leaves)
        else:
            self.words = line
            self.ntok = len(line)

        # when dealing with dep trees, the index in the actual sentence
        # is no the same as the index in the bracketed sentence 
        if dep_tree:
            node_names = [int(n.split(' ')[2]) for n in labels]
            order = list(np.argsort(node_names))
            self.words = [self.words[i] for i in order]
            word_names = [node_names.index(i) for i in range(self.ntok)]
        else:
            node_names = list(range(len(labels)))
            word_names = list(range(len(labels)))
            
        self.node2word = np.array(node_names) # indices of each node in the sequence
        self.word2node = np.array(word_names) # indices of each word in the node list
        self.node2term = np.array([leaves.tolist().index(i) if i in leaves else -1 \
                                   for i in node_names])
        
        self.node_tags = [w.split(' ')[0] for w in self.nodes]
        self.pos_tags = list(np.array(self.node_tags)[leaves])
        
    def __repr__(self):
        return self.bracketed_string

    def binary(self):

        d = len(self.nodes)
        B = np.zeros((self.ntok, d))
        for i,w in enumerate(self.term2word):
            B[i, self.path_to_root([], w)] = 1

        return B[:,1:]

    def parse_depth(self, i, term=True):
        """Distance to root"""
        if term: # indexing terminal tokens?
            i = self.word2node[self.term2word[i]]
        else:
            i = self.word2node[i]
        
        return self.depth[i]
    
    def path_to_root(self, path, tok):
        """Recursion to fill `path` with the ancestors of `tok`"""
        path.append(tok)
        whichedge = np.where(np.array(self.edges)[:,1]==tok)[0]
        if len(whichedge)>0:
            parent = self.edges[whichedge[0]][0]
            path = self.path_to_root(path, parent)
        return path
    
    def tree_dist(self, i, j, term=None):
        """
        Compute d = depth(i) + depth(j) - 2*depth(nearest common ancestor)
        """
        if term is None:
            term = not self.dep_parse
        if term: # indexing terminal tokens?
            i = self.word2node[self.term2word[i]]
            j = self.word2node[self.term2word[j]]
        else:
            i = self.word2node[i]
            j = self.word2node[j]
            
        # take care of pathological cases
        if i>j:
            i_ = j
            j_ = i
        elif i<j:
            i_ = i
            j_ = j
        else:
            return 0
        if i==0:
            return self.depth[j_]
        
        # get ancestors of both
        anci = np.array(self.path_to_root([], i_))  # [i, parent(i), ..., 0]
        ancj = np.array(self.path_to_root([], j_))  # [j, parent(j), ..., 0]
        
        # get nearest common ancestor
        nearest = np.array(anci)[np.isin(anci,ancj)][0]
        
        return self.depth[i_] + self.depth[j_] - 2*self.depth[nearest]
    
    def ancestor_tags(self, i, n=2):
        i_ = self.word2node[self.term2word[i]]
        anc = np.array(self.path_to_root([], i_)) # all ancestors
        return self.node_tags[anc[np.min([n, len(anc)-1])]] # choose the nth one
    
    def phrases(self, order=1, min_length=2, strict=False):
        """
        Phrases of a given order are subtrees whose deepest member is at 
        most `order' away from the subtree root
        
        Returns a list of arrays, which contain the indices of all phrases of 
        specified order (if strict) or at least specified order (if not strict)
        
        Note that if strict=False, some indices might appear twice, as phrases 
        of lower order are nested in phrases of higher order.
        """
        if order == 0:
            is_fake = True
            order = 1
        else:
            is_fake = False
        if strict:
            phrs = np.where((self.subtree_order!=0)&(self.subtree_order==order))[0]
        else:
            phrs = np.where((self.subtree_order!=0)&(self.subtree_order<=order))[0]
        
        # if is_fake:
        #     phrs = np.sort(np.random.choice(len(self.node2term),6,replace=False))

        phrases = [np.array(range(i+1,self.node_span[i])) for i in phrs]
        # print(phrases)

        chunks = [self.node2term[p[np.isin(p, self.term2word)]] for p in phrases]
        chunks = [c for c in chunks if len(c)>=min_length]

        if is_fake:
            c0 = np.array([c[0]+len(c)//2 for c in chunks])
            len_phrs = np.array([len(c) for c in chunks])
            ovlp = np.array([c0[i] - (c0[i-1]+len_phrs[i-1]) for i in range(1,len(c0))] + [0])
            c0 += ovlp*(ovlp<0)
            chunks = [np.arange(c0[i], c0[i]+len(chunks[i])) for i in range(len(chunks))]
        
        return [c for c in chunks if len(c)>=min_length and max(c)<=self.ntok]
    
    def bracket_dist(self,i,j):
        """Number of (non-terminal) brackets between i and j. Only guaranteed 
        to be meaningful for adjacent terminal tokens."""
        i = self.node2term[self.word2node[self.term2word[i]]]
        j = self.node2term[self.word2node[self.term2word[j]]]
            
        return np.abs(self.brackets)[self.term2brak[i]:self.term2brak[j]+1].sum()
        
    def is_relative(self, i, j, order=1, term=None):
        """
        Bool, are tokens i and j part of the same n-th order subtree?
        Equivalently: is the nearest common ancestor of i and j of the specified
        order?
        """
        if term is None:
            term = not self.dep_parse
        
        if term: # indexing terminal tokens?
            i = self.word2node[self.term2word[i]]
            j = self.word2node[self.term2word[j]]
        else:
            i = self.word2node[i]
            j = self.word2node[j]
        
        if order==1 and (self.depth[i]!=self.depth[j]):
            return False # a necessary condition
        
        anci = np.array(self.path_to_root([], i)) # these are node indices
        ancj = np.array(self.path_to_root([], j)) # these are word indices
        nearest = np.array(anci)[np.isin(anci,ancj)][0]
        
        return self.subtree_order[nearest]<=order

    def ngram_shuffle(self, n):
        """ shuffle sentence in units of n """
        if self.ntok<n:
            raise ValueError
        n_pad = int(n-np.mod(self.ntok,n))
        # padded = np.insert(swap_idx.astype(float), 
        #                    ntok-n_pad-1, 
        #                    np.ones(n_pad)*np.nan)
        padded = np.append(np.arange(self.ntok, dtype=float), np.ones(n_pad)*np.nan)
        while 1:
            shuf_idx = np.random.permutation(padded.reshape((-1,n))).flatten()
            shuf_idx = shuf_idx[~np.isnan(shuf_idx)].astype(int)
            if np.any(shuf_idx != np.arange(self.ntok)):
                break

        return shuf_idx

    def phrase_swap(self, order=1):
        """ swap two phrases of a given order """
        phr = self.phrases(order=order, strict=True)

        if len(phr)<2:
            return []

        these_phr = np.sort(np.random.choice(len(phr),2,replace=False))
        phra = [phr[i] for i in these_phr]

        splt_idx = np.concatenate([(s[0],s[-1]+1) for s in phra])
        chunked = np.split(np.arange(self.ntok),splt_idx)
        swap_idx = np.concatenate(np.array(chunked)[[0,3,2,1,4]])
        
        return swap_idx

    def adjacent_swap(self, t):
        """ Swap adjacent words that are distance t apart """

        crossings = np.diff(np.abs(self.brackets).cumsum()[self.term2brak])

        if not np.any(np.isin(crossings, t-2)):
            return []

        i = int(np.random.choice(np.where(crossings==t-2)[0]))

        swap_idx = np.array(range(self.ntok))
        swap_idx[i] = i+1
        swap_idx[i+1] = i

        return swap_idx


###########################################################
###### PCFG ###############
##########################################################

EOS = 0
NT, T = 0, 1
_TOL = 1e-12


def _pick(rng, options, weights):
    w = np.asarray(weights, float)
    if not len(w) or w.sum() <= 0:
        raise ValueError("no options to sample from")
    return options[int(np.searchsorted(np.cumsum(w), rng.random() * w.sum()))]


def _spec_repr(v):
    """Deterministic text for a constructor argument: graphs by their sorted nodes and
    edges (with attributes), sets sorted, callables by name, arrays as lists, the
    rest by repr."""
    if isinstance(v, nx.Graph):
        sort = lambda xs: sorted(xs, key=repr)
        node = lambda n, d: (n, sort(d.items()))
        if v.is_multigraph():
            edges = sort((u, w, k, sort(d.items())) for u, w, k, d in v.edges(keys=True, data=True))
        else:
            edges = sort((u, w, sort(d.items())) for u, w, d in v.edges(data=True))
        return '%s(nodes=%r, edges=%r)' % (type(v).__name__, sort(node(n, d) for n, d in v.nodes(data=True)), edges)
    if callable(v) and not isinstance(v, Sequential):
        return getattr(v, '__qualname__', type(v).__name__)
    if isinstance(v, np.ndarray):
        return repr(v.tolist())
    if isinstance(v, (set, frozenset)):     # sorted: set order varies between runs
        inner = ', '.join(sorted(_spec_repr(x) for x in v))
        if isinstance(v, frozenset):
            return 'frozenset({%s})' % inner if v else 'frozenset()'
        return '{%s}' % inner if v else 'set()'
    if isinstance(v, (list, tuple)):
        inner = ', '.join(_spec_repr(x) for x in v)
        return '[%s]' % inner if isinstance(v, list) else '(%s)' % inner
    if isinstance(v, dict):
        return '{%s}' % ', '.join('%r: %s' % (k, _spec_repr(x)) for k, x in sorted(v.items(), key=repr))
    return repr(v)


def _record_init(init):
    """Wrap a Sequential subclass's __init__ to remember the call that built the object
    (the outermost constructor's, when they chain through super().__init__)."""
    sig = inspect.signature(init)

    @functools.wraps(init)
    def wrapped(self, *args, **kwargs):
        if '_init_spec' not in self.__dict__:
            bound = sig.bind(self, *args, **kwargs)
            bound.apply_defaults()
            self._init_spec = (type(self).__name__, dict(list(bound.arguments.items())[1:]))
        init(self, *args, **kwargs)
    return wrapped


class Sequential:
    """Everything that does not depend on how the state is represented."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if '__init__' in cls.__dict__:
            cls.__init__ = _record_init(cls.__init__)

    def __repr__(self):
        """The constructor call that built this object, e.g. FSA(graph=MultiDiGraph(...),
        length=1). Equal arguments give equal reprs, which the experiment hashing in
        super_experiments relies on (the default repr contains a memory address)."""
        name, args = self.__dict__.get('_init_spec', (type(self).__name__, {}))
        return '%s(%s)' % (name, ', '.join('%s=%s' % (k, _spec_repr(v)) for k, v in args.items()))

    def state(self, seq=()):
        """A state with `seq` pushed. Stops early if the prefix dies, so
        len(st.tokens) - 1 is then the position that killed it."""
        st = self.State(self)
        for t in (self.encode(seq, strict=False) if isinstance(seq, str) else seq):
            if not st.push(t):
                break
        return st

    def continuations(self, prefix=()):
        """(tokens, probs) for what may follow `prefix`. Token 0 means EOS."""
        return self.state(prefix).continuations()

    def is_valid_prefix(self, prefix=()):
        return self.state(prefix).viable

    def generate(self, rng=None, tree=False, max_len=10_000):
        """Sample a sentence left to right, with the distribution used at each
        step. tree=True also returns a structure for the sampled sentence."""
        rng = np.random.default_rng() if rng is None else rng
        nt = len(self.vocab)
        st = self.state()
        tokens, probs = [], []
        while len(tokens) <= max_len:
            cands, ps = st.continuations()
            v = np.zeros(nt)
            v[cands] = ps
            probs.append(v)
            t = _pick(rng, cands, ps)
            if t == EOS:
                return (tokens, probs, st.tree(rng)) if tree else (tokens, probs)
            st.push(t)
            tokens.append(t)
        raise RuntimeError("max_len exceeded")

    def parse(self, seq, best=True, rng=None):
        """Parse a sentence (a string, or a list of token ints). Returns the
        structure, or None if the sequence is not a sentence of the grammar.
        best=True gives the most probable one, best=False samples."""
        st = self.state(seq)
        if not st.complete:
            return None
        return st.best_tree() if best else st.tree(rng)

    sep = ''          # separator between terminals in a decoded string

    def encode(self, text, strict=True):
        """Symbols outside the alphabet map to -1, which never matches."""
        # "".split(sep) is [''], not [], so the empty string needs a guard
        syms = (text.split(self.sep) if self.sep else list(text)) if text else []
        if strict and not set(syms) <= set(self.token_of):
            raise KeyError(f"not terminals: {sorted(set(syms) - set(self.token_of))}")
        return [self.token_of.get(c, -1) for c in syms]

    def decode(self, tokens):
        return self.sep.join(self.vocab[t] for t in tokens if t != EOS)


class PCFG(Sequential):
    """Reduced, epsilon-free, integerised context-free grammar, plus Stolcke's
    two closure matrices. Rules are strings: each character is a symbol,
    uppercase = nonterminal."""

    def __init__(self, rules, init="S", weights=None):
        alts = {A: (list(r) if type(r) is list else [r])
                for A, r in rules.items()}
        todo = list(alts)
        while todo:                                    # used but undefined
            for rhs in alts[todo.pop()]:
                for c in rhs:
                    if c.isupper() and c not in alts:
                        alts[c] = []
                        todo.append(c)
        probs = {}
        for A in alts:
            if weights and A in weights:
                w = np.asarray(weights[A], float)
                probs[A] = list(w / w.sum())
            else:
                probs[A] = [1.0 / len(alts[A])] * len(alts[A]) if alts[A] else []

        alts, probs = self._reduce(alts, probs, init)
        n = self._fixpoint(alts, probs, empty_only=True)   # P(A =>* eps)
        z = self._fixpoint(alts, probs, empty_only=False)  # P(A halts)
        self.p_finite = z[init]
        self.p_empty = n[init] / z[init]
        self.q_start = z[init] - n[init]
        alts, probs = self._unepsilon(alts, probs, n, z)
        if init not in alts:
            raise ValueError(f"{init!r} derives only the empty string")
        alts, probs = self._reduce(alts, probs, init)

        terms = sorted({c for A in alts for r in alts[A]
                        for c in r if not c.isupper()})
        self.vocab = [None] + terms
        self.token_of = {c: i + 1 for i, c in enumerate(terms)}

        names = sorted(alts)
        idx = {A: i for i, A in enumerate(names)}
        m = len(names) + 1
        self.phi_nt = len(names)
        self.nt_name = names + ["<start>"]
        self.init_nt = idx[init]
        self.lhs, self.rhs, self.prob = [], [], []
        self.rules_of = [[] for _ in range(m)]
        for A in names:
            for r, p in zip(alts[A], probs[A]):
                self.rules_of[idx[A]].append(len(self.lhs))
                self.lhs.append(idx[A])
                self.rhs.append(tuple((NT, idx[c]) if c.isupper()
                                      else (T, self.token_of[c]) for c in r))
                self.prob.append(p)
        self.phi = len(self.lhs)
        self.rules_of[self.phi_nt].append(self.phi)
        self.lhs.append(self.phi_nt)
        self.rhs.append(((NT, idx[init]),))
        self.prob.append(1.0)
        self.is_unit = [len(r) == 1 and r[0][0] == NT for r in self.rhs]

        PL, PU = np.zeros((m, m)), np.zeros((m, m))
        for i, r in enumerate(self.rhs):
            if r[0][0] == NT:
                PL[self.lhs[i], r[0][1]] += self.prob[i]
                if self.is_unit[i]:
                    PU[self.lhs[i], r[0][1]] += self.prob[i]
        self.RL, self.RU = self._closure(PL), self._closure(PU)

    # -- compilation helpers ----------------------------------------------

    @staticmethod
    def _reduce(alts, probs, init):
        """Drop nonterminals deriving no terminal string, then unreachable
        ones; renormalise what survives."""
        while True:
            gen, changed = set(), True
            while changed:
                changed = False
                for A in alts:
                    if A not in gen and any(
                            all((not c.isupper()) or c in gen for c in r)
                            for r in alts[A]):
                        gen.add(A)
                        changed = True
            new_a, new_p, dropped = {}, {}, False
            for A in alts:
                if A not in gen:
                    dropped = True
                    continue
                keep = [(r, p) for r, p in zip(alts[A], probs[A])
                        if all((not c.isupper()) or c in gen for c in r)]
                dropped |= len(keep) != len(alts[A])
                if keep:
                    tot = sum(p for _, p in keep)
                    new_a[A] = [r for r, _ in keep]
                    new_p[A] = [p / tot for _, p in keep]
            alts, probs = new_a, new_p
            if not dropped:
                break
        if init not in alts:
            raise ValueError(f"the language generated from {init!r} is empty")
        reach, stack = {init}, [init]
        while stack:
            for r in alts[stack.pop()]:
                for c in r:
                    if c.isupper() and c not in reach:
                        reach.add(c)
                        stack.append(c)
        return {A: alts[A] for A in reach}, {A: probs[A] for A in reach}

    @staticmethod
    def _fixpoint(alts, probs, empty_only, iters=20000):
        """Least fixpoint of x[A] = sum_r p * prod x[children]. With
        empty_only, a terminal kills the term (giving P(A =>* eps)); otherwise
        it contributes 1 (giving P(A has a finite derivation))."""
        x = {A: 0.0 for A in alts}
        for _ in range(iters):
            delta = 0.0
            for A in alts:
                tot = 0.0
                for r, p in zip(alts[A], probs[A]):
                    q = p
                    for c in r:
                        q *= x[c] if c.isupper() else (0.0 if empty_only else 1.0)
                    tot += q
                delta = max(delta, abs(tot - x[A]))
                x[A] = tot
            if delta < 1e-14:
                break
        return x

    @staticmethod
    def _unepsilon(alts, probs, n, z):
        """Delete nullable symbols from each RHS in every combination, weighting
        by n (child went empty) or q = z - n (child did not). The weights over
        non-empty outcomes total q[A], so dividing by it renormalises."""
        q = {A: z[A] - n[A] for A in alts}
        out_a, out_p = {}, {}
        for A in alts:
            if q[A] <= _TOL:
                continue
            acc = defaultdict(float)
            for rhs, p in zip(alts[A], probs[A]):
                if any(c.isupper() and z[c] <= _TOL for c in rhs):
                    continue
                pos = [i for i, c in enumerate(rhs) if c.isupper() and n[c] > _TOL]
                for mask in range(1 << len(pos)):
                    drop = {pos[b] for b in range(len(pos)) if mask >> b & 1}
                    kept = [c for i, c in enumerate(rhs) if i not in drop]
                    if not kept:
                        continue
                    w = p
                    for i, c in enumerate(rhs):
                        w *= n[c] if i in drop else (q[c] if c.isupper() else 1.0)
                    if w > _TOL:
                        acc[''.join(kept)] += w
            if acc:
                out_a[A] = list(acc)
                out_p[A] = [v / q[A] for v in acc.values()]
        return out_a, out_p

    @staticmethod
    def _closure(P):
        try:
            R = np.linalg.inv(np.eye(len(P)) - P)
        except np.linalg.LinAlgError:
            raise ValueError("closure is singular; the grammar is improper")
        R[np.abs(R) < _TOL] = 0.0
        if np.any(R < 0):
            raise ValueError("closure diverged; the grammar is improper")
        return R


class EarleyState:
    """One left-to-right parse. Items are (rule, dot, origin) -> [alpha, gamma].

    Forward probabilities are rescaled at each position by the one-step prefix
    probability. That stops them underflowing on long sentences and makes the
    denominator in `continuations` exactly 1. It is exact: an item's inner
    probability picks up s_i / s_origin, and the factors cancel in both
    completion and prediction.
    """

    def __init__(self, grammar):
        g = self.g = grammar
        seed = (g.phi, 0, 0)
        self.chart = [{seed: [1.0, 1.0]}]
        self.kernel = [[seed]]          # items prediction may fire from
        self.log_scale = 0.0
        self.tokens = []
        self.viable = True
        self._done = {}                 # position -> {(lhs, origin): [(rule, gamma)]}
        self._process(0)

    def _process(self, i):
        g, chart = self.g, self.chart[i]

        # completion, origins descending: an epsilon-free grammar makes
        # completion origins strictly decrease, so one pass suffices. Unit
        # rules are skipped because RU already sums over unit chains.
        for k in range(i - 1, -1, -1):
            batch = [(it[0], v[1]) for it, v in chart.items()
                     if it[2] == k and it[1] == len(g.rhs[it[0]])
                     and not g.is_unit[it[0]]]
            for ridx, gam in batch:
                col = g.RU[:, g.lhs[ridx]]
                for (prid, pdot, porig), pv in self.chart[k].items():
                    prhs = g.rhs[prid]
                    if pdot >= len(prhs) or prhs[pdot][0] != NT:
                        continue
                    ru = col[prhs[pdot][1]]
                    if ru == 0.0:
                        continue
                    new = (prid, pdot + 1, porig)
                    ent = chart.get(new)
                    if ent is None:
                        ent = chart[new] = [0.0, 0.0]
                        self.kernel[i].append(new)
                    ent[0] += pv[0] * ru * gam
                    ent[1] += pv[1] * ru * gam

        # prediction, from kernel items only: RL already sums over left-corner
        # chains, so predicting from predictions would double count.
        for it in self.kernel[i]:
            ridx, dot, _ = it
            rhs = g.rhs[ridx]
            if dot >= len(rhs) or rhs[dot][0] != NT:
                continue
            a = chart[it][0]
            if a == 0.0:
                continue
            row = g.RL[rhs[dot][1]]
            for Y in np.nonzero(row)[0]:
                for r2 in g.rules_of[Y]:
                    p = g.prob[r2]
                    new = (r2, 0, i)
                    ent = chart.get(new)
                    if ent is None:
                        ent = chart[new] = [0.0, p]
                    ent[0] += a * row[Y] * p

    def push(self, token):
        """Scan one token. Returns False, leaving the state dead, if no
        sentence of the grammar starts with the resulting prefix."""
        g = self.g
        i = len(self.tokens)
        sym = (T, token)
        hits, c = [], 0.0
        for (ridx, dot, origin), v in self.chart[i].items():
            if dot < len(g.rhs[ridx]) and g.rhs[ridx][dot] == sym:
                hits.append(((ridx, dot + 1, origin), v))
                c += v[0]
        self.tokens.append(token)
        if c <= 0.0:
            self.chart.append({})
            self.kernel.append([])
            self.viable = False
            return False
        self.chart.append({new: [v[0] / c, v[1] / c] for new, v in hits})
        self.kernel.append([new for new, _ in hits])
        self.log_scale += float(np.log(c))
        self._process(i + 1)
        return True

    # -- phrase structure --------------------------------------------------

    def _completed(self, i):
        """Completed items of set i, grouped by (nonterminal, origin)."""
        by = self._done.get(i)
        if by is None:
            g, by = self.g, defaultdict(list)
            for (r, dot, o), v in self.chart[i].items():
                if dot == len(g.rhs[r]) and v[1] > 0.0:
                    by[(g.lhs[r], o)].append((r, v[1]))
            self._done[i] = by
        return by

    def tree(self, rng=None, max_depth=1000):
        """Sample a parse of the tokens read so far, proportional to inner
        probability -- i.e. from P(tree | sentence). Ambiguous sentences give a
        different tree each call."""
        g = self.g
        if not self.complete:
            raise ValueError("the tokens so far are not a complete sentence")
        rng = np.random.default_rng() if rng is None else rng
        if not self.tokens:
            return (g.nt_name[g.init_nt], ())
        return self._sub(g.init_nt, 0, len(self.tokens), rng, max_depth)

    def _sub(self, Z, j, i, rng, depth):
        if depth <= 0:
            raise RecursionError("unit-production cycle while extracting a tree")
        opts = self._completed(i).get((Z, j), ())
        r = _pick(rng, [o[0] for o in opts], [o[1] for o in opts])
        return (self.g.nt_name[Z], self._kids(r, j, i, rng, depth - 1))

    def _kids(self, r, j, i, rng, depth):
        """Split [j, i) across the RHS of r, right to left. Item (r, t, j) sits
        in set p exactly when the first t symbols cover [j, p), so the splits
        are read straight off the chart; weighting each by the inner
        probability of the two parts samples the forest correctly, the
        rescaling factors being constant across split points."""
        g = self.g
        rhs = g.rhs[r]
        plan, end = [], i
        for t in range(len(rhs) - 1, -1, -1):
            kind, val = rhs[t]
            if kind == T:
                plan.append((kind, val, end - 1, end))
                end -= 1
            else:
                cands, ws = [], []
                for p in range(j, end + 1):
                    left = self.chart[p].get((r, t, j))
                    if left is None or left[1] <= 0.0:
                        continue
                    tot = sum(gm for _, gm in self._completed(end).get((val, p), ()))
                    if tot > 0.0:
                        cands.append(p)
                        ws.append(left[1] * tot)
                p = _pick(rng, cands, ws)
                plan.append((kind, val, p, end))
                end = p
        return tuple(val if kind == T else self._sub(val, a, b, rng, depth)
                     for kind, val, a, b in reversed(plan))

    def best_tree(self):
        """The most probable parse, by dynamic programming over the chart.
        Spans are filled shortest first; within a span, rules with two or more
        symbols depend only on strictly shorter spans, so only unit rules need
        relaxing, and at most |N| rounds since a repeated nonterminal in a unit
        chain can only lower the probability."""
        g = self.g
        if not self.complete:
            raise ValueError("the tokens so far are not a complete sentence")
        n = len(self.tokens)
        if not n:
            return (g.nt_name[g.init_nt], ())

        best = {}                       # (Z, j, i) -> (logprob, rule, plan)
        for width in range(1, n + 1):
            for j in range(0, n - width + 1):
                i = j + width
                units = []
                for (Z, o), items in self._completed(i).items():
                    if o != j:
                        continue
                    for r, _ in items:
                        if g.is_unit[r]:
                            units.append((Z, r))
                            continue
                        sc, plan = self._best_rhs(r, j, i, best)
                        if sc is not None:
                            sc += np.log(g.prob[r])
                            if sc > best.get((Z, j, i), (-np.inf,))[0]:
                                best[(Z, j, i)] = (sc, r, plan)
                for _ in range(len(g.nt_name)):
                    changed = False
                    for Z, r in units:
                        child = g.rhs[r][0][1]
                        if (child, j, i) not in best:
                            continue
                        sc = np.log(g.prob[r]) + best[(child, j, i)][0]
                        if sc > best.get((Z, j, i), (-np.inf,))[0]:
                            best[(Z, j, i)] = (sc, r, ((NT, child, j, i),))
                            changed = True
                    if not changed:
                        break

        return self._build(g.init_nt, 0, n, best)

    def _best_rhs(self, r, j, i, best):
        """Best split of [j, i) across the RHS of r. layers[t] maps an end
        position to (logprob, backpointer) for the first t symbols; membership
        of (r, t, j) in the chart prunes positions that are not reachable."""
        g = self.g
        rhs = g.rhs[r]
        layers = [{j: (0.0, None)}]
        for t, (kind, val) in enumerate(rhs):
            nxt = {}
            for p, (sc, _) in layers[t].items():
                if kind == T:
                    q = p + 1
                    if p < i and self.tokens[p] == val and (r, t + 1, j) in self.chart[q]:
                        if sc > nxt.get(q, (-np.inf,))[0]:
                            nxt[q] = (sc, (kind, val, p, q))
                else:
                    for q in range(p + 1, i + 1):
                        sub = best.get((val, p, q))
                        if sub is None or (r, t + 1, j) not in self.chart[q]:
                            continue
                        cand = sc + sub[0]
                        if cand > nxt.get(q, (-np.inf,))[0]:
                            nxt[q] = (cand, (kind, val, p, q))
            if not nxt:
                return None, None
            layers.append(nxt)
        if i not in layers[-1]:
            return None, None
        plan, q = [], i
        for t in range(len(rhs), 0, -1):
            bp = layers[t][q][1]
            plan.append(bp)
            q = bp[2]
        return layers[-1][i][0], tuple(reversed(plan))

    def _build(self, Z, j, i, best):
        g = self.g
        sc, r, plan = best[(Z, j, i)]
        kids = tuple(val if kind == T else self._build(val, a, b, best)
                     for kind, val, a, b in plan)
        return (g.nt_name[Z], kids)

    @property
    def prefix_logprob(self):
        """log P(some sentence starts with the tokens so far), terminating
        derivations only."""
        if not self.viable:
            return -np.inf
        if not self.tokens:
            return float(np.log(self.g.p_finite))
        return self.log_scale + float(np.log(self.g.q_start))

    @property
    def sentence_logprob(self):
        """log P(the tokens so far are exactly a sentence)."""
        g = self.g
        if not self.viable:
            return -np.inf
        if not self.tokens:
            return float(np.log(g.p_empty * g.p_finite)) if g.p_empty else -np.inf
        gam = self.chart[len(self.tokens)].get((g.phi, 1, 0))
        if not gam or gam[1] <= 0.0:
            return -np.inf
        return self.log_scale + float(np.log(gam[1] * g.q_start))

    @property
    def complete(self):
        return self.sentence_logprob > -np.inf

    def continuations(self):
        """(tokens, probs): the exact next-token distribution, token 0 = EOS.
        ([], []) once the prefix is dead. Rescaling makes the denominator 1."""
        g = self.g
        i = len(self.tokens)
        if not self.viable:
            return [], []
        acc = defaultdict(float)
        for (ridx, dot, _), v in self.chart[i].items():
            rhs = g.rhs[ridx]
            if dot < len(rhs) and rhs[dot][0] == T:
                acc[rhs[dot][1]] += v[0]

        toks = sorted(acc)
        probs = [float(acc[t]) for t in toks]
        if i == 0:
            stop = g.p_empty
            probs = [p * (1.0 - stop) for p in probs]
        else:
            gam = self.chart[i].get((g.phi, 1, 0))
            stop = float(gam[1]) if gam else 0.0
        if stop > 0.0:
            toks, probs = [EOS] + toks, [stop] + probs
        return toks, probs


PCFG.State = EarleyState


class WalkState:
    """`pending` is the distribution over the next node given the tokens so
    far, rescaled at each step so `continuations` needs no denominator. The
    grammar decides how a step is taken, so fixed-length and halting walks use
    the same state."""

    def __init__(self, grammar):
        self.g = grammar
        self.pending = grammar.pi.copy()
        self.alphas = []                 # belief over the node at each position
        self.stop = 0.0                  # a walk emits at least one token
        self.log_scale = 0.0
        self.tokens = []
        self.viable = True

    def push(self, token):
        g = self.g
        self.tokens.append(token)
        rows = g.by_token.get(token)
        if rows is None or not self.viable:
            self.viable = False
            return False
        a = np.zeros(len(g.pi))
        a[rows] = self.pending[rows]
        c = a.sum()
        if c <= 0.0:
            self.viable = False
            return False
        a /= c
        self.alphas.append(a)
        self.pending, self.stop = g.advance(a, len(self.tokens))
        self.log_scale += float(np.log(c))
        return True

    def continuations(self):
        """(tokens, probs): the exact next-token distribution, token 0 = EOS."""
        if not self.viable:
            return [], []
        acc = {t: float(self.pending[r].sum()) for t, r in self.g.by_token.items()}
        toks = [t for t in sorted(acc) if acc[t] > 0.0]
        probs = [acc[t] for t in toks]
        if self.stop > 0.0:
            toks, probs = [EOS] + toks, [self.stop] + probs
        return toks, probs

    @property
    def prefix_logprob(self):
        if not self.viable:
            return -np.inf
        return self.log_scale + float(np.log(self.g.p_finite))

    @property
    def sentence_logprob(self):
        if not self.viable or not self.tokens or self.stop <= 0.0:
            return -np.inf
        return self.log_scale + float(np.log(self.stop * self.g.p_finite))

    @property
    def complete(self):
        return self.sentence_logprob > -np.inf

    def tree(self, rng=None):
        """Sample a walk from P(walk | sentence), backwards through the
        forward probabilities, as a right-linear tree."""
        if not self.complete:
            raise ValueError("the tokens so far are not a complete sentence")
        rng = np.random.default_rng() if rng is None else rng
        g = self.g
        states = np.arange(len(g.pi))
        n = len(self.tokens)
        path = [_pick(rng, states, self.alphas[-1] * g.back(n))]
        for t in range(n - 2, -1, -1):
            path.append(_pick(rng, states, self.alphas[t] * g.link(t + 1)[:, path[-1]]))
        return self._tree(path[::-1])

    def best_tree(self):
        """The most probable walk (Viterbi), as a right-linear tree."""
        if not self.complete:
            raise ValueError("the tokens so far are not a complete sentence")
        g = self.g
        n = len(self.tokens)
        with np.errstate(divide="ignore"):
            v = np.log(self.alphas[0])
            back = []
            for t in range(1, n):
                cand = v[:, None] + np.log(g.link(t))
                cand[:, self.alphas[t] <= 0] = -np.inf
                back.append(np.argmax(cand, axis=0))
                v = cand[back[-1], np.arange(len(v))]
            path = [int(np.argmax(v + np.log(g.back(n))))]
        for t in range(n - 2, -1, -1):
            path.append(int(back[t][path[-1]]))
        return self._tree(path[::-1])

    def _tree(self, path):
        node = (self.g.nt_name[path[-1]], (self.tokens[-1],))
        for t in range(len(path) - 2, -1, -1):
            node = (self.g.nt_name[path[t]], (self.tokens[t], node))
        return node


class RegularGrammar(Sequential):
    """A networkx graph as a regular grammar: a node is a state emitting one
    terminal on arrival (its `label` attribute, or its name), an edge is a
    transition weighted by `weight`, and a sentence is the label sequence of a
    walk. Three ways for a walk to end, in order of precedence:

      length=n        exactly n tokens, no halting: a plain random walk
      mean_length=k   halt at rate 1/k, so lengths are geometric with mean k
      neither         the graph's sinks halt; used automatically when it has
                      any, otherwise mean_length falls back to 10

    Walks that cannot finish are conditioned away, so `continuations` always
    sums to 1 and `p_finite` says how much mass that discarded: below 1 when
    some region cannot reach a sink, or cannot support a walk of the required length.
    """

    State = WalkState

    def __init__(self, graph, init=None, length=None, mean_length=None,
                 weights="weight", label="label"):
        if length is not None and mean_length is not None:
            raise ValueError("give length or mean_length, not both")
        nodes = list(graph)
        if not nodes:
            raise ValueError("the graph has no nodes")
        idx = {s: i for i, s in enumerate(nodes)}
        m = len(nodes)

        P = np.zeros((m, m))
        for s in nodes:
            for t, d in graph.adj[s].items():
                P[idx[s], idx[t]] += float(d.get(weights, 1.0))
        row = P.sum(1, keepdims=True)
        A0 = np.divide(P, row, out=np.zeros_like(P), where=row > 0)

        if length is None and mean_length is None and not np.any(row <= 0):
            mean_length = 10.0           # no sinks, so nothing would ever stop
        self.length = None if length is None else int(length)
        self.mean_length = mean_length
        if self.length is not None and self.length < 1:
            raise ValueError("length must be at least 1")
        if mean_length is not None and mean_length < 1:
            raise ValueError("mean_length must be at least 1")

        pi = np.zeros(m)
        if init is None:
            w = np.array([float(graph.nodes[s].get("start", 0.0)) for s in nodes])
            pi = w if w.sum() > 0 else np.ones(m)
        elif isinstance(init, dict):
            for s, w in init.items():
                pi[idx[s]] = float(w)
        elif isinstance(init, (list, tuple, set)):
            for s in init:
                pi[idx[s]] = 1.0
        else:
            pi[idx[init]] = 1.0
        if pi.sum() <= 0:
            raise ValueError("the start distribution is empty")
        pi /= pi.sum()

        if self.length is not None:
            # Z[k, s] = P(k further steps are possible from s). Conditioning is
            # time-dependent here, so it is applied per step in `advance`.
            Z = np.ones((self.length, m))
            for k in range(1, self.length):
                Z[k] = A0 @ Z[k - 1]
            self.Z, self.A0 = Z, A0
            self.p_finite = float(pi @ Z[-1])
            if self.p_finite <= _TOL:
                raise ValueError(f"no walk of length {self.length} exists")
            pi = pi * Z[-1] / self.p_finite
            keep = np.arange(m)
        else:
            h = (np.full(m, 1.0 / float(mean_length)) if mean_length is not None
                 else np.zeros(m))
            h[row[:, 0] <= 0] = 1.0                   # a sink can only halt
            A = (1.0 - h)[:, None] * A0
            # z[s] = P(the walk from s halts eventually), solved only where a
            # halt is reachable -- a halt-free cycle makes I - A singular
            can = h > 0
            while True:
                nxt = can | ((A[:, can].sum(1) > 0) if can.any() else False)
                if np.array_equal(nxt, can):
                    break
                can = nxt
            z = np.zeros(m)
            if can.any():
                sub = np.where(can)[0]
                z[sub] = np.linalg.solve(np.eye(len(sub)) - A[np.ix_(sub, sub)], h[sub])
            z = np.clip(z, 0.0, 1.0)
            self.p_finite = float(pi @ z)
            if self.p_finite <= _TOL:
                raise ValueError("no walk can halt: pass length or mean_length")
            grow = set(np.where((z > _TOL) & (pi > 0))[0])
            while True:
                more = {t for s in grow for t in np.where(A[s] > 0)[0] if z[t] > _TOL}
                if more <= grow:
                    break
                grow |= more
            keep = np.array(sorted(grow))
            A, h, z = A[np.ix_(keep, keep)], h[keep], z[keep]
            pi = pi[keep]
            # condition on halting (an h-transform by z), so the chain is proper
            self.A = A * z[None, :] / z[:, None]
            self.halt = h / z
            pi = pi * z / (pi @ z)

        self.pi = pi
        nodes = [nodes[i] for i in keep]
        labels = [str(graph.nodes[s].get(label, s)) for s in nodes]
        # node names like (0, 1) are not single characters, so strings of them
        # need a separator; `sep` is empty whenever the labels are single chars
        self.sep = '' if all(len(c) == 1 for c in labels) else '|'
        if self.sep and any(self.sep in c for c in labels):
            raise ValueError(f"node labels may not contain {self.sep!r}")
        self.vocab = [None] + sorted(set(labels))
        self.token_of = {c: i for i, c in enumerate(self.vocab) if i}
        emit = np.array([self.token_of[c] for c in labels])
        self.by_token = {t: np.where(emit == t)[0] for t in range(1, len(self.vocab))}
        self.states = nodes
        self.nt_name = [str(s) for s in nodes]

    # -- how one step is taken, which is all that differs between the modes --

    def advance(self, a, t):
        """Given the belief `a` over the node that emitted token t, return the
        belief over the next node and the probability of stopping here."""
        if self.length is None:
            return a @ self.A, float(a @ self.halt)
        rem = self.length - t
        if rem <= 0:
            return np.zeros_like(a), 1.0
        den = np.where(self.Z[rem] > 0, self.Z[rem], 1.0)
        return ((a / den) @ self.A0) * self.Z[rem - 1], 0.0

    def link(self, t):
        """Transition matrix used between tokens t and t+1."""
        if self.length is None:
            return self.A
        rem = self.length - t
        if rem <= 0:
            return np.zeros_like(self.A0)
        den = np.where(self.Z[rem] > 0, self.Z[rem], 1.0)
        return self.A0 * self.Z[rem - 1][None, :] / den[:, None]

    def back(self, n):
        """Probability of stopping at each node after n tokens."""
        if self.length is None:
            return self.halt
        return np.ones(len(self.pi)) if n >= self.length else np.zeros(len(self.pi))


def bracket(tree, g):
    """(S (E (T (F a))) ...) -- a bracketed phrase structure."""
    label, kids = tree
    if not kids:
        return f"({label})"
    parts = [g.vocab[k] if isinstance(k, int) else bracket(k, g) for k in kids]
    return f"({label} {' '.join(parts)})"


def label_paths(tree):
    """One tuple of labels per token, root first. The last entry of each is the
    token's preterminal, so [p[-1] for p in label_paths(t)] is a tag sequence."""
    out = []

    def walk(node, path):
        label, kids = node
        path += (label,)
        for k in kids:
            out.append(path) if isinstance(k, int) else walk(k, path)

    walk(tree, ())
    return out