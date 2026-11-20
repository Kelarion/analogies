"""
Learned token categories: binary latent codes fitted by sbmf (github.com/Kelarion/concepts)
to a model's token embeddings, stored per model, for geometry (ICLModel's group and categories).

A stored fit is named by method and dimension, e.g. 'semibmf300' or 'bipca300', and fitted on
the embedding the geometry measures similarity with ('input' or 'canonical', see geometry):

    ICLModel(..., group={'block'}, categories={'semibmf300'}, embedding='canonical')
        reads $ICL_CATEGORIES/<model>/semibmf300-canonical.npz for the run's model

ICL_CATEGORIES defaults to $SCRATCH/icl_categories, and <model> is the model's name with '/'
as '--' (as in the Hugging Face cache). Everything else about a fit is shared by every model
and kept here: METHODS holds each method's hyperparameters, in sbmf's own names (constructor
fields under 'model', fit() keywords under 'fit'); FIT_TOKENS is the vocabulary every fit is
made on (later pools are matched to it by token string); embeddings are rescaled by one
scalar to unit standard deviation (sbmf's intercept absorbs their mean). The entries are
meant to be chosen once by a sweep (bmf_sweep.py) and then left alone: each file records the
entry it was fitted with, and load() refuses one that no longer matches (regenerating keeps
the old file as <name>-<embedding>.stale-<date>.npz).

Files are made on demand, like model weights: a run whose geometry names stored codes
declares them (icl_model.ICLModel.prerequisites), and server_utils.send_to_server checks for
them on the cluster and, for any missing or stale, first runs a job that makes them
(jobs/make_categories.py, through jobs/prerequisite_template.sh), with the array waiting for
it. By hand:

    python jobs/make_categories.py MODEL semibmf300 --embedding canonical      # make (if needed)
    python jobs/make_categories.py MODEL semibmf300 --embedding canonical --check
    python jobs/make_categories.py MODEL --list

A model's embeddings of FIT_TOKENS are cached alongside (embeddings-<embedding>.npz), so a
fit, or a sweep's many fits, load the weights at most once.
"""
import json
import os
import re
import shutil
import tempfile
import time

import numpy as np

import vocabulary as voc

# Hyperparameters shared by every fit of a method. PROVISIONAL: sbmf's defaults (and its
# tests' learning rates) until bmf_sweep picks better ones. Changing an entry makes codes
# fitted with the old one stale: load() refuses them until they are regenerated.
METHODS = {
    'semibmf': {'class': 'SemiBMF', 'model': {}, 'fit': {'lr': 0.1, 'max_iter': 600}},
    'bipca': {'class': 'BiPCA', 'model': {}, 'fit': {'lr': 0.25, 'max_iter': 600}},
}
FIT_TOKENS = voc.Vocabulary('words')   # what every fit is made on
EMBEDDINGS = ('input', 'canonical')
PREPROCESS = 'one scalar: X / X.std()'
SEED = 0
NAME = re.compile(r'^([a-z]+)([0-9]+)$')


# -- where things are ---------------------------------------------------------------------

def root():
    """$ICL_CATEGORIES, else $SCRATCH/icl_categories (else under ~)."""
    base = os.environ.get('SCRATCH', os.path.expanduser('~'))
    return os.environ.get('ICL_CATEGORIES', os.path.join(base, 'icl_categories'))


def model_dir(model):
    return os.path.join(root(), model.replace('/', '--'))


def parse_name(name):
    """(method, dim) of a stored fit's name, e.g. 'semibmf300' -> ('semibmf', 300)."""
    m = NAME.match(name) if isinstance(name, str) else None
    if not m or m.group(1) not in METHODS or int(m.group(2)) < 1:
        raise ValueError(f"{name!r} isn't a stored fit's name: <method><dim>, method one of "
                         f"{sorted(METHODS)}, e.g. 'semibmf300'")
    return m.group(1), int(m.group(2))


def is_name(name):
    try:
        parse_name(name)
        return True
    except ValueError:
        return False


def _check_embedding(embedding):
    if embedding not in EMBEDDINGS:
        raise ValueError(f"stored codes are fitted on one of {EMBEDDINGS}, not {embedding!r}")


def path(model, name, embedding):
    parse_name(name); _check_embedding(embedding)
    return os.path.join(model_dir(model), f'{name}-{embedding}.npz')


def embeddings_path(model, embedding):
    _check_embedding(embedding)
    return os.path.join(model_dir(model), f'embeddings-{embedding}.npz')


# -- what a fit is ----------------------------------------------------------------------------

def recipe(name):
    """Everything that determines a stored fit, apart from model and embedding: what its file
    records, and what it must still match to be used."""
    method, dim = parse_name(name)
    return {'method': method, 'dim': dim, **json.loads(json.dumps(METHODS[method])),
            'fit_tokens': repr(FIT_TOKENS), 'preprocess': PREPROCESS, 'seed': SEED}


def _write(p, **arrays):
    """np.savez to p atomically, so a job dying midway leaves nothing half-written."""
    os.makedirs(os.path.dirname(p), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(p), suffix='.part')
    os.close(fd)
    try:
        with open(tmp, 'wb') as f:
            np.savez(f, **arrays)
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _read_meta(p):
    with np.load(p, allow_pickle=False) as z:
        return json.loads(str(z['meta']))


def status(model, name, embedding):
    """'ok', 'missing', or 'stale: <why>' for a stored fit."""
    p = path(model, name, embedding)
    if not os.path.exists(p):
        return 'missing'
    meta = _read_meta(p)
    if meta.get('model') != model:
        return f"stale: made for {meta.get('model')!r}"
    want = recipe(name)
    diff = sorted(k for k in want if meta.get('recipe', {}).get(k) != want[k])
    return f"stale: {', '.join(diff)} changed since it was fitted" if diff else 'ok'


def load(model, name, embedding):
    """(tokens, codes, meta) of a stored fit, if it is current."""
    s = status(model, name, embedding)
    if s != 'ok':
        how = f"python jobs/make_categories.py {model} {name} --embedding {embedding}"
        raise FileNotFoundError(f"codes {name!r} ({embedding}) for {model} are {s} "
                                f"({path(model, name, embedding)}); make them with: {how} "
                                f"(send_to_server does this itself)")
    with np.load(path(model, name, embedding), allow_pickle=False) as z:
        return z['tokens'], z['codes'].astype(bool), json.loads(str(z['meta']))


def available(model):
    """{'<name>-<embedding>': status} of the fits stored for a model."""
    d = model_dir(model)
    out = {}
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        stem = f[:-4]
        if f.endswith('.npz') and not f.startswith('embeddings-') and '.stale-' not in f:
            name, _, emb = stem.rpartition('-')
            out[stem] = status(model, name, emb) if is_name(name) and emb in EMBEDDINGS else 'unrecognised'
    return out


# -- embeddings -------------------------------------------------------------------------------

def save_embeddings(model, embedding, tokens, vectors, **meta):
    """Cache a model's embeddings of FIT_TOKENS (as made by embeddings(), or given)."""
    meta = dict(meta, model=model, embedding=embedding, fit_tokens=repr(FIT_TOKENS),
                created=time.strftime('%Y-%m-%dT%H:%M:%S'))
    _write(embeddings_path(model, embedding), tokens=np.asarray([str(t) for t in tokens]),
           vectors=np.asarray(vectors, np.float16), meta=np.array(json.dumps(meta)))


def embeddings(model, embedding, tokenizer=None):
    """(tokens, vectors as float32): the model's embeddings of FIT_TOKENS, from the cache, else
    computed from its weights (which loads the model) and cached."""
    p = embeddings_path(model, embedding)
    if os.path.exists(p):
        with np.load(p, allow_pickle=False) as z:
            meta = json.loads(str(z['meta']))
            if meta.get('model') == model and meta.get('fit_tokens') == repr(FIT_TOKENS):
                return z['tokens'], z['vectors'].astype(np.float32)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import geometry
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(model)
    pool = voc.get_pool(FIT_TOKENS, tokenizer)
    hf = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16, device_map='auto')
    X = geometry.model_vectors(hf, pool.ids, embedding, len(tokenizer))
    tokens = [pool.strs[i] for i in pool.ids]
    save_embeddings(model, embedding, tokens, X, tokenizer=getattr(tokenizer, 'name_or_path', None))
    return np.asarray(tokens), X.astype(np.float32)


def preprocess(X):
    """The rescaling every fit uses (PREPROCESS)."""
    X = np.asarray(X, np.float64)
    return X / X.std()


# -- fitting ----------------------------------------------------------------------------------

def build(method, dim, **model_overrides):
    """An sbmf model for a method, with METHODS' constructor fields (and any overrides)."""
    from sbmf import bae_models
    spec = METHODS[method]
    return getattr(bae_models, spec['class'])(dim, **{**spec['model'], **model_overrides})


def fit_settings(method, **overrides):
    """METHODS' fit() keywords for a method (and any overrides), with a prior_schedule given by
    name (e.g. 'AdaptiveTemp') made into the sbmf object."""
    s = {**METHODS[method]['fit'], **overrides}
    if isinstance(s.get('prior_schedule'), str):
        from sbmf import bae_priors
        s['prior_schedule'] = getattr(bae_priors, s['prior_schedule'])()
    return s


def seed_all(seed):
    """Seed numpy's and numba's random streams, as sbmf's own cross-validation does."""
    np.random.seed(seed)
    try:
        from sbmf.bae_util import _seed_numba
        _seed_numba(seed)
    except ImportError:
        pass


def fit_codes(X, method, dim, seed=SEED):
    """Binary codes (n x dim) for the rows of X, by METHODS[method]."""
    seed_all(seed)
    model = build(method, dim)
    model.fit(preprocess(X), verbose=False, **fit_settings(method))
    if getattr(model, 'n_chains', 1) > 1:
        model.collapse()
    return np.asarray(model.S) > 0.5


def generate(model, name, embedding, tokenizer=None, overwrite=False):
    """Make a stored fit if it is missing or stale (or overwrite); returns its path. A current
    fit is left alone, so a second job asked for the same one does nothing."""
    s = status(model, name, embedding)
    p = path(model, name, embedding)
    if s == 'ok' and not overwrite:
        print(f'{p} is current; nothing to do', flush=True)
        return p
    if s.startswith('stale'):
        keep = p[:-4] + time.strftime('.stale-%Y%m%d-%H%M%S.npz')
        shutil.move(p, keep)
        print(f'{p} was {s}; kept as {keep}', flush=True)
    method, dim = parse_name(name)
    tokens, X = embeddings(model, embedding, tokenizer)
    t = time.time()
    codes = fit_codes(X, method, dim)
    try:
        import sbmf
        version = sbmf.__version__
    except ImportError:
        version = None
    meta = {'model': model, 'name': name, 'embedding': embedding, 'recipe': recipe(name),
            'n_tokens': int(len(tokens)), 'sbmf': version, 'fit_seconds': round(time.time() - t, 1),
            'mean_on': float(codes.mean()), 'created': time.strftime('%Y-%m-%dT%H:%M:%S')}
    _write(p, tokens=np.asarray([str(t) for t in tokens]), codes=codes, meta=np.array(json.dumps(meta)))
    print(f'saved {p}: {len(tokens)} tokens x {dim}, {codes.mean():.3f} on, in {meta["fit_seconds"]}s', flush=True)
    return p


# -- for send_to_server ---------------------------------------------------------------------

def prerequisite(model, name, embedding):
    """What send_to_server needs to make sure a stored fit exists before a sweep runs: commands
    (scripts in the repo and their arguments) to check for it and to make it."""
    parse_name(name); _check_embedding(embedding)
    args = ['jobs/make_categories.py', model, name, '--embedding', embedding]
    return {'key': f'categories {model} {name}-{embedding}', 'check': args + ['--check'],
            'make': args, 'what': f'{name} codes ({embedding} embedding) for {model}'}


def embeddings_prerequisite(model, embedding):
    """The same, for a model's cached embeddings of FIT_TOKENS (what a sweep needs)."""
    _check_embedding(embedding)
    args = ['jobs/make_categories.py', model, '--embeddings', embedding]
    return {'key': f'embeddings {model} {embedding}', 'check': args + ['--check'],
            'make': args, 'what': f'{embedding} embeddings of the fit vocabulary for {model}'}


def embeddings_status(model, embedding):
    p = embeddings_path(model, embedding)
    if not os.path.exists(p):
        return 'missing'
    meta = _read_meta(p)
    ok = meta.get('model') == model and meta.get('fit_tokens') == repr(FIT_TOKENS)
    return 'ok' if ok else 'stale: made for another model or fit vocabulary'