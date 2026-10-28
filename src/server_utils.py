import os

# Where things live, from the environment so the same file works on the laptop and the cluster.
# SAVE_DIR: results (Experiment.save_experiment / load_experiments).
# SWEEP_DIR: one folder per submitted sweep: its task/model pickles, job script and logs.
_BASE = os.environ.get('SCRATCH', os.path.expanduser('~'))
SAVE_DIR = os.environ.get('ICL_RESULTS', os.path.join(_BASE, 'icl_results')) + '/'
SWEEP_DIR = os.environ.get('ICL_SWEEPS', os.path.join(_BASE, 'icl_sweeps'))
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # src/.. = repo root
JOB_TEMPLATE = os.path.join(REPO_DIR, 'jobs', 'array_template.sh')
WORKER = os.path.join(REPO_DIR, 'jobs', 'run_experiment.py')
DOWNLOAD_TEMPLATE = os.path.join(REPO_DIR, 'jobs', 'download_template.sh')
WEIGHTS = 'jobs/weights.py'          # relative to the repo, run on the cluster
STATUS = 'jobs/sweep_status.py'

# Submitting from elsewhere (e.g. the laptop): the ssh host, and the repo's path there.
REMOTE_HOST = os.environ.get('ICL_REMOTE_HOST', 'mila')
REMOTE_REPO = os.environ.get('ICL_REMOTE_REPO', '~/analogies')

import socket
import sys
import time
import json
import shlex
import shutil
import glob
import importlib
import pickle as pkl
import subprocess
import copy
from dataclasses import dataclass, fields, is_dataclass

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import itertools as itt

import util

class Build:
    """
    A deferred constructor call, for task_args / mod_args values that are expensive to make:

        'grammar': Build(auto.FSA, Build(auto.cycle, 5), length=1)

    stands for auto.FSA(auto.cycle(5), length=1). Nothing is built when the sweep is
    defined or submitted: Task and Model call build() on their fields when they are
    constructed, i.e. inside the job (nested Builds first). The un-built Build is what is
    recorded in .args, hashed into the save name, and matched by load_experiments, via its
    repr, which spells out the call: automata.FSA(automata.cycle(5), length=1).
    """
    def __init__(self, func, *args, **kwargs):
        self.func, self.args, self.kwargs = func, args, kwargs

    def build(self):
        return self.func(*resolve(self.args), **resolve(self.kwargs))

    def __repr__(self):
        name = f'{self.func.__module__}.{self.func.__qualname__}'
        parts = [_build_repr(a) for a in self.args]
        parts += [f'{k}={_build_repr(v)}' for k, v in sorted(self.kwargs.items())]
        return f'{name}({", ".join(parts)})'

    def __eq__(self, other):
        return isinstance(other, Build) and repr(self) == repr(other)

    def __hash__(self):
        return hash(repr(self))


def _build_repr(v):
    if isinstance(v, np.ndarray):
        return f'array({v.tolist()!r})'           # numpy's own repr truncates large arrays
    if isinstance(v, (list, tuple)):
        inner = ', '.join(_build_repr(x) for x in v)
        return f'[{inner}]' if isinstance(v, list) else f'({inner})'
    if isinstance(v, dict):
        return '{' + ', '.join(f'{k!r}: {_build_repr(x)}' for k, x in sorted(v.items(), key=repr)) + '}'
    if isinstance(v, type) or (callable(v) and hasattr(v, '__qualname__')):
        return f'{v.__module__}.{v.__qualname__}'
    return repr(v)


def contains_build(v):
    if isinstance(v, Build):
        return True
    if isinstance(v, (list, tuple)):
        return any(contains_build(x) for x in v)
    if isinstance(v, dict):
        return any(contains_build(x) for x in v.values())
    return False


def resolve(v):
    """v with every Build in it (including inside lists, tuples and dicts) built."""
    if isinstance(v, Build):
        return v.build()
    if isinstance(v, (list, tuple)) and contains_build(v):
        return type(v)(resolve(x) for x in v)
    if isinstance(v, dict) and contains_build(v):
        return {k: resolve(x) for k, x in v.items()}
    return v


class Parameter(np.lib.mixins.NDArrayOperatorsMixin):
    """
    Define an parameter and its range using symbolic expressions. Can be used to 
    define dependencies between iterators by abusing python's native syntax. 

    For example, to iterate k over integers from 1 to 5, and to make j go from 
    1 to 2^k, you'd write:

    1 << k << 5
    1 << j << 2**k
    """

    def __init__(self):

        self.funcs = []
        self.value = None
        self.root_instance = True

        self.checks = [] # logical functions of the parameter (constraints via `|`)

    def generate_values(self):
        """
        This function needs to be implemented in descendent classes, 
        it should return an interable
        """

        raise NotImplementedError

    def __iter__(self):

        return ParamSet(self).__iter__()

    def __bool__(self):

        return bool(np.all([c() for c in self.checks]))

    def __eq__(self, other):

        if isinstance(other, Parameter):
            return id(self) == id(other)
        else:
            return False

    def __hash__(self):

        return id(self)

    def __deepcopy__(self, memo):
        """
        Custom behavior for deepcopy, so root refers to original instance
        """

        clss = self.__class__
        result = clss.__new__(clss)
        memo[id(self)] = result
        for k,v in self.__dict__.items():
            if k in ['roots', 'checks']:
                setattr(result, k, v)
            elif k in ['funcs']:
                setattr(result, k, copy.copy(v))
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        result.root_instance = False

        return result

    def __call__(self, *vals):

        if len(vals) == len(self.roots):
            for r,v in zip(self.roots, vals):
                r.value = v

        if self.root_instance:
            return self.value

        for f, args, kwargs in self.funcs:
            arrgs = []
            for arg in args:
                if isinstance(arg, Parameter):
                    arrgs.append(arg())
                else:
                    arrgs.append(arg)
            ret = f(*tuple(arrgs), **kwargs)

        return ret

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Keep track of functions, which will be applied to value
        """

        # logics = ['greater', 'less', 'greater_than', 'less_than', 'equal']

        if method == '__call__':

            clone = copy.deepcopy(self)

            ## Union the roots of Parameter inputs
            for this_inp in inputs:
                if isinstance(this_inp, Parameter):
                    if this_inp.root_instance:
                        clone.roots = clone.roots | {this_inp}
                    else:
                        clone.roots = clone.roots | this_inp.roots

            clone.funcs += [(ufunc, inputs, kwargs)]

            return clone

        else:
            return NotImplemented

    def __or__(self, other):
        # print('or')

        if isinstance(other, tuple):
            self.checks.extend(other)
        else:
            self.checks.append(other)

        return self

    def __rshift__(self, other):

        if self.root_instance:
            return ParamSet(self) >> other
        elif isinstance(other, Parameter):
            if other.root_instance:
                return ParamSet(other) << self
            else:
                return ParamSet(self) >> other
        elif isinstance(other,ParamSet):
            return other >> self
        else:
            return ParamSet(self) >> other

    def __lshift__(self, other):

        if self.root_instance:
            return ParamSet(self) << other
        elif isinstance(other, Parameter):
            if other.root_instance:
                return ParamSet(other) >> self
            else:
                return ParamSet(self) << other
        elif isinstance(other,ParamSet):
            return other >> self
        else:
            return ParamSet(self) << other

    def __rrshift__(self, other):

        if self.root_instance:
            return ParamSet(self) << other
        elif isinstance(other, Parameter):
            if other.root_instance:
                return ParamSet(other) >> self
            else:
                return ParamSet(self) << other
        elif isinstance(other,ParamSet):
            return other >> self
        else:
            return ParamSet(self) << other

    def __rlshift__(self, other):

        if self.root_instance:
            return ParamSet(self) >> other
        elif isinstance(other, Parameter):
            if other.root_instance:
                return ParamSet(other) << self
            else:
                return ParamSet(self) >> other
        elif isinstance(other,ParamSet):
            return other >> self
        else:
            return ParamSet(self) >> other


## Clothing
class Set(Parameter):
    """
    A parameter which takes values from a pre-defined set
    """

    def __init__(self, values):

        super(Set, self).__init__()

        self.values = values

        roots = set()
        for val in values:
            roots = roots | _param_roots(val)

        if len(roots) > 0:
            self.roots = roots
        else:
            self.roots = {self}

    def generate_values(self):

        return [_eval_params(v) for v in self.values]


def _param_roots(v):
    """The root parameters a value depends on, looking inside lists, tuples and dicts
    (so Set([[k, 2 * k]]) depends on k)."""
    if isinstance(v, Parameter):
        return set(v.roots)
    if isinstance(v, (list, tuple)):
        return set().union(*(_param_roots(x) for x in v)) if v else set()
    if isinstance(v, dict):
        return set().union(*(_param_roots(x) for x in v.values())) if v else set()
    return set()


def _eval_params(v):
    """v with every parameter in it (inside lists, tuples and dicts too) evaluated."""
    if isinstance(v, Parameter):
        return v()
    if isinstance(v, (list, tuple)):
        return type(v)(_eval_params(x) for x in v)
    if isinstance(v, dict):
        return {k: _eval_params(x) for k, x in v.items()}
    return v


class Integer(Parameter):

    def __init__(self, num=None, step=None):

        super(Integer, self).__init__()

        self.num = num
        self.step = step

        self.roots = {self} # base parameter, without any functions applied

    def generate_values(self, lb, ub):

        if self.step is not None:
            step = self.step
        elif self.num is not None:
            # `num` sets the step via floor-division; clamp to >=1 so that asking
            # for more points than the integer span gives the whole span rather
            # than a zero step (np.arange ZeroDivisionError).
            step = max(1, int((ub - lb) // self.num))
        else:
            raise ValueError('Need to define either step size or number')

        return np.arange(lb, ub+1, step)


class Real(Parameter):

    def __init__(self, num=None, step=None):

        super(Real, self).__init__()

        self.num = num
        self.step = step

        self.roots = {self} # base parameter, without any functions applied

    def generate_values(self, lb, ub):

        if self.num is not None:
            num = self.num
        elif self.step is not None:
            num = int((ub - lb) // self.step)
        else:
            raise ValueError('Need to define either step size or number')

        return np.linspace(lb, ub, num=num) 


## utility classes that should be hidden to user
class ParamSet:
    """
    A set of values of a parameter
    """

    def __init__(self, param):

        self.param = param

        self.lb = None 
        self.ub = None 

    def __iter__(self):

        ## Disallow iteration with parametric bounds
        if isinstance(self.lb, Parameter) or isinstance(self.ub, Parameter):
            raise TypeError('Not iterable: range depends on other parameters')

        return ParamIter(self).__iter__()

    def init_values(self):
        """
        Return the set of values which satisfy constraints
        """

        # if self.param.root_instance:
        if isinstance(self.param, Set):
            values = self.param.generate_values()

        else:
            if isinstance(self.lb, Parameter):
                lb = self.lb()
            else:
                lb = self.lb

            if isinstance(self.ub, Parameter):
                ub = self.ub()
            else:
                ub = self.ub

            values = self.param.generate_values(lb, ub)

        if len(values) == 0:          # e.g. a range whose bounds cross for this outer value
            return iter(())
        self.param.value = values[0]

        if self.param.root_instance:
            return filter(self.valid , values) 
        else:
            return values.__iter__()


    def dependencies(self):
        """
        Return a set of all parameters which constrain this set
        """

        deps = self.param.roots - {self.param}
        for c in self.param.checks:
            deps = deps | c.roots - {self.param}

        if isinstance(self.lb, Parameter):
            deps = deps | self.lb.roots
            
        if isinstance(self.ub, Parameter):
            deps = deps | self.ub.roots

        return deps 

    def valid(self, val):

        # if self.param not in self.param.roots:
        #     raise Exception('Can only check constraints on root parameters')
        # else:
        self.param.value = val

        return bool(np.all([c() for c in self.param.checks]))

    def __bool__(self):
        return bool(np.all([c() for c in self.param.checks]))

    def __rshift__(self, other):
        # print('rshift')
        self.lb = other

        return self

    def __lshift__(self, other):
        # print('lshift')
        self.ub = other

        return self

    def __rrshift__(self, other):
        # print('rrshift')
        self.ub = other

        return self

    def __rlshift__(self, other):
        # print('rlshift')
        self.lb = other

        return self

    def __mod__(self, other):

        if isinstance(other, ParamIter):
            return ParamIter(self, *other.sets)
        elif isinstance(other, ParamSet):
            return ParamIter(self, other)
        elif isinstance(other, Parameter):
            return ParamIter(self, ParamSet(other))
        else:
            raise Exception('Multiplication not defined for %s'%type(other))

    def __rmod__(self, other):

        if isinstance(other, ParamIter):
            return ParamIter(self, *other.sets)
        elif isinstance(other, ParamSet):
            return ParamIter(self, other)
        elif isinstance(other, Parameter):
            return ParamIter(self, ParamSet(other))
        else:
            raise Exception('Multiplication not defined for %s'%type(other))

    def __or__(self, other):
        # print('or')

        if isinstance(other, tuple):
            self.param.checks.extend(other)
        else:
            self.param.checks.append(other)

        return self

class ParamIter:
    """
    Iterator over the product of parameter sets
    """

    def __init__(self, *sets):

        self.sets = sets

    def __iter__(self):

        params = [s.param for s in self.sets]

        ## Construct a partial order of parameter dependencies
        self.couplings = {i: set() for i in range(len(self.sets))}
        for i,s in enumerate(self.sets):

            for dep in s.dependencies():
                if dep in params:
                    j = params.index(dep)
                    self.couplings[j].add(i)
                else:
                    raise Exception('Constraining parameters not included!')

        ## coupling graph needs to be acyclic
        if util.is_cyclic(self.couplings):
            raise Exception("Parameters have cyclic dependencies, that's bad (`^`)")

        self.order = util.topsort(self.couplings)
        self.inv_ord = np.argsort(self.order)

        return self._combos(0, [None] * len(self.order))

    def _combos(self, depth, current):
        """Nested for loops over the sets in dependency order. Each set's values are made
        once the sets it depends on hold their current values, so a range that is empty
        for some outer value skips just that value."""
        if depth == len(self.order):
            yield tuple(current[i] for i in self.inv_ord)
            return
        s = self.sets[self.order[depth]]
        for val in s.init_values():
            if s.param.root_instance:
                s.param.value = val       # what the sets below see
            current[depth] = val
            yield from self._combos(depth + 1, current)

    def __mod__(self, other):

        ## Support products between paramter sets, parameters, lists, and tuples

        if isinstance(other, ParamIter):
            return ParamIter(*self.sets, *other.sets)
        elif isinstance(other, ParamSet):
            return ParamIter(*self.sets, other)
        elif isinstance(other, Parameter):
            return ParamIter(*self.sets, ParamSet(other))
        else:
            raise Exception('Multiplication not defined for %s'%type(other))

    def __rmod__(self, other):

        ## Support products between paramter sets, parameters, lists, and tuples

        if isinstance(other, ParamIter):
            return ParamIter(*self.sets, *other.sets)
        elif isinstance(other, ParamSet):
            return ParamIter(*self.sets, other)
        elif isinstance(other, Parameter):
            return ParamIter(*self.sets, ParamSet(other))
        else:
            raise Exception('Multiplication not defined for %s'%type(other))


##############################################################################
########### Utility functions ################################################
##############################################################################

def _plain(v):
    """numpy scalars as Python ones, also inside lists, tuples and dicts."""
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (list, tuple)):
        return type(v)(_plain(x) for x in v)
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    return v


def parse_params(dictionary, forbidden_keys=None):
    """
    This takes dict with variable values, and outputs each combination
    """

    if forbidden_keys is None:
        forbidden_keys = []

    variable_prms = {}
    matched_prms = {}
    fixed_prms = {}
    for k,v in dictionary.items():
        if k in forbidden_keys:
            continue
        if isinstance(v, Parameter):
            variable_prms[k] = ParamSet(v)
        elif isinstance(v, ParamSet):
            variable_prms[k] = v
        elif type(v) is list:
            variable_prms[k] = ParamSet(Set(v))
        elif type(v) is tuple:
            matched_prms[k] = v 
        else:
            fixed_prms[k] = v

    ### Handle variables
    if len(matched_prms)>0: 
        if len(np.unique([len(v) for v in matched_prms.values()]))>1:
            raise ValueError('Tuple arguments must all be the same length you dingus!')
        tup_keys = tuple(matched_prms.keys())
        tup_vals = list(zip(*matched_prms.values()))
    else:
        tup_keys = ()
        tup_vals = [()]

    if len(variable_prms)>0:
        var_k, var_v = zip(*variable_prms.items())
    else:
        var_k = ()
        var_v = ()

    ## Make experiment dictionaries (iterators yield numpy scalars: hand out plain Python
    ## numbers, so a swept k=6 gets the same save name as a typed k=6)
    out_dicts = []
    for var_vals in ParamIter(*var_v):
        var_vals = tuple(_plain(v) for v in var_vals)
        for mch_vals in tup_vals:
            out_dicts.append( dict(zip(var_k+tup_keys, var_vals+mch_vals), **fixed_prms) )

    return out_dicts


def _remote(cmd, capture=True):
    """Run `cmd` in a login shell on REMOTE_HOST (so $SCRATCH, PATH etc. are set)."""
    out = subprocess.run(['ssh', REMOTE_HOST, 'bash -lc ' + shlex.quote(cmd)],
                         capture_output=capture, text=True)
    if out.returncode:
        raise RuntimeError(f'on {REMOTE_HOST}: {cmd}\n{out.stderr if capture else ""}')
    return out.stdout.strip() if capture else None


def _git(*args, cwd=REPO_DIR):
    return subprocess.run(['git', *args], cwd=cwd, capture_output=True, text=True).stdout.strip()


def send_to_server(task_args, mod_args, run_remote=True, verbose=False,
                   name=None, max_parallel=8, template=None, remote=None, check_commit=True,
                   ensure_weights=True):
    """
    Submit every task x model combination as one Slurm job array.

    Any values which are list-type are combined independently; tuple-type values are
    matched 1-1 (see parse_params). Only the arguments are pickled; tasks and models are
    built inside the jobs.

    Each call gets its own folder <sweeps>/<name or timestamp>/ with the task_i / model_j
    pickles, the job script and the logs, so a new sweep can't overwrite the inputs of one
    still in the queue. Array task k runs task k % n_task with model k // n_task (see
    jobs/run_experiment.py); results go to the cluster's SAVE_DIR.

    remote: None submits with sbatch here if this machine has it, else over ssh to
        REMOTE_HOST (pickles written here, copied over with rsync). True/False forces it.
    check_commit: when submitting remotely, refuse unless the cluster's checkout is at
        this repo's commit, since the jobs run the cluster's code on pickles made here.
    ensure_weights: if the model class has a model_name field, check that each model in the
        sweep is fully in the cluster's Hugging Face cache; for any that isn't, first submit
        a CPU job that downloads it, and make the array wait for that job to succeed.
    run_remote=False runs the first combination here instead, for debugging.
    max_parallel caps how many array tasks run at once.
    """
    if remote is None:
        remote = run_remote and shutil.which('sbatch') is None

    tasks = parse_params(task_args, forbidden_keys=['task'])
    models = parse_params(mod_args, forbidden_keys=['model'])
    n_task, n_tot = len(tasks), len(tasks) * len(models)
    name = name or time.strftime('%Y%m%d-%H%M%S')

    if remote:
        here = _git('rev-parse', 'HEAD')
        if _git('status', '--porcelain', '--untracked-files=no'):
            print('warning: uncommitted changes here; the jobs will run the cluster\'s code')
        info = _remote('echo $SCRATCH; echo ${ICL_RESULTS:-$SCRATCH/icl_results}; '
                       'echo ${ICL_SWEEPS:-$SCRATCH/icl_sweeps}; '
                       f'cd {REMOTE_REPO} && pwd && git rev-parse HEAD').splitlines()
        scratch, results, sweeps, repo, there = info[-5:]   # last lines, past any login chatter
        if check_commit and here != there:
            raise RuntimeError(f'cluster checkout is at {there[:8]}, this repo at {here[:8]}: '
                               'push here and pull there first (or check_commit=False)')
        sweep = os.path.join(sweeps, name)
    else:
        scratch, repo, results = _BASE, REPO_DIR, SAVE_DIR.rstrip('/')
        sweep = os.path.join(SWEEP_DIR, name)

    local = os.path.join(SWEEP_DIR, name)        # where the files are written
    if os.path.exists(local):
        raise FileExistsError(f'{local} already exists; pick another name')
    os.makedirs(os.path.join(local, 'logs'))

    for i, args in enumerate(tasks):
        with open(os.path.join(local, 'task_%d.pkl' % i), 'wb') as f:
            pkl.dump({'task': task_args['task'], 'args': args}, f)
    for j, args in enumerate(models):
        with open(os.path.join(local, 'model_%d.pkl' % j), 'wb') as f:
            pkl.dump({'model': mod_args['model'], 'args': args}, f)
    if verbose:
        print(f'{n_task} tasks x {len(models)} models = {n_tot} jobs, in {sweep}'
              + (f' on {REMOTE_HOST}' if remote else ''))

    if not run_remote:
        print('Running the first job here...')
        subprocess.check_call([sys.executable, WORKER, '0', str(n_task), local])
        return local

    fill = dict(n_tot=n_tot - 1, n_task=n_task, sweep_dir=sweep, repo=repo, scratch=scratch,
                results=results, max_parallel=max_parallel)
    with open(template or JOB_TEMPLATE) as f:
        script = f.read().format(**fill)
    with open(os.path.join(local, 'job_script.sh'), 'w') as f:
        f.write(script)

    names = _model_names(mod_args, models) if ensure_weights else []
    missing = _missing_weights(names, scratch, repo, remote) if names else {}
    if missing:
        with open(DOWNLOAD_TEMPLATE) as f:
            script = f.read().format(models=' '.join(shlex.quote(m) for m in missing), **fill)
        with open(os.path.join(local, 'download.sh'), 'w') as f:
            f.write(script)

    if remote:
        _remote(f'mkdir -p {sweep}')
        subprocess.run(['rsync', '-a', local + '/', f'{REMOTE_HOST}:{sweep}/'], check=True)

    after = None
    if missing:
        after = _sbatch(os.path.join(sweep, 'download.sh'), remote)
        for m, why in missing.items():
            print(f'Not cached on the cluster: {m}\n    ({why})')
        print(f'Downloading first (job {after}); the array starts once that succeeds.')
    job = _sbatch(os.path.join(sweep, 'job_script.sh'), remote, after=after)
    _record_job(sweep, job, remote)
    print(f'Submitted batch job {job} ({n_tot} array tasks)')
    return sweep


def _model_names(mod_args, models):
    """The model_name of every model in the sweep (the field's default where not given),
    or [] if the model class has no model_name field."""
    cls = mod_args['model']
    if not is_dataclass(cls):
        return []
    default = {f.name: f.default for f in fields(cls)}
    if 'model_name' not in default:
        return []
    return sorted({m.get('model_name', default['model_name']) for m in models})


def _missing_weights(names, scratch, repo, remote):
    """{name: why} for the `names` not fully in the cluster's Hugging Face cache (jobs/weights.py)."""
    hf_home = f'{scratch}/.cache/huggingface'
    if remote:
        args = ' '.join(shlex.quote(n) for n in names)
        out = _remote(f'cd {repo} && HF_HOME={hf_home} .venv/bin/python {WEIGHTS} check {args}')
    else:
        out = subprocess.run([sys.executable, os.path.join(repo, WEIGHTS), 'check', *names],
                             capture_output=True, text=True, check=True,
                             env={**os.environ, 'HF_HOME': hf_home}).stdout
    return json.loads(out.strip().splitlines()[-1])


def _sbatch(script, remote, after=None, extra=()):
    """Submit `script`, optionally to start only once job `after` has succeeded; returns the job id."""
    args = (['sbatch', '--parsable'] + ([f'--dependency=afterok:{after}'] if after else [])
            + list(extra) + [script])
    if remote:
        out = _remote(' '.join(shlex.quote(a) for a in args))
    else:
        res = subprocess.run(args, capture_output=True, text=True)
        if res.returncode:
            raise RuntimeError(res.stderr.strip())
        out = res.stdout
    return out.strip().splitlines()[-1].split(';')[0]



def _record_job(sweep, job, remote):
    """Append an array job id to the sweep's jobids file (read by jobs/sweep_status.py)."""
    if remote:
        _remote(f'echo {job} >> {sweep}/jobids')
    else:
        with open(os.path.join(sweep, 'jobids'), 'a') as f:
            f.write(f'{job}\n')


def _cluster_script(args, remote):
    """Run one of the repo's scripts on the cluster (over ssh if remote); its stdout."""
    if remote:
        return _remote(f'cd {REMOTE_REPO} && .venv/bin/python ' + ' '.join(shlex.quote(a) for a in args))
    return subprocess.run([sys.executable, *args], cwd=REPO_DIR, capture_output=True,
                          text=True, check=True).stdout


def sweep_status(sweep=None, remote=None, verbose=True):
    """
    What happened to each array task of a sweep: whether its result was saved, Slurm's
    state, the node and start time, and for the ones that didn't finish, why (from Slurm
    and the task's log). Prints a summary, including failures per node, and returns
    {'sweep', 'n_tot', 'tasks': {index: {...}}, ...}.

    sweep: the folder name send_to_server printed (or name= you gave it), a path, or None
        for the most recent sweep. remote: as in send_to_server (None: decide by whether
        this machine has sbatch).
    """
    remote = shutil.which('sbatch') is None if remote is None else remote
    out = _cluster_script([STATUS, sweep or ''], remote)
    st = json.loads(out.strip().splitlines()[-1])
    st['tasks'] = {int(i): t for i, t in st['tasks'].items()}
    if verbose:
        _print_status(st)
    return st


def _print_status(st):
    tasks = st['tasks']
    live = {i: t for i, t in tasks.items() if not t['done'] and t['cause'] is None}
    failed = {i: t for i, t in tasks.items() if not t['done'] and t['cause'] is not None}
    print(f"{st['sweep']}: {sum(t['done'] for t in tasks.values())} of {st['n_tot']} done, "
          f"{len(live)} queued or running, {len(failed)} failed")
    if not failed:
        return
    causes = {}
    for t in failed.values():
        causes[t['cause']] = causes.get(t['cause'], 0) + 1
    print('  by cause: ' + ', '.join(f'{c} {n}' for c, n in sorted(causes.items(), key=lambda x: -x[1])))
    ran, bad = {}, {}
    for t in tasks.values():
        if t['node']:
            ran[t['node']] = ran.get(t['node'], 0) + 1
            if not t['done'] and t['cause'] is not None:
                bad[t['node']] = bad.get(t['node'], 0) + 1
    if bad:
        print('  failures per node: ' + ', '.join(f'{n} {bad[n]}/{ran[n]}'
                                                 for n in sorted(bad, key=lambda n: -bad[n])))
    for i, t in sorted(failed.items()):
        print(f"  {i:>4}  {t['node'] or '?':<12} {(t['start'] or '?')[:19]:<19}  {t['cause']}"
              + ('' if t['transient'] else '  (not transient)'))
        if t['detail']:
            print(f"        {t['detail'][:110]}")


def _ranges(ix):
    """[1, 2, 3, 7] -> '1-3,7' (Slurm's array syntax)"""
    ix, out, start = sorted(ix), [], None
    for a, b in zip(ix, ix[1:] + [None]):
        start = a if start is None else start
        if b != a + 1:
            out.append(f'{start}-{a}' if a != start else f'{a}')
            start = None
    return ','.join(out)


def resubmit_missing(sweep=None, exclude=None, include_errors=False, remote=None, dry_run=False):
    """
    Resubmit the array tasks of a sweep that didn't save a result and aren't queued or
    running, as a new array job running the sweep's own job script (same pickles, so the
    results land where the originals' would have).

    Tasks that failed for reasons a retry won't fix (a Python error, the time limit, running
    out of memory, missing weights, ...) are listed and left out unless include_errors=True,
    so a bug doesn't burn another round of GPU time. exclude: nodes to keep the reruns off,
    or 'failed' for the nodes where this sweep's tasks failed transiently.
    """
    remote = shutil.which('sbatch') is None if remote is None else remote
    st = sweep_status(sweep, remote, verbose=False)
    tasks = st['tasks']
    missing = {i: t for i, t in tasks.items() if not t['done'] and t['cause'] is not None}
    todo = sorted(i for i, t in missing.items() if t['transient'] or include_errors)
    skipped = sorted(set(missing) - set(todo))
    for i in skipped:
        print(f"leaving out {i}: {tasks[i]['cause']} ({(tasks[i]['detail'] or '')[:90]})")
    if skipped:
        print('  (fix the cause, then resubmit_missing(..., include_errors=True))')
    if not todo:
        print('nothing to resubmit')
        return None

    if exclude == 'failed':
        exclude = sorted({t['node'] for t in missing.values() if t['transient'] and t['node']})
    spec = _ranges(todo) + (f"%{st['max_parallel']}" if st['max_parallel'] else '')
    extra = [f'--array={spec}'] + ([f'--exclude={",".join(exclude)}'] if exclude else [])
    print(f'resubmitting {len(todo)} task(s): {_ranges(todo)}'
          + (f', not on {",".join(exclude)}' if exclude else ''))
    if dry_run:
        return None
    job = _sbatch(os.path.join(st['sweep'], 'job_script.sh'), remote, extra=extra)
    _record_job(st['sweep'], job, remote)
    print(f'Submitted batch job {job}')
    return job

def fetch_results(dest=None):
    """Copy the cluster's results into SAVE_DIR here (or `dest`), e.g. for plotting on the
    laptop. Only new or changed files are transferred."""
    dest = dest or SAVE_DIR
    os.makedirs(dest, exist_ok=True)
    results = _remote('echo ${ICL_RESULTS:-$SCRATCH/icl_results}').splitlines()[-1]
    subprocess.run(['rsync', '-a', f'{REMOTE_HOST}:{results}/', dest], check=True)


def get_all_experiments(task_args, model_args, bool_friendly=True):

    ##### Loop over experiment parameters
    #########################################

    exps = []

    all_keys = list(task_args.keys()) + list(model_args.keys())
    params = {k:[] for k in all_keys if k not in ['task', 'model']}
    for this_dset in parse_params(task_args, forbidden_keys=['task']):

        exp_info = {'task':task_args['task'], 'args':this_dset}

        net_idx = 0
        for this_net in parse_params(model_args, forbidden_keys=['model']):

            net_info = {'model':model_args['model'], 'args':this_net}

            exps.append({'task_args':exp_info, 'model_args':net_info})

            for k,v in this_dset.items():
                params[k].append(v)
            for k,v in this_net.items():
                params[k].append(v)

    for k,v in params.items():
        # if bool_friendly:
        #     params[k] = np.array([stringify(vv) for vv in v])
        # else:
        arr_v = np.array(v)
        if arr_v.dtype.name == 'object':
            arr_v = np.array([stringify(vv) for vv in arr_v])
        
        params[k] = arr_v

    return exps, params


def stringify(thing):
    """
    because python is doodoo caca  
    """

    whatisit = type(thing)

    # if 'builtin' not in whatisit.__module__:
    #     if '__name__' in dir(thing):
    #         return thing.__name__
    #     else:
    #         return thing.__class__.__name__
    if callable(thing):
        return thing.__name__
    else:
        return thing


def _canon(v):
    """A hashable, numpy/version-stable key for value comparison (mirrors how
    arg_digest canonicalizes): callables -> name, numpy scalars/arrays -> native,
    anything unhashable -> repr."""
    v = stringify(v)
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return tuple(v.tolist())
    if type(v).__eq__ is object.__eq__ and v is not None:
        return repr(v)    # identity equality can't match across processes; compare by repr
    try:
        hash(v)
        return v
    except TypeError:
        return repr(v)


def _grid_signatures(param_dict, forbidden):
    """(keys, signatures) for one side (task or model) of a sweep dict.

    Enumerates the sweep grid with parse_params -- so it honors Set/Real/Integer,
    matched tuples, and constraints exactly like get_all_experiments -- and
    returns the SPECIFIED keys plus the set of their (key, canon-value) tuples
    over every combination.  A saved run matches this side iff its args, read on
    those keys, reproduce one of these signatures; keys absent from `param_dict`
    are never looked at, so they are unconstrained."""
    keys = tuple(sorted(k for k in param_dict if k not in forbidden))
    sigs = {tuple((k, _canon(combo[k])) for k in keys)
            for combo in parse_params(param_dict, forbidden_keys=list(forbidden))}
    return keys, sigs


def _run_signature(args, keys):
    """Signature of a saved run's `args` over `keys`, or None if any key is
    absent (so the run can't match -- e.g. it predates a field being queried)."""
    try:
        return tuple((k, _canon(args[k])) for k in keys)
    except KeyError:
        return None


def load_experiments(task_args, model_args, SAVE_DIR=None,
                     reconstitute=False, verbose=True):
    """Load every saved run matching a task/model parameter sweep -- same dict
    interface as get_all_experiments, but robust to missing/renamed folders and
    to PARTIAL specs.

    task_args / model_args carry a 'task'/'model' class plus parameter specs
    (scalars, lists, su.Set/Real/Integer, matched tuples), exactly as passed to
    get_all_experiments and send_to_server.  A saved run is kept iff, on the keys
    PRESENT in the dicts, its stored args reproduce one of the enumerated sweep
    combinations.  Keys omitted from a dict are unconstrained -- so to "load every
    run regardless of posterior_samps", just leave posterior_samps out of
    model_args.

    Unlike the old grid-walk (get_all_experiments + Experiment.load_experiment),
    this never recomputes a folder hash from a live object, so it is immune to the
    two things that orphan runs across cluster sends: numpy/env repr drift and
    dataclass field add/remove.  It keys the search on the class NAMES recorded in
    each arguments_* sidecar, matching both the flat TaskClass/ModelClass layout
    and any older nested-hash folders.

    Returns
    -------
    records : list of dict
        One per matching run: task, model (names), task_args, model_args, folder,
        metrics, and (if reconstitute) task_obj/model_obj.
    prm : dict {name: np.ndarray}
        Every arg name seen, stacked across records (object-dtype stringified,
        like get_all_experiments) for boolean-mask plotting: these = prm['x']==v.
    """
    if SAVE_DIR is None:
        SAVE_DIR = globals()['SAVE_DIR']

    task_cls, model_cls = task_args['task'], model_args['model']
    task_name = task_cls if isinstance(task_cls, str) else task_cls.__name__
    model_name = model_cls if isinstance(model_cls, str) else model_cls.__name__

    tkeys, tsigs = _grid_signatures(task_args, {'task'})
    mkeys, msigs = _grid_signatures(model_args, {'model'})

    # sidecars keyed by class name: the flat TaskClass/ModelClass/ layout, plus
    # older nested TaskClass/<hash>/ModelClass/<hash>/ saves.
    patterns = [os.path.join(SAVE_DIR, task_name, model_name, 'arguments_*.pkl'),
                os.path.join(SAVE_DIR, task_name, '*', model_name, '*',
                             'arguments_*.pkl')]
    sidecars = sorted({p for pat in patterns for p in glob.glob(pat)})

    records = []
    for side in tqdm(sidecars, disable=not verbose):
        try:
            with open(side, 'rb') as f:
                rec = pkl.load(f)
        except Exception:
            continue

        targs, margs = rec.get('task_args', {}), rec.get('model_args', {})
        if _run_signature(targs, tkeys) not in tsigs:
            continue
        if _run_signature(margs, mkeys) not in msigs:
            continue

        # load the sibling metrics file (arguments_X.pkl -> metrics_X.pkl)
        folder = os.path.dirname(side)
        mfile = os.path.join(folder,
            os.path.basename(side).replace('arguments_', 'metrics_', 1))
        try:
            with open(mfile, 'rb') as f:
                metrics = pkl.load(f)
        except Exception:
            continue

        out = {'task': rec.get('task'), 'model': rec.get('model'),
               'task_args': targs, 'model_args': margs,
               'folder': folder, 'metrics': metrics}
        if reconstitute:
            out['task_obj'] = _rebuild(rec.get('task_module'), rec.get('task'), targs)
            out['model_obj'] = _rebuild(rec.get('model_module'), rec.get('model'), margs)
        records.append(out)

    # stack params across records for mask-style filtering downstream
    keys = set()
    for r in records:
        keys |= set(r['task_args']) | set(r['model_args'])
    prm = {}
    for k in keys:
        col = [{**r['task_args'], **r['model_args']}.get(k, None) for r in records]
        arr = np.array(col)
        if arr.dtype.name == 'object':
            arr = np.array([stringify(v) for v in col])
        prm[k] = arr

    if verbose:
        print(f'loaded {len(records)} / {len(sidecars)} runs '
              f'({task_name}/{model_name})')
    return records, prm


def _rebuild(module_name, class_name, args):
    """Reconstitute a Task/Model instance from a sidecar record: import its class
    and call it with the saved (full) args.  Returns None if the class can't be
    found (e.g. it was renamed/moved since the run was saved)."""
    try:
        return getattr(importlib.import_module(module_name), class_name)(**args)
    except Exception:
        return None


def pad_to_dense(M):
    """Appends the minimal required amount of zeroes at the end of each
     array in the jagged array `M`, such that `M` looses its jagedness."""

    shapes = np.array([m.shape for m in M])

    buff = shapes.max(0, keepdims=True) - shapes
    
    Z = []
    for enu, row in enumerate(M):
        
        Z.append(
            np.pad( 
                1.0*row, [(0,s) for s in buff[enu]], mode='constant', constant_values=np.nan 
                )
            )
        
    return np.array(Z)