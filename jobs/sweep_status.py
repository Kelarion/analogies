"""
What happened to each array task of a sweep (run on the cluster, by server_utils.sweep_status):

    python jobs/sweep_status.py [SWEEP]      SWEEP: a folder name in the sweeps folder, a path,
                                             or nothing for the most recent sweep

Prints JSON: {'sweep': path, 'n_tot': ..., 'max_parallel': ..., 'tasks': {index: {...}}}, each
task with: done (its result was saved), state (Slurm's, from sacct, if the job ids were
recorded), node, start, cause (a short label for why it ended, from Slurm and its log), and
detail (the matching log line). Only the standard library, so it starts instantly.
"""
import glob
import json
import os
import re
import subprocess
import sys

# log signatures, most specific first: (label, pattern, transient?)
# transient: plausibly fine on a retry (and on another node); the others would just fail again
SIGNATURES = [
    ('filesystem I/O error', r'Remote I/O error|Input/output error|Stale file handle', True),
    ('GPU error', r'CUDA error|uncorrectable ECC|Xid|NVML|unspecified launch failure', True),
    ('node failure', r'NODE_FAIL|DUE TO NODE FAILURE', True),
    ('preempted', r'DUE TO PREEMPTION', True),
    ('time limit', r'DUE TO TIME LIMIT', False),
    ('CPU memory (oom-kill)', r'oom[-_]kill|Out Of Memory|OUT_OF_MEMORY', False),
    ('GPU out of memory', r'CUDA out of memory|OutOfMemoryError', False),
    ('weights not cached', r"LocalEntryNotFoundError|couldn't find them in the cached files", False),
    ('environment out of date', r'environment out of date', False),
    ('cancelled', r'CANCELLED AT', True),          # last: Slurm's other kill messages say this too
]
STATES = {'TIMEOUT': 'time limit', 'OUT_OF_MEMORY': 'CPU memory (oom-kill)',
          'NODE_FAIL': 'node failure', 'PREEMPTED': 'preempted', 'CANCELLED': 'cancelled'}
TRANSIENT = {label for label, _, t in SIGNATURES if t} | {'no log', 'ended without an error'}


def sweep_dir(arg):
    root = os.environ.get('ICL_SWEEPS', os.path.join(os.environ.get('SCRATCH', os.path.expanduser('~')), 'icl_sweeps'))
    if not arg:
        dirs = [d for d in glob.glob(os.path.join(root, '*')) if os.path.isdir(d)]
        return max(dirs, key=os.path.getmtime)
    return arg if os.path.isabs(arg) else os.path.join(root, arg)


def expand(spec):
    """'3' / '[3-5,7%8]' -> [3, 4, 5, 7]"""
    out = []
    for part in spec.strip('[]').split('%')[0].split(','):
        a, _, b = part.partition('-')
        out += list(range(int(a), int(b or a) + 1))
    return out


def sacct(job_ids):
    """{index: {'state', 'node', 'start'}} from the recorded array jobs, later ones winning."""
    info = {}
    for jid in job_ids:
        out = subprocess.run(['sacct', '-j', jid, '-n', '-P', '-X',
                              '--format=JobID,State,NodeList,Start'],
                             capture_output=True, text=True).stdout
        for line in out.splitlines():
            job, state, node, start = (line.split('|') + [''] * 4)[:4]
            if '_' not in job:
                continue
            for i in expand(job.split('_', 1)[1]):
                info[i] = {'state': state.split()[0], 'node': None if node in ('', 'None assigned') else node,
                           'start': None if start in ('', 'Unknown', 'None') else start}
    return info


def from_log(path):
    if not os.path.exists(path):
        return {}
    with open(path, errors='replace') as f:
        text = f.read()
    out = {}
    m = re.search(r'^node: (\S+)\s+start: (\S+)', text, re.M)
    if m:
        out['node'], out['start'] = m.groups()
    for label, pat, _ in SIGNATURES:
        hit = re.search(f'^.*(?:{pat}).*$', text, re.M)
        if hit:
            out['cause'], out['detail'] = label, hit.group(0).strip()[-200:]
            return out
    errors = re.findall(r'^\w*(?:Error|Exception)\b.*$', text, re.M)
    if errors:
        out['cause'], out['detail'] = 'python error', errors[-1].strip()[-200:]
    return out


def status(d):
    n_tot = len(glob.glob(os.path.join(d, 'task_*.pkl'))) * len(glob.glob(os.path.join(d, 'model_*.pkl')))
    script = open(os.path.join(d, 'job_script.sh')).read()
    m = re.search(r'--array=\S*%(\d+)', script)
    ids_file = os.path.join(d, 'jobids')
    job_ids = open(ids_file).read().split() if os.path.exists(ids_file) else []
    slurm = sacct(job_ids) if job_ids else {}

    tasks = {}
    for i in range(n_tot):
        log = os.path.join(d, 'logs', f'{i}.out')
        t = {'done': os.path.exists(os.path.join(d, 'done', str(i))), 'state': None,
             'node': None, 'start': None, 'cause': None, 'detail': None}
        if not t['done'] and os.path.exists(log):      # sweeps from before the markers
            with open(log, errors='replace') as f:
                t['done'] = any(line.startswith('saved ') for line in f)
        t.update({k: v for k, v in slurm.get(i, {}).items() if v})
        if not t['done']:
            t.update({k: v for k, v in from_log(log).items() if v})
            if t['state'] in STATES:                    # Slurm knows best why it ended a job
                t['cause'] = STATES[t['state']]
            if t['state'] in ('PENDING', 'RUNNING', 'REQUEUED'):
                t['cause'] = None
            elif t['cause'] is None:
                t['cause'] = 'no log' if not os.path.exists(log) else 'ended without an error'
        t['transient'] = t['cause'] in TRANSIENT
        tasks[i] = t
    return {'sweep': d, 'n_tot': n_tot, 'max_parallel': int(m.group(1)) if m else None,
            'job_ids': job_ids, 'tasks': tasks}


if __name__ == '__main__':
    print(json.dumps(status(sweep_dir(sys.argv[1] if len(sys.argv) > 1 else ''))))
    