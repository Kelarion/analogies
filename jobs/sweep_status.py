"""
What happened to each array task of a sweep (run on the cluster, by server_utils.sweep_status):

    python3 jobs/sweep_status.py [SWEEP] [--text] [--job-ids=ID,ID]

SWEEP: a folder name in the sweeps folder, a path, or nothing for the most recent sweep.
--job-ids gives the sweep's array job ids (recorded in SWEEP/jobids for next time), for
sweeps whose ids weren't recorded and can't be found in Slurm's accounting.
--text prints a readable summary instead of JSON (no project environment needed, so a lone
copy of this file works too: python3 ~/sweep_status.py SWEEP --text).

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
                              '--format=JobID,State,NodeList,Start,ExitCode'],
                             capture_output=True, text=True).stdout
        for line in out.splitlines():
            job, state, node, start, code = (line.split('|') + [''] * 5)[:5]
            if '_' not in job:
                continue
            for i in expand(job.split('_', 1)[1]):
                info[i] = {'state': state.split()[0], 'node': None if node in ('', 'None assigned') else node,
                           'start': None if start in ('', 'Unknown', 'None') else start,
                           'exit': code or None}
    return info


def find_job_ids(d, warnings):
    """The array jobs that ran this sweep's job script, for sweeps that didn't record them.
    Slurm's accounting is searched by the submit command (it names the script), else by the
    log path (it names the sweep's folder); else the queue, for jobs still alive. Fields a
    Slurm version doesn't have are skipped, and its complaints collected in `warnings`."""
    script, logs = os.path.join(d, 'job_script.sh'), os.path.join(d, 'logs')
    for field, needle in (('SubmitLine', script), ('StdOut', logs)):
        res = subprocess.run(['sacct', '-u', os.environ.get('USER', ''), '-S', 'now-60days', '-X',
                              '-n', '-P', f'--format=JobID,{field}'],
                             capture_output=True, text=True)
        if res.returncode or res.stderr.strip():
            warnings.append(f'sacct --format=JobID,{field}: ' + (res.stderr.strip() or f'exit {res.returncode}'))
        ids = [line.split('|', 1)[0].split('_')[0] for line in res.stdout.splitlines()
               if needle in line.partition('|')[2]]
        if ids:
            return sorted(set(ids), key=int)
    res = subprocess.run(['squeue', '--me', '-h', '-o', '%F|%o'], capture_output=True, text=True)
    ids = [job for job, _, cmd in (l.partition('|') for l in res.stdout.splitlines()) if script in cmd]
    return sorted(set(ids), key=int)


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


def status(d, given_ids=()):
    meta = os.path.join(d, 'sweep.json')
    if os.path.exists(meta):                      # array task j runs combinations j*per_job, ...
        with open(meta) as f:
            meta = json.load(f)
        n_tot, per_job = meta['n_tot'], meta['per_job']
    else:                                         # older sweeps: one combination per task
        n_tot = len(glob.glob(os.path.join(d, 'task_*.pkl'))) * len(glob.glob(os.path.join(d, 'model_*.pkl')))
        per_job = 1
    n_jobs = -(-n_tot // per_job)
    script = open(os.path.join(d, 'job_script.sh')).read()
    m = re.search(r'--array=\S*%(\d+)', script)
    ids_file = os.path.join(d, 'jobids')
    job_ids = open(ids_file).read().split() if os.path.exists(ids_file) else []
    new = [j for j in given_ids if j not in job_ids]
    if new:                                       # given by hand: remember them
        job_ids += new
        with open(ids_file, 'w') as f:
            f.write('\n'.join(job_ids) + '\n')
    warnings = []
    if not job_ids:                               # a sweep from before job ids were recorded
        job_ids = find_job_ids(d, warnings)
        if job_ids:
            with open(ids_file, 'w') as f:
                f.write('\n'.join(job_ids) + '\n')
    slurm = sacct(job_ids) if job_ids else {}

    tasks = {}                                    # by array index
    for i in range(n_jobs):
        log = os.path.join(d, 'logs', f'{i}.out')
        combos = range(i * per_job, min((i + 1) * per_job, n_tot))
        n_done = sum(os.path.exists(os.path.join(d, 'done', str(c))) for c in combos)
        t = {'done': n_done == len(combos), 'combos': [n_done, len(combos)], 'state': None,
             'node': None, 'start': None, 'exit': None, 'cause': None, 'detail': None}
        if not t['done'] and per_job == 1 and os.path.exists(log):   # sweeps from before the markers
            with open(log, errors='replace') as f:
                t['done'] = any(line.startswith('saved ') for line in f)
        t.update({k: v for k, v in slurm.get(i, {}).items() if v})
        if t['state'] == 'COMPLETED':             # exit 0 comes only after every result is saved
            t['done'] = True
            t['combos'][0] = t['combos'][1]
        if not t['done']:
            t.update({k: v for k, v in from_log(log).items() if v})
            if t['state'] in STATES:                    # Slurm knows best why it ended a job
                t['cause'] = STATES[t['state']]
            if t['state'] in ('PENDING', 'RUNNING', 'REQUEUED'):
                t['cause'] = None
            elif t['cause'] is None:
                t['cause'] = 'no log' if not os.path.exists(log) else 'ended without an error'
                if t['state']:
                    t['detail'] = f"Slurm: {t['state']}, exit code {t.get('exit')}"
        t['transient'] = t['cause'] in TRANSIENT
        if not job_ids and t['cause'] in ('no log', 'ended without an error'):
            t['transient'] = False                    # might be queued or running: don't rerun
            t['detail'] = ('no Slurm record found, so it may still be queued or running '
                           "(give the array job id: job_ids=[...])")
        tasks[i] = t
    return {'sweep': d, 'n_tot': n_tot, 'per_job': per_job, 'n_jobs': n_jobs,
            'max_parallel': int(m.group(1)) if m else None,
            'job_ids': job_ids, 'warnings': warnings, 'tasks': tasks}


def summary(st):
    """Readable summary: counts, failures by cause and per node, then each failed task."""
    tasks = {int(i): t for i, t in st['tasks'].items()}
    live = {i: t for i, t in tasks.items() if not t['done'] and t['cause'] is None}
    failed = {i: t for i, t in tasks.items() if not t['done'] and t['cause'] is not None}
    per_job = st.get('per_job', 1)
    n_done = sum(t.get('combos', [t['done']])[0] for t in tasks.values())
    if per_job == 1:
        lines = [f"{st['sweep']}: {n_done} of {st['n_tot']} done, "
                 f"{len(live)} queued or running, {len(failed)} failed"]
    else:
        lines = [f"{st['sweep']}: {n_done} of {st['n_tot']} combinations done; of its "
                 f"{len(tasks)} jobs ({per_job} each), {len(live)} queued or running, "
                 f"{len(failed)} failed"]
    if not st.get('job_ids'):
        lines.append('  (no Slurm job ids for this sweep: states, nodes and times unknown)')
    lines += [f'  warning: {w}' for w in st.get('warnings', [])]
    if not failed:
        return '\n'.join(lines)
    causes = {}
    for t in failed.values():
        causes[t['cause']] = causes.get(t['cause'], 0) + 1
    lines.append('  by cause: ' + ', '.join(f'{c} {n}' for c, n in sorted(causes.items(), key=lambda x: -x[1])))
    ran, bad = {}, {}
    for t in tasks.values():
        if t['node']:
            ran[t['node']] = ran.get(t['node'], 0) + 1
            if not t['done'] and t['cause'] is not None:
                bad[t['node']] = bad.get(t['node'], 0) + 1
    if bad:
        lines.append('  failures per node: ' + ', '.join(f'{n} {bad[n]}/{ran[n]}'
                                                       for n in sorted(bad, key=lambda n: -bad[n])))
    for i, t in sorted(failed.items()):
        part = f"  ({t['combos'][0]}/{t['combos'][1]} done)" if per_job > 1 else ''
        lines.append(f"  {i:>4}  {t['node'] or '?':<12} {(t['start'] or '?')[:19]:<19}  {t['cause']}"
                     + ('' if t['transient'] else '  (not transient)') + part)
        if t['detail']:
            lines.append(f"        {t['detail'][:110]}")
    return '\n'.join(lines)


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    ids = [a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('--job-ids=')]
    ids = [j for a in ids for j in a.split(',') if j]
    st = status(sweep_dir(args[0] if args else ''), ids)
    print(summary(st) if '--text' in sys.argv else json.dumps(st))