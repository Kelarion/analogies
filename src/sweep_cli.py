"""
Command line for the sweep tools in server_utils, registered in pyproject.toml as `sweeps`:

    uv run sweeps send SPEC [--name NAME] [--per-job N] [--exclude NODE ...] [--dry-run] [--here]
    uv run sweeps status [SWEEP]
    uv run sweeps resubmit [SWEEP] [--per-job N] [--exclude NODE ...] [--include-errors] [--dry-run]
    uv run sweeps fetch

SPEC is a Python file defining the sweep's arguments, as send_to_server takes them:

    import icl_tasks, icl_model
    task_args = {'task': icl_tasks.GridTask, 'n': 3, 'seed': [0, 1]}
    mod_args = {'model': icl_model.ICLModel, 'model_name': 'Qwen/Qwen3.8-27B'}
    send = {'per_job': 8}            # optional: this sweep's defaults for send's options

Options given on the command line override `send`, which overrides send_to_server's defaults.
The sweep is named after the spec file and the time unless --name (or send['name']) says
otherwise, and the spec file is copied into its folder.

SWEEP is a sweep's folder name (as send_to_server printed it), one of its array job ids, or
left out for the most recent sweep. From the laptop these run on the cluster over ssh, as
the functions do. `uv run sweeps <command> --help` lists each command's options.

--cluster NAME (before the command) picks a cluster from clusters.toml, e.g.
`uv run sweeps --cluster della send spec.py`. Left out: for send, the cluster this is
running on, else the file's default; for status and resubmit, the cluster a sweep sent from
here was recorded on; `fetch --cluster all` fetches from every cluster. A spec's `send` can
name its cluster too.
"""
import argparse
import os
import sys


def main(argv=None):
    p = argparse.ArgumentParser(prog='sweeps', description=__doc__.split('\n\n')[0].strip())
    p.add_argument('--remote', action=argparse.BooleanOptionalAction, default=None,
                   help="go through ssh (default: unless this is the cluster)")
    p.add_argument('--cluster', help="a cluster in clusters.toml ('all' for fetch)")
    sub = p.add_subparsers(dest='command', required=True)

    n = sub.add_parser('send', help='submit the sweep a spec file describes')
    n.add_argument('spec', help='a Python file defining task_args and mod_args (and optionally send)')
    n.add_argument('--name', help="the sweep's folder name (default: the spec's name and the time)")
    n.add_argument('--per-job', type=int, help='combinations each array task runs')
    n.add_argument('--max-parallel', type=int, help='array tasks at once')
    n.add_argument('--exclude', nargs='+', metavar='NODE', help='nodes to keep the jobs off')
    n.add_argument('--dry-run', action='store_true', help='say what would be submitted, and stop')
    n.add_argument('--here', action='store_true', help='run the first combination here instead of submitting')
    n.add_argument('--no-check-commit', action='store_true', help="don't require the cluster's checkout to match")
    n.add_argument('--no-weights', action='store_true', help="don't check or download model weights")
    n.add_argument('--no-prerequisites', action='store_true',
                   help="don't check or make the files the runs need (stored token categories)")
    n.add_argument('--quiet', action='store_true', help='print less')

    s = sub.add_parser('status', help='what happened to each array task of a sweep')
    s.add_argument('sweep', nargs='?', help='folder name or array job id (default: the latest)')
    s.add_argument('--job-ids', nargs='+', help="the sweep's array job ids, for sweeps that "
                                                "didn't record them")

    r = sub.add_parser('resubmit', help='rerun the array tasks that never saved their results')
    r.add_argument('sweep', nargs='?', help='folder name or array job id (default: the latest)')
    r.add_argument('--per-job', type=int, help='rechunk: resubmit as a continuation sweep, '
                                               'N combinations per job')
    r.add_argument('--max-parallel', type=int, help='array tasks at once (default: as before)')
    r.add_argument('--exclude', nargs='+', metavar='NODE',
                   help="nodes to keep the reruns off, or 'failed' for where this sweep failed")
    r.add_argument('--include-errors', action='store_true',
                   help='also rerun tasks that failed for reasons a retry won\'t fix')
    r.add_argument('--dry-run', action='store_true', help='say what would be submitted, and stop')
    r.add_argument('--job-ids', nargs='+', help='as for status')

    f = sub.add_parser('fetch', help="copy the cluster's results here (from the laptop)")
    f.add_argument('--dest', help='where to (default: SAVE_DIR)')

    a = p.parse_args(argv)
    import server_utils as su                     # after parsing, so --help is instant

    try:
        if a.command == 'send':
            send(a)
        elif a.command == 'status':
            su.sweep_status(a.sweep, remote=a.remote, job_ids=a.job_ids, cluster=a.cluster)
        elif a.command == 'resubmit':
            exclude = a.exclude[0] if a.exclude == ['failed'] else a.exclude
            su.resubmit_missing(a.sweep, exclude=exclude, include_errors=a.include_errors,
                                remote=a.remote, dry_run=a.dry_run, job_ids=a.job_ids,
                                per_job=a.per_job, max_parallel=a.max_parallel, cluster=a.cluster)
        elif a.command == 'fetch':
            su.fetch_results(a.dest, cluster=a.cluster)
    except (RuntimeError, FileExistsError, ValueError, KeyError) as e:
        sys.exit(f'sweeps {a.command}: {e}')


SEND_OPTIONS = ('name', 'per_job', 'max_parallel', 'exclude', 'check_commit', 'ensure_weights',
                'ensure_prerequisites', 'verbose', 'template', 'cluster')


def load_spec(path):
    """(task_args, mod_args, send) from a spec file (see the module docstring)."""
    import runpy
    if not os.path.isfile(path):
        raise ValueError(f'no spec file {path}')
    spec = runpy.run_path(path, run_name='__sweep_spec__')
    for key, cls in (('task_args', 'task'), ('mod_args', 'model')):
        if not isinstance(spec.get(key), dict):
            raise ValueError(f'{path} must define {key}, a dict')
        if cls not in spec[key]:
            raise ValueError(f"{path}: {key} has no '{cls}' (the {cls} class)")
    send_ = spec.get('send', {})
    unknown = set(send_) - set(SEND_OPTIONS)
    if unknown:
        raise ValueError(f'{path}: send has {sorted(unknown)}; it may set {list(SEND_OPTIONS)}')
    return spec['task_args'], spec['mod_args'], dict(send_)


def send(a):
    """`sweeps send`: send_to_server with the spec's arguments, and its options from the spec's
    `send`, overridden by the command line."""
    import time
    import server_utils as su
    task_args, mod_args, opts = load_spec(a.spec)
    given = {'name': a.name, 'per_job': a.per_job, 'max_parallel': a.max_parallel, 'exclude': a.exclude,
             'cluster': a.cluster}
    opts.update({k: v for k, v in given.items() if v is not None})
    if a.no_check_commit:
        opts['check_commit'] = False
    if a.no_weights:
        opts['ensure_weights'] = False
    if a.no_prerequisites:
        opts['ensure_prerequisites'] = False
    opts.setdefault('verbose', not a.quiet)
    opts.setdefault('name', os.path.splitext(os.path.basename(a.spec))[0] + time.strftime('-%Y%m%d-%H%M%S'))
    return su.send_to_server(task_args, mod_args, run_remote=not a.here, remote=a.remote,
                             dry_run=a.dry_run, attach=[a.spec], **opts)


if __name__ == '__main__':
    main()