"""
Command line for the sweep tools in server_utils, registered in pyproject.toml as `sweeps`:

    uv run sweeps status [SWEEP]
    uv run sweeps resubmit [SWEEP] [--per-job N] [--exclude NODE ...] [--include-errors] [--dry-run]
    uv run sweeps fetch

SWEEP is a sweep's folder name (as send_to_server printed it), one of its array job ids, or
left out for the most recent sweep. From the laptop these run on the cluster over ssh, as
the functions do. `uv run sweeps <command> --help` lists each command's options.
"""
import argparse
import sys


def main(argv=None):
    p = argparse.ArgumentParser(prog='sweeps', description=__doc__.split('\n\n')[0].strip())
    p.add_argument('--remote', action=argparse.BooleanOptionalAction, default=None,
                   help='go through ssh (default: only if this machine has no sbatch)')
    sub = p.add_subparsers(dest='command', required=True)

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
        if a.command == 'status':
            su.sweep_status(a.sweep, remote=a.remote, job_ids=a.job_ids)
        elif a.command == 'resubmit':
            exclude = a.exclude[0] if a.exclude == ['failed'] else a.exclude
            su.resubmit_missing(a.sweep, exclude=exclude, include_errors=a.include_errors,
                                remote=a.remote, dry_run=a.dry_run, job_ids=a.job_ids,
                                per_job=a.per_job, max_parallel=a.max_parallel)
        elif a.command == 'fetch':
            su.fetch_results(a.dest)
    except (RuntimeError, FileExistsError) as e:
        sys.exit(f'sweeps {a.command}: {e}')


if __name__ == '__main__':
    main()