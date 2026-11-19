"""
Make (or check) a model's stored token categories, as categories.py describes:

    python jobs/make_categories.py MODEL NAME --embedding canonical [--check] [--overwrite]
    python jobs/make_categories.py MODEL --embeddings canonical [--check]
    python jobs/make_categories.py MODEL --list

NAME is a stored fit's name, e.g. semibmf300. Making does nothing if the fit is current, and
replaces it (keeping the old file) if stale. --check makes nothing: it exits 0 if the fit (or
the cached embeddings) is current, 1 if not, which is how server_utils.send_to_server decides
whether to run this first. --embeddings caches the model's embeddings of the fit vocabulary
(what bmf_sweep's fits read).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import categories


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0].strip())
    p.add_argument('model')
    p.add_argument('name', nargs='?', help='a stored fit, e.g. semibmf300')
    p.add_argument('--embedding', default='canonical', choices=categories.EMBEDDINGS)
    p.add_argument('--embeddings', choices=categories.EMBEDDINGS,
                   help='cache these embeddings of the fit vocabulary instead of fitting')
    p.add_argument('--check', action='store_true', help='exit 0 if current, 1 if not; make nothing')
    p.add_argument('--overwrite', action='store_true', help='refit even if current')
    p.add_argument('--list', action='store_true', help="list the model's stored fits")
    a = p.parse_args(argv)
    if a.list:
        for stem, s in categories.available(a.model).items():
            print(f'{stem:28s} {s}')
        return 0
    if a.embeddings:
        s = categories.embeddings_status(a.model, a.embeddings)
        if a.check:
            print(f'{a.embeddings} embeddings for {a.model}: {s}')
            return 0 if s == 'ok' else 1
        tokens, X = categories.embeddings(a.model, a.embeddings)
        print(f'{categories.embeddings_path(a.model, a.embeddings)}: {len(tokens)} tokens x {X.shape[1]}')
        return 0
    if not a.name:
        p.error('give a NAME (e.g. semibmf300), --embeddings or --list')
    if a.check:
        s = categories.status(a.model, a.name, a.embedding)
        print(f'{a.name} ({a.embedding}) for {a.model}: {s}')
        return 0 if s == 'ok' else 1
    categories.generate(a.model, a.name, a.embedding, overwrite=a.overwrite)
    return 0


if __name__ == '__main__':
    sys.exit(main())