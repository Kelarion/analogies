"""
The NLTK data the code uses, for server_utils.send_to_server (run on the cluster):

    python jobs/nltk_data.py check       prints, as JSON, the packages that aren't installed
    python jobs/nltk_data.py download    installs those (into ~/nltk_data, where NLTK looks)

The code fetches some of these itself on first use, but jobs can't on clusters whose compute
nodes have no internet (Della, most of DRAC), so send_to_server makes sure they're there first.
Add a package here when the code starts using a new one.
"""
import json
import sys

import nltk

PACKAGES = {                          # package: where nltk.data.find looks for it
    'stopwords': 'corpora/stopwords',       # vocabulary.function_words
    'wordnet': 'corpora/wordnet',           # vocabulary._wordnet
    'framenet_v17': 'corpora/framenet_v17',  # geometry._source_categories('frames')
}


def installed(path):
    for p in (path, path + '.zip'):        # some packages stay zipped (wordnet)
        try:
            nltk.data.find(p)
            return True
        except LookupError:
            pass
    return False


def missing():
    return [pkg for pkg, path in PACKAGES.items() if not installed(path)]


if __name__ == '__main__':
    if sys.argv[1:] == ['check']:
        print(json.dumps(missing()))
    elif sys.argv[1:] == ['download']:
        failed = [pkg for pkg in missing() if not nltk.download(pkg, quiet=True)]
        if failed:
            sys.exit(f'could not download {failed}')
        print('NLTK data installed:', ', '.join(PACKAGES))
    else:
        sys.exit('usage: nltk_data.py check | download')