"""
Model weights in the Hugging Face cache (HF_HOME), for server_utils.send_to_server:

    python jobs/weights.py check NAME ...      prints, as JSON, {model: why} for those not fully cached
    python jobs/weights.py download NAME ...   fetches whatever of them is missing

"Fully cached" means a snapshot exists, none of its files is a link to a deleted blob
(the scratch cleanup removes files one by one), and every weight shard is present.
Downloads take only what loading needs (configs, tokenizer, chat template, safetensors),
not the extra copies some repos carry (.bin, .pth, original/, gguf).
"""
import json
import os
import sys

from huggingface_hub import snapshot_download

PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken", "*.jinja"]


def problem(name):
    """Why `name` isn't fully in the cache, or None if it is."""
    try:
        # same patterns as the download: newer huggingface_hub checks the snapshot against
        # the repo's file listing, which would otherwise include files we never fetch
        d = snapshot_download(name, local_files_only=True, allow_patterns=PATTERNS)
    except Exception as e:
        return f'{type(e).__name__}: {e}'.splitlines()[0]
    files = os.listdir(d)
    gone = [f for f in files if not os.path.exists(os.path.join(d, f))]
    if gone:
        return f'{len(gone)} file(s) deleted from the cache in {d}, e.g. {gone[0]}'
    index = os.path.join(d, 'model.safetensors.index.json')
    if os.path.exists(index):
        with open(index) as f:
            shards = sorted(set(json.load(f)['weight_map'].values()))
        missing = [s for s in shards if not os.path.exists(os.path.join(d, s))]
        if missing:
            return f'{len(missing)} of {len(shards)} weight shards missing in {d}, e.g. {missing[0]}'
    elif not any(f.endswith('.safetensors') for f in files):
        return f'no .safetensors weights in {d} (files: {", ".join(sorted(files)[:8])})'
    return None


if __name__ == '__main__':
    cmd, names = sys.argv[1], sys.argv[2:]
    if cmd == 'check':
        print(json.dumps({n: p for n in names if (p := problem(n))}))
    elif cmd == 'download':
        for n in names:
            print('downloading', n, flush=True)
            print(snapshot_download(n, allow_patterns=PATTERNS), flush=True)
    else:
        sys.exit(f'unknown command {cmd!r}')