"""
Model weights in the Hugging Face cache (HF_HOME), for server_utils.send_to_server:

    python jobs/weights.py check NAME ...      prints, as JSON, the models not fully cached
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


def complete(name):
    try:
        d = snapshot_download(name, local_files_only=True)
    except Exception:
        return False
    files = os.listdir(d)
    if not all(os.path.exists(os.path.join(d, f)) for f in files):     # dangling links
        return False
    index = os.path.join(d, 'model.safetensors.index.json')
    if os.path.exists(index):
        with open(index) as f:
            shards = set(json.load(f)['weight_map'].values())
        return all(os.path.exists(os.path.join(d, s)) for s in shards)
    return any(f.endswith('.safetensors') for f in files)


if __name__ == '__main__':
    cmd, names = sys.argv[1], sys.argv[2:]
    if cmd == 'check':
        print(json.dumps([n for n in names if not complete(n)]))
    elif cmd == 'download':
        for n in names:
            print('downloading', n, flush=True)
            print(snapshot_download(n, allow_patterns=PATTERNS), flush=True)
    else:
        sys.exit(f'unknown command {cmd!r}')