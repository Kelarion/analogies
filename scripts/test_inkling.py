"""
Checks scoring for an eager-only model (Inkling 2), which ChatLM.score runs in blocks of
prefill_block positions through the KV cache:

    python test_inkling.py [MODEL] [--n 3000] [--block 1024] [--lengths 4000 8000 12000 16000]

1. correctness: lm.check_blocked_scoring, one pass vs blocks, on n tokens. n should be well
   over the 1024-token sliding window so blocks cross it; the one-pass version has to fit
   at n (it does at 3000).
2. memory: peak GPU memory of score() at increasing lengths, with the traceback on OOM.
"""
import argparse
import time
import traceback

import numpy as np
import torch

from icl_model import get_lm

p = argparse.ArgumentParser()
p.add_argument('model', nargs='?', default='thinkingmachines/Inkling-2-Mini')
p.add_argument('--n', type=int, default=3000)
p.add_argument('--block', type=int, default=1024)
p.add_argument('--lengths', type=int, nargs='+', default=[4000, 8000, 12000, 16000])
args = p.parse_args()

GB = 2 ** 30
lm = get_lm(args.model)
print(f'loaded {args.model}: sdpa_ok = {lm.sdpa_ok}, prefill_block = {lm.prefill_block}')
print(f'  allocated after loading: {torch.cuda.memory_allocated() / GB:.1f} GB')
if lm.sdpa_ok:
    print('  (this model scores on SDPA in one pass; the blocked path below is not what it uses)')

# -- 1. correctness -------------------------------------------------------------------------------
print(f'\n1. one pass vs blocks of {args.block}:')
lm.check_blocked_scoring(n=args.n, block=args.block)
print('  differences at bf16 rounding (up to ~0.1 nats, same top token nearly everywhere) mean\n'
      '  the cache handles multi-token blocks; large ones mean it does not, and the fallback\n'
      '  is needed (prefill_block = None, or blocked attention).')
torch.cuda.empty_cache()

# -- 2. memory -----------------------------------------------------------------------------------
print(f'\n2. peak memory of score() by length (prefill_block = {lm.prefill_block})')
task_toks = np.arange(1000, 1020)
lo, hi = 1000, min(len(lm.tokenizer), 100_000)
for T in args.lengths:
    x = torch.randint(lo, hi, (T,), device=lm.device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    t0 = time.time()
    try:
        lm.score(x, task_toks)
        torch.cuda.synchronize()
    except torch.OutOfMemoryError:
        print(f'  T={T:6d}: out of memory; where:')
        traceback.print_exc()
        break
    peak = torch.cuda.max_memory_allocated()
    print(f'  T={T:6d}: peak {peak / GB:5.1f} GB (+{(peak - base) / GB:4.1f} over weights), '
          f'{time.time() - t0:5.1f} s')
    del x